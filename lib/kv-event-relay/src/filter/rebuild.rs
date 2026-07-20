// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-model relay-side CKF state. Each model owns one upstream
//! [`DcCkfState`] fed by a synthetic member; the deduped event stream drives
//! its authoritative aggregation (exact refcounts plus the packed CKF lane).
//! The absolute bucket images the state publishes are mirrored into a
//! per-model word array (the source of full snapshots) and their bucket ids
//! accumulated in a dirty set, so a periodic publisher can ship one coalesced
//! CBI1 delta per tick instead of one frame per event.
//!
//! The upstream filter is fixed-capacity: it is sized once from
//! `expected_blocks_per_dc` and never resized. An insert past capacity fails
//! the event (surfaced as a warning); size the capacity hint for the DC's
//! working set.
//!
//! Why not the upstream `DcCkfPublisher`/`DcCkfDeltaSink`: that
//! publisher sequences per-command absolute-image batches for an in-process
//! consumer. This relay instead needs a WAN publication shaped for the CBI1
//! gRPC transport: coalesce all bucket changes across many commands into ONE
//! delta per `--filter-interval-ms` tick, and ship full state as chunked CBI1
//! snapshots on (re)connect. So the relay pins `publish_every_n_events = 1`,
//! drains raw `outcome.publication()` images, and owns its own per-tick
//! coalescing (mirror + dirty set) and CBI1 epoch continuity — deliberately
//! not the upstream `last_sequence` protocol.

use std::sync::{Arc, Mutex as StdMutex};

use rustc_hash::FxHashSet;

use dynamo_kv_event_relay_proto::wire::images::{
    BucketImage, FINGERPRINT_BITS, FORMAT_VERSION, FilterFormat, SLOTS_PER_BUCKET, encode_delta,
    max_delta_images,
};
use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState};
use dynamo_kv_router::protocols::{KvCacheEvent, RouterEvent, WorkerWithDpRank};

use crate::model_registry::ModelIdentity;
use crate::state::lock_recovering;

use super::registry::FilterRegistry;

/// This relay is one DC: all of its workers collapse onto one DC-scoped state.
/// The upstream aggregator refcounts contributions across members, so a single
/// synthetic member is enough — the ingest stream is already DC-deduped.
const RELAY_LANE_MEMBER: WorkerWithDpRank = WorkerWithDpRank {
    worker_id: 0,
    dp_rank: 0,
};

/// One publisher-tick outcome for a model.
pub(crate) enum Publish {
    /// Full-lane snapshot at `epoch`; the words are chunked per subscriber.
    Full {
        format: FilterFormat,
        epoch: u64,
        words: Arc<[u64]>,
    },
    /// Coalesced CBI1 delta frame advancing the model's epoch.
    Delta(Vec<u8>),
    /// Nothing changed since the last tick — heartbeat.
    Unchanged,
}

/// Per-model relay-side CKF state driven by the upstream DC aggregator.
pub(crate) struct ModelFilterState {
    /// Public model id — label for the scrape-time filter gauges.
    pub(crate) model_id: Arc<str>,
    /// Replica-side format identity (seed + bucket count) every published frame
    /// carries so the Global Gateway can gate drift.
    format: FilterFormat,
    /// Authoritative aggregation; exact member refcounts + packed CKF lane,
    /// fed by the single synthetic member.
    state: DcCkfState,
    /// Latest absolute word per published bucket — the shadow of the
    /// published lane and the source of full snapshots.
    mirror: Vec<u64>,
    /// Buckets whose image changed since the last emitted frame.
    dirty: FxHashSet<u32>,
    /// Last emitted publication epoch (0 before the first frame).
    epoch: u64,
}

impl ModelFilterState {
    pub(crate) fn new(model_id: Arc<str>, config: CkfConfig) -> Self {
        // The relay does its own per-tick coalescing (mirror + dirty), so the
        // state drains one publication per applied command instead of batching.
        let config = CkfConfig {
            publish_every_n_events: 1,
            ..config
        };
        let state = DcCkfState::new(config).expect("CKF state (validated config)");
        let identity = state.format();
        // The CBI1 header stamps FORMAT_VERSION/FINGERPRINT_BITS/SLOTS_PER_BUCKET
        // as compile-time constants, and FilterFormat carries only seed+bucket_count.
        // If a future upstream bump changes the CKF layout, silently shipping a CBI1
        // v1/16/4 header over the new layout would make the Global Gateway install
        // corrupt filters. Fail loudly at construction instead.
        assert_eq!(
            (
                identity.format_version(),
                identity.fingerprint_bits(),
                identity.slots_per_bucket()
            ),
            (FORMAT_VERSION, FINGERPRINT_BITS, SLOTS_PER_BUCKET),
            "upstream DcCkfFormatIdentity drifted from the CBI1 wire constants; \
             the CBI1 header/encoder must be updated before this relay can publish"
        );
        let format = FilterFormat::new(identity.seed(), identity.bucket_count())
            .expect("upstream bucket_count is a power of two >= 2");
        Self {
            model_id,
            format,
            state,
            mirror: vec![0; format.bucket_count],
            dirty: FxHashSet::default(),
            epoch: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn format(&self) -> FilterFormat {
        self.format
    }

    /// Blocks resident in this model's lane (scrape-time gauge).
    pub(crate) fn block_count(&self) -> usize {
        self.state.member_block_count(RELAY_LANE_MEMBER)
    }

    /// Current-state snapshot for a fresh subscriber: format, epoch, and a
    /// copy of the lane words behind a cheap shared pointer.
    pub(crate) fn snapshot(&self) -> (FilterFormat, u64, Arc<[u64]>) {
        (self.format, self.epoch, Arc::from(self.mirror.as_slice()))
    }

    /// Fold one deduped event into the DC state and mirror its published
    /// images. The event's worker identity is rewritten onto the state's
    /// synthetic member (the DC-level dedup already collapsed real workers).
    /// Returns whether the state reported an apply error (typically a
    /// capacity-exhausted insert) — the caller surfaces it as a counter.
    fn ingest(&mut self, event: RouterEvent) -> bool {
        let storage_tier = event.storage_tier;
        let event_id = event.event.event_id;
        let data = event.event.data;
        let rewritten = RouterEvent::with_storage_tier(
            RELAY_LANE_MEMBER.worker_id,
            KvCacheEvent {
                event_id,
                dp_rank: RELAY_LANE_MEMBER.dp_rank,
                data,
            },
            storage_tier,
        );

        let outcome = self.state.apply_event(rewritten);
        let had_error = if let Some(error) = outcome.first_error() {
            tracing::warn!(
                model_id = %self.model_id,
                %error,
                "CKF apply_event error; lane content may be incomplete (capacity?)"
            );
            true
        } else {
            false
        };
        // Every touched bucket arrives as an absolute image (a `Cleared` drain
        // included — the dirty window enumerates the buckets it zeroes), so
        // deltas alone keep consumers in sync.
        let Some(publication) = outcome.publication() else {
            return had_error;
        };
        for image in publication.images() {
            self.mirror[image.bucket()] = image.value();
            self.dirty.insert(image.bucket() as u32);
        }
        had_error
    }

    /// Compute this tick's output, advancing the epoch when a frame is emitted.
    fn publish(&mut self) -> Publish {
        // A delta too large for one frame becomes a full dump. The mirror
        // already reflects every applied image.
        if self.dirty.len() > max_delta_images() {
            self.epoch += 1;
            self.dirty.clear();
            return Publish::Full {
                format: self.format,
                epoch: self.epoch,
                words: Arc::from(self.mirror.as_slice()),
            };
        }
        if self.dirty.is_empty() {
            return Publish::Unchanged;
        }
        let base = self.epoch;
        self.epoch += 1;
        let words = &self.mirror;
        let images: Vec<BucketImage> = self
            .dirty
            .iter()
            .map(|&bucket| BucketImage {
                bucket,
                value: words[bucket as usize],
            })
            .collect();
        self.dirty.clear();
        Publish::Delta(encode_delta(
            self.format,
            RELAY_LANE_MEMBER.worker_id,
            base,
            self.epoch,
            &images,
        ))
    }
}

/// Lock a model filter, recovering from poisoning.
pub(crate) fn lock_filter(
    mutex: &StdMutex<ModelFilterState>,
) -> std::sync::MutexGuard<'_, ModelFilterState> {
    lock_recovering(mutex, "model filter")
}

/// Fold a deduped event batch into the model's CKF state. Each event is
/// applied to the authoritative aggregator and its published images coalesced
/// into the per-model mirror + dirty set the publisher drains on its tick.
/// Returns how many events hit an apply error (capacity exhaustion) —
/// those blocks are missing from the published filter.
pub(crate) fn update_model_filter(
    filters: &FilterRegistry,
    model: &ModelIdentity,
    events: Vec<RouterEvent>,
    config: CkfConfig,
) -> usize {
    let model_state = filters.get_or_create(model, config);
    let mut state = lock_filter(&model_state);
    let mut apply_errors = 0;
    for event in events {
        if state.ingest(event) {
            apply_errors += 1;
        }
    }
    apply_errors
}

/// This model's publisher-tick output under a single lock.
pub(crate) fn tick_publish(model_state: &Arc<StdMutex<ModelFilterState>>) -> Publish {
    lock_filter(model_state).publish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::registry::FilterRegistry;
    use crate::filter::test_support::{model, removed_event, stored_event};
    use dynamo_kv_event_relay_proto::wire::images::{
        ImagesFrame, SnapshotAssembly, decode, encode_snapshot_chunks,
    };

    const CAPACITY: usize = 4096;

    fn config() -> CkfConfig {
        CkfConfig::new(CAPACITY)
    }

    /// Apply decoded publications to plain lane words. Search semantics belong
    /// to the global-gateway replica tests; relay tests pin publication bytes.
    struct Consumer {
        format: FilterFormat,
        words: Vec<u64>,
        assembly: SnapshotAssembly,
    }

    impl Consumer {
        fn new(format: FilterFormat) -> Self {
            Self {
                format,
                words: vec![0; format.bucket_count],
                assembly: SnapshotAssembly::new(format),
            }
        }

        fn full(&mut self, epoch: u64, words: &[u64]) {
            for bytes in encode_snapshot_chunks(self.format, 0, epoch, words) {
                let frame = decode(self.format, &bytes).unwrap();
                if let Some((_, images)) = self.assembly.absorb(&frame).unwrap() {
                    self.words.fill(0);
                    for image in images {
                        self.words[image.bucket as usize] = image.value;
                    }
                }
            }
        }

        fn delta(&mut self, bytes: &[u8]) {
            match decode(self.format, bytes).unwrap() {
                ImagesFrame::Delta { images, .. } => {
                    for image in images {
                        self.words[image.bucket as usize] = image.value;
                    }
                }
                other => panic!("expected delta, got {other:?}"),
            }
        }
    }

    fn drive(state: &Arc<StdMutex<ModelFilterState>>, consumer: &mut Consumer) {
        match tick_publish(state) {
            Publish::Full { epoch, words, .. } => consumer.full(epoch, &words),
            Publish::Delta(bytes) => consumer.delta(&bytes),
            Publish::Unchanged => {}
        }
    }

    #[test]
    fn stored_blocks_replicate_through_delta() {
        let filters = FilterRegistry::new(256);
        update_model_filter(&filters, &model(), vec![stored_event(0..8)], config());
        let state = filters.model_state(7).unwrap();
        let format = lock_filter(&state).format();
        let mut consumer = Consumer::new(format);

        drive(&state, &mut consumer);

        let (_, _, expected) = lock_filter(&state).snapshot();
        assert_eq!(consumer.words, expected.as_ref());
        assert_eq!(lock_filter(&state).block_count(), 8);
    }

    /// Removing blocks lowers the replicated depth: absolute images zero the
    /// affected buckets without a lane reset.
    #[test]
    fn removal_lowers_replicated_depth() {
        let filters = FilterRegistry::new(256);
        update_model_filter(&filters, &model(), vec![stored_event(0..8)], config());
        let state = filters.model_state(7).unwrap();
        let format = lock_filter(&state).format();
        let mut consumer = Consumer::new(format);
        drive(&state, &mut consumer);

        update_model_filter(&filters, &model(), vec![removed_event(4..8)], config());
        drive(&state, &mut consumer);

        let (_, _, expected) = lock_filter(&state).snapshot();
        assert_eq!(consumer.words, expected.as_ref());
        assert_eq!(lock_filter(&state).block_count(), 4);
    }

    /// A snapshot taken mid-stream plus the subsequent delta reproduce the full
    /// state — the base-epoch/current-epoch handoff a fresh subscriber relies
    /// on.
    #[test]
    fn snapshot_then_delta_converges() {
        let filters = FilterRegistry::new(256);
        update_model_filter(&filters, &model(), vec![stored_event(0..4)], config());
        let state = filters.model_state(7).unwrap();
        let format = lock_filter(&state).format();
        let mut consumer = Consumer::new(format);
        drive(&state, &mut consumer);

        // A fresh subscriber bootstraps from the current snapshot...
        let (snap_format, snap_epoch, words) = lock_filter(&state).snapshot();
        let mut fresh = Consumer::new(snap_format);
        fresh.full(snap_epoch, &words);
        // ...then applies the next live delta.
        update_model_filter(&filters, &model(), vec![stored_event(4..6)], config());
        drive(&state, &mut fresh);

        let (_, _, expected) = lock_filter(&state).snapshot();
        assert_eq!(fresh.words, expected.as_ref());
        assert_eq!(lock_filter(&state).block_count(), 6);
    }
}
