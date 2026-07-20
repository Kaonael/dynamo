// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Periodic filter publisher: per model a coalesced CBI1 delta when buckets
//! changed, a full snapshot after a lane reset (drain) or when a delta would
//! exceed one frame, a heartbeat otherwise. A full snapshot rides the
//! broadcast as one `Arc` to the lane words and is serialized into CBI1
//! chunks per-subscriber in the gRPC stream, so no per-subscriber copy
//! happens here. `lock_publication` is deliberately held across
//! `tick_publish` — including the (rare, Full-only) copy of the lane words —
//! because the epoch cut, the seq assignment, and the frame send must stay
//! atomic with respect to `subscribe_with_snapshot`; releasing the lock
//! around the copy would let a fresh subscriber bootstrap at an epoch whose
//! delta it then receives as a spurious `EpochGap`.

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use dynamo_kv_event_relay_proto::wire::images::{IMAGES_HEADER_LEN, SNAPSHOT_CHUNK_BUCKETS};

use crate::clock::unix_micros;
use crate::observability::RelayMetrics;

use super::rebuild::{Publish, tick_publish};
use super::registry::{FilterFrame, FilterRegistry};

/// Bytes and chunk count a full-lane snapshot of `bucket_count` words will
/// materialize into on the wire (one CBI1 chunk frame per
/// `SNAPSHOT_CHUNK_BUCKETS` words).
fn snapshot_chunk_lens(bucket_count: usize) -> impl Iterator<Item = usize> {
    let full_chunks = bucket_count / SNAPSHOT_CHUNK_BUCKETS;
    let remainder = bucket_count % SNAPSHOT_CHUNK_BUCKETS;
    (0..full_chunks)
        .map(|_| IMAGES_HEADER_LEN + 16 + SNAPSHOT_CHUNK_BUCKETS * 8)
        .chain((remainder != 0).then_some(IMAGES_HEADER_LEN + 16 + remainder * 8))
}

fn publish_filter_tick(
    filters: &FilterRegistry,
    metrics: &Option<Arc<RelayMetrics>>,
    send_ts_us: u64,
) {
    for (model_key, model_state) in filters.model_states() {
        let mut publication = filters.lock_publication();
        let publish_started = std::time::Instant::now();
        let output = tick_publish(&model_state);
        if let Some(metrics) = metrics {
            metrics
                .filter_publish_seconds
                .observe(publish_started.elapsed().as_secs_f64());
        }
        let frame = match output {
            Publish::Unchanged => {
                record_frame_metrics(metrics, "heartbeat", std::iter::once(0));
                FilterFrame::Heartbeat {
                    model_key,
                    seq: publication.next_seq(),
                    send_ts_us,
                }
            }
            Publish::Delta(bytes) => {
                record_frame_metrics(metrics, "delta", std::iter::once(bytes.len()));
                record_delta_buckets(metrics, &bytes);
                FilterFrame::Delta {
                    model_key,
                    seq: publication.next_seq(),
                    send_ts_us,
                    bytes: bytes.into(),
                }
            }
            Publish::Full {
                format,
                epoch,
                words,
            } => {
                // Account each chunk the subscriber stream will materialize,
                // sized from the layout so we never serialize under the lock.
                record_frame_metrics(metrics, "full", snapshot_chunk_lens(words.len()));
                FilterFrame::Full {
                    model_key,
                    seq: publication.next_seq(),
                    send_ts_us,
                    format,
                    epoch,
                    words,
                }
            }
        };
        let _ = publication.sender.send(frame);
    }
}

/// Periodic filter publisher. The mirror exactly tracks the published lane, so
/// periodic full snapshots are unnecessary: a consumer that detects any desync
/// re-subscribes (the stream prepends full snapshots on connect). Each tick's
/// CPU-bound encoding runs on a blocking worker.
pub async fn run_filter_publisher(
    filters: Arc<FilterRegistry>,
    metrics: Option<Arc<RelayMetrics>>,
    interval: Duration,
    cancel: CancellationToken,
) {
    let mut tick = tokio::time::interval(interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            _ = tick.tick() => {
                let filters = filters.clone();
                let metrics = metrics.clone();
                let send_ts_us = unix_micros();
                if let Err(error) = tokio::task::spawn_blocking(move || {
                    publish_filter_tick(&filters, &metrics, send_ts_us);
                })
                .await
                {
                    tracing::error!(%error, "filter publish tick panicked");
                }
            }
        }
    }
}

/// One `filter_updates_total`+`filter_update_bytes` observation per wire frame
/// the subscriber stream will emit for this publish (`byte_lens` yields one
/// length per frame). A full snapshot rides the broadcast as a single `Arc`, so
/// its per-chunk bytes are accounted here from the layout, not from serialized
/// buffers.
fn record_frame_metrics(
    metrics: &Option<Arc<RelayMetrics>>,
    kind: &'static str,
    byte_lens: impl Iterator<Item = usize>,
) {
    let Some(m) = metrics else {
        return;
    };
    for len in byte_lens {
        m.filter_updates_total.with_label_values(&[kind]).inc();
        m.filter_update_bytes
            .with_label_values(&[kind])
            .observe(len as f64);
    }
}

/// A CBI1 delta body is `base_epoch: u64` then `image count: u32`; read the
/// count to record how many buckets this delta carried.
fn record_delta_buckets(metrics: &Option<Arc<RelayMetrics>>, delta: &[u8]) {
    if let Some(m) = metrics {
        let offset = IMAGES_HEADER_LEN + 8;
        if delta.len() >= offset + 4 {
            let changed = u32::from_le_bytes(delta[offset..offset + 4].try_into().unwrap());
            m.filter_delta_buckets.observe(changed as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::rebuild::update_model_filter;
    use crate::filter::test_support::{model, stored_event};
    use dynamo_kv_event_relay_proto::wire::images::{ImagesFrame, SnapshotAssembly, decode};
    use dynamo_kv_router::indexer::cuckoo::CkfConfig;

    /// A fresh subscribe snapshot is the base for the first live delta: apply
    /// the bootstrap chunks then the delta and the consumer matches the
    /// producer's resident prefix.
    #[test]
    fn subscription_snapshot_is_base_for_first_live_delta() {
        let filters = FilterRegistry::new(256);
        let config = CkfConfig::new(4096);
        update_model_filter(&filters, &model(), vec![stored_event(0..4)], config);
        publish_filter_tick(&filters, &None, 1);

        update_model_filter(&filters, &model(), vec![stored_event(4..5)], config);
        let mut subscription = filters.subscribe_with_snapshot();
        assert_eq!(subscription.snapshots.len(), 1);
        let snap = subscription.snapshots.pop().unwrap();
        let format = snap.format;

        let mut consumer_words = vec![0u64; format.bucket_count];
        let mut assembly = SnapshotAssembly::new(format);
        for bytes in dynamo_kv_event_relay_proto::wire::images::encode_snapshot_chunks(
            format,
            0,
            snap.epoch,
            &snap.words,
        ) {
            let frame = decode(format, &bytes).unwrap();
            if let Some((_, images)) = assembly.absorb(&frame).unwrap() {
                consumer_words.fill(0);
                for image in images {
                    consumer_words[image.bucket as usize] = image.value;
                }
            }
        }

        publish_filter_tick(&filters, &None, 2);
        let frame = subscription.receiver.try_recv().expect("first live update");
        let FilterFrame::Delta { bytes, .. } = frame else {
            panic!("first live frame after a full publish must be a delta");
        };
        match decode(format, &bytes).unwrap() {
            ImagesFrame::Delta { images, .. } => {
                for image in images {
                    consumer_words[image.bucket as usize] = image.value;
                }
            }
            other => panic!("expected delta, got {other:?}"),
        }

        let state = filters.model_state(7).unwrap();
        let (_, _, expected) = super::super::rebuild::lock_filter(&state).snapshot();
        assert_eq!(consumer_words, expected.as_ref());
    }
}
