// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-source KV-event subscriber: cold-start bootstrap from the source's
//! advertised `recovery_target`, then a long-lived endpoint-scoped event-plane
//! subscription with cursor-based gap detection.
//!
//! The event transport is whatever `dynamo-runtime` resolves for the
//! deployment — ZMQ (local backends / `DYN_EVENT_PLANE=zmq`) or NATS
//! (distributed backends). This module sits above the envelope layer
//! via [`EventSubscriber`], so gap detection (on the application-level
//! `event_id`) and recovery behave identically on either. ZMQ PUB/SUB
//! drops silently at its HWM with no replay, which is exactly why the
//! cursor + slot re-bootstrap path matters.
//!
//! The subscriber feeds [`RefCountedDedup`] (shared via
//! [`IngestContext`]) and publishes filtered batches into the relay's
//! per-model cuckoo filter.
//!
//! [`RefCountedDedup`]: crate::events::dedup::RefCountedDedup

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::{
    protocols::{DpRank, KV_EVENT_SUBJECT, RouterEvent, WorkerId, WorkerWithDpRank},
    recovery::{CursorObservation, CursorState},
};
use dynamo_runtime::{
    DistributedRuntime, component::Instance, protocols::EndpointId,
    transports::event_plane::EventSubscriber,
};

use crate::events::batcher::EventBatcher;
use crate::events::recovery::{bootstrap_worker_slot, recover_gap_for_slot};
use crate::events::worker_state::RuntimeWorkerStateSource;
use crate::model_registry::ModelIdentityCache;
use crate::state::IngestContext;

pub(crate) async fn run_source_subscriber(
    drt: DistributedRuntime,
    source_endpoint: EndpointId,
    worker: WorkerWithDpRank,
    recovery_target: Option<Instance>,
    state: Arc<IngestContext>,
    cancel: CancellationToken,
) -> Result<()> {
    let component = source_endpoint.component.clone();
    let worker_id = worker.worker_id;
    let worker_state = RuntimeWorkerStateSource::new(&drt, recovery_target.as_ref());
    // Subscribe BEFORE bootstrap so the transport starts buffering
    // events. Any event with `event_id <= last_event_id_from_dump`
    // arriving in the bootstrap window hits the pre-initialised cursor
    // and is classified `Stale` (filtered).
    // KV events ride the plane as batches (`Vec<RouterEvent>`, kv-router #11776),
    // so decode a batch per frame and fold each event below. Decoding a single
    // `RouterEvent` here mis-parses the batch frame ("wrong msgpack marker").
    let mut subscriber = EventSubscriber::for_endpoint_id(&drt, &source_endpoint, KV_EVENT_SUBJECT)
        .await?
        .typed::<Vec<RouterEvent>>();

    // Model identity comes from the MDC; without it events are not
    // attributable and would leak a phantom model key downstream. The MDC
    // watcher runs in parallel — block until it has seen this worker.
    tracing::info!(component = %component, worker_id, "kv-events subscriber waiting for MDC");
    if state
        .models
        .wait_registered(&component, worker_id, &cancel)
        .await
        .is_none()
    {
        return Ok(()); // cancelled while waiting
    }

    tracing::info!(component = %component, worker_id, "kv-events subscriber connected");

    let mut cursors: HashMap<(WorkerId, DpRank), CursorState> = HashMap::new();
    bootstrap_worker_slot(
        &worker_state,
        &component,
        worker_id,
        worker.dp_rank,
        &state,
        &mut cursors,
    )
    .await;

    // Optional event batching: accumulate dedup-filtered events per model and
    // flush on a timer or size threshold. `batch_window == 0` keeps the legacy
    // one-frame-per-event behaviour (the accumulator stays empty).
    let batching = !state.batch_window.is_zero();
    let mut model_cache = ModelIdentityCache::default();
    let mut batcher = EventBatcher::new(state.batch_max_events);
    let tick_period = if batching {
        state.batch_window
    } else {
        Duration::from_secs(3600)
    };
    let mut flush_tick =
        tokio::time::interval_at(tokio::time::Instant::now() + tick_period, tick_period);
    flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                batcher.flush(state.event_publisher.as_ref());
                tracing::info!(component = %component, "kv-events subscriber cancelled");
                return Ok(());
            }
            _ = flush_tick.tick() => {
                if batching {
                    batcher.flush(state.event_publisher.as_ref());
                }
            }
            msg = subscriber.next() => {
                let events = match msg {
                    Some(Ok((_envelope, evs))) => evs,
                    Some(Err(e)) => {
                        tracing::warn!(component = %component, error = %e, "kv-events recv error");
                        continue;
                    }
                    None => {
                        batcher.flush(state.event_publisher.as_ref());
                        tracing::info!(component = %component, "kv-events stream ended");
                        return Ok(());
                    }
                };

                for event in events {
                // The component topic carries every worker's events (and a
                // durable transport replays dead workers' history). One
                // subscriber instance exists per live worker — processing
                // only our own keeps duplicates and departed-worker ghosts
                // out of the dedup state.
                if event.worker_id != worker_id || event.event.dp_rank != worker.dp_rank {
                    continue;
                }

                // Gap detection. On a gap we re-bootstrap the affected
                // slot from the worker's authoritative dump and forward
                // a corrective delta to the global gateway (Phase 4).
                let key = (event.worker_id, event.event.dp_rank);
                let got = event.event.event_id;
                let observation = cursors.entry(key).or_default().observe(got);
                match observation {
                    CursorObservation::Initial { got }
                    | CursorObservation::Contiguous { got } => {
                        if let Some(c) = cursors.get_mut(&key) {
                            *c = c.advance_to(got);
                        }
                    }
                    CursorObservation::Stale { .. } => {
                        tracing::trace!(
                            component = %component,
                            worker_id = event.worker_id,
                            event_id = got,
                            "stale event ignored"
                        );
                        continue;
                    }
                    CursorObservation::Gap { expected, got } => {
                        if let Some(m) = &state.metrics {
                            m.gap_detected_total.inc();
                        }
                        tracing::warn!(
                            component = %component,
                            worker_id = event.worker_id,
                            dp_rank = event.event.dp_rank,
                            expected,
                            got,
                            "gap in upstream RouterEvent stream; attempting slot re-bootstrap"
                        );
                        // Preserve ordering: live events accumulated so far must
                        // ship (with earlier seq) before the corrective batch the
                        // recovery path forwards.
                        if batching {
                            batcher.flush(state.event_publisher.as_ref());
                        }
                        let recovered = recover_gap_for_slot(
                            &worker_state,
                            &component,
                            event.worker_id,
                            event.event.dp_rank,
                            &state,
                            &mut cursors,
                        )
                        .await;
                        if let Some(m) = &state.metrics {
                            let result = if recovered { "recovered" } else { "failed" };
                            m.gap_recovery_total.with_label_values(&[result]).inc();
                        }
                        if !recovered {
                            // Worker unreachable: advance past the gap so
                            // we don't re-trigger on every following
                            // event. Affected blocks heal on later
                            // eviction/re-add.
                            if let Some(c) = cursors.get_mut(&key) {
                                *c = c.advance_to(got);
                            }
                        }
                        // The triggering event is superseded by the fresh
                        // dump (or dropped on the unreachable fallback) —
                        // don't forward it on this path.
                        continue;
                    }
                }

                let Some(model) = state
                    .models
                    .resolve(&component, event.worker_id, &mut model_cache)
                    .await
                else {
                    // Model unregistered mid-flight (teardown window). Dropped
                    // blocks are reconciled by the departure eviction.
                    if let Some(m) = &state.metrics {
                        m.events_dropped_unresolved_total.inc();
                    }
                    tracing::debug!(
                        component = %component,
                        worker_id = event.worker_id,
                        "dropping event: model no longer registered"
                    );
                    continue;
                };
                let dedup_started = std::time::Instant::now();
                let filtered = state
                    .dedup
                    .process_event_for_model(model.model_key, &event);
                if let Some(metrics) = &state.metrics {
                    metrics.dedup_seconds.observe(dedup_started.elapsed().as_secs_f64());
                }
                let Some(filtered_event) = filtered else { continue };

                if batching {
                    if batcher.push(model, filtered_event) {
                        batcher.flush(state.event_publisher.as_ref());
                    }
                } else {
                    state.event_publisher.publish_batch(model, vec![filtered_event]);
                }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::test_ingest_context;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash,
    };

    fn stored_event(hashes: std::ops::Range<u64>) -> RouterEvent {
        RouterEvent::new(
            1,
            dynamo_kv_router::protocols::KvCacheEvent {
                event_id: 0,
                dp_rank: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: hashes
                        .map(|h| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(
                                h.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                            ),
                            tokens_hash: LocalBlockHash(h),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
            },
        )
    }

    fn stored_event_for_worker(worker: WorkerId, hashes: std::ops::Range<u64>) -> RouterEvent {
        let mut event = stored_event(hashes);
        event.worker_id = worker;
        event
    }

    /// A worker leaving discovery must produce a corrective `Removed` for the
    /// blocks only it held, and the model filter must shrink accordingly —
    /// otherwise dead workers' blocks poison routing until a relay restart
    /// (and resurrect from durable-transport replay even then).
    #[tokio::test(flavor = "multi_thread")]
    async fn departed_source_eviction_forwards_removed_and_clears_filter() {
        let (state, _filter_rx) = test_ingest_context();
        state
            .models
            .register("backend".to_string(), 7, "llama".to_string())
            .await;
        state
            .models
            .register("backend".to_string(), 9, "llama".to_string())
            .await;
        let model = state
            .models
            .resolve_once("backend", 7)
            .await
            .expect("registered");

        // Worker 7 holds blocks 0..8; worker 9 shares block 0.
        for (worker, hashes) in [(7u64, 0..8u64), (9u64, 0..1u64)] {
            let event = stored_event_for_worker(worker, hashes);
            let filtered = state
                .dedup
                .process_event_for_model(model.model_key, &event)
                .into_iter()
                .collect::<Vec<_>>();
            if !filtered.is_empty() {
                state.event_publisher.publish_batch(model.clone(), filtered);
            }
        }
        let resident_before = state.filters.resident_len(model.model_key).unwrap();
        assert_eq!(resident_before, 8);

        state.models.remove("backend", 7).await;
        crate::events::recovery::evict_departed_slot(&state, "backend", 7, 0).await;

        // Block 0 is still held by worker 9 — only the 7 unique blocks go.
        let resident_after = state.filters.resident_len(model.model_key).unwrap();
        assert_eq!(resident_after, 1);

        // Idempotent: a second eviction has nothing left to remove.
        crate::events::recovery::evict_departed_slot(&state, "backend", 7, 0).await;
        assert_eq!(state.filters.resident_len(model.model_key).unwrap(), 1);
    }
}
