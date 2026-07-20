// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::{
    indexer::WorkerKvQueryResponse,
    protocols::{DpRank, RouterEvent, WorkerId},
    recovery::CursorState,
};

use crate::events::worker_state::{FetchOutcome, WorkerStateSource};
use crate::model_registry::ModelIdentity;
use crate::state::IngestContext;

/// Cold-start handshake for one advertised `(worker_id, dp_rank)` source:
/// query its recovery target and replay the returned `TreeDump` into
/// `state.dedup`. Side effect: `cursors` is populated so the message
/// loop's `CursorState::observe` will correctly classify transport-
/// buffered re-deliveries (`event_id <= TreeDump.last_event_id`) as
/// `Stale`.
///
/// Best-effort: if any step fails we log and continue with empty
/// state for that slot. Natural events that arrive afterward will
/// re-populate it; the cost of a missed bootstrap is the same
/// transient inconsistency the relay had before this feature
/// existed.
///
/// Mirrors `apply_tree_dump_replace_locked` in
/// `lib/llm/src/kv_router/indexer/worker_query.rs`. The wipe
/// is necessary because `state.dedup` may already hold partial state
/// for this slot — either from a previous run of this same
/// subscriber that was cancelled mid-flight, or from stray live
/// events between worker discovery and bootstrap.
pub(crate) async fn bootstrap_worker_slot(
    source: &impl WorkerStateSource,
    component: &str,
    worker_id: WorkerId,
    dp_rank: DpRank,
    state: &Arc<IngestContext>,
    cursors: &mut HashMap<(WorkerId, DpRank), CursorState>,
) {
    match source.fetch(worker_id, dp_rank).await {
        FetchOutcome::NoEndpoint | FetchOutcome::QueryFailed => {
            record_bootstrap_rank(state, "failed");
        }
        FetchOutcome::Response(
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            }
            | WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            },
        ) => {
            let count = events.len();
            let Some(model) = state.models.resolve_once(component, worker_id).await else {
                record_bootstrap_rank(state, "failed");
                tracing::warn!(
                    worker_id,
                    dp_rank,
                    "bootstrap: model no longer registered; skipping dump replay"
                );
                return;
            };
            // Evict the old slot state WITH a corrective Removed for any
            // block that lost its last holder, then replay the dump. Using the
            // silent `clear_worker_dp_for_model` here would drop those blocks
            // from dedup without telling the gateway, leaving them resident in
            // the CKF forever (and a later real Removed skipped on was_held).
            // Ordering: the corrective Removed must ship before the re-added
            // Stored events.
            let mut corrective: Vec<RouterEvent> = Vec::new();
            if let Some(removed) = state.dedup.evict_slot_forwarding_for_model(
                model.model_key,
                worker_id,
                dp_rank,
                last_event_id,
            ) {
                corrective.push(removed);
            }
            for event in &events {
                if let Some(stored) = state.dedup.process_event_for_model(model.model_key, event) {
                    corrective.push(stored);
                }
            }
            if !corrective.is_empty() {
                state.event_publisher.publish_batch(model, corrective);
            }
            cursors.insert((worker_id, dp_rank), CursorState::Live(last_event_id));
            record_bootstrap_rank(state, "replayed");
            tracing::info!(
                worker_id,
                dp_rank,
                events = count,
                last_event_id,
                "bootstrap: replayed worker dump into state.dedup"
            );
        }
        FetchOutcome::Response(WorkerKvQueryResponse::TooNew {
            newest_available, ..
        }) => {
            record_bootstrap_rank(state, "failed");
            tracing::warn!(
                worker_id,
                dp_rank,
                newest_available,
                "bootstrap: requested range too new; skipping"
            );
        }
        FetchOutcome::Response(WorkerKvQueryResponse::InvalidRange { start_id, end_id }) => {
            record_bootstrap_rank(state, "failed");
            tracing::warn!(
                worker_id,
                dp_rank,
                start_id,
                end_id,
                "bootstrap: invalid range from worker"
            );
        }
        FetchOutcome::Response(WorkerKvQueryResponse::Error(msg)) => {
            record_bootstrap_rank(state, "failed");
            tracing::warn!(
                worker_id, dp_rank, error = %msg,
                "bootstrap: worker returned error"
            );
        }
        FetchOutcome::Response(WorkerKvQueryResponse::TreeDumpFailed {
            last_event_id,
            message,
        }) => {
            record_bootstrap_rank(state, "failed");
            tracing::warn!(
                worker_id, dp_rank, last_event_id, error = %message,
                "bootstrap: worker tree dump failed; slot heals from live events"
            );
        }
        // The upstream response enum is #[non_exhaustive]; treat variants this
        // relay predates as a failed bootstrap and heal from live events.
        FetchOutcome::Response(other) => {
            record_bootstrap_rank(state, "failed");
            tracing::warn!(
                worker_id,
                dp_rank,
                response = ?other,
                "bootstrap: unhandled worker response variant; skipping"
            );
        }
    }
}

/// Record a per-dp-rank bootstrap outcome on the relay metrics, if
/// instrumentation is enabled. `result` ∈ {`replayed`, `failed`}.
fn record_bootstrap_rank(state: &IngestContext, result: &str) {
    if let Some(m) = &state.metrics {
        m.bootstrap_rank_total.with_label_values(&[result]).inc();
    }
}

/// Gap recovery (Phase 4): re-fetch the worker's authoritative state
/// for `(worker_id, dp_rank)` and forward a corrective delta so the
/// global gateway's index re-syncs.
///
/// Procedure (all dedup mutation under one lock):
///   1. evict the slot's current blocks, forwarding a `Removed` for any
///      that no longer have a holder anywhere in the DC;
///   2. replay the fresh dump, forwarding `Stored` for blocks newly
///      first-seen in the DC.
///
/// Multi-holder blocks correctly produce no forward; blocks uniquely
/// held by this slot churn (Removed then Stored) but converge.
///
/// Returns `true` when the worker was re-queried and the cursor
/// advanced to the dump's `last_event_id`; `false` when the worker was
/// unreachable (caller falls back to advancing past the gap).
pub(crate) async fn recover_gap_for_slot(
    source: &impl WorkerStateSource,
    component: &str,
    worker_id: WorkerId,
    dp_rank: u32,
    state: &Arc<IngestContext>,
    cursors: &mut HashMap<(WorkerId, DpRank), CursorState>,
) -> bool {
    let dp = dp_rank as DpRank;
    let (events, last_event_id) = match source.fetch(worker_id, dp_rank).await {
        FetchOutcome::Response(
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            }
            | WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            },
        ) => (events, last_event_id),
        _ => return false,
    };

    let Some(model) = state.models.resolve_once(component, worker_id).await else {
        tracing::warn!(
            worker_id,
            dp_rank,
            "gap recovery: model no longer registered; skipping corrective delta"
        );
        return false;
    };

    let mut corrective: Vec<RouterEvent> = Vec::new();
    if let Some(removed) =
        state
            .dedup
            .evict_slot_forwarding_for_model(model.model_key, worker_id, dp, last_event_id)
    {
        corrective.push(removed);
    }
    for event in &events {
        if let Some(stored) = state.dedup.process_event_for_model(model.model_key, event) {
            corrective.push(stored);
        }
    }
    let forwarded = corrective.len();
    if !corrective.is_empty() {
        state.event_publisher.publish_batch(model, corrective);
    }
    cursors.insert((worker_id, dp), CursorState::Live(last_event_id));
    tracing::info!(
        worker_id,
        dp_rank,
        dump_events = events.len(),
        forwarded,
        last_event_id,
        "gap recovery: re-synced slot and forwarded corrective delta"
    );
    true
}

/// Evict exactly one advertised KV source slot. A worker may publish multiple
/// DP ranks independently, so removing one publisher must not drop its sibling
/// ranks from the DC residency view.
pub(crate) async fn evict_departed_slot(
    state: &Arc<IngestContext>,
    component: &str,
    worker_id: WorkerId,
    dp_rank: DpRank,
) {
    for (model_key, corrective) in state
        .dedup
        .evict_worker_dp_forwarding(worker_id, dp_rank, 0)
    {
        let model_id = state
            .filters
            .model_id(model_key)
            .unwrap_or_else(|| Arc::from(component));
        tracing::info!(
            component,
            worker_id,
            dp_rank,
            model_key,
            "departed KV source slot evicted; forwarding corrective Removed"
        );
        state.event_publisher.publish_batch(
            ModelIdentity {
                model_id,
                model_key,
            },
            vec![corrective],
        );
    }
}
