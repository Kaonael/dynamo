// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Periodic emission of `DcModelMetricsSnapshot` frames on the `metrics_tx`
//! broadcast (served by the gRPC `SubscribeMetrics` stream).
//!
//! Two data sources merge into each tick's snapshots:
//!
//!   * a [`MetricSource`](super::source::MetricSource) — the DC's in-cluster
//!     Prometheus over PromQL (optional; latency/queue/util fields);
//!   * DC-local readiness — live worker counts from the discovery-fed
//!     [`ModelRegistry`] and frontend liveness from
//!     [`FrontendHealth`](crate::frontend_health::FrontendHealth) (also
//!     discovery-fed). These are deliberately not PromQL: they must keep
//!     flowing when the metrics stack (or the frontend that feeds it) is
//!     down, which is exactly when the router needs them.
//!
//! The `seq` counter is **per-model**, **per-relay-process**, and independent of
//! the KV-event sequence, so the global side detects gaps per model
//! independently.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use dynamo_kv_event_relay_proto::metrics::DcModelMetricsSnapshot;

use crate::clock::{unix_micros, unix_ms};
use crate::frontend_health::FrontendHealth;
use crate::model_registry::ModelRegistry;

use super::catalog::{QueryCatalog, collect_snapshots};
use super::source::PromQlClient;

/// Long-lived publisher loop. Wakes every `interval`, assembles one snapshot
/// per known model (PromQL catalog fields when a source is configured, plus
/// readiness), packs each into a `wire::MetricFrame`, and publishes on the
/// `metrics_tx` broadcast. Exits on cancellation.
#[allow(clippy::too_many_arguments)]
pub async fn run_metrics_publisher(
    metrics_tx: broadcast::Sender<dynamo_kv_event_relay_proto::MetricsSnapshot>,
    source: Option<PromQlClient>,
    catalog: QueryCatalog,
    models: Arc<ModelRegistry>,
    frontend_health: Arc<FrontendHealth>,
    dc_id: String,
    interval: Duration,
    cancel: CancellationToken,
) -> Result<()> {
    let mut seqs: HashMap<String, u64> = HashMap::new();
    // Models that ever had a worker, a frontend association, or a PromQL
    // series. A model whose pool drains to zero vanishes from the registry
    // and from Prometheus — this set is what lets us still report
    // worker_count=0 for it instead of going silent (silence reads as
    // "healthy" downstream).
    let mut known_models: HashSet<String> = HashSet::new();
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    tracing::info!(
        dc_id = %dc_id,
        interval_ms = interval.as_millis(),
        promql = source.is_some(),
        "metrics publisher started"
    );

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                tracing::info!("metrics publisher cancelled");
                return Ok(());
            }
            _ = ticker.tick() => {
                let now_ms = unix_ms();
                let prom_snapshots = match &source {
                    Some(source) =>
                        collect_snapshots(source, &catalog, &dc_id, now_ms, &mut seqs).await,
                    None => Vec::new(),
                };
                let worker_counts = models.worker_counts().await;
                let frontend_by_model = frontend_health.health_by_model().await;
                let snapshots = merge_readiness(
                    prom_snapshots,
                    &worker_counts,
                    &frontend_by_model,
                    &mut known_models,
                    &mut seqs,
                    &dc_id,
                    now_ms,
                );
                for snap in snapshots {
                    publish_one(&metrics_tx, snap);
                }
            }
        }
    }
}

/// Stamp readiness onto the PromQL snapshots and synthesize snapshots for
/// known models PromQL didn't cover this tick (metrics stack down, model
/// fully drained, or PromQL disabled). Every known model gets an explicit
/// `worker_count` — `Some(0)` is the signal the router excludes on.
fn merge_readiness(
    prom_snapshots: Vec<DcModelMetricsSnapshot>,
    worker_counts: &HashMap<String, u32>,
    frontend_health: &HashMap<String, bool>,
    known_models: &mut HashSet<String>,
    seqs: &mut HashMap<String, u64>,
    dc_id: &str,
    now_ms: u64,
) -> Vec<DcModelMetricsSnapshot> {
    known_models.extend(worker_counts.keys().cloned());
    known_models.extend(frontend_health.keys().cloned());
    known_models.extend(prom_snapshots.iter().map(|s| s.model_id.clone()));

    let mut by_model: HashMap<String, DcModelMetricsSnapshot> = prom_snapshots
        .into_iter()
        .map(|s| (s.model_id.clone(), s))
        .collect();
    for model in known_models.iter() {
        let snap = by_model.entry(model.clone()).or_insert_with(|| {
            let mut snap = DcModelMetricsSnapshot::empty(dc_id, model);
            let seq = seqs.entry(model.clone()).or_insert(0);
            *seq = seq.wrapping_add(1);
            snap.seq = *seq;
            snap.captured_at_unix_ms = now_ms;
            snap
        });
        snap.worker_count = Some(worker_counts.get(model).copied().unwrap_or(0));
        snap.frontend_healthy = frontend_health.get(model).copied();
    }
    by_model.into_values().collect()
}

fn publish_one(
    metrics_tx: &broadcast::Sender<dynamo_kv_event_relay_proto::MetricsSnapshot>,
    snap: DcModelMetricsSnapshot,
) {
    // Pack only the metrics actually present into a `MetricFrame` (see `wire`).
    // dc_id is implied by the per-DC connection and model identity travels as
    // `model_key`, so neither string rides here.
    let payload = dynamo_kv_event_relay_proto::wire::to_frame(&snap).encode();
    let send_ts_us = unix_micros();
    let frame = dynamo_kv_event_relay_proto::MetricsSnapshot {
        seq: snap.seq,
        send_ts_us,
        model_key: dynamo_kv_event_relay_proto::wire::model_id_to_key(&snap.model_id),
        payload: payload.into(),
        // Stamped per-stream by the gRPC layer on the first frame only.
        instance_id: bytes::Bytes::new(),
    };
    // `send` errs only when no receivers — fine; subscribers come and go.
    let _ = metrics_tx.send(frame);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(model: &str, seq: u64) -> DcModelMetricsSnapshot {
        let mut s = DcModelMetricsSnapshot::empty("dc1", model);
        s.seq = seq;
        s.captured_at_unix_ms = 1_000;
        s.ttft_p50_ms = Some(100.0);
        s
    }

    /// PromQL snapshots get readiness stamped in place; a probe-only model
    /// with no workers gets a synthesized snapshot with an explicit zero.
    #[test]
    fn merge_stamps_prom_snapshots_and_synthesizes_for_known_models() {
        let mut known: HashSet<String> = HashSet::from(["probed".to_string()]);
        let mut seqs = HashMap::from([("live".to_string(), 5u64)]);
        let counts = HashMap::from([("live".to_string(), 2u32)]);
        let health = HashMap::from([("probed".to_string(), false)]);

        let out = merge_readiness(
            vec![snap("live", 5)],
            &counts,
            &health,
            &mut known,
            &mut seqs,
            "dc1",
            2_000,
        );

        let by_model: HashMap<_, _> = out.into_iter().map(|s| (s.model_id.clone(), s)).collect();
        let live = &by_model["live"];
        assert_eq!(live.worker_count, Some(2));
        assert_eq!(live.frontend_healthy, None);
        assert_eq!(live.ttft_p50_ms, Some(100.0));
        assert_eq!(live.seq, 5);

        let probed = &by_model["probed"];
        assert_eq!(probed.worker_count, Some(0));
        assert_eq!(probed.frontend_healthy, Some(false));
        assert_eq!(probed.seq, 1);
        assert_eq!(probed.captured_at_unix_ms, 2_000);

        assert!(known.contains("live"));
    }

    /// A model whose workers all die disappears from both PromQL and the
    /// registry — the sticky known-model set must keep reporting an explicit
    /// worker_count=0 for it, with seq still advancing.
    #[test]
    fn drained_model_keeps_reporting_zero() {
        let mut known = HashSet::new();
        let mut seqs = HashMap::new();

        // Tick 1: model alive.
        let counts = HashMap::from([("m".to_string(), 1u32)]);
        let out = merge_readiness(
            Vec::new(),
            &counts,
            &HashMap::new(),
            &mut known,
            &mut seqs,
            "dc1",
            1_000,
        );
        assert_eq!(out[0].worker_count, Some(1));
        let first_seq = out[0].seq;

        // Tick 2: workers gone, PromQL empty — snapshot still flows.
        let out = merge_readiness(
            Vec::new(),
            &HashMap::new(),
            &HashMap::new(),
            &mut known,
            &mut seqs,
            "dc1",
            2_000,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].model_id, "m");
        assert_eq!(out[0].worker_count, Some(0));
        assert_eq!(out[0].seq, first_seq + 1);
    }
}
