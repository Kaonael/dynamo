// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The relay's own Prometheus instrumentation + a lightweight `/metrics` HTTP
//! sidecar — internal health, distinct from the upstream routing telemetry the
//! relay feeds the global gateway (see [`crate::telemetry`]).
//!
//! The gRPC server is mTLS-only and not a good fit for scrape traffic, so this
//! sidecar lives on a separate plaintext port (conventionally firewalled to the
//! monitoring network), mirroring the global gateway's metrics sidecar.
//!
//! Counters/gauges are recorded inline on the hot paths
//! (`publisher::publish_batch`, the gap arm, `bootstrap_worker_state`, and the
//! gRPC broadcast adapter). The filter-size *level* gauges are refreshed
//! lazily at scrape time from the `FilterRegistry` handle — see
//! `refresh_state_gauges` — so we never scan per-model CKF state on the
//! event path.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode, body::Incoming};
use hyper_util::rt::TokioIo;
use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGaugeVec, Opts, Registry,
    TextEncoder,
};

use crate::filter::FilterRegistry;

/// All relay Prometheus handles plus the registry they belong to.
/// Cloning is cheap (every field is `Arc`-backed internally).
#[derive(Clone)]
pub struct RelayMetrics {
    pub(crate) registry: Arc<Registry>,
    /// Per-dp-rank bootstrap results: `result` ∈ {`replayed`, `failed`}.
    /// `replayed` counts ranks whose TreeDump/Events were applied;
    /// `failed` counts ranks discovered but whose query errored/empty.
    /// A worker with `failed > 0` alongside `replayed > 0` means a
    /// partial cold start: some ranks recovered, some are serving a degraded
    /// (incomplete) view of this DC until they heal from live events.
    pub(crate) bootstrap_rank_total: IntCounterVec,
    /// Gaps observed in the upstream event-plane RouterEvent stream (ZMQ or NATS).
    pub(crate) gap_detected_total: IntCounter,
    /// Outcome of slot re-bootstrap after a gap: `result` ∈
    /// {`recovered`, `failed`}. `recovered` means the worker's current
    /// state was re-fetched and the corrective delta forwarded;
    /// `failed` means the worker was unreachable and the cursor was
    /// advanced past the gap (blocks heal on later eviction/re-add).
    pub(crate) gap_recovery_total: IntCounterVec,
    /// gRPC subscribers disconnected with `resource_exhausted` after a
    /// broadcast `Lagged`, by `channel` ∈ {`metrics`, `filter`}.
    pub(crate) subscriber_lagged_total: IntCounterVec,
    /// Currently-connected gRPC subscribers, by `channel`.
    pub(crate) active_subscribers: IntGaugeVec,
    /// Total deduplicated event batches folded into the per-model filters.
    pub(crate) forwarded_batches_total: IntCounter,
    /// Filter snapshots/deltas published on `SubscribeFilter`, by `kind` ∈
    /// {`full`, `delta`}. A `full`-heavy ratio signals frequent lane resets;
    /// zero growth signals a stalled producer.
    pub(crate) filter_updates_total: IntCounterVec,
    /// Blocks in the per-model producer cuckoo filter (refreshed at scrape
    /// time). Cross-check against the GG's `dc_index_blocks` for the same
    /// (dc, model): sustained divergence means silent filter drift.
    pub(crate) filter_blocks: IntGaugeVec,
    /// Events dropped because their model unregistered mid-flight (the
    /// subscriber gates on MDC, so this only counts the teardown window).
    pub(crate) events_dropped_unresolved_total: IntCounter,
    /// Events the CKF pipeline failed to apply — in practice capacity
    /// exhaustion of the fixed-size filter. Any nonzero rate means the
    /// published filter is missing blocks: raise `--filter-capacity-hint`.
    pub(crate) ckf_apply_errors_total: IntCounter,
    /// `notAfter` (unix seconds) of the TLS material loaded at startup, by
    /// `material` ∈ {`server_cert`, `client_ca`}. Rotation requires a restart,
    /// so this is constant per process — alert well before it passes now().
    pub(crate) tls_expiry_timestamp_seconds: IntGaugeVec,
    pub(crate) dedup_seconds: Histogram,
    pub(crate) filter_publish_seconds: Histogram,
    pub(crate) event_batch_events: Histogram,
    pub(crate) filter_update_bytes: HistogramVec,
    pub(crate) filter_delta_buckets: Histogram,
}

impl RelayMetrics {
    pub fn new() -> Result<Self> {
        let registry = Arc::new(Registry::new());

        let bootstrap_rank_total = IntCounterVec::new(
            Opts::new(
                "dynamo_kv_event_relay_bootstrap_rank_total",
                "Per-dp-rank cold-start bootstrap results (replayed|failed).",
            ),
            &["result"],
        )
        .context("constructing bootstrap_rank_total")?;
        let gap_detected_total = IntCounter::new(
            "dynamo_kv_event_relay_gap_detected_total",
            "Gaps detected in the upstream event-plane RouterEvent stream (ZMQ or NATS).",
        )
        .context("constructing gap_detected_total")?;
        let gap_recovery_total = IntCounterVec::new(
            Opts::new(
                "dynamo_kv_event_relay_gap_recovery_total",
                "Slot re-bootstrap outcomes after a gap (recovered|failed).",
            ),
            &["result"],
        )
        .context("constructing gap_recovery_total")?;
        let subscriber_lagged_total = IntCounterVec::new(
            Opts::new(
                "dynamo_kv_event_relay_subscriber_lagged_total",
                "gRPC subscribers dropped after a broadcast lag, by channel.",
            ),
            &["channel"],
        )
        .context("constructing subscriber_lagged_total")?;
        let active_subscribers = IntGaugeVec::new(
            Opts::new(
                "dynamo_kv_event_relay_active_subscribers",
                "Currently connected gRPC subscribers, by channel.",
            ),
            &["channel"],
        )
        .context("constructing active_subscribers")?;
        let forwarded_batches_total = IntCounter::new(
            "dynamo_kv_event_relay_forwarded_batches_total",
            "Total deduplicated event batches folded into the per-model filters.",
        )
        .context("constructing forwarded_batches_total")?;
        let filter_updates_total = IntCounterVec::new(
            Opts::new(
                "dynamo_kv_event_relay_filter_updates_total",
                "Filter snapshots/deltas published on SubscribeFilter, by kind.",
            ),
            &["kind"],
        )
        .context("constructing filter_updates_total")?;
        let filter_blocks = IntGaugeVec::new(
            Opts::new(
                "dynamo_kv_event_relay_filter_blocks",
                "Blocks in the per-model producer cuckoo filter; compare with the \
                 global gateway's dc_index_blocks to detect silent drift.",
            ),
            &["model_id"],
        )
        .context("constructing filter_blocks")?;
        let events_dropped_unresolved_total = IntCounter::new(
            "dynamo_kv_event_relay_events_dropped_unresolved_total",
            "Events dropped because their model unregistered mid-flight.",
        )
        .context("constructing events_dropped_unresolved_total")?;
        let ckf_apply_errors_total = IntCounter::new(
            "dynamo_kv_event_relay_ckf_apply_errors_total",
            "Events the CKF pipeline failed to apply (filter capacity exhausted); \
             the published filter is missing these blocks.",
        )
        .context("constructing ckf_apply_errors_total")?;
        let tls_expiry_timestamp_seconds = IntGaugeVec::new(
            Opts::new(
                "dynamo_kv_event_relay_tls_expiry_timestamp_seconds",
                "notAfter of the TLS material loaded at startup (rotation requires restart).",
            ),
            &["material"],
        )
        .context("constructing tls_expiry_timestamp_seconds")?;

        let latency_buckets = vec![
            0.000_001, 0.000_005, 0.000_01, 0.000_05, 0.000_1, 0.000_5, 0.001, 0.005, 0.01, 0.05,
            0.1, 0.5, 1.0,
        ];
        let make_latency = |name: &str, help: &str| {
            Histogram::with_opts(HistogramOpts::new(name, help).buckets(latency_buckets.clone()))
        };
        let dedup_seconds = make_latency(
            "dynamo_kv_event_relay_dedup_seconds",
            "Time spent applying one upstream event to refcounted dedup.",
        )?;
        let filter_publish_seconds = make_latency(
            "dynamo_kv_event_relay_filter_publish_seconds",
            "Time spent building one model filter update.",
        )?;
        let event_batch_events = Histogram::with_opts(
            HistogramOpts::new(
                "dynamo_kv_event_relay_event_batch_events",
                "Deduplicated events per outbound batch.",
            )
            .buckets(vec![
                1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
            ]),
        )?;
        let filter_update_bytes = HistogramVec::new(
            HistogramOpts::new(
                "dynamo_kv_event_relay_filter_update_bytes",
                "Encoded filter update bytes by full or delta kind.",
            )
            .buckets(prometheus::exponential_buckets(64.0, 2.0, 22)?),
            &["kind"],
        )?;
        let filter_delta_buckets = Histogram::with_opts(
            HistogramOpts::new(
                "dynamo_kv_event_relay_filter_delta_buckets",
                "Changed buckets per CKF delta.",
            )
            .buckets(prometheus::exponential_buckets(1.0, 2.0, 20)?),
        )?;

        registry.register(Box::new(bootstrap_rank_total.clone()))?;
        registry.register(Box::new(gap_detected_total.clone()))?;
        registry.register(Box::new(gap_recovery_total.clone()))?;
        registry.register(Box::new(subscriber_lagged_total.clone()))?;
        registry.register(Box::new(active_subscribers.clone()))?;
        registry.register(Box::new(forwarded_batches_total.clone()))?;
        registry.register(Box::new(filter_updates_total.clone()))?;
        registry.register(Box::new(filter_blocks.clone()))?;
        registry.register(Box::new(events_dropped_unresolved_total.clone()))?;
        registry.register(Box::new(ckf_apply_errors_total.clone()))?;
        registry.register(Box::new(tls_expiry_timestamp_seconds.clone()))?;
        registry.register(Box::new(dedup_seconds.clone()))?;
        registry.register(Box::new(filter_publish_seconds.clone()))?;
        registry.register(Box::new(event_batch_events.clone()))?;
        registry.register(Box::new(filter_update_bytes.clone()))?;
        registry.register(Box::new(filter_delta_buckets.clone()))?;

        Ok(Self {
            registry,
            bootstrap_rank_total,
            gap_detected_total,
            gap_recovery_total,
            subscriber_lagged_total,
            active_subscribers,
            forwarded_batches_total,
            filter_updates_total,
            filter_blocks,
            events_dropped_unresolved_total,
            ckf_apply_errors_total,
            tls_expiry_timestamp_seconds,
            dedup_seconds,
            filter_publish_seconds,
            event_batch_events,
            filter_update_bytes,
            filter_delta_buckets,
        })
    }

    /// RAII guard for a connected subscriber: bumps `active_subscribers`
    /// on creation and decrements it when the stream is dropped (client
    /// disconnect or graceful end). `channel` selects the label.
    pub(crate) fn subscriber_guard(&self, channel: &'static str) -> ActiveSubscriberGuard {
        let gauge = self.active_subscribers.with_label_values(&[channel]);
        gauge.inc();
        ActiveSubscriberGuard { gauge }
    }
}

pub(crate) struct ActiveSubscriberGuard {
    gauge: prometheus::IntGauge,
}

impl Drop for ActiveSubscriberGuard {
    fn drop(&mut self) {
        self.gauge.dec();
    }
}

/// Repopulate filter-size level gauges at scrape time so publication does not
/// pay the Prometheus update cost.
fn refresh_state_gauges(filters: &FilterRegistry, metrics: &RelayMetrics) {
    for stats in filters.stats() {
        metrics
            .filter_blocks
            .with_label_values(&[stats.model_id.as_ref()])
            .set(stats.blocks as i64);
    }
}

/// Serve `GET /metrics` (Prometheus text format) on `addr` until the
/// task is dropped. Bind failures are logged and end the task.
pub async fn serve_metrics(addr: SocketAddr, filters: Arc<FilterRegistry>, metrics: RelayMetrics) {
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(addr = %addr, error = %e, "relay metrics listener bind failed");
            return;
        }
    };
    tracing::info!(addr = %addr, "relay metrics endpoint listening");
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "relay metrics accept failed");
                continue;
            }
        };
        let filters = filters.clone();
        let metrics = metrics.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(move |req: Request<Incoming>| {
                let filters = filters.clone();
                let metrics = metrics.clone();
                async move { Ok::<_, std::convert::Infallible>(respond(req, filters, metrics).await) }
            });
            // The port is plaintext and often reachable beyond the scraper;
            // bound header reads so an idle half-open connection can't pin
            // a file descriptor indefinitely.
            if let Err(e) = http1::Builder::new()
                .timer(hyper_util::rt::TokioTimer::new())
                .header_read_timeout(std::time::Duration::from_secs(5))
                .serve_connection(io, svc)
                .await
            {
                tracing::debug!(error = %e, "relay metrics connection ended with error");
            }
        });
    }
}

async fn respond(
    req: Request<Incoming>,
    filters: Arc<FilterRegistry>,
    metrics: RelayMetrics,
) -> Response<Full<Bytes>> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/metrics") => {
            refresh_state_gauges(&filters, &metrics);
            let encoder = TextEncoder::new();
            match encoder.encode_to_string(&metrics.registry.gather()) {
                Ok(body) => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/plain; version=0.0.4")
                    .body(Full::new(Bytes::from(body)))
                    .expect("static response builds"),
                Err(e) => Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::new(Bytes::from(format!("encode error: {e}"))))
                    .expect("static response builds"),
            }
        }
        _ => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::new()))
            .expect("static response builds"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registers_all_series_and_encodes() {
        let m = RelayMetrics::new().expect("construct");
        m.forwarded_batches_total.inc();
        m.gap_detected_total.inc();
        m.bootstrap_rank_total
            .with_label_values(&["replayed"])
            .inc();
        m.subscriber_lagged_total
            .with_label_values(&["metrics"])
            .inc();
        // Label-vec gauges emit nothing until a child exists; keep one
        // alive across the encode so the family is rendered.
        let _sub = m.subscriber_guard("metrics");

        let body = TextEncoder::new()
            .encode_to_string(&m.registry.gather())
            .expect("encode");

        for name in [
            "dynamo_kv_event_relay_bootstrap_rank_total",
            "dynamo_kv_event_relay_gap_detected_total",
            "dynamo_kv_event_relay_subscriber_lagged_total",
            "dynamo_kv_event_relay_active_subscribers",
            "dynamo_kv_event_relay_forwarded_batches_total",
        ] {
            assert!(body.contains(name), "missing metric {name} in:\n{body}");
        }
    }

    #[test]
    fn subscriber_guard_increments_then_decrements() {
        let m = RelayMetrics::new().expect("construct");
        let gauge = m.active_subscribers.with_label_values(&["events"]);
        assert_eq!(gauge.get(), 0);
        {
            let _g = m.subscriber_guard("events");
            assert_eq!(m.active_subscribers.with_label_values(&["events"]).get(), 1);
        }
        assert_eq!(m.active_subscribers.with_label_values(&["events"]).get(), 0);
    }
}
