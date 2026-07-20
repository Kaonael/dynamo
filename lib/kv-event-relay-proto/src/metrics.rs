// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep the relay telemetry schema small and engine-agnostic so the global
//! router can score capacity and health without depending on frontend internals.

/// Per-DC × per-model telemetry snapshot.
///
/// Optional fields stay optional because some aggregates are undefined under
/// zero traffic or missing capacity, and the router falls back when a signal is
/// absent.
#[derive(Debug, Clone, PartialEq)]
pub struct DcModelMetricsSnapshot {
    pub dc_id: String,
    pub model_id: String,
    /// Separate monotonic counters make metrics recovery independent of the
    /// event stream.
    pub seq: u64,
    /// Capture time lets the router age out stale telemetry.
    pub captured_at_unix_ms: u64,

    pub queue_depth: u32,

    pub ttft_p50_ms: Option<f32>,
    pub ttft_p95_ms: Option<f32>,
    pub ttft_p99_ms: Option<f32>,
    pub itl_p50_ms: Option<f32>,
    pub itl_p95_ms: Option<f32>,
    pub itl_p99_ms: Option<f32>,

    pub gpu_util_pct: Option<f32>,
    pub kv_util_pct: Option<f32>,
    /// Server-side error rate stays separate from client-side noise so bad
    /// routing signals don't get conflated with cancellations.
    pub server_error_rate: Option<f32>,

    /// Live worker instances for this model, from the relay's discovery
    /// watchers rather than metrics scrape — it reacts in k8s-watch time
    /// and stays defined under zero traffic, where the error rate (a
    /// per-request signal) can never register a fully dead pool.
    /// `Some(0)` is an explicit "no workers" the router must act on;
    /// `None` means the relay has no discovery datum for this model.
    pub worker_count: Option<u32>,
    /// Whether the DC-local frontend answered its health probe. The
    /// frontend registers nothing in discovery, so worker_count cannot
    /// see it die; this is the probe-based complement.
    pub frontend_healthy: Option<bool>,
}

impl DcModelMetricsSnapshot {
    /// A snapshot with identity only — no signals, so the router learns the
    /// (DC, model) tuple without biasing decisions before data arrives.
    pub fn empty(dc_id: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            dc_id: dc_id.into(),
            model_id: model_id.into(),
            seq: 0,
            captured_at_unix_ms: 0,
            queue_depth: 0,
            ttft_p50_ms: None,
            ttft_p95_ms: None,
            ttft_p99_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: None,
            itl_p99_ms: None,
            gpu_util_pct: None,
            kv_util_pct: None,
            server_error_rate: None,
            worker_count: None,
            frontend_healthy: None,
        }
    }
}
