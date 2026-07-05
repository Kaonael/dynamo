// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encode the telemetry fields the router actually scores in a compact wire
//! frame so relay and router share one small metric contract.

use crate::metrics::DcModelMetricsSnapshot;

/// Typed decode failures let the subscriber separate truncation from shape
/// mismatch without parsing strings.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MetricDecodeError {
    #[error("truncated metric frame")]
    Truncated,
    #[error("metric frame length mismatch: count={count} body={body_len}")]
    LengthMismatch { count: usize, body_len: usize },
}

/// Keep the metric keys stable so both sides can add fields without changing
/// the frame layout.
pub mod metric_key {
    pub const QUEUE_DEPTH: u16 = 0;
    pub const TTFT_P50_MS: u16 = 1;
    pub const TTFT_P95_MS: u16 = 2;
    pub const TTFT_P99_MS: u16 = 3;
    pub const ITL_P50_MS: u16 = 4;
    pub const ITL_P95_MS: u16 = 5;
    pub const ITL_P99_MS: u16 = 6;
    pub const GPU_UTIL_PCT: u16 = 7;
    pub const KV_UTIL_PCT: u16 = 8;
    pub const SERVER_ERROR_RATE: u16 = 9;
    pub const WORKER_COUNT: u16 = 10;
    /// Boolean on the wire: 0.0 = unhealthy, anything else = healthy.
    pub const FRONTEND_HEALTHY: u16 = 11;
}

/// Self-delimiting frames keep the metric payload forward-compatible and cheap
/// to parse.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricFrame {
    pub samples: Vec<(u16, f64)>,
}

impl MetricFrame {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: u16, value: f64) {
        if let Some(slot) = self.samples.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.samples.push((key, value));
        }
    }

    pub fn get(&self, key: u16) -> Option<f64> {
        self.samples
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.samples.len() * 10);
        out.extend_from_slice(&(self.samples.len() as u16).to_le_bytes());
        for (key, value) in &self.samples {
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, MetricDecodeError> {
        if bytes.len() < 2 {
            return Err(MetricDecodeError::Truncated);
        }
        let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        let body = &bytes[2..];
        if body.len() != count * 10 {
            return Err(MetricDecodeError::LengthMismatch {
                count,
                body_len: body.len(),
            });
        }
        let mut samples = Vec::with_capacity(count);
        for chunk in body.chunks_exact(10) {
            let key = u16::from_le_bytes([chunk[0], chunk[1]]);
            let value = f64::from_le_bytes(chunk[2..10].try_into().unwrap());
            samples.push((key, value));
        }
        Ok(Self { samples })
    }
}

/// Project the engine snapshot onto the compact wire form so transport stays
/// small.
pub fn to_frame(snapshot: &DcModelMetricsSnapshot) -> MetricFrame {
    let mut frame = MetricFrame::new();
    frame.set(metric_key::QUEUE_DEPTH, snapshot.queue_depth as f64);
    for (key, value) in [
        (metric_key::TTFT_P50_MS, snapshot.ttft_p50_ms),
        (metric_key::TTFT_P95_MS, snapshot.ttft_p95_ms),
        (metric_key::TTFT_P99_MS, snapshot.ttft_p99_ms),
        (metric_key::ITL_P50_MS, snapshot.itl_p50_ms),
        (metric_key::ITL_P95_MS, snapshot.itl_p95_ms),
        (metric_key::ITL_P99_MS, snapshot.itl_p99_ms),
        (metric_key::GPU_UTIL_PCT, snapshot.gpu_util_pct),
        (metric_key::KV_UTIL_PCT, snapshot.kv_util_pct),
        (metric_key::SERVER_ERROR_RATE, snapshot.server_error_rate),
    ] {
        if let Some(value) = value {
            frame.set(key, value as f64);
        }
    }
    if let Some(count) = snapshot.worker_count {
        frame.set(metric_key::WORKER_COUNT, count as f64);
    }
    if let Some(healthy) = snapshot.frontend_healthy {
        frame.set(metric_key::FRONTEND_HEALTHY, if healthy { 1.0 } else { 0.0 });
    }
    frame
}

/// Rebuild the snapshot using the identity supplied by the gRPC envelope.
pub fn from_frame(frame: &MetricFrame, dc_id: &str, model_id: &str) -> DcModelMetricsSnapshot {
    let mut snapshot = DcModelMetricsSnapshot::empty(dc_id, model_id);
    if let Some(value) = frame.get(metric_key::QUEUE_DEPTH) {
        snapshot.queue_depth = value as u32;
    }
    snapshot.ttft_p50_ms = frame.get(metric_key::TTFT_P50_MS).map(|v| v as f32);
    snapshot.ttft_p95_ms = frame.get(metric_key::TTFT_P95_MS).map(|v| v as f32);
    snapshot.ttft_p99_ms = frame.get(metric_key::TTFT_P99_MS).map(|v| v as f32);
    snapshot.itl_p50_ms = frame.get(metric_key::ITL_P50_MS).map(|v| v as f32);
    snapshot.itl_p95_ms = frame.get(metric_key::ITL_P95_MS).map(|v| v as f32);
    snapshot.itl_p99_ms = frame.get(metric_key::ITL_P99_MS).map(|v| v as f32);
    snapshot.gpu_util_pct = frame.get(metric_key::GPU_UTIL_PCT).map(|v| v as f32);
    snapshot.kv_util_pct = frame.get(metric_key::KV_UTIL_PCT).map(|v| v as f32);
    snapshot.server_error_rate = frame.get(metric_key::SERVER_ERROR_RATE).map(|v| v as f32);
    snapshot.worker_count = frame.get(metric_key::WORKER_COUNT).map(|v| v as u32);
    snapshot.frontend_healthy = frame.get(metric_key::FRONTEND_HEALTHY).map(|v| v != 0.0);
    snapshot
}
