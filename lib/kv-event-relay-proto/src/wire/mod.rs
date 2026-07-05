// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep the transport-private encodings isolated so relay and router share one
//! compact contract without coupling the domain types together.

mod metrics;
mod model;

pub use metrics::{MetricDecodeError, MetricFrame, from_frame, metric_key, to_frame};
pub use model::model_id_to_key;

#[cfg(test)]
use crate::metrics::DcModelMetricsSnapshot;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_to_key_is_deterministic_and_distinct() {
        assert_eq!(model_id_to_key("llama"), model_id_to_key("llama"));
        assert_ne!(model_id_to_key("llama"), model_id_to_key("qwen"));
        assert_eq!(model_id_to_key("llama"), 11_279_576_970_053_723_506);
    }

    #[test]
    fn metric_frame_round_trips() {
        let mut f = MetricFrame::new();
        f.set(metric_key::QUEUE_DEPTH, 12.0);
        f.set(metric_key::TTFT_P50_MS, 120.5);
        f.set(metric_key::QUEUE_DEPTH, 13.0);

        let bytes = f.encode();
        assert_eq!(
            bytes,
            [
                0x02, 0x00, // sample count
                0x00, 0x00, // queue-depth key
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a, 0x40, // 13.0
                0x01, 0x00, // TTFT p50 key
                0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x5e, 0x40, // 120.5
            ]
        );
        let back = MetricFrame::decode(&bytes).expect("decode");
        assert_eq!(back.get(metric_key::QUEUE_DEPTH), Some(13.0));
        assert_eq!(back.get(metric_key::TTFT_P50_MS), Some(120.5));
        assert_eq!(back.get(metric_key::GPU_UTIL_PCT), None);
        assert_eq!(back.samples.len(), 2);
    }

    #[test]
    fn metric_frame_preserves_unknown_keys() {
        let mut f = MetricFrame::new();
        f.set(60000, 1.0);
        f.set(metric_key::QUEUE_DEPTH, 4.0);
        let back = MetricFrame::decode(&f.encode()).expect("decode");
        assert_eq!(back.get(60000), Some(1.0));
        assert_eq!(back.get(metric_key::QUEUE_DEPTH), Some(4.0));
    }

    #[test]
    fn metric_frame_rejects_truncated() {
        assert_eq!(MetricFrame::decode(&[]), Err(MetricDecodeError::Truncated));
        assert_eq!(
            MetricFrame::decode(&[1, 0]),
            Err(MetricDecodeError::LengthMismatch {
                count: 1,
                body_len: 0,
            })
        );
    }

    #[test]
    fn snapshot_frame_round_trips_populated_keys() {
        let mut snap = DcModelMetricsSnapshot::empty("dc1", "llama");
        snap.queue_depth = 12;
        snap.ttft_p50_ms = Some(120.5);
        snap.ttft_p99_ms = Some(800.0);
        snap.itl_p50_ms = Some(35.0);
        snap.gpu_util_pct = Some(72.0);
        snap.kv_util_pct = Some(50.0);
        snap.server_error_rate = None; // absent -> not emitted

        let frame = to_frame(&snap);
        assert_eq!(frame.get(metric_key::SERVER_ERROR_RATE), None);

        let back = from_frame(&frame, "dc1", "llama");
        assert_eq!(back.queue_depth, 12);
        assert_eq!(back.ttft_p50_ms, Some(120.5));
        assert_eq!(back.ttft_p95_ms, None);
        assert_eq!(back.ttft_p99_ms, Some(800.0));
        assert_eq!(back.itl_p50_ms, Some(35.0));
        assert_eq!(back.gpu_util_pct, Some(72.0));
        assert_eq!(back.kv_util_pct, Some(50.0));
        assert_eq!(back.server_error_rate, None);
        assert_eq!(back.worker_count, None);
        assert_eq!(back.frontend_healthy, None);
    }

    /// `Some(0)` / `Some(false)` are load-bearing "definitely down" values —
    /// they must survive the round trip distinct from absent.
    #[test]
    fn readiness_zero_values_round_trip_distinct_from_absent() {
        let mut snap = DcModelMetricsSnapshot::empty("dc1", "llama");
        snap.worker_count = Some(0);
        snap.frontend_healthy = Some(false);

        let back = from_frame(&to_frame(&snap), "dc1", "llama");
        assert_eq!(back.worker_count, Some(0));
        assert_eq!(back.frontend_healthy, Some(false));

        let mut snap = DcModelMetricsSnapshot::empty("dc1", "llama");
        snap.worker_count = Some(3);
        snap.frontend_healthy = Some(true);
        let back = from_frame(&to_frame(&snap), "dc1", "llama");
        assert_eq!(back.worker_count, Some(3));
        assert_eq!(back.frontend_healthy, Some(true));
    }
}
