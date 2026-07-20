// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Upstream routing telemetry the relay feeds the global gateway — distinct from
//! the relay's own Prometheus health (see [`crate::observability`]).
//!
//! [`source`] is the metric backend abstraction (a [`MetricSource`]; a PromQL
//! HTTP client in production, swappable and fakeable in tests); [`catalog`]
//! turns one tick's samples into per-model `DcModelMetricsSnapshot`s; and
//! [`publisher`] merges in discovery-fed readiness (worker counts + frontend
//! liveness) and broadcasts on `metrics_tx` for the gRPC `SubscribeMetrics`
//! stream.

pub mod catalog;
pub mod publisher;
pub mod source;

pub use catalog::{QueryCatalog, collect_snapshots};
pub use publisher::run_metrics_publisher;
pub use source::{MetricSource, PromQlClient, Sample};
