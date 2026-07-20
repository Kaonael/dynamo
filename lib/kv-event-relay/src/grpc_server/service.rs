// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `KvEventRelay` tonic service. Holds exactly the transport handles each RPC
//! needs — `metrics_tx` and `FilterRegistry` for the live feeds, plus
//! instrumentation — rather than the process-wide ingest context.
//! Stream bodies live in the sibling [`super::streams`] module; the handlers
//! here are thin delegates.

use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use tokio::sync::broadcast;
use tonic::{Request, Response, Status};

use dynamo_kv_event_relay_proto::{
    KvEventRelay, MetricsSnapshot, RelayInfo, RelayInfoRequest, SubscribeRequest,
};

use crate::filter::FilterRegistry;
use crate::observability::RelayMetrics;
use crate::state::BlockSizeTracker;

use super::streams::{
    SubscribeFilterStream, SubscribeMetricsStream, broadcast_to_stream, filter_subscribe_stream,
};

pub struct KvEventRelayService {
    instance_id: bytes::Bytes,
    /// Metrics broadcast — `run_metrics_publisher` is the sole writer.
    metrics_tx: broadcast::Sender<MetricsSnapshot>,
    /// Per-model relay-side filters backing `SubscribeFilter`.
    filters: Arc<FilterRegistry>,
    /// Prometheus handles. `None` disables instrumentation (tests, or no
    /// metrics port).
    metrics: Option<Arc<RelayMetrics>>,
    /// DC-wide KV block size observed from worker MDCs, reported in
    /// `GetRelayInfo` so the global gateway can validate its topology.
    block_size: Arc<BlockSizeTracker>,
    /// Shared shutdown token so live subscribe streams end promptly on
    /// cancellation instead of blocking graceful shutdown.
    cancel: CancellationToken,
}

/// The subscriber id is a client-controlled string that lands in logs — cap
/// its length; `?`-formatting at the log sites escapes control characters.
fn sanitized_subscriber_id(raw: String) -> String {
    const MAX_LEN: usize = 128;
    if raw.chars().count() <= MAX_LEN {
        raw
    } else {
        raw.chars().take(MAX_LEN).collect()
    }
}

impl KvEventRelayService {
    pub fn new(
        instance_id: bytes::Bytes,
        metrics_tx: broadcast::Sender<MetricsSnapshot>,
        filters: Arc<FilterRegistry>,
        metrics: Option<Arc<RelayMetrics>>,
        block_size: Arc<BlockSizeTracker>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            instance_id,
            metrics_tx,
            filters,
            metrics,
            block_size,
            cancel,
        }
    }
}

#[tonic::async_trait]
impl KvEventRelay for KvEventRelayService {
    type SubscribeMetricsStream = SubscribeMetricsStream;
    type SubscribeFilterStream = SubscribeFilterStream;

    async fn get_relay_info(
        &self,
        _req: Request<RelayInfoRequest>,
    ) -> Result<Response<RelayInfo>, Status> {
        Ok(Response::new(RelayInfo {
            instance_id: self.instance_id.clone(),
            block_size: self.block_size.get(),
        }))
    }

    async fn subscribe_metrics(
        &self,
        req: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeMetricsStream>, Status> {
        let subscriber_id = sanitized_subscriber_id(req.into_inner().subscriber_id);
        tracing::info!(subscriber_id = ?subscriber_id, "SubscribeMetrics started");
        let rx = self.metrics_tx.subscribe();
        let s = broadcast_to_stream(
            rx,
            "metrics",
            subscriber_id,
            self.metrics.clone(),
            self.instance_id.clone(),
            self.cancel.clone(),
        );
        Ok(Response::new(Box::pin(s)))
    }

    async fn subscribe_filter(
        &self,
        req: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeFilterStream>, Status> {
        let subscriber_id = sanitized_subscriber_id(req.into_inner().subscriber_id);
        tracing::info!(subscriber_id = ?subscriber_id, "SubscribeFilter started");

        let subscription = self.filters.subscribe_with_snapshot();
        let s = filter_subscribe_stream(
            subscription.snapshots,
            subscription.receiver,
            self.instance_id.clone(),
            self.metrics.clone(),
            subscriber_id,
            self.cancel.clone(),
        );
        Ok(Response::new(Box::pin(s)))
    }
}
