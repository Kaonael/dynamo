// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::indexer::cuckoo::CkfConfig;
use dynamo_kv_router::protocols::RouterEvent;

use crate::filter::{FilterRegistry, update_model_filter};
use crate::model_registry::ModelIdentity;
use crate::observability::RelayMetrics;

/// Terminal sink of the ingest pipeline: folds each deduplicated event batch
/// into the per-model CKF pipeline that `SubscribeFilter` ships to routers.
pub struct EventPublisher {
    filters: Arc<FilterRegistry>,
    ckf_config: CkfConfig,
    metrics: Option<Arc<RelayMetrics>>,
}

impl EventPublisher {
    pub fn new(
        filters: Arc<FilterRegistry>,
        ckf_config: CkfConfig,
        metrics: Option<Arc<RelayMetrics>>,
    ) -> Self {
        Self {
            filters,
            ckf_config,
            metrics,
        }
    }

    pub fn publish_batch(&self, model: ModelIdentity, events: Vec<RouterEvent>) {
        if let Some(metrics) = &self.metrics {
            metrics.event_batch_events.observe(events.len() as f64);
            metrics.forwarded_batches_total.inc();
        }
        tracing::debug!(model_id = %model.model_id, events = events.len(), "batch forwarded");
        let apply_errors = update_model_filter(&self.filters, &model, events, self.ckf_config);
        if apply_errors > 0
            && let Some(metrics) = &self.metrics
        {
            metrics.ckf_apply_errors_total.inc_by(apply_errors as u64);
        }
    }
}
