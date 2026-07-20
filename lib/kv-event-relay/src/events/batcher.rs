// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use rustc_hash::FxHashMap;

use dynamo_kv_router::protocols::RouterEvent;

use super::publisher::EventPublisher;
use crate::model_registry::ModelIdentity;

pub(crate) struct EventBatcher {
    // FxHashMap: numeric model_key key, matching the lib/kv-router convention.
    pending: FxHashMap<u64, (Arc<str>, Vec<RouterEvent>)>,
    count: usize,
    max_events: usize,
}

impl EventBatcher {
    pub(crate) fn new(max_events: usize) -> Self {
        Self {
            pending: FxHashMap::default(),
            count: 0,
            max_events: max_events.max(1),
        }
    }

    pub(crate) fn push(&mut self, model: ModelIdentity, event: RouterEvent) -> bool {
        self.pending
            .entry(model.model_key)
            .or_insert_with(|| (model.model_id, Vec::new()))
            .1
            .push(event);
        self.count += 1;
        self.count >= self.max_events
    }

    pub(crate) fn flush(&mut self, sink: &EventPublisher) {
        self.count = 0;
        for (model_key, (model_id, events)) in self.pending.drain() {
            if !events.is_empty() {
                sink.publish_batch(
                    ModelIdentity {
                        model_id,
                        model_key,
                    },
                    events,
                );
            }
        }
    }
}
