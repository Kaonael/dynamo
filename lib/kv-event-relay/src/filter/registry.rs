// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ownership of the per-model filter map and the live publication channel.
//! [`FilterRegistry`] serializes snapshot cuts with epoch advancement and live
//! publication so a subscriber never receives a delta older than its bootstrap
//! snapshot, and is the only door to the per-model [`ModelFilterState`] — the
//! map is never handed out, so callers reach state through narrow accessors.

use std::sync::{Arc, Mutex as StdMutex, RwLock as StdRwLock};

use rustc_hash::FxHashMap;

use tokio::sync::broadcast;

use dynamo_kv_event_relay_proto::wire::images::FilterFormat;
use dynamo_kv_router::indexer::cuckoo::CkfConfig;

use crate::model_registry::ModelIdentity;
use crate::state::{lock_recovering, read_recovering, write_recovering};

use super::rebuild::{ModelFilterState, lock_filter};

type ModelFilters = StdRwLock<FxHashMap<u64, Arc<StdMutex<ModelFilterState>>>>;

/// Internal broadcast item. A full snapshot rides as a single cheap `Arc`
/// reference to the lane words and is materialized into CBI1 chunks
/// per-subscriber in the stream, so the publisher never serializes a whole
/// filter while holding [`FilterRegistry::lock_publication`].
#[derive(Clone)]
pub(crate) enum FilterFrame {
    Full {
        model_key: u64,
        seq: u64,
        send_ts_us: u64,
        format: FilterFormat,
        epoch: u64,
        words: Arc<[u64]>,
    },
    Delta {
        model_key: u64,
        seq: u64,
        send_ts_us: u64,
        bytes: bytes::Bytes,
    },
    Heartbeat {
        model_key: u64,
        seq: u64,
        send_ts_us: u64,
    },
}

/// A fresh subscriber's per-model bootstrap: the current lane words plus the
/// format identity and epoch the stream stamps onto the snapshot chunks.
pub(crate) struct FilterSnapshot {
    pub model_key: u64,
    pub format: FilterFormat,
    pub epoch: u64,
    pub words: Arc<[u64]>,
}

pub(crate) struct FilterSubscription {
    pub snapshots: Vec<FilterSnapshot>,
    pub receiver: broadcast::Receiver<FilterFrame>,
}

/// Per-model observable counts captured under the model lock. Exposes only
/// numbers, never the `ModelFilterState`, so metrics/gRPC stay decoupled from
/// the filter internals.
pub struct FilterModelStats {
    pub model_id: Arc<str>,
    pub blocks: usize,
}

/// Serializes snapshot cuts with epoch advancement and live publication so a
/// subscriber never receives a delta older than its bootstrap snapshot.
pub struct FilterRegistry {
    models: ModelFilters,
    publication: StdMutex<FilterPublication>,
}

pub(crate) struct FilterPublication {
    pub(crate) sender: broadcast::Sender<FilterFrame>,
    pub(crate) next_seq: u64,
}

impl FilterPublication {
    /// Hand out the next monotonic frame sequence. Serialized under
    /// `lock_publication`, so a subscriber's bootstrap snapshot always precedes
    /// the first live frame it observes.
    pub(crate) fn next_seq(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        seq
    }
}

impl FilterRegistry {
    pub fn new(channel_capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(channel_capacity);
        Self {
            models: StdRwLock::new(FxHashMap::default()),
            publication: StdMutex::new(FilterPublication {
                sender,
                next_seq: 0,
            }),
        }
    }

    pub(crate) fn lock_publication(&self) -> std::sync::MutexGuard<'_, FilterPublication> {
        lock_recovering(&self.publication, "filter publication")
    }

    /// Hot-path entry from the deduped ingest stream: the existing per-model
    /// state or a freshly created one.
    pub(crate) fn get_or_create(
        &self,
        model: &ModelIdentity,
        config: CkfConfig,
    ) -> Arc<StdMutex<ModelFilterState>> {
        if let Some(existing) = read_recovering(&self.models, "filter models").get(&model.model_key)
        {
            return existing.clone();
        }
        write_recovering(&self.models, "filter models")
            .entry(model.model_key)
            .or_insert_with(|| {
                Arc::new(StdMutex::new(ModelFilterState::new(
                    model.model_id.clone(),
                    config,
                )))
            })
            .clone()
    }

    /// Snapshot of `(model_key, state)` pairs for one publisher tick. Cloning
    /// the `Arc`s out releases the map lock before any per-model work.
    pub(crate) fn model_states(&self) -> Vec<(u64, Arc<StdMutex<ModelFilterState>>)> {
        read_recovering(&self.models, "filter models")
            .iter()
            .map(|(&model_key, state)| (model_key, state.clone()))
            .collect()
    }

    pub(crate) fn model_id(&self, model_key: u64) -> Option<Arc<str>> {
        let model_state = self.models.read().ok()?.get(&model_key)?.clone();
        let model_id = lock_filter(&model_state).model_id.clone();
        Some(model_id)
    }

    pub(crate) fn subscribe_with_snapshot(&self) -> FilterSubscription {
        let publication = self.lock_publication();
        let receiver = publication.sender.subscribe();
        let model_states = self.model_states();
        let snapshots = model_states
            .into_iter()
            .map(|(model_key, model_state)| {
                let (format, epoch, words) = lock_filter(&model_state).snapshot();
                FilterSnapshot {
                    model_key,
                    format,
                    epoch,
                    words,
                }
            })
            .collect();
        FilterSubscription {
            snapshots,
            receiver,
        }
    }

    /// Per-model block counts for the scrape-time gauges. Returns numbers only
    /// — the model lock is taken and released inside.
    pub fn stats(&self) -> Vec<FilterModelStats> {
        self.model_states()
            .into_iter()
            .map(|(_, model_state)| {
                let state = lock_filter(&model_state);
                FilterModelStats {
                    model_id: state.model_id.clone(),
                    blocks: state.block_count(),
                }
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn model_state(&self, model_key: u64) -> Option<Arc<StdMutex<ModelFilterState>>> {
        read_recovering(&self.models, "filter models")
            .get(&model_key)
            .cloned()
    }

    #[cfg(test)]
    pub(crate) fn resident_len(&self, model_key: u64) -> Option<usize> {
        let model_state = read_recovering(&self.models, "filter models")
            .get(&model_key)?
            .clone();
        Some(lock_filter(&model_state).block_count())
    }
}
