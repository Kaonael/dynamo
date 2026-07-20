// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-model relay-side CKF pipeline. [`registry`] owns the per-model filter
//! map and the live publication channel; [`rebuild`] folds the deduped event
//! stream into each model's upstream `DcCkfState` and coalesces its
//! published bucket images; [`publisher`] periodically ships CBI1 snapshots /
//! deltas / heartbeats to `SubscribeFilter` consumers.

mod publisher;
mod rebuild;
mod registry;

pub use publisher::run_filter_publisher;
pub use registry::{FilterModelStats, FilterRegistry};

pub(crate) use rebuild::update_model_filter;
pub(crate) use registry::{FilterFrame, FilterSnapshot};

#[cfg(test)]
pub(crate) mod test_support {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
    };
    use std::sync::Arc;

    use crate::model_registry::ModelIdentity;

    pub(crate) fn stored_event(hashes: std::ops::Range<u64>) -> RouterEvent {
        RouterEvent::new(
            1,
            KvCacheEvent {
                event_id: 0,
                dp_rank: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: hashes
                        .map(|h| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(
                                h.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                            ),
                            tokens_hash: LocalBlockHash(h),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
            },
        )
    }

    pub(crate) fn removed_event(hashes: std::ops::Range<u64>) -> RouterEvent {
        RouterEvent::new(
            1,
            KvCacheEvent {
                event_id: 0,
                dp_rank: 0,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes
                        .map(|h| ExternalSequenceBlockHash(h.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
                        .collect(),
                }),
            },
        )
    }

    pub(crate) fn model() -> ModelIdentity {
        ModelIdentity {
            model_id: Arc::from("llama"),
            model_key: 7,
        }
    }
}
