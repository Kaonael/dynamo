// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DC-wide refcounted dedup state ([`RefCountedDedup`]).
//!
//! Multiple workers in a DC may cache the same block; the global
//! router only needs one `Stored` / `Removed` pair per
//! `(block_hash, DC)`. `RefCountedDedup` counts holders per
//! external `block_hash` so a `Stored` is forwarded the first time the DC
//! sees the block and a `Removed` only after the last worker drops it.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use rustc_hash::{FxHashMap, FxHashSet};

use dynamo_kv_router::protocols::{
    DpRank, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, RouterEvent, StorageTier, WorkerId,
};

/// Wrap a cache-event payload into a `RouterEvent` carrying the same
/// `worker_id`, `event_id`, `dp_rank`, and `storage_tier` as `src`.
/// Avoids cloning `src.event.data` — the dedup paths always replace
/// it with a filtered `Vec` anyway.
fn rewrap(src: &RouterEvent, data: KvCacheEventData) -> RouterEvent {
    RouterEvent::with_storage_tier(
        src.worker_id,
        KvCacheEvent {
            event_id: src.event.event_id,
            dp_rank: src.event.dp_rank,
            data,
        },
        src.storage_tier,
    )
}

const DEDUP_SHARDS: usize = 64;

#[derive(Default)]
struct DedupShard {
    refcounts: FxHashMap<(u64, u64), u32>,
    per_holder: FxHashMap<(u64, WorkerId, DpRank), FxHashSet<u64>>,
}

fn lock_shard(mutex: &Mutex<DedupShard>) -> MutexGuard<'_, DedupShard> {
    crate::state::lock_recovering(mutex, "dedup shard")
}

fn actual_holder_count(shard: &DedupShard, model_key: u64, hash: u64) -> u32 {
    let count = shard
        .per_holder
        .iter()
        .filter(|((model, _, _), holder)| *model == model_key && holder.contains(&hash))
        .count();
    u32::try_from(count).unwrap_or(u32::MAX)
}

fn increment_refcount(shard: &mut DedupShard, model_key: u64, hash: u64) -> bool {
    let key = (model_key, hash);
    let current = shard.refcounts.get(&key).copied().unwrap_or(0);
    let Some(next) = current.checked_add(1) else {
        let repaired = actual_holder_count(shard, model_key, hash);
        tracing::error!(
            hash,
            current,
            repaired,
            "dedup refcount overflow; repaired from holder state"
        );
        shard.refcounts.insert(key, repaired);
        return current == 0;
    };
    shard.refcounts.insert(key, next);
    next == 1
}

/// Decrement after the holder entry has already been removed. Returns true
/// when no holder remains and the caller should forward a Removed event.
fn decrement_refcount(shard: &mut DedupShard, model_key: u64, hash: u64) -> bool {
    let key = (model_key, hash);
    let current = shard.refcounts.get(&key).copied();
    match current.and_then(|count| count.checked_sub(1)) {
        Some(0) => {
            shard.refcounts.remove(&key);
            true
        }
        Some(next) => {
            shard.refcounts.insert(key, next);
            false
        }
        None => {
            let repaired = actual_holder_count(shard, model_key, hash);
            tracing::error!(
                hash,
                current = ?current,
                repaired,
                "dedup refcount invariant violated; repaired from holder state"
            );
            if repaired == 0 {
                shard.refcounts.remove(&key);
                true
            } else {
                shard.refcounts.insert(key, repaired);
                false
            }
        }
    }
}

pub struct RefCountedDedup {
    shards: [Mutex<DedupShard>; DEDUP_SHARDS],
}

impl Default for RefCountedDedup {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(DedupShard::default())),
        }
    }
}

impl RefCountedDedup {
    #[inline]
    fn shard_index(model_key: u64, hash: u64) -> usize {
        ((hash ^ model_key.rotate_left(32)) as usize) & (DEDUP_SHARDS - 1)
    }

    /// Apply a single event. Each block holds only its hash shard while its
    /// holder/refcount pair is updated; no critical section contains an await.
    pub fn process_event_for_model(&self, model_key: u64, ev: &RouterEvent) -> Option<RouterEvent> {
        let holder_key = (model_key, ev.worker_id, ev.event.dp_rank);
        match &ev.event.data {
            KvCacheEventData::Stored(store) => {
                let mut first_new = None;
                for (index, block) in store.blocks.iter().enumerate() {
                    let hash = block.block_hash.0;
                    let mut shard = lock_shard(&self.shards[Self::shard_index(model_key, hash)]);
                    let newly_held = shard.per_holder.entry(holder_key).or_default().insert(hash);
                    if newly_held
                        && increment_refcount(&mut shard, model_key, hash)
                        && first_new.is_none()
                    {
                        first_new = Some(index);
                    }
                }
                first_new.map(|index| {
                    let parent_hash = if index == 0 {
                        store.parent_hash
                    } else {
                        Some(store.blocks[index - 1].block_hash)
                    };
                    rewrap(
                        ev,
                        KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash,
                            start_position: store
                                .start_position
                                .and_then(|position| position.checked_add(index as u32)),
                            blocks: store.blocks[index..].to_vec(),
                        }),
                    )
                })
            }
            KvCacheEventData::Removed(remove) => {
                // Lazy: most Removes forward nothing (multi-holder DC), so defer
                // the allocation to the first block that loses its last holder
                // instead of reserving the full batch up front.
                let mut kept: Vec<ExternalSequenceBlockHash> = Vec::new();
                for block_hash in &remove.block_hashes {
                    let hash = block_hash.0;
                    let mut shard = lock_shard(&self.shards[Self::shard_index(model_key, hash)]);
                    let was_held = shard
                        .per_holder
                        .get_mut(&holder_key)
                        .is_some_and(|holder| holder.remove(&hash));
                    if !was_held {
                        continue;
                    }
                    if decrement_refcount(&mut shard, model_key, hash) {
                        kept.push(*block_hash);
                    }
                }
                (!kept.is_empty()).then(|| {
                    rewrap(
                        ev,
                        KvCacheEventData::Removed(KvCacheRemoveData { block_hashes: kept }),
                    )
                })
            }
            KvCacheEventData::Cleared => {
                let mut guards: Vec<_> = self.shards.iter().map(lock_shard).collect();
                let mut zeroed = Vec::new();
                for shard in &mut guards {
                    let blocks = shard.per_holder.remove(&holder_key).unwrap_or_default();
                    zeroed.reserve(blocks.len());
                    for hash in blocks {
                        if decrement_refcount(shard, model_key, hash) {
                            zeroed.push(ExternalSequenceBlockHash::from(hash));
                        }
                    }
                }
                let all_quiet = guards.iter().all(|shard| {
                    !shard
                        .per_holder
                        .keys()
                        .any(|(model, _, _)| *model == model_key)
                        && !shard.refcounts.keys().any(|(model, _)| *model == model_key)
                });
                if all_quiet {
                    Some(rewrap(ev, KvCacheEventData::Cleared))
                } else if zeroed.is_empty() {
                    None
                } else {
                    Some(rewrap(
                        ev,
                        KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: zeroed,
                        }),
                    ))
                }
            }
        }
    }

    pub fn evict_slot_forwarding_for_model(
        &self,
        model_key: u64,
        worker: WorkerId,
        dp: DpRank,
        event_id: u64,
    ) -> Option<RouterEvent> {
        let holder_key = (model_key, worker, dp);
        let mut removed = Vec::new();
        for shard in &self.shards {
            let mut shard = lock_shard(shard);
            let Some(blocks) = shard.per_holder.remove(&holder_key) else {
                continue;
            };
            removed.reserve(blocks.len());
            for hash in blocks {
                if decrement_refcount(&mut shard, model_key, hash) {
                    removed.push(ExternalSequenceBlockHash::from(hash));
                }
            }
        }
        if removed.is_empty() {
            return None;
        }
        Some(RouterEvent::with_storage_tier(
            worker,
            KvCacheEvent {
                event_id,
                dp_rank: dp,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: removed,
                }),
            },
            StorageTier::Device,
        ))
    }

    /// Remove one worker DP slot across every model without relying on model
    /// discovery, which may already have removed the worker attribution.
    pub fn evict_worker_dp_forwarding(
        &self,
        worker: WorkerId,
        dp: DpRank,
        event_id: u64,
    ) -> Vec<(u64, RouterEvent)> {
        let mut guards: Vec<_> = self.shards.iter().map(lock_shard).collect();
        let mut removed_by_model: BTreeMap<u64, Vec<ExternalSequenceBlockHash>> = BTreeMap::new();

        for shard in &mut guards {
            let holder_keys: Vec<_> = shard
                .per_holder
                .keys()
                .filter(|(_, holder_worker, holder_dp)| {
                    *holder_worker == worker && *holder_dp == dp
                })
                .copied()
                .collect();
            for holder_key @ (model_key, _, _) in holder_keys {
                let blocks = shard.per_holder.remove(&holder_key).unwrap_or_default();
                let removed = removed_by_model.entry(model_key).or_default();
                for hash in blocks {
                    if decrement_refcount(shard, model_key, hash) {
                        removed.push(ExternalSequenceBlockHash::from(hash));
                    }
                }
            }
        }

        removed_by_model
            .into_iter()
            .filter_map(|(model_key, block_hashes)| {
                (!block_hashes.is_empty()).then(|| {
                    (
                        model_key,
                        RouterEvent::with_storage_tier(
                            worker,
                            KvCacheEvent {
                                event_id,
                                dp_rank: dp,
                                data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
                            },
                            StorageTier::Device,
                        ),
                    )
                })
            })
            .collect()
    }

    /// Remove every slot owned by `worker` without relying on discovery state
    /// that may already have forgotten which models the worker served.
    pub fn evict_worker_forwarding(
        &self,
        worker: WorkerId,
        event_id: u64,
    ) -> Vec<(u64, RouterEvent)> {
        let mut guards: Vec<_> = self.shards.iter().map(lock_shard).collect();
        let mut removed_by_slot: BTreeMap<(u64, DpRank), Vec<ExternalSequenceBlockHash>> =
            BTreeMap::new();

        for shard in &mut guards {
            let holder_keys: Vec<_> = shard
                .per_holder
                .keys()
                .filter(|(_, holder_worker, _)| *holder_worker == worker)
                .copied()
                .collect();
            for holder_key @ (model_key, _, dp_rank) in holder_keys {
                let blocks = shard.per_holder.remove(&holder_key).unwrap_or_default();
                let removed = removed_by_slot.entry((model_key, dp_rank)).or_default();
                for hash in blocks {
                    if decrement_refcount(shard, model_key, hash) {
                        removed.push(ExternalSequenceBlockHash::from(hash));
                    }
                }
            }
        }

        removed_by_slot
            .into_iter()
            .filter_map(|((model_key, dp_rank), block_hashes)| {
                (!block_hashes.is_empty()).then(|| {
                    (
                        model_key,
                        RouterEvent::with_storage_tier(
                            worker,
                            KvCacheEvent {
                                event_id,
                                dp_rank,
                                data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
                            },
                            StorageTier::Device,
                        ),
                    )
                })
            })
            .collect()
    }

    #[cfg(test)]
    fn refcount(&self, hash: u64) -> Option<u32> {
        self.shards[Self::shard_index(0, hash)]
            .lock()
            .unwrap()
            .refcounts
            .get(&(0, hash))
            .copied()
    }

    #[cfg(test)]
    fn holder_contains(&self, holder: (WorkerId, DpRank), hash: u64) -> bool {
        self.shards[Self::shard_index(0, hash)]
            .lock()
            .unwrap()
            .per_holder
            .get(&(0, holder.0, holder.1))
            .is_some_and(|hashes| hashes.contains(&hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheRemoveData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash,
    };

    fn store_event(worker: WorkerId, dp: DpRank, tokens_hashes: &[u64]) -> RouterEvent {
        let blocks: Vec<(u64, u64)> = tokens_hashes.iter().map(|&h| (h, h)).collect();
        store_event_with_blocks(worker, dp, &blocks)
    }

    fn store_event_with_blocks(worker: WorkerId, dp: DpRank, blocks: &[(u64, u64)]) -> RouterEvent {
        RouterEvent::new(
            worker,
            KvCacheEvent {
                event_id: 0,
                dp_rank: dp,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: blocks
                        .iter()
                        .map(|&(block_hash, tokens_hash)| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(block_hash),
                            tokens_hash: LocalBlockHash(tokens_hash),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
            },
        )
    }

    fn remove_event(worker: WorkerId, dp: DpRank, block_hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker,
            KvCacheEvent {
                event_id: 0,
                dp_rank: dp,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: block_hashes
                        .iter()
                        .map(|&h| ExternalSequenceBlockHash(h))
                        .collect(),
                }),
            },
        )
    }

    fn removed_hashes(ev: Option<RouterEvent>) -> Vec<u64> {
        match ev.expect("expected forwarded event").event.data {
            KvCacheEventData::Removed(remove) => {
                remove.block_hashes.into_iter().map(|h| h.0).collect()
            }
            other => panic!("expected Removed event, got {other:?}"),
        }
    }

    fn stored_hashes(ev: Option<RouterEvent>) -> Vec<u64> {
        match ev.expect("expected forwarded event").event.data {
            KvCacheEventData::Stored(store) => {
                store.blocks.into_iter().map(|b| b.block_hash.0).collect()
            }
            other => panic!("expected Stored event, got {other:?}"),
        }
    }

    #[test]
    fn poisoned_shard_is_recovered() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let dedup = RefCountedDedup::default();
        let hash = 10;
        let shard = &dedup.shards[RefCountedDedup::shard_index(0, hash)];
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _guard = shard.lock().unwrap();
            panic!("poison test shard");
        }));
        assert!(panic.is_err());

        assert_eq!(
            stored_hashes(dedup.process_event_for_model(0, &store_event(1, 0, &[hash]))),
            vec![hash]
        );
        assert!(!shard.is_poisoned());
    }

    #[test]
    fn invalid_zero_refcount_is_repaired_without_panicking() {
        let dedup = RefCountedDedup::default();
        dedup.process_event_for_model(0, &store_event(1, 0, &[10]));
        {
            let mut shard = lock_shard(&dedup.shards[RefCountedDedup::shard_index(0, 10)]);
            shard.refcounts.insert((0, 10), 0);
        }

        assert_eq!(
            removed_hashes(dedup.process_event_for_model(0, &remove_event(1, 0, &[10]))),
            vec![10]
        );
        assert_eq!(dedup.refcount(10), None);
    }

    #[test]
    fn missing_refcount_is_repaired_from_remaining_holders() {
        let dedup = RefCountedDedup::default();
        dedup.process_event_for_model(0, &store_event(1, 0, &[10]));
        dedup.process_event_for_model(0, &store_event(2, 0, &[10]));
        {
            let mut shard = lock_shard(&dedup.shards[RefCountedDedup::shard_index(0, 10)]);
            shard.refcounts.remove(&(0, 10));
        }

        assert!(
            dedup
                .process_event_for_model(0, &remove_event(1, 0, &[10]))
                .is_none()
        );
        assert_eq!(dedup.refcount(10), Some(1));
        assert_eq!(
            removed_hashes(dedup.process_event_for_model(0, &remove_event(2, 0, &[10]))),
            vec![10]
        );
    }

    #[test]
    fn concurrent_overlapping_producers_match_reference_counts() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        const PRODUCERS: u64 = 32;
        let dedup = Arc::new(RefCountedDedup::default());
        let barrier = Arc::new(Barrier::new(PRODUCERS as usize));
        let mut handles = Vec::with_capacity(PRODUCERS as usize);
        for worker in 0..PRODUCERS {
            let dedup = dedup.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                dedup
                    .process_event_for_model(0, &store_event(worker, 0, &[10, 20, 30]))
                    .map(|event| stored_hashes(Some(event)))
                    .unwrap_or_default()
            }));
        }
        let mut stored: Vec<u64> = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        stored.sort_unstable();
        assert_eq!(stored, vec![10, 20, 30]);
        for hash in [10, 20, 30] {
            assert_eq!(dedup.refcount(hash), Some(PRODUCERS as u32));
        }

        let barrier = Arc::new(Barrier::new(PRODUCERS as usize));
        let mut handles = Vec::with_capacity(PRODUCERS as usize);
        for worker in 0..PRODUCERS {
            let dedup = dedup.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                dedup
                    .process_event_for_model(0, &remove_event(worker, 0, &[10, 20, 30]))
                    .map(|event| removed_hashes(Some(event)))
                    .unwrap_or_default()
            }));
        }
        let mut removed: Vec<u64> = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        removed.sort_unstable();
        assert_eq!(removed, vec![10, 20, 30]);
    }

    #[test]
    fn evict_slot_forwarding_removes_last_holder_blocks() {
        let d = RefCountedDedup::default();
        d.process_event_for_model(0, &store_event(1, 0, &[10, 20]));

        let mut hs = removed_hashes(d.evict_slot_forwarding_for_model(0, 1, 0, 99));
        hs.sort_unstable();
        assert_eq!(hs, vec![10, 20]);

        // Slot is empty now → nothing more to evict.
        assert!(d.evict_slot_forwarding_for_model(0, 1, 0, 100).is_none());
    }

    #[test]
    fn evict_slot_forwarding_keeps_multi_holder_blocks() {
        let d = RefCountedDedup::default();
        d.process_event_for_model(0, &store_event(1, 0, &[10, 20])); // slot A: 10, 20
        d.process_event_for_model(0, &store_event(2, 0, &[10])); // slot B also holds 10

        // Evicting A drops only blocks with no other holder: 20.
        // Block 10 still held by B → not forwarded.
        let hs = removed_hashes(d.evict_slot_forwarding_for_model(0, 1, 0, 99));
        assert_eq!(hs, vec![20]);
    }

    #[test]
    fn evict_slot_forwarding_none_for_unknown_slot() {
        let d = RefCountedDedup::default();
        assert!(d.evict_slot_forwarding_for_model(0, 7, 0, 1).is_none());
    }

    #[test]
    fn evict_then_restore_churns_block_for_global_resync() {
        // Models the gap-recovery delta: a uniquely-held block is
        // removed then re-stored, so the global gateway sees a clean
        // Removed+Stored pair (converging) rather than a stale entry.
        let d = RefCountedDedup::default();
        d.process_event_for_model(0, &store_event(1, 0, &[10]));

        assert_eq!(
            removed_hashes(d.evict_slot_forwarding_for_model(0, 1, 0, 99)),
            vec![10]
        );
        // Re-store is first-seen again → forwarded as Stored.
        assert_eq!(
            stored_hashes(d.process_event_for_model(0, &store_event(1, 0, &[10]))),
            vec![10]
        );
    }

    #[test]
    fn removed_decrements_by_external_block_hash_not_tokens_hash() {
        let d = RefCountedDedup::default();
        d.process_event_for_model(0, &store_event_with_blocks(1, 0, &[(10, 1000)]));

        let forwarded = d.process_event_for_model(0, &remove_event(1, 0, &[10]));

        assert_eq!(removed_hashes(forwarded), vec![10]);
        assert!(d.refcount(10).is_none());
        assert!(!d.holder_contains((1, 0), 10));
    }

    #[test]
    fn removed_waits_until_last_holder_drops_external_block_hash() {
        let d = RefCountedDedup::default();
        assert!(
            d.process_event_for_model(0, &store_event_with_blocks(1, 0, &[(10, 1000)]))
                .is_some()
        );
        assert!(
            d.process_event_for_model(0, &store_event_with_blocks(2, 0, &[(10, 1000)]))
                .is_none()
        );

        assert!(
            d.process_event_for_model(0, &remove_event(1, 0, &[10]))
                .is_none()
        );
        assert_eq!(d.refcount(10), Some(1));

        let forwarded = d.process_event_for_model(0, &remove_event(2, 0, &[10]));
        assert_eq!(removed_hashes(forwarded), vec![10]);
        assert!(d.refcount(10).is_none());
    }

    #[test]
    fn same_tokens_hash_with_different_external_block_hashes_are_distinct() {
        let d = RefCountedDedup::default();
        let forwarded =
            d.process_event_for_model(0, &store_event_with_blocks(1, 0, &[(10, 1000), (20, 1000)]));

        match forwarded.expect("expected forwarded Stored").event.data {
            KvCacheEventData::Stored(store) => {
                let hashes: Vec<u64> = store.blocks.into_iter().map(|b| b.block_hash.0).collect();
                assert_eq!(hashes, vec![10, 20]);
            }
            other => panic!("expected Stored event, got {other:?}"),
        }

        assert_eq!(d.refcount(10), Some(1));
        assert_eq!(d.refcount(20), Some(1));

        let forwarded = d.process_event_for_model(0, &remove_event(1, 0, &[10]));
        assert_eq!(removed_hashes(forwarded), vec![10]);
        assert!(d.refcount(10).is_none());
        assert_eq!(d.refcount(20), Some(1));
    }

    #[test]
    fn evict_slot_forwards_corrective_removed_for_dropped_blocks() {
        // Evicting a slot must forward a corrective Removed for exactly
        // the blocks that lost their last holder — not silently drop them.
        let d = RefCountedDedup::default();
        d.process_event_for_model(0, &store_event(1, 0, &[100, 200]));
        d.process_event_for_model(0, &store_event(2, 0, &[100, 300]));

        assert_eq!(d.refcount(100), Some(2));
        assert_eq!(d.refcount(200), Some(1));
        assert_eq!(d.refcount(300), Some(1));

        // Evict slot (1,0): B1(100) is still held by worker 2, B2(200) drops.
        let corrective = d
            .evict_slot_forwarding_for_model(0, 1, 0, 42)
            .expect("a block lost its last holder -> corrective Removed");
        assert_eq!(corrective.worker_id, 1);
        assert_eq!(corrective.event.event_id, 42);
        match corrective.event.data {
            KvCacheEventData::Removed(r) => {
                let hashes: Vec<u64> = r.block_hashes.iter().map(|h| h.0).collect();
                assert_eq!(hashes, vec![200], "only the last-holder-dropped block");
            }
            other => panic!("expected Removed, got {other:?}"),
        }

        assert_eq!(d.refcount(100), Some(1), "B1 still held by worker 2");
        assert!(
            d.refcount(200).is_none(),
            "B2 dropped when last holder evicted"
        );
        assert_eq!(d.refcount(300), Some(1), "B3 untouched");
        assert!(
            !d.holder_contains((1, 0), 100) || d.holder_contains((1, 0), 200),
            "slot entry removed"
        );
        assert!(
            d.holder_contains((2, 0), 100) || d.holder_contains((2, 0), 300),
            "other slot untouched"
        );
    }

    #[test]
    fn stored_suffix_keeps_existing_parent_context() {
        let dedup = RefCountedDedup::default();
        assert!(
            dedup
                .process_event_for_model(1, &store_event(1, 0, &[10]))
                .is_some()
        );

        let forwarded = dedup
            .process_event_for_model(1, &store_event(2, 0, &[10, 20]))
            .expect("new suffix must be forwarded");
        let KvCacheEventData::Stored(store) = forwarded.event.data else {
            panic!("expected Stored");
        };
        assert_eq!(store.parent_hash.map(|hash| hash.0), Some(10));
        assert_eq!(
            store
                .blocks
                .iter()
                .map(|block| block.block_hash.0)
                .collect::<Vec<_>>(),
            vec![20]
        );
    }

    #[test]
    fn same_block_hash_is_isolated_by_model() {
        let dedup = RefCountedDedup::default();
        let event = store_event_with_blocks(1, 0, &[(10, 1000)]);

        assert!(dedup.process_event_for_model(11, &event).is_some());
        assert!(dedup.process_event_for_model(22, &event).is_some());
        assert!(
            dedup
                .process_event_for_model(11, &remove_event(1, 0, &[10]))
                .is_some()
        );
        assert!(
            dedup
                .process_event_for_model(22, &remove_event(1, 0, &[10]))
                .is_some()
        );
    }

    #[test]
    fn worker_eviction_discovers_models_from_holder_state() {
        let dedup = RefCountedDedup::default();
        let event = store_event_with_blocks(7, 0, &[(10, 1000)]);
        dedup.process_event_for_model(11, &event);
        dedup.process_event_for_model(22, &event);

        let evictions = dedup.evict_worker_forwarding(7, 99);

        assert_eq!(
            evictions
                .iter()
                .map(|(model_key, _)| *model_key)
                .collect::<Vec<_>>(),
            vec![11, 22]
        );
        for (_, event) in evictions {
            assert_eq!(event.event.event_id, 99);
            assert_eq!(removed_hashes(Some(event)), vec![10]);
        }
        assert!(dedup.evict_worker_forwarding(7, 100).is_empty());
    }

    #[test]
    fn evict_slot_noop_on_unknown_slot() {
        let d = RefCountedDedup::default();
        d.process_event_for_model(0, &store_event(1, 0, &[100]));
        assert!(d.evict_slot_forwarding_for_model(0, 99, 5, 7).is_none());
        assert_eq!(d.refcount(100), Some(1));
    }
}
