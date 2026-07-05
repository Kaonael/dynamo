// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guard the allocation profile of the producer's insert/remove hot path.
//!
//! The relay calls `SnapshotProducer::insert` once per stored block, so this
//! path must stay allocation-free — including when both candidate buckets are
//! full and the insert resolves through a cuckoo kick chain, which is the
//! common case at steady-state load factors.

mod common;

use std::sync::atomic::Ordering;

use common::{ALLOCATED_BYTES, ALLOCATIONS, CountingAllocator, spread};
use dynamo_kv_event_relay_proto::cuckoo::{DEFAULT_FILTER_SEED, SnapshotProducer};

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

const ENTRIES: usize = 1_000_000;
/// Overfill past the provisioning target so the load factor sits at the upper
/// end of the steady-state range and kick chains are frequent.
const FILL: usize = ENTRIES * 3 / 2;
const CHURN: usize = 100_000;

#[test]
fn producer_churn_does_not_allocate() {
    let mut producer = SnapshotProducer::new(7, ENTRIES, DEFAULT_FILTER_SEED);
    for value in 0..FILL as u64 {
        assert!(
            producer.insert(spread(value)),
            "overfilled filter unexpectedly hit Full"
        );
    }

    let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
    let allocated_before = ALLOCATED_BYTES.load(Ordering::Relaxed);

    for value in 0..CHURN as u64 {
        producer.remove(spread(value));
        assert!(
            producer.insert(spread(FILL as u64 + value)),
            "churn insert overflowed"
        );
    }

    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations_before;
    let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed) - allocated_before;
    println!(
        "churn: {CHURN} removes+inserts over {} resident: allocations={allocations} bytes={allocated}",
        producer.len(),
    );

    // The only legitimate allocations here are amortized capacity doublings of
    // the filter's reusable `kick_scratch` buffer, which is bounded by
    // MAX_KICKS and grows a handful of times over the filter's lifetime.
    // Per-insert allocations (one per kicked insert, ~16% of inserts at this
    // load) blow through this budget by three orders of magnitude.
    const ALLOCATION_BUDGET: u64 = 8;
    const BYTE_BUDGET: u64 = 64 * 1024;
    assert!(
        allocations <= ALLOCATION_BUDGET && allocated <= BYTE_BUDGET,
        "insert/remove hot path allocated {allocations} times ({allocated} bytes) \
         during {CHURN} churn operations (budget: {ALLOCATION_BUDGET} allocations, \
         {BYTE_BUDGET} bytes); kicked inserts must not allocate per insert"
    );
}
