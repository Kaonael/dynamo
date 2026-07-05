// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Guard the allocation profile of delta publication.
//!
//! Building a delta must not buffer the payload more than once: the producer
//! already pays one `Vec` for the sorted dirty-bucket list and one for the
//! final frame, and anything beyond that is avoidable memcpy that scales with
//! `MAX_DELTA_BYTES` (32 MiB worst case, every publish tick, per model).
//!
//! Allocation counts are deterministic for a fixed fixture, so unlike a timing
//! benchmark this can assert hard bounds and run in CI.

mod common;

use std::sync::atomic::Ordering;

use common::{
    ALLOCATED_BYTES, ALLOCATIONS, CountingAllocator, LIVE_BYTES, PEAK_LIVE_BYTES, spread,
};
use dynamo_kv_event_relay_proto::cuckoo::{DEFAULT_FILTER_SEED, Publish, SnapshotProducer};

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

const ENTRIES: usize = 1_000_000;
/// Churn enough hashes to dirty ~20% of the buckets: a large delta that still
/// stays under the `churn_wants_full` switch (dirty ≥ num_buckets / 3).
const CHURN: usize = ENTRIES / 16;

/// The publish path legitimately owns two payload-sized buffers: the sorted
/// dirty-bucket list (8 bytes per dirty bucket vs 12 on the wire) and the
/// delta frame itself. Entry headroom covers dirty buckets that compare
/// net-unchanged and get dropped from the frame.
const BUDGET_FACTOR_NUM: u64 = 5;
const BUDGET_FACTOR_DEN: u64 = 2;
/// Fixed slack for small side allocations that don't scale with the delta.
const BUDGET_SLACK: u64 = 64 * 1024;

#[test]
fn delta_publish_allocates_at_most_one_extra_payload_buffer() {
    let mut producer = SnapshotProducer::new(7, ENTRIES, DEFAULT_FILTER_SEED);
    for value in 0..ENTRIES as u64 {
        assert!(
            producer.insert(spread(value)),
            "provisioned filter overflowed"
        );
    }
    // Baseline ship, then churn CHURN resident hashes so the next publish has
    // to serialize a large delta.
    drop(producer.full_snapshot());
    for value in 0..CHURN as u64 {
        producer.remove(spread(value));
        assert!(
            producer.insert(spread(ENTRIES as u64 + value)),
            "churn insert overflowed"
        );
    }

    let allocations_before = ALLOCATIONS.load(Ordering::Relaxed);
    let allocated_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let live_before = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(live_before, Ordering::Relaxed);

    let delta = match producer.publish() {
        Publish::Delta(delta) => delta,
        Publish::Full(_) => panic!("churn fixture unexpectedly crossed the full-snapshot switch"),
        Publish::Unchanged => panic!("churn fixture produced no delta"),
    };

    let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed) - allocated_before;
    let allocation_count = ALLOCATIONS.load(Ordering::Relaxed) - allocations_before;
    let peak_extra = PEAK_LIVE_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(live_before);
    let delta_len = delta.len() as u64;

    println!(
        "delta publish: frame={delta_len} bytes, allocated={allocated} bytes ({:.2}x frame), \
         peak_extra={peak_extra} bytes ({:.2}x frame), allocations={allocation_count}",
        allocated as f64 / delta_len as f64,
        peak_extra as f64 / delta_len as f64,
    );

    // Sanity: the fixture really produced a multi-megabyte delta.
    assert!(
        delta_len > 1024 * 1024,
        "fixture delta unexpectedly small: {delta_len} bytes"
    );

    let budget = delta_len * BUDGET_FACTOR_NUM / BUDGET_FACTOR_DEN + BUDGET_SLACK;
    assert!(
        allocated <= budget,
        "delta build over-buffers: allocated {allocated} bytes for a {delta_len}-byte frame \
         (budget {budget}); the frame is being copied through intermediate buffers"
    );
    assert!(
        peak_extra <= budget,
        "delta build peak memory {peak_extra} bytes for a {delta_len}-byte frame \
         (budget {budget}); intermediate buffers are alive simultaneously"
    );
}
