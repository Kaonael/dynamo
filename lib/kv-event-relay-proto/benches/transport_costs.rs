// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measure the remaining transport-copy, decode, and search costs so the
//! optimization work stays grounded in numbers.
//!
//! Run the quick defaults with:
//! `cargo bench -p dynamo-kv-event-relay-proto --bench transport_costs`
//!
//! Scale the filter and iteration counts with:
//! - `PROTO_BENCH_FILTER_ENTRIES` (default 1,000,000)
//! - `PROTO_BENCH_FILTER_ITERS` (default 50)
//! - `PROTO_BENCH_CHUNK_ITERS` (default 20)
//! - `PROTO_BENCH_DELTA_ITERS` (default 20)
//! - `PROTO_BENCH_CHURN_ITERS` (default 50)
//! - `PROTO_BENCH_SEARCH_DCS` (default 8)
//! - `PROTO_BENCH_SEARCH_DEPTH` (default 128)
//! - `PROTO_BENCH_SEARCH_ITERS` (default 20,000)

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use dynamo_kv_event_relay_proto::cuckoo::{
    CuckooFilter, DEFAULT_FILTER_SEED, Publish, SnapshotProducer, apply_delta, assemble_chunks,
    overlap_depth_searched, overlap_depth_searched_seq, probes_for,
};

// Reuse the allocation-budget tests' counting allocator so the bench reports
// the same counters those tests assert on.
#[path = "../tests/common/mod.rs"]
mod common;

use common::{
    ALLOCATED_BYTES, ALLOCATIONS, CountingAllocator, LIVE_BYTES, PEAK_LIVE_BYTES, spread,
};

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct AllocationSnapshot {
    count: u64,
    bytes: u64,
}

impl AllocationSnapshot {
    fn capture() -> Self {
        Self {
            count: ALLOCATIONS.load(Ordering::Relaxed),
            bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .max(1)
}

fn percentile(samples: &mut [f64], fraction: f64) -> f64 {
    samples.sort_by(f64::total_cmp);
    let index = ((samples.len() - 1) as f64 * fraction).round() as usize;
    samples[index]
}

fn measure(mut operation: impl FnMut(), name: &str, iterations: usize) {
    for _ in 0..3 {
        operation();
    }

    let mut samples = Vec::with_capacity(iterations);
    let allocations_before = AllocationSnapshot::capture();
    let live_before = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(live_before, Ordering::Relaxed);
    let total_started = Instant::now();
    for _ in 0..iterations {
        let started = Instant::now();
        operation();
        samples.push(started.elapsed().as_secs_f64() * 1e9);
    }
    let elapsed = total_started.elapsed();
    let allocations_after = AllocationSnapshot::capture();
    let allocated = allocations_after
        .bytes
        .saturating_sub(allocations_before.bytes);
    let allocation_count = allocations_after
        .count
        .saturating_sub(allocations_before.count);
    let peak_extra = PEAK_LIVE_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(live_before);
    let p50 = percentile(&mut samples, 0.50);
    let p99 = percentile(&mut samples, 0.99);
    let average = elapsed.as_secs_f64() * 1e9 / iterations as f64;

    println!(
        "{name:30} avg={average:12.0}ns p50={p50:12.0}ns p99={p99:12.0}ns alloc/op={:8.2} bytes/op={:14.0} peak+={peak_extra:10}",
        allocation_count as f64 / iterations as f64,
        allocated as f64 / iterations as f64,
    );
}

struct FilterFixture {
    producer: SnapshotProducer,
    base: CuckooFilter,
    base_epoch: u64,
    delta: Vec<u8>,
}

fn filter_fixture(entries: usize) -> FilterFixture {
    let mut producer = SnapshotProducer::new(7, entries, DEFAULT_FILTER_SEED);
    for value in 0..entries as u64 {
        assert!(
            producer.insert(spread(value)),
            "provisioned filter overflowed"
        );
    }
    let full = producer.full_snapshot();
    let (base, meta) = assemble_chunks(full.chunks()).expect("assemble baseline filter");

    assert!(producer.insert(spread(entries as u64 + 1)));
    let delta = match producer.publish() {
        Publish::Delta(delta) => delta,
        _ => panic!("single insert after full snapshot must produce a delta"),
    };
    FilterFixture {
        producer,
        base,
        base_epoch: meta.filter_epoch,
        delta,
    }
}

fn bench_filter_costs(entries: usize, iterations: usize, chunk_iterations: usize) -> CuckooFilter {
    let fixture = filter_fixture(entries);
    let filter_bytes = fixture.base.bytes();
    println!(
        "filter: entries={} buckets={} bytes={} ({:.2} MiB), delta={} bytes",
        entries,
        fixture.base.num_buckets(),
        filter_bytes,
        filter_bytes as f64 / (1024.0 * 1024.0),
        fixture.delta.len(),
    );

    measure(
        || {
            black_box(fixture.base.clone());
        },
        "filter clone only",
        iterations,
    );

    let mut apply_only = fixture.base.clone();
    measure(
        || {
            black_box(
                apply_delta(&mut apply_only, fixture.base_epoch, &fixture.delta)
                    .expect("apply reusable delta"),
            );
        },
        "delta apply in place",
        iterations,
    );

    measure(
        || {
            let mut next = fixture.base.clone();
            black_box(
                apply_delta(&mut next, fixture.base_epoch, &fixture.delta)
                    .expect("apply delta to clone"),
            );
            black_box(next);
        },
        "global clone + delta",
        iterations,
    );

    measure(
        || {
            black_box(fixture.producer.current_snapshot());
        },
        "snapshot bucket copy",
        iterations,
    );

    let snapshot = fixture.producer.current_snapshot();
    let chunk_count = snapshot.chunks().count();
    println!("snapshot: {chunk_count} chunk(s), {filter_bytes} bucket bytes");
    measure(
        || {
            for chunk in snapshot.chunks() {
                black_box(chunk);
            }
        },
        "snapshot chunk encode",
        chunk_iterations,
    );
    fixture.base
}

/// Measure large-delta publication: the re-dirty phase runs un-timed so the
/// samples isolate `publish()` — dirty-list drain, serialization, and the
/// `last_shipped` re-baseline.
fn bench_delta_publish(entries: usize, iterations: usize) {
    let mut producer = SnapshotProducer::new(7, entries, DEFAULT_FILTER_SEED);
    for value in 0..entries as u64 {
        assert!(
            producer.insert(spread(value)),
            "provisioned filter overflowed"
        );
    }
    drop(producer.full_snapshot());
    // Churn ~20% of the buckets per round: a large delta that still stays
    // under the full-snapshot switch. The removal window trails the insert
    // window by exactly `entries`, so removals always target resident hashes.
    let churn = entries / 16;

    let mut samples = Vec::with_capacity(iterations);
    let mut allocated = 0u64;
    let mut allocation_count = 0u64;
    let mut peak_extra = 0u64;
    let mut delta_len = 0usize;
    for iteration in 0..iterations as u64 {
        let remove_base = iteration * churn as u64;
        let insert_base = entries as u64 + remove_base;
        for value in 0..churn as u64 {
            producer.remove(spread(remove_base + value));
            assert!(
                producer.insert(spread(insert_base + value)),
                "churn insert overflowed"
            );
        }

        let allocations_before = AllocationSnapshot::capture();
        let live_before = LIVE_BYTES.load(Ordering::Relaxed);
        PEAK_LIVE_BYTES.store(live_before, Ordering::Relaxed);
        let started = Instant::now();
        let delta = match producer.publish() {
            Publish::Delta(delta) => delta,
            Publish::Full(_) => panic!("churn fixture crossed the full-snapshot switch"),
            Publish::Unchanged => panic!("churn fixture produced no delta"),
        };
        samples.push(started.elapsed().as_secs_f64() * 1e9);
        let allocations_after = AllocationSnapshot::capture();
        allocated += allocations_after.bytes - allocations_before.bytes;
        allocation_count += allocations_after.count - allocations_before.count;
        peak_extra = peak_extra.max(
            PEAK_LIVE_BYTES
                .load(Ordering::Relaxed)
                .saturating_sub(live_before),
        );
        delta_len = delta.len();
        black_box(delta);
    }

    let average = samples.iter().sum::<f64>() / iterations as f64;
    let p50 = percentile(&mut samples, 0.50);
    let p99 = percentile(&mut samples, 0.99);
    println!(
        "delta: frame={} bytes ({:.2} MiB), churn={churn} hashes/round",
        delta_len,
        delta_len as f64 / (1024.0 * 1024.0),
    );
    println!(
        "{:30} avg={average:12.0}ns p50={p50:12.0}ns p99={p99:12.0}ns alloc/op={:8.2} bytes/op={:14.0} peak+={peak_extra:10}",
        "delta build publish",
        allocation_count as f64 / iterations as f64,
        allocated as f64 / iterations as f64,
    );
}

/// Measure producer insert/remove churn at elevated load, where cuckoo kick
/// chains are frequent. This is the relay's hottest path: one insert per
/// stored block.
fn bench_insert_churn(entries: usize, iterations: usize) {
    let mut producer = SnapshotProducer::new(7, entries, DEFAULT_FILTER_SEED);
    // Overfill past the provisioning target so the load factor reaches the
    // upper end of the steady-state range and kick chains actually happen.
    let fill = entries * 3 / 2;
    for value in 0..fill as u64 {
        assert!(
            producer.insert(spread(value)),
            "overfilled filter unexpectedly hit Full"
        );
    }
    // Mirror `CuckooFilter::provisioned` geometry so the printed load factor
    // stays honest: ceil(entries / (4 * 0.8)) rounded up to a power of two.
    let buckets = ((entries as f64 / (4.0 * 0.8)).ceil() as usize).next_power_of_two();
    let load = producer.len() as f64 / (buckets * 4) as f64;
    println!(
        "churn: resident={} load={load:.2} batch={CHURN_BATCH} inserts+removes/op",
        producer.len(),
    );

    let mut round = 0u64;
    measure(
        || {
            let remove_base = round * CHURN_BATCH as u64;
            let insert_base = fill as u64 + remove_base;
            for value in 0..CHURN_BATCH as u64 {
                producer.remove(spread(remove_base + value));
                assert!(
                    producer.insert(spread(insert_base + value)),
                    "churn insert overflowed"
                );
            }
            round += 1;
        },
        "insert+remove churn batch",
        iterations,
    );
}

const CHURN_BATCH: usize = 10_000;

fn bench_search_speed(
    base: &CuckooFilter,
    entries: usize,
    dcs: usize,
    depth: usize,
    iterations: usize,
) {
    assert!(
        depth <= entries,
        "search depth must fit in resident fixture"
    );
    let filters = vec![base.clone(); dcs];
    let resident: Vec<u64> = (0..depth as u64).map(spread).collect();
    let mut absent = Vec::with_capacity(depth);
    let mut offset = 0u64;
    while absent.len() < depth {
        let hash = spread(entries as u64 + 10_000 + offset);
        if !base.contains(hash) {
            absent.push(hash);
        }
        offset += 1;
    }
    let partial_depth = depth / 2;
    let mut partial = resident[..partial_depth].to_vec();
    partial.extend_from_slice(&absent[partial_depth..]);
    let probes = probes_for(&resident, base.seed());
    let partial_probes = probes_for(&partial, base.seed());

    assert_eq!(
        overlap_depth_searched_seq(base, &partial),
        partial_depth as u32,
        "partial-prefix fixture must stop at its first absent hash"
    );

    measure(
        || {
            black_box(overlap_depth_searched_seq(
                black_box(base),
                black_box(&resident),
            ));
        },
        "search seq 1dc full hit",
        iterations,
    );

    measure(
        || {
            let mut total = 0u32;
            for filter in &filters {
                total += overlap_depth_searched_seq(filter, black_box(&resident));
            }
            black_box(total);
        },
        "search seq multi-dc hit",
        iterations,
    );

    measure(
        || {
            let mut total = 0u32;
            for filter in &filters {
                total += overlap_depth_searched_seq(filter, black_box(&absent));
            }
            black_box(total);
        },
        "search seq multi-dc miss",
        iterations,
    );

    measure(
        || {
            let mut total = 0u32;
            for filter in &filters {
                total += overlap_depth_searched_seq(filter, black_box(&partial));
            }
            black_box(total);
        },
        "search seq multi-dc partial",
        iterations,
    );

    measure(
        || {
            let mut best = 0u32;
            for filter in &filters {
                best = best.max(overlap_depth_searched(filter, black_box(&probes)));
            }
            black_box(best);
        },
        "search probes multi-dc hit",
        iterations,
    );

    measure(
        || {
            let mut best = 0u32;
            for filter in &filters {
                best = best.max(overlap_depth_searched(filter, black_box(&partial_probes)));
            }
            black_box(best);
        },
        "search probes multi-dc partial",
        iterations,
    );

    println!(
        "search: dcs={dcs} depth={depth} partial_depth={partial_depth} iterations={iterations}"
    );
}

fn main() {
    let entries = env_usize("PROTO_BENCH_FILTER_ENTRIES", 1_000_000);
    let filter_iterations = env_usize("PROTO_BENCH_FILTER_ITERS", 50);
    let chunk_iterations = env_usize("PROTO_BENCH_CHUNK_ITERS", 20);
    let delta_iterations = env_usize("PROTO_BENCH_DELTA_ITERS", 20);
    let search_dcs = env_usize("PROTO_BENCH_SEARCH_DCS", 8);
    let search_depth = env_usize("PROTO_BENCH_SEARCH_DEPTH", 128);
    let search_iterations = env_usize("PROTO_BENCH_SEARCH_ITERS", 20_000);

    println!("kv-event-relay-proto transport cost baselines");
    let search_base = bench_filter_costs(entries, filter_iterations, chunk_iterations);
    bench_delta_publish(entries, delta_iterations);
    bench_insert_churn(entries, env_usize("PROTO_BENCH_CHURN_ITERS", 50));
    bench_search_speed(
        &search_base,
        entries,
        search_dcs,
        search_depth.min(entries),
        search_iterations,
    );
}
