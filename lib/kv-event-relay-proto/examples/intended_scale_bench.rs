// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intended-scale index-kernel benchmark, mirroring the independent methodology
//! from the DEP #11225 review: D data centers at ~10M resident memberships
//! each, deterministic 128-block query families (one full-hit DC, deterministic
//! partial coverage everywhere else), 20k shuffled query families, >=1M queries
//! per trial, 15 trials per mode, single-core pinned runs.
//!
//! Backends:
//! - `radix_*` — the upstream kv-router [`RadixTree`] (one worker id per DC),
//!   the exact baseline the independent review measured; one `find_matches`
//!   walk returns every DC's overlap.
//! - `exact_*` — deduplicated open-addressing `seq_hash -> dc_mask` table; one
//!   monotone walk answers every DC at once (an upper bound for any exact
//!   index, stronger than a per-DC radix because it dedups across DCs).
//! - `cuckoo_full` — per-DC exponential+binary boundary search over the shared
//!   probe set (`probes_for` + `overlap_depth_searched`).
//! - `cuckoo_full_seq` — the production `find_all_overlaps_by_hashes` kernel:
//!   per-DC `overlap_depth_searched_seq`, re-deriving probes per DC.
//! - `cuckoo_best` — `argmax_overlap_dc`, the O(D + log B) tournament lookup.
//!
//! Environment knobs (defaults in parentheses):
//! - `ISB_DCS` (4) — data-center count
//! - `ISB_TARGET` (10_000_000) — resident memberships per DC
//! - `ISB_FAMQ` (20_000) — query families sampled from the resident set
//! - `ISB_QUERIES` (1_000_000) — queries per trial
//! - `ISB_TRIALS` (15) — trials per mode
//! - `ISB_WARMUP` (50_000) — unmeasured warmup queries per mode
//! - `ISB_SEED` (1) — run seed (vary per process repetition)
//! - `ISB_MODES` (all) — comma list to restrict modes; `all` covers the
//!   exact/cuckoo modes, while `radix_full`/`radix_best` must be requested
//!   explicitly (their tree is a large separate build, run one backend per
//!   process like the independent review did)
//!
//! Run pinned, e.g.:
//! `numactl --physcpubind=60 --membind=1 cargo run --release -p
//!  dynamo-kv-event-relay-proto --example intended_scale_bench`

use std::hint::black_box;
use std::time::Instant;

use dynamo_kv_event_relay_proto::cuckoo::{
    COHORT_LANES, CuckooFilter, DEFAULT_FILTER_SEED, MultiOverlapScratch, TransposedCohort,
    argmax_overlap_dc, overlap_depth_searched, overlap_depth_searched_seq, overlap_depths_multi,
    probes_for,
};
use dynamo_kv_router::indexer::RadixTree;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
};

const FAMILY_BLOCKS: usize = 128;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic prefix-chained sequence hashes for family `f`. Zero is
/// remapped so the exact table can use it as its empty sentinel.
fn family_chain(seed: u64, f: u64, out: &mut [u64; FAMILY_BLOCKS]) {
    let mut state = seed ^ f.wrapping_mul(0xA24B_AED4_963E_E407);
    for slot in out.iter_mut() {
        let mut h = splitmix64(&mut state);
        if h == 0 {
            h = 0x9E37_79B9_7F4A_7C15;
        }
        *slot = h;
    }
}

/// Stored event feeding the radix baseline: the family chain doubles as both
/// the token-hash walk key and the unique block id, mirroring how the PR#4
/// harness derives both from one `hash_ids` chain.
fn stored_event(worker: u64, event_id: u64, chain: &[u64]) -> RouterEvent {
    RouterEvent::new(
        worker,
        KvCacheEvent {
            event_id,
            dp_rank: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: chain
                    .iter()
                    .map(|&h| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(h),
                        tokens_hash: LocalBlockHash(h),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
        },
    )
}

/// Deterministic partial coverage depth of family `f` in non-home DC `d`,
/// uniform in [0, 127] so the home DC (128) is the unique deepest.
#[inline]
fn partial_coverage(seed: u64, f: u64, d: u64) -> usize {
    (mix64(seed ^ f.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ d.wrapping_mul(0xCA5A_8263_9512_1157))
        % FAMILY_BLOCKS as u64) as usize
}

// ---------------------------------------------------------------------------
// Exact baseline: open-addressing seq_hash -> dc_mask table.
// ---------------------------------------------------------------------------

struct ExactTable {
    keys: Vec<u64>,
    masks: Vec<u64>,
    slot_mask: usize,
    len: usize,
}

impl ExactTable {
    fn with_capacity(expected: usize) -> Self {
        let slots = ((expected as f64 / 0.8).ceil() as usize)
            .next_power_of_two()
            .max(16);
        Self {
            keys: vec![0; slots],
            masks: vec![0; slots],
            slot_mask: slots - 1,
            len: 0,
        }
    }

    fn insert(&mut self, key: u64, dc: usize) {
        let mut i = (mix64(key) as usize) & self.slot_mask;
        loop {
            let k = self.keys[i];
            if k == key {
                self.masks[i] |= 1u64 << dc;
                return;
            }
            if k == 0 {
                self.keys[i] = key;
                self.masks[i] = 1u64 << dc;
                self.len += 1;
                return;
            }
            i = (i + 1) & self.slot_mask;
        }
    }

    #[inline]
    fn get(&self, key: u64) -> u64 {
        let mut i = (mix64(key) as usize) & self.slot_mask;
        loop {
            let k = self.keys[i];
            if k == key {
                return self.masks[i];
            }
            if k == 0 {
                return 0;
            }
            i = (i + 1) & self.slot_mask;
        }
    }

    fn bytes(&self) -> usize {
        self.keys.len() * 16
    }
}

/// One monotone walk fills every DC's contiguous overlap depth.
#[inline]
fn exact_full_map(table: &ExactTable, seq: &[u64], dcs: usize, depths: &mut [u32]) {
    let all = if dcs == 64 {
        u64::MAX
    } else {
        (1u64 << dcs) - 1
    };
    depths[..dcs].fill(0);
    let mut live = all;
    for (i, &h) in seq.iter().enumerate() {
        let m = table.get(h);
        let mut died = live & !m;
        while died != 0 {
            let b = died.trailing_zeros() as usize;
            depths[b] = i as u32;
            died &= died - 1;
        }
        live &= m;
        if live == 0 {
            return;
        }
    }
    let mut survivors = live;
    while survivors != 0 {
        let b = survivors.trailing_zeros() as usize;
        depths[b] = seq.len() as u32;
        survivors &= survivors - 1;
    }
}

/// Same walk, returning only the deepest DC (ties -> lowest index, matching
/// `argmax_overlap_dc`).
#[inline]
fn exact_best_dc(table: &ExactTable, seq: &[u64], dcs: usize) -> (usize, u32) {
    let all = if dcs == 64 {
        u64::MAX
    } else {
        (1u64 << dcs) - 1
    };
    let mut live = all;
    for (i, &h) in seq.iter().enumerate() {
        let m = table.get(h);
        let next = live & m;
        if next == 0 {
            return (live.trailing_zeros() as usize, i as u32);
        }
        live = next;
    }
    (live.trailing_zeros() as usize, seq.len() as u32)
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

fn percentile(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

/// Two-sided 95% Student-t critical values for n-1 degrees of freedom.
fn t_crit(n: usize) -> f64 {
    const TABLE: [f64; 15] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131,
    ];
    if n <= 1 {
        return f64::NAN;
    }
    TABLE[(n - 2).min(TABLE.len() - 1)]
}

fn vm_rss_gib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|kb| kb.parse::<f64>().ok())
            })
        })
        .map(|kb| kb / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

struct ModeStats {
    trial_kops: Vec<f64>,
    latencies_ns: Vec<u32>,
}

impl ModeStats {
    fn new() -> Self {
        Self {
            trial_kops: Vec::new(),
            latencies_ns: Vec::new(),
        }
    }
}

fn main() {
    let dcs = env_usize("ISB_DCS", 4);
    assert!((1..=64).contains(&dcs), "ISB_DCS must be in 1..=64");
    let target = env_usize("ISB_TARGET", 10_000_000);
    let fam_q = env_usize("ISB_FAMQ", 20_000);
    let queries = env_usize("ISB_QUERIES", 1_000_000);
    let trials = env_usize("ISB_TRIALS", 15);
    let warmup = env_usize("ISB_WARMUP", 50_000);
    let seed = env_u64("ISB_SEED", 1);
    let mode_filter = std::env::var("ISB_MODES").unwrap_or_else(|_| "all".to_string());

    // Family count so that expected per-DC memberships ~= target:
    // per-DC = F * (128 + p_avg * (D-1)) / D with p_avg = 63.5.
    let families =
        ((target as f64 * dcs as f64) / (FAMILY_BLOCKS as f64 + 63.5 * (dcs as f64 - 1.0))).ceil()
            as u64;
    assert!(
        fam_q as u64 <= families,
        "ISB_FAMQ exceeds the resident family count"
    );

    println!(
        "CONFIG dcs={dcs} target_per_dc={target} families={families} query_families={fam_q} \
         queries_per_trial={queries} trials={trials} seed={seed} modes={mode_filter}"
    );

    // -- Modes -----------------------------------------------------------------
    // `all` deliberately excludes the radix modes: their tree is a large
    // separate build, so radix runs go in their own process (one backend per
    // process, as in the independent review).
    let default_modes = [
        "exact_full",
        "exact_best",
        "cuckoo_full",
        "cuckoo_full_seq",
        "cuckoo_best",
    ];
    let all_modes = [
        "exact_full",
        "exact_best",
        "cuckoo_full",
        "cuckoo_full_seq",
        "cuckoo_best",
        "cuckoo_full_tr",
        "cuckoo_best_tr",
        "radix_full",
        "radix_best",
    ];
    let modes: Vec<&str> = if mode_filter == "all" {
        default_modes.to_vec()
    } else {
        all_modes
            .iter()
            .copied()
            .filter(|m| mode_filter.split(',').any(|w| w == *m))
            .collect()
    };
    assert!(!modes.is_empty(), "ISB_MODES matched no known mode");
    let needs_cuckoo = modes.iter().any(|m| m.starts_with("cuckoo"));
    let needs_exact = modes.iter().any(|m| m.starts_with("exact"));
    let needs_radix = modes.iter().any(|m| m.starts_with("radix"));
    let needs_transposed = modes.iter().any(|m| m.ends_with("_tr"));

    // -- Build ---------------------------------------------------------------
    let mut filters: Vec<CuckooFilter> = if needs_cuckoo {
        (0..dcs)
            .map(|_| CuckooFilter::provisioned(target, DEFAULT_FILTER_SEED))
            .collect()
    } else {
        Vec::new()
    };
    let mut exact = needs_exact.then(|| ExactTable::with_capacity(families as usize * FAMILY_BLOCKS));
    let mut radix = needs_radix.then(RadixTree::new);

    let mut chain = [0u64; FAMILY_BLOCKS];
    let mut memberships_per_dc = vec![0u64; dcs];
    let mut cuckoo_insert_failures = 0u64;
    let mut event_id = 0u64;

    let mut cuckoo_ns = 0u128;
    let mut exact_ns = 0u128;
    let mut radix_ns = 0u128;
    for f in 0..families {
        family_chain(seed, f, &mut chain);
        let home = (f % dcs as u64) as usize;
        for d in 0..dcs {
            let cov = if d == home {
                FAMILY_BLOCKS
            } else {
                partial_coverage(seed, f, d as u64)
            };
            memberships_per_dc[d] += cov as u64;
            if needs_cuckoo {
                let t = Instant::now();
                for &h in &chain[..cov] {
                    if !filters[d].insert(h) {
                        cuckoo_insert_failures += 1;
                    }
                }
                cuckoo_ns += t.elapsed().as_nanos();
            }
            if let Some(exact) = exact.as_mut() {
                let t = Instant::now();
                for &h in &chain[..cov] {
                    exact.insert(h, d);
                }
                exact_ns += t.elapsed().as_nanos();
            }
            if let Some(radix) = radix.as_mut()
                && cov > 0
            {
                // Event assembly stays outside the timer; only the structure
                // insert is the build cost being compared.
                let event = stored_event(d as u64, event_id, &chain[..cov]);
                event_id += 1;
                let t = Instant::now();
                radix.apply_event(event).expect("radix apply_event");
                radix_ns += t.elapsed().as_nanos();
            }
        }
    }

    let total_memberships: u64 = memberships_per_dc.iter().sum();
    let cuckoo_bytes: usize = filters.iter().map(|f| f.bytes()).sum();
    let mps = |ns: u128| {
        if ns == 0 {
            0.0
        } else {
            total_memberships as f64 / (ns as f64 / 1e3)
        }
    };
    println!(
        "BUILD memberships_total={total_memberships} per_dc_min={} per_dc_max={} \
         cuckoo_build_mps={:.2} exact_build_mps={:.2} radix_build_mps={:.2} \
         insert_failures={cuckoo_insert_failures}",
        memberships_per_dc.iter().min().unwrap(),
        memberships_per_dc.iter().max().unwrap(),
        mps(cuckoo_ns),
        mps(exact_ns),
        mps(radix_ns),
    );
    println!(
        "MEM cuckoo_payload_gib={:.3} exact_table_gib={:.3} exact_distinct_keys={} \
         radix_blocks={} rss_gib={:.3}",
        cuckoo_bytes as f64 / (1u64 << 30) as f64,
        exact.as_ref().map_or(0.0, |e| e.bytes() as f64 / (1u64 << 30) as f64),
        exact.as_ref().map_or(0, |e| e.len),
        radix.as_ref().map_or(0, |r| r.current_size()),
        vm_rss_gib(),
    );

    // -- Transposed read cache (derived from the authoritative filters) -------
    let cohorts: Vec<TransposedCohort> = if needs_transposed {
        let t = Instant::now();
        let cohorts: Vec<TransposedCohort> = filters
            .chunks(COHORT_LANES)
            .map(|chunk| {
                let refs: Vec<&CuckooFilter> = chunk.iter().collect();
                TransposedCohort::from_filters(&refs).expect("same-shape cohort")
            })
            .collect();
        println!(
            "TRANSPOSED cohorts={} bytes_gib={:.3} build_secs={:.3}",
            cohorts.len(),
            cohorts.iter().map(|c| c.bytes()).sum::<usize>() as f64 / (1u64 << 30) as f64,
            t.elapsed().as_secs_f64(),
        );
        cohorts
    } else {
        Vec::new()
    };

    // -- Query families -------------------------------------------------------
    // Fisher-Yates over the resident families, take the first `fam_q`.
    let mut order: Vec<u64> = (0..families).collect();
    let mut shuffle_state = seed ^ 0x5157_5CA1_B1E5_D00D;
    for i in (1..order.len()).rev() {
        let j = (splitmix64(&mut shuffle_state) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let query_fams: Vec<u64> = order[..fam_q].to_vec();
    drop(order);

    let mut qchains: Vec<u64> = Vec::with_capacity(fam_q * FAMILY_BLOCKS);
    for &f in &query_fams {
        family_chain(seed, f, &mut chain);
        qchains.extend_from_slice(&chain);
    }
    let qlocal: Vec<LocalBlockHash> = if needs_radix {
        qchains.iter().map(|&h| LocalBlockHash(h)).collect()
    } else {
        Vec::new()
    };

    // -- Accuracy (every query family, exact as oracle) -----------------------
    if let Some(exact) = exact.as_ref().filter(|_| needs_cuckoo) {
        let mut exact_depths = vec![0u32; dcs];
        let mut tr_depths = vec![0u32; dcs];
        let mut tr_scratch = MultiOverlapScratch::default();
        let mut inflated = 0u64;
        let mut underestimated = 0u64;
        let mut wrong_best = 0u64;
        let mut tr_mismatch = 0u64;
        for qi in 0..fam_q {
            let seq = &qchains[qi * FAMILY_BLOCKS..(qi + 1) * FAMILY_BLOCKS];
            exact_full_map(exact, seq, dcs, &mut exact_depths);
            let probes = probes_for(seq, DEFAULT_FILTER_SEED);
            for (d, filter) in filters.iter().enumerate() {
                let depth = overlap_depth_searched(filter, &probes);
                if depth > exact_depths[d] {
                    inflated += 1;
                } else if depth < exact_depths[d] {
                    underestimated += 1;
                }
            }
            let (ebest, _) = exact_best_dc(exact, seq, dcs);
            let (cbest, _) = argmax_overlap_dc(&filters, &probes);
            if ebest != cbest {
                wrong_best += 1;
            }
            // The transposed walk must agree with the per-filter search.
            if needs_transposed {
                overlap_depths_multi(&cohorts, &probes, &mut tr_depths, &mut tr_scratch);
                for (d, filter) in filters.iter().enumerate() {
                    if tr_depths[d] != overlap_depth_searched(filter, &probes) {
                        tr_mismatch += 1;
                    }
                }
            }
        }
        println!(
            "ACCURACY checked={} inflated={inflated} underestimated={underestimated} \
             wrong_best_dc={wrong_best}/{fam_q} transposed_mismatch={tr_mismatch}",
            fam_q * dcs,
        );
    }

    // -- Timed modes -----------------------------------------------------------
    let mut stats: Vec<ModeStats> = modes.iter().map(|_| ModeStats::new()).collect();
    let mut depths_out = vec![0u32; dcs];
    let mut tr_scratch = MultiOverlapScratch::default();

    let mut run_queries = |mode: &str,
                           count: usize,
                           trial_seed: u64,
                           latencies: Option<&mut Vec<u32>>|
     -> f64 {
        // Per-trial shuffled family order, cycled to reach `count` queries.
        let mut idx: Vec<u32> = (0..fam_q as u32).collect();
        let mut st = trial_seed;
        for i in (1..idx.len()).rev() {
            let j = (splitmix64(&mut st) % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        let mut lat = latencies;
        let is_radix = mode.starts_with("radix");
        let start = Instant::now();
        let mut sink = 0u64;
        for qi in 0..count {
            let fam = idx[qi % fam_q] as usize;
            let seq = &qchains[fam * FAMILY_BLOCKS..(fam + 1) * FAMILY_BLOCKS];
            // `find_matches` consumes its query, so the clone happens outside
            // the per-query timer (the PR#4 harness measures the same way).
            let radix_query = is_radix
                .then(|| qlocal[fam * FAMILY_BLOCKS..(fam + 1) * FAMILY_BLOCKS].to_vec());
            let t = Instant::now();
            match mode {
                "exact_full" => {
                    exact_full_map(exact.as_ref().unwrap(), seq, dcs, &mut depths_out);
                    sink ^= depths_out[0] as u64;
                }
                "exact_best" => {
                    let (dc, depth) = exact_best_dc(exact.as_ref().unwrap(), seq, dcs);
                    sink ^= (dc as u64) << 32 | depth as u64;
                }
                "radix_full" => {
                    let scores = radix
                        .as_ref()
                        .unwrap()
                        .find_matches(radix_query.unwrap(), false);
                    depths_out[..dcs].fill(0);
                    for (worker, &depth) in &scores.scores {
                        let dc = worker.worker_id as usize;
                        if depths_out[dc] < depth {
                            depths_out[dc] = depth;
                        }
                    }
                    sink ^= depths_out[0] as u64;
                }
                "radix_best" => {
                    let scores = radix
                        .as_ref()
                        .unwrap()
                        .find_matches(radix_query.unwrap(), false);
                    let mut best_dc = 0usize;
                    let mut best = 0u32;
                    for (worker, &depth) in &scores.scores {
                        let dc = worker.worker_id as usize;
                        // Tie -> lowest index, independent of map iteration order.
                        if depth > best || (depth == best && depth > 0 && dc < best_dc) {
                            best = depth;
                            best_dc = dc;
                        }
                    }
                    sink ^= (best_dc as u64) << 32 | best as u64;
                }
                "cuckoo_full" => {
                    let probes = probes_for(seq, DEFAULT_FILTER_SEED);
                    for (d, filter) in filters.iter().enumerate() {
                        depths_out[d] = overlap_depth_searched(filter, &probes);
                    }
                    sink ^= depths_out[0] as u64;
                }
                "cuckoo_full_seq" => {
                    for (d, filter) in filters.iter().enumerate() {
                        depths_out[d] = overlap_depth_searched_seq(filter, seq);
                    }
                    sink ^= depths_out[0] as u64;
                }
                "cuckoo_best" => {
                    let probes = probes_for(seq, DEFAULT_FILTER_SEED);
                    let (dc, depth) = argmax_overlap_dc(&filters, &probes);
                    sink ^= (dc as u64) << 32 | depth as u64;
                }
                "cuckoo_full_tr" => {
                    let probes = probes_for(seq, DEFAULT_FILTER_SEED);
                    overlap_depths_multi(&cohorts, &probes, &mut depths_out, &mut tr_scratch);
                    sink ^= depths_out[0] as u64;
                }
                "cuckoo_best_tr" => {
                    let probes = probes_for(seq, DEFAULT_FILTER_SEED);
                    overlap_depths_multi(&cohorts, &probes, &mut depths_out, &mut tr_scratch);
                    let mut best_dc = 0usize;
                    let mut best = 0u32;
                    for (d, &v) in depths_out[..dcs].iter().enumerate() {
                        if v > best {
                            best = v;
                            best_dc = d;
                        }
                    }
                    sink ^= (best_dc as u64) << 32 | best as u64;
                }
                other => unreachable!("unknown mode {other}"),
            }
            if let Some(lat) = lat.as_deref_mut() {
                lat.push(t.elapsed().as_nanos().min(u32::MAX as u128) as u32);
            }
        }
        black_box(sink);
        start.elapsed().as_secs_f64()
    };

    // Warmup each mode once, unmeasured.
    for mode in &modes {
        run_queries(mode, warmup, seed ^ 0xDEAD_BEEF, None);
    }

    // Alternate mode order across trials to spread drift evenly.
    for trial in 0..trials {
        for k in 0..modes.len() {
            let mi = (k + trial) % modes.len();
            let mode = modes[mi];
            let before = stats[mi].latencies_ns.len();
            let mut lat = std::mem::take(&mut stats[mi].latencies_ns);
            let secs = run_queries(mode, queries, seed.wrapping_add(trial as u64 * 7919), Some(&mut lat));
            stats[mi].latencies_ns = lat;
            let kops = queries as f64 / secs / 1e3;
            stats[mi].trial_kops.push(kops);
            let new = &mut stats[mi].latencies_ns[before..];
            new.sort_unstable();
            println!(
                "RESULT mode={mode} trial={trial} queries={queries} secs={secs:.3} kops={kops:.1} \
                 p50_ns={} p99_ns={}",
                percentile(new, 0.50),
                percentile(new, 0.99),
            );
        }
    }

    for (mi, mode) in modes.iter().enumerate() {
        let s = &mut stats[mi];
        let n = s.trial_kops.len();
        if n == 0 {
            continue;
        }
        let mean = s.trial_kops.iter().sum::<f64>() / n as f64;
        let ci = if n > 1 {
            let var =
                s.trial_kops.iter().map(|k| (k - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            t_crit(n) * (var / n as f64).sqrt()
        } else {
            0.0
        };
        s.latencies_ns.sort_unstable();
        println!(
            "SUMMARY mode={mode} trials={n} kops_mean={mean:.1} kops_ci95={ci:.1} \
             p50_ns={} p99_ns={} samples={}",
            percentile(&s.latencies_ns, 0.50),
            percentile(&s.latencies_ns, 0.99),
            s.latencies_ns.len(),
        );
    }
    println!("DONE rss_gib={:.3}", vm_rss_gib());
}
