// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Apples-to-apples comparison of two per-DC block-availability indexes for
//! cross-DC KV-aware routing:
//!
//!   * `radix`  — the upstream kv-router [`RadixTree`] (exact index; one tree,
//!     one worker id per DC), queried through `find_matches`;
//!   * `cuckoo` — this crate's seeded cuckoo filter (one filter per DC), built
//!     on the relay side by [`SnapshotProducer`], shipped through the real
//!     CKF1 wire encoding (full-snapshot chunks, then a delta), and queried
//!     with [`overlap_depth_searched`] / [`argmax_overlap_dc`].
//!
//! Both backends ingest identical Stored events derived from a mooncake-style
//! trace (`hash_ids` per request are prefix-aware block ids: shared prompt
//! prefixes share leading ids). Every request is then re-queried against
//! both, and an exact per-DC `HashSet` oracle provides ground truth for
//! routing quality, so the filter's false positives are measured rather than
//! assumed away.
//!
//! Quality + latency (default mode):
//!
//! ```text
//! cargo run -p dynamo-kv-event-relay-proto --release --example radix_vs_cuckoo -- \
//!     lib/bench/testdata/mooncake_trace_1000.jsonl 4
//! ```
//!
//! Memory — run each backend in its OWN process so RSS attributes cleanly:
//!
//! ```text
//! ... --example radix_vs_cuckoo -- <trace> 4 mem-radix
//! ... --example radix_vs_cuckoo -- <trace> 4 mem-cuckoo
//! ```
//!
//! The trace path accepts any JSONL with a `hash_ids: [u64]` field per line
//! (and an optional `dc` field pinning a request's home DC), so larger traces
//! from the mooncake generators drop in unchanged.

use std::collections::HashSet;
use std::time::Instant;

use dynamo_kv_event_relay_proto::cuckoo::{
    CuckooFilter, DEFAULT_FILTER_SEED, Publish, SnapshotProducer, apply_delta, argmax_overlap_dc,
    assemble_chunks, overlap_depth_searched, probes_for,
};
use dynamo_kv_router::indexer::RadixTree;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, compute_seq_hash_for_block,
};

/// One replayed request: the per-block local-hash chain (the radix query
/// input) and its chained sequence-hash chain (what the relay stores and the
/// cuckoo side probes). Both derive from the same `hash_ids` through the same
/// kv-router recurrence, so prefix identity is consistent across backends.
struct Request {
    local: Vec<LocalBlockHash>,
    seq: Vec<u64>,
    dc: Option<usize>,
}

#[derive(serde::Deserialize)]
struct Row {
    #[serde(default)]
    hash_ids: Option<Vec<u64>>,
    #[serde(default)]
    dc: Option<u32>,
}

fn load_requests(path: &str) -> Vec<Request> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading trace {path}: {e}"));
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(line).expect("trace line parses as JSON");
        let Some(ids) = row.hash_ids else { continue };
        if ids.is_empty() {
            continue;
        }
        let local: Vec<LocalBlockHash> = ids.iter().map(|&h| LocalBlockHash(h)).collect();
        let seq = compute_seq_hash_for_block(&local);
        out.push(Request {
            local,
            seq,
            dc: row.dc.map(|d| d as usize),
        });
    }
    out
}

/// Home DC per request: the trace's explicit `dc` when present (synthetic
/// multi-DC traces), else a deterministic scatter by the first block so the
/// same prompt family always lands on the same DC — which is what produces
/// cross-request prefix reuse inside a DC.
fn plan_dcs(reqs: &[Request], num_dcs: usize) -> Vec<usize> {
    reqs.iter()
        .map(|r| match r.dc {
            Some(dc) => dc % num_dcs,
            None => (r.seq[0] as usize) % num_dcs,
        })
        .collect()
}

fn stored_event(worker: u64, event_id: u64, req: &Request) -> RouterEvent {
    RouterEvent::new(
        worker,
        KvCacheEvent {
            event_id,
            dp_rank: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: req
                    .local
                    .iter()
                    .zip(&req.seq)
                    .map(|(&local, &seq)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(seq),
                        tokens_hash: local,
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
        },
    )
}

/// Longest leading run of `seq` present in `set` — the exact-oracle overlap.
fn prefix_overlap(seq: &[u64], set: &HashSet<u64>) -> u32 {
    let mut depth = 0;
    for h in seq {
        if set.contains(h) {
            depth += 1;
        } else {
            break;
        }
    }
    depth
}

/// argmax over DC sets by exact prefix overlap; tie → lowest dc.
fn argmax_exact(seq: &[u64], sets: &[HashSet<u64>]) -> (usize, u32) {
    let mut best_dc = 0;
    let mut best = 0u32;
    for (dc, set) in sets.iter().enumerate() {
        let d = prefix_overlap(seq, set);
        if d > best {
            best = d;
            best_dc = dc;
        }
    }
    (best_dc, best)
}

/// argmax over a per-DC depth slice; tie → lowest dc.
fn argmax_depths(depths: &[u32]) -> (usize, u32) {
    let mut best_dc = 0usize;
    let mut best = 0u32;
    for (dc, &d) in depths.iter().enumerate() {
        if d > best {
            best = d;
            best_dc = dc;
        }
    }
    (best_dc, best)
}

/// Cold-start spread: when no DC holds the prefix, scatter deterministically
/// so fresh prompts don't all pile onto dc0 — applied identically to every
/// backend so choice comparison stays meaningful at depth 0.
fn choose_with_fallback(dc: usize, depth: u32, seq: &[u64], num_dcs: usize) -> usize {
    if depth > 0 {
        dc
    } else {
        (seq[0] as usize) % num_dcs
    }
}

fn percentile(sorted_ns: &[u64], p: f64) -> u64 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((sorted_ns.len() as f64 - 1.0) * p).round() as usize;
    sorted_ns[idx]
}

fn proc_status_kib(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field).and_then(|r| r.strip_prefix(':'))
            && let Some(kib) = rest.split_whitespace().next()
        {
            return kib.parse().unwrap_or(0);
        }
    }
    0
}

/// Exact per-DC resident sets (the oracle, and the dedup guard for ingest:
/// the real relay dedups upstream events the same way).
fn build_oracle_sets(reqs: &[Request], plan: &[usize], num_dcs: usize) -> Vec<HashSet<u64>> {
    let mut sets = vec![HashSet::new(); num_dcs];
    for (req, &dc) in reqs.iter().zip(plan) {
        for &h in &req.seq {
            sets[dc].insert(h);
        }
    }
    sets
}

fn build_radix(reqs: &[Request], plan: &[usize]) -> RadixTree {
    let mut tree = RadixTree::new();
    for (i, (req, &dc)) in reqs.iter().zip(plan).enumerate() {
        tree.apply_event(stored_event(dc as u64, i as u64, req))
            .expect("radix apply_event");
    }
    tree
}

/// Relay-side producers → CKF1 wire → router-side filters. The first 90% of
/// each DC's blocks travel as a chunked full snapshot, the rest as a delta,
/// so the round trip exercises both frame kinds a real subscriber sees.
fn build_cuckoo_via_wire(sets: &[HashSet<u64>]) -> (Vec<CuckooFilter>, usize) {
    let mut filters = Vec::with_capacity(sets.len());
    let mut wire_bytes_total = 0usize;
    for (dc, set) in sets.iter().enumerate() {
        let resident: Vec<u64> = set.iter().copied().collect();
        let split = resident.len() * 9 / 10;
        let mut producer =
            SnapshotProducer::new(dc as u64, resident.len().max(1), DEFAULT_FILTER_SEED);
        for &h in &resident[..split] {
            assert!(producer.insert(h), "provisioned filter must not overflow");
        }
        let snap = producer.full_snapshot();
        let chunks: Vec<Vec<u8>> = snap.chunks().collect();
        wire_bytes_total += chunks.iter().map(Vec::len).sum::<usize>();
        let (mut filter, meta) = assemble_chunks(chunks).expect("snapshot assembles");
        let mut epoch = meta.filter_epoch;

        for &h in &resident[split..] {
            assert!(producer.insert(h), "provisioned filter must not overflow");
        }
        match producer.publish() {
            Publish::Delta(bytes) => {
                wire_bytes_total += bytes.len();
                let info = apply_delta(&mut filter, epoch, &bytes).expect("delta applies");
                epoch = info.new_epoch;
            }
            Publish::Full(snap) => {
                let chunks: Vec<Vec<u8>> = snap.chunks().collect();
                wire_bytes_total += chunks.iter().map(Vec::len).sum::<usize>();
                let (f, meta) = assemble_chunks(chunks).expect("snapshot assembles");
                filter = f;
                epoch = meta.filter_epoch;
            }
            Publish::Unchanged => {}
        }
        let _ = epoch;
        filters.push(filter);
    }
    (filters, wire_bytes_total)
}

#[allow(clippy::too_many_lines)]
fn run_quality(reqs: &[Request], plan: &[usize], num_dcs: usize) {
    let sets = build_oracle_sets(reqs, plan, num_dcs);
    let resident: usize = sets.iter().map(HashSet::len).sum();
    let tree = build_radix(reqs, plan);
    let (filters, wire_bytes) = build_cuckoo_via_wire(&sets);
    let filter_bytes: usize = filters.iter().map(CuckooFilter::bytes).sum();

    let mut radix_ns = Vec::with_capacity(reqs.len());
    let mut ck_full_ns = Vec::with_capacity(reqs.len());
    let mut ck_best_ns = Vec::with_capacity(reqs.len());
    let (mut radix_match, mut ck_full_match, mut ck_best_match) = (0usize, 0usize, 0usize);
    let (mut inflated, mut under) = (0usize, 0usize);

    for req in reqs {
        let (odc, odepth) = argmax_exact(&req.seq, &sets);
        let oracle_choice = choose_with_fallback(odc, odepth, &req.seq, num_dcs);

        // radix: one walk returns every DC's overlap. The Vec clone is API
        // shape (find_matches consumes its query), kept outside the timer.
        let query = req.local.clone();
        let t = Instant::now();
        let scores = tree.find_matches(query, false);
        radix_ns.push(t.elapsed().as_nanos() as u64);
        let mut radix_depths = vec![0u32; num_dcs];
        for (worker, depth) in &scores.scores {
            let dc = worker.worker_id as usize;
            radix_depths[dc] = radix_depths[dc].max(*depth);
        }
        let (rdc, rdepth) = argmax_depths(&radix_depths);
        if choose_with_fallback(rdc, rdepth, &req.seq, num_dcs) == oracle_choice {
            radix_match += 1;
        }

        // cuckoo, full per-DC map (what weighted multi-signal scoring needs).
        // probes_for is the per-request hashing the router pays, so it is
        // inside the timed region.
        let t = Instant::now();
        let probes = probes_for(&req.seq, DEFAULT_FILTER_SEED);
        let depths: Vec<u32> = filters
            .iter()
            .map(|f| overlap_depth_searched(f, &probes))
            .collect();
        ck_full_ns.push(t.elapsed().as_nanos() as u64);
        let (cdc, cdepth) = argmax_depths(&depths);
        if choose_with_fallback(cdc, cdepth, &req.seq, num_dcs) == oracle_choice {
            ck_full_match += 1;
        }
        let exact_at_choice = prefix_overlap(&req.seq, &sets[cdc]);
        if cdepth > exact_at_choice {
            inflated += 1;
        }
        if cdepth < exact_at_choice {
            under += 1;
        }

        // cuckoo, best-DC-only tournament path.
        let t = Instant::now();
        let probes = probes_for(&req.seq, DEFAULT_FILTER_SEED);
        let (bdc, bdepth) = argmax_overlap_dc(&filters, &probes);
        ck_best_ns.push(t.elapsed().as_nanos() as u64);
        if choose_with_fallback(bdc, bdepth, &req.seq, num_dcs) == oracle_choice {
            ck_best_match += 1;
        }
    }

    radix_ns.sort_unstable();
    ck_full_ns.sort_unstable();
    ck_best_ns.sort_unstable();
    let pct = |n: usize| n as f64 / reqs.len() as f64 * 100.0;

    println!(
        "\ntrace: {} requests, {num_dcs} DCs, {resident} unique resident blocks",
        reqs.len()
    );
    println!(
        "cuckoo index: {:.2} MiB in-memory ({:.2} B/block), {:.2} MiB shipped over the CKF1 wire",
        filter_bytes as f64 / (1024.0 * 1024.0),
        filter_bytes as f64 / resident.max(1) as f64,
        wire_bytes as f64 / (1024.0 * 1024.0),
    );
    println!(
        "radix tree size: {} blocks (bytes: see mem-radix mode)\n",
        tree.current_size()
    );
    println!(
        "{:<22} {:>10} {:>10} {:>12}",
        "backend", "p50 ns", "p99 ns", "oracle match"
    );
    for (name, ns, matched) in [
        ("radix find_matches", &radix_ns, radix_match),
        ("cuckoo full map", &ck_full_ns, ck_full_match),
        ("cuckoo best-DC", &ck_best_ns, ck_best_match),
    ] {
        println!(
            "{:<22} {:>10} {:>10} {:>11.2}%",
            name,
            percentile(ns, 0.50),
            percentile(ns, 0.99),
            pct(matched),
        );
    }
    println!(
        "\ncuckoo depth accuracy vs exact oracle: {inflated} inflated, {under} under-reported \
         (of {} lookups)",
        reqs.len()
    );
}

fn run_mem_radix(reqs: &[Request], plan: &[usize]) {
    let base = proc_status_kib("VmRSS");
    let tree = build_radix(reqs, plan);
    let grown = proc_status_kib("VmRSS");
    println!(
        "radix tree: {} blocks, RSS growth {:.1} MiB",
        tree.current_size(),
        (grown.saturating_sub(base)) as f64 / 1024.0
    );
    std::hint::black_box(&tree);
}

fn run_mem_cuckoo(reqs: &[Request], plan: &[usize], num_dcs: usize) {
    let sets = build_oracle_sets(reqs, plan, num_dcs);
    let resident: usize = sets.iter().map(HashSet::len).sum();
    let base = proc_status_kib("VmRSS");
    let (filters, _) = build_cuckoo_via_wire(&sets);
    let grown = proc_status_kib("VmRSS");
    let exact: usize = filters.iter().map(CuckooFilter::bytes).sum();
    println!(
        "cuckoo filters: {resident} blocks, exact filter bytes {:.2} MiB, RSS growth {:.1} MiB",
        exact as f64 / (1024.0 * 1024.0),
        (grown.saturating_sub(base)) as f64 / 1024.0
    );
    std::hint::black_box(&filters);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let trace = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("lib/bench/testdata/mooncake_trace_1000.jsonl");
    let num_dcs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4).max(1);
    let mode = args.get(3).map(String::as_str).unwrap_or("quality");

    let reqs = load_requests(trace);
    assert!(!reqs.is_empty(), "trace {trace} yielded no requests");
    let plan = plan_dcs(&reqs, num_dcs);

    match mode {
        "quality" => run_quality(&reqs, &plan, num_dcs),
        "mem-radix" => run_mem_radix(&reqs, &plan),
        "mem-cuckoo" => run_mem_cuckoo(&reqs, &plan, num_dcs),
        other => panic!("unknown mode {other}; expected quality | mem-radix | mem-cuckoo"),
    }
}
