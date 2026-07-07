// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Answer the routing question — how deep a contiguous prefix of the request
//! is cached in each DC — in logarithmic probes instead of a linear scan.

use super::filter::{CuckooFilter, I1_SEED_TWEAK, derive_fp, mix};

/// Precompute per-block probes once so large DC fan-outs pay the hash cost only
/// once per request.
pub struct Probe {
    pub(super) fp: u16,
    pub(super) m_index: u64,
    pub(super) alt_base: u64,
}

impl Probe {
    /// Derive a probe from one sequence-hash so the search can reuse it or
    /// derive it lazily only at the indices it touches.
    #[inline]
    fn for_hash(h: u64, seed: u64) -> Probe {
        let fp = derive_fp(h, seed);
        Probe {
            fp,
            m_index: mix(h, seed ^ I1_SEED_TWEAK),
            alt_base: mix(fp as u64, seed),
        }
    }
}

/// Precompute probes when the caller expects to reuse them across many DCs.
pub fn probes_for(seq: &[u64], seed: u64) -> Vec<Probe> {
    seq.iter().map(|&h| Probe::for_hash(h, seed)).collect()
}

/// Keep the linear reference so the search variants stay pinned to the same
/// answer in tests.
#[cfg(test)]
pub(super) fn filter_overlap(filter: &CuckooFilter, probes: &[Probe]) -> u32 {
    let mut depth = 0u32;
    for p in probes {
        let i1 = (p.m_index & filter.mask) as usize;
        if filter.has_at(i1, p.fp) {
            depth += 1;
            continue;
        }
        let i2 = ((i1 as u64) ^ ((p.alt_base & filter.mask) | 1)) as usize;
        if filter.has_at(i2, p.fp) {
            depth += 1;
        } else {
            break;
        }
    }
    depth
}

/// Use authoritative misses to bound overlap depth; only hits can lie.
#[inline]
fn probe_present(filter: &CuckooFilter, p: &Probe) -> bool {
    let i1 = (p.m_index & filter.mask) as usize;
    if filter.has_at(i1, p.fp) {
        return true;
    }
    let i2 = ((i1 as u64) ^ ((p.alt_base & filter.mask) | 1)) as usize;
    filter.has_at(i2, p.fp)
}

/// Derive the probe lazily when a search wants to avoid materializing the full
/// probe array.
#[inline]
fn probe_present_hash(filter: &CuckooFilter, h: u64) -> bool {
    probe_present(filter, &Probe::for_hash(h, filter.seed()))
}

/// Recheck the tail window so a mid-search false positive cannot inflate the
/// reported contiguous prefix.
const OVERLAP_VERIFY_WINDOW: usize = 8;

/// Use exponential plus binary search so deep prefixes stay logarithmic while
/// authoritative misses still bound the answer.
pub fn overlap_depth_searched(filter: &CuckooFilter, probes: &[Probe]) -> u32 {
    let n = probes.len();
    if n == 0 || !probe_present(filter, &probes[0]) {
        return 0;
    }
    // Grow exponentially so the first reliable miss appears without a linear
    // scan.
    let mut hi = 1usize;
    while hi < n && probe_present(filter, &probes[hi]) {
        hi <<= 1;
    }
    let mut lo = hi >> 1;
    let mut r = hi.min(n);
    while r - lo > 1 {
        let m = lo + (r - lo) / 2;
        if probe_present(filter, &probes[m]) {
            lo = m;
        } else {
            r = m;
        }
    }
    // Recheck the tail so a false positive cannot overstate the contiguous
    // prefix.
    let mut depth = r;
    for (k, probe) in probes
        .iter()
        .enumerate()
        .take(r)
        .skip(r.saturating_sub(OVERLAP_VERIFY_WINDOW))
    {
        if !probe_present(filter, probe) {
            depth = k;
            break;
        }
    }
    depth as u32
}

/// Derive probes lazily from the sequence-hash chain so deep prefixes avoid
/// materializing one probe per block.
pub fn overlap_depth_searched_seq(filter: &CuckooFilter, seq: &[u64]) -> u32 {
    let n = seq.len();
    if n == 0 || !probe_present_hash(filter, seq[0]) {
        return 0;
    }
    let mut hi = 1usize;
    while hi < n && probe_present_hash(filter, seq[hi]) {
        hi <<= 1;
    }
    let mut lo = hi >> 1;
    let mut r = hi.min(n);
    while r - lo > 1 {
        let m = lo + (r - lo) / 2;
        if probe_present_hash(filter, seq[m]) {
            lo = m;
        } else {
            r = m;
        }
    }
    let mut depth = r;
    for (k, &hash) in seq
        .iter()
        .enumerate()
        .take(r)
        .skip(r.saturating_sub(OVERLAP_VERIFY_WINDOW))
    {
        if !probe_present_hash(filter, hash) {
            depth = k;
            break;
        }
    }
    depth as u32
}

/// Prune non-contenders with one authoritative probe so only plausible winners
/// pay the logarithmic search cost.
pub fn argmax_overlap_dc(filters: &[CuckooFilter], probes: &[Probe]) -> (usize, u32) {
    let n = probes.len();
    let mut best_dc = 0usize;
    let mut best = 0u32;
    for (dc, filter) in filters.iter().enumerate() {
        // Use the current leader's boundary block as the prune test; a miss
        // there cannot win.
        let bar = if best > 0 { best as usize } else { 0 };
        if bar >= n || !probe_present(filter, &probes[bar]) {
            continue;
        }
        let depth = overlap_depth_searched(filter, probes);
        if depth > best {
            best = depth;
            best_dc = dc;
        }
    }
    (best_dc, best)
}
