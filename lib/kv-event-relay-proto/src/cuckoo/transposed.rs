// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transposed read cache over a cohort of per-DC filters: `bucket -> DC ->
//! slots` instead of `DC -> bucket -> slots`, so one probe answers membership
//! for every DC in the cohort from two cache lines instead of two scattered
//! bucket reads per DC.
//!
//! This is a derived, router-side layout only: the per-DC [`CuckooFilter`]s
//! stay authoritative and the relay protocol is unchanged. Each lane remains
//! an independent filter — fingerprints are compared per lane, never merged —
//! so false-positive and deletion semantics are exactly those of the source
//! filters.

use super::filter::{CuckooFilter, SLOTS};
use super::overlap::Probe;

/// Fixed cohort width: 8 lanes x one 8-byte slot word = a 64-byte bucket row,
/// exactly one cache line. Cohorts narrower than 8 pad with empty lanes (an
/// all-zero word can never match: fingerprint 0 is remapped at derivation).
pub const COHORT_LANES: usize = 8;

/// One bucket's slot words for every lane, cache-line aligned so a probe's two
/// row loads touch exactly two lines.
#[repr(align(64))]
#[derive(Clone, Copy)]
struct Row([u64; COHORT_LANES]);

const EMPTY_ROW: Row = Row([0; COHORT_LANES]);

/// Broadcast a fingerprint into all four u16 lanes of a slot word.
#[inline]
fn broadcast(fp: u16) -> u64 {
    (fp as u64) * 0x0001_0001_0001_0001
}

/// True when any u16 lane of `x` is zero (SWAR zero-detect).
#[inline]
fn any_u16_zero(x: u64) -> bool {
    x.wrapping_sub(0x0001_0001_0001_0001) & !x & 0x8000_8000_8000_8000 != 0
}

#[inline]
fn pack_slots(slots: &[u16; SLOTS]) -> u64 {
    (slots[0] as u64)
        | (slots[1] as u64) << 16
        | (slots[2] as u64) << 32
        | (slots[3] as u64) << 48
}

/// Why a set of filters cannot share one transposed cohort.
#[derive(Debug, PartialEq, Eq)]
pub enum CohortShapeError {
    /// More filters than [`COHORT_LANES`]; split into several cohorts.
    TooManyLanes,
    /// Probes are derived once per cohort, so every lane must share the seed.
    SeedMismatch,
    /// Bucket indices are derived once per cohort, so every lane must share
    /// the bucket count.
    BucketCountMismatch,
}

/// Transposed read table for up to [`COHORT_LANES`] same-shape filters.
pub struct TransposedCohort {
    rows: Vec<Row>,
    mask: u64,
    lanes: usize,
}

impl TransposedCohort {
    /// Build from a cohort of same-shape filters. Lane `d` mirrors
    /// `filters[d]`; missing lanes stay empty.
    pub fn from_filters(filters: &[&CuckooFilter]) -> Result<Self, CohortShapeError> {
        if filters.len() > COHORT_LANES {
            return Err(CohortShapeError::TooManyLanes);
        }
        let Some(first) = filters.first() else {
            return Ok(Self {
                rows: Vec::new(),
                mask: 0,
                lanes: 0,
            });
        };
        if filters.iter().any(|f| f.seed() != first.seed()) {
            return Err(CohortShapeError::SeedMismatch);
        }
        if filters.iter().any(|f| f.num_buckets() != first.num_buckets()) {
            return Err(CohortShapeError::BucketCountMismatch);
        }
        let num_buckets = first.num_buckets();
        let mut rows = vec![EMPTY_ROW; num_buckets];
        for (lane, filter) in filters.iter().enumerate() {
            for (bucket, row) in rows.iter_mut().enumerate() {
                row.0[lane] = pack_slots(&filter.bucket_slots(bucket));
            }
        }
        Ok(Self {
            rows,
            mask: (num_buckets as u64) - 1,
            lanes: filters.len(),
        })
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    pub fn bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<Row>()
    }

    /// Mirror one source-filter bucket into its lane (the delta-apply path:
    /// after a relay delta touches `bucket` of lane `lane`'s filter).
    pub fn update_lane_bucket(&mut self, lane: usize, bucket: usize, slots: &[u16; SLOTS]) {
        self.rows[bucket].0[lane] = pack_slots(slots);
    }

    #[inline]
    fn row_indices(&self, probe: &Probe) -> (usize, usize) {
        let i1 = (probe.m_index & self.mask) as usize;
        let i2 = i1 ^ ((probe.alt_base & self.mask) | 1) as usize;
        (i1, i2)
    }

    /// Hint both candidate rows into cache ahead of `query_mask`, so batched
    /// walks overlap their memory latency instead of serializing misses.
    #[inline]
    pub fn prefetch_probe(&self, probe: &Probe) {
        let (i1, i2) = self.row_indices(probe);
        prefetch_line(&self.rows[i1] as *const Row as *const u8);
        prefetch_line(&self.rows[i2] as *const Row as *const u8);
    }

    #[inline]
    fn all_lanes_mask(&self) -> u8 {
        if self.lanes == COHORT_LANES {
            u8::MAX
        } else {
            (1u8 << self.lanes) - 1
        }
    }

    /// Membership mask over the cohort for one probe: bit `d` set when lane
    /// `d` may contain the hash. Two row loads answer all lanes at once.
    #[inline]
    pub fn query_mask(&self, probe: &Probe) -> u8 {
        let (i1, i2) = self.row_indices(probe);
        let needle = broadcast(probe.fp);
        let r1 = &self.rows[i1].0;
        let r2 = &self.rows[i2].0;
        let mut mask = 0u8;
        for lane in 0..COHORT_LANES {
            let hit = any_u16_zero(r1[lane] ^ needle) || any_u16_zero(r2[lane] ^ needle);
            mask |= (hit as u8) << lane;
        }
        mask
    }

    /// Contiguous overlap depth for every lane in one shared walk. Two phases:
    ///
    /// 1. **Linear phase** over the first [`LINEAR_PHASE`] positions — probe in
    ///    order, retire a lane at its first miss (authoritative), stop when no
    ///    lane is live. Shallow real-traffic overlaps exit here.
    /// 2. **Shared bisection** for lanes still live: their boundaries lie in
    ///    `(LINEAR_PHASE-1, n]`, and one midpoint probe splits the segment for
    ///    every pending lane at once — O(lanes · log n) probes worst case
    ///    instead of O(n), with each probe answering all lanes from two rows.
    ///
    /// Misses are authoritative, so bisection can never under-report a lane;
    /// like the per-filter searched lookup, a rare false positive can only
    /// inflate a boundary by a bounded window. Depths land in
    /// `depths[..lanes]`.
    pub fn overlap_depths(&self, probes: &[Probe], depths: &mut [u32]) {
        let lanes = self.lanes;
        depths[..lanes].fill(0);
        if lanes == 0 || self.rows.is_empty() || probes.is_empty() {
            return;
        }
        let all = if lanes == COHORT_LANES {
            u8::MAX
        } else {
            (1u8 << lanes) - 1
        };
        let n = probes.len();
        let mut live = all;
        let linear_end = n.min(LINEAR_PHASE);
        for (i, probe) in probes[..linear_end].iter().enumerate() {
            let m = self.query_mask(probe);
            let mut died = live & !m;
            while died != 0 {
                let lane = died.trailing_zeros() as usize;
                depths[lane] = i as u32;
                died &= died - 1;
            }
            live &= m;
            if live == 0 {
                return;
            }
        }
        if linear_end == n {
            let mut survivors = live;
            while survivors != 0 {
                let lane = survivors.trailing_zeros() as usize;
                depths[lane] = n as u32;
                survivors &= survivors - 1;
            }
            return;
        }
        // Segment stack: every entry (lo, hi, mask) says each lane in `mask`
        // is present at `lo` and absent at `hi` (`hi == n` is the virtual
        // end-of-prompt miss). Segments partition the live lanes, so at most
        // `lanes` entries are ever pending.
        let mut stack = [(0usize, 0usize, 0u8); COHORT_LANES + 1];
        let mut sp = 0usize;
        stack[sp] = (linear_end - 1, n, live);
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let (lo, hi, m) = stack[sp];
            if hi - lo == 1 {
                let mut resolved = m;
                while resolved != 0 {
                    let lane = resolved.trailing_zeros() as usize;
                    depths[lane] = hi as u32;
                    resolved &= resolved - 1;
                }
                continue;
            }
            let mid = lo + (hi - lo) / 2;
            let pm = self.query_mask(&probes[mid]);
            let up = m & pm;
            let down = m & !pm;
            if down != 0 {
                stack[sp] = (lo, mid, down);
                sp += 1;
            }
            if up != 0 {
                stack[sp] = (mid, hi, up);
                sp += 1;
            }
        }
    }
}

/// Length of the ordered prefix walk before switching to shared bisection:
/// long enough that shallow overlaps resolve without any bisection probes,
/// short enough that deep prompts pay O(log n) rather than O(n).
const LINEAR_PHASE: usize = 16;

/// How many positions ahead the linear walk prefetches. Adaptive decisions
/// never change which rows position `i` needs, so the lookahead only wastes
/// prefetches on an early exit.
const LINEAR_LOOKAHEAD: usize = 4;

#[inline]
fn prefetch_line(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch is a pure cache hint; it cannot fault and touches no
    // memory architecturally.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

/// One pending bisection segment: every lane in `mask` of cohort `cohort` is
/// present at `lo` and absent at `hi` (`hi == probes.len()` is the virtual
/// end-of-prompt miss).
#[derive(Clone, Copy)]
struct Segment {
    cohort: u32,
    lo: u32,
    hi: u32,
    mask: u8,
}

/// Reusable buffers for [`overlap_depths_multi`], so the hot path never
/// allocates.
#[derive(Default)]
pub struct MultiOverlapScratch {
    lives: Vec<u8>,
    cur: Vec<Segment>,
    next: Vec<Segment>,
}

/// Per-lane overlap depths across many cohorts in one memory-parallel walk.
/// Lane `l` of `cohorts[c]` lands in `depths[c * COHORT_LANES + l]`.
///
/// Semantically identical to running [`TransposedCohort::overlap_depths`] per
/// cohort (the bisection probes the same midpoints; only the order differs).
/// The difference is memory-level parallelism: the linear phase probes every
/// cohort per position with a lookahead prefetch, and the bisection runs
/// breadth-first across all cohorts — each level prefetches every pending
/// midpoint's two rows before reading any of them, so the walk pays ~log(B)
/// rounds of overlapped misses instead of a serial dependent-miss chain.
pub fn overlap_depths_multi(
    cohorts: &[TransposedCohort],
    probes: &[Probe],
    depths: &mut [u32],
    scratch: &mut MultiOverlapScratch,
) {
    let n = probes.len();
    depths.fill(0);
    if n == 0 || cohorts.is_empty() {
        return;
    }
    let lives = &mut scratch.lives;
    lives.clear();
    lives.extend(cohorts.iter().map(TransposedCohort::all_lanes_mask));

    // Linear phase, lockstep across cohorts: per position, every live cohort
    // issues two independent row loads, plus a lookahead prefetch.
    let linear_end = n.min(LINEAR_PHASE);
    for (ci, cohort) in cohorts.iter().enumerate() {
        if lives[ci] != 0 {
            cohort.prefetch_probe(&probes[0]);
        }
    }
    let mut any_live = true;
    for i in 0..linear_end {
        if i + LINEAR_LOOKAHEAD < linear_end {
            for (ci, cohort) in cohorts.iter().enumerate() {
                if lives[ci] != 0 {
                    cohort.prefetch_probe(&probes[i + LINEAR_LOOKAHEAD]);
                }
            }
        }
        let mut live_union = 0u8;
        for (ci, cohort) in cohorts.iter().enumerate() {
            let live = lives[ci];
            if live == 0 {
                continue;
            }
            let m = cohort.query_mask(&probes[i]);
            let mut died = live & !m;
            while died != 0 {
                let lane = died.trailing_zeros() as usize;
                depths[ci * COHORT_LANES + lane] = i as u32;
                died &= died - 1;
            }
            lives[ci] = live & m;
            live_union |= lives[ci];
        }
        if live_union == 0 {
            any_live = false;
            break;
        }
    }
    if !any_live {
        return;
    }
    if linear_end == n {
        for (ci, &live) in lives.iter().enumerate() {
            let mut survivors = live;
            while survivors != 0 {
                let lane = survivors.trailing_zeros() as usize;
                depths[ci * COHORT_LANES + lane] = n as u32;
                survivors &= survivors - 1;
            }
        }
        return;
    }

    // Breadth-first shared bisection: all pending segments of a level are
    // independent, so their rows prefetch together before any is read.
    let cur = &mut scratch.cur;
    let next = &mut scratch.next;
    cur.clear();
    next.clear();
    for (ci, &live) in lives.iter().enumerate() {
        if live != 0 {
            cur.push(Segment {
                cohort: ci as u32,
                lo: (linear_end - 1) as u32,
                hi: n as u32,
                mask: live,
            });
        }
    }
    while !cur.is_empty() {
        for seg in cur.iter() {
            let mid = seg.lo + (seg.hi - seg.lo) / 2;
            cohorts[seg.cohort as usize].prefetch_probe(&probes[mid as usize]);
        }
        for seg in cur.iter() {
            let ci = seg.cohort as usize;
            let mid = seg.lo + (seg.hi - seg.lo) / 2;
            let pm = cohorts[ci].query_mask(&probes[mid as usize]);
            for (part, lo, hi) in [
                (seg.mask & !pm, seg.lo, mid),
                (seg.mask & pm, mid, seg.hi),
            ] {
                if part == 0 {
                    continue;
                }
                if hi - lo == 1 {
                    let mut resolved = part;
                    while resolved != 0 {
                        let lane = resolved.trailing_zeros() as usize;
                        depths[ci * COHORT_LANES + lane] = hi;
                        resolved &= resolved - 1;
                    }
                } else {
                    next.push(Segment {
                        cohort: seg.cohort,
                        lo,
                        hi,
                        mask: part,
                    });
                }
            }
        }
        std::mem::swap(cur, next);
        next.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::filter::DEFAULT_FILTER_SEED;
    use super::super::overlap::probes_for;
    use super::*;

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The transposed mask must agree with per-filter `contains` for every
    /// lane, on both present and absent hashes.
    #[test]
    fn query_mask_matches_per_filter_contains() {
        let mut state = 7u64;
        let mut filters: Vec<CuckooFilter> = (0..5)
            .map(|_| CuckooFilter::provisioned(4096, DEFAULT_FILTER_SEED))
            .collect();
        let mut hashes = Vec::new();
        for _ in 0..4096 {
            let h = splitmix(&mut state).max(1);
            for (d, f) in filters.iter_mut().enumerate() {
                if splitmix(&mut state) % (d as u64 + 2) == 0 {
                    assert!(f.insert(h));
                }
            }
            hashes.push(h);
        }
        let refs: Vec<&CuckooFilter> = filters.iter().collect();
        let cohort = TransposedCohort::from_filters(&refs).unwrap();
        // Absent hashes exercise the miss path too.
        for _ in 0..4096 {
            hashes.push(splitmix(&mut state).max(1));
        }
        for &h in &hashes {
            let probes = probes_for(&[h], DEFAULT_FILTER_SEED);
            let mask = cohort.query_mask(&probes[0]);
            for (d, f) in filters.iter().enumerate() {
                assert_eq!(
                    mask >> d & 1 == 1,
                    f.contains(h),
                    "lane {d} disagrees with its filter for hash {h:#x}"
                );
            }
            // Padded lanes can never report membership.
            assert_eq!(mask >> filters.len(), 0, "padding lanes must stay silent");
        }
    }

    /// The shared walk must report exactly the per-filter linear overlap for
    /// every lane.
    #[test]
    fn overlap_depths_match_per_filter_linear_reference() {
        let mut state = 11u64;
        let mut filters: Vec<CuckooFilter> = (0..COHORT_LANES)
            .map(|_| CuckooFilter::provisioned(8192, DEFAULT_FILTER_SEED))
            .collect();
        // Prefix-chained families with per-lane coverage, like the DC index.
        let mut chains = Vec::new();
        for _ in 0..64 {
            let chain: Vec<u64> = (0..96).map(|_| splitmix(&mut state).max(1)).collect();
            for (d, f) in filters.iter_mut().enumerate() {
                let cov = (splitmix(&mut state) % 97) as usize;
                for &h in &chain[..cov] {
                    assert!(f.insert(h));
                }
                let _ = d;
            }
            chains.push(chain);
        }
        let refs: Vec<&CuckooFilter> = filters.iter().collect();
        let cohort = TransposedCohort::from_filters(&refs).unwrap();
        let mut depths = [0u32; COHORT_LANES];
        for chain in &chains {
            let probes = probes_for(chain, DEFAULT_FILTER_SEED);
            cohort.overlap_depths(&probes, &mut depths);
            for (d, f) in filters.iter().enumerate() {
                let mut expect = 0u32;
                for &h in chain {
                    if f.contains(h) {
                        expect += 1;
                    } else {
                        break;
                    }
                }
                assert_eq!(depths[d], expect, "lane {d} depth diverged");
            }
        }
    }

    /// The lockstep multi-cohort walk must resolve exactly the same depths as
    /// the per-cohort walk (same midpoints, different probe order).
    #[test]
    fn multi_walk_matches_per_cohort_walk() {
        let mut state = 23u64;
        let mut filters: Vec<CuckooFilter> = (0..13)
            .map(|_| CuckooFilter::provisioned(8192, DEFAULT_FILTER_SEED))
            .collect();
        let mut chains = Vec::new();
        for _ in 0..64 {
            let chain: Vec<u64> = (0..160).map(|_| splitmix(&mut state).max(1)).collect();
            for f in filters.iter_mut() {
                let cov = (splitmix(&mut state) % 161) as usize;
                for &h in &chain[..cov] {
                    assert!(f.insert(h));
                }
            }
            chains.push(chain);
        }
        let cohorts: Vec<TransposedCohort> = filters
            .chunks(COHORT_LANES)
            .map(|chunk| {
                let refs: Vec<&CuckooFilter> = chunk.iter().collect();
                TransposedCohort::from_filters(&refs).unwrap()
            })
            .collect();
        let mut scratch = MultiOverlapScratch::default();
        let mut multi = vec![0u32; cohorts.len() * COHORT_LANES];
        let mut single = vec![0u32; cohorts.len() * COHORT_LANES];
        for chain in &chains {
            let probes = probes_for(chain, DEFAULT_FILTER_SEED);
            overlap_depths_multi(&cohorts, &probes, &mut multi, &mut scratch);
            for (ci, cohort) in cohorts.iter().enumerate() {
                cohort.overlap_depths(&probes, &mut single[ci * COHORT_LANES..]);
            }
            assert_eq!(multi, single);
        }
    }

    /// A lane update after a source-filter mutation keeps the cohort in sync.
    #[test]
    fn update_lane_bucket_tracks_source_filter() {
        let mut filters: Vec<CuckooFilter> = (0..2)
            .map(|_| CuckooFilter::provisioned(1024, DEFAULT_FILTER_SEED))
            .collect();
        let refs: Vec<&CuckooFilter> = filters.iter().collect();
        let mut cohort = TransposedCohort::from_filters(&refs).unwrap();

        let h = 0xDEAD_BEEF_u64;
        let mut dirty = Vec::new();
        assert!(filters[1].insert_with(h, |b| dirty.push(b)));
        for &b in &dirty {
            cohort.update_lane_bucket(1, b, &filters[1].bucket_slots(b));
        }
        let probes = probes_for(&[h], DEFAULT_FILTER_SEED);
        assert_eq!(cohort.query_mask(&probes[0]), 0b10);
    }
}
