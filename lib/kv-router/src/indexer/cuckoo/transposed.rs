// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::ensure;

use super::{CuckooFilter, DeltaEntry, OVERLAP_VERIFY_WINDOW, Probe, SLOTS};

/// Widest cohort one table serves; also bounds every fixed-capacity scratch
/// buffer the search uses, so the hot path never allocates beyond the returned
/// depth vector.
const MAX_DCS: usize = 16;

pub struct TransposedTable {
    num_dcs: usize,
    num_buckets: usize,
    lanes: Vec<AtomicU64>,
    generations: Vec<AtomicU64>,
}

#[derive(Debug)]
pub struct MaskLookup {
    pub depths: Vec<u32>,
    pub conflict_mask: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchPhase {
    GenerationSnapshot,
    FirstProbe,
    ExponentialProbe,
    BinaryProbe,
    VerificationProbe,
    GenerationValidation,
}

/// Hint a bucket's lane row toward cache ahead of its read, so a batch of
/// independent probes overlaps its memory latency instead of serializing
/// misses. A row spans `num_dcs * 8` bytes.
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

impl TransposedTable {
    pub fn from_filters(filters: &[CuckooFilter]) -> anyhow::Result<Self> {
        let Some(first) = filters.first() else {
            anyhow::bail!("transposed table requires at least one filter");
        };
        ensure!(
            filters.len() <= MAX_DCS,
            "transposed table supports at most {MAX_DCS} DCs"
        );
        ensure!(
            filters.iter().all(|filter| {
                filter.num_buckets() == first.num_buckets() && filter.seed() == first.seed()
            }),
            "all transposed filters must have identical shape and seed"
        );
        let num_dcs = filters.len();
        let num_buckets = first.num_buckets();
        let lanes = (0..num_buckets)
            .flat_map(|bucket| {
                filters
                    .iter()
                    .map(move |filter| AtomicU64::new(pack_slots(filter.bucket_slots(bucket))))
            })
            .collect();
        let table = Self {
            num_dcs,
            num_buckets,
            lanes,
            generations: (0..num_dcs).map(|_| AtomicU64::new(0)).collect(),
        };
        table.verify_filters(filters)?;
        Ok(table)
    }

    pub fn num_dcs(&self) -> usize {
        self.num_dcs
    }

    pub fn begin_update(&self, dc: usize) -> u64 {
        let previous = self.generations[dc].fetch_add(1, Ordering::Release);
        assert_eq!(previous & 1, 0, "concurrent writers for DC {dc}");
        previous
    }

    pub fn apply_entries(&self, dc: usize, entries: &[DeltaEntry]) {
        for entry in entries {
            self.lanes[self.lane_index(entry.bucket, dc)]
                .store(pack_slots(entry.slots), Ordering::Release);
        }
    }

    pub fn rebuild_dc(&self, dc: usize, filter: &CuckooFilter) {
        assert_eq!(filter.num_buckets(), self.num_buckets);
        for bucket in 0..self.num_buckets {
            self.lanes[self.lane_index(bucket, dc)]
                .store(pack_slots(filter.bucket_slots(bucket)), Ordering::Release);
        }
    }

    pub fn end_update(&self, dc: usize, previous_even: u64) {
        self.generations[dc].store(previous_even + 2, Ordering::Release);
    }

    pub fn search(&self, probes: &[Probe], available_mask: u16) -> MaskLookup {
        self.search_with_phase_hook(probes, available_mask, |_| {})
    }

    fn search_with_phase_hook(
        &self,
        probes: &[Probe],
        available_mask: u16,
        mut hook: impl FnMut(SearchPhase),
    ) -> MaskLookup {
        let all_mask = if self.num_dcs == MAX_DCS {
            u16::MAX
        } else {
            (1u16 << self.num_dcs) - 1
        };
        let eligible = available_mask & all_mask;
        let mut before = [0u64; MAX_DCS];
        for (dc, generation) in self.generations.iter().enumerate() {
            before[dc] = generation.load(Ordering::Acquire);
        }
        hook(SearchPhase::GenerationSnapshot);
        let mut stable = eligible;
        for (dc, generation) in before.iter().take(self.num_dcs).enumerate() {
            if generation & 1 != 0 {
                stable &= !(1u16 << dc);
            }
        }
        let mut depths = self.search_stable(probes, stable, &mut hook);
        let mut conflict_mask = eligible & !stable;
        hook(SearchPhase::GenerationValidation);
        for (dc, generation) in self.generations.iter().enumerate() {
            let after = generation.load(Ordering::Acquire);
            if after != before[dc] || after & 1 != 0 {
                conflict_mask |= 1u16 << dc;
                depths[dc] = 0;
            }
        }
        MaskLookup {
            depths,
            conflict_mask,
        }
    }

    pub fn verify_filters(&self, filters: &[CuckooFilter]) -> anyhow::Result<()> {
        ensure!(filters.len() == self.num_dcs, "filter count mismatch");
        for (dc, filter) in filters.iter().enumerate() {
            ensure!(
                filter.num_buckets() == self.num_buckets,
                "bucket count mismatch"
            );
            for bucket in 0..self.num_buckets {
                let actual = self.lanes[self.lane_index(bucket, dc)].load(Ordering::Acquire);
                let expected = pack_slots(filter.bucket_slots(bucket));
                ensure!(
                    actual == expected,
                    "transposed lane mismatch: dc={dc}, bucket={bucket}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "bench")]
    pub fn touch_for_benchmark(&self) {
        for lane in &self.lanes {
            std::hint::black_box(lane.load(Ordering::Acquire));
        }
        for generation in &self.generations {
            std::hint::black_box(generation.load(Ordering::Acquire));
        }
    }

    /// Resolve every stable lane's contiguous overlap depth. The probe plan is
    /// the same first-probe / exponential-bracket / binary-refine / verify
    /// sequence as the per-filter searched lookup, but each position is probed
    /// once for all pending lanes, and every batch of position-independent
    /// probes is prefetched before any row is read, so misses overlap instead
    /// of forming a serial dependency chain.
    fn search_stable(
        &self,
        probes: &[Probe],
        stable_mask: u16,
        hook: &mut impl FnMut(SearchPhase),
    ) -> Vec<u32> {
        let mut depths = vec![0u32; self.num_dcs];
        if probes.is_empty() || stable_mask == 0 {
            return depths;
        }
        self.prefetch_probe(&probes[0]);
        // Exponential positions are result-independent, so the opening can
        // stay one probe ahead of its own reads.
        if probes.len() > 1 {
            self.prefetch_probe(&probes[1]);
        }
        let first = self.presence_mask(&probes[0], stable_mask);
        hook(SearchPhase::FirstProbe);
        let active = stable_mask & first;
        let mut lo = [0usize; MAX_DCS];
        let mut hi = [1usize; MAX_DCS];
        let mut unresolved = active;

        let mut probe_index = 1usize;
        while unresolved != 0 && probe_index < probes.len() {
            let lookahead = probe_index << 1;
            if lookahead < probes.len() {
                self.prefetch_probe(&probes[lookahead]);
            }
            let present = self.presence_mask(&probes[probe_index], unresolved);
            hook(SearchPhase::ExponentialProbe);
            let missed = unresolved & !present;
            for_each_dc(missed, |dc| hi[dc] = probe_index);
            for_each_dc(present, |dc| lo[dc] = probe_index);
            unresolved = present;
            probe_index <<= 1;
        }
        if unresolved != 0 {
            for_each_dc(unresolved, |dc| hi[dc] = probes.len());
        }

        // Binary refinement, breadth-first: lanes sharing a midpoint share one
        // probe, and a whole level's midpoints prefetch together before any is
        // read. Lanes partition across groups, so a level holds at most
        // MAX_DCS entries.
        let mut level: [(usize, u16); MAX_DCS] = [(0, 0); MAX_DCS];
        loop {
            let mut level_len = 0usize;
            for_each_dc(active, |dc| {
                if hi[dc] - lo[dc] > 1 {
                    let midpoint = lo[dc] + (hi[dc] - lo[dc]) / 2;
                    if let Some(entry) = level[..level_len]
                        .iter_mut()
                        .find(|(position, _)| *position == midpoint)
                    {
                        entry.1 |= 1u16 << dc;
                    } else {
                        level[level_len] = (midpoint, 1u16 << dc);
                        level_len += 1;
                    }
                }
            });
            if level_len == 0 {
                break;
            }
            for &(midpoint, _) in &level[..level_len] {
                self.prefetch_probe(&probes[midpoint]);
            }
            for &(midpoint, group) in &level[..level_len] {
                let present = self.presence_mask(&probes[midpoint], group);
                hook(SearchPhase::BinaryProbe);
                for_each_dc(group & present, |dc| lo[dc] = midpoint);
                for_each_dc(group & !present, |dc| hi[dc] = midpoint);
            }
        }

        for_each_dc(active, |dc| depths[dc] = hi[dc] as u32);

        // Verification window, ascending so each lane's earliest miss wins —
        // the same boundary recheck as the per-filter searched lookup. Lanes
        // with clustered depths share position probes.
        let mut verification: [(usize, u16); MAX_DCS * OVERLAP_VERIFY_WINDOW] =
            [(0, 0); MAX_DCS * OVERLAP_VERIFY_WINDOW];
        let mut verification_len = 0usize;
        for_each_dc(active, |dc| {
            let end = depths[dc] as usize;
            for index in end.saturating_sub(OVERLAP_VERIFY_WINDOW)..end {
                if let Some(entry) = verification[..verification_len]
                    .iter_mut()
                    .find(|(position, _)| *position == index)
                {
                    entry.1 |= 1u16 << dc;
                } else {
                    verification[verification_len] = (index, 1u16 << dc);
                    verification_len += 1;
                }
            }
        });
        let verification = &mut verification[..verification_len];
        verification.sort_unstable_by_key(|&(position, _)| position);
        for &(position, _) in verification.iter() {
            self.prefetch_probe(&probes[position]);
        }
        let mut verified = active;
        for &(position, candidates) in verification.iter() {
            let candidates = candidates & verified;
            if candidates == 0 {
                continue;
            }
            let present = self.presence_mask(&probes[position], candidates);
            hook(SearchPhase::VerificationProbe);
            let missed = candidates & !present;
            for_each_dc(missed, |dc| depths[dc] = position as u32);
            verified &= !missed;
        }
        depths
    }

    #[inline]
    fn presence_mask(&self, probe: &Probe, candidates: u16) -> u16 {
        let mask = (self.num_buckets - 1) as u64;
        let (first, second) = probe.bucket_indices(mask);
        let needle = broadcast(probe.fingerprint());
        let mut present = 0u16;
        for_each_dc(candidates, |dc| {
            let first_lane = self.lanes[self.lane_index(first, dc)].load(Ordering::Acquire);
            if any_u16_zero(first_lane ^ needle) {
                present |= 1u16 << dc;
                return;
            }
            let second_lane = self.lanes[self.lane_index(second, dc)].load(Ordering::Acquire);
            if any_u16_zero(second_lane ^ needle) {
                present |= 1u16 << dc;
            }
        });
        present
    }

    #[inline]
    fn prefetch_probe(&self, probe: &Probe) {
        let mask = (self.num_buckets - 1) as u64;
        let (first, second) = probe.bucket_indices(mask);
        for bucket in [first, second] {
            let row = &self.lanes[self.lane_index(bucket, 0)] as *const AtomicU64 as *const u8;
            prefetch_line(row);
            if self.num_dcs > 8 {
                // A 9..=16-lane row spans a second cache line.
                prefetch_line(unsafe { row.add(64) });
            }
        }
    }

    #[inline]
    fn lane_index(&self, bucket: usize, dc: usize) -> usize {
        bucket * self.num_dcs + dc
    }
}

/// Broadcast a fingerprint into all four u16 lanes of a slot word.
#[inline]
fn broadcast(fingerprint: u16) -> u64 {
    (fingerprint as u64) * 0x0001_0001_0001_0001
}

/// True when any u16 lane of `x` is zero (SWAR zero-detect). Empty slots hold
/// fingerprint 0, which derivation never produces, so an all-zero word can
/// never match a real fingerprint.
#[inline]
fn any_u16_zero(x: u64) -> bool {
    x.wrapping_sub(0x0001_0001_0001_0001) & !x & 0x8000_8000_8000_8000 != 0
}

#[inline]
fn pack_slots(slots: [u16; SLOTS]) -> u64 {
    slots
        .iter()
        .enumerate()
        .fold(0u64, |packed, (index, slot)| {
            packed | (u64::from(*slot) << (index * 16))
        })
}

fn for_each_dc(mut mask: u16, mut callback: impl FnMut(usize)) {
    while mask != 0 {
        let dc = mask.trailing_zeros() as usize;
        callback(dc);
        mask &= mask - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DEFAULT_FILTER_SEED, overlap_depth_searched, probes_for};
    use super::*;

    #[test]
    fn transposed_and_native_search_agree() {
        let mut filters: Vec<CuckooFilter> = (0..4)
            .map(|_| CuckooFilter::provisioned(512, DEFAULT_FILTER_SEED))
            .collect();
        let sequence: Vec<u64> = (1..=128).collect();
        for (dc, filter) in filters.iter_mut().enumerate() {
            for &hash in sequence.iter().take(17 + dc * 13) {
                assert!(filter.insert(hash));
            }
        }
        let probes = probes_for(&sequence, DEFAULT_FILTER_SEED);
        let table = TransposedTable::from_filters(&filters).unwrap();
        let result = table.search(&probes, 0b1111);
        assert_eq!(result.conflict_mask, 0);
        for (dc, filter) in filters.iter().enumerate() {
            assert_eq!(
                result.depths[dc],
                overlap_depth_searched(filter, &probes),
                "DC {dc}"
            );
        }
    }

    #[test]
    fn updating_one_generation_only_conflicts_that_dc() {
        let filters: Vec<CuckooFilter> = (0..2)
            .map(|_| CuckooFilter::provisioned(64, DEFAULT_FILTER_SEED))
            .collect();
        let table = TransposedTable::from_filters(&filters).unwrap();
        let even = table.begin_update(1);
        let probes = probes_for(&[1, 2, 3], DEFAULT_FILTER_SEED);
        let result = table.search(&probes, 0b11);
        assert_eq!(result.conflict_mask, 0b10);
        table.end_update(1, even);
    }

    #[test]
    fn generation_changes_are_detected_during_every_search_phase() {
        let sequence: Vec<u64> = (1..=128).collect();
        let mut filters: Vec<CuckooFilter> = (0..2)
            .map(|_| CuckooFilter::provisioned(512, DEFAULT_FILTER_SEED))
            .collect();
        for filter in &mut filters {
            for &hash in sequence.iter().take(40) {
                assert!(filter.insert(hash));
            }
        }
        let probes = probes_for(&sequence, DEFAULT_FILTER_SEED);
        let table = TransposedTable::from_filters(&filters).unwrap();
        for target in [
            SearchPhase::GenerationSnapshot,
            SearchPhase::FirstProbe,
            SearchPhase::ExponentialProbe,
            SearchPhase::BinaryProbe,
            SearchPhase::VerificationProbe,
            SearchPhase::GenerationValidation,
        ] {
            let mut changed = false;
            let result = table.search_with_phase_hook(&probes, 0b11, |phase| {
                if phase == target && !changed {
                    let generation = table.begin_update(1);
                    table.end_update(1, generation);
                    changed = true;
                }
            });
            assert!(changed, "search did not reach {target:?}");
            assert_eq!(result.conflict_mask, 0b10, "phase {target:?}");
            assert_eq!(
                result.depths[0],
                overlap_depth_searched(&filters[0], &probes),
                "stable DC was lost during {target:?}"
            );
        }
    }

    /// The SWAR lane compare must agree with per-filter `contains` for both
    /// present and absent hashes, including empty (all-zero) lanes.
    #[test]
    fn presence_matches_per_filter_contains() {
        let mut filters: Vec<CuckooFilter> = (0..3)
            .map(|_| CuckooFilter::provisioned(2048, DEFAULT_FILTER_SEED))
            .collect();
        let mut state = 42u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)).max(1)
        };
        let mut hashes = Vec::new();
        for _ in 0..2048 {
            let hash = next();
            for (dc, filter) in filters.iter_mut().enumerate() {
                if hash % (dc as u64 + 2) == 0 {
                    assert!(filter.insert(hash));
                }
            }
            hashes.push(hash);
        }
        for _ in 0..2048 {
            hashes.push(next());
        }
        let table = TransposedTable::from_filters(&filters).unwrap();
        for &hash in &hashes {
            let probes = probes_for(&[hash], DEFAULT_FILTER_SEED);
            let present = table.presence_mask(&probes[0], 0b111);
            for (dc, filter) in filters.iter().enumerate() {
                assert_eq!(
                    present >> dc & 1 == 1,
                    filter.contains(hash),
                    "lane {dc} disagrees for hash {hash:#x}"
                );
            }
        }
    }
}
