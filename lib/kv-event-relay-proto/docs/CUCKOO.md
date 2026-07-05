<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CKF1 cuckoo filter reference

Internals of the seeded cuckoo filter and its CKF1 wire encoding
(`src/cuckoo/`). All multi-byte integers are **little-endian**. Structural
tests in `src/cuckoo/tests.rs` pin the format.

## Filter structure (`filter.rs`)

- **Fingerprints**: 16-bit (`u16`), derived from the 64-bit block hash with a
  murmur-style finaliser mix + seed. Fingerprint `0` is the empty-slot
  sentinel; `derive_fp` remaps 0 → 1. False-positive rate is ≈ 2⁻¹³ per probe
  at 4 slots/bucket.
- **Buckets**: power-of-two count, 4 slots each → filter size is
  `num_buckets × 4 × 2` bytes.
- **Addressing**: primary bucket `i1 = mix(h, seed ^ I1_SEED_TWEAK) & mask`;
  alternate `i2 = i1 XOR ((mix(fp, seed) & mask) | 1)`. The `| 1` keeps the
  offset odd so `i2 ≠ i1` always.
- **Insertion**: try `i1`, then `i2`, then a kick-out chain of up to
  `MAX_KICKS = 500` displacements. On failure the chain is **rolled back**
  (pre-mutation bucket snapshots are kept, restored in reverse), so a full
  filter is left unmodified and the caller gets a clean "needs rebuild"
  signal.
- **Sizing**: `provisioned(n)` targets 80% load (headroom for churn),
  `with_capacity_seeded(n)` targets 95%.
- **Seed**: `DEFAULT_FILTER_SEED = 0x5DEE_CE66_D1B5_4A33`; producer and
  consumer must share the seed, and it travels in every CKF1 header.
- **Storage**: `BucketPages` — page-granular copy-on-write fingerprint array.
  Cloning a filter for a point-in-time snapshot shares untouched pages;
  mutation copies only dirty pages.

## CKF1 frame header

Every frame (chunk or delta) starts with a 48-byte header, magic `b"CKF1"`:

```
[0..4]   magic: b"CKF1"
[4..6]   version: u16       (=1)
[6..8]   flags: u16         1=DELTA, 2=CHUNK (exact values, not a bitmask)
[8]      fp_bits: u8        (=16)
[9]      slots_per_bucket: u8 (=4)
[10..12] reserved
[12..20] seed: u64
[20..28] num_buckets: u64   (must be a power of two)
[28..36] dc_worker_id: u64
[36..44] epoch: u64
[44..48] body_checksum: u32 (xxh3_64 of body, truncated)
```

The checksum detects corruption only; integrity/authenticity comes from mTLS.
`fp_bits`/`slots_per_bucket` mismatches are rejected as
`IncompatibleFilterParams`.

## Full snapshot chunks (`flags = CHUNK`)

Large filters are split into ~4 MiB chunks (`CHUNK_BUCKETS` buckets each) to
stay under gRPC message limits. Chunk body:

```
[0..4]   chunk_index: u32
[4..8]   chunk_count: u32
[8..16]  bucket_offset: u64
[16..]   slot data: buckets_in_chunk × 4 × u16
```

`SnapshotAssembler` reconstructs the filter in place as chunks arrive: chunk 0
allocates the filter from header metadata, each subsequent chunk writes its
slot range directly into the bucket array. Chunks must arrive in order with
matching metadata; any violation (`ChunkSequenceMismatch`,
`IncompleteCoverage`, checksum failure) drops partial state and forces a fresh
snapshot. No whole-snapshot intermediate buffer ever exists on the receiver.

## Incremental deltas (`flags = DELTA`)

Only changed buckets are sent. Body:

```
[0..8]   base_epoch: u64    (must equal the receiver's current epoch)
[8..12]  changed_count: u32
for each changed bucket:
  [0..4]  bucket_index: u32
  [4..12] new slots: 4 × u16
```

`apply_delta` is all-or-nothing per contract violation: a `base_epoch` gap,
shape/seed mismatch, checksum failure or out-of-range bucket index is a typed
`DeltaError` that forces the consumer to resync from a full snapshot instead
of silently diverging. Filter `len` is maintained incrementally from the slot
diff.

## Producer (`producer.rs`)

`SnapshotProducer` keeps one DC's filter in sync with the authoritative
resident set and decides what to publish:

- Dirty buckets are tracked in a **bitmap** (allocation-free marking on the
  event hot path; draining yields sorted indices for free).
- `publish()` returns `Full(SnapshotState)` when there is no stable delta base
  (first publish, after `rebuild()`), when churn makes a delta pointless
  (`delta_bytes × 2 ≥ full_bytes`), or when the delta would exceed
  `MAX_DELTA_BYTES = 32 MiB`; otherwise `Delta(bytes)` or `Unchanged`.
- Epochs advance on every successful publish; `new_with_epoch` lets a
  replacement producer resume an epoch chain so consumers don't misread a
  handover as a replay reset.
- `full_snapshot()` captures a point-in-time `SnapshotState` (COW page share)
  so chunk serialization happens outside the caller's lock.
- If `insert()` returns false the filter is full and the caller must
  `rebuild()` from the authoritative resident set.

## Overlap search (`overlap.rs`)

The routing query: given the request's sequence of block hashes, find the
longest **contiguous cached prefix** per DC.

- `overlap_depth_searched(filter, probes)` — exponential growth (double `hi`
  until first miss) then binary search on `[hi/2, hi]`: O(log n) probes
  instead of a linear scan. A trailing `OVERLAP_VERIFY_WINDOW = 8` recheck
  re-probes the last blocks before the boundary so a single mid-search false
  positive cannot overstate the depth. Misses are authoritative, so the search
  can only err by the (rare) false-positive margin.
- `overlap_depth_searched_seq(filter, seq)` — same algorithm but derives each
  probe lazily from the hash slice; avoids materializing a `Vec<Probe>`. This
  is what the global-router's `SnapshotBackend` uses per DC.
- `probes_for(seq, seed)` + `argmax_overlap_dc(filters, probes)` — best-DC
  path: probes are precomputed once, then each DC is first tested at the
  current leader's boundary block — a miss there cannot win, so
  non-contenders skip the logarithmic search entirely. O(D + log B) for D DCs
  and prefix depth B.
