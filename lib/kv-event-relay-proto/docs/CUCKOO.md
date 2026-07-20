<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CKF bucket-image transport

The relay consumes the public CKF API from `dynamo-kv-router`; that crate owns
event aggregation, cuckoo mutation, and the authoritative bucket-image format.
This crate defines the CBI1 transport for those images.

All multi-byte CBI1 values are little-endian.

## Upstream filter identity

`FilterFormat` carries the fields required to validate and allocate a consumer:

- format version `1`;
- 16-bit nonzero fingerprints;
- four fingerprint slots packed into one `u64` bucket word;
- deterministic xxh3 seed;
- a power-of-two bucket count of at least two.

The relay obtains the seed and bucket count from `DcCkfState::format()`. The
bucket count is fixed when the state is constructed from
`CkfConfig::expected_blocks_per_dc`; the filter does not resize.

## CBI1 header

Every snapshot chunk and delta begins with the same 48-byte header:

```text
[0..4]   magic: b"CBI1"
[4..6]   wire_version: u16       (=1)
[6..8]   flags: u16              (1=SNAPSHOT_CHUNK, 2=DELTA)
[8]      fingerprint_bits: u8    (=16)
[9]      slots_per_bucket: u8    (=4)
[10..12] reserved
[12..20] seed: u64
[20..28] bucket_count: u64
[28..36] dc_worker_id: u64
[36..44] epoch: u64
[44..48] body_checksum: u32      (low 32 bits of xxh3_64(body))
```

The consumer validates magic, wire version, flags, checksum, and filter format
before applying a frame. The DC used for routing is bound to the authenticated
subscriber connection; the header field is not used to select a lane.

## Snapshot chunks

A full lane is encoded as dense `u64` bucket words. Each chunk body is:

```text
[0..4]   chunk_index: u32
[4..8]   chunk_count: u32
[8..16]  bucket_offset: u64
[16..]   bucket words: u64[]
```

`SNAPSHOT_CHUNK_BUCKETS` limits a chunk to 512 Ki buckets, or 4 MiB of words.
`SnapshotAssembly` requires ordered, contiguous chunks with one epoch and one
chunk count. On completion it emits nonzero `BucketImage`s for a reset apply;
any sequence or coverage violation aborts the assembly and forces a new
subscription.

## Deltas

A delta advances one model lane from `base_epoch` to the epoch in its header:

```text
[0..8]   base_epoch: u64
[8..12]  image_count: u32
for each image:
  [0..4]   bucket: u32
  [4..12]  absolute bucket word: u64
```

Bucket words are absolute images rather than operations. The relay coalesces
all touches to a bucket during one publisher interval to its latest value, and
reapplying an image is idempotent. A delta is capped by `max_delta_images()` so
it cannot exceed `IMAGES_MAX_FRAME_BYTES`; a larger publication is sent as a
chunked snapshot.

Epoch continuity is enforced by the consuming backend. A delta
without an installed snapshot or with a mismatched `base_epoch` closes the
connection, and the replacement subscription begins with a current snapshot.

## Ownership boundary

The relay owns its published-lane mirror, dirty tracking, and publication epoch. The
global gateway owns CKF addressing, cohort layout, COW application, and prefix
lookup. `dynamo-kv-event-relay-proto` owns only the format identity, CBI1
encoder/decoder, and ordered snapshot assembly shared by both endpoints.
