<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-event-relay-proto

Shared transport contract between the per-DC `kv-event-relay` (server) and the
`global-router` (client). Both crates depend on this one, so neither side can
drift from the other's encoding.

The crate covers three things:

1. **gRPC protocol** — how a router talks to a relay.
2. **Wire codec** — how metrics are packed into bytes, and how model ids
   become fixed 64-bit keys.
3. **CKF1 cuckoo filter** — how a whole DC's KV cache is summarized into a
   compact structure the router can query locally.

This README is a high-level overview. Exact byte layouts and data-structure internals
live in [docs/WIRE_FORMAT.md](docs/WIRE_FORMAT.md) and
[docs/CUCKOO.md](docs/CUCKOO.md).

---

## 1. gRPC protocol

Defined in [`proto/relay.proto`](proto/relay.proto), exposed as tonic stubs
under `v1::KvEventRelay`.

The router opens **one mTLS connection per DC** and multiplexes several
long-lived streams over it:

- **`SubscribeFilter`** — a ready-made cuckoo filter of the DC's cache,
  shipped as a full snapshot on connect plus small incremental updates. This
  is the feed that keeps the router's `SnapshotCuckoo` index in sync; a
  detected gap or reconnect heals via a fresh full snapshot.
- **`SubscribeMetrics`** — periodic telemetry (queue depth, latency
  percentiles, GPU/KV utilization) that the router uses to score DCs.
- **`GetRelayInfo`** — the relay's instance id and the DC's KV block size,
  which the router validates against its topology before ingesting.

### Surviving restarts

A relay restart silently resets its state, which would corrupt a router that
kept applying updates as if nothing happened. To prevent this, each relay
process generates a random **instance id** at startup and stamps it on the
first frame of every stream. When the router sees an unfamiliar id, it knows
the relay restarted and does a full resync instead of trusting its cursor.

### Keeping frames small

Frames avoid carrying anything that can be derived elsewhere: the DC is
implied by the connection, the model travels as a fixed 64-bit hash instead of
a string, and the instance id is sent only once per stream. The actual data
rides in an opaque `payload` field encoded by this crate's own codec (below),
not as nested protobuf.

---

## 2. Wire codec (`src/wire/`)

Payloads are flat little-endian binary: fixed-size numbers written in order,
no field tags, no varints. This makes encoding a single buffer append and
decoding a linear scan.

- **Metrics** (`wire/metrics.rs`) — a list of `(numeric key, f64 value)`
  pairs. Keys are stable numeric ids (`metric_key::*`), so either side can add
  new metrics without breaking the other: unknown keys pass through, missing
  keys read as "not reported".
- **Model keys** (`wire/model.rs`) — `model_id_to_key` hashes a model-id
  string to the 64-bit key used in every frame. The hash is seeded and
  deterministic, so relay and router always agree on it.

The decoder returns typed errors (truncation vs. shape mismatch), so recovery
logic can decide whether to retry, resync, or drop.

Byte-level layouts: [docs/WIRE_FORMAT.md](docs/WIRE_FORMAT.md).

---

## 3. Cuckoo filter (`src/cuckoo/`)

The idea behind the `SnapshotCuckoo` backend: instead of streaming every cache
event to the router, the **relay** maintains a [cuckoo
filter](https://en.wikipedia.org/wiki/Cuckoo_filter) — a compact "is this
block cached here?" structure — and ships it to the router. The router then
answers routing queries entirely from local memory.

A cuckoo filter is an approximate set: it stores small fingerprints instead of
full hashes, so it is orders of magnitude smaller than the real block set. The
trade-off is rare false positives; a "no" is always reliable, a "yes" almost
always. It also supports deletion, which a plain Bloom filter does not — and
the relay needs that, because cache blocks get evicted.

### How it travels (the CKF1 format)

Filter bytes move over `SubscribeFilter` in frames tagged with the magic
`CKF1`. Two kinds:

- **Full snapshot** — the entire filter, split into ~4 MiB chunks to stay
  under gRPC message limits. The router reassembles chunks directly into a
  pre-allocated filter (`SnapshotAssembler`), never buffering the whole
  snapshot twice.
- **Delta** — only the buckets that changed since the last publish. The relay
  tracks dirty buckets as it applies events (`SnapshotProducer`), so a quiet
  DC ships a few hundred bytes instead of megabytes.

Every frame carries an **epoch** number, and a delta names the epoch it was
built against. If the router's epoch doesn't match — a dropped update, a
reconnect — the delta is rejected and a fresh full snapshot heals the
divergence. Checksums catch corruption; malformed frames are typed errors, not
silent misreads. When churn is high enough that a delta wouldn't be
meaningfully smaller than a snapshot, the relay just sends the snapshot.

### How it's queried (`overlap.rs`)

The routing question is not "is this one block cached?" but "**how long a
prefix** of this request is already cached in each DC?" — the DC with the
deepest cached prefix skips the most prefill work.

The naive approach probes block by block until the first miss. For prompts
thousands of blocks deep that's slow, so `overlap_depth_searched` uses
exponential-then-binary search over the prefix, landing in O(log n) probes. A
small verification window at the end re-checks the boundary, so a lone false
positive can't inflate the reported depth. When comparing many DCs at once,
`argmax_overlap_dc` first checks each candidate at the current leader's depth
— a DC that misses there can't win and is skipped without a full search.

Filter internals and the CKF1 frame layout: [docs/CUCKOO.md](docs/CUCKOO.md).

### Try it: compare against the kv-router radix tree

`examples/radix_vs_cuckoo.rs` feeds identical Stored events into both the
upstream kv-router `RadixTree` and this crate's per-DC cuckoo filters (built
through the real CKF1 snapshot + delta wire path), then re-queries every
request against both, with an exact `HashSet` oracle as ground truth:

```text
cargo run -p dynamo-kv-event-relay-proto --release --example radix_vs_cuckoo -- \
    lib/bench/testdata/mooncake_trace_1000.jsonl 4
```

Reports per-lookup p50/p99 latency, routing-choice agreement with the oracle,
cuckoo depth inflation/under-reporting, and index size. `mem-radix` /
`mem-cuckoo` modes measure RSS growth of each backend in isolation. Any JSONL
trace with per-request `hash_ids` (mooncake format) works, so the comparison
scales to millions of blocks with generated traces.
