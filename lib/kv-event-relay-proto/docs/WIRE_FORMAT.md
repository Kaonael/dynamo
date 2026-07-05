<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wire format reference

Exact byte layouts for the relay → global-router transport. All multi-byte
integers are **little-endian**. Golden-byte tests in `src/wire/mod.rs` pin
these layouts.

## gRPC envelope

Protobuf only wraps the envelope; the domain data rides in opaque `payload`
fields encoded as described below.

Common envelope fields (see [`proto/relay.proto`](../proto/relay.proto)):

- `seq: uint64` — monotonic counter per stream; a gap triggers a full filter
  resync.
- `send_ts_us: uint64` — relay wall-clock in µs at send time; used by the
  router to measure delivery lag and staleness.
- `model_key: fixed64` — xxh3 hash of the model-id string with fixed seed
  `0x4D4F_4444_454C_4B31` (`wire::model_id_to_key`). `fixed64` avoids varint
  overhead for high-entropy values. Neither the model-id string nor the DC id
  travels in frames: the DC is implied by the connection, the model by this
  key.
- `instance_id: bytes` — random 128-bit relay-process identity. Carried only
  on the **first** frame of each stream (empty afterwards, saving ~18 bytes
  per message). A mismatch against `GetRelayInfo` means the relay restarted
  and the router must resync.
- `FilterUpdate.heartbeat: bool` — when true the frame carries no payload and
  only refreshes the consumer's freshness clock without advancing the filter
  epoch.

## Metric payload (`wire/metrics.rs`)

Body of `MetricsSnapshot.payload`:

```
[sample_count: u16]
for each sample:
  [key: u16]     stable numeric id (metric_key consts)
  [value: f64]
```

Unknown keys round-trip transparently (forward compatibility). Absent optional
fields are simply not emitted. Defined keys:

| Key | Const | Field |
|-----|-------|-------|
| 0 | `QUEUE_DEPTH` | pending request count |
| 1 | `TTFT_P50_MS` | time-to-first-token p50 |
| 2 | `TTFT_P95_MS` | time-to-first-token p95 |
| 3 | `TTFT_P99_MS` | time-to-first-token p99 |
| 4 | `ITL_P50_MS` | inter-token latency p50 |
| 5 | `ITL_P95_MS` | inter-token latency p95 |
| 6 | `ITL_P99_MS` | inter-token latency p99 |
| 7 | `GPU_UTIL_PCT` | GPU utilisation 0–100 |
| 8 | `KV_UTIL_PCT` | KV cache utilisation 0–100 |
| 9 | `SERVER_ERROR_RATE` | error fraction 0–1 |

`to_frame` / `from_frame` project between the wire frame and the engine-side
`DcModelMetricsSnapshot`; the identity (`dc_id`, `model_id`) is supplied by
the gRPC envelope, never by the frame.
