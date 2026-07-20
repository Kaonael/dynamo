<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-event-relay-proto

Shared transport contract between the per-DC `kv-event-relay` server and the
`global-gateway` client. Both endpoints depend on this crate so protobuf stubs,
CBI1 framing, format validation, and integrity checks remain identical.

The crate provides:

1. Generated tonic client and server types for `KvEventRelay`.
2. Compact wire codecs for model keys, routing metrics, and CKF bucket images.

Exact payload layouts are documented in
[docs/WIRE_FORMAT.md](docs/WIRE_FORMAT.md). The bucket-image contract is
documented in [docs/CUCKOO.md](docs/CUCKOO.md).

## gRPC contract

The service is defined in [`proto/relay.proto`](proto/relay.proto). A global
router opens one mTLS connection per DC and uses three RPCs:

- `SubscribeFilter` streams a full per-model CBI1 snapshot on subscription,
  followed by deltas or heartbeats.
- `SubscribeMetrics` streams compact per-model routing telemetry.
- `GetRelayInfo` returns the relay process identity and KV block size.

The relay stamps its process `instance_id` on the first frame of each stream.
The subscriber fences the stream against the identity returned by
`GetRelayInfo`; a mismatch closes the connection and starts a new subscription.

The DC identity comes from the authenticated connection and topology entry.
The model travels as a deterministic 64-bit `model_key`, so filter payloads do
not repeat either string identity.

## CBI1 publication

Each relay owns one upstream `DcCkfState` per model. Deduplicated DC-level
events feed it, and each publication yields absolute bucket images. The relay
mirrors those images and publishes:

- a delta containing the latest value of every bucket changed during the
  publisher interval;
- a bounded sequence of dense snapshot chunks after a lane reset, when a delta
  would exceed the frame cap, and at the start of every subscription;
- a heartbeat when the lane did not change.

Every CBI1 frame carries the upstream format identity: seed, bucket count,
fingerprint width, and slots per bucket. The global gateway adopts the first
format seen for a model and rejects drift from any DC. Deltas carry
`base_epoch` and `epoch`; a missing base or epoch gap closes the connection so
the next subscription installs a current snapshot.

## Compatibility rules

- All relays serving one global-gateway model must use the same CKF seed and
  capacity, producing the same `FilterFormat`.
- Bucket images are absolute values, so repeated application is idempotent.
- Epoch continuity is enforced by the global-gateway backend, not the CBI1
  decoder.
- The CBI1 `dc_worker_id` header field is informational; routing uses the DC
  bound to the subscriber connection.

Relay publication state lives in `dynamo-kv-event-relay`; replica layout and
lookup live in `dynamo-global-gateway`. This crate owns only their shared
transport contract.
