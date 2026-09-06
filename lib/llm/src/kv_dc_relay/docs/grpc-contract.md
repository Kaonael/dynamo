<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV DC Relay: gRPC Contract

Service `dynamo.kvrelay.v1.KvEventRelay`, defined in
[`../protocol/relay.proto`](../protocol/relay.proto). This document is the
contract-level view: RPCs, message semantics, validation, and lifecycle rules.
Byte-level framing of the CKF payload (CBI1) and identity derivation are
specified in [`../protocol/README.md`](../protocol/README.md); the producer
model behind the contract is in [`architecture.md`](architecture.md).

## Envelope and versioning

Every top-level request and response carries a `fixed32 contract_marker` at
field number 127. Its required value is `0x4B565231` (`KVR1`). Every response
also carries `protocol_version` and the typed `RelayIdentity`. The server rejects
a mismatched request marker with `FAILED_PRECONDITION` / `CONTRACT_MISMATCH`.
Consumers reject a mismatched response marker or version before applying any state.

`RELAY_PROTOCOL_VERSION` is **1**. This is the first WAN contract intended for
deployment. Earlier branch-local prototypes are not accepted; Relay and
consumers must use the same compatibility major, not necessarily the same Dynamo release.

### v1 Evolution

- Keep package `dynamo.kvrelay.v1`, protocol version `1`, and marker `KVR1` for additive changes.
- Preserve existing field numbers, types, meanings, RPC names, and error reason names/numbers.
  Reserve both numbers and names when removing fields or enum values; never reuse them.
- Add fields only when absence preserves previous behavior. Readers ignore unknown protobuf
  fields; writers cannot assume old readers retain them when reconstructing a request.
- New enum values and `oneof` alternatives are not automatically semantically compatible.
  Advertise new per-pool formats only with an explicit discriminator; old readers quarantine
  the affected entry. Do not introduce new mandatory filter frame kinds into a v1 stream.
- Identity composition, existing hash/CKF interpretations, and required behavior cannot change
  under v1. Use a new package/service major for incompatible changes and keep v1 during migration.

### Unknown Types and Validation

Validate the envelope first, then each complete snapshot entry independently. The wire helpers
return errors without mutating state; `WireIdentityError::is_unsupported()` distinguishes an
unsupported entry from malformed known data. `*_UNSPECIFIED`, missing required fields, duplicate
sets, and namespace inconsistencies remain invalid. `ServingReadinessState.UNKNOWN` is a valid
state, not an unknown enum value.

| Observation | Required scope and action |
| --- | --- |
| Unknown pool identity source/version, CKF/hash format, pool role, or model target | Quarantine the containing pool, including its previous CKF/load state. Keep unrelated supported pools from the complete snapshot. |
| Unknown topology state or role, including member/adapter dependencies | Quarantine the entire topology entry; never remove the unknown dependency and infer `READY`. Keep unrelated entries. |
| Unknown filter frame kind or unsupported CBI1 payload | Invalidate only that pool replica and close its stream. Do not skip a frame and continue applying deltas. |
| Missing or unknown `ModelTarget.target` alternative | Treat the pool as unsupported. Protobuf readers cannot distinguish an unknown alternative from an unset `oneof`; never infer a base target. |
| Malformed entry with a trustworthy key | Quarantine that entry; never retain its old usable state as a fallback. |
| Invalid envelope, duplicate top-level keys, or entries that cannot be identified safely | Invalidate the affected state plane; do not partially apply an ambiguous snapshot. |

An unsupported entry is not permission to use default semantics. A newly received complete
snapshot replaces the previous supported view, even if some entries are quarantined. These are
consumer requirements; this package does not implement a consumer state machine.

### Producer Identity Key

`ProducerIdentity` is the frozen v1 subscription key. Equality includes:

- `pool_id.identity_version`, both domain digests **and their sources**, and `dc_id`;
- `producer_incarnation` and `layout_generation`;
- every existing `ckf_format` field: version, seed, bucket count, fingerprint bits, and slots per bucket.

Rust callers use `ProducerKey::try_from(&identity)` to validate and compare these explicit fields.
Metadata belongs in `KvPoolDescriptor`, not in the identity or its nested messages. Endpoint,
registrations, roles, and future descriptive fields do not participate in generation equality.
Their updates still require catalog/query/readiness revalidation, but do not alone restart a CKF
subscription. Never change the meaning of an existing key field or silently drop it from equality.

### Machine-Readable Errors

Relay application errors include the ASCII trailing metadata key `kv-relay-error-reason`.
Its value is the exact protobuf `RelayErrorReason` name, for example
`RELAY_ERROR_REASON_PRODUCER_CHANGED`. This is a custom trailer, not `google.rpc.Status`
inside `grpc-status-details-bin`; no error payload is inserted into the data stream.
Rust callers can use `relay_error_reason(&tonic::Status)`. Diagnostic message text is not stable.

The table omits the common `RELAY_ERROR_REASON_` prefix:

| Reason | gRPC code | Client action |
| --- | --- | --- |
| `CONTRACT_MISMATCH` | `FAILED_PRECONDITION` | Stop retries with this contract; use a compatible client/server major. |
| `INVALID_REQUEST` | `INVALID_ARGUMENT` | Fix the request; do not retry unchanged. |
| `UNSUPPORTED_FEATURE` | `FAILED_PRECONDITION` | Quarantine the requested pool; wait for supported semantics or upgrade. |
| `POOL_NOT_FOUND` | `NOT_FOUND` | Refresh catalog, drop withdrawn state; subscribe only to an advertised producer. |
| `PRODUCER_CHANGED` | `FAILED_PRECONDITION` | Drop the old replica, refresh catalog, and subscribe to the current key. |
| `RESOURCE_LIMIT` | `RESOURCE_EXHAUSTED` | Retry admission with bounded exponential backoff and jitter; reduce subscriptions if persistent. |
| `SUBSCRIBER_LAGGED` | `RESOURCE_EXHAUSTED` | Invalidate this stream's state and reopen from a full snapshot/window with backoff. Other planes remain independent. |
| `SNAPSHOT_PROGRESS_TIMEOUT` | `RESOURCE_EXHAUSTED` | Discard incomplete assembly, fix draining/backpressure, and reopen with backoff. |
| `PUBLICATION_UNAVAILABLE` | `UNAVAILABLE` | Invalidate the affected stream state, refresh catalog for pool streams, and reconnect with backoff. |
| `INVALID_PUBLICATION` | `FAILED_PRECONDITION` | Invalidate the replica and refresh catalog; do not keep retrying the same fenced generation. |
| `INTERNAL` | `INTERNAL` | Invalidate the affected stream state, refresh catalog, and retry with backoff; surface persistent failures. |

Older servers, proxies, and gRPC itself can return statuses without this trailer. Unknown or
`UNSPECIFIED` reasons are equivalent to absent reasons: use the status code, never message parsing.
For `FAILED_PRECONDITION`, refresh identity/catalog and verify the envelope; do not blindly retry
an unchanged request. For `RESOURCE_EXHAUSTED` or `UNAVAILABLE`, reconnect with backoff; a pool
stream always starts with a new snapshot. Request errors need correction. Authentication and
authorization errors belong to the external proxy. An unexpected clean EOF also invalidates the
affected stream state and requires reconnecting; it is not proof that cached state remains fresh.

## RPCs

| RPC | Kind | Purpose |
| --- | --- | --- |
| `GetRelayInfo` | unary | Protocol version and typed Relay identity |
| `WatchKvPoolCatalog` | server stream | Complete revisioned pool-catalog snapshots |
| `SubscribeKvPool` | server stream | CKF snapshot + deltas for one exact producer generation |
| `SubscribeServingReadiness` | server stream | Complete namespace topology projections |
| `SubscribeKvPoolLoad` | server stream | Complete pool-load windows |

Streaming requests require a non-empty `subscriber_id` (≤ 128 bytes). Each
stream type has an independent subscriber limit; pool publication additionally
bounds total pool streams, subscribers per pool, and initialized publication
hubs. Breaching any bound returns `RESOURCE_EXHAUSTED`.

The server's default maximum message size is 8 MiB. Configure consumers and sidecars to accept
that size as well: a full snapshot chunk carries 4 MiB of bucket words plus CBI1 and protobuf
headers, exceeding tonic's default 4 MiB client receive limit. Rust clients can use
`KvEventRelayClient::max_decoding_message_size(8 * 1024 * 1024)`.

## Transport security boundary

The Relay server is plaintext HTTP/2 gRPC. It does not provide encryption,
server authentication, or client authentication. mTLS is optional and implemented only by an
external sidecar; Relay has no TLS flags or certificate configuration. A sidecar is not required
on a trusted, isolated network. A deployment that crosses a
trust boundary must bind Relay to loopback or another isolated interface and
place a TLS- or mutual-TLS-terminating gRPC proxy in front of it. Only the proxy
listener should be exposed; a firewall or NetworkPolicy must prevent direct
access to the Relay listener. Certificate validation, authorization, rotation,
and expiry monitoring belong to the proxy.

## Stream contracts

### WatchKvPoolCatalog

Emits `KvPoolCatalogUpdate` — a **complete snapshot** of all live pools per
revision. There are no tombstones anywhere in the contract: a pool absent from
the next snapshot is withdrawn.

`KvPoolDescriptor`:

| Field | Semantics |
| --- | --- |
| `producer: ProducerIdentity` | Pool identity + generation + CKF format. The subscription key for `SubscribeKvPool`. |
| `serving_endpoint` | The Dynamo endpoint owning the pool. Descriptor metadata for consumer-side resolution — **not** an ingress and not a `PoolId` dimension. |
| `registrations[]` | Canonical model id, `ModelTarget` (base, or LoRA `{base_model, adapter}`), normalized aliases. Cross-pool name uniqueness is deliberately not enforced. |
| `query_semantics` | Atomic token→sequence-hash pipeline: `kv_block_size` (> 0 required) + closed `KvQueryHashFormat`. Consumers must reject unknown formats. |
| `pool_roles[]` | Worker roles **declared** by the endpoint's current base cards, independent of liveness (`LEGACY` for cards without a role). Non-empty required. Live roles are published in the topology stream instead. |

Same-target name repeats across pools are allowed. If one lookup name resolves to more than one
distinct `BindingIdentity`, a consumer name index omits it fail-closed.

### SubscribeKvPool

The request names an exact `expected_producer: ProducerIdentity` taken from
the catalog — not just a pool. A stale generation is rejected with
`FAILED_PRECONDITION` (producer mismatch), an unknown pool with `NOT_FOUND`,
and retirement while waiting for admission or lazy hub initialization can return
`UNAVAILABLE` through cancellation. A mismatch detected by the identity check after
initialization still returns `FAILED_PRECONDITION`. Refresh the catalog before retrying
these responses. A subscriber can never silently read a replacement generation's filter.

The stream delivers `FilterUpdate` frames: `SNAPSHOT_CHUNK`s assembling one
consistent CBI1 image, then contiguous `DELTA`s, with `HEARTBEAT`s in between.
Heartbeats start only after the complete initial snapshot and do not advance its sequence.
These payloads come directly from the universal publisher; the gRPC adapter does not re-encode
the CKF or maintain an independent publication hub.
A subscriber that falls behind the bounded queue is terminated with
`RESOURCE_EXHAUSTED` and must resubscribe from scratch.
The initial snapshot producer has an independent per-frame progress timeout. If a client stops
draining the bounded bootstrap queue, the Relay stops that producer and releases encoder admission
at the deadline. The stream yields `RESOURCE_EXHAUSTED` after gRPC resumes polling it; this
client-local timeout does not fence the producer generation. A client that never resumes can retain
one of the bounded total pool-stream slots until its transport disconnects or the Relay shuts down.

### SubscribeServingReadiness

Emits complete `ServingReadinessUpdate` snapshots with a monotonic revision
maintained by the topology projection (independent of catalog revisions).
Entry key: `(namespace, canonical_model_id)`.

`TopologyEntry`:

| Field | Semantics |
| --- | --- |
| `state` | `READY` / `UNAVAILABLE` / `UNKNOWN` — serving readiness in core-frontend semantics (namespace-wide role disjunctive normal form (DNF); see the mental model). |
| `present_roles[]` / `missing_roles[]` | Live vs missing typed roles across the namespace. Cleared when `state = UNKNOWN`. |
| `members[]` | One per participating endpoint: `{endpoint, declared roles, optional pool_id}`. `pool_id` is the **stable** pool link, present only while the pool is materialized; the current generation is resolved through the catalog. |
| `duplicate_role_endpoints[]` | Typed roles declared by more than one endpoint under this key. Only `PREFILL`/`DECODE` are valid values. An observable fact; any "disaggregation degraded" interpretation is version-dependent consumer policy. |
| `legacy_fallback_active` | True when any card without a `worker_type` disabled strict gating (readiness = any live worker). |
| `adapters[]` | Per-LoRA readiness: `{canonical_model_id, state, missing_roles}`. Adapters never appear as top-level entries. |

Validation: at least one member; member roles non-empty; entry/member
namespace consistency; duplicate members, adapters, and roles rejected;
`duplicate_role_endpoints` values outside `PREFILL`/`DECODE` rejected.

### SubscribeKvPoolLoad

Emits complete `KvPoolLoadUpdate` windows (`window_sequence`, `observed_ms`,
`window_ms`); a pool absent from the next window is gone. Per-pool
`KvPoolLoadEntry` carries worker-authoritative KV occupancy
(`kv_used_blocks` / `total_kv_blocks`) and observed/expected rank coverage.
An aggregate is authoritative only when every declared rank has reported and
every rank has a nonzero known capacity. Incomplete usage is encoded with
`observed < expected`; unknown capacity is encoded as zero capacity. Aggregate
sums use the same saturating `u64` arithmetic as the local load aggregator, so
the Relay forwards a saturated value as `u64::MAX`. Neither incomplete coverage
nor zero capacity may be read as zero load.

## Consumer lifecycle rules

| Observation | Required consumer reaction |
| --- | --- |
| New `ProducerIdentity` for a known `PoolId` in the catalog | Drop the CKF replica, resubscribe with the new `expected_producer`. |
| Pool absent from a catalog snapshot | Drop its CKF and load state; the topology member loses its `pool_id`. |
| `TopologyEntry` turns `UNAVAILABLE` | Stop routing to this key even though its pools remain published. |
| `TopologyEntry` `UNKNOWN` | Consumer policy; conservatively skip while READY alternatives exist. |
| Stream lag → `RESOURCE_EXHAUSTED` | Resubscribe from scratch (snapshot + deltas). |
| Marker/version mismatch → `FAILED_PRECONDITION` | Deployment skew; do not retry without upgrading. |

Routing is always gated by the topology plane and matched by the pool plane:
pool presence alone never implies routability, and the two planes may disagree
transiently at revision boundaries.
