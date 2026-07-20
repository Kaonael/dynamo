<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# kv-event-relay operations guide

Production reference for running `dynamo-kv-event-relay`: process lifecycle,
memory sizing, TLS material, the full configuration surface, monitoring, and
known limitations. For what the relay *is*, start with the
[README](../README.md); for the wire contract, see
[`kv-event-relay-proto`](../../kv-event-relay-proto/README.md).

## Process lifecycle

The relay runs on the standard `dynamo_runtime::Worker` harness:

- **Shutdown**: SIGINT or SIGTERM triggers a graceful shutdown — gRPC health
  flips to `NOT_SERVING` *before* the server stops accepting streams (so
  Kubernetes removes the pod from Endpoints first), live streams end, watchers
  drain their subscribers, and the process exits. If the application does not
  finish within `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` seconds (default 30 in
  release builds) it is force-terminated with exit code 911.
- **Fail-fast on discovery loss**: each discovery watcher retries a broken
  watch with backoff (250 ms doubling to 5 s). If a watch stays broken for
  **60 s of consecutive failures**, the watcher gives up and the relay shuts
  down with a nonzero exit code. This is deliberate: a restart resyncs all
  state from scratch (worker dumps + a fresh gRPC snapshot to every
  consumer), which is strictly safer than serving a frozen view. Expect
  `CrashLoopBackOff` during a full etcd outage — that is the designed
  behavior, not a relay bug.
- **Watch-gap reconciliation**: when a watch reconnects *within* the failure
  budget, the relay diffs a fresh discovery list against its accumulated
  state and synthesizes the removals the gap swallowed (departed workers are
  drained and their blocks evicted; stale model registrations pruned). The
  small window between the list and the new watch is not covered — a removal
  landing exactly there heals on the next reconcile or restart.
- **Restart cost**: every start generates a fresh 128-bit `instance_id`.
  Consumers (the global gateway) treat a changed id as "full resync
  required" and rebuild their per-model filters from the snapshot the new
  stream prepends. The cost is one full snapshot per model per consumer —
  see [Memory](#memory-sizing) for snapshot sizes.

A Kubernetes probe setup that matches these semantics:

```yaml
ports:
  - name: grpc
    containerPort: 5560
  - name: metrics
    containerPort: 9090
livenessProbe:
  grpc: { port: 5560 }        # standard grpc.health.v1.Health, k8s >= 1.24
readinessProbe:
  grpc: { port: 5560 }
```

## Memory sizing

Relay memory has three independent components. All figures below are
estimates for capacity planning — validate against RSS and the
`dynamo_kv_event_relay_filter_blocks` gauge under real load.

### 1. Per-model cuckoo filter — driven by `--filter-capacity-hint`

For a capacity hint of `C` blocks, the filter allocates
`bucket_count = next_power_of_two(max(2, (5·C + 15) / 16))` buckets — that is
`C/4` buckets (4 slots per bucket) with 25 % headroom (80 % target load),
rounded up to a power of two. Because of the rounding, the effective load
factor at capacity varies between 40 % and 80 %.

Per model, the relay holds the published-lane mirror plus the authoritative
pipeline (a replica of the same bucket array and a per-block contribution
map at roughly 40 B per resident block):

| `--filter-capacity-hint` | buckets | mirror + replica | contributions | ≈ total per model |
|---:|---:|---:|---:|---:|
| 65 536 (default) | 32 768 | 0.5 MiB | ~2.6 MiB | **~3 MiB** |
| 1 000 000 | 524 288 | 8 MiB | ~40 MiB | **~50 MiB** |
| 10 000 000 | 4 194 304 | 64 MiB | ~400 MiB | **~0.5 GiB** |

This allocation happens when the model's first event arrives and is **never
freed until restart** — a model that leaves the DC keeps its filter resident
(see [Known limitations](#known-limitations)). Multiply by the number of
models the DC serves.

### 2. DC-wide dedup — driven by actual resident blocks, not the hint

The dedup layer tracks every block every worker holds:

- **per-holder sets**: ~16 B × Σ (resident blocks per worker) — a block
  cached by N workers is counted N times;
- **refcounts**: ~40 B × unique (model, block) pairs in the DC.

Example: 8 workers × 1 M resident blocks each, 50 % overlap →
8 M holder entries (~128 MiB) + ~4 M refcounts (~160 MiB) ≈ **~300 MiB**.
This scales with the DC's real KV footprint and dominates at large scale.

### 3. Transport

- A full snapshot publish or a fresh `SubscribeFilter` connect materializes
  one extra copy of the lane words (`bucket_count × 8 B`), shared across all
  subscribers via `Arc`.
- Snapshots ship in bounded CBI1 chunks of ≤ 4 MiB payload; `--max-msg-bytes`
  (default 64 MiB) caps the encoded gRPC frame.
- The internal broadcast buffers up to 1024 frames. Deltas are typically
  KB-sized; the theoretical worst case (1024 consecutive maximum deltas) is
  4 GiB, but a subscriber that lags behind the buffer is disconnected long
  before that and re-syncs from a snapshot.

**Rule of thumb**:
`RSS ≈ Σ_models filter(C) + dedup(Σ resident blocks) + ~50 MiB base`.
For a small DC (few models, default hint) request 512 MiB; for a 10 M-block
working set budget several GiB and set limits with ~30 % headroom over the
formula.

## TLS and certificates

mTLS is mandatory; there is no plaintext mode. Three PEM files are required:

| Flag | Contents |
|---|---|
| `--tls-server-cert` | Server certificate the relay presents; leaf first, then intermediates |
| `--tls-server-key` | Matching private key (PKCS#8 or RSA PEM) |
| `--tls-client-ca` | CA bundle used to verify connecting clients; may contain several CAs |

Operational facts to plan around:

- **SANs**: the server certificate must cover the DNS name or IP the global
  gateway dials. IP SANs work.
- **Trust model**: there is no authorization beyond the client CA. *Any*
  certificate signed by a CA in the bundle can read every filter and metric
  this DC publishes. Use a CA dedicated to the relay↔gateway link, not an
  organization-wide one.
- **No revocation**: CRL/OCSP are not checked (rustls). Compromised client
  certificates can only be fenced by rotating the CA — prefer short-lived
  certificates.
- **Rotation = restart**: TLS files are read once at startup; there is no hot
  reload. Rotating means restarting the pod, which mints a new `instance_id`
  and triggers a full consumer resync (bounded by snapshot size × models).
  Do rolling rotations in a low-traffic window.
- **Expiry is exported**:
  `dynamo_kv_event_relay_tls_expiry_timestamp_seconds{material="server_cert"|"client_ca"}`
  carries the earliest `notAfter` of each loaded bundle. Alert at least two
  weeks out (see [Monitoring](#monitoring-and-alerting)).
- **Protocol versions**: both ends negotiate TLS 1.3 in practice, but TLS 1.2
  support is still compiled in through transitive dependencies and is not
  rejected at the protocol level. Do not present this link as "TLS 1.3 only"
  in a security review.

With cert-manager, a minimal server certificate:

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: kv-event-relay
spec:
  secretName: kv-event-relay-tls
  duration: 720h        # short-lived; rotation requires a pod restart
  renewBefore: 240h
  dnsNames: ["relay.dc-a.example.internal"]
  issuerRef: { name: relay-link-ca, kind: Issuer }
```

## Configuration reference

Every flag has an environment equivalent (shown in `--help`). Guidance below
is *when to change it*, not just what it is.

### Identity and listeners

| Flag (env) | Default | Guidance |
|---|---|---|
| `--dc-id` (`DYN_DC_ID`) | required | Must match the DC id in the global gateway's topology. |
| `--namespaces` (`DYN_RELAY_NAMESPACES`) | `dynamo` | Comma-separated Dynamo namespaces to ingest. All pools publish through one stream stamped with `--dc-id`; models differentiate pools. |
| `--bind` (`DYN_RELAY_BIND`) | `0.0.0.0:5560` | mTLS gRPC. Expose to the WAN/gateway only. |
| `--metrics-listen` (`DYN_RELAY_METRICS_LISTEN`) | `0.0.0.0:9090` | Plaintext, unauthenticated `/metrics`. Keep firewalled to the monitoring network; header reads time out after 5 s. |

### Filter pipeline

| Flag | Default | Guidance |
|---|---|---|
| `--filter-capacity-hint` | `65536` | **The one flag you must size.** The filter is fixed-size and never resized; inserts past capacity are *rejected* — those blocks silently vanish from routing. Size from `sum(dynamo_frontend_model_total_kv_blocks)` for the model across the DC's workers, ×1.2 headroom. Overflow is visible as a nonzero `ckf_apply_errors_total` and a plateau in `filter_blocks` under growing load. Memory cost: see [Memory](#memory-sizing). |
| `--filter-interval-ms` | `1000` (min 50) | Publish cadence: freshness vs WAN bandwidth. Deltas are cheap; lowering below ~250 ms mostly buys latency on cache-churn visibility. |
| `--batch-window-ms` | `1` | Event coalescing before the filter fold. `0` = one fold per upstream event (higher overhead, minimal latency win). |
| `--batch-max-events` | `256` | Early-flush bound when batching; caps added latency under burst. |
| `--max-msg-bytes` | `67108864` | Encoding cap; must clear the 4 MiB CBI1 chunk + envelope (validated at startup). **The global gateway must accept at least this on decode** — change the two together. |

### Telemetry (PromQL)

| Flag | Default | Guidance |
|---|---|---|
| `--prometheus-url` | unset | PromQL API root of the DC's Prometheus (any compatible store). Unset ⇒ metrics snapshots carry discovery-fed readiness only. |
| `--prometheus-selector` | unset | Extra label matcher (e.g. `namespace="dc-ams"`) scoping every query. **Required whenever the metrics backend holds more than this DC** (shared VictoriaMetrics, several per-namespace DCs in one cluster) — without it, same-model series from other DCs corrupt each other's snapshots. |
| `--metrics-interval-ms` | `1000` (min 100) | Snapshot cadence; also the per-query HTTP timeout (min 2 s). Queries run concurrently — one tick costs one round-trip. |
| `DYN_RELAY_PROMETHEUS_BEARER_TOKEN` | unset | **Environment-only** (no CLI flag) so the secret never appears in `/proc/*/cmdline`. |

### gRPC keepalive

| Flag | Default | Guidance |
|---|---|---|
| `--grpc-keepalive-interval-ms` | `20000` | HTTP/2 ping cadence; reaps half-open WAN connections. |
| `--grpc-keepalive-timeout-ms` | `10000` | Raise the pair on very-high-RTT or lossy links; lower to detect dead gateways faster. |

### Runtime environment (inherited from `dynamo-runtime`)

| Variable | Meaning |
|---|---|
| `DYN_DISCOVERY_BACKEND` | `etcd` (default), `kubernetes`, `file`, `mem`. |
| `DYN_KUBE_WATCH_NAMESPACES` | With the `kubernetes` backend: comma-separated k8s namespaces to watch. Required when the relay runs in a central namespace overseeing per-tenant inference namespaces. |
| `DYN_LOG` / `RUST_LOG` | Log filter (`info` default). |
| `DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` | Seconds before a hung shutdown is force-terminated (30 in release builds). |

### Fixed behavior (not configurable)

These are compile-time constants; they define the contract operators observe:

- Discovery watch retry: backoff 250 ms → 5 s, failure budget 60 s, then
  process exit (see [Lifecycle](#process-lifecycle)).
- Subscriber drain grace on worker departure or shutdown: 5 s, then abort.
- Internal broadcast depth: 1024 frames; a slower consumer is disconnected
  (`subscriber_lagged_total`) and re-syncs via reconnect.
- Each KV `EventSource` identifies one `(worker_id, dp_rank)` and may advertise
  its native `recovery_target`. The relay bootstraps exactly that source slot;
  sources without a recovery target continue in live-only mode.

## Monitoring and alerting

All series are prefixed `dynamo_kv_event_relay_`. The ones that should page:

| Metric | Alert on | Meaning |
|---|---|---|
| `ckf_apply_errors_total` | any increase | **Filter capacity exhausted** — blocks are missing from routing. Raise `--filter-capacity-hint`. |
| `tls_expiry_timestamp_seconds{material}` | `< time() + 14d` | TLS material expiring; rotation requires a restart, schedule it. |
| `gap_recovery_total{result="failed"}` | sustained rate | Workers unreachable for re-query after event loss; their filter view degrades until natural healing. |
| process restarts | CrashLoop | Watch failure budget exhausted — check etcd/discovery health. |

Worth dashboards and warning thresholds:

| Metric | Watch for |
|---|---|
| `filter_blocks{model_id}` | Divergence from the gateway's `dc_index_blocks` for the same (dc, model) = silent filter drift; plateau under growing load = capacity overflow. |
| `gap_detected_total` | Event-plane loss rate (ZMQ HWM drops, NATS hiccups). |
| `subscriber_lagged_total{channel}` | Consumers slower than the publish rate; each hit forces a snapshot re-sync. |
| `bootstrap_rank_total{result="failed"}` | Partial cold starts — a worker joined with an incomplete replay. |
| `events_dropped_unresolved_total` | Nonzero outside model-teardown windows indicates MDC/discovery trouble. |
| `active_subscribers{channel}` | 0 while the gateway should be connected = link/TLS trouble. |
| `dedup_seconds`, `filter_publish_seconds`, `filter_update_bytes` | Stage latency and WAN volume baselines. |

## Known limitations

Documented behavior, accepted for the current design:

- **Per-model filters are never freed.** A model whose pool leaves the DC
  keeps its filter allocated and heart-beating until the relay restarts.
  Bounded by the DC's historical model catalog.
- **Readiness snapshots are sticky.** A model that ever existed keeps
  reporting `worker_count=0` snapshots (deliberately — silence reads as
  healthy downstream) until restart.
- **Reconciliation window.** A worker removal landing exactly between the
  post-gap discovery list and the new watch is missed until the next
  reconcile or restart.
- **`Cleared` events serialize ingest.** A cache-clear takes all dedup shards
  and scans the model's state; engines that emit `Cleared` frequently will
  bottleneck the ingest path.
- **dp_rank cap.** Cold-start replay covers dp ranks 0–7 only.
- **Rotation = restart = resync.** No TLS hot-reload; see
  [TLS](#tls-and-certificates).
