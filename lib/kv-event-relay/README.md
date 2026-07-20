# dynamo-kv-event-relay

Per-DC bridge between a datacenter's internal Dynamo event plane and the
cross-DC **global gateway**. One relay runs in each DC; the global gateway
connects to all of them and uses what they report to decide which DC should
serve each request.

The problem it solves: inside a DC, workers publish KV cache events over the
local event plane (ZMQ or NATS) — high volume, per-worker, not reachable from
outside. The global gateway needs a **DC-level** view of that cache, delivered
over the WAN, that survives restarts and missed messages. The relay sits in
between: it folds the per-worker firehose into a deduplicated per-model cuckoo
filter and serves it over a single mTLS gRPC endpoint.

The wire contract (protobuf, payload encodings, cuckoo filter format) lives in
the sibling crate [`kv-event-relay-proto`](../kv-event-relay-proto/README.md);
this crate is the server that implements it. Production topics — memory
sizing, certificates, the full configuration surface, monitoring, known
limitations — are covered in the [operations guide](docs/OPERATIONS.md).

## What the relay serves

One gRPC port, several server-streams (see
[`relay.proto`](../kv-event-relay-proto/proto/relay.proto)):

- **Filters** — a per-model cuckoo filter summarizing the DC's cached blocks,
  shipped as a full snapshot on connect plus small deltas afterwards. This is
  what the global gateway's `SnapshotCuckoo` index backend consumes; a
  reconnect always starts from a fresh snapshot, so no cross-connection
  cursor or replay machinery is needed.
- **Metrics** — routing telemetry (queue depth, TTFT/ITL percentiles, GPU and
  KV cache utilization) collected from the DC's Prometheus via PromQL.

## How it works

```text
 ┌──────────────────────────────────────────────────────────┐
 │ kv-event-relay process                                   │
 │                                                          │
 │ discovery (etcd watch: workers, models)                  │
 │   └─ per-worker subscriber                               │
 │        ├─ bootstrap (authoritative state dump)           │
 │        ├─ event-plane subscription (ZMQ/NATS)            │
 │        └─ DC-wide dedup ──► per-model cuckoo filter      │
 │                                                          │
 │ telemetry: PromQL ──► metrics broadcast                  │
 │                                                          │
 │ gRPC server (TLS + mTLS, single port)                    │
 │   SubscribeFilter / SubscribeMetrics / GetRelayInfo      │
 └──────────────────────────────────────────────────────────┘
```

### Discovery

The relay watches the discovery plane (via `dynamo-runtime`) for two things:
**event channels** (a worker appeared or left — start or drain its
subscriber) and **model cards** (which public model id, e.g.
`Qwen/Qwen3-0.6B`, each worker serves). Everything downstream is keyed by
model, so the router can hold per-model routing state; a worker's events are
never processed until its model card is known.

A broken discovery watch is retried with backoff, and each retry reconciles
against a fresh list so removals lost in the gap are synthesized. A watch
that stays broken for a minute fails the process — a restart resyncs
everything from scratch, which beats serving a frozen view. See the
[operations guide](docs/OPERATIONS.md#process-lifecycle).

### Dedup: many workers, one DC view

Several workers in a DC usually cache the same blocks. The router doesn't care
*which* worker holds a block — only that the DC does. The relay keeps a
DC-wide reference count per block: a `Stored` event is forwarded only for
blocks the DC didn't already hold, and a `Removed` only when the *last* holder
drops a block. This keeps the per-model filter's refcounting exact and the
outbound filter deltas small.

### Correctness under failure

The relay assumes everything around it can fail, and recovers in layers:

- **Cold start / late start.** Events that happened before the relay came up
  are gone from the event plane. On startup (and whenever a new worker
  appears) the relay queries the worker's authoritative KV state and replays
  it through dedup, so the outbound view starts complete rather than empty.
- **Missed events.** Every worker event carries an id; the relay tracks a
  cursor per worker. A gap means events were lost — the relay re-fetches that
  worker's authoritative state and emits a corrective delta rather than
  guessing.
- **Worker departure.** When a worker leaves, blocks only it held are
  synthetically `Removed`, so a dead worker's cache can't keep attracting
  traffic.
- **Router disconnect.** Every `SubscribeFilter` connection starts with a
  full filter snapshot before live deltas, so a reconnecting router heals by
  simply reconnecting — no replay state is kept on either side.
- **Relay restart.** The relay's own state resets silently on restart. Each
  process generates a random instance id, stamped on every stream; a router
  seeing a new id discards its view and resyncs from scratch.

### Filters

For each model, the relay feeds the deduplicated DC-level event stream into an
upstream `DcCkfState`. The relay mirrors the absolute bucket images it
publishes and, on a fixed cadence, emits either a coalesced CBI1 delta, a full
snapshot after a lane reset or oversized delta, or a heartbeat.
The upstream CKF is fixed-capacity; size `--filter-capacity-hint` for the DC's
working set because inserts beyond that capacity are rejected (counted by
`ckf_apply_errors_total` and logged) — sizing guidance and memory cost are in
the [operations guide](docs/OPERATIONS.md#memory-sizing).

### Metrics

Routing telemetry is not measured by the relay itself: it runs PromQL queries
against the DC's Prometheus (any API-compatible store works) on an interval,
maps the results to models, and streams compact per-model snapshots. The query
catalog covers frontend latency histograms, queue depth, DCGM GPU utilization
and KV cache usage. Without `--prometheus-url` the metrics stream is simply
disabled — event and filter streams don't depend on it.

## Running

TLS is mandatory: the relay listens on a single mTLS port and verifies client
certificates against a CA bundle. There is no plaintext mode. Certificate
requirements (SANs, trust model, rotation) are in the
[operations guide](docs/OPERATIONS.md#tls-and-certificates).

Key flags (each has a `DYN_RELAY_*` / `DYN_*` env var equivalent, see
`--help`; the [full reference](docs/OPERATIONS.md#configuration-reference)
covers everything with sizing guidance):

| Flag | Default | Purpose |
|---|---|---|
| `--dc-id` | (required) | This DC's identity, stamped on outbound metrics |
| `--namespaces` | `dynamo` | Dynamo namespaces to relay |
| `--bind` | `0.0.0.0:5560` | gRPC listen address (mTLS) |
| `--tls-server-cert` / `--tls-server-key` / `--tls-client-ca` | (required) | mTLS material |
| `--filter-capacity-hint` | `65536` | Per-model filter capacity in blocks — **must be sized for the DC's KV working set** |
| `--prometheus-url` | unset | PromQL endpoint; unset disables the metrics stream |
| `--prometheus-selector` | unset | Label matcher scoping queries; required on shared metrics backends |
| `--filter-interval-ms` | `1000` | Filter publish cadence |
| `--batch-window-ms` | `1` | Event coalescing window; `0` = frame per event |
| `--metrics-listen` | `0.0.0.0:9090` | The relay's own plaintext Prometheus endpoint |

The Prometheus bearer token (when the metrics backend requires auth) is
passed only via the `DYN_RELAY_PROMETHEUS_BEARER_TOKEN` environment variable,
never as a flag, so it stays out of the process command line.

Minimal invocation:

```bash
dynamo-kv-event-relay \
  --dc-id dc-a \
  --tls-server-cert server.pem --tls-server-key server.key \
  --tls-client-ca clients-ca.pem \
  --prometheus-url http://prometheus:9090
```

## Observability

The relay exposes its own health on a plaintext `/metrics` endpoint
(`dynamo_kv_event_relay_*`): forwarded batches, detected gaps and recovery
outcomes, per-model filter sizes, filter-capacity overflows, TLS expiry,
active subscribers per stream, and stage latencies (dedup, filter publish).
gRPC health and server reflection are served on the main port; health flips
to not-serving on SIGTERM/SIGINT before streams are cut, so load balancers
drain cleanly. The metric-by-metric alerting guide is in the
[operations guide](docs/OPERATIONS.md#monitoring-and-alerting).
