// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Three parallel discovery watchers:
//!
//!   1. [`run_event_source_watcher`] — watches typed KV `EventSource`
//!      advertisements in the relay's namespace and spawns/cancels one
//!      endpoint-scoped subscriber per publisher incarnation.
//!   2. [`run_mdc_watcher`] — watches `NamespacedModels` and maintains
//!      `(component, instance_id) → model_id` in
//!      [`crate::model_registry::ModelRegistry`] so the forward path can
//!      stamp the correct topic on outgoing batches. It also records the
//!      `model_id → namespace` association for frontend-health attribution.
//!   3. [`run_router_discovery_watcher`] — watches endpoints for the
//!      frontend KV router's `router-discovery` registration, tracking
//!      DC-local frontend liveness in [`crate::frontend_health::FrontendHealth`].
//!
//! The underlying discovery watch does not survive an etcd reconnect and does
//! not replay `Removed` events that happened while it was down. Each watcher
//! therefore retries with backoff, and every re-attempt first *reconciles*: it
//! diffs a fresh `discovery.list()` against the state it accumulated and
//! synthesizes the removals the gap swallowed. A watch that stays broken past
//! the failure budget fails the watcher — the relay shuts down and the
//! orchestrator restart resyncs everything from scratch.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use futures::StreamExt;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, WorkerWithDpRank};
use dynamo_runtime::{
    DistributedRuntime,
    component::Instance,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, EventScope,
        EventSourceQuery,
    },
    protocols::EndpointId,
};

use crate::events::recovery::evict_departed_slot;
use crate::events::subscriber::run_source_subscriber;
use crate::events::supervisor::{SUBSCRIBER_SHUTDOWN_GRACE, SubscriberSupervisor};
use crate::state::IngestContext;

/// Endpoint name the frontend's KV router registers for discovery. Source of
/// truth is `dynamo_llm::kv_router::KV_ROUTER_ENDPOINT`; duplicated here as a
/// literal so the relay need not depend on lib/llm for one string.
const ROUTER_DISCOVERY_ENDPOINT: &str = "router-discovery";

const WATCH_BACKOFF_INITIAL: Duration = Duration::from_millis(250);
const WATCH_BACKOFF_MAX: Duration = Duration::from_secs(5);
/// Consecutive-failure window after which a watcher gives up: transient etcd
/// blips are retried; a persistent outage is surfaced so the process restarts
/// into a clean resync instead of serving frozen state.
const WATCH_FAILURE_BUDGET: Duration = Duration::from_secs(60);

/// Consecutive-failure tracker for one watch loop. An attempt that ran at
/// least as long as the budget counts as healthy and resets the window.
struct WatchRetry {
    backoff: Duration,
    failing_since: Option<Instant>,
}

impl WatchRetry {
    fn new() -> Self {
        Self {
            backoff: WATCH_BACKOFF_INITIAL,
            failing_since: None,
        }
    }

    /// Back off before the next attempt. `false` when the failure budget is
    /// exhausted. Cancellation cuts the sleep short; the caller re-checks the
    /// token before retrying.
    async fn pause(&mut self, ran_for: Duration, cancel: &CancellationToken) -> bool {
        if ran_for >= WATCH_FAILURE_BUDGET {
            self.backoff = WATCH_BACKOFF_INITIAL;
            self.failing_since = None;
        }
        let since = *self.failing_since.get_or_insert_with(Instant::now);
        if since.elapsed() >= WATCH_FAILURE_BUDGET {
            return false;
        }
        tokio::select! {
            _ = cancel.cancelled() => {}
            _ = tokio::time::sleep(self.backoff) => {}
        }
        self.backoff = (self.backoff * 2).min(WATCH_BACKOFF_MAX);
        true
    }
}

/// Interpret one finished watch attempt: `Ok` under cancellation is a clean
/// exit; anything else (error, or the stream just ending) is a failure to
/// retry.
fn watch_failure(outcome: Result<()>) -> anyhow::Error {
    match outcome {
        Ok(()) => anyhow::anyhow!("watch stream ended"),
        Err(error) => error,
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct KvSourceAdvertisement {
    kv_state_endpoint: EndpointId,
    worker: WorkerWithDpRank,
    publisher_id: u64,
    #[serde(default)]
    recovery_target: Option<Instance>,
}

pub async fn run_event_source_watcher(
    drt: DistributedRuntime,
    namespace: String,
    state: Arc<IngestContext>,
    cancel: CancellationToken,
) -> Result<()> {
    let mut supervisor = SubscriberSupervisor::new(cancel.clone());
    let mut known = HashMap::new();
    let mut retry = WatchRetry::new();
    let mut reconcile = false;
    let result = loop {
        let attempt_started = Instant::now();
        let outcome = async {
            if reconcile {
                reconcile_event_sources(&drt, &namespace, &state, &mut supervisor, &mut known)
                    .await?;
            }
            watch_event_sources(
                &drt,
                &namespace,
                &state,
                &cancel,
                &mut supervisor,
                &mut known,
            )
            .await
        }
        .await;
        if cancel.is_cancelled() {
            break Ok(());
        }
        let error = watch_failure(outcome);
        reconcile = true;
        tracing::warn!(namespace = %namespace, error = %format!("{error:#}"), "EventSource watch interrupted; retrying");
        if !retry.pause(attempt_started.elapsed(), &cancel).await {
            break Err(error.context(format!(
                "EventSource watch ({namespace}): failure budget exhausted"
            )));
        }
    };
    // Graceful drain: cancel and join every live subscriber before returning,
    // so a shutting-down relay doesn't leave detached ingest tasks behind.
    supervisor.shutdown_all(SUBSCRIBER_SHUTDOWN_GRACE).await;
    result
}

async fn watch_event_sources(
    drt: &DistributedRuntime,
    namespace: &str,
    state: &Arc<IngestContext>,
    cancel: &CancellationToken,
    supervisor: &mut SubscriberSupervisor,
    known: &mut HashMap<u64, KvSourceAdvertisement>,
) -> Result<()> {
    let mut stream = drt
        .discovery()
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::all()),
            Some(cancel.clone()),
        )
        .await?;

    while let Some(event) = stream.next().await {
        if cancel.is_cancelled() {
            break;
        }
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "EventSource watch yielded error");
                continue;
            }
        };
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::EventSource {
                scope,
                topic,
                publisher_id,
                metadata,
            }) => {
                let Some(source) = decode_source(namespace, scope, topic, publisher_id, metadata)
                else {
                    continue;
                };
                if let Some(existing) = known.get(&publisher_id)
                    && existing != &source
                {
                    tracing::warn!(
                        publisher_id,
                        "KV source changed immutable attribution; ignoring replacement"
                    );
                    continue;
                }
                known.insert(publisher_id, source.clone());

                let component = source.kv_state_endpoint.component.clone();
                let key = (component.clone(), publisher_id);
                let drt_c = drt.clone();
                let endpoint_c = source.kv_state_endpoint.clone();
                let worker_c = source.worker;
                let recovery_target_c = source.recovery_target.clone();
                let comp_c = component.clone();
                let state_c = state.clone();
                supervisor.spawn(key, move |task_cancel| async move {
                    if let Err(e) = run_source_subscriber(
                        drt_c,
                        endpoint_c,
                        worker_c,
                        recovery_target_c,
                        state_c,
                        task_cancel,
                    )
                    .await
                    {
                        tracing::warn!(component = %comp_c, publisher_id, error = %e, "KV source subscriber exited");
                    }
                });
            }
            DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(id)) => {
                if id.topic != KV_EVENT_SUBJECT || id.scope.namespace() != namespace {
                    continue;
                }
                let Some(source) = known.remove(&id.publisher_id) else {
                    continue;
                };
                drain_departed_source(supervisor, state, source);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Cancel + join a departed worker's subscriber, then evict its blocks and
/// forward the corrective `Removed` downstream — its events will never carry
/// their own.
fn drain_departed_source(
    supervisor: &mut SubscriberSupervisor,
    state: &Arc<IngestContext>,
    source: KvSourceAdvertisement,
) {
    let component = source.kv_state_endpoint.component.clone();
    let key = (component.clone(), source.publisher_id);
    let state = state.clone();
    let worker = source.worker;
    supervisor.begin_drain(key, SUBSCRIBER_SHUTDOWN_GRACE, move || async move {
        evict_departed_slot(&state, &component, worker.worker_id, worker.dp_rank).await;
    });
}

/// Diff the supervised subscriber set against a fresh discovery list and drain
/// the workers whose `Removed` fell into the watch gap.
async fn reconcile_event_sources(
    drt: &DistributedRuntime,
    namespace: &str,
    state: &Arc<IngestContext>,
    supervisor: &mut SubscriberSupervisor,
    known: &mut HashMap<u64, KvSourceAdvertisement>,
) -> Result<()> {
    let live: HashMap<u64, KvSourceAdvertisement> = drt
        .discovery()
        .list(DiscoveryQuery::EventSources(EventSourceQuery::all()))
        .await
        .context("listing event sources for reconciliation")?
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::EventSource {
                scope,
                topic,
                publisher_id,
                metadata,
            } => decode_source(namespace, scope, topic, publisher_id, metadata)
                .map(|source| (publisher_id, source)),
            _ => None,
        })
        .collect();
    for key in supervisor.active_keys() {
        if !live.contains_key(&key.1) {
            tracing::info!(
                component = %key.0,
                publisher_id = key.1,
                "KV source departed during a watch gap; draining its subscriber"
            );
            if let Some(source) = known.remove(&key.1) {
                drain_departed_source(supervisor, state, source);
            }
        }
    }
    Ok(())
}

fn decode_source(
    namespace: &str,
    scope: EventScope,
    topic: String,
    publisher_id: u64,
    metadata: serde_json::Value,
) -> Option<KvSourceAdvertisement> {
    if topic != KV_EVENT_SUBJECT || scope.namespace() != namespace {
        return None;
    }
    let source: KvSourceAdvertisement = match serde_json::from_value(metadata) {
        Ok(source) => source,
        Err(error) => {
            tracing::warn!(publisher_id, %error, "ignoring malformed KV source advertisement");
            return None;
        }
    };
    let expected_scope = EventScope::Endpoint {
        endpoint: source.kv_state_endpoint.clone(),
    };
    if scope != expected_scope || source.publisher_id != publisher_id {
        tracing::warn!(
            publisher_id,
            "ignoring inconsistently attributed KV source advertisement"
        );
        return None;
    }
    Some(source)
}

pub async fn run_mdc_watcher(
    drt: DistributedRuntime,
    namespace: String,
    state: Arc<IngestContext>,
    cancel: CancellationToken,
) -> Result<()> {
    let mut retry = WatchRetry::new();
    // (component, instance) pairs this watcher registered. The registry is
    // shared across namespace watchers, so gap reconciliation may prune only
    // what this watcher has seen.
    let mut known: HashSet<(String, u64)> = HashSet::new();
    let mut reconcile = false;
    loop {
        let attempt_started = Instant::now();
        let outcome = async {
            if reconcile {
                reconcile_models(&drt, &namespace, &state, &mut known).await?;
            }
            watch_models(&drt, &namespace, &state, &cancel, &mut known).await
        }
        .await;
        if cancel.is_cancelled() {
            return Ok(());
        }
        let error = watch_failure(outcome);
        reconcile = true;
        tracing::warn!(namespace = %namespace, error = %format!("{error:#}"), "MDC watch interrupted; retrying");
        if !retry.pause(attempt_started.elapsed(), &cancel).await {
            return Err(error.context(format!("MDC watch ({namespace}): failure budget exhausted")));
        }
    }
}

async fn watch_models(
    drt: &DistributedRuntime,
    namespace: &str,
    state: &Arc<IngestContext>,
    cancel: &CancellationToken,
    known: &mut HashSet<(String, u64)>,
) -> Result<()> {
    let mut stream = drt
        .discovery()
        .list_and_watch(
            DiscoveryQuery::NamespacedModels {
                namespace: namespace.to_string(),
            },
            Some(cancel.clone()),
        )
        .await?;

    while let Some(event) = stream.next().await {
        if cancel.is_cancelled() {
            break;
        }
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "MDC watch yielded error");
                continue;
            }
        };
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Model {
                component,
                instance_id,
                card_json,
                ..
            }) => {
                // MDC `display_name` is what `dynamo.{vllm,sglang}`
                // registers as the public model id (e.g.
                // `Qwen/Qwen3-0.6B`).
                if let Some(display) = card_json.get("display_name").and_then(|v| v.as_str()) {
                    known.insert((component.clone(), instance_id));
                    state
                        .models
                        .register(component.clone(), instance_id, display.to_string())
                        .await;
                    // The frontend that fronts this model lives in the same
                    // Dynamo namespace; record the link so its router-discovery
                    // liveness can be attributed to the model.
                    state.frontend_health.associate(display, namespace).await;
                } else {
                    // An MDC with no readable `display_name`
                    // leaves the model unregistered, so every subscriber for
                    // this component blocks forever in `wait_registered` and
                    // ingests nothing — the exact shape of the relay↔worker
                    // CR-schema-skew incident. Make it loud instead of silent.
                    tracing::warn!(
                        component = %component,
                        instance_id,
                        "MDC has no readable display_name; model left unregistered \
                         (subscribers will block on wait_registered) — likely a \
                         worker CR schema mismatch"
                    );
                }
                // The MDC also carries the worker's KV block size — the DC-wide
                // consensus is reported to the global gateway via `GetRelayInfo`
                // so it can refuse a topology/DC block-size mismatch.
                if let Some(block_size) = card_json
                    .get("kv_cache_block_size")
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok())
                {
                    state
                        .block_size
                        .observe(&component, instance_id, block_size);
                }
            }
            // Without this, departed workers' entries pile up forever AND keep
            // advertising their model to the global gateway via the metrics
            // publisher (it derives its worker→model map from this map).
            DiscoveryEvent::Removed(DiscoveryInstanceId::Model(id)) => {
                known.remove(&(id.component.clone(), id.instance_id));
                state.models.remove(&id.component, id.instance_id).await;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Prune registry entries whose `Removed` fell into the watch gap. Frontend
/// associations and the block size are sticky by design and not reconciled.
async fn reconcile_models(
    drt: &DistributedRuntime,
    namespace: &str,
    state: &Arc<IngestContext>,
    known: &mut HashSet<(String, u64)>,
) -> Result<()> {
    let live: HashSet<(String, u64)> = drt
        .discovery()
        .list(DiscoveryQuery::NamespacedModels {
            namespace: namespace.to_string(),
        })
        .await
        .context("listing models for reconciliation")?
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::Model {
                component,
                instance_id,
                ..
            } => Some((component, instance_id)),
            _ => None,
        })
        .collect();
    let pruned = state.models.retain_instances(known, &live).await;
    if pruned > 0 {
        tracing::info!(
            namespace,
            pruned,
            "pruned model registrations lost in a watch gap"
        );
    }
    known.retain(|key| live.contains(key));
    Ok(())
}

/// Watch endpoint registrations in `namespace` and track the frontend KV
/// router's `router-discovery` endpoint as DC-local frontend liveness.
///
/// The frontend registers this endpoint in a `DynamoWorkerMetadata` CR owned
/// by its pod, so it vanishes on pod death — worker discovery cannot see the
/// frontend, this is the only DC-local signal for it. Unlike the event-source
/// watcher this spawns nothing: it just flips presence in
/// [`crate::frontend_health::FrontendHealth`], which the metrics publisher
/// joins to per-model `frontend_healthy`.
pub async fn run_router_discovery_watcher(
    drt: DistributedRuntime,
    namespace: String,
    state: Arc<IngestContext>,
    cancel: CancellationToken,
) -> Result<()> {
    let mut retry = WatchRetry::new();
    let mut reconcile = false;
    loop {
        let attempt_started = Instant::now();
        let outcome = async {
            if reconcile {
                reconcile_router_discovery(&drt, &namespace, &state).await?;
            }
            watch_router_discovery(&drt, &namespace, &state, &cancel).await
        }
        .await;
        if cancel.is_cancelled() {
            return Ok(());
        }
        let error = watch_failure(outcome);
        reconcile = true;
        tracing::warn!(namespace = %namespace, error = %format!("{error:#}"), "router-discovery watch interrupted; retrying");
        if !retry.pause(attempt_started.elapsed(), &cancel).await {
            return Err(error.context(format!(
                "router-discovery watch ({namespace}): failure budget exhausted"
            )));
        }
    }
}

async fn watch_router_discovery(
    drt: &DistributedRuntime,
    namespace: &str,
    state: &Arc<IngestContext>,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut stream = drt
        .discovery()
        .list_and_watch(
            DiscoveryQuery::NamespacedEndpoints {
                namespace: namespace.to_string(),
            },
            Some(cancel.clone()),
        )
        .await?;

    while let Some(event) = stream.next().await {
        if cancel.is_cancelled() {
            break;
        }
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "router-discovery watch yielded error");
                continue;
            }
        };
        match event {
            DiscoveryEvent::Added(DiscoveryInstance::Endpoint(inst))
                if inst.endpoint == ROUTER_DISCOVERY_ENDPOINT =>
            {
                state
                    .frontend_health
                    .add_router(&inst.namespace, inst.instance_id)
                    .await;
            }
            DiscoveryEvent::Removed(DiscoveryInstanceId::Endpoint(id))
                if id.endpoint == ROUTER_DISCOVERY_ENDPOINT =>
            {
                state
                    .frontend_health
                    .remove_router(&id.namespace, id.instance_id)
                    .await;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Prune frontend instances whose `Removed` fell into the watch gap — a lost
/// removal would report a dead frontend as healthy forever.
async fn reconcile_router_discovery(
    drt: &DistributedRuntime,
    namespace: &str,
    state: &Arc<IngestContext>,
) -> Result<()> {
    let live: HashSet<u64> = drt
        .discovery()
        .list(DiscoveryQuery::NamespacedEndpoints {
            namespace: namespace.to_string(),
        })
        .await
        .context("listing endpoints for reconciliation")?
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::Endpoint(inst) if inst.endpoint == ROUTER_DISCOVERY_ENDPOINT => {
                Some(inst.instance_id)
            }
            _ => None,
        })
        .collect();
    let pruned = state.frontend_health.retain_routers(namespace, &live).await;
    if pruned > 0 {
        tracing::info!(
            namespace,
            pruned,
            "pruned frontend instances lost in a watch gap"
        );
    }
    Ok(())
}
