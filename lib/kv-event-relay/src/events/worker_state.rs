// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;

use dynamo_kv_router::{
    indexer::{WorkerKvQueryRequest, WorkerKvQueryResponse},
    protocols::WorkerId,
};
use dynamo_runtime::{
    DistributedRuntime,
    component::Instance,
    pipeline::{PushRouter, RouterMode, SingleIn},
};

/// How long to wait for the PushRouter's discovery watch to surface
/// our target `worker_id`. `endpoint.client()` returns immediately
/// after subscribing to an etcd watch, but the first snapshot of
/// registered instances arrives asynchronously. Without this poll,
/// `router.direct(worker_id)` would frequently fail with
/// "instance_id not found" on startup even though the worker is up.
const BOOTSTRAP_DISCOVERY_WAIT: Duration = Duration::from_secs(5);

const BOOTSTRAP_DISCOVERY_POLL: Duration = Duration::from_millis(50);

/// Exponential-backoff caps total wait around 4s (200, 400, 800,
/// 1600 ms). Covers a worker visible in discovery but mid-TCP
/// handshake.
const BOOTSTRAP_DIRECT_MAX_ATTEMPTS: u32 = 4;
const BOOTSTRAP_DIRECT_INITIAL_BACKOFF: Duration = Duration::from_millis(200);

/// Splits "no instance" (discovery race) from "transport handshake
/// mid-flight" — the latter is what this loop covers; the former is
/// filtered upstream by the `instance_ids()` poll. Bails early on
/// permanent NotFound so we don't burn the full backoff budget on a
/// worker that just doesn't expose this rank.
async fn try_direct_with_retry(
    router: &PushRouter<WorkerKvQueryRequest, WorkerKvQueryResponse>,
    request: WorkerKvQueryRequest,
    route_instance_id: u64,
    worker_id: WorkerId,
    dp_rank: u32,
) -> Result<dynamo_runtime::pipeline::ManyOut<WorkerKvQueryResponse>> {
    let mut backoff = BOOTSTRAP_DIRECT_INITIAL_BACKOFF;
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 0..BOOTSTRAP_DIRECT_MAX_ATTEMPTS {
        match router
            .direct(SingleIn::new(request.clone()), route_instance_id)
            .await
        {
            Ok(stream) => {
                if attempt > 0 {
                    tracing::debug!(
                        worker_id,
                        dp_rank,
                        attempt,
                        "bootstrap: query succeeded after retry"
                    );
                }
                return Ok(stream);
            }
            Err(e) => {
                // FRAGILE: classifies permanent-vs-retryable by matching
                // a dynamo-runtime PushRouter error *string*. A wording change
                // upstream silently flips bootstrap retry semantics. Unclassifiable
                // errors fall through to the retryable path (safe default); replace
                // with a downcast to a typed NotFound once upstream exposes one.
                let msg = e.to_string();
                if msg.contains("not found for endpoint") {
                    return Err(e);
                }
                last_err = Some(e);
                if attempt + 1 < BOOTSTRAP_DIRECT_MAX_ATTEMPTS {
                    tokio::time::sleep(backoff).await;
                    backoff *= 2;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| {
        anyhow::anyhow!("query exhausted {BOOTSTRAP_DIRECT_MAX_ATTEMPTS} attempts")
    }))
}

/// Outcome of a single-slot KV-indexer query: endpoint resolution
/// (worker-specific with legacy fallback), discovery wait, and the RPC
/// itself. Response matching is left to the caller because cold-start
/// bootstrap and gap recovery diverge — the former replays into the
/// dedup silently, the latter forwards a corrective delta.
pub(crate) enum FetchOutcome {
    /// No worker/legacy query endpoint, router setup failed, or the
    /// worker never surfaced in discovery for this dp_rank.
    NoEndpoint,
    /// Endpoint reachable but the query failed (retries exhausted or an
    /// empty response stream).
    QueryFailed,
    /// The worker answered; caller matches the variant.
    Response(WorkerKvQueryResponse),
}

async fn fetch_worker_dp_response(
    drt: &DistributedRuntime,
    target: Option<&Instance>,
    worker_id: WorkerId,
    dp_rank: u32,
) -> FetchOutcome {
    let Some(target) = target else {
        return FetchOutcome::NoEndpoint;
    };
    let namespace = match drt.namespace(&target.namespace) {
        Ok(namespace) => namespace,
        Err(error) => {
            tracing::debug!(worker_id, dp_rank, %error, "query: invalid recovery target namespace");
            return FetchOutcome::NoEndpoint;
        }
    };
    let component = match namespace.component(&target.component) {
        Ok(component) => component,
        Err(error) => {
            tracing::debug!(worker_id, dp_rank, %error, "query: invalid recovery target component");
            return FetchOutcome::NoEndpoint;
        }
    };
    let endpoint_name = target.endpoint.clone();
    let route_instance_id = target.instance_id;
    let endpoint = component.endpoint(&endpoint_name);
    let client = match endpoint.client().await {
        Ok(client) => client,
        Err(error) => {
            tracing::debug!(
                worker_id,
                dp_rank,
                endpoint = %endpoint_name,
                %error,
                "query: advertised recovery endpoint is unavailable"
            );
            return FetchOutcome::NoEndpoint;
        }
    };
    let router = match PushRouter::<WorkerKvQueryRequest, WorkerKvQueryResponse>::
        from_client_no_fault_detection(client, RouterMode::RoundRobin).await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(
                worker_id, dp_rank, endpoint = %endpoint_name, error = %e,
                "query: PushRouter setup failed"
            );
            return FetchOutcome::NoEndpoint;
        }
    };

    // Wait for discovery to surface the advertised recovery target under the
    // endpoint. `router.client.instance_ids()` is updated by an etcd
    // watch racing with our setup; without this poll the first
    // `direct()` call almost always loses.
    let discovery_start = std::time::Instant::now();
    let mut discovered = false;
    loop {
        if router.client.instance_ids().contains(&route_instance_id) {
            discovered = true;
            break;
        }
        if discovery_start.elapsed() >= BOOTSTRAP_DISCOVERY_WAIT {
            break;
        }
        tokio::time::sleep(BOOTSTRAP_DISCOVERY_POLL).await;
    }
    if !discovered {
        tracing::debug!(
            worker_id,
            dp_rank,
            route_instance_id,
            endpoint = %endpoint_name,
            wait_ms = BOOTSTRAP_DISCOVERY_WAIT.as_millis() as u64,
            "query: discovery did not surface worker for this dp_rank"
        );
        return FetchOutcome::NoEndpoint;
    }

    let request = WorkerKvQueryRequest {
        worker_id,
        dp_rank,
        start_event_id: None,
        end_event_id: None,
        supports_tree_dump_failed: true,
    };
    let mut stream = match try_direct_with_retry(
        &router,
        request,
        route_instance_id,
        worker_id,
        dp_rank,
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                worker_id, dp_rank, error = %e,
                "query: failed after retries"
            );
            return FetchOutcome::QueryFailed;
        }
    };
    match stream.next().await {
        Some(r) => FetchOutcome::Response(r),
        None => {
            tracing::warn!(worker_id, dp_rank, "query: empty response stream");
            FetchOutcome::QueryFailed
        }
    }
}

pub(crate) trait WorkerStateSource {
    async fn fetch(&self, worker_id: WorkerId, dp_rank: u32) -> FetchOutcome;
}

pub(crate) struct RuntimeWorkerStateSource<'a> {
    drt: &'a DistributedRuntime,
    target: Option<&'a Instance>,
}

impl<'a> RuntimeWorkerStateSource<'a> {
    pub(crate) fn new(drt: &'a DistributedRuntime, target: Option<&'a Instance>) -> Self {
        Self { drt, target }
    }
}

impl WorkerStateSource for RuntimeWorkerStateSource<'_> {
    async fn fetch(&self, worker_id: WorkerId, dp_rank: u32) -> FetchOutcome {
        fetch_worker_dp_response(self.drt, self.target, worker_id, dp_rank).await
    }
}
