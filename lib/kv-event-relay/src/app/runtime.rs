// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The running relay: spawns the watchers, publishers, and metrics sidecar,
//! serves the mTLS gRPC endpoint, and tears everything down on shutdown.
//!
//! Shutdown is driven by the runtime's cancellation token: the
//! [`Worker`](dynamo_runtime::Worker) signal handler (SIGINT/SIGTERM) cancels
//! it, and a watcher that exhausts its failure budget cancels it too — in
//! that case [`RelayApp::run`] returns the watcher's error so the process
//! exits nonzero and the orchestrator restarts it into a clean resync.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use futures::future::BoxFuture;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tonic::codec::CompressionEncoding;
use tonic::transport::{Server, ServerTlsConfig};

use dynamo_kv_event_relay_proto::{FILE_DESCRIPTOR_SET, KvEventRelayServer, MetricsSnapshot};
use dynamo_runtime::DistributedRuntime;

use crate::discovery::{run_event_source_watcher, run_mdc_watcher, run_router_discovery_watcher};
use crate::filter::{FilterRegistry, run_filter_publisher};
use crate::grpc_server::KvEventRelayService;
use crate::observability::{RelayMetrics, serve_metrics};
use crate::state::IngestContext;
use crate::telemetry::{PromQlClient, QueryCatalog, run_metrics_publisher};

use super::config::RelayConfig;

/// A fully-wired relay, ready to [`run`](RelayApp::run). Holds the ingest
/// context shared by the watchers and the transport-side handles the gRPC
/// service / publishers / metrics sidecar read directly.
pub struct RelayApp {
    pub(super) config: RelayConfig,
    pub(super) drt: DistributedRuntime,
    pub(super) instance_id: bytes::Bytes,
    pub(super) filters: Arc<FilterRegistry>,
    pub(super) metrics_tx: broadcast::Sender<MetricsSnapshot>,
    pub(super) relay_metrics: Arc<RelayMetrics>,
    pub(super) ingest: Arc<IngestContext>,
    pub(super) tls_config: ServerTlsConfig,
}

/// Run one discovery watcher to completion. A watcher error is fatal for the
/// whole relay: cancel the shared token so everything drains, and surface the
/// error to `run()` for a nonzero exit.
fn spawn_watcher(
    name: &'static str,
    namespace: &str,
    cancel: &CancellationToken,
    watcher: BoxFuture<'static, Result<()>>,
) -> tokio::task::JoinHandle<Result<()>> {
    let cancel = cancel.clone();
    let namespace = namespace.to_string();
    tokio::spawn(async move {
        let result = watcher
            .await
            .with_context(|| format!("{name} watcher ({namespace})"));
        if let Err(error) = &result {
            tracing::error!(
                watcher = name,
                namespace = %namespace,
                error = %format!("{error:#}"),
                "watcher failed; shutting the relay down"
            );
            cancel.cancel();
        }
        result
    })
}

impl RelayApp {
    /// Run until the runtime cancellation token fires (signal handler or a
    /// failed watcher) or the gRPC server hits a fatal error.
    pub async fn run(self) -> Result<()> {
        let RelayApp {
            config,
            drt,
            instance_id,
            filters,
            metrics_tx,
            relay_metrics,
            ingest,
            tls_config,
        } = self;

        tracing::info!(
            dc_id = %config.dc_id,
            namespaces = ?config.namespaces,
            bind = %config.bind,
            "starting kv-event-relay (gRPC + mTLS)"
        );

        // Child of the runtime's primary token: the Worker signal handler
        // (SIGINT/SIGTERM) cancels the parent; watcher failures cancel this
        // child directly.
        let cancel = drt.child_token();

        // Prometheus `/metrics` sidecar (plaintext, separate port). Runs on the
        // main tokio runtime; aborted on shutdown alongside the watchers.
        let metrics_task = {
            let addr = config.metrics_listen;
            let filters = filters.clone();
            let metrics = (*relay_metrics).clone();
            tokio::spawn(async move { serve_metrics(addr, filters, metrics).await })
        };

        // One MDC + event-source + router-discovery watcher per Dynamo
        // namespace, all writing into the same ingest context. From the global
        // gateway's perspective they all live in a single logical DC stamped by
        // `--dc-id`; model_id differentiates pools.
        let mut watcher_tasks = Vec::with_capacity(config.namespaces.len() * 3);
        for ns in &config.namespaces {
            watcher_tasks.push(spawn_watcher(
                "mdc",
                ns,
                &cancel,
                Box::pin(run_mdc_watcher(
                    drt.clone(),
                    ns.clone(),
                    ingest.clone(),
                    cancel.clone(),
                )),
            ));
            watcher_tasks.push(spawn_watcher(
                "event-source",
                ns,
                &cancel,
                Box::pin(run_event_source_watcher(
                    drt.clone(),
                    ns.clone(),
                    ingest.clone(),
                    cancel.clone(),
                )),
            ));
            watcher_tasks.push(spawn_watcher(
                "router-discovery",
                ns,
                &cancel,
                Box::pin(run_router_discovery_watcher(
                    drt.clone(),
                    ns.clone(),
                    ingest.clone(),
                    cancel.clone(),
                )),
            ));
        }

        // Metrics publisher: merges the PromQL catalog (when `--prometheus-url`
        // is set) with discovery-fed worker counts and frontend liveness.
        // Runs even without Prometheus — readiness must keep flowing when the
        // metrics stack is down; that is exactly the failure it reports.
        let interval = std::time::Duration::from_millis(config.metrics_interval_ms.max(100));
        let source = match config.prometheus_url.clone() {
            Some(url) => Some(
                PromQlClient::new(url, config.prometheus_bearer_token.clone(), interval)
                    .context("building PromQL client")?,
            ),
            None => {
                tracing::warn!("no --prometheus-url; metrics snapshots carry readiness only");
                None
            }
        };
        let metrics_pub_task = {
            let catalog = QueryCatalog::with_selector(config.prometheus_selector.as_deref());
            let metrics_tx = metrics_tx.clone();
            let models = ingest.models.clone();
            let frontend_health = ingest.frontend_health.clone();
            let dc_id = config.dc_id.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move {
                if let Err(e) = run_metrics_publisher(
                    metrics_tx,
                    source,
                    catalog,
                    models,
                    frontend_health,
                    dc_id,
                    interval,
                    cancel,
                )
                .await
                {
                    tracing::error!(error = %e, "metrics publisher exited");
                }
            })
        };

        // Per-model cuckoo filter publisher is mandatory for SnapshotCuckoo consumers.
        let filter_pub_task = {
            let filters = filters.clone();
            let metrics = Some(relay_metrics.clone());
            let interval = std::time::Duration::from_millis(config.filter_interval_ms.max(50));
            let cancel = cancel.clone();
            tokio::spawn(async move {
                run_filter_publisher(filters, metrics, interval, cancel).await;
            })
        };

        // CBI1 snapshots and repeated metric frames compress well. Only the
        // encoding cap is raised (it must admit the largest bounded CBI1
        // chunk); inbound requests are tiny, so decoding keeps tonic's
        // conservative default.
        let svc = KvEventRelayServer::new(KvEventRelayService::new(
            instance_id,
            metrics_tx,
            filters,
            Some(relay_metrics),
            ingest.block_size.clone(),
            cancel.clone(),
        ))
        .accept_compressed(CompressionEncoding::Zstd)
        .send_compressed(CompressionEncoding::Zstd)
        .max_encoding_message_size(config.max_msg_bytes);

        // Standard `grpc.health.v1.Health` so k8s 1.24+ can use a native `grpc:`
        // probe instead of a side HTTP `/healthz`. Mark the service Serving once
        // we're past TLS setup; flip it to NotServing on graceful shutdown.
        let (health_reporter, health_svc) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<KvEventRelayServer<KvEventRelayService>>()
            .await;

        // Reflection (`grpc.reflection.v1`) — lets `grpcurl` describe the service
        // without a .proto checkout.
        let reflection_svc = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .context("building tonic reflection service")?;

        tracing::info!(addr = %config.bind, "gRPC server listening (mTLS required)");
        let shutdown = {
            let cancel = cancel.clone();
            async move {
                cancel.cancelled().await;
                // Tell health probes we're going down BEFORE tonic stops accepting
                // new streams, so k8s removes us from Endpoints before the cut.
                health_reporter
                    .set_not_serving::<KvEventRelayServer<KvEventRelayService>>()
                    .await;
            }
        };
        let serve = Server::builder()
            .http2_keepalive_interval(Some(std::time::Duration::from_millis(
                config.grpc_keepalive_interval_ms,
            )))
            .http2_keepalive_timeout(Some(std::time::Duration::from_millis(
                config.grpc_keepalive_timeout_ms,
            )))
            .tls_config(tls_config)?
            .add_service(svc)
            .add_service(health_svc)
            .add_service(reflection_svc)
            .serve_with_shutdown(config.bind, shutdown);

        // A serve error (bind conflict, TLS material rejected at runtime, ...)
        // is fatal: capture it so `run()` returns nonzero instead of exiting 0
        // on the watcher results alone.
        let serve_error = serve.await.err().map(|e| {
            tracing::error!(error = %e, "gRPC server exited with error");
            anyhow::Error::from(e).context("gRPC server failed")
        });

        // Cancellation propagates to watchers (which drain their subscriber
        // supervisors), publishers, and the sidecar.
        cancel.cancel();
        metrics_task.abort();
        let mut watcher_failure: Option<anyhow::Error> = None;
        for task in watcher_tasks {
            if let Ok(Err(error)) = task.await {
                watcher_failure.get_or_insert(error);
            }
        }
        let _ = metrics_pub_task.await;
        let _ = filter_pub_task.await;
        match serve_error.or(watcher_failure) {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}
