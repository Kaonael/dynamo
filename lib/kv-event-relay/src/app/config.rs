// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI configuration for the relay process: parse with [`clap`], then
//! [`RelayConfig::validate`] before any wiring.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

/// Slack on top of the delta cap for the frame header and the protobuf
/// envelope around the payload bytes.
const DELTA_ENVELOPE_HEADROOM: usize = 64 * 1024;

#[derive(Debug, Parser)]
#[command(
    name = "dynamo-kv-event-relay",
    about = "Bridges per-DC Dynamo KV events to a cross-DC gRPC server-streaming endpoint."
)]
pub struct RelayConfig {
    /// Human-readable DC id, surfaced in logs and stamped on every
    /// outbound `MetricsSnapshot`.
    #[arg(long, env = "DYN_DC_ID")]
    pub dc_id: String,

    /// One or more Dynamo namespaces whose kv-events we relay (must
    /// match `dynamo.vllm`/`dynamo.sglang` `--namespace`). Comma-
    /// separated; one watcher per namespace shares the same ingest
    /// context, so all pools in a DC publish through a single
    /// gRPC server-stream stamped with `--dc-id`.
    #[arg(
        long,
        env = "DYN_RELAY_NAMESPACES",
        value_delimiter = ',',
        default_value = "dynamo"
    )]
    pub namespaces: Vec<String>,

    /// TCP address to bind the gRPC server on. mTLS is mandatory —
    /// see `--tls-*` flags.
    #[arg(long, env = "DYN_RELAY_BIND", default_value = "0.0.0.0:5560")]
    pub bind: SocketAddr,

    /// TCP address for the plaintext Prometheus `/metrics` sidecar.
    /// Conventionally distinct from `--bind` so mTLS gRPC traffic and
    /// scrape traffic can be firewalled independently.
    #[arg(long, env = "DYN_RELAY_METRICS_LISTEN", default_value = "0.0.0.0:9090")]
    pub metrics_listen: SocketAddr,

    /// HTTP/2 keepalive ping interval. Reaps half-open / stalled connections so
    /// a non-reading client cannot pin a live stream slot indefinitely.
    #[arg(
        long,
        env = "DYN_RELAY_GRPC_KEEPALIVE_INTERVAL_MS",
        default_value_t = 20_000
    )]
    pub grpc_keepalive_interval_ms: u64,

    /// How long to wait for a keepalive ping ACK before closing the connection.
    #[arg(
        long,
        env = "DYN_RELAY_GRPC_KEEPALIVE_TIMEOUT_MS",
        default_value_t = 10_000
    )]
    pub grpc_keepalive_timeout_ms: u64,

    /// gRPC max message size in bytes (encode on this server; the global gateway
    /// mirrors it on decode). CBI1 snapshot chunks can exceed tonic's 4 MiB
    /// default and would break `SubscribeFilter`; raise this so they pass.
    /// Default 64 MiB.
    #[arg(long, env = "DYN_RELAY_MAX_MSG_BYTES", default_value_t = 67_108_864)]
    pub max_msg_bytes: usize,

    /// Event-batching coalesce window in milliseconds. `0` keeps the
    /// legacy behaviour of one frame per upstream event. When `>0`, dedup-
    /// filtered events are accumulated per model and flushed on this tick (or
    /// when `--batch-max-events` is reached), amortising the per-frame envelope
    /// and HTTP/2 framing across many events.
    // Default 10 ms: an honest coalesce window for a WAN-bound pipeline whose
    // filter publisher only ticks at ~1 s. 1 ms armed a needless ~1 kHz flush
    // timer per subscriber; lazy deadline arming is a follow-up.
    #[arg(long, env = "DYN_RELAY_BATCH_WINDOW_MS", default_value_t = 10)]
    pub batch_window_ms: u64,

    /// Max events buffered before an early flush when batching is enabled
    /// (`--batch-window-ms > 0`). Bounds per-batch size and added latency.
    #[arg(long, env = "DYN_RELAY_BATCH_MAX_EVENTS", default_value_t = 256)]
    pub batch_max_events: usize,

    /// How often to publish a `FilterUpdate` (full snapshot / delta) per model.
    #[arg(long, env = "DYN_RELAY_FILTER_INTERVAL_MS", default_value_t = 1000)]
    pub filter_interval_ms: u64,

    /// Per-model, per-DC filter capacity in blocks. The upstream CKF is
    /// fixed-size (sized once from this, not resized), so set it for the DC's
    /// working set; inserts past capacity fail the event and are logged.
    #[arg(long, env = "DYN_RELAY_FILTER_CAPACITY_HINT", default_value_t = 65536)]
    pub filter_capacity_hint: usize,

    /// Base URL of the DC's in-cluster Prometheus (PromQL API root, e.g.
    /// `http://prometheus-kube-prometheus-prometheus.monitoring:9090`). When
    /// set, the relay queries the routing metric catalog here each tick. When
    /// unset, no metrics snapshots are published.
    #[arg(long, env = "DYN_RELAY_PROMETHEUS_URL")]
    pub prometheus_url: Option<String>,

    /// Optional bearer token for the Prometheus HTTP API (when it sits behind
    /// auth). Sent as `Authorization: Bearer <token>`. Env-only
    /// (`DYN_RELAY_PROMETHEUS_BEARER_TOKEN`) — a CLI flag would leak the
    /// secret through the process command line.
    #[arg(skip)]
    pub prometheus_bearer_token: Option<String>,

    /// Extra PromQL label matcher (e.g. `namespace="dc-ams"`) scoping every
    /// catalog query to this DC's series. Required when the Prometheus
    /// backend holds more than this DC (shared VictoriaMetrics, or one
    /// cluster hosting several per-namespace DCs) — otherwise same-model
    /// series from other DCs pollute this DC's snapshots.
    #[arg(long, env = "DYN_RELAY_PROMETHEUS_SELECTOR")]
    pub prometheus_selector: Option<String>,

    /// How often to query Prometheus and emit a `DcModelMetricsSnapshot` on the
    /// metrics channel.
    #[arg(long, env = "DYN_RELAY_METRICS_INTERVAL_MS", default_value_t = 1000)]
    pub metrics_interval_ms: u64,

    /// Path to the PEM-encoded server certificate the relay presents
    /// during the TLS handshake. **Required** — cleartext is not
    /// supported.
    #[arg(long, env = "DYN_RELAY_TLS_SERVER_CERT")]
    pub tls_server_cert: PathBuf,

    /// Path to the PEM-encoded private key matching
    /// `--tls-server-cert`.
    #[arg(long, env = "DYN_RELAY_TLS_SERVER_KEY")]
    pub tls_server_key: PathBuf,

    /// Path to the PEM-encoded CA bundle the relay uses to verify
    /// connecting global-gateway clients. **Required** — mTLS is
    /// always-on; cleartext or anonymous TLS are rejected.
    #[arg(long, env = "DYN_RELAY_TLS_CLIENT_CA")]
    pub tls_client_ca: PathBuf,
}

impl RelayConfig {
    /// Parse from the process arguments.
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }

    /// Reject configurations that clap can't express as constraints, and pull
    /// in the env-only secrets clap deliberately does not parse.
    pub fn validate(mut self) -> Result<Self> {
        self.prometheus_bearer_token = std::env::var("DYN_RELAY_PROMETHEUS_BEARER_TOKEN")
            .ok()
            .filter(|token| !token.is_empty());
        if self.namespaces.is_empty() {
            anyhow::bail!("--namespaces requires at least one value");
        }
        if self.filter_capacity_hint == 0 {
            anyhow::bail!("--filter-capacity-hint must be greater than zero");
        }
        // A CBI1 snapshot chunk (and any delta is capped to the same size) can
        // reach IMAGES_MAX_FRAME_BYTES. With a message cap below that, frames
        // are rejected by the transport and `SubscribeFilter` silently breaks
        // while everything still "works". Fail fast instead.
        let min_msg_bytes = dynamo_kv_event_relay_proto::wire::images::IMAGES_MAX_FRAME_BYTES
            + DELTA_ENVELOPE_HEADROOM;
        if self.max_msg_bytes < min_msg_bytes {
            anyhow::bail!(
                "--max-msg-bytes {} is below the CBI1 frame cap ({} + {} envelope headroom); \
                 large frames would be rejected and SubscribeFilter would break",
                self.max_msg_bytes,
                dynamo_kv_event_relay_proto::wire::images::IMAGES_MAX_FRAME_BYTES,
                DELTA_ENVELOPE_HEADROOM,
            );
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_args() -> Vec<&'static str> {
        vec![
            "relay",
            "--dc-id",
            "dc1",
            "--tls-server-cert",
            "server.crt",
            "--tls-server-key",
            "server.key",
            "--tls-client-ca",
            "ca.crt",
        ]
    }

    #[test]
    fn msg_cap_below_frame_cap_fails_validation() {
        let mut argv = required_args();
        argv.extend(["--max-msg-bytes", "4194304"]);
        let config = RelayConfig::try_parse_from(argv).unwrap();
        let error = config.validate().unwrap_err().to_string();
        assert!(
            error.contains("CBI1 frame cap"),
            "unexpected error: {error}"
        );

        let default_config = RelayConfig::try_parse_from(required_args()).unwrap();
        assert!(default_config.validate().is_ok());
    }

    #[test]
    fn zero_still_disables_batching() {
        let mut argv = required_args();
        argv.extend(["--batch-window-ms", "0"]);
        let config = RelayConfig::try_parse_from(argv).unwrap();
        assert_eq!(config.batch_window_ms, 0);
    }

    #[test]
    fn empty_namespaces_is_rejected() {
        let config = RelayConfig::try_parse_from(required_args()).unwrap();
        let config = RelayConfig {
            namespaces: vec![],
            ..config
        };
        assert!(config.validate().is_err());
    }
}
