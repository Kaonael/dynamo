// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wires a validated [`RelayConfig`] into a ready-to-run [`RelayApp`]: the
//! distributed runtime, TLS identity, shared component graph, and Prometheus
//! handles. No tasks are spawned here — that is [`RelayApp::run`].

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use tokio::sync::broadcast;
use tonic::transport::{Identity, ServerTlsConfig};

use dynamo_runtime::{DistributedRuntime, Runtime};

use crate::events::dedup::RefCountedDedup;
use crate::events::publisher::EventPublisher;
use crate::filter::FilterRegistry;
use crate::model_registry::ModelRegistry;
use crate::observability::RelayMetrics;
use crate::state::{BROADCAST_CAPACITY, IngestContext};

use super::config::RelayConfig;
use super::runtime::RelayApp;

/// A fresh 128-bit relay epoch. The global gateway treats a changed
/// `instance_id` as "this relay restarted; full resync required".
pub(crate) fn relay_instance_id() -> bytes::Bytes {
    bytes::Bytes::copy_from_slice(uuid::Uuid::new_v4().as_bytes())
}

impl RelayApp {
    /// Construct the full component graph from `config` on the
    /// [`Worker`](dynamo_runtime::Worker)-provided runtime. Fails on TLS
    /// material or distributed-runtime errors before any task is spawned.
    pub async fn build(config: RelayConfig, runtime: Runtime) -> Result<RelayApp> {
        let drt = DistributedRuntime::from_settings(runtime).await?;

        let (tls_config, tls_expiry) = build_tls_config(
            &config.tls_server_cert,
            &config.tls_server_key,
            &config.tls_client_ca,
        )?;

        let relay_metrics = Arc::new(RelayMetrics::new().context("constructing relay metrics")?);
        for (material, not_after) in tls_expiry {
            relay_metrics
                .tls_expiry_timestamp_seconds
                .with_label_values(&[material])
                .set(not_after);
        }
        let (metrics_tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        let instance_id = relay_instance_id();

        let filters = Arc::new(FilterRegistry::new(BROADCAST_CAPACITY));
        let ckf_config =
            dynamo_kv_router::indexer::cuckoo::CkfConfig::new(config.filter_capacity_hint);
        let event_publisher = Arc::new(EventPublisher::new(
            filters.clone(),
            ckf_config,
            Some(relay_metrics.clone()),
        ));
        // Shared ingest context: only what the watchers/subscribers/recovery
        // mutate. Transport handles ride on `RelayApp` separately.
        let ingest = Arc::new(IngestContext {
            dedup: Arc::new(RefCountedDedup::default()),
            event_publisher,
            models: Arc::new(ModelRegistry::default()),
            frontend_health: Arc::new(crate::frontend_health::FrontendHealth::default()),
            block_size: Arc::new(crate::state::BlockSizeTracker::default()),
            filters: filters.clone(),
            batch_window: Duration::from_millis(config.batch_window_ms),
            batch_max_events: config.batch_max_events,
            metrics: Some(relay_metrics.clone()),
        });

        Ok(RelayApp {
            config,
            drt,
            instance_id,
            filters,
            metrics_tx,
            relay_metrics,
            ingest,
            tls_config,
        })
    }
}

/// Wire the relay's TLS identity + the CA bundle used to verify
/// global-gateway clients (mTLS, always required). Also returns the earliest
/// `notAfter` per material for the expiry gauge — the files are read once at
/// startup and never reloaded, so expiry is a restart-the-pod alert.
fn build_tls_config(
    server_cert: &PathBuf,
    server_key: &PathBuf,
    client_ca: &PathBuf,
) -> Result<(ServerTlsConfig, Vec<(&'static str, i64)>)> {
    let cert = std::fs::read(server_cert)
        .with_context(|| format!("reading TLS server cert {}", server_cert.display()))?;
    let key = std::fs::read(server_key)
        .with_context(|| format!("reading TLS server key {}", server_key.display()))?;
    let ca = std::fs::read(client_ca)
        .with_context(|| format!("reading client-CA bundle {}", client_ca.display()))?;

    let mut expiry = Vec::new();
    for (material, pem) in [("server_cert", &cert), ("client_ca", &ca)] {
        match earliest_not_after(pem) {
            Some(not_after) => expiry.push((material, not_after)),
            None => tracing::warn!(
                material,
                "could not parse notAfter from PEM; expiry gauge not set"
            ),
        }
    }

    let identity = Identity::from_pem(cert, key);
    let ca_cert = tonic::transport::Certificate::from_pem(ca);
    Ok((
        ServerTlsConfig::new()
            .identity(identity)
            .client_ca_root(ca_cert),
        expiry,
    ))
}

/// Earliest `notAfter` (unix seconds) across every certificate in a PEM
/// bundle — for a chain or CA bundle the first expiry is what breaks the
/// handshake. `None` when nothing parseable is found (tonic will surface a
/// real TLS-material error on its own).
fn earliest_not_after(pem: &[u8]) -> Option<i64> {
    x509_parser::pem::Pem::iter_from_buffer(pem)
        .filter_map(|pem| {
            let pem = pem.ok()?;
            let cert = pem.parse_x509().ok()?;
            Some(cert.validity().not_after.timestamp())
        })
        .min()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_instance_ids_are_random_128_bit_epochs() {
        let first = relay_instance_id();
        let second = relay_instance_id();
        assert_eq!(first.len(), 16);
        assert_eq!(second.len(), 16);
        assert_ne!(first, second);
    }

    /// Self-signed test certificate, notAfter = 2026-07-16T17:15:05Z.
    const TEST_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIDBzCCAe+gAwIBAgIUPlwUBjTCvHFKr288WGaDmjFA5egwDQYJKoZIhvcNAQEL
BQAwEzERMA8GA1UEAwwIc21va2UtY2EwHhcNMjYwNzE0MTcxNTA1WhcNMjYwNzE2
MTcxNTA1WjATMREwDwYDVQQDDAhzbW9rZS1jYTCCASIwDQYJKoZIhvcNAQEBBQAD
ggEPADCCAQoCggEBANJzBlZQz7UysbejNPzBMEsVzZHCRx1Eu0TXzj1/FANYjIKz
xjN+v49q2jwsvl7HMNK7jMWJ2V1oBv0W7ZNmSUYA1qyERN2j8c8Z257Cf+c0tGS7
I/wbtDX/g9knCrUdelKYxub8oRhyGMP+iIMR5w1LUQvmAULoaszLRab4+GOV8ijU
IYVZvxTsMFY0ztdG6pP/H7gIJXkuwdfqC+BRCXoO/ppWa2MdGz3zz+uaG8nPhX7u
sBqohwHug8DPnBHCZT2jJisNcV3zylVNnGtPS/TnV288mgJbKecP2IDzS5GP35XZ
rYToMZ9k7IDU97+BzCdLIpqc9ZFsWD92ANcaxA0CAwEAAaNTMFEwHQYDVR0OBBYE
FEi1N71BRzWnbtAqVF3D3OQQLkOtMB8GA1UdIwQYMBaAFEi1N71BRzWnbtAqVF3D
3OQQLkOtMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAIY8H/6Y
R3E+hB24zGRInHAWP5HZQRPpPg8kR+eMvMN5xoJ5ShtmBXSUPERTqP8Y32qVGCvZ
VFu026QlQf2itEXeVjXH/Uj60m3lBnu7/oEK08miNtIXC1fDQof3zcEa25794Tyr
Zygnb7ujYQjJDHAoP0DG0XPfGt08iP3BNCFLPmomz1CBSpXRri8W20/Enbv7XfRW
Y1DzvndIwXiastq4lcR02EoP3rDX1WQpodnt+8bVl35Knb//1dxlyAr4V2y6YZrG
Edu8xGIZutL0KaA15LGy07BVivS3sy5uvXZaghtvHeYHf7up8g2Il3eHLuaiOjbB
WAQz2ykKZ+IVQoM=
-----END CERTIFICATE-----
";

    #[test]
    fn not_after_is_parsed_from_pem_bundles() {
        assert_eq!(
            earliest_not_after(TEST_CERT_PEM.as_bytes()),
            Some(1_784_222_105)
        );
        // A bundle reports the earliest expiry (here: two identical certs).
        let bundle = format!("{TEST_CERT_PEM}{TEST_CERT_PEM}");
        assert_eq!(earliest_not_after(bundle.as_bytes()), Some(1_784_222_105));
        assert_eq!(earliest_not_after(b"not pem"), None);
    }
}
