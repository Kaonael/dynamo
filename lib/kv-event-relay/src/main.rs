// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint: process-level init (logging, crypto provider), then the
//! relay lifecycle on the standard [`dynamo_runtime::Worker`] harness — which
//! owns the tokio runtime and traps SIGINT/SIGTERM into a cancellation of the
//! runtime's primary token. All wiring lives in [`dynamo_kv_event_relay::app`].

use anyhow::Result;

use dynamo_kv_event_relay::app::{RelayApp, RelayConfig};
use dynamo_runtime::Worker;

fn main() -> Result<()> {
    dynamo_runtime::logging::init();

    // rustls 0.23 needs an explicit `CryptoProvider::install_default` before any
    // TLS handshake. tonic-0.14's `tls-ring` feature pulls ring in but doesn't
    // auto-install it.
    let _ = rustls::crypto::ring::default_provider().install_default();

    // Parse and validate before the runtime spins up so `--help` and config
    // errors exit immediately.
    let config = RelayConfig::parse().validate()?;
    Worker::from_settings()?
        .execute(move |runtime| async move { RelayApp::build(config, runtime).await?.run().await })
}
