// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Application lifecycle: [`RelayConfig`] (CLI), [`RelayApp::build`] (wiring),
//! and [`RelayApp::run`] (serve + shutdown). The binary entrypoint is just
//! `RelayApp::build(RelayConfig::parse().validate()?).await?.run().await`.

mod builder;
mod config;
mod runtime;

pub use config::RelayConfig;
pub use runtime::RelayApp;
