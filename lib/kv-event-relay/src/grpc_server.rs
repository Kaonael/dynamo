// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `KvEventRelay` tonic service — relay transport RPC handlers:
//!
//!   * [`KvEventRelayService::subscribe_metrics`] streams 1 Hz
//!     `MetricsSnapshot`s from the `metrics_tx` broadcast.
//!   * [`KvEventRelayService::subscribe_filter`] streams CBI1 `FilterUpdate`s
//!     from `FilterRegistry`, prepending a full snapshot per model on connect.
//!
//! [`service`] holds the trait impl and the transport handles; [`streams`] is
//! the live server-streaming plumbing. The broadcast `RecvError::Lagged` path
//! terminates the stream with `resource_exhausted`; the global subscriber
//! reconnects, and `SubscribeFilter` re-syncs it from a fresh full snapshot.

mod service;
mod streams;

pub use service::KvEventRelayService;
