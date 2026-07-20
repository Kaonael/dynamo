// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dynamo-kv-event-relay` — per-DC bridge from the intra-DC Dynamo
//! event plane (`dynamo-runtime`; ZMQ or NATS depending on the
//! deployment backend) to a cross-DC gRPC server-streaming endpoint
//! consumed by the global gateway.
//!
//! Layout:
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────┐
//!  │ kv-event-relay process                                   │
//!  │                                                          │
//!  │ discovery::run_event_source_watcher                      │
//!  │   └─ events::subscriber::run_component_subscriber        │
//!  │        ├─ bootstrap_worker_state (TreeDump replay)       │
//!  │        ├─ event-plane subscription (ZMQ/NATS)            │
//!  │        └─ events::dedup (DC-wide refcount) →             │
//!  │           events::publisher::publish_batch ──►           │
//!  │             filter::update_model_filter ──►              │
//!  │             FilterRegistry                               │
//!  │                                                          │
//!  │ telemetry::publisher::run_metrics_publisher ──►         │
//!  │             metrics_tx (broadcast)                       │
//!  │ filter::run_filter_publisher ──►                         │
//!  │             filter_tx (broadcast)                        │
//!  │                                                          │
//!  │ tonic::transport::Server                                 │
//!  │   ├─ SubscribeMetrics ── reads metrics_tx                │
//!  │   └─ SubscribeFilter  ── reads FilterRegistry           │
//!  │                                                          │
//!  │  ↓ TLS + mTLS (rustls) — single port                     │
//!  └──────────────────────────────────────────────────────────┘
//! ```
//!
//! `payload` fields carry transport-private encodings from
//! `dynamo_kv_event_relay_proto::wire` (packed `MetricFrame`s and CBI1 bucket
//! images); model identity rides as a `model_key` hash. Kept compact and
//! decoupled from the `dynamo-kv-router` domain types so neither side
//! re-declares them in protobuf.

pub mod app;
pub mod discovery;
pub mod events;
pub mod filter;
pub mod frontend_health;
pub mod grpc_server;
pub mod model_registry;
pub mod observability;
pub mod state;
pub mod telemetry;

pub(crate) mod clock;
