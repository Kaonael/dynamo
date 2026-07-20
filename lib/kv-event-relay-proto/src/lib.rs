// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep the relay and global-gateway on one generated transport contract so
//! protocol changes stay synchronized across both ends.

pub mod v1 {
    // Generated code is allowed to carry upstream lint noise.
    #![allow(clippy::all)]
    tonic::include_proto!("dynamo.kvrelay.v1");
}

/// Keep the compact transport encodings and model-key derivation local to this
/// crate so the wire contract can evolve without duplicating logic elsewhere.
pub mod wire;

/// Keep relay telemetry in one schema so the router can score it without
/// depending on engine-specific internals.
pub mod metrics;

pub use v1::{
    FilterUpdate, MetricsSnapshot, RelayInfo, RelayInfoRequest, SubscribeRequest,
    kv_event_relay_client::KvEventRelayClient,
    kv_event_relay_server::{KvEventRelay, KvEventRelayServer},
};

pub use wire::{MetricDecodeError, MetricFrame, metric_key, model_id_to_key};

/// Export the descriptor set so the relay can serve reflection without
/// shipping a separate `.proto` tree.
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/relay_descriptor.bin"));
