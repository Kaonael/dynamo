// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV-event pipeline: discovery-spawned [`subscriber`]s ingest the
//! intra-DC event plane, [`dedup`] collapses per-worker events to one
//! `Stored`/`Removed` per `(block, DC)`, and [`publisher`] folds each
//! forwarded batch into the per-model cuckoo filter.

pub(crate) mod batcher;
pub mod dedup;
pub mod publisher;
pub(crate) mod recovery;
pub mod subscriber;
pub(crate) mod supervisor;
pub(crate) mod worker_state;
