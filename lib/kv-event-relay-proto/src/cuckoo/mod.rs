// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep the cuckoo membership, overlap search, and CKF1 wire assembly together
//! so relay and router share one authoritative implementation.

mod filter;
mod overlap;
mod pages;
mod producer;
mod snapshot;
mod transposed;

pub use filter::{CuckooFilter, DEFAULT_FILTER_SEED};
pub use overlap::{
    Probe, argmax_overlap_dc, overlap_depth_searched, overlap_depth_searched_seq, probes_for,
};
pub use transposed::{
    COHORT_LANES, CohortShapeError, MultiOverlapScratch, TransposedCohort, overlap_depths_multi,
};
pub use producer::{MAX_DELTA_BYTES, Publish, SnapshotProducer};
pub use snapshot::{
    DeltaError, DeltaInfo, SNAP_HEADER_LEN, SnapshotAssembler, SnapshotError, SnapshotMeta,
    SnapshotState, apply_delta, assemble_chunks, is_chunk, is_delta,
};

#[cfg(test)]
mod tests;
