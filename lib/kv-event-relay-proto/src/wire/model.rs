// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable conversion from public model identifiers to compact wire keys.

/// Fixed seed shared by relay and global-router.
const MODEL_HASH_SEED: u64 = 0x4D4F_4444_454C_4B31;

/// Deterministic hash of a `model_id` to the `fixed64 model_key` carried on
/// relay frames.
pub fn model_id_to_key(model_id: &str) -> u64 {
    xxhash_rust::xxh3::xxh3_64_with_seed(model_id.as_bytes(), MODEL_HASH_SEED)
}
