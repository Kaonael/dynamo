// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock send timestamps stamped on outbound frames.

use std::time::{SystemTime, UNIX_EPOCH};

/// Microsecond resolution is enough to register intra-DC latency (typically
/// 100us–2ms), which ms resolution truncated to 0. u64 fits μs-since-epoch
/// through year ~586524. Saturating to 0 handles the impossible-but-typed
/// `UNIX_EPOCH > now` case; the global side's `observe_latency` already drops
/// nonsensical values.
pub(crate) fn unix_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

pub(crate) fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
