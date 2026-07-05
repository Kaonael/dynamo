// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Counting global allocator shared by the allocation-budget tests and the
//! transport-costs bench. Each binary installs it via `#[global_allocator]`
//! and reads the counters around the section under measurement; allocation
//! counts are deterministic for a fixed fixture, so the tests can assert hard
//! bounds.

#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct CountingAllocator;

pub static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
pub static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: every operation delegates to `System` with the caller-provided
// pointer/layout and only updates independent relaxed counters.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated with the original valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed)
                + layout.size() as u64;
            PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: delegated with the pointer/layout pair supplied by caller.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: delegated with the original pointer/layout and requested size.
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            let old_size = layout.size() as u64;
            let new_size = new_size as u64;
            let live = if new_size >= old_size {
                LIVE_BYTES.fetch_add(new_size - old_size, Ordering::Relaxed) + new_size - old_size
            } else {
                LIVE_BYTES.fetch_sub(old_size - new_size, Ordering::Relaxed) - old_size + new_size
            };
            PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }
}

/// Spread sequential fixture values across the hash space so they don't
/// cluster in a few buckets.
pub fn spread(value: u64) -> u64 {
    value.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}
