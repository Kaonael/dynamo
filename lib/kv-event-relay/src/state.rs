// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared ingest-path state: one [`IngestContext`] is read by the per-source
//! subscribers, the discovery watchers, and the gap/departure recovery helpers
//! as they fold the intra-DC event plane into dedup and the per-model filters.
//!
//! Transport-side handles — the relay `instance_id` and the metrics broadcast
//! — are *not* here: they live on `RelayApp` and reach the gRPC service /
//! publishers / metrics sidecar directly, so the ingest path only sees what
//! it actually mutates.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use crate::events::dedup::RefCountedDedup;
use crate::events::publisher::EventPublisher;
use crate::filter::FilterRegistry;
use crate::frontend_health::FrontendHealth;
use crate::model_registry::ModelRegistry;
use crate::observability::RelayMetrics;

/// Broadcast-channel capacity for metrics and filter frames. Slow gRPC
/// subscribers that fall behind by more than this trigger
/// `RecvError::Lagged`; the handler logs and closes the stream so the
/// client re-syncs from a fresh snapshot.
pub const BROADCAST_CAPACITY: usize = 1024;

/// DC-wide KV block size, learned from worker MDCs (`kv_cache_block_size`).
///
/// Block hashes only compare equal when producer and consumer agree on the
/// block size, so a single value must hold across every worker this relay
/// ingests. First non-zero observation wins and is reported to the global
/// router via `GetRelayInfo`; a conflicting later observation is a
/// deployment error — it is logged loudly and ignored, since flipping the
/// reported value mid-flight would just move the corruption around.
#[derive(Default)]
pub struct BlockSizeTracker {
    size: AtomicU32,
}

impl BlockSizeTracker {
    /// Record `size` for `(component, worker)`.
    pub fn observe(&self, component: &str, worker_id: u64, size: u32) {
        if size == 0 {
            return; // MDC predates the field or the engine didn't set it
        }
        match self
            .size
            .compare_exchange(0, size, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => {
                tracing::info!(
                    component,
                    worker_id,
                    block_size = size,
                    "DC block size learned from MDC"
                );
            }
            Err(current) if current == size => {}
            Err(current) => {
                tracing::error!(
                    component,
                    worker_id,
                    block_size = size,
                    dc_block_size = current,
                    "worker MDC disagrees with the DC-wide KV block size; its block hashes will never match — fix the deployment"
                );
            }
        }
    }

    /// DC-wide block size; 0 until the first worker MDC arrives.
    pub fn get(&self) -> u32 {
        self.size.load(Ordering::Relaxed)
    }
}

pub struct IngestContext {
    pub dedup: Arc<RefCountedDedup>,
    pub event_publisher: Arc<EventPublisher>,
    pub models: Arc<ModelRegistry>,
    /// DC-local frontend liveness from the discovery plane, joined to models
    /// via the MDC watcher; published as `frontend_healthy` per snapshot.
    pub frontend_health: Arc<FrontendHealth>,
    /// Shared with the gRPC service, which reports it in `GetRelayInfo`.
    pub block_size: Arc<BlockSizeTracker>,
    /// Per-model relay-side filters — the terminal sink of the ingest
    /// pipeline, shipped to routers via `SubscribeFilter`.
    pub filters: Arc<FilterRegistry>,
    /// Event-batching coalesce window. Zero ⇒ disabled: forward one frame per
    /// upstream event instead of coalescing per window.
    pub batch_window: Duration,
    /// Max events accumulated before an early flush when batching is enabled.
    pub batch_max_events: usize,
    /// Prometheus handles for the hot-path counters. `None` disables
    /// instrumentation (tests, or running without the metrics port).
    pub metrics: Option<Arc<RelayMetrics>>,
}

/// Lock `mutex`, recovering from poisoning. Callers deliberately prefer
/// logging the poisoned structure and preserving availability over cascading
/// a worker-task panic through the process.
pub(crate) fn lock_recovering<'a, T>(
    mutex: &'a std::sync::Mutex<T>,
    what: &'static str,
) -> std::sync::MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(what, "mutex was poisoned; recovering guarded state");
            mutex.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// RwLock read guard that recovers from poisoning, matching the crate's single
/// poison policy instead of `.unwrap()`.
pub(crate) fn read_recovering<'a, T>(
    lock: &'a std::sync::RwLock<T>,
    what: &'static str,
) -> std::sync::RwLockReadGuard<'a, T> {
    match lock.read() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(what, "rwlock read was poisoned; recovering guarded state");
            lock.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// RwLock write guard that recovers from poisoning; see [`read_recovering`].
pub(crate) fn write_recovering<'a, T>(
    lock: &'a std::sync::RwLock<T>,
    what: &'static str,
) -> std::sync::RwLockWriteGuard<'a, T> {
    match lock.write() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::error!(what, "rwlock write was poisoned; recovering guarded state");
            lock.clear_poison();
            poisoned.into_inner()
        }
    }
}

/// Minimal ingest context for filter/subscriber tests: batching off,
/// instrumentation off. Returns the context plus a filter receiver the
/// tests assert against.
#[cfg(test)]
pub(crate) fn test_ingest_context() -> (
    Arc<IngestContext>,
    tokio::sync::broadcast::Receiver<crate::filter::FilterFrame>,
) {
    use dynamo_kv_router::indexer::cuckoo::CkfConfig;

    let filters = Arc::new(FilterRegistry::new(256));
    let filter_rx = filters.subscribe_with_snapshot().receiver;
    let event_publisher = Arc::new(EventPublisher::new(
        filters.clone(),
        CkfConfig::new(4096),
        None,
    ));
    let ingest = Arc::new(IngestContext {
        dedup: Arc::new(RefCountedDedup::default()),
        event_publisher,
        models: Arc::new(ModelRegistry::default()),
        frontend_health: Arc::new(FrontendHealth::default()),
        block_size: Arc::new(BlockSizeTracker::default()),
        filters,
        batch_window: Duration::ZERO,
        batch_max_events: 1,
        metrics: None,
    });
    (ingest, filter_rx)
}

#[cfg(test)]
mod tests {
    use super::BlockSizeTracker;

    #[test]
    fn block_size_first_nonzero_wins_and_mismatch_is_flagged() {
        let tracker = BlockSizeTracker::default();
        assert_eq!(tracker.get(), 0);
        tracker.observe("backend", 1, 0); // unset MDC field: ignored
        assert_eq!(tracker.get(), 0);
        tracker.observe("backend", 1, 16);
        tracker.observe("backend", 2, 16);
        tracker.observe("backend", 3, 32);
        assert_eq!(
            tracker.get(),
            16,
            "conflicting observation must not flip the DC value"
        );
    }
}
