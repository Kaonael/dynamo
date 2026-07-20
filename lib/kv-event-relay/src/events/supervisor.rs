// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owns the per-`(component, publisher)` subscriber tasks the discovery watcher
//! spawns: a child cancellation token plus the task's `JoinHandle`. Holding the
//! handle is what lets worker removal run **cancel → join → dedup eviction** in
//! order — the subscriber's cancel-time final flush mutates dedup/filters, so
//! the eviction must not start until that flush has fully drained, or it would
//! compute its corrective `Removed` against a racing producer.
//!
//! Removal runs off the watch loop ([`begin_drain`](SubscriberSupervisor::begin_drain)),
//! so a worker that flaps — `Removed` then `Added` of the same
//! `(component, instance)` within the eviction window — could otherwise let the
//! stale eviction strip the freshly re-bootstrapped slot. The supervisor keeps
//! the in-flight drain's handle so a re-`spawn` of the same key waits for that
//! eviction to finish before bootstrapping again.

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// How long a cancelled subscriber gets to drain its final flush before it is
/// force-aborted. A well-behaved subscriber returns near-instantly on cancel;
/// the grace only bounds a task wedged on a slow await.
pub(crate) const SUBSCRIBER_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// A live subscriber's cancellation token and join handle.
pub(crate) struct SubscriberHandle {
    cancel: CancellationToken,
    join: JoinHandle<()>,
}

impl SubscriberHandle {
    /// Cancel the subscriber and wait for it to finish (including its
    /// cancel-time flush). If it overruns `grace`, abort it — the subsequent
    /// dedup eviction repairs any partially-applied state from the refcount
    /// holder set regardless.
    pub(crate) async fn shutdown(self, grace: Duration) {
        self.cancel.cancel();
        let abort = self.join.abort_handle();
        match tokio::time::timeout(grace, self.join).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) if error.is_panic() => {
                tracing::warn!("subscriber task panicked during shutdown");
            }
            Ok(Err(_)) => {}
            Err(_) => {
                tracing::warn!("subscriber did not stop within grace; aborting");
                abort.abort();
            }
        }
    }
}

/// Registry of running subscriber tasks, keyed by `(component, publisher_id)`.
pub(crate) struct SubscriberSupervisor {
    parent_cancel: CancellationToken,
    tasks: HashMap<(String, u64), SubscriberHandle>,
    /// In-flight `shutdown → evict` drains, keyed the same way. A re-`spawn` of
    /// a draining key waits on its handle before bootstrapping so the stale
    /// eviction can't strip the new incarnation's blocks.
    draining: HashMap<(String, u64), JoinHandle<()>>,
}

impl SubscriberSupervisor {
    pub(crate) fn new(parent_cancel: CancellationToken) -> Self {
        Self {
            parent_cancel,
            tasks: HashMap::new(),
            draining: HashMap::new(),
        }
    }

    /// Spawn a subscriber under a fresh child token, keyed by `key`. No-op if a
    /// subscriber for `key` already runs. Returns whether it spawned. `make`
    /// receives the child token and returns the subscriber future.
    ///
    /// If `key` is mid-drain (a recent `Removed` whose eviction hasn't
    /// finished), the new subscriber awaits that drain before running `make`,
    /// so bootstrap always observes a post-eviction dedup state.
    pub(crate) fn spawn<F, Fut>(&mut self, key: (String, u64), make: F) -> bool
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        // Reap dead subscribers first: a task that exited on its own (e.g. a
        // transport error) must not block this key forever — the next
        // discovery `Added` (including a watch-reconnect re-list) respawns it.
        self.tasks.retain(|_, handle| !handle.join.is_finished());
        if self.tasks.contains_key(&key) {
            return false;
        }
        self.reap_finished_drains();
        let prior_drain = self.draining.remove(&key);
        let cancel = self.parent_cancel.child_token();
        let child = cancel.clone();
        let join = tokio::spawn(async move {
            if let Some(drain) = prior_drain {
                let _ = drain.await;
            }
            make(child).await;
        });
        self.tasks.insert(key, SubscriberHandle { cancel, join });
        true
    }

    /// Begin draining a removed worker off the watch loop: cancel + join its
    /// subscriber (bounded by `grace`), then run `evict`. The drain handle is
    /// retained so a re-`spawn` of the same key can serialize behind it.
    pub(crate) fn begin_drain<F, Fut>(&mut self, key: (String, u64), grace: Duration, evict: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.reap_finished_drains();
        let handle = self.tasks.remove(&key);
        let drain = tokio::spawn(async move {
            if let Some(handle) = handle {
                handle.shutdown(grace).await;
            }
            evict().await;
        });
        self.draining.insert(key, drain);
    }

    /// Drop the handles of drains that have already finished so the map stays
    /// bounded under steady worker churn.
    fn reap_finished_drains(&mut self) {
        self.draining.retain(|_, drain| !drain.is_finished());
    }

    /// Keys of currently supervised subscribers — the reconciliation set the
    /// watcher diffs against a fresh discovery list after a watch gap.
    pub(crate) fn active_keys(&self) -> Vec<(String, u64)> {
        self.tasks.keys().cloned().collect()
    }

    #[cfg(test)]
    fn take(&mut self, key: &(String, u64)) -> Option<SubscriberHandle> {
        self.tasks.remove(key)
    }

    /// Cancel and join every remaining subscriber, then await any in-flight
    /// drains, e.g. when the owning watcher is shutting down. Each subscriber
    /// gets the same bounded grace before abort.
    pub(crate) async fn shutdown_all(&mut self, grace: Duration) {
        for (_, handle) in self.tasks.drain().collect::<Vec<_>>() {
            handle.shutdown(grace).await;
        }
        for (_, drain) in self.draining.drain().collect::<Vec<_>>() {
            let _ = drain.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// `shutdown` must not return until the cancelled task has fully finished
    /// its cancel-time work — so the caller's next step (eviction) can never
    /// race the subscriber's final flush.
    #[tokio::test]
    async fn shutdown_joins_before_caller_proceeds() {
        let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let mut sup = SubscriberSupervisor::new(CancellationToken::new());
        let key = ("c".to_string(), 1u64);
        {
            let log = log.clone();
            sup.spawn(key.clone(), move |cancel| async move {
                cancel.cancelled().await;
                // Simulate the cancel-time final flush taking real time.
                tokio::time::sleep(Duration::from_millis(50)).await;
                log.lock().unwrap().push("flushed");
            });
        }

        let handle = sup.take(&key).expect("supervised");
        handle.shutdown(Duration::from_secs(5)).await;
        log.lock().unwrap().push("evicted");

        assert_eq!(*log.lock().unwrap(), vec!["flushed", "evicted"]);
    }

    /// A task that ignores cancellation is force-aborted at the grace deadline,
    /// so removal can never wedge the watcher.
    #[tokio::test]
    async fn shutdown_aborts_unresponsive_task() {
        let mut sup = SubscriberSupervisor::new(CancellationToken::new());
        let key = ("c".to_string(), 1u64);
        sup.spawn(key.clone(), |_cancel| async move {
            loop {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        let handle = sup.take(&key).unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            handle.shutdown(Duration::from_millis(50)),
        )
        .await
        .expect("shutdown must return after aborting the wedged task");
    }

    /// A worker that flaps (`Removed` then `Added` of the same key before the
    /// eviction finishes) must bootstrap only after its prior eviction lands —
    /// otherwise the stale `evict` would strip the new incarnation's blocks.
    #[tokio::test(flavor = "multi_thread")]
    async fn re_add_waits_for_prior_eviction() {
        let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let mut sup = SubscriberSupervisor::new(CancellationToken::new());
        let key = ("c".to_string(), 1u64);

        sup.spawn(key.clone(), |cancel| async move {
            cancel.cancelled().await;
        });

        {
            let log = log.clone();
            sup.begin_drain(key.clone(), Duration::from_secs(5), move || async move {
                // Make the race observable: a stale eviction would otherwise
                // land after the re-add's bootstrap below.
                tokio::time::sleep(Duration::from_millis(50)).await;
                log.lock().unwrap().push("evicted");
            });
        }

        {
            let log = log.clone();
            sup.spawn(key.clone(), move |_cancel| async move {
                log.lock().unwrap().push("bootstrapped");
            });
        }

        for _ in 0..200 {
            if log.lock().unwrap().len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(*log.lock().unwrap(), vec!["evicted", "bootstrapped"]);
    }

    #[tokio::test]
    async fn spawn_is_idempotent_per_key() {
        let mut sup = SubscriberSupervisor::new(CancellationToken::new());
        let key = ("c".to_string(), 1u64);
        assert!(sup.spawn(
            key.clone(),
            |cancel| async move { cancel.cancelled().await }
        ));
        assert!(!sup.spawn(
            key.clone(),
            |cancel| async move { cancel.cancelled().await }
        ));
    }

    /// A subscriber that died on its own (transport error) must not occupy its
    /// key: the next discovery `Added` for the same worker respawns it.
    #[tokio::test]
    async fn finished_task_does_not_block_respawn() {
        let mut sup = SubscriberSupervisor::new(CancellationToken::new());
        let key = ("c".to_string(), 1u64);
        assert!(sup.spawn(key.clone(), |_cancel| async move {}));

        // Let the task finish.
        for _ in 0..100 {
            if sup
                .tasks
                .get(&key)
                .is_some_and(|handle| handle.join.is_finished())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        assert!(
            sup.spawn(
                key.clone(),
                |cancel| async move { cancel.cancelled().await }
            ),
            "finished subscriber must be reaped so the key can respawn"
        );
    }
}
