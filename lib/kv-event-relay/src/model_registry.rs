// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;
use tokio::sync::{Notify, RwLock};
use tokio_util::sync::CancellationToken;

use dynamo_kv_event_relay_proto::wire;
use dynamo_kv_router::protocols::WorkerId;

#[derive(Clone)]
pub struct ModelIdentity {
    pub model_id: Arc<str>,
    pub model_key: u64,
}

pub(crate) struct ModelIdentityCache {
    generation: u64,
    by_worker: FxHashMap<WorkerId, ModelIdentity>,
}

impl Default for ModelIdentityCache {
    fn default() -> Self {
        Self {
            generation: u64::MAX,
            by_worker: FxHashMap::default(),
        }
    }
}

/// ModelDeploymentCard (MDC)-fed `(component, instance) → model_id` map. There
/// is deliberately no fallback identity: events are attributable only once the
/// worker's MDC has
/// arrived, so subscribers gate on [`wait_registered`](Self::wait_registered)
/// and drop events whose model has since unregistered — a made-up model key
/// would leak phantom filters to the global gateway instead.
///
/// The generation counter invalidates event-path [`ModelIdentityCache`]s
/// whenever the map changes.
#[derive(Default)]
pub struct ModelRegistry {
    by_instance: RwLock<HashMap<(String, u64), String>>,
    generation: AtomicU64,
    registered: Notify,
}

impl ModelRegistry {
    pub async fn register(&self, component: String, instance_id: u64, model_id: String) {
        self.by_instance
            .write()
            .await
            .insert((component, instance_id), model_id);
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.registered.notify_waiters();
    }

    pub async fn remove(&self, component: &str, instance_id: u64) -> bool {
        let removed = self
            .by_instance
            .write()
            .await
            .remove(&(component.to_string(), instance_id))
            .is_some();
        if removed {
            self.generation.fetch_add(1, Ordering::Relaxed);
        }
        removed
    }

    async fn lookup(&self, component: &str, worker_id: WorkerId) -> Option<ModelIdentity> {
        self.by_instance
            .read()
            .await
            .get(&(component.to_string(), worker_id))
            .map(|model_id| {
                let model_id = Arc::<str>::from(model_id.as_str());
                ModelIdentity {
                    model_key: wire::model_id_to_key(&model_id),
                    model_id,
                }
            })
    }

    /// Block until an MDC for `(component, worker_id)` is registered, or
    /// `cancel` fires (`None`). Subscribers call this before bootstrapping so
    /// no event is ever processed under an unknown model identity.
    pub(crate) async fn wait_registered(
        &self,
        component: &str,
        worker_id: WorkerId,
        cancel: &CancellationToken,
    ) -> Option<ModelIdentity> {
        loop {
            // Arm before the lookup so a concurrent `register` cannot slip
            // between the miss and the wait.
            let registered = self.registered.notified();
            if let Some(model) = self.lookup(component, worker_id).await {
                return Some(model);
            }
            tokio::select! {
                _ = cancel.cancelled() => return None,
                _ = registered => {}
            }
        }
    }

    pub(crate) async fn resolve(
        &self,
        component: &str,
        worker_id: WorkerId,
        cache: &mut ModelIdentityCache,
    ) -> Option<ModelIdentity> {
        let generation = self.generation.load(Ordering::Relaxed);
        if cache.generation != generation {
            cache.by_worker.clear();
            cache.generation = generation;
        }
        if let Some(model) = cache.by_worker.get(&worker_id) {
            return Some(model.clone());
        }
        let model = self.lookup(component, worker_id).await?;
        cache.by_worker.insert(worker_id, model.clone());
        Some(model)
    }

    pub async fn resolve_once(
        &self,
        component: &str,
        worker_id: WorkerId,
    ) -> Option<ModelIdentity> {
        self.lookup(component, worker_id).await
    }

    /// Drop registrations for `component` instances absent from `live` —
    /// reconciliation after a discovery-watch gap, during which `Removed`
    /// events may have been lost. Returns how many entries were pruned.
    pub(crate) async fn retain_instances(
        &self,
        known: &std::collections::HashSet<(String, u64)>,
        live: &std::collections::HashSet<(String, u64)>,
    ) -> usize {
        let mut map = self.by_instance.write().await;
        let before = map.len();
        map.retain(|key, _| live.contains(key) || !known.contains(key));
        let pruned = before - map.len();
        if pruned > 0 {
            self.generation.fetch_add(1, Ordering::Relaxed);
        }
        pruned
    }

    /// Live discovery-registered worker instances per model. Only models
    /// with at least one instance appear — the caller is responsible for
    /// remembering models whose pool drained to zero (this registry no
    /// longer knows they exist).
    pub async fn worker_counts(&self) -> HashMap<String, u32> {
        let mut counts: HashMap<String, u32> = HashMap::new();
        for model_id in self.by_instance.read().await.values() {
            *counts.entry(model_id.clone()).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolution_requires_mdc_and_invalidates_cached_identity() {
        let registry = ModelRegistry::default();
        let mut cache = ModelIdentityCache::default();
        assert!(registry.resolve("backend", 7, &mut cache).await.is_none());

        registry.register("backend".into(), 7, "llama".into()).await;
        assert_eq!(
            registry
                .resolve("backend", 7, &mut cache)
                .await
                .expect("registered")
                .model_id
                .as_ref(),
            "llama"
        );

        assert!(registry.remove("backend", 7).await);
        assert!(
            registry.resolve("backend", 7, &mut cache).await.is_none(),
            "removal must invalidate the cached identity"
        );
    }

    #[tokio::test]
    async fn wait_registered_unblocks_on_registration_and_cancel() {
        let registry = Arc::new(ModelRegistry::default());
        let cancel = CancellationToken::new();

        let waiter = {
            let registry = registry.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move { registry.wait_registered("backend", 7, &cancel).await })
        };
        registry.register("backend".into(), 7, "llama".into()).await;
        let model = waiter.await.unwrap().expect("registration arrived");
        assert_eq!(model.model_id.as_ref(), "llama");
        assert_eq!(model.model_key, wire::model_id_to_key("llama"));

        let waiter = {
            let registry = registry.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move { registry.wait_registered("backend", 8, &cancel).await })
        };
        cancel.cancel();
        assert!(waiter.await.unwrap().is_none(), "cancel unblocks with None");
    }

    #[tokio::test]
    async fn retain_prunes_only_known_stale_instances() {
        let registry = ModelRegistry::default();
        registry.register("a".into(), 1, "m1".into()).await;
        registry.register("a".into(), 2, "m1".into()).await;
        registry.register("other-ns".into(), 9, "m2".into()).await;

        // This watcher only ever saw ("a", 1) and ("a", 2); ("a", 2) is gone
        // from the live list. ("other-ns", 9) belongs to another watcher and
        // must survive even though it is absent from `live`.
        let known = [("a".to_string(), 1), ("a".to_string(), 2)].into();
        let live = [("a".to_string(), 1)].into();
        assert_eq!(registry.retain_instances(&known, &live).await, 1);

        assert!(registry.resolve_once("a", 1).await.is_some());
        assert!(registry.resolve_once("a", 2).await.is_none());
        assert!(registry.resolve_once("other-ns", 9).await.is_some());
    }
}
