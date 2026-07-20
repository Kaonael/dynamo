// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DC-local frontend (KV-router) liveness, observed through the discovery
//! plane rather than an HTTP probe.
//!
//! Each frontend's embedded KV router registers a `router-discovery` endpoint
//! in its Dynamo namespace, in a `DynamoWorkerMetadata` CR owned by the
//! frontend pod — so the endpoint disappears the moment that pod dies. The
//! router-discovery watcher maintains the live-instance set per namespace; the
//! MDC watcher records which model each namespace serves. Their join answers
//! "is model M's frontend up in this DC?" with zero configuration and no probe
//! URL. This is the frontend counterpart to the worker liveness the model
//! registry already derives from discovery.

use std::collections::{HashMap, HashSet};

use tokio::sync::RwLock;

#[derive(Default)]
pub struct FrontendHealth {
    /// namespace → live `router-discovery` instance ids. A namespace is "up"
    /// while this set is non-empty; tracking ids (not a bool) keeps a frontend
    /// with multiple replicas up until the last one leaves.
    routers: RwLock<HashMap<String, HashSet<u64>>>,
    /// model_id → namespaces observed serving it (from worker MDCs). Sticky:
    /// a namespace that once served a model stays associated, so a frontend
    /// that dies while its workers linger is still attributable. The set is
    /// bounded by the DC's model catalog.
    model_namespaces: RwLock<HashMap<String, HashSet<String>>>,
}

impl FrontendHealth {
    /// Record a live router-discovery endpoint (watcher `Added`).
    pub async fn add_router(&self, namespace: &str, instance_id: u64) {
        self.routers
            .write()
            .await
            .entry(namespace.to_string())
            .or_default()
            .insert(instance_id);
    }

    /// Drop a router-discovery endpoint (watcher `Removed`); prune the
    /// namespace once its last frontend instance is gone.
    pub async fn remove_router(&self, namespace: &str, instance_id: u64) {
        let mut routers = self.routers.write().await;
        if let Some(ids) = routers.get_mut(namespace) {
            ids.remove(&instance_id);
            if ids.is_empty() {
                routers.remove(namespace);
            }
        }
    }

    /// Reconcile one namespace after a discovery-watch gap: keep only the
    /// instances present in `live` (lost `Removed` events would otherwise
    /// report a dead frontend as healthy forever). Returns how many router
    /// instances were pruned.
    pub(crate) async fn retain_routers(
        &self,
        namespace: &str,
        live: &std::collections::HashSet<u64>,
    ) -> usize {
        let mut routers = self.routers.write().await;
        let Some(ids) = routers.get_mut(namespace) else {
            return 0;
        };
        let before = ids.len();
        ids.retain(|id| live.contains(id));
        let pruned = before - ids.len();
        if ids.is_empty() {
            routers.remove(namespace);
        }
        pruned
    }

    /// Associate a model with the namespace serving it (MDC watcher).
    pub async fn associate(&self, model_id: &str, namespace: &str) {
        self.model_namespaces
            .write()
            .await
            .entry(model_id.to_string())
            .or_default()
            .insert(namespace.to_string());
    }

    /// `frontend_healthy` per model: healthy iff some namespace serving the
    /// model has a live frontend. Only models with a known namespace appear —
    /// a model we've never seen a worker for yields no entry, so the router
    /// keeps its cold-start default rather than treating unknown as down.
    pub async fn health_by_model(&self) -> HashMap<String, bool> {
        let model_ns = self.model_namespaces.read().await;
        let routers = self.routers.read().await;
        model_ns
            .iter()
            .map(|(model, namespaces)| {
                let up = namespaces.iter().any(|ns| routers.contains_key(ns));
                (model.clone(), up)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_joins_model_namespace_to_live_routers() {
        let fh = FrontendHealth::default();
        fh.associate("model-a", "namespace-a").await;
        fh.associate("model-b", "namespace-b").await;

        // Only model-a's frontend is up.
        fh.add_router("namespace-a", 1).await;

        let health = fh.health_by_model().await;
        assert_eq!(health.get("model-a"), Some(&true));
        assert_eq!(health.get("model-b"), Some(&false));
    }

    #[tokio::test]
    async fn namespace_stays_up_until_last_replica_leaves() {
        let fh = FrontendHealth::default();
        fh.associate("m", "ns").await;
        fh.add_router("ns", 1).await;
        fh.add_router("ns", 2).await;

        fh.remove_router("ns", 1).await;
        assert_eq!(fh.health_by_model().await.get("m"), Some(&true));

        fh.remove_router("ns", 2).await;
        assert_eq!(fh.health_by_model().await.get("m"), Some(&false));
    }

    #[tokio::test]
    async fn unknown_model_has_no_entry() {
        let fh = FrontendHealth::default();
        fh.add_router("ns", 1).await;
        assert!(fh.health_by_model().await.is_empty());
    }
}
