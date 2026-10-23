// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `LoadUpdate` windows from the Relay's serving-load aggregate.

use std::time::Duration;

use super::identity::{pool_id_to_wire, producer_to_wire, relay_identity_to_wire, unix_timestamp};
use super::load::window_ms;
use super::protocol as proto;
use crate::pool_relay::identity::PoolRelayIdentity;
use crate::pool_relay::load::{Coverage, PoolLoadSnapshot};
use crate::pool_relay::serving_load::{LoadSnapshot, ModelLoad, PoolLoad};

pub(super) fn load_update(
    relay: PoolRelayIdentity,
    snapshot: &LoadSnapshot,
    window: Duration,
    sequence: u64,
) -> proto::LoadUpdate {
    let observed_ms = unix_timestamp::<1_000>();
    proto::LoadUpdate {
        protocol_version: proto::RELAY_PROTOCOL_VERSION,
        relay: Some(relay_identity_to_wire(relay)),
        window_sequence: sequence,
        observed_ms,
        window_ms: window_ms(window),
        pools: snapshot
            .pools
            .iter()
            .map(|pool| pool_to_wire(pool, observed_ms))
            .collect(),
        models: snapshot.models.iter().map(model_to_wire).collect(),
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

/// The aggregate when complete, and the view's status.
fn coverage_to_wire<T: Copy>(coverage: Coverage<T>) -> (Option<T>, proto::DataStatus) {
    match coverage {
        Coverage::Complete(value) => (Some(value), proto::DataStatus::Complete),
        Coverage::Partial => (None, proto::DataStatus::Degraded),
        Coverage::Missing => (None, proto::DataStatus::Unavailable),
    }
}

fn pool_to_wire(pool: &PoolLoad, observed_ms: u64) -> proto::PoolLoad {
    let (gauges, status) = coverage_to_wire(pool.scheduler.coverage);
    proto::PoolLoad {
        producer: Some(producer_to_wire(pool.producer)),
        kv: Some(kv_usage_to_wire(pool.kv, observed_ms)),
        load: Some(proto::LoadView {
            requests: None,
            tokens: Some(proto::TokenLoadStats {
                active_prefill_tokens: gauges.map(|gauges| gauges.active_prefill_tokens),
                active_decode_blocks: gauges.map(|gauges| gauges.active_decode_blocks),
                ..Default::default()
            }),
            status: status as i32,
            source_observed_ms: pool.scheduler.source_observed_ms,
        }),
        deployment: Some(proto::PoolDeploymentStatus {
            live_workers: pool.deployment.map(|deployment| deployment.live_workers),
            max_concurrency: pool
                .deployment
                .and_then(|deployment| deployment.max_concurrency),
        }),
    }
}

fn kv_usage_to_wire(snapshot: Option<PoolLoadSnapshot>, observed_ms: u64) -> proto::KvUsage {
    let Some(snapshot) = snapshot else {
        return proto::KvUsage {
            status: proto::DataStatus::Unavailable as i32,
            ..Default::default()
        };
    };
    let status = if snapshot.kv_observed_ranks == 0 {
        proto::DataStatus::Unavailable
    } else if snapshot.has_degraded_coverage() {
        proto::DataStatus::Degraded
    } else {
        proto::DataStatus::Complete
    };
    proto::KvUsage {
        kv_used_blocks: snapshot.kv_used_blocks.unwrap_or_default(),
        total_kv_blocks: snapshot.total_kv_blocks.unwrap_or_default(),
        kv_observed_ranks: snapshot.kv_observed_ranks.try_into().unwrap_or(u32::MAX),
        kv_expected_ranks: snapshot.kv_expected_ranks.try_into().unwrap_or(u32::MAX),
        status: status as i32,
        observed_ms: if snapshot.kv_observed_ranks == 0 {
            0
        } else {
            observed_ms
        },
    }
}

fn model_to_wire(model: &ModelLoad) -> proto::ModelLoad {
    let (gauges, status) = coverage_to_wire(model.gauges);
    let totals = model.totals;
    proto::ModelLoad {
        namespace: model.namespace.clone(),
        canonical_model_id: model.model.clone(),
        load: Some(proto::LoadView {
            requests: Some(proto::RequestLifecycleStats {
                requests_started_total: totals.requests_started,
                requests_completed_total: totals.requests_completed,
                requests_failed_total: totals.requests_failed,
                requests_cancelled_total: totals.requests_cancelled,
                requests_awaiting_first_token: gauges
                    .map(|gauges| gauges.requests_awaiting_first_token),
                requests_generating: gauges.map(|gauges| gauges.requests_generating),
            }),
            tokens: Some(proto::TokenLoadStats {
                awaiting_first_token_input_tokens: gauges
                    .map(|gauges| gauges.awaiting_first_token_input_tokens),
                inflight_input_tokens: gauges.map(|gauges| gauges.inflight_input_tokens),
                input_tokens_total: Some(totals.input_tokens),
                output_tokens_total: Some(totals.output_tokens),
                ..Default::default()
            }),
            status: status as i32,
            source_observed_ms: model.source_observed_ms,
        }),
        deployment: Some(proto::ModelDeploymentStatus {
            expected_frontends: model.expected_frontends,
            observed_frontends: model.observed_frontends,
            ready_frontends: model.ready_frontends,
            serving_pools: model
                .serving_pools
                .iter()
                .copied()
                .map(pool_id_to_wire)
                .collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState, ProducerIdentity};

    use super::*;
    use crate::frontend_load::{RequestGauges, RequestTotals};
    use crate::pool_relay::load::{PoolDeployment, PoolSchedulerLoad, SchedulerGauges};

    fn producer(seed: u8) -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32)).unwrap().format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([seed; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                ClusterId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    fn model(gauges: Coverage<RequestGauges>) -> ModelLoad {
        ModelLoad {
            namespace: "ns".to_string(),
            model: "llama".to_string(),
            gauges,
            totals: RequestTotals {
                requests_started: 9,
                output_tokens: 70,
                ..RequestTotals::default()
            },
            source_observed_ms: 5,
            expected_frontends: 2,
            observed_frontends: 1,
            ready_frontends: 1,
            serving_pools: vec![producer(1).pool_id()],
        }
    }

    #[test]
    fn window_carries_every_pool_and_follows_presence_rules() {
        let snapshot = LoadSnapshot {
            pools: vec![
                PoolLoad {
                    producer: producer(1),
                    kv: None,
                    scheduler: PoolSchedulerLoad {
                        coverage: Coverage::Complete(SchedulerGauges {
                            active_decode_blocks: 12,
                            active_prefill_tokens: 34,
                        }),
                        source_observed_ms: 4,
                    },
                    deployment: Some(PoolDeployment {
                        live_workers: 2,
                        max_concurrency: None,
                    }),
                },
                // A pool without serving data is still listed.
                PoolLoad {
                    producer: producer(2),
                    kv: None,
                    scheduler: PoolSchedulerLoad {
                        coverage: Coverage::Missing,
                        source_observed_ms: 0,
                    },
                    deployment: None,
                },
            ],
            models: vec![model(Coverage::Partial)],
        };
        let update = load_update(
            PoolRelayIdentity::new(1, 2),
            &snapshot,
            Duration::from_millis(500),
            3,
        );
        proto::validate_load_update(&update).unwrap();
        assert_eq!((update.window_sequence, update.window_ms), (3, 500));
        for pool in &update.pools {
            proto::validate_pool_load(pool).unwrap();
        }

        let complete = update.pools[0].load.as_ref().unwrap();
        assert_eq!(complete.status, proto::DataStatus::Complete as i32);
        let tokens = complete.tokens.as_ref().unwrap();
        assert_eq!(
            (tokens.active_decode_blocks, tokens.active_prefill_tokens),
            (Some(12), Some(34))
        );
        assert_eq!(
            update.pools[0].deployment,
            Some(proto::PoolDeploymentStatus {
                live_workers: Some(2),
                max_concurrency: None,
            })
        );
        let unavailable = update.pools[1].load.as_ref().unwrap();
        assert_eq!(unavailable.status, proto::DataStatus::Unavailable as i32);
        assert_eq!(
            unavailable.tokens.as_ref().unwrap().active_decode_blocks,
            None
        );
        assert_eq!(
            update.pools[1].deployment,
            Some(proto::PoolDeploymentStatus::default())
        );

        let degraded = &update.models[0];
        proto::validate_model_load(degraded).unwrap();
        let load = degraded.load.as_ref().unwrap();
        assert_eq!(load.status, proto::DataStatus::Degraded as i32);
        let requests = load.requests.as_ref().unwrap();
        // Totals survive partial coverage; gauges do not.
        assert_eq!(requests.requests_started_total, 9);
        assert_eq!(requests.requests_awaiting_first_token, None);
        assert_eq!(load.tokens.as_ref().unwrap().output_tokens_total, Some(70));
        assert_eq!(degraded.deployment.as_ref().unwrap().serving_pools.len(), 1);

        let complete = model_to_wire(&model(Coverage::Complete(RequestGauges {
            requests_awaiting_first_token: 3,
            ..RequestGauges::default()
        })));
        proto::validate_model_load(&complete).unwrap();
        let requests = complete.load.as_ref().unwrap().requests.as_ref().unwrap();
        assert_eq!(requests.requests_awaiting_first_token, Some(3));
        assert_eq!(requests.requests_generating, Some(0));
    }
    #[test]
    fn kv_coverage_is_independent_of_scheduler_load() {
        let mut snapshot = PoolLoadSnapshot {
            producer: producer(1),
            kv_used_blocks: Some(u64::MAX),
            total_kv_blocks: Some(u64::MAX),
            kv_observed_ranks: 2,
            kv_capacity_ranks: 2,
            kv_expected_ranks: 2,
        };
        let complete = kv_usage_to_wire(Some(snapshot), 42);
        proto::validate_kv_usage(&complete).unwrap();
        assert_eq!(complete.status, proto::DataStatus::Complete as i32);
        assert_eq!(
            (complete.kv_used_blocks, complete.total_kv_blocks),
            (u64::MAX, u64::MAX)
        );
        assert_eq!(complete.observed_ms, 42);

        let pool = PoolLoad {
            producer: producer(1),
            scheduler: PoolSchedulerLoad {
                coverage: Coverage::Missing,
                source_observed_ms: 0,
            },
            kv: Some(snapshot),
            deployment: None,
        };
        let wire = pool_to_wire(&pool, 42);
        proto::validate_pool_load(&wire).unwrap();
        assert_eq!(
            wire.load.as_ref().unwrap().status,
            proto::DataStatus::Unavailable as i32
        );
        assert_eq!(
            wire.kv.as_ref().unwrap().status,
            proto::DataStatus::Complete as i32
        );

        snapshot.kv_observed_ranks = 1;
        snapshot.kv_used_blocks = None;
        let partial = kv_usage_to_wire(Some(snapshot), 43);
        proto::validate_kv_usage(&partial).unwrap();
        assert_eq!(partial.status, proto::DataStatus::Degraded as i32);
        assert_eq!(partial.kv_used_blocks, 0);
        assert_eq!(partial.total_kv_blocks, u64::MAX);

        snapshot.kv_observed_ranks = 2;
        snapshot.kv_used_blocks = Some(0);
        snapshot.kv_capacity_ranks = 1;
        snapshot.total_kv_blocks = None;
        let unknown_capacity = kv_usage_to_wire(Some(snapshot), 44);
        proto::validate_kv_usage(&unknown_capacity).unwrap();
        assert_eq!(unknown_capacity.status, proto::DataStatus::Degraded as i32);

        snapshot.kv_observed_ranks = 0;
        snapshot.kv_used_blocks = None;
        let missing = kv_usage_to_wire(Some(snapshot), 45);
        proto::validate_kv_usage(&missing).unwrap();
        assert_eq!(missing.status, proto::DataStatus::Unavailable as i32);
        assert_eq!(missing.observed_ms, 0);
        proto::validate_kv_usage(&kv_usage_to_wire(None, 45)).unwrap();
    }
}
