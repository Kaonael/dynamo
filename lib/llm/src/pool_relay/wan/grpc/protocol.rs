// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned WAN contract for cluster-local Relay pool publications.

pub mod v1 {
    #![allow(clippy::all)]
    tonic::include_proto!("dynamo.poolrelay.v1");
}

/// Compact transport encodings and validation shared by producers and consumers.
pub mod wire;

mod errors;
pub use errors::{RELAY_ERROR_REASON_HEADER, relay_error_reason};

pub use v1::{
    AdapterReadiness, BaseModelTarget, CkfFormat, DigestIdentity, DynamoEndpointId, GetInfoRequest,
    IdentitySource, IndexerDomainId, KvIndexUpdate, KvIndexUpdateKind, KvQueryHashFormat,
    KvQuerySemantics, LoraModelTarget, ModelRegistration, ModelTarget, PoolCatalogSnapshot,
    PoolCatalogUpdate, PoolDescriptor, PoolId, ProducerIdentity, ReadinessState, ReadinessUpdate,
    RelayErrorReason, RelayIdentity, RelayInfo, TopologyEntry, TopologyMember, WatchCatalogRequest,
    WatchKvIndexRequest, WatchReadinessRequest, WorkerRole, pool_relay_client::PoolRelayClient,
};
pub use v1::{
    DataStatus, KvUsage, LoadUpdate, LoadView, ModelDeploymentStatus, ModelLoad,
    PoolDeploymentStatus, PoolLoad, RequestLifecycleStats, TokenLoadStats, WatchLoadRequest,
};

pub use v1::pool_relay_server::{PoolRelay, PoolRelayServer};
pub use wire::{
    ProducerKey, WireIdentityError, validate_ckf_format, validate_contract_marker,
    validate_endpoint_id, validate_model_registration, validate_pool_descriptor, validate_pool_id,
    validate_producer_identity, validate_protocol_envelope, validate_query_semantics,
    validate_topology_entry, validate_worker_roles,
};
pub use wire::{validate_kv_usage, validate_load_update, validate_model_load, validate_pool_load};

/// Compatibility major; additive v1 changes do not increment this value.
pub const RELAY_PROTOCOL_VERSION: u32 = 1;
/// Marker carried by every top-level message for the Relay WAN v1 contract.
pub const RELAY_CONTRACT_MARKER: u32 = 0x4B56_5231;
/// Current composition of cache semantics, routing scope, and cluster identity.
pub const POOL_IDENTITY_VERSION: u32 = 1;

/// Descriptor set used by the WAN reflection service.
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/relay_descriptor.bin"));

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use prost::Message as _;

    use super::*;

    fn relay_identity() -> RelayIdentity {
        RelayIdentity {
            drt_instance_id: 17,
            relay_incarnation: 23,
        }
    }

    fn pool_id() -> PoolId {
        PoolId {
            identity_version: POOL_IDENTITY_VERSION,
            indexer_domain: Some(IndexerDomainId {
                cache_semantics: Some(DigestIdentity {
                    digest: Bytes::from_static(&[1; 16]),
                    source: IdentitySource::Explicit as i32,
                }),
                routing_scope: Some(DigestIdentity {
                    digest: Bytes::from_static(&[2; 16]),
                    source: IdentitySource::DefaultDerived as i32,
                }),
            }),
            cluster_id: 7,
        }
    }

    fn producer() -> ProducerIdentity {
        ProducerIdentity {
            pool_id: Some(pool_id()),
            producer_incarnation: 23,
            layout_generation: 5,
            ckf_format: Some(CkfFormat {
                format_version: 1,
                seed: 42,
                bucket_count: 1 << 10,
                fingerprint_bits: 16,
                slots_per_bucket: 4,
            }),
        }
    }

    // Independent schemas model additive evolution without changing the v1 package/marker.
    #[derive(Clone, PartialEq, prost::Message)]
    struct ExtendedInfoRequest {
        #[prost(fixed32, tag = "127")]
        contract_marker: u32,
        #[prost(string, optional, tag = "100")]
        diagnostic_label: Option<String>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct ExtendedRelayInfo {
        #[prost(uint32, tag = "1")]
        protocol_version: u32,
        #[prost(message, optional, tag = "2")]
        relay: Option<RelayIdentity>,
        #[prost(fixed32, tag = "127")]
        contract_marker: u32,
        #[prost(string, optional, tag = "100")]
        diagnostic_label: Option<String>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct ExtendedDescriptor {
        #[prost(message, optional, tag = "1")]
        producer: Option<ProducerIdentity>,
        #[prost(string, optional, tag = "100")]
        diagnostic_label: Option<String>,
    }

    #[test]
    fn additive_v1_fields_work_in_both_directions_without_entering_identity() {
        let future_request = ExtendedInfoRequest {
            contract_marker: RELAY_CONTRACT_MARKER,
            diagnostic_label: Some("future client".into()),
        };
        let old_request =
            GetInfoRequest::decode(future_request.encode_to_vec().as_slice()).unwrap();
        validate_contract_marker(old_request.contract_marker).unwrap();
        let new_request =
            ExtendedInfoRequest::decode(old_request.encode_to_vec().as_slice()).unwrap();
        assert_eq!(new_request.contract_marker, RELAY_CONTRACT_MARKER);
        assert_eq!(new_request.diagnostic_label, None);

        let future_info = ExtendedRelayInfo {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity()),
            contract_marker: RELAY_CONTRACT_MARKER,
            diagnostic_label: Some("future server".into()),
        };
        let old_info = RelayInfo::decode(future_info.encode_to_vec().as_slice()).unwrap();
        validate_protocol_envelope(old_info.protocol_version, old_info.contract_marker).unwrap();
        assert_eq!(old_info.relay, future_info.relay);
        let new_info = ExtendedRelayInfo::decode(old_info.encode_to_vec().as_slice()).unwrap();
        assert_eq!(new_info.protocol_version, future_info.protocol_version);
        assert_eq!(new_info.contract_marker, future_info.contract_marker);
        assert_eq!(new_info.relay, future_info.relay);
        assert_eq!(new_info.diagnostic_label, None);

        let future_descriptor = ExtendedDescriptor {
            producer: Some(producer()),
            diagnostic_label: Some("not a generation dimension".into()),
        };
        let old_descriptor =
            PoolDescriptor::decode(future_descriptor.encode_to_vec().as_slice()).unwrap();
        let key = ProducerKey::try_from(future_descriptor.producer.as_ref().unwrap()).unwrap();
        assert_eq!(
            ProducerKey::try_from(old_descriptor.producer.as_ref().unwrap()).unwrap(),
            key
        );
        let new_descriptor =
            ExtendedDescriptor::decode(old_descriptor.encode_to_vec().as_slice()).unwrap();
        assert_eq!(
            ProducerKey::try_from(new_descriptor.producer.as_ref().unwrap()).unwrap(),
            key
        );
        assert_eq!(new_descriptor.diagnostic_label, None);
    }

    #[test]
    fn producer_key_fences_every_v1_identity_dimension() {
        let original = producer();
        let key = ProducerKey::try_from(&original).unwrap();
        let mutations: &[fn(&mut ProducerIdentity)] = &[
            |p| p.pool_id.as_mut().unwrap().identity_version += 1,
            |p| p.pool_id.as_mut().unwrap().cluster_id += 1,
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .cache_semantics
                    .as_mut()
                    .unwrap()
                    .digest = Bytes::from_static(&[8; 16])
            },
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .routing_scope
                    .as_mut()
                    .unwrap()
                    .digest = Bytes::from_static(&[9; 16])
            },
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .cache_semantics
                    .as_mut()
                    .unwrap()
                    .source = IdentitySource::DefaultDerived as i32
            },
            |p| {
                p.pool_id
                    .as_mut()
                    .unwrap()
                    .indexer_domain
                    .as_mut()
                    .unwrap()
                    .routing_scope
                    .as_mut()
                    .unwrap()
                    .source = IdentitySource::Explicit as i32
            },
            |p| p.producer_incarnation += 1,
            |p| p.layout_generation += 1,
            |p| p.ckf_format.as_mut().unwrap().format_version += 1,
            |p| p.ckf_format.as_mut().unwrap().seed += 1,
            |p| p.ckf_format.as_mut().unwrap().bucket_count *= 2,
            |p| p.ckf_format.as_mut().unwrap().fingerprint_bits += 1,
            |p| p.ckf_format.as_mut().unwrap().slots_per_bucket += 1,
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut changed = original.clone();
            mutate(&mut changed);
            assert_ne!(
                ProducerKey::try_from(&changed),
                Ok(key),
                "key dimension {index} was ignored"
            );
        }
    }

    #[test]
    fn unknown_semantics_are_distinguished_from_malformed_known_values() {
        for value in [99, -1] {
            let error = validate_query_semantics(&KvQuerySemantics {
                kv_block_size: 64,
                hash_format: value,
            })
            .unwrap_err();
            assert!(error.is_unsupported());
            assert!(
                validate_worker_roles(&[value])
                    .unwrap_err()
                    .is_unsupported()
            );
        }
        assert!(
            !validate_query_semantics(&KvQuerySemantics {
                kv_block_size: 64,
                hash_format: 0,
            })
            .unwrap_err()
            .is_unsupported()
        );
        assert!(!validate_worker_roles(&[0]).unwrap_err().is_unsupported());
        let known = WorkerRole::Prefill as i32;
        assert!(
            !validate_worker_roles(&[known, known])
                .unwrap_err()
                .is_unsupported()
        );

        // A future target alternative at tag 3 decodes as an unset oneof in v1.
        let target = ModelTarget::decode(&[0x1a, 0x00][..]).unwrap();
        let registration = ModelRegistration {
            canonical_model_id: "future".into(),
            target: Some(target),
            aliases: vec![],
        };
        assert_eq!(
            validate_model_registration(&registration),
            Err(WireIdentityError::UnsupportedModelTarget)
        );
        let topology = TopologyEntry {
            namespace: "ns".into(),
            canonical_model_id: "model".into(),
            state: 99,
            ..Default::default()
        };
        assert!(
            validate_topology_entry(&topology)
                .unwrap_err()
                .is_unsupported()
        );
        assert!(
            !validate_topology_entry(&TopologyEntry {
                state: ReadinessState::Unknown as i32,
                ..topology
            })
            .unwrap_err()
            .is_unsupported()
        ); // Missing members, not an unknown readiness enum.
        validate_query_semantics(&KvQuerySemantics {
            kv_block_size: 64,
            hash_format: KvQueryHashFormat::DynamoStandardV1 as i32,
        })
        .unwrap();
    }

    #[test]
    fn catalog_round_trips_typed_identity_endpoint_and_lora_registration() {
        let update = PoolCatalogUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity()),
            revision: 9,
            snapshot: Some(PoolCatalogSnapshot {
                pools: vec![PoolDescriptor {
                    producer: Some(producer()),
                    serving_endpoint: Some(DynamoEndpointId {
                        namespace: "ns".into(),
                        component: "backend".into(),
                        endpoint: "generate".into(),
                    }),
                    registrations: vec![ModelRegistration {
                        canonical_model_id: "llama-lora".into(),
                        target: Some(ModelTarget {
                            target: Some(v1::model_target::Target::Lora(LoraModelTarget {
                                base_model: "llama".into(),
                                adapter: "llama-lora".into(),
                            })),
                        }),
                        aliases: vec!["chat".into()],
                    }],
                    query_semantics: Some(KvQuerySemantics {
                        kv_block_size: 64,
                        hash_format: KvQueryHashFormat::DynamoStandardV1 as i32,
                    }),
                    pool_roles: vec![WorkerRole::Decode as i32],
                }],
            }),
            contract_marker: RELAY_CONTRACT_MARKER,
        };

        let decoded = PoolCatalogUpdate::decode(update.encode_to_vec().as_slice())
            .expect("catalog update must decode");
        assert_eq!(decoded, update);
        let descriptor = &decoded.snapshot.as_ref().expect("snapshot").pools[0];
        validate_pool_descriptor(descriptor).expect("pool descriptor");
    }

    fn load_update() -> LoadUpdate {
        let tokens = |prefill, decode| TokenLoadStats {
            active_prefill_tokens: prefill,
            active_decode_blocks: decode,
            ..Default::default()
        };
        LoadUpdate {
            protocol_version: RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity()),
            window_sequence: 1,
            observed_ms: 1_000,
            window_ms: 1_000,
            pools: vec![PoolLoad {
                producer: Some(producer()),
                load: Some(LoadView {
                    requests: None,
                    tokens: Some(tokens(Some(3), Some(4))),
                    status: DataStatus::Complete as i32,
                    source_observed_ms: 900,
                }),
                deployment: Some(PoolDeploymentStatus::default()),
                kv: Some(KvUsage {
                    kv_used_blocks: 0,
                    total_kv_blocks: 100,
                    kv_observed_ranks: 2,
                    kv_expected_ranks: 2,
                    status: DataStatus::Complete as i32,
                    observed_ms: 1_000,
                }),
            }],
            models: vec![ModelLoad {
                namespace: "ns".into(),
                canonical_model_id: "llama".into(),
                load: Some(LoadView {
                    requests: Some(RequestLifecycleStats {
                        requests_started_total: 5,
                        ..Default::default()
                    }),
                    tokens: Some(TokenLoadStats {
                        input_tokens_total: Some(10),
                        output_tokens_total: Some(20),
                        ..Default::default()
                    }),
                    status: DataStatus::Degraded as i32,
                    source_observed_ms: 900,
                }),
                deployment: Some(ModelDeploymentStatus {
                    serving_pools: vec![pool_id()],
                    ..Default::default()
                }),
            }],
            contract_marker: RELAY_CONTRACT_MARKER,
        }
    }

    #[test]
    fn unified_load_roundtrips_with_independent_statuses() {
        let update = load_update();
        let decoded = LoadUpdate::decode(update.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded, update);
        validate_load_update(&decoded).unwrap();
        validate_pool_load(&decoded.pools[0]).unwrap();
        validate_model_load(&decoded.models[0]).unwrap();
        let idle = decoded.pools[0].kv.as_ref().unwrap();
        assert_eq!(idle.status, DataStatus::Complete as i32);
        assert_eq!(idle.kv_used_blocks, 0);

        let mut pool = decoded.pools[0].clone();
        pool.kv = None;
        assert!(validate_pool_load(&pool).is_err());
        pool.kv = Some(KvUsage {
            status: 99,
            ..Default::default()
        });
        assert!(validate_pool_load(&pool).unwrap_err().is_unsupported());
        pool.kv = Some(KvUsage {
            status: DataStatus::Complete as i32,
            ..Default::default()
        });
        assert!(validate_pool_load(&pool).is_err());
        pool.kv = Some(KvUsage {
            status: DataStatus::Unavailable as i32,
            ..Default::default()
        });
        validate_pool_load(&pool).unwrap();
        assert_eq!(
            pool.load.as_ref().unwrap().status,
            DataStatus::Complete as i32
        );
    }

    /// A bad envelope or a duplicate key invalidates the whole window.
    #[test]
    fn serving_load_window_rejects_bad_envelopes_and_duplicate_keys() {
        let update = load_update();
        validate_load_update(&update).unwrap();

        let mut wrong_version = update.clone();
        wrong_version.protocol_version += 1;
        assert_eq!(
            validate_load_update(&wrong_version),
            Err(WireIdentityError::ProtocolVersion(2))
        );
        let mut no_relay = update.clone();
        no_relay.relay = None;
        assert!(validate_load_update(&no_relay).is_err());
        let mut anonymous = update.clone();
        anonymous.pools[0].producer = None;
        assert!(validate_load_update(&anonymous).is_err());

        let mut duplicate_pool = update.clone();
        duplicate_pool.pools.push(update.pools[0].clone());
        assert_eq!(
            validate_load_update(&duplicate_pool),
            Err(WireIdentityError::DuplicateLoadPool)
        );
        let mut duplicate_model = update.clone();
        duplicate_model.models.push(update.models[0].clone());
        assert!(matches!(
            validate_load_update(&duplicate_model),
            Err(WireIdentityError::DuplicateLoadModel { .. })
        ));
    }

    /// An entry this consumer cannot interpret is quarantined on its own.
    #[test]
    fn serving_load_entries_with_unknown_status_or_identity_are_unsupported() {
        let update = load_update();
        for status in [99, -1] {
            let mut pool = update.pools[0].clone();
            pool.load.as_mut().unwrap().status = status;
            assert!(validate_pool_load(&pool).unwrap_err().is_unsupported());
        }
        let mut future_pool = update.clone();
        future_pool.pools[0]
            .producer
            .as_mut()
            .unwrap()
            .pool_id
            .as_mut()
            .unwrap()
            .identity_version += 1;
        validate_load_update(&future_pool).unwrap();
        assert!(
            validate_pool_load(&future_pool.pools[0])
                .unwrap_err()
                .is_unsupported()
        );
    }

    /// Gauges are set exactly when COMPLETE, totals always, and the other view
    /// kind's fields never; a malformed entry is rejected, not unsupported.
    #[test]
    fn serving_load_entries_enforce_presence_rules() {
        let update = load_update();
        validate_pool_load(&update.pools[0]).unwrap();
        validate_model_load(&update.models[0]).unwrap();

        let model_violations: &[fn(&mut ModelLoad)] = &[
            |model| model.load.as_mut().unwrap().status = DataStatus::Unspecified as i32,
            // DEGRADED gauges would read as a partial sum.
            |model| {
                let requests = model.load.as_mut().unwrap().requests.as_mut().unwrap();
                requests.requests_generating = Some(1);
            },
            // COMPLETE requires every gauge.
            |model| model.load.as_mut().unwrap().status = DataStatus::Complete as i32,
            // Totals are always set; pool fields never.
            |model| {
                let tokens = model.load.as_mut().unwrap().tokens.as_mut().unwrap();
                tokens.output_tokens_total = None;
            },
            |model| {
                let tokens = model.load.as_mut().unwrap().tokens.as_mut().unwrap();
                tokens.active_decode_blocks = Some(1);
            },
            |model| {
                model
                    .deployment
                    .as_mut()
                    .unwrap()
                    .serving_pools
                    .push(pool_id())
            },
        ];
        for (index, violate) in model_violations.iter().enumerate() {
            let mut model = update.models[0].clone();
            violate(&mut model);
            let error = validate_model_load(&model).unwrap_err();
            assert!(
                !error.is_unsupported(),
                "model violation {index}: {error:?}"
            );
        }

        let mut incomplete = update.pools[0].clone();
        incomplete.load.as_mut().unwrap().status = DataStatus::Degraded as i32;
        assert!(matches!(
            validate_pool_load(&incomplete),
            Err(WireIdentityError::LoadViewField("active_prefill_tokens"))
        ));
        let mut with_requests = update.pools[0].clone();
        with_requests.load.as_mut().unwrap().requests = Some(RequestLifecycleStats::default());
        assert!(matches!(
            validate_pool_load(&with_requests),
            Err(WireIdentityError::LoadViewField(_))
        ));
    }
}
