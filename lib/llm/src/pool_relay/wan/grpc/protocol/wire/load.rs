// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation of `LoadUpdate` windows.
//!
//! [`validate_load_update`] decides whether a window is usable at all;
//! the per-entry validators decide whether one entry is. Any entry error
//! quarantines that entry only; `is_unsupported()` marks the ones a newer
//! producer may legitimately send (an unknown `DataStatus` or pool identity).

use std::collections::HashSet;

use super::super::{DataStatus, KvUsage, LoadUpdate, ModelLoad, PoolLoad, ProducerIdentity};
use super::identity::{
    ProducerKey, WireIdentityError, validate_pool_id, validate_producer_identity,
    validate_protocol_envelope, validate_text,
};

/// Validates the envelope and entry keys. Missing or malformed keys and
/// duplicate keys make the whole window unusable.
pub fn validate_load_update(update: &LoadUpdate) -> Result<(), WireIdentityError> {
    validate_protocol_envelope(update.protocol_version, update.contract_marker)?;
    update
        .relay
        .as_ref()
        .ok_or(WireIdentityError::MissingField("relay identity"))?;
    let mut pools = HashSet::with_capacity(update.pools.len());
    for pool in &update.pools {
        match ProducerKey::try_from(pool_producer(pool)?) {
            Ok(key) => {
                if !pools.insert(key) {
                    return Err(WireIdentityError::DuplicateLoadPool);
                }
            }
            // Quarantined by validate_pool_load.
            Err(error) if error.is_unsupported() => {}
            Err(error) => return Err(error),
        }
    }
    let mut models = HashSet::with_capacity(update.models.len());
    for model in &update.models {
        validate_text("serving load namespace", &model.namespace)?;
        validate_text("serving load canonical model ID", &model.canonical_model_id)?;
        if !models.insert((&model.namespace, &model.canonical_model_id)) {
            return Err(WireIdentityError::DuplicateLoadModel {
                namespace: model.namespace.clone(),
                model: model.canonical_model_id.clone(),
            });
        }
    }
    Ok(())
}

pub fn validate_pool_load(pool: &PoolLoad) -> Result<(), WireIdentityError> {
    validate_producer_identity(pool_producer(pool)?)?;
    validate_kv_usage(
        pool.kv
            .as_ref()
            .ok_or(WireIdentityError::MissingField("pool KV usage"))?,
    )?;
    let load = pool
        .load
        .as_ref()
        .ok_or(WireIdentityError::MissingField("pool load view"))?;
    let complete = validate_status(load.status)?;
    if load.requests.is_some() {
        return Err(WireIdentityError::LoadViewField("pool requests"));
    }
    let tokens = load
        .tokens
        .as_ref()
        .ok_or(WireIdentityError::MissingField("pool token load"))?;
    require_set_exactly_when(
        "active_prefill_tokens",
        tokens.active_prefill_tokens,
        complete,
    )?;
    require_set_exactly_when(
        "active_decode_blocks",
        tokens.active_decode_blocks,
        complete,
    )?;
    for (field, value) in [
        (
            "pool awaiting_first_token_input_tokens",
            tokens.awaiting_first_token_input_tokens,
        ),
        ("pool inflight_input_tokens", tokens.inflight_input_tokens),
        ("pool input_tokens_total", tokens.input_tokens_total),
        ("pool output_tokens_total", tokens.output_tokens_total),
    ] {
        require_set_exactly_when(field, value, false)?;
    }
    pool.deployment
        .as_ref()
        .ok_or(WireIdentityError::MissingField("pool deployment status"))?;
    Ok(())
}

/// KV status is independent of the scheduler view. A zero value is idle
/// only with complete rank/capacity coverage.
pub fn validate_kv_usage(kv: &KvUsage) -> Result<(), WireIdentityError> {
    let complete = validate_status(kv.status)?;
    if kv.kv_observed_ranks > kv.kv_expected_ranks {
        return Err(WireIdentityError::LoadViewField("KV rank coverage"));
    }
    if complete
        && (kv.kv_expected_ranks == 0
            || kv.kv_observed_ranks != kv.kv_expected_ranks
            || kv.total_kv_blocks == 0)
    {
        return Err(WireIdentityError::LoadViewField("complete KV usage"));
    }
    let unavailable = kv.status == DataStatus::Unavailable as i32;
    if unavailable != (kv.kv_observed_ranks == 0) {
        return Err(WireIdentityError::LoadViewField("KV usage status"));
    }
    if unavailable && (kv.kv_used_blocks != 0 || kv.observed_ms != 0) {
        return Err(WireIdentityError::LoadViewField("unavailable KV usage"));
    }
    Ok(())
}

pub fn validate_model_load(model: &ModelLoad) -> Result<(), WireIdentityError> {
    validate_text("serving load namespace", &model.namespace)?;
    validate_text("serving load canonical model ID", &model.canonical_model_id)?;
    let load = model
        .load
        .as_ref()
        .ok_or(WireIdentityError::MissingField("model load view"))?;
    let complete = validate_status(load.status)?;
    let requests = load
        .requests
        .as_ref()
        .ok_or(WireIdentityError::MissingField("model request counts"))?;
    require_set_exactly_when(
        "requests_awaiting_first_token",
        requests.requests_awaiting_first_token,
        complete,
    )?;
    require_set_exactly_when(
        "requests_generating",
        requests.requests_generating,
        complete,
    )?;
    let tokens = load
        .tokens
        .as_ref()
        .ok_or(WireIdentityError::MissingField("model token load"))?;
    require_set_exactly_when(
        "awaiting_first_token_input_tokens",
        tokens.awaiting_first_token_input_tokens,
        complete,
    )?;
    require_set_exactly_when(
        "inflight_input_tokens",
        tokens.inflight_input_tokens,
        complete,
    )?;
    require_set_exactly_when("input_tokens_total", tokens.input_tokens_total, true)?;
    require_set_exactly_when("output_tokens_total", tokens.output_tokens_total, true)?;
    require_set_exactly_when(
        "model active_prefill_tokens",
        tokens.active_prefill_tokens,
        false,
    )?;
    require_set_exactly_when(
        "model active_decode_blocks",
        tokens.active_decode_blocks,
        false,
    )?;
    let deployment = model
        .deployment
        .as_ref()
        .ok_or(WireIdentityError::MissingField("model deployment status"))?;
    for (index, pool_id) in deployment.serving_pools.iter().enumerate() {
        validate_pool_id(pool_id)?;
        // A model is served by a handful of pools; a quadratic scan is cheapest.
        if deployment.serving_pools[..index].contains(pool_id) {
            return Err(WireIdentityError::DuplicateServingPool);
        }
    }
    Ok(())
}

fn pool_producer(pool: &PoolLoad) -> Result<&ProducerIdentity, WireIdentityError> {
    pool.producer
        .as_ref()
        .ok_or(WireIdentityError::MissingField("pool producer"))
}

/// Returns whether the view is COMPLETE.
fn validate_status(status: i32) -> Result<bool, WireIdentityError> {
    match DataStatus::try_from(status) {
        Ok(DataStatus::Unspecified) | Err(_) => Err(WireIdentityError::DataStatus(status)),
        Ok(status) => Ok(status == DataStatus::Complete),
    }
}

fn require_set_exactly_when(
    field: &'static str,
    value: Option<u64>,
    condition: bool,
) -> Result<(), WireIdentityError> {
    if value.is_some() != condition {
        return Err(WireIdentityError::LoadViewField(field));
    }
    Ok(())
}
