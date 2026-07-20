// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The metric backend behind the relay's telemetry: a [`MetricSource`] runs one
//! instant query and returns a label-set + scalar vector. [`PromQlClient`] is
//! the production impl over the Prometheus HTTP API; the trait is the single
//! seam, so the collector can run against any PromQL-compatible backend
//! (Thanos/Mimir/VictoriaMetrics) or a fake in tests — no HTTP required.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

/// One instant-vector sample: its label set and scalar value. Provider-neutral
/// — nothing here is Prometheus-specific.
#[derive(Debug, Clone)]
pub struct Sample {
    pub labels: HashMap<String, String>,
    pub value: f64,
}

/// A backend that answers instant queries with a result vector. The only I/O
/// seam in the telemetry path.
pub trait MetricSource: Send + Sync {
    /// Run one instant query. A query that errors or returns a non-`success`
    /// status yields an error; individual samples whose value is NaN/±Inf or
    /// unparseable are skipped (treated as "no data" by the caller).
    fn instant(&self, query: &str)
    -> impl std::future::Future<Output = Result<Vec<Sample>>> + Send;
}

/// Minimal client for the Prometheus HTTP API (`GET /api/v1/query`).
#[derive(Clone)]
pub struct PromQlClient {
    http: reqwest::Client,
    /// Base URL of the Prometheus server, e.g. `http://prometheus:9090`.
    base: String,
    bearer: Option<String>,
}

impl PromQlClient {
    pub fn new(base: String, bearer: Option<String>, timeout: Duration) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(timeout.max(Duration::from_secs(2)))
            .build()
            .context("building PromQL reqwest client")?;
        // Trim a trailing slash so `{base}/api/v1/query` joins cleanly.
        let base = base.trim_end_matches('/').to_string();
        Ok(Self { http, base, bearer })
    }
}

impl MetricSource for PromQlClient {
    async fn instant(&self, query: &str) -> Result<Vec<Sample>> {
        let url = format!("{}/api/v1/query", self.base);
        let mut req = self.http.get(&url).query(&[("query", query)]);
        if let Some(token) = &self.bearer {
            req = req.bearer_auth(token);
        }
        let body = req
            .send()
            .await
            .with_context(|| format!("PromQL GET {url}"))?
            .error_for_status()
            .with_context(|| format!("non-2xx from PromQL {url}"))?
            .text()
            .await
            .context("reading PromQL body")?;
        parse_instant(&body).with_context(|| format!("parsing PromQL response for {query:?}"))
    }
}

#[derive(Deserialize)]
struct PromResponse {
    status: String,
    #[serde(default)]
    data: Option<PromData>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct PromData {
    #[serde(rename = "resultType")]
    result_type: String,
    result: Vec<PromResult>,
}

#[derive(Deserialize)]
struct PromResult {
    metric: HashMap<String, String>,
    /// `[unix_ts: f64, value: string]`.
    value: (f64, String),
}

/// Parse a Prometheus `/api/v1/query` instant-vector response. Pure; no I/O.
pub fn parse_instant(text: &str) -> Result<Vec<Sample>> {
    let resp: PromResponse = serde_json::from_str(text).context("decoding PromQL JSON")?;
    if resp.status != "success" {
        anyhow::bail!(
            "PromQL status {:?}: {}",
            resp.status,
            resp.error.unwrap_or_default()
        );
    }
    let data = resp.data.context("PromQL success without data")?;
    if data.result_type != "vector" {
        anyhow::bail!("expected vector result, got {:?}", data.result_type);
    }
    let mut out = Vec::with_capacity(data.result.len());
    for r in data.result {
        // Prometheus encodes the scalar as a string ("12.3", "NaN", "+Inf");
        // a NaN/Inf or unparseable value means "no usable datum" → skip.
        match r.value.1.parse::<f64>() {
            Ok(v) if v.is_finite() => out.push(Sample {
                labels: r.metric,
                value: v,
            }),
            _ => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_instant_vector_and_skips_nan() {
        let body = r#"{"status":"success","data":{"resultType":"vector","result":[
            {"metric":{"model":"m1"},"value":[1700000000.0,"12.5"]},
            {"metric":{"model":"m2"},"value":[1700000000.0,"NaN"]}
        ]}}"#;
        let s = parse_instant(body).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].labels.get("model").unwrap(), "m1");
        assert_eq!(s[0].value, 12.5);
    }

    #[test]
    fn errors_on_non_success() {
        let body = r#"{"status":"error","error":"bad query"}"#;
        assert!(parse_instant(body).is_err());
    }

    #[test]
    fn rejects_non_vector() {
        let body = r#"{"status":"success","data":{"resultType":"matrix","result":[]}}"#;
        assert!(parse_instant(body).is_err());
    }
}
