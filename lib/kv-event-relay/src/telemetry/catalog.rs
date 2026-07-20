// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query catalog + per-tick assembly: the single source of the relay's
//! per-(dc, model) telemetry. All metrics come from the DC's in-cluster
//! Prometheus, engine-agnostically off the Dynamo frontend layer
//! (`dynamo_frontend_*`) plus DCGM for GPU utilisation.
//!
//! Attribution: latency / error metrics carry a `model` label directly. The
//! queue, GPU-util and KV-util series do not (DCGM keys by `pod`; the queue and
//! worker gauges key by `worker_type`/`worker_id`), so they are grouped by the
//! per-pod `nvidia.com/dynamo-graph-deployment-name` label (surfaced via
//! `kube_pod_labels` after the KSM allowlist) and translated DGD → model using a
//! resolver query that joins a model-labelled frontend metric to the same pod
//! label.

use std::collections::HashMap;

use dynamo_kv_event_relay_proto::metrics::DcModelMetricsSnapshot;

use super::source::MetricSource;

/// Which `DcModelMetricsSnapshot` field a query populates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    QueueDepth,
    TtftP50,
    TtftP95,
    TtftP99,
    ItlP50,
    ItlP95,
    ItlP99,
    GpuUtil,
    KvUtil,
    ServerErrorRate,
}

/// How a query's result vector is keyed back to a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyBy {
    /// The series carries a `model` label directly.
    Model,
    /// The series is keyed by the DGD label (no `model`); translate via the
    /// resolver's DGD → model map.
    Dgd,
}

/// One catalog entry: the field it fills, its PromQL, and how to key it.
#[derive(Debug, Clone)]
pub struct MetricQuery {
    pub field: Field,
    pub key_by: KeyBy,
    pub query: String,
}

/// The full catalog plus the DGD→model resolver and the DGD label name.
#[derive(Debug, Clone)]
pub struct QueryCatalog {
    pub queries: Vec<MetricQuery>,
    /// PromQL whose result carries both `model` and the DGD label, used to
    /// build the DGD → model map for `KeyBy::Dgd` queries.
    pub dgd_resolver: String,
    /// PromQL label name carrying the DGD (kube_pod_labels-sanitised form of
    /// `nvidia.com/dynamo-graph-deployment-name`).
    pub dgd_label: String,
}

impl Default for QueryCatalog {
    fn default() -> Self {
        Self::with_selector(None)
    }
}

impl QueryCatalog {
    /// `selector` is an extra PromQL label matcher (e.g. `namespace="dc-ams"`)
    /// ANDed into the `kube_pod_labels` join and the DGD resolver. Every
    /// catalog query is DGD-keyed through that join, so this one injection
    /// point scopes the whole catalog. Needed when the queried backend holds
    /// more than this DC's series (shared VictoriaMetrics / one cluster with
    /// per-DC namespaces): without it, same-model DGDs from other DCs
    /// overwrite each other's snapshot fields in arbitrary order.
    pub fn with_selector(selector: Option<&str>) -> Self {
        // `<J>` = the standard pod-label join used for DGD-keyed series.
        let dgd_label = "label_nvidia_com_dynamo_graph_deployment_name".to_string();
        let extra = selector
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| format!(", {s}"))
            .unwrap_or_default();
        let join = format!(
            "* on(namespace,pod) group_left({dgd_label}) kube_pod_labels{{{dgd_label}!=\"\"{extra}}}"
        );
        let ttft = "dynamo_frontend_time_to_first_token_seconds_bucket";
        let itl = "dynamo_frontend_inter_token_latency_seconds_bucket";
        let q = |field, key_by, query: String| MetricQuery {
            field,
            key_by,
            query,
        };
        // Quantiles are keyed by DGD (not the metric's own `model` label):
        // the frontend exposes `model` in inconsistent case across metric
        // families (e.g. histograms lowercase it), which wouldn't match the
        // GR topology allow-list. Joining to the DGD pod-label and resolving
        // DGD→model via `dgd_resolver` yields one canonical model_id for all
        // queries.
        let quant = |hist: &str, qq: f64| {
            format!(
                "histogram_quantile({qq}, sum by(le,{dgd_label})(rate({hist}[1m]) {join})) * 1000"
            )
        };
        Self {
            queries: vec![
                q(
                    Field::QueueDepth,
                    KeyBy::Dgd,
                    format!(
                        "sum by({dgd_label})(dynamo_frontend_router_queue_pending_requests {join})"
                    ),
                ),
                q(Field::TtftP50, KeyBy::Dgd, quant(ttft, 0.50)),
                q(Field::TtftP95, KeyBy::Dgd, quant(ttft, 0.95)),
                q(Field::TtftP99, KeyBy::Dgd, quant(ttft, 0.99)),
                q(Field::ItlP50, KeyBy::Dgd, quant(itl, 0.50)),
                q(Field::ItlP95, KeyBy::Dgd, quant(itl, 0.95)),
                q(Field::ItlP99, KeyBy::Dgd, quant(itl, 0.99)),
                q(
                    Field::GpuUtil,
                    KeyBy::Dgd,
                    // dcgm-exporter is scraped without honorLabels, so its own
                    // pod/namespace (the GPU *consumer* pod) collide with the
                    // target's and land as exported_pod/exported_namespace.
                    // Rewrite them back to pod/namespace so the kube_pod_labels
                    // join (on the consumer pod) matches.
                    format!(
                        "avg by({dgd_label})(label_replace(label_replace(\
                         DCGM_FI_DEV_GPU_UTIL, \"pod\", \"$1\", \"exported_pod\", \"(.+)\"), \
                         \"namespace\", \"$1\", \"exported_namespace\", \"(.+)\") {join})"
                    ),
                ),
                q(
                    Field::KvUtil,
                    KeyBy::Dgd,
                    format!(
                        "100 * sum by({dgd_label})(dynamo_frontend_worker_active_decode_blocks {join}) \
                         / sum by({dgd_label})(dynamo_frontend_model_total_kv_blocks {join})"
                    ),
                ),
                q(
                    Field::ServerErrorRate,
                    KeyBy::Dgd,
                    format!(
                        "sum by({dgd_label})(rate(dynamo_frontend_requests_total{{status=\"error\", \
                         error_type!=\"cancelled\", error_type!=\"not_found\"}}[1m]) {join}) \
                         / sum by({dgd_label})(rate(dynamo_frontend_requests_total[1m]) {join})"
                    ),
                ),
            ],
            dgd_resolver: format!("dynamo_frontend_model_total_kv_blocks {join}"),
            dgd_label,
        }
    }
}

/// Apply one sample's value to the matching field of a snapshot.
fn apply_field(snap: &mut DcModelMetricsSnapshot, field: Field, value: f64) {
    match field {
        Field::QueueDepth => snap.queue_depth = value.max(0.0) as u32,
        Field::TtftP50 => snap.ttft_p50_ms = Some(value as f32),
        Field::TtftP95 => snap.ttft_p95_ms = Some(value as f32),
        Field::TtftP99 => snap.ttft_p99_ms = Some(value as f32),
        Field::ItlP50 => snap.itl_p50_ms = Some(value as f32),
        Field::ItlP95 => snap.itl_p95_ms = Some(value as f32),
        Field::ItlP99 => snap.itl_p99_ms = Some(value as f32),
        Field::GpuUtil => snap.gpu_util_pct = Some(value as f32),
        Field::KvUtil => snap.kv_util_pct = Some(value as f32),
        Field::ServerErrorRate => snap.server_error_rate = Some(value as f32),
    }
}

/// Run the catalog against `source` and assemble one snapshot per model seen.
/// `seqs` holds the per-model, per-relay counter, bumped in place. A failing
/// query is logged and skipped (its field stays absent) rather than failing the
/// whole tick — a single broken series shouldn't blank the others.
pub async fn collect_snapshots<S: MetricSource>(
    source: &S,
    catalog: &QueryCatalog,
    dc_id: &str,
    captured_at_unix_ms: u64,
    seqs: &mut HashMap<String, u64>,
) -> Vec<DcModelMetricsSnapshot> {
    // The queries are independent — run them (and the DGD→model resolver)
    // concurrently so a tick costs one round-trip, not the sum of eleven.
    let (dgd_to_model, results) = tokio::join!(
        resolve_dgd_to_model(source, catalog),
        futures::future::join_all(
            catalog
                .queries
                .iter()
                .map(|mq| async move { (mq, source.instant(&mq.query).await) }),
        )
    );

    let mut by_model: HashMap<String, DcModelMetricsSnapshot> = HashMap::new();
    let ensure = |model: &str, map: &mut HashMap<String, DcModelMetricsSnapshot>| {
        map.entry(model.to_string())
            .or_insert_with(|| DcModelMetricsSnapshot::empty(dc_id, model));
    };

    for (mq, samples) in results {
        let samples = match samples {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(field = ?mq.field, error = %format!("{e:#}"), "PromQL query failed; field skipped");
                continue;
            }
        };
        for s in samples {
            let model = match mq.key_by {
                KeyBy::Model => s.labels.get("model").cloned(),
                KeyBy::Dgd => s
                    .labels
                    .get(&catalog.dgd_label)
                    .and_then(|dgd| dgd_to_model.get(dgd).cloned()),
            };
            let Some(model) = model.filter(|m| !m.is_empty()) else {
                continue;
            };
            ensure(&model, &mut by_model);
            apply_field(by_model.get_mut(&model).unwrap(), mq.field, s.value);
        }
    }

    by_model
        .into_values()
        .map(|mut snap| {
            let seq = seqs.entry(snap.model_id.clone()).or_insert(0);
            *seq = seq.wrapping_add(1);
            snap.seq = *seq;
            snap.captured_at_unix_ms = captured_at_unix_ms;
            snap
        })
        .collect()
}

/// Build the DGD → model map from the resolver query. Returns empty on failure
/// (DGD-keyed queries then contribute nothing this tick, logged by the caller).
async fn resolve_dgd_to_model<S: MetricSource>(
    source: &S,
    catalog: &QueryCatalog,
) -> HashMap<String, String> {
    let mut map = HashMap::new();
    match source.instant(&catalog.dgd_resolver).await {
        Ok(samples) => {
            for s in samples {
                if let (Some(dgd), Some(model)) =
                    (s.labels.get(&catalog.dgd_label), s.labels.get("model"))
                    && !dgd.is_empty()
                    && !model.is_empty()
                {
                    map.insert(dgd.clone(), model.clone());
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "DGD→model resolver query failed; GPU/queue/kv-util skipped this tick");
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::source::Sample;
    use anyhow::Result;

    #[test]
    fn apply_field_sets_each_target() {
        let mut snap = DcModelMetricsSnapshot::empty("dc1", "m");
        apply_field(&mut snap, Field::QueueDepth, 7.0);
        apply_field(&mut snap, Field::TtftP99, 800.0);
        apply_field(&mut snap, Field::GpuUtil, 73.0);
        apply_field(&mut snap, Field::KvUtil, 42.0);
        apply_field(&mut snap, Field::ServerErrorRate, 0.1);
        assert_eq!(snap.queue_depth, 7);
        assert_eq!(snap.ttft_p99_ms, Some(800.0));
        assert_eq!(snap.gpu_util_pct, Some(73.0));
        assert_eq!(snap.kv_util_pct, Some(42.0));
        assert_eq!(snap.server_error_rate, Some(0.1));
    }

    #[test]
    fn default_catalog_has_all_ten_fields() {
        let c = QueryCatalog::default();
        assert_eq!(c.queries.len(), 10);
        assert!(c.dgd_resolver.contains("model_total_kv_blocks"));
        assert!(c.dgd_label.starts_with("label_"));
    }

    /// The selector must land in every query and the resolver — all are
    /// DGD-keyed through the shared `kube_pod_labels` join.
    #[test]
    fn selector_scopes_every_query_and_resolver() {
        let sel = "namespace=\"dc-ams\"";
        let c = QueryCatalog::with_selector(Some(sel));
        assert!(c.dgd_resolver.contains(sel));
        for q in &c.queries {
            assert!(q.query.contains(sel), "missing selector: {}", q.query);
        }
        // Blank / whitespace selectors degrade to the unscoped catalog.
        let unscoped = QueryCatalog::with_selector(Some("  "));
        assert_eq!(unscoped.dgd_resolver, QueryCatalog::default().dgd_resolver);
    }

    /// In-memory `MetricSource`: maps a query string to its canned result
    /// vector, so the whole assembly path (KeyBy resolution, DGD translation,
    /// per-model seq) is exercised without any HTTP.
    struct FakeSource {
        responses: HashMap<String, Vec<Sample>>,
    }

    impl MetricSource for FakeSource {
        async fn instant(&self, query: &str) -> Result<Vec<Sample>> {
            Ok(self.responses.get(query).cloned().unwrap_or_default())
        }
    }

    fn sample(pairs: &[(&str, &str)], value: f64) -> Sample {
        Sample {
            labels: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            value,
        }
    }

    /// A model-keyed and a DGD-keyed query merge into one snapshot, with the
    /// DGD-keyed field resolved through the resolver map and the per-model seq
    /// advancing across ticks.
    #[tokio::test]
    async fn collect_merges_model_and_dgd_keyed_and_bumps_seq() {
        let catalog = QueryCatalog {
            queries: vec![
                MetricQuery {
                    field: Field::TtftP50,
                    key_by: KeyBy::Model,
                    query: "q_ttft".to_string(),
                },
                MetricQuery {
                    field: Field::QueueDepth,
                    key_by: KeyBy::Dgd,
                    query: "q_queue".to_string(),
                },
            ],
            dgd_resolver: "q_resolve".to_string(),
            dgd_label: "dgd".to_string(),
        };
        let source = FakeSource {
            responses: HashMap::from([
                (
                    "q_resolve".to_string(),
                    vec![sample(&[("dgd", "dgdA"), ("model", "modelA")], 1.0)],
                ),
                (
                    "q_ttft".to_string(),
                    vec![sample(&[("model", "modelA")], 120.0)],
                ),
                ("q_queue".to_string(), vec![sample(&[("dgd", "dgdA")], 5.0)]),
            ]),
        };

        let mut seqs = HashMap::new();
        let first = collect_snapshots(&source, &catalog, "dc1", 1_000, &mut seqs).await;
        assert_eq!(first.len(), 1);
        let snap = &first[0];
        assert_eq!(snap.model_id, "modelA");
        assert_eq!(snap.ttft_p50_ms, Some(120.0));
        assert_eq!(snap.queue_depth, 5);
        assert_eq!(snap.seq, 1);
        assert_eq!(snap.captured_at_unix_ms, 1_000);

        // Per-model seq advances on the next tick.
        let second = collect_snapshots(&source, &catalog, "dc1", 2_000, &mut seqs).await;
        assert_eq!(second[0].seq, 2);
    }

    /// A failing query drops only its own field; a sibling query's data still
    /// lands. (The resolver succeeds, so the DGD-keyed field resolves.)
    #[tokio::test]
    async fn failing_query_skips_only_its_field() {
        struct ResolverThenErr;
        impl MetricSource for ResolverThenErr {
            async fn instant(&self, query: &str) -> Result<Vec<Sample>> {
                match query {
                    "q_resolve" => Ok(vec![sample(&[("dgd", "dgdA"), ("model", "modelA")], 1.0)]),
                    "q_ok" => Ok(vec![sample(&[("dgd", "dgdA")], 9.0)]),
                    _ => anyhow::bail!("boom"),
                }
            }
        }
        let catalog = QueryCatalog {
            queries: vec![
                MetricQuery {
                    field: Field::QueueDepth,
                    key_by: KeyBy::Dgd,
                    query: "q_ok".to_string(),
                },
                MetricQuery {
                    field: Field::GpuUtil,
                    key_by: KeyBy::Dgd,
                    query: "q_boom".to_string(),
                },
            ],
            dgd_resolver: "q_resolve".to_string(),
            dgd_label: "dgd".to_string(),
        };
        let mut seqs = HashMap::new();
        let snaps = collect_snapshots(&ResolverThenErr, &catalog, "dc1", 1, &mut seqs).await;
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].queue_depth, 9);
        assert_eq!(snaps[0].gpu_util_pct, None);
    }
}
