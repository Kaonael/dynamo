// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared dataset loading, parity, and reporting support for tokenizer benchmarks.
//!
//! This module deliberately depends only on `hf-hub`, `serde_json`, and the
//! standard library. It can therefore be shared by Dynamo's workspace bench and
//! the standalone Gigatoken experiment without importing either implementation.

use std::path::Path;
use std::time::{Duration, Instant};

/// Default HuggingFace model for the tokenizer.
pub const DEFAULT_HF_MODEL: &str = "Qwen/Qwen3-0.6B";

/// Default HuggingFace Hub dataset.
pub const DEFAULT_DATASET: &str = "zai-org/LongBench-v2";

/// Default number of dataset samples.
pub const DEFAULT_MAX_SAMPLES: usize = 503;

/// Common interface implemented by each tokenizer backend under comparison.
pub trait TokenizerBench {
    fn name(&self) -> &'static str;
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String>;
}

/// Resolve a tokenizer path from `TOKENIZER_PATH`, a HuggingFace model name, or
/// the default model.
pub fn resolve_tokenizer_path() -> String {
    let input = std::env::var("TOKENIZER_PATH").ok();

    if let Some(ref path) = input
        && Path::new(path).is_file()
    {
        eprintln!("[setup] Using local tokenizer: {path}");
        return path.clone();
    }

    let model_name = input.as_deref().unwrap_or(DEFAULT_HF_MODEL);
    eprintln!("[setup] Downloading tokenizer for {model_name}...");

    let cache = hf_hub::Cache::default();
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache)
        .with_progress(true)
        .build()
        .expect("Failed to create HuggingFace API client");

    let repo = api.model(model_name.to_string());
    let path = repo
        .get("tokenizer.json")
        .expect("Failed to download tokenizer.json");
    let path = path.display().to_string();
    eprintln!("[setup] Tokenizer: {path}");
    path
}

/// Return the JSON filename for a supported HuggingFace Hub dataset.
fn dataset_json_file(dataset: &str) -> &'static str {
    match dataset {
        "RyokoAI/ShareGPT52K" => "sg_90k_part1.json",
        "zai-org/LongBench-v2" => "data.json",
        _ => panic!(
            "Unknown dataset: {dataset}. Supported: zai-org/LongBench-v2, RyokoAI/ShareGPT52K"
        ),
    }
}

/// Extract one input text from a dataset record.
fn extract_text(dataset: &str, item: &serde_json::Value) -> Option<String> {
    match dataset {
        "RyokoAI/ShareGPT52K" => {
            let messages = item.get("conversations")?.as_array()?;
            let parts: Vec<String> = messages
                .iter()
                .filter_map(|message| {
                    let role = message.get("from")?.as_str()?;
                    let value = message.get("value")?.as_str()?;
                    (!value.is_empty()).then(|| format!("[{role}]: {value}"))
                })
                .collect();
            (!parts.is_empty()).then(|| parts.join("\n\n"))
        }
        "zai-org/LongBench-v2" => {
            let context = item.get("context")?.as_str()?;
            (!context.is_empty()).then(|| context.to_string())
        }
        _ => None,
    }
}

/// Load text samples from a supported HuggingFace Hub dataset.
pub fn load_dataset(dataset: &str, max_items: usize) -> Vec<String> {
    let json_file = dataset_json_file(dataset);

    eprintln!("[setup] Downloading dataset {dataset}...");
    let api = hf_hub::api::sync::Api::new().expect("Failed to create HuggingFace API client");
    let repo = api.dataset(dataset.to_string());
    let json_path = repo.get(json_file).expect("Failed to download dataset");

    let text = std::fs::read_to_string(&json_path).expect("Failed to read dataset JSON");
    let data: Vec<serde_json::Value> =
        serde_json::from_str(&text).expect("Failed to parse dataset JSON");
    let samples: Vec<String> = data
        .iter()
        .take(max_items)
        .filter_map(|item| extract_text(dataset, item))
        .collect();

    eprintln!("[setup] Loaded {} samples", samples.len());
    samples
}

/// Warm every backend with the first sample before timing.
pub fn warm_up(samples: &[String], tokenizers: &[&dyn TokenizerBench]) -> Result<(), String> {
    let Some(sample) = samples.first() else {
        return Ok(());
    };
    for tokenizer in tokenizers {
        tokenizer.encode(sample)?;
    }
    Ok(())
}

#[derive(Default)]
struct Stats {
    duration: Duration,
}

fn report(
    label: &str,
    sample_count: usize,
    total_chars: u64,
    total_tokens: u64,
    tokenizers: &[&dyn TokenizerBench],
    stats: &[Stats],
) {
    println!();
    println!("=== {label} ({sample_count} samples) ===");
    println!("  Total chars:        {total_chars}");
    println!("  Total tokens:       {total_tokens}");
    println!("  ---");

    for (tokenizer, stat) in tokenizers.iter().zip(stats) {
        let milliseconds = stat.duration.as_secs_f64() * 1000.0;
        println!("  {} total: {:>14.2} ms", tokenizer.name(), milliseconds);
    }

    println!("  ---");
    for (index, (tokenizer, stat)) in tokenizers.iter().zip(stats).enumerate() {
        let seconds = stat.duration.as_secs_f64();
        println!(
            "  {} avg/sample: {:>9.3} ms",
            tokenizer.name(),
            seconds * 1000.0 / sample_count as f64
        );
        println!(
            "  {} throughput: {:>9.2} MB/s",
            tokenizer.name(),
            total_chars as f64 / seconds / 1_000_000.0
        );
        if index > 0 {
            println!(
                "  {} speedup vs {}: {:>7.2}x",
                tokenizer.name(),
                tokenizers[0].name(),
                stats[0].duration.as_secs_f64() / seconds
            );
        }
    }
}

fn compare(
    sample_index: usize,
    reference: &[u32],
    actual: &[u32],
    reference_name: &str,
    actual_name: &str,
    mismatches: &mut u64,
) {
    if reference != actual {
        *mismatches += 1;
        if *mismatches <= 3 {
            eprintln!(
                "[MISMATCH] sample {sample_index}: {reference_name}={} tokens, {actual_name}={} tokens",
                reference.len(),
                actual.len(),
            );
        }
    }
}

/// Run an exact per-sample parity check and measure sequential encoding.
pub fn bench_sequential(
    samples: &[String],
    tokenizers: &[&dyn TokenizerBench],
) -> Result<(), String> {
    assert!(!tokenizers.is_empty(), "at least one tokenizer is required");
    let mut stats = (0..tokenizers.len())
        .map(|_| Stats::default())
        .collect::<Vec<_>>();
    let mut total_tokens = 0u64;
    let mut total_chars = 0u64;
    let mut mismatches = 0u64;

    for (sample_index, sample) in samples.iter().enumerate() {
        let mut results = Vec::with_capacity(tokenizers.len());
        for (index, tokenizer) in tokenizers.iter().enumerate() {
            let start = Instant::now();
            let ids = tokenizer.encode(sample)?;
            stats[index].duration += start.elapsed();
            results.push(ids);
        }

        for (index, result) in results.iter().enumerate().skip(1) {
            compare(
                sample_index,
                &results[0],
                result,
                tokenizers[0].name(),
                tokenizers[index].name(),
                &mut mismatches,
            );
        }
        total_tokens += results[0].len() as u64;
        total_chars += sample.len() as u64;

        if (sample_index + 1) % 20 == 0 {
            eprintln!("[progress] {}/{}", sample_index + 1, samples.len());
        }
    }

    if mismatches == 0 {
        eprintln!("[OK] All samples produced identical token IDs");
    } else {
        eprintln!("[WARNING] {mismatches} samples had mismatched token IDs");
    }
    report(
        "Sequential Benchmark",
        samples.len(),
        total_chars,
        total_tokens,
        tokenizers,
        &stats,
    );
    Ok(())
}

/// Run an exact per-sample parity check and measure batched encoding.
pub fn bench_batched(
    samples: &[String],
    tokenizers: &[&dyn TokenizerBench],
    batch_size: usize,
) -> Result<(), String> {
    assert!(!tokenizers.is_empty(), "at least one tokenizer is required");
    assert!(batch_size > 0, "batch size must be positive");
    let mut stats = (0..tokenizers.len())
        .map(|_| Stats::default())
        .collect::<Vec<_>>();
    let mut total_tokens = 0u64;
    let mut total_chars = 0u64;
    let mut mismatches = 0u64;
    let batch_count = samples.len().div_ceil(batch_size);

    for (batch_index, batch) in samples.chunks(batch_size).enumerate() {
        let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
        let mut results = Vec::with_capacity(tokenizers.len());
        for (index, tokenizer) in tokenizers.iter().enumerate() {
            let start = Instant::now();
            let ids = tokenizer.encode_batch(&refs)?;
            stats[index].duration += start.elapsed();
            results.push(ids);
        }

        for (index, result) in results.iter().enumerate().skip(1) {
            if results[0].len() != result.len() {
                mismatches += 1;
                if mismatches <= 3 {
                    eprintln!(
                        "[LENGTH MISMATCH] batch {batch_index}: {} returned {}, {} returned {}",
                        tokenizers[0].name(),
                        results[0].len(),
                        tokenizers[index].name(),
                        result.len(),
                    );
                }
            }
            for row in 0..results[0].len().max(result.len()) {
                let sample_index = batch_index * batch_size + row;
                match (results[0].get(row), result.get(row)) {
                    (Some(reference), Some(actual)) => compare(
                        sample_index,
                        reference,
                        actual,
                        tokenizers[0].name(),
                        tokenizers[index].name(),
                        &mut mismatches,
                    ),
                    (Some(_), None) | (None, Some(_)) => {
                        mismatches += 1;
                    }
                    (None, None) => unreachable!(),
                }
            }
        }

        total_tokens += results[0].iter().map(|ids| ids.len() as u64).sum::<u64>();
        total_chars += batch.iter().map(|sample| sample.len() as u64).sum::<u64>();

        if (batch_index + 1) % 5 == 0 {
            eprintln!("[progress] batch {}/{}", batch_index + 1, batch_count);
        }
    }

    if mismatches == 0 {
        eprintln!("[OK] All samples produced identical token IDs");
    } else {
        eprintln!("[WARNING] {mismatches} samples had mismatched token IDs");
    }
    report(
        &format!("Batched Benchmark (batch_size={batch_size})"),
        samples.len(),
        total_chars,
        total_tokens,
        tokenizers,
        &stats,
    );
    Ok(())
}
