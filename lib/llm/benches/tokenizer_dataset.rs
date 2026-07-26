// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dataset-driven HuggingFace and Fastokens tokenizer benchmark.
//!
//! The dataset loading, parity checking, and reporting implementation lives in
//! `tokenizer_dataset_support.rs` so the standalone Gigatoken experiment can
//! exercise the identical scenario without entering Dynamo's Cargo workspace.

mod tokenizer_dataset_support;

use dynamo_llm::tokenizers::{FastTokenizer, HuggingFaceTokenizer, traits::Encoder};
use tokenizer_dataset_support::{
    DEFAULT_DATASET, DEFAULT_MAX_SAMPLES, TokenizerBench, bench_batched, bench_sequential,
    load_dataset, resolve_tokenizer_path, warm_up,
};

struct HuggingFaceBench(HuggingFaceTokenizer);

impl TokenizerBench for HuggingFaceBench {
    fn name(&self) -> &'static str {
        "HuggingFace"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.0
            .encode(text)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|error| error.to_string())
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        self.0
            .encode_batch(texts)
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|encoding| encoding.token_ids().to_vec())
                    .collect()
            })
            .map_err(|error| error.to_string())
    }
}

struct FastokensBench(FastTokenizer);

impl TokenizerBench for FastokensBench {
    fn name(&self) -> &'static str {
        "Fastokens"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.0
            .encode(text)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|error| error.to_string())
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        self.0
            .encode_batch(texts)
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|encoding| encoding.token_ids().to_vec())
                    .collect()
            })
            .map_err(|error| error.to_string())
    }
}

fn main() {
    // This benchmark downloads a large dataset and takes several minutes.
    // It is opt-in to avoid blocking `cargo test --all-targets` in CI.
    if std::env::var("RUN_BENCH").is_err() {
        eprintln!("[skip] tokenizer_dataset benchmark skipped. Set RUN_BENCH=1 to run it.");
        eprintln!("[skip] See lib/llm/benches/README.md for usage.");
        return;
    }

    let tokenizer_path = resolve_tokenizer_path();
    let dataset = std::env::var("DATASET").unwrap_or_else(|_| DEFAULT_DATASET.to_string());
    let max_samples = std::env::var("MAX_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_SAMPLES);
    let batch_size = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse().ok());

    let samples = load_dataset(&dataset, max_samples);
    let hf = HuggingFaceBench(
        HuggingFaceTokenizer::from_file(&tokenizer_path)
            .expect("Failed to load HuggingFace tokenizer"),
    );
    let fast = FastokensBench(
        FastTokenizer::from_file(&tokenizer_path).expect("Failed to load Fastokens tokenizer"),
    );
    let tokenizers: [&dyn TokenizerBench; 2] = [&hf, &fast];

    warm_up(&samples, &tokenizers).expect("Tokenizer warm-up failed");
    match batch_size {
        Some(batch_size) => {
            bench_batched(&samples, &tokenizers, batch_size).expect("Batched benchmark failed")
        }
        None => bench_sequential(&samples, &tokenizers).expect("Sequential benchmark failed"),
    }
}
