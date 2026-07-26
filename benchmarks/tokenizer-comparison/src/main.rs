// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Check token-ID parity between HuggingFace tokenizers, Fastokens, and Gigatoken.
//!
//! This is intentionally a standalone nightly-only experiment: it keeps the
//! Gigatoken dependency out of Dynamo's stable production dependency graph.

#[allow(dead_code)]
#[path = "../../../lib/llm/benches/tokenizer_dataset_support.rs"]
mod tokenizer_dataset_support;

use std::{env, fs, path::PathBuf};

use gigatoken_rs::{
    WorkerPool, encode_docs_ragged,
    load_tokenizer::hf::{HfTokenizer, load_hf_slice},
    sp_encode_docs_ragged,
};
use tokenizer_dataset_support::{
    DEFAULT_MAX_SAMPLES, TokenizerBench, bench_batched, bench_sequential, load_dataset, warm_up,
};

// Keep this identical to `lib/llm/benches/tokenizer_simple.rs`.
const SIMPLE_PROMPT: &str = "The cat sat by the window, watching raindrops race down the glass. Far thunder rumbled. She purred softly, feeling safe at home.";

struct Args {
    tokenizer: PathBuf,
    input: Option<PathBuf>,
    simple: bool,
    documents: usize,
    dataset: Option<String>,
    max_samples: usize,
    batch_size: Option<usize>,
    backend: Option<Backend>,
}
#[derive(Clone, Copy)]
enum Backend {
    HuggingFace,
    Fastokens,
    Gigatoken,
}

impl Backend {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "huggingface" => Ok(Self::HuggingFace),
            "fastokens" => Ok(Self::Fastokens),
            "gigatoken" => Ok(Self::Gigatoken),
            _ => Err("--backend must be huggingface, fastokens, or gigatoken".to_string()),
        }
    }
}

fn required_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut tokenizer = None;
        let mut input = None;
        let mut simple = false;
        let mut documents = 1;
        let mut dataset = None;
        let mut max_samples = DEFAULT_MAX_SAMPLES;
        let mut batch_size = None;
        let mut backend = None;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--tokenizer" => {
                    tokenizer = Some(PathBuf::from(required_value(&mut args, "--tokenizer")?))
                }
                "--input" => input = Some(PathBuf::from(required_value(&mut args, "--input")?)),
                "--simple" => simple = true,
                "--documents" => {
                    documents = required_value(&mut args, "--documents")?
                        .parse()
                        .map_err(|_| "--documents must be a positive integer".to_string())?
                }
                "--dataset" => dataset = Some(required_value(&mut args, "--dataset")?),
                "--max-samples" => {
                    max_samples = required_value(&mut args, "--max-samples")?
                        .parse()
                        .map_err(|_| "--max-samples must be a positive integer".to_string())?
                }
                "--batch-size" => {
                    batch_size = Some(
                        required_value(&mut args, "--batch-size")?
                            .parse()
                            .map_err(|_| "--batch-size must be a positive integer".to_string())?,
                    )
                }
                "--backend" => {
                    backend = Some(Backend::parse(&required_value(&mut args, "--backend")?)?)
                }
                "-h" | "--help" => return Err(Self::usage()),
                _ => return Err(format!("unknown argument {arg}\n\n{}", Self::usage())),
            }
        }

        let args = Self {
            tokenizer: tokenizer.ok_or_else(Self::usage)?,
            input,
            simple,
            documents,
            dataset,
            max_samples,
            batch_size,
            backend,
        };
        if args.documents == 0 || args.max_samples == 0 || args.batch_size == Some(0) {
            return Err(
                "--documents, --max-samples, and --batch-size must be positive".to_string(),
            );
        }
        if (args.input.is_some() || args.simple) == args.dataset.is_some() {
            return Err(format!(
                "select exactly one input mode\n\n{}",
                Self::usage()
            ));
        }
        if args.backend.is_some() && args.dataset.is_none() {
            return Err("--backend is supported only with --dataset".to_string());
        }
        Ok(args)
    }

    fn usage() -> String {
        "Usage:\n  cargo +nightly -Zprofile-rustflags run -- --tokenizer PATH (--simple | --input PATH) [--documents N]\n  cargo +nightly -Zprofile-rustflags run -- --tokenizer PATH --dataset DATASET [--max-samples N] [--batch-size N] [--backend huggingface|fastokens|gigatoken]".to_string()
    }
}

struct HuggingFaceBench(tokenizers::Tokenizer);

impl TokenizerBench for HuggingFaceBench {
    fn name(&self) -> &'static str {
        "HuggingFace"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.0
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| error.to_string())
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        self.0
            .encode_batch(texts.to_vec(), false)
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|encoding| encoding.get_ids().to_vec())
                    .collect()
            })
            .map_err(|error| error.to_string())
    }
}

struct FastokensBench(fastokens::Tokenizer);

impl TokenizerBench for FastokensBench {
    fn name(&self) -> &'static str {
        "Fastokens"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.0
            .encode_batch(&[text], false)
            .map_err(|error| error.to_string())?
            .into_iter()
            .next()
            .ok_or_else(|| "Fastokens returned no encoding".to_string())
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        self.0
            .encode_batch(texts, false)
            .map_err(|error| error.to_string())
    }
}

struct GigatokenBench {
    tokenizer: HfTokenizer,
    workers: WorkerPool,
}

impl GigatokenBench {
    fn unpack(ids: Vec<u32>, lengths: Vec<i64>) -> Result<Vec<Vec<u32>>, String> {
        let mut offset = 0usize;
        let mut documents = Vec::with_capacity(lengths.len());
        for length in lengths {
            let length =
                usize::try_from(length).map_err(|_| "negative token length".to_string())?;
            let end = offset
                .checked_add(length)
                .ok_or_else(|| "token length overflow".to_string())?;
            let tokens = ids
                .get(offset..end)
                .ok_or_else(|| "invalid Gigatoken ragged result".to_string())?;
            documents.push(tokens.to_vec());
            offset = end;
        }
        if offset != ids.len() {
            return Err("Gigatoken ragged result did not consume all token IDs".to_string());
        }
        Ok(documents)
    }
}

impl TokenizerBench for GigatokenBench {
    fn name(&self) -> &'static str {
        "Gigatoken"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_batch(&[text])?
            .into_iter()
            .next()
            .ok_or_else(|| "Gigatoken returned no encoding".to_string())
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        let (ids, lengths) = match &self.tokenizer {
            HfTokenizer::Bpe(tokenizer) => {
                let docs: Vec<&[u8]> = texts.iter().map(|text| text.as_bytes()).collect();
                encode_docs_ragged(&self.workers, tokenizer, &docs)
            }
            HfTokenizer::SentencePiece(tokenizer) => sp_encode_docs_ragged(tokenizer, texts),
        };
        Self::unpack(ids, lengths)
    }
}

fn split_documents(input: &str, count: usize) -> Vec<&str> {
    if count == 1 {
        return vec![input];
    }

    let target = input.len().div_ceil(count);
    let mut documents = Vec::with_capacity(count);
    let mut remaining = input;
    while !remaining.is_empty() && documents.len() + 1 < count {
        let mut split_at = target.min(remaining.len());
        while split_at > 0 && !remaining.is_char_boundary(split_at) {
            split_at -= 1;
        }
        if let Some(newline) = remaining[..split_at].rfind('\n') {
            split_at = newline + 1;
        }
        if split_at == 0 {
            split_at = remaining
                .char_indices()
                .nth(1)
                .map_or(remaining.len(), |(index, _)| index);
        }
        let (document, rest) = remaining.split_at(split_at);
        documents.push(document);
        remaining = rest;
    }
    if !remaining.is_empty() {
        documents.push(remaining);
    }
    documents
}

fn input_parity(documents: &[&str], tokenizers: &[&dyn TokenizerBench]) -> Result<(), String> {
    let results = tokenizers
        .iter()
        .map(|tokenizer| tokenizer.encode_batch(documents))
        .collect::<Result<Vec<_>, _>>()?;
    for (index, result) in results.iter().enumerate().skip(1) {
        if results[0] != *result {
            return Err(format!(
                "token parity failed: {} and {} returned different token IDs",
                tokenizers[0].name(),
                tokenizers[index].name(),
            ));
        }
    }
    let tokens = results[0].iter().map(Vec::len).sum::<usize>();
    println!("token parity: OK ({tokens} token IDs for all backends)");
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;
    let hf = HuggingFaceBench(
        tokenizers::Tokenizer::from_file(&args.tokenizer)
            .map_err(|error| format!("failed to load HF tokenizer: {error}"))?,
    );
    let fast = FastokensBench(
        fastokens::Tokenizer::from_file(&args.tokenizer)
            .map_err(|error| format!("failed to load Fastokens tokenizer: {error}"))?,
    );
    let tokenizer_json = fs::read(&args.tokenizer)
        .map_err(|error| format!("failed to read {}: {error}", args.tokenizer.display()))?;
    let gigatoken = GigatokenBench {
        tokenizer: load_hf_slice(&tokenizer_json)
            .map_err(|error| format!("failed to load Gigatoken tokenizer: {error}"))?,
        workers: WorkerPool::new(),
    };
    let all_tokenizers: [&dyn TokenizerBench; 3] = [&hf, &fast, &gigatoken];

    if let Some(dataset) = args.dataset {
        let tokenizers = match args.backend {
            Some(Backend::HuggingFace) => vec![&hf as &dyn TokenizerBench],
            Some(Backend::Fastokens) => vec![&fast as &dyn TokenizerBench],
            Some(Backend::Gigatoken) => vec![&gigatoken as &dyn TokenizerBench],
            None => all_tokenizers.to_vec(),
        };
        let samples = load_dataset(&dataset, args.max_samples);
        warm_up(&samples, &tokenizers)?;
        return match args.batch_size {
            Some(batch_size) => bench_batched(&samples, &tokenizers, batch_size),
            None => bench_sequential(&samples, &tokenizers),
        };
    }

    let input = if args.simple {
        SIMPLE_PROMPT.repeat(8_000 / SIMPLE_PROMPT.len())
    } else {
        fs::read_to_string(
            args.input
                .as_ref()
                .expect("input mode checked during parsing"),
        )
        .map_err(|error| format!("failed to read input: {error}"))?
    };
    let documents = split_documents(&input, args.documents);
    let bytes = documents
        .iter()
        .map(|document| document.len())
        .sum::<usize>();
    println!("input={bytes} bytes, documents={}", documents.len());
    input_parity(&documents, &all_tokenizers)
}
