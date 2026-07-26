<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tokenizer parity experiment

This standalone Rust experiment compares HuggingFace `tokenizers`, Fastokens,
and Gigatoken without adding Gigatoken to Dynamo's production workspace or
runtime dependency graph. It requires nightly Rust because Gigatoken currently
does.

The simple and input modes validate exact flat token-ID parity:

```bash
cd benchmarks/tokenizer-comparison
cargo +nightly -Zprofile-rustflags run -- \
  --tokenizer /path/to/tokenizer.json \
  --simple
```

```bash
cargo +nightly -Zprofile-rustflags run -- \
  --tokenizer /path/to/tokenizer.json \
  --input /path/to/corpus.txt \
  --documents 8
```

`--documents` splits the UTF-8 input at newline or character boundaries. Every
backend receives the same resulting documents.

## Shared dataset benchmark

`--dataset` reuses the dataset loading, batch boundaries, token-ID parity
checks, and reporting harness from
`lib/llm/benches/tokenizer_dataset_support.rs`. The in-workspace
`tokenizer_dataset` benchmark uses that same harness for HuggingFace and
Fastokens; this standalone runner adds Gigatoken as a third backend.

Run the LongBench-v2 scenario used by Dynamo's dataset bench:

```bash
cargo +nightly -Zprofile-rustflags run --release -- \
  --tokenizer /path/to/tokenizer.json \
  --dataset zai-org/LongBench-v2
```

Run its batched variant:

```bash
cargo +nightly -Zprofile-rustflags run --release -- \
  --tokenizer /path/to/tokenizer.json \
  --dataset zai-org/LongBench-v2 \
  --max-samples 503 \
  --batch-size 64
```



To measure one backend memory without constructing the other two, pass both
`--backend` and `--memory`:

```bash
cargo +nightly -Zprofile-rustflags run --release -- --tokenizer /path/to/tokenizer.json --dataset zai-org/LongBench-v2 --max-samples 503 --batch-size 64 --backend gigatoken --memory
```

This reports Linux process `VmRSS` and `VmHWM` after dataset loading, tokenizer
initialization, warmup, and the benchmark. Deltas use the post-dataset snapshot
as their baseline; they are process-level measurements, not per-allocation
attribution.

Supported datasets and their extraction rules match Dynamo's existing bench:

- `zai-org/LongBench-v2`: the `context` field from `data.json`.
- `RyokoAI/ShareGPT52K`: formatted conversation turns from `sg_90k_part1.json`.

The shared harness warms every backend once, measures each backend separately,
checks per-document token-ID parity, and reports total time, average latency,
throughput, and speedup relative to HuggingFace.
