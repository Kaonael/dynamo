# LongBench-v2 isolated memory results

<text>

| Mode | Backend | Dataset baseline RSS (GiB) | Tokenizer init ΔRSS (GiB) | RSS after benchmark (GiB) | ΔRSS after benchmark (GiB) | Peak process RSS / VmHWM (GiB) | Throughput (MB/s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Sequential | HuggingFace | 0.43 | +0.13 | 0.91 | +0.47 | 2.21 | 1.66 |
| Sequential | Fastokens | 0.43 | +0.15 | 0.95 | +0.51 | 1.27 | 94.97 |
| Sequential | Gigatoken | 0.43 | +0.09 | 0.77 | +0.34 | 1.27 | 216.83 |
| Batch=64 | HuggingFace | 0.43 | +0.13 | 18.16 | +17.73 | 19.64 | 5.67 |
| Batch=64 | Fastokens | 0.43 | +0.15 | 2.81 | +2.38 | 2.97 | 255.06 |
| Batch=64 | Gigatoken | 0.43 | +0.09 | 1.56 | +1.12 | 1.60 | 718.23 |

</text>

LongBench-v2, 503 contexts / 446.09 MB, Qwen3-14B tokenizer, `taskset -c 0-55`.
The dataset baseline is the current `VmRSS` after loading the extracted samples. The process high-water mark can retain the higher transient RSS from JSON parsing; this is why the sequential Fastokens and Gigatoken peak remains 1.27 GiB.
