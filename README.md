# Paged-Infer

**A Bare-Metal Rust Inference Engine with PagedAttention and Continuous Batching**

## Overview
`Paged-Infer` is a from-scratch, dependency-light inference engine designed to serve modern LLMs efficiently. By implementing a custom OS-style memory allocator (PagedAttention) alongside an iteration-level scheduler, this engine eliminates KV Cache memory fragmentation and maximizes generation throughput during batched autoregressive decoding. 

This project demonstrates deep expertise in low-level systems engineering, Rust memory management, and modern Transformer architecture—skills critical for ML Infrastructure roles at top-tier AI labs.

## Technical Specifications
* **Language:** Rust (Edition 2021)
* **Target Model:** Llama 3.2 1B (via `.safetensors` format)
* **Key Optimizations:**
    * Memory-mapped weight loading (`mmap`).
    * Paged KV Cache Allocation & Iteration-Level Scheduling (Continuous Batching).
    * **Block Size:** 16 tokens per physical block.
* **Core Exclusions:** No PyTorch, No TensorFlow, No HuggingFace `transformers` library for inference. All forward passes and attention mechanisms are written entirely from scratch.
* **Engineering Tradeoffs & Realities:**
    * **Floating Point Precision:** Llama 3.2 1B weights are natively `bfloat16`. To balance memory footprint with compute simplicity, weights are memory-mapped as `bf16` (via the `half` crate) and cast to `f32` upon entering the CPU cache for the forward pass.
    * **Tokenization:** Uses the `tokenizers` crate to handle Llama 3's complex ~128k BPE vocabulary, strictly managing special tokens like `<|begin_of_text|>` (128000) and `<|eot_id|>` (128009) to control sequence lifecycle.

## Architecture & Directory Structure

    paged-infer/
    ├── README.md               # Architecture details, benchmark graphs, and run instructions
    ├── scripts/
    │   └── download_model.py   # Script to fetch Llama 3.2 1B safetensors and tokenizer
    ├── Cargo.toml              # Rust dependencies (safetensors, memmap2, tokenizers, half)
    ├── src/
    │   ├── main.rs             # CLI entry point, scheduler, and continuous batching loop
    │   ├── tensor.rs           # Multi-dimensional array structs and basic ops
    │   ├── math.rs             # Matrix multiplication (GEMM), RoPE, and SwiGLU
    │   ├── model.rs            # Llama 3.2 Architecture (RMSNorm, Transformer Blocks)
    │   ├── memory/
    │   │   ├── mod.rs          # Memory module exports
    │   │   ├── allocator.rs    # Manages the pre-allocated physical KV cache pool
    │   │   └── block_table.rs  # Maps logical token sequences to physical memory blocks
    └── tests/
        ├── math_tests.rs       # Unit tests for tensor operations
        └── paged_tests.rs      # Verifies PagedAttention outputs match naive attention

## Development Roadmap

### Phase 1: Foundation (Naive Inference)
* **Goal:** Build a working, unoptimized forward pass for Llama 3.2 1B.
* **Tasks:**
    1. Parse and load the model weights into memory using `memmap2` and the `safetensors` crate.
    2. Implement basic ND-array tensor structs and pure Rust matrix multiplication.
    3. Implement Rotary Positional Embeddings (RoPE), SwiGLU, and RMSNorm.
    4. Write a naive Grouped-Query Attention (GQA) mechanism.
* **Deliverable:** A CLI interface that takes a prompt and successfully generates text one token at a time.

### Phase 2: The Block Allocator (Memory Management)
* **Goal:** Step away from neural networks and build the OS-level memory structures.
* **Tasks:**
    1. Define the physical constraints: Pre-allocate a large chunk of heap memory to act as the global KV Cache.
    2. Build the `BlockAllocator` struct that divides this memory into fixed chunks of **16 tokens**.
    3. Implement `allocate()` and `free()` methods to hand out blocks and reclaim them when a generation sequence finishes.
    4. Build the `BlockTable` mapping system (Logical -> Physical).
* **Deliverable:** Rigorous Rust unit tests proving the allocator can handle thousands of allocate/free cycles without leaking memory.

### Phase 3: PagedAttention & Continuous Batching
* **Goal:** Wire the memory manager into the model and achieve high throughput.
* **Tasks:**
    1. Rewrite the naive Attention function to query the `BlockTable` and fetch fragmented 16-token chunks during the Key/Value lookup.
    2. Implement an **Iteration-Level Scheduler**.
    3. Inject new requests into the batch at the exact moment another request generates its `<|eot_id|>` token, utilizing the newly freed physical blocks immediately.
* **Deliverable:** The engine seamlessly processes a continuous stream of prompts of varying lengths without stalling or wasting memory.

### Phase 4: Benchmarking & SIMD Optimization
* **Goal:** Quantify the engineering impact and make the engine practically viable.
* **Tasks:**
    1. Optimize the bare-metal Matrix Multiplication (GEMM). Replace pure Rust `for` loops with `std::arch` SIMD intrinsics (AVX2/AVX-512) to make the 1B parameter forward pass performant.
    2. Track memory usage (Resident Set Size) between a contiguous baseline and the Paged-Infer engine.
    3. Generate a chart showing how Paged-Infer achieves near 0% memory waste.
* **Deliverable:** A highly polished `README.md` featuring architecture diagrams and performance metrics.
## Optimization Pass 2 (March 26, 2026) — int8 Quantization + Speculative Decoding

### Feature 1: int8 Weight-Only Quantization

Per-row symmetric int8 quantization of all projection matrices. Each row's weights are scaled to the range `[-127, 127]` using a per-row `f32` scale factor, reducing weight memory by **~4x** with a parallel dequantizing matvec kernel.

- `quantize_rows_i8(weight, rows, cols)` — one-shot quantization at model load time
- `matvec_i8_weight_parallel()` — Rayon-parallel kernel; dequantizes on-the-fly during accumulation
- `QuantizedLinear` struct — drop-in replacement for `PackedLinear` with 4x lower memory footprint

Benchmark (`cargo run --release --bin benchmark`, 2048×2048 matrix, 20 iters):

| Kernel | Time | vs Baseline | vs Packed f32 | Memory |
|---|---:|---:|---:|---:|
| Baseline (bf16 convert each iter) | 0.1420s | 1.00x | — | 16.00 MB |
| Stream bf16 | 0.0704s | 2.02x | — | — |
| Packed f32 + parallel | 0.0193s | 7.35x | 1.00x | 16.00 MB |
| **int8 + parallel** | **0.0152s** | **9.36x** | **1.27x** | **4.01 MB** |

- **9.36x throughput vs baseline**, **1.27x vs packed f32** from better cache utilization
- **3.99x memory reduction** vs f32 (4 bytes → 1 byte per weight + tiny per-row scale overhead)
- For TinyLlama 1.1B: projection weights shrink from ~4.3 GB to ~1.1 GB

### Feature 2: Speculative Decoding with N-gram Drafting

`NgramDrafter` maintains an in-memory n-gram frequency table (default n=3). At each decode step it proposes K cheap draft tokens via O(1) table lookup; the main model verifies each candidate and accepts if its argmax agrees.

- `src/speculative.rs` — `NgramDrafter::observe()` / `::draft()` with majority-vote update rule
- `src/bin/speculative_benchmark.rs` — measures acceptance rate and theoretical throughput gain

```bash
MODEL_PATH=models/tinyllama-1.1b/model.safetensors \
SPEC_STEPS=50 SPEC_K=4 SPEC_N=3 \
cargo run --release --bin speculative_benchmark
```

**Acceptance rate** (the key metric) measures how often the cheap n-gram prediction matches the verifier's argmax. With batched prefill verification (future work), an acceptance rate of `α` with `K` draft tokens gives a throughput multiplier of approximately `(1 + α·K)` since the drafter is free.

| α (acceptance rate) | K=4 theoretical multiplier |
|---:|---:|
| 20% | ~1.8x |
| 40% | ~2.6x |
| 60% | ~3.4x |

Run `cargo run --release --bin speculative_benchmark` with a model to measure real acceptance rates on your workload.

## Optimization Pass 1 (March 26, 2026) — Prepack + Buffer Reuse

We implemented and benchmarked the two highest-impact follow-ups for decode throughput:

1. **Prepack bf16 weights into cache-friendly f32 layout + parallelize row matvec**
   Projection weights are converted once during model load and stored contiguously for fast row access; matvec now runs in parallel across output rows.
2. **Reuse attention score buffers instead of allocating per head per step**
   Attention keeps a reusable score scratch buffer and resets it in-place.

### Benchmark setup
- Command: `cargo run --release --bin benchmark`
- CPU benchmark is synthetic and isolates kernel behavior:
  - Matvec: hidden=2048, rows=2048, 20 iterations
  - Attention score path: head_dim=64, seq_len=1024, 200 iterations

### Results
| Kernel | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| Matvec (bf16 convert each iter) | 0.1420s | 0.0704s (stream bf16) | **2.02x** |
| Matvec (bf16 convert each iter) | 0.1420s | 0.0193s (packed + parallel) | **7.35x** |
| Attention score scratch handling | 0.0051s | 0.0051s | **1.00x** |

### Takeaway
- The largest win comes from **one-time prepacking + parallel row matvec** on projection kernels.
- int8 quantization pushes the matvec win further to **9.36x** with a 4x memory reduction.

## Final Validation Checklist (Correctness + E2E + Memory)

To make this project interview-ready and reproducible, use:

1. **Kernel / attention correctness parity**
   - `cargo test --test parity_tests`
   - Verifies paged attention matches a naive reference implementation on deterministic pseudo-random inputs.

2. **int8 quantization correctness**
   - `cargo test --test math_tests test_quantize_rows_i8_roundtrip`
   - `cargo test --test math_tests test_matvec_i8_matches_f32`
   - Verifies quantize→dequantize roundtrip and int8 matvec parity against f32 reference (within 2% relative error).

3. **Kernel microbenchmarks (no model required)**
   - `cargo run --release --bin benchmark`
   - Reports matvec throughput across bf16/f32/int8 kernels, memory footprints, and attention buffer reuse.

4. **End-to-end decode benchmark (throughput + latency + memory)**
   - `MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin e2e_benchmark`
   - Reports:
     - tokens/sec throughput
     - avg token latency
     - p50/p95 token latency
     - peak RSS memory (MB)

5. **Speculative decoding acceptance rate**
   - `MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin speculative_benchmark`
   - Reports acceptance rate and theoretical throughput multiplier for n-gram drafting with K=4.

If `MODEL_PATH` is missing, model-dependent binaries exit early with a clear message rather than failing.

### Larger local sweeps (recommended for final README table)

Use the helper script:

```bash
MODEL_PATH=/absolute/path/to/model.safetensors \
STEPS_LIST="64 128 256" \
BATCH_LIST="1 2 4 8" \
OUT_CSV=e2e_sweep.csv \
./scripts/run_e2e_sweep.sh
```

This produces a CSV with:
- `batch`, `steps`, `total_tokens`
- `throughput_tok_s`
- `avg_token_latency_ms`
- `p50_token_latency_us`, `p95_token_latency_us`
- `peak_rss_mb`

Note: `e2e_benchmark` now uses `/proc/self/status` when available and falls back to `ps -o rss=` for platforms like macOS, so RSS should no longer show `0.00` unless collection genuinely fails.

## Local E2E Sweep Results (TinyLlama 1.1B, user-provided)

| batch | steps | total_tokens | throughput_tok_s | avg_token_latency_ms | p50_us | p95_us | peak_rss_mb |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64  | 64   | 4.28 | 233.416 | 198181 | 262967 | 5234.39 |
| 1 | 128 | 128  | 3.87 | 258.206 | 247382 | 288431 | 4922.09 |
| 1 | 256 | 256  | 3.71 | 269.629 | 253601 | 296986 | 5208.28 |
| 2 | 64  | 128  | 3.90 | 256.486 | 245933 | 302792 | 5433.16 |
| 2 | 128 | 256  | 3.76 | 265.657 | 249403 | 311337 | 5318.77 |
| 2 | 256 | 512  | 3.66 | 273.450 | 253876 | 314387 | 5226.61 |
| 4 | 64  | 256  | 3.72 | 268.742 | 250367 | 323788 | 5137.52 |
| 4 | 128 | 512  | 3.83 | 261.184 | 247273 | 285391 | 5269.30 |
| 4 | 256 | 1024 | 3.76 | 265.742 | 252306 | 299577 | 5162.23 |
| 8 | 64  | 512  | 3.76 | 265.823 | 249151 | 300425 | 5069.22 |
| 8 | 128 | 1024 | 3.59 | 278.215 | 251924 | 347618 | 5106.17 |
| 8 | 256 | 2048 | 3.60 | 277.696 | 254201 | 348173 | 5277.98 |

### What this indicates
- **Throughput stabilizes around ~3.6–3.9 tok/s** across medium/large sweeps, peaking at **4.28 tok/s** for the shortest run (`batch=1`, `steps=64`).
- **Latency increases with longer contexts and larger batches**, especially p95 (up to ~348 ms at `batch=8`, long runs), which is expected from growing attention history.
- **Peak RSS is consistently ~4.9–5.4 GB**, suggesting the memory footprint is stable under sweep load and compatible with the paged KV design.

## Scaling Behavior Summary

### 1→32 batch scaling (currently measured 1→8)

| batch | throughput_tok_s | avg_latency_ms | p95_us | peak_rss_mb |
|---:|---:|---:|---:|---:|
| 1 | 3.95 | 253.75 | 282795 | 5121.59 |
| 2 | 3.77 | 265.20 | 309505 | 5326.18 |
| 4 | 3.77 | 265.22 | 302919 | 5189.68 |
| 8 | 3.65 | 273.91 | 332072 | 5151.12 |

```mermaid
xychart-beta
    title "Throughput vs Batch"
    x-axis "batch" [1, 2, 4, 8]
    y-axis "tok/s" 0 --> 5
    line [3.95, 3.77, 3.77, 3.65]
```

### Throughput vs context length

| steps | throughput_tok_s | avg_latency_ms | p95_us |
|---:|---:|---:|---:|
| 64 | 3.92 | 256.12 | 297493 |
| 128 | 3.76 | 265.82 | 308194 |
| 256 | 3.68 | 271.63 | 314781 |

### Memory vs sequence length

| steps | peak_rss_mb |
|---:|---:|
| 64 | 5218.57 |
| 128 | 5154.08 |
| 256 | 5218.77 |

You can regenerate these summaries from CSV using:
- `./scripts/analyze_sweep.py e2e_sweep.csv`

## Product Mode: Simple OpenAI-Compatible HTTP API

Run:

```bash
HOST=0.0.0.0 PORT=8080 \
MODEL_PATH=models/tinyllama-1.1b/model.safetensors \
TOKENIZER_PATH=models/tinyllama-1.1b/tokenizer.json \
cargo run --release --bin http_server
```

Health check:

```bash
curl http://localhost:8080/health
```

Chat completion (OpenAI-style path):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tinyllama-1.1b",
    "messages": [{"role":"user","content":"Explain paged attention in one paragraph."}],
    "max_tokens": 64
  }'
```

If model/tokenizer files are unavailable, the server still runs in a `dry-run` mode for API integration tests.

## Killer Feature: KV Cache Eviction (LRU)

`KvCacheManager` introduces an eviction-aware KV allocator policy:
- Tracks per-sequence KV ownership and `last_used_tick`
- On allocation pressure, evicts the least-recently-used sequence (excluding current requester)
- Keeps the allocator from hard-failing under contention-heavy workloads

Run benchmark:

```bash
cargo run --release --bin eviction_benchmark
```

Latest result on this branch:

| Policy | completed | dropped | active_end |
|---|---:|---:|---:|
| No eviction | 11 | 93 | 8 |
| LRU eviction | 104 | 0 | 8 |

`completed_gain_pct = 845.45%`

This benchmark simulates long-running and short-running requests under tight KV block pressure; LRU prevents starvation and dramatically improves request completion throughput.
