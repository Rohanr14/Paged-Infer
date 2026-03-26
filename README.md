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
## Latest Optimization Pass (March 26, 2026)

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
| Matvec (bf16 convert each iter) | 0.3186s | 0.1555s (stream bf16) | **2.05x** |
| Matvec (bf16 convert each iter) | 0.3186s | 0.0777s (packed + parallel) | **4.10x** |
| Attention score scratch handling | 0.0135s | 0.0130s | **1.04x** |

### Takeaway
- The largest win now comes from **one-time prepacking + parallel row matvec** on projection kernels.
- Buffer reuse in attention still helps allocator churn and consistency, even when throughput gain is modest.

## Final Validation Checklist (Correctness + E2E + Memory)

To make this project interview-ready and reproducible, use:

1. **Kernel / attention correctness parity**
   - `cargo test --test parity_tests`
   - Verifies paged attention matches a naive reference implementation on deterministic pseudo-random inputs.

2. **End-to-end decode benchmark (throughput + latency + memory)**
   - `MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin e2e_benchmark`
   - Reports:
     - tokens/sec throughput
     - avg token latency
     - p50/p95 token latency
     - peak RSS memory (MB)

If `MODEL_PATH` is missing, the benchmark binary exits early with a clear message rather than failing.

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
