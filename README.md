# Paged-Infer

**An LLM inference engine written from scratch in Rust: PagedAttention, automatic prefix caching, copy-on-write forking, batched decode, and hand-written SIMD kernels.**

No PyTorch, no TensorFlow, no `transformers`. The forward pass, the attention
kernels, the KV-cache allocator, the scheduler, and the HTTP server are all in
this repository. The only meaningful dependencies are `safetensors` and
`memmap2` for reading checkpoints, `tokenizers` for BPE, and `rayon` for thread
pooling.

The interesting part is not the transformer — it's the memory manager. Serving
LLMs is a memory allocation problem wearing a linear algebra costume, and this
engine treats it that way: KV cache is paged into fixed blocks, blocks are
reference counted, identical prefixes across requests map the same physical
block, and forked sequences split blocks copy-on-write.

---

## Verified against a reference implementation

Every claim below rests on this. `scripts/gen_golden_fixture.py` builds a small
Llama checkpoint in the exact HuggingFace tensor layout and computes reference
logits with an independent NumPy transcription of `modeling_llama.py`.
`tests/golden_parity_tests.rs` drives that same checkpoint through the real
engine — loader, paged cache, block table, SIMD kernels — and compares.

```
incremental decode: max|delta|=0.000003, argmax mismatches=0/40
prefill:            max|delta|=0.000002
```

The reference is written against HuggingFace semantics rather than against this
engine's code, so agreement is evidence rather than a tautology. The fixture is
~400 KB, spans three KV blocks, and runs in CI in under a second.

**This is how three real bugs were found.** Before the harness existed the engine
diverged from the reference by `max|Δ| = 0.67` and picked a *different greedy
token at 5 of 40 positions*:

| Bug | Effect |
|---|---|
| Rotary embeddings used the GPT-J interleaved convention | HF checkpoints are permuted for NeoX `rotate_half`; every attention score was computed on wrongly rotated Q/K |
| Key heads rotated once per *query* head | Under GQA, `kv_group` query heads share a key head — an 8× over-rotation on TinyLlama |
| Unmapped positions scored `0.0` instead of `-inf` | They survived the softmax and took a full share of attention weight |

Separately, the engine had **no prefill at all**: the scheduler ran a forward
pass only on the *last* prompt token, leaving positions `0..n-1` of the KV cache
zeroed. The model could not see the prompt.

---

## Quick start

Nothing below needs model weights:

```bash
cargo test                                        # 95 tests, incl. golden parity
cargo run --release --bin benchmark               # kernel attribution ladder
cargo run --release --bin batch_benchmark         # batched vs sequential decode
cargo run --release --bin prefix_cache_benchmark  # what prefix caching is worth
```

With weights:

```bash
pip install huggingface_hub
python3 scripts/download_model.py                 # TinyLlama 1.1B, ~2.2 GB

cargo run --release                               # CLI demo
QUANT=int8 cargo run --release                    # ~4x less weight memory

MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin e2e_benchmark
MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin batch_benchmark

MODEL_PATH=models/tinyllama-1.1b/model.safetensors \
TOKENIZER_PATH=models/tinyllama-1.1b/tokenizer.json \
cargo run --release --bin http_server
```

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Explain paged attention."}],
  "max_tokens": 64, "n": 2, "temperature": 0.8
}'

curl localhost:8080/metrics   # prefix cache hit rate, tokens reused, tok/s
```

The engine reads the architecture from the checkpoint's own `config.json`, so
any HuggingFace-format Llama checkpoint works, not just TinyLlama.

---

## The memory manager

```
                    ┌───────────────────────────────────────────┐
   request ────────►│  Engine::step()                           │
                    │    admit → prefill → decode → reclaim     │
                    └───────────┬───────────────────────────────┘
                                │
                    ┌───────────▼───────────────────────────────┐
                    │  KvCacheManager                           │
                    │    prefix reuse · copy-on-write · evict   │
                    └──────┬──────────────────────┬─────────────┘
                           │                      │
                ┌──────────▼─────────┐  ┌─────────▼──────────┐
                │  PrefixCache       │  │  BlockAllocator    │
                │  hash → block      │  │  refcounted pool   │
                └──────────┬─────────┘  └─────────┬──────────┘
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────▼─────────────────────────┐
                    │  KV cache: [layer][block][tok][kv_head]   │
                    │  BlockTable maps logical → physical       │
                    └───────────────────────────────────────────┘
```

### Reference-counted blocks

A physical block returns to the free pool only when its last reference drops.
That single change is what makes everything else possible: several sequences can
map the same block at once. Double-free and incref-after-free are assertions,
not silent corruption. `tests` run 5,000 alloc/free rounds with interleaved
sharing and check the pool returns to exactly its starting state.

### Automatic prefix caching

Serving traffic is enormously repetitive — the same system prompt, the same
few-shot preamble, the same retrieved document, on request after request. The KV
state for a token sequence is a deterministic function of that sequence, so it
can be computed once and shared.

Blocks are keyed by a hash chained through the previous block's hash, so a
block's identity covers its **entire prefix**. This matters for soundness: two
prompts can share sixteen tokens at some offset while having completely
different histories, and their KV state for those tokens genuinely differs.
Chaining means a hit implies identical history — exactly the condition under
which reuse is valid. Lookups walk blocks in order and stop at the first miss.

`tests/prefix_cache_parity_tests.rs` proves the reuse is numerically invisible:
a batch with a shared preamble produces logits agreeing to `<1e-5` with a
cache-disabled run, while recomputing 48 prompt tokens instead of 120.

### Copy-on-write forking

Drawing *n* continuations from one prompt maps the parent's blocks into each
child instead of copying them. Nothing moves at fork time. The one block the
siblings both write to is split lazily, on first write — so a sample that stops
early never pays for a copy at all.

---

## What a run looks like

`cargo run --release` submits three requests that share one system prompt; the
third asks for four continuations, which are forked rather than re-prefilled.

```
Mapped 22 layers, 4.14 GB of F32 weights. SIMD backend: neon.
KV cache: 352.00 MB across 512 blocks of 16 tokens.

[req 0 / seq 0] 32 tokens, Length, ttft 666.74ms
  Answer: PagedAttention solves the problem of training deep neural networks on
  large datasets by using a technique called paging. This technique allows the model

[req 1 / seq 1] 32 tokens, Length, ttft 293.42ms
  Answer: Continuous batching improves throughput by reducing the number of
  batches that need to be processed. This reduces the time required to

[req 2 / seq 2] 32 tokens, Length, ttft 260.24ms
  Our system achieves grouped-query attention by using the attention scores
  across all attention heads for a single input word, and the attention scores for the

[req 2 / seq 3] 32 tokens, Length, ttft 260.85ms
  Answer the question clearly and concisely. ...

[req 2 / seq 4] ...
[req 2 / seq 5] ...

Completed 6 sequences in 4.72s over 31 steps.
  prompt tokens     : 103 total, 71 prefilled, 32 reused from cache
  generated tokens  : 192
  prefix cache      : 2/5 block lookups hit (40.0%)
  copy-on-write     : 3 block copies
  prefill 1.22s / decode 3.50s (54.92 tok/s)
```

Requests 1 and 2 have a lower TTFT than request 0 because they inherit its
system-prompt blocks. The four sequences under request 2 share one prefill and
diverge into visibly different continuations — that is copy-on-write forking
doing its job, and the three block copies are the partial block they all wrote
into.

The 40% hit rate is a property of that prompt set, not a ceiling: the demo's
system prompt has since been lengthened to a realistic size, because a preamble
shorter than a block or two has nothing to share. Output is TinyLlama 1.1B
being TinyLlama — fluent, and wrong about what PagedAttention is.

---

## End-to-end decode

TinyLlama 1.1B, full causal attention (no sliding window), greedy decode, on an
**Apple Silicon MacBook Air** with the NEON kernels. `e2e_benchmark` walks the
batch one sequence at a time, so this is the *sequential* path — see
[batched decode](#batched-decode) below for what running them together buys:

| batch | steps | throughput | avg latency | p50 | p95 | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 24 | **11.37 tok/s** | 88.0 ms | 74.0 ms | 83.7 ms | 5,870 MB |

```bash
MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin e2e_benchmark
```

That 5,870 MB is the cost of widening bf16 to f32 at load: a 1.1B model carries
~4.14 GB of projections plus the KV pool. `QUANT=int8` stores them per-row int8
instead, and because decode is memory-bound it is a *throughput* knob, not only
a footprint one:

| | weights | throughput | avg latency | p95 | peak RSS |
|---|---:|---:|---:|---:|---:|
| f32 | 4.14 GB | 11.37 tok/s | 88.0 ms | 83.7 ms | 5,870 MB |
| **int8** | **1.23 GB** | **31.79 tok/s** | **31.5 ms** | 39.9 ms | **3,320 MB** |

A quarter of the weight bytes buys **2.8x the throughput**. The ratio is 3.36x
rather than 4x because the LM head stays f32 on purpose, and RSS falls by less
than the weights do because the KV pool and the mapped checkpoint are unchanged.

```bash
QUANT=int8 MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin e2e_benchmark
```

### Batched decode

Decoding is memory-bound — each step streams every weight matrix to do one
multiply-add per element — so running `B` sequences one at a time reads the
weights `B` times. `decode_batch_into` runs the active batch through the model
together, streaming each matrix once and reusing every row across the batch.

**TinyLlama 1.1B (4.14 GB of f32 weights), Apple Silicon, NEON, 8 threads, 128
tokens of context:**

| batch | sequential | batched | speedup | weight GB/s |
|---:|---:|---:|---:|---:|
| 1 | 13.71 tok/s | 13.93 tok/s | 1.02x | 57.6 |
| 2 | 13.72 tok/s | 26.49 tok/s | 1.93x | 54.8 |
| 4 | 13.49 tok/s | 50.50 tok/s | **3.74x** | 52.2 |
| 8 | 13.81 tok/s | 59.80 tok/s | **4.33x** | 30.9 |

The same sweep on **4-core x86_64 (AVX2+FMA)** with 0.77 GB of weights reaches
2.51x at batch 4 and 3.17x at batch 8. Apple Silicon does better because it
sustains ~57 GB/s to the CPU against that machine's ~25, so it stays
bandwidth-bound — the regime where batching pays — out to a larger batch.

Sequential throughput is flat in batch size, which is the point: it is doing `B`
times the work for `B` times the traffic. Batch 1 costs nothing (0.99x), so
single-sequence decode is not penalized.

The speedup is sublinear, and the bandwidth column explains why — it *falls* as
batch grows, meaning the kernel stops being bandwidth-bound and the parts that
do not batch take over: attention (each sequence has its own KV, position and
block table) plus per-sequence RMSNorm, RoPE and SwiGLU. Context length moves
this less than batch size does: at batch 8, 3.8x at 32 tokens of context against
3.3x at 512.

### Batched prefill

Prefill sets time-to-first-token, and it batches along *positions* instead of
sequences — same kernel, same reason. It is correct for the same reason too:
every position's K/V is written to the cache before any attention runs, and a
position's attention loop ends at itself, so it sees the positions before it in
the chunk and none after. That is exactly causal masking, for free.

**TinyLlama 1.1B, 256-token prompt, Apple Silicon, NEON:**

| positions per pass | prefill | throughput | speedup |
|---:|---:|---:|---:|
| 1 (sequential) | 18.275 s | 14.0 tok/s | 1.00x |
| 4 | 4.811 s | 53.2 tok/s | 3.80x |
| 8 | 4.073 s | 62.9 tok/s | **4.49x** |
| 16 | 4.293 s | 59.6 tok/s | 4.26x |
| 32 | 4.604 s | 55.6 tok/s | 3.97x |
| 64 | 4.193 s | 61.0 tok/s | 4.36x |

The gain plateaus after about 8 positions per pass; past that the spread is
run-to-run noise on both machines measured, so the default chunk of 32 sits well
clear of the knee and there is no reason to spend more scratch on it. The same
sweep on 4-core x86_64 with synthetic weights peaks at 3.55x.

On the CLI demo this took time-to-first-token from 3.63 s to 667 ms and cut
prefill from 6.23 s to 1.22 s, roughly halving the wall clock of the whole run.

Only the final position pays for the LM head. That projection is
`vocab_size x hidden_size`, the largest matrix in the model, so running every
prompt position through it would cost more than the layers do.

Batched decode and prefill are both **bit-identical** to the paths they replace
on the fixture model (`max|Δ| = 0.00000000`) — ragged batches, block-boundary
crossings, per-sequence sliding windows, ragged final chunks, resumed
mid-prompt prefill after a cache hit, and an explicit causality check that a
position inside a chunk cannot see the ones after it. Only the loop nesting
changed.

```bash
cargo run --release --bin batch_benchmark          # synthetic weights, no download
MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin batch_benchmark
```

The synthetic default is sized to exceed last-level cache on purpose. A
cache-resident model shows almost no benefit from batching and would make the
feature look worthless; a real 1B+ model never fits.

## What prefix caching is worth

`cargo run --release --bin prefix_cache_benchmark`. Model-free: prompt tokens
prefilled and blocks allocated are exact properties of the memory manager, so
these numbers are deterministic and reproduce anywhere.

**64 requests · 240-token shared system prompt · three 480-token documents · 48 unique tokens each**

| metric | no cache | prefix cache | saved |
|---|---:|---:|---:|
| prompt tokens prefilled | 49,152 | 4,752 | **90.3%** |
| KV blocks allocated | 3,072 | 297 | **90.3%** |
| block cache hit rate | — | 97.7% | |

Prefill is O(prompt tokens), so this is roughly a 10× cut in time-to-first-token
for requests that hit.

That headline is a property of *the workload*, not of the engine, so the
benchmark also reports the full range:

| shared prefix | prefilled (no cache) | prefilled (cache) | saved |
|---|---:|---:|---:|
| 0 tokens | 3,072 | 3,072 | **0.0%** |
| 64 tokens | 7,168 | 3,136 | 56.2% |
| 240 tokens | 18,432 | 3,312 | 82.0% |
| 480 tokens | 33,792 | 3,552 | 89.5% |
| 960 tokens | 64,512 | 4,032 | **93.8%** |

With nothing shared there is nothing to reuse, and the cache degrades to the
uncached path at the cost of the lookups.

**Parallel sampling** — KV blocks for *n* continuations of a 1,024-token prompt:

| samples | independent | copy-on-write | saved |
|---|---:|---:|---:|
| 2 | 128 | 64 | 50.0% |
| 4 | 256 | 64 | 75.0% |
| 8 | 512 | 64 | 87.5% |
| 16 | 1,024 | 64 | **93.8%** |

Verified live over HTTP: six concurrent clients sharing a 16-token prefix
produced 5/6 block cache hits and reused 80 of 123 prompt tokens.

---

## Kernel performance

`cargo run --release --bin benchmark`. Structured as an attribution ladder —
each rung adds exactly one change to the one above — so the numbers say what
each optimization is worth rather than only reporting a total.

**Why hand-written SIMD is necessary here:** f32 addition is not associative, so
LLVM may not reassociate a reduction. The obvious `acc += a[i] * b[i]` loop stays
scalar at any optimization level. That constraint, not compiler weakness, is why
these kernels exist.

**16384×8192 f32 (512 MiB working set), 4-core Xeon @ 2.10 GHz, AVX2+FMA**

| kernel | per iter | vs base | vs prev |
|---|---:|---:|---:|
| bf16 convert + serial scalar | 642.5 ms | 1.00× | 1.00× |
| + stream bf16 (no realloc) | 163.9 ms | 3.92× | 3.92× |
| + prepack f32, serial scalar | 159.2 ms | 4.04× | 1.03× |
| + 4 accumulators (serial) | 162.9 ms | 3.94× | 0.98× |
| + AVX2/FMA (serial) | 59.8 ms | 10.75× | **2.73×** |
| + rayon across rows (4t) | 13.7 ms | 47.01× | **4.37×** |
| + int8 weights | 3.4 ms | **191×** | **4.07×** |

Measured on a shared cloud VM; across three runs the per-stage figures ranged
AVX2 2.7–3.2×, rayon 2.9–4.5×, int8 2.7–4.1×, total 120–191×. Run it on your own
hardware rather than trusting these. The naive baseline reallocates 512 MB per
iteration and is the noisiest row.

Two results worth reading carefully:

- **Accumulator unrolling measures ~1.0×, not a win.** A scalar loop is limited
  by load throughput long before the FMA dependency chain becomes the
  bottleneck; multiple accumulators only pay once the loop issues wide FMAs. The
  benchmark reports this rather than claiming a speedup it does not measure.
- **The default 2048×2048 size is misleading.** That is 16 MiB, which fits
  entirely in this machine's 260 MiB L3 — it measures cache bandwidth, not DRAM,
  and no 1B-parameter model fits in cache. The benchmark now detects this and
  says so. Use `BENCH_ROWS`/`BENCH_COLS` to exceed your LLC.

Weight memory: 512.00 MB f32 → 128.06 MB int8 (**4.00×**). The LM head stays f32
deliberately — its quantization error lands directly on token choice, and it is
one matrix rather than seven per layer.

### GPU path (Metal / Vulkan via `wgpu`)

A WGSL compute shader does one output row per 256-thread workgroup, with
`vec4<f32>` loads and an 8-step binary tree reduction in workgroup memory —
the standard GPU parallel-reduction pattern, in WGSL for cross-platform support.

Measured previously on an **Apple M2**, 2048×2048 f32 matvec:

| kernel | per iter | vs CPU |
|---|---:|---:|
| CPU packed + parallel | 1.555 ms | 1.0× |
| GPU scalar WGSL | 0.394 ms | 3.9× |
| GPU vec4 WGSL | 0.330 ms | **4.7×** |

Max `|CPU − GPU|` = 1.6e-4. At ~100 GB/s of M2 bandwidth the theoretical floor is
~0.16 ms/iter, so the vec4 kernel reaches ~48% of peak. This is a standalone
matvec benchmark and is unaffected by the correctness fixes above. On Apple
Silicon the CPU and GPU share physical memory, so `mmap`ed weights are directly
addressable and only the 8 KB activation vector moves per step.

`cargo run --release --bin gpu_benchmark` — exits with a CPU reference number on
headless machines.

### KV eviction under pressure

`cargo run --release --bin eviction_benchmark` — long- and short-lived requests
against a deliberately tight block pool:

| policy | completed | dropped |
|---|---:|---:|
| no eviction | 11 | 93 |
| LRU eviction | **104** | **0** |

---

## Correctness and testing

95 tests, no model download required.

| suite | covers |
|---|---|
| `golden_parity_tests` | full forward pass vs the NumPy reference: decode, prefill, resumed prefill |
| `prefix_cache_parity_tests` | reuse is numerically invisible; CoW isolation under the real model; a colliding suffix with a different history is *not* reused |
| `batched_decode_tests` | batched decode and prefill are bit-identical to the paths they replace: ragged batches, block boundaries, per-sequence windows, ragged chunks, resumed prefill, causality |
| `engine_tests` | the scheduler: token budgets, EOS, block-boundary growth, determinism, seeded sampling, memory-pressure staging, unfittable prompts, int8 |
| `simd_tests` | every kernel vs scalar at every length 0–80, `i8::MIN` sign extension, row independence |
| `math_tests` | both rotary conventions, that they are *not* interchangeable, rotation preserves norm, masking |
| memory unit tests | refcount invariants, 5,000 leak-free cycles, hash chaining, LRU, CoW |

CI runs `fmt`, `clippy -D warnings`, and the suite on **x86_64 (AVX2+FMA)** and
**aarch64 (NEON)**, so a kernel that only works on the machine it was written on
gets caught. A separate job regenerates the golden fixture and diffs it — a
committed reference is only trustworthy if its generator still reproduces it
byte for byte.

---

## Honest limitations

- **Attention does not batch.** Each sequence has its own block table, position
  and KV history, so there is no shared operand to amortize. It is parallelized
  across (sequence, head) pairs instead. This is what caps the batched-decode
  speedup below the batch size.
- **The GPU path covers matvec only.** Attention, RMSNorm and SwiGLU stay on the
  CPU, and every call round-trips through a blocking readback — so it is
  currently a demonstration of a WGSL reduction kernel, not a fast end-to-end
  path.
- **No end-to-end throughput numbers are published here.** The previous README
  carried a batch×steps sweep on an Apple M3, but those runs predate the
  correctness fixes: they were measured with no prefill, the wrong rotary
  convention, and a silent 256-token attention window. They measured something
  that was not the model, so they have been removed rather than restated. The
  same applies to the speculative-decoding acceptance rate, which depends on the
  model producing sensible tokens. `e2e_benchmark` and `speculative_benchmark`
  still run; the numbers need re-measuring on hardware with the weights present.
- **Speculative decoding is drafting plus verification accounting**, not batched
  tree verification.
- **The prefix cache publishes prompt blocks only.** Blocks that fill during
  decode are not published, since their contents depend on sampled tokens the
  hash chain does not cover.

## Repository layout

```
src/
  engine.rs          scheduler: admit → prefill → decode → reclaim
  model.rs           Llama forward pass, weight loading, quantization
  math.rs            RMSNorm, RoPE, SwiGLU, softmax, matvec dispatch
  simd.rs            AVX2/FMA and NEON kernels + scalar reference
  sampling.rs        temperature / top-p / top-k over a seeded RNG
  memory/
    allocator.rs     refcounted physical block pool
    block_table.rs   logical → physical mapping
    prefix_cache.rs  content-addressed KV reuse
    kv_cache_manager.rs  admission, copy-on-write, eviction
    layout.rs        physical KV addressing
  bin/
    http_server.rs         OpenAI-shaped API over the real engine
    benchmark.rs           kernel attribution ladder
    batch_benchmark.rs     batched vs sequential decode
    prefix_cache_benchmark.rs
    e2e_benchmark.rs  eviction_benchmark.rs  gpu_benchmark.rs
scripts/
  gen_golden_fixture.py    reference model + logits (NumPy)
  download_model.py        fetch TinyLlama 1.1B
tests/                     83 tests; fixtures/ holds the reference checkpoint
```

## License

MIT — see [LICENSE](LICENSE).
