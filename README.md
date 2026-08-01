# Paged-Infer

**An LLM inference engine written from scratch in Rust: PagedAttention, automatic prefix caching, copy-on-write forking, batched decode and prefill, lossless speculative decoding, and hand-written SIMD kernels.**

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
cargo test                                        # 144 tests, incl. golden parity
cargo run --release --bin benchmark               # kernel attribution ladder
cargo run --release --bin batch_benchmark         # batched vs sequential decode
cargo run --release --bin prefix_cache_benchmark  # what prefix caching is worth
cargo run --release --bin speculative_benchmark   # lossless speculative decoding
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

# Tokens as they are produced, in OpenAI's server-sent-event format.
curl -N localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Explain paged attention."}],
  "max_tokens": 128, "stream": true
}'

curl localhost:8080/metrics   # Prometheus text format
curl localhost:8080/stats     # the same counters as JSON
```

The engine reads the architecture from the checkpoint's own `config.json`, so
any HuggingFace-format Llama checkpoint works, not just TinyLlama.
`DRAFT_TOKENS=4` turns on speculative decoding, `QUANT=int8` quantizes the
projections, and `WARMUP=0` disables the startup pass.

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

`cargo run --release` submits three requests sharing one system prompt; the
third asks for four continuations, which are forked rather than re-prefilled.

```
Mapped 22 layers, 4.14 GB of F32 weights. SIMD backend: neon.
KV cache: 352.00 MB across 512 blocks of 16 tokens.

[req 0 / seq 0] 32 tokens, Length, ttft 5.18s
[req 1 / seq 1] 32 tokens, Length, ttft 188.71ms
  Answer: Continuous batching is a technique that allows the system to process
  batches of data in parallel, rather than in a single thread.

[req 2 / seq 2] 32 tokens, Length, ttft 163.28ms
  Questioner: Explain the reasons behind grouping queries into sub-sets when
  computing the attention weights in the neural machine translation model.
[req 2 / seq 3] ... Grouped-query attention is a way to use attention mechanisms to
[req 2 / seq 4] ... Grouped-query attention is a more recent attention mechanism
[req 2 / seq 5] ... Grouped-query attention is a type of attention mechanism often

Completed 6 sequences in 9.18s over 31 steps.
  prompt tokens     : 511 total, 191 prefilled, 320 reused from cache
  generated tokens  : 192
  prefix cache      : 20/21 block lookups hit (95.2%)
  copy-on-write     : 3 block copies
  prefill 5.53s / decode 3.64s (52.68 tok/s)
```

The TTFT column is the prefix cache doing its job. Request 0 pays 5.18 s to
build the system prompt's KV; requests 1 and 2 inherit those blocks and start in
**188 ms and 163 ms — about 30x faster**. Across the run, 320 of 511 prompt
tokens never went through the model at all.

The four sequences under request 2 come from a single prefill and diverge into
visibly different continuations. The three copy-on-write copies are the one
partial block they all wrote into; the blocks before it stayed shared.

Output is TinyLlama 1.1B being TinyLlama — fluent, and wrong about what
PagedAttention is.

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

### Tiling the batched matmul

The batched matmul had the wrong loop nesting, and it was the real ceiling on
batched decode:

```rust
for b in 0..batch {
    out[b] = dot(w_row, x[b]);   // w_row reloaded, once per batch entry
}
```

Every element of `w_row` is loaded into a register, multiplied against **one**
activation vector, and discarded — then loaded again for the next. That is two
loads to feed each fused multiply-add, one for the weight and one for the
activation. A core that retires two loads and two FMAs per cycle is then
waiting on its load ports while its arithmetic sits idle, and no amount of
memory bandwidth helps: the kernel has left the bandwidth-bound regime and
become issue-bound. That, not attention, is what flattens the batching curve.

`dot_multi` loads each weight element once and multiplies it into `TILE`
independent accumulator sets, cutting the ratio from `2` loads per FMA toward
`1 + 1/TILE`. The int8 path gets the same treatment and gains more from it: an
`i8` weight must be loaded, sign-extended and converted before it can be
multiplied, and the untiled nesting repeated all three per batch entry.

`TILE` is bounded by the register file rather than by preference — `dot_multi`
keeps `4 * TILE` accumulators live, and spilling them hands back the loads it
saved. Four fits aarch64's 32 vector registers comfortably and AVX2's 16
tightly.

**4-core x86_64 (AVX2+FMA), `batch_benchmark`, fastest of 3:**

| weights | context | batch | untiled | tiled | batched speedup |
|:--|---:|---:|---:|---:|---:|
| f32 | 128 | 8 | 171.7 tok/s | 187.3 tok/s | 3.28× → 3.53× |
| f32 | 128 | 16 | 181.5 tok/s | 204.5 tok/s | 3.52× → 3.92× |
| f32 | 128 | 32 | 243.9 tok/s | 262.2 tok/s | 4.68× → **4.97×** |
| int8 | 1024 | 8 | 145.4 tok/s | 149.6 tok/s | 2.90× → 3.09× |
| int8 | 1024 | 16 | 162.9 tok/s | **185.4 tok/s** | 3.22× → **3.81×** |

7–14% throughput for a loop reordering, and the gain grows with batch size
because that is how many activation vectors each weight load now serves.

This is also a scheduling change and nothing more. `dot_multi::<1>` keeps the
same four accumulators, the same 32-element chunking, the same
`(acc0+acc1)+(acc2+acc3)` reduction and the same scalar-tail order as `dot`, so
it *is* `dot`, element for element — `tests/simd_tests.rs` asserts equality at
every length from 0 to 129 and every tile width from 1 to 6, for both the f32
and int8 kernels. A tile that collapsed to one accumulator per output would
have been faster still and would have quietly broken the bit-identity the
batched path is claimed to have.

`PAGED_INFER_MATMUL_TILE=1` turns tiling off, which is how the table above was
measured: same binary, same run, one environment variable.

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

### Paged attention

Batching cannot make attention cheaper. Every other operand in a decode step is
shared — one weight matrix serves the whole batch — but the KV cache is not:
each sequence has its own history, its own position, its own scattered blocks.
There is nothing to amortize.

What it can do is stop attention costing more than it has to. Two costs in the
obvious implementation are pure overhead:

**Grouped-query replication.** Under GQA, `kv_group` query heads share one
key/value head — eight of them on TinyLlama. Parallelizing over
`(sequence, query head)` gives each of those eight its own task, and each task
streams *the same* K and V vectors out of the cache independently. Grouping
them into one task reads it once.

**Per-token address translation.** A token's physical slot is
`block_table[t / block_size]` plus `t % block_size`: a division, a modulo and a
bounds-checked lookup — per token, twice per pass, per head, per layer. The
mapping is constant *within* a block, so it belongs outside the token loop.

`src/attention.rs` does both, and picks how many query heads share a task from
the batch size: wide lanes read the cache least, narrow lanes produce more tasks
to keep cores busy.

**The kernel alone**, TinyLlama's attention shape, speedup over
one-task-per-query-head (`attention_benchmark`):

| context | batch | Apple M-series, 8 threads (NEON) | 4-core x86_64 (AVX2+FMA) |
|---:|---:|---:|---:|
| 256 | 1 | 1.65× (d=4) | 1.16× (d=4) |
| 256 | 8 | 1.50× (d=4) | 1.09× (d=8) |
| 1024 | 1 | 1.27× (d=8) | 1.47× (d=4) |
| 1024 | 8 | 1.39× (d=8) | 1.57× (d=8) |
| 4096 | 1 | 1.41× (d=8) | 1.38× (d=4) |
| 4096 | 4 | 2.48× (d=8) | — |
| 4096 | 8 | **2.65× (d=8)** | 1.86× (d=8) |

The width in brackets is the one that actually won. Two things are worth
reading off this table.

**The widest lane is not always best.** On the 4-core machine at batch 1, `d=8`
gives four tasks for four threads, rayon has nothing left to steal, and the
schedule runs at the speed of its slowest core — `d=4` beats it by 30%. Reading
the cache once is worthless if half the cores are idle waiting.

**The optimum is a property of the machine, not just the shape**, and the two
machines disagree in a way no single rule can reconcile. On the 8-thread
machine `d=4` wins at 256 tokens of context and `d=8` wins at 1024 — *at every
batch size*. A scheduler that sees only the batch size therefore cannot be
right in both cells, which is a proof rather than a tuning failure. What it can
do is be right where it counts: the shipped rule (two tasks per thread) picks
the measured optimum in every cell at batch ≥ 4 and context ≥ 1024 on both
machines, and gives up 6–19% of the *kernel* at batch 1 or short context —
precisely the cells where attention is a negligible share of the step.
`PAGED_INFER_ATTN_LANES_PER_THREAD` overrides it; `attention_benchmark` is how
you would pick a value.

#### What it is worth end to end, which is much less

Up to 1.86× on the kernel buys **2–3%** of a decode step at f32 (4-core x86_64,
`batch_benchmark`, fastest of 3):

| weights | context | batch | per-head | grouped | batched speedup |
|:--|---:|---:|---:|---:|---:|
| f32 | 128 | 8 | 178.3 tok/s | 171.4 tok/s | 3.30× → 3.29× |
| f32 | 1536 | 8 | 142.5 tok/s | 145.9 tok/s | 3.02× → 3.05× |
| f32 | 4096 | 8 | 114.0 tok/s | 116.7 tok/s | 2.63× → 2.77× |
| **int8** | 4096 | 8 | 106.5 tok/s | **115.5 tok/s** | 2.46× → **2.61×** |

And on Apple Silicon (8 threads, real TinyLlama, 22 layers, int8, 2048 tokens of
context) — the regime where the step has left the bandwidth roof entirely:

| | per-head | grouped | |
|:--|---:|---:|---:|
| batched decode | 39.27 tok/s | **43.44 tok/s** | **+10.6%** |
| batched speedup | 1.37× | **1.48×** | |

Working backwards from the f32 rows, attention is only about **5% of a decode
step** even at 4096 tokens of context — so a 1.86× kernel is worth 2%. The int8
row is the same arithmetic with the weights cut 4×: attention becomes a much
larger share of what is left, and the same change is suddenly worth 8.5%. The
value of an attention optimization is set almost entirely by what it is being
compared against.

**This corrects a claim this README used to make.** The old text said attention
not batching "is what caps the batched-decode speedup below the batch size."
That is wrong. At 128 tokens of context attention is around 1% of the step, and
removing it entirely would not move the ceiling. What actually caps it is the
projection kernel: batching gives it an arithmetic intensity of `B/2`
flop-per-byte, so it crosses from bandwidth-bound to issue-bound at roughly
`B = 2C/BW` — batch 4–5 on both machines measured, which is exactly where the
measured curve flattens. Attention is what caps it at *long* context and small
weights, not in general.

None of it changes a single arithmetic operation. Scores are still computed in
increasing `t`, the softmax still runs over the whole window, the value
accumulation still runs in increasing `t`. `tests/attention_tests.rs` holds the
kernel against a deliberately naive reference written in the test file — one
head at a time, one token at a time, addresses resolved per token — and demands
**bit-identical** output across six head configurations, every legal lane width,
ragged batches, sliding windows narrower than a block, unmapped blocks, and
several positions of one sequence sharing a block table. The benchmark asserts
it too, on every run, before printing a number.

```bash
cargo run --release --bin attention_benchmark   # sweeps the lane width
```

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

144 tests, no model download required.

| suite | covers |
|---|---|
| `golden_parity_tests` | full forward pass vs the NumPy reference: decode, prefill, resumed prefill |
| `prefix_cache_parity_tests` | reuse is numerically invisible; CoW isolation under the real model; a colliding suffix with a different history is *not* reused |
| `batched_decode_tests` | batched decode and prefill are bit-identical to the paths they replace: ragged batches, block boundaries, per-sequence windows, ragged chunks, resumed prefill, causality |
| `engine_tests` | the scheduler: token budgets, EOS, block-boundary growth, determinism, seeded sampling, memory-pressure staging, unfittable prompts, int8 |
| `speculative_tests` | speculative output is token-identical to greedy: draft depths, mixed batches, chunk splits, block boundaries, EOS mid-run, memory pressure |
| `streaming_tests` | streamed deltas reconstruct the completion exactly, under speculation and forking; every sequence gets one terminal delta; cancellation returns blocks; warm-up leaves no trace |
| `attention_tests` | the paged attention kernel is bit-identical to a naive per-head, per-token reference at every lane width: six head shapes, ragged batches, sub-block sliding windows, unmapped blocks, shared block tables |
| `simd_tests` | every kernel vs scalar at every length 0–80, `i8::MIN` sign extension, row independence |
| `math_tests` | both rotary conventions, that they are *not* interchangeable, rotation preserves norm, masking |
| memory unit tests | refcount invariants, 5,000 leak-free cycles, hash chaining, LRU, CoW |

CI runs `fmt`, `clippy -D warnings`, and the suite on **x86_64 (AVX2+FMA)** and
**aarch64 (NEON)**, so a kernel that only works on the machine it was written on
gets caught. A separate job regenerates the golden fixture and diffs it — a
committed reference is only trustworthy if its generator still reproduces it
byte for byte.

---

## Speculative decoding

The optimizations above all trade on having *more* work available — more
sequences, more positions. None of them help one user waiting on one stream.
Speculative decoding does.

A cheap drafter guesses `K` tokens; the model checks all `K+1` positions in one
batched pass. Because decoding is memory-bound, that pass costs barely more than
producing a single token, so every accepted draft is nearly free. The
verification primitive is exactly the batched decode above, with the batch being
positions of one sequence.

**It is lossless, not an approximation.** A draft is accepted only where it
equals what the model itself would have chosen, and the token after the last
accepted one comes from the model's own logits — so the output is *exactly* the
greedy output. `tests/speculative_tests.rs` asserts that token-for-token across
draft depths, mixed batches, chunk splits, block boundaries, EOS landing inside
an accepted run, and memory pressure. Sampled sequences do not speculate at all:
acceptance-by-argmax is a greedy rule, and applying it to a sampled sequence
would quietly change its distribution.

The drafter is **prompt lookup** — find the longest recent suffix that occurred
earlier in the context and propose what followed it. No draft model, so nothing
extra to load or keep resident. It works because serving workloads copy
constantly: summarize this, answer from this passage, rewrite this function.

```bash
cargo run --release --bin speculative_benchmark   # runs without a checkpoint
MODEL_PATH=models/tinyllama-1.1b/model.safetensors cargo run --release --bin speculative_benchmark
```

Acceptance is a property of the workload, so the benchmark reports several
side by side — including the open-ended case, where there is nothing to copy and
speculation costs a little and saves nothing. It asserts on every run that the
token count is unchanged.

### Measured

TinyLlama 1.1B in f32 on an Apple MacBook Air (NEON), greedy, 64 tokens per
prompt. `tok/step` is tokens emitted per sequence-step — ordinary decoding is
exactly 1.00.

**Verbatim copy** — "repeat this passage exactly", 85-token prompt. The output
*is* the input, the best case prompt lookup can ever see:

| drafts | accepted | tok/step | decode | speedup |
|-------:|---------:|---------:|-------:|--------:|
| 0 (off) | –       | 1.00     | 4.752 s | 1.00× |
| 2      | 100.0%   | 3.00     | 1.670 s | 2.85× |
| 4      | 100.0%   | 4.85     | 1.249 s | 3.80× |
| 8      | 100.0%   | 9.00     | 1.094 s | **4.34×** |

**Extractive QA** — "answer using only this passage", 91-token prompt. The
answer quotes the passage but is not the passage:

| drafts | accepted | tok/step | decode | speedup |
|-------:|---------:|---------:|-------:|--------:|
| 0 (off) | –       | 1.00     | 2.423 s | 1.00× |
| 2      | 50.0%    | 1.28     | 2.421 s | 1.00× |
| 4      | 45.0%    | 1.39     | 1.807 s | **1.34×** |
| 8      | 22.5%    | 1.39     | 2.148 s | 1.13× |

**Open-ended** — "explain why decoding is memory-bound", 27-token prompt.
Nothing to copy:

| drafts | accepted | tok/step | decode | speedup |
|-------:|---------:|---------:|-------:|--------:|
| 0 (off) | –       | 1.00     | 4.747 s | 1.00× |
| 2      | 55.6%    | 1.09     | 4.398 s | 1.08× |
| 4      | 29.4%    | 1.09     | 5.034 s | **0.94×** |
| 8      | 16.1%    | 1.09     | 4.626 s | 1.03× |

Three things in these tables are worth more than the headline 4.34×.

**Deeper drafting stops helping at a workload-specific depth, and the tables say
exactly where.** Look down the `tok/step` column, not the acceptance column.
Open-ended is pinned at 1.09 for every `K`, while acceptance falls 55.6% → 29.4%
→ 16.1% — almost exactly halving as `K` doubles. That is the signature of a
*constant* number of accepted tokens divided by a doubling number of drafted
ones: past the point where the model and the drafter diverge, every additional
draft is a position that gets embedded, rotated, attended over and pushed
through the LM head to be thrown away. Extractive QA saturates the same way at
1.39. Verbatim copy never saturates — `tok/step` tracks `K+1` exactly (3.00,
4.85, 9.00) because the model agrees with the drafter indefinitely.

**Acceptance rate is not the metric.** Open-ended shows 55.6% acceptance at
`K=2` — higher than extractive QA's 50% — and buys almost nothing, because the
drafter only *fires* on a small fraction of steps there. A high acceptance rate
on rare guesses is worth less than a mediocre one on frequent guesses. `tok/step`
conflates the two correctly; acceptance alone flatters the hard case.

**`K=4` on open-ended is 0.94×** — speculation made it slower. That is the
honest cost of a wrong guess, and it is why `draft_tokens` defaults to 0 and why
the benchmark sweeps `K` rather than shipping a favourite value.

*On measurement noise:* acceptance and `tok/step` are deterministic — the same
prompt drafts and accepts identically every run. Wall clock is not. Repeats of
one configuration on this laptop have landed 24% apart, wide enough to reorder
`K=2` and `K=4` between runs. The benchmark therefore reports the **fastest of
three** runs per configuration (`SPEC_REPEATS`) and prints the widest observed
spread underneath each table, so a gap narrower than the noise can be recognized
as one. The numbers above predate that change and are single measurements.

## Serving

The HTTP front end is not a wrapper that re-implements batching; it holds the
engine on one thread and funnels every connection into the same scheduler, so
requests from *different clients* land in the same batch and share prefix-cache
blocks.

- **Streaming.** `"stream": true` returns OpenAI-shaped server-sent events, one
  chunk per scheduler step. A speculative step that had four drafts accepted
  emits four tokens in one chunk — they really are all available at that
  instant. Detokenization is incremental but not per-token: a BPE token can
  carry a fragment of a UTF-8 character, so the whole prefix is decoded each
  time and only the newly-appeared text is sent.
- **Cancellation.** A client that hangs up is noticed on the next chunk write,
  and the engine stops: the sequence retires at the next step and its KV blocks
  go back to the pool. Measured against the fixture, a stream abandoned after
  four chunks stopped at 25 generated tokens out of a requested 4000, and the
  pool returned to 511 of 512 blocks free.
- **Warm-up.** `Engine::warm_up()` runs one throwaway prefill before the
  listener opens, so the rayon pool, the scratch arenas and the checkpoint's
  pages are all touched by something other than the first real request. It then
  returns the cache and every counter to its initial state — `tests/streaming_tests.rs`
  asserts a warmed engine is indistinguishable from a fresh one. `WARMUP=0`
  disables it, which is how you measure what it was worth.
- **Metrics.** `/metrics` speaks Prometheus text format (counters carry
  `_total`, so `rate()` works); `/stats` returns the same numbers as JSON.

## Honest limitations

- **Attention still cannot amortize across sequences**, because there is no
  shared operand to amortize — each sequence has its own KV. The kernel removes
  the redundant work (see *Paged attention*), but the floor it leaves is real.
- **The batched matmul is tiled across the batch but not across output rows.**
  Tiling rows as well would cut activation loads the way tiling the batch cut
  weight loads, but `4 * TILE` accumulators per output already crowd the AVX2
  register file, so it needs the accumulator count per output to drop from four
  to one — which would end the bit-identity with `dot`.
- **The attention kernel does not split the token axis.** With few sequences and
  a multi-query model there can be fewer tasks than cores, and the lane
  scheduler's only remedy is to narrow lanes, which costs traffic.
  Flash-decoding-style splitting with a log-sum-exp combine would fix that, at
  the price of no longer being bit-identical.
- **The GPU path covers matvec only.** Attention, RMSNorm and SwiGLU stay on the
  CPU, and every call round-trips through a blocking readback — so it is
  currently a demonstration of a WGSL reduction kernel, not a fast end-to-end
  path.
- **Speculative decoding verifies a single draft sequence**, not a tree. Tree
  verification explores several branches per pass and lifts acceptance further —
  which is exactly where the extractive-QA numbers above run out of road.
- **Prompt lookup rarely fires on novel text.** That is the real ceiling on the
  open-ended row — not low acceptance, but no draft offered at all on most
  steps. Fixing it needs a drafter that can propose tokens the context has never
  contained, which prompt lookup by construction cannot.
- **Only streaming clients can be cancelled.** Disconnection is detected on a
  failed chunk write, and a buffered request writes nothing until it is
  finished, so there is no write to fail on.
- **Sampled sequences never speculate.** Extending it there needs the
  rejection-sampling correction to stay distributionally exact.
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
  attention.rs       paged attention: GQA head grouping, block-wise addressing
  detokenizer.rs     incremental token → text for streaming
  speculative.rs     drafters and lossless greedy verification
  bin/
    http_server.rs         OpenAI-shaped API, SSE streaming, Prometheus
    attention_benchmark.rs lane-width sweep for the paged attention kernel
    benchmark.rs           kernel attribution ladder
    batch_benchmark.rs     batched vs sequential decode
    prefix_cache_benchmark.rs
    e2e_benchmark.rs  eviction_benchmark.rs  gpu_benchmark.rs
scripts/
  gen_golden_fixture.py    reference model + logits (NumPy)
  download_model.py        fetch TinyLlama 1.1B
    speculative_benchmark.rs   acceptance and speedup per workload
tests/                     144 tests; fixtures/ holds the reference checkpoint
```

## License

MIT — see [LICENSE](LICENSE).
