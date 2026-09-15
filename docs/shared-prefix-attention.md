# Shared-prefix attention prototype

The optional decode kernel reuses physical KV loads across up to four query
entries while retaining GQA head grouping for private history. It is disabled
by default. `EngineConfig.shared_prefix_attention` enables it, or set
`SHARED_PREFIX_ATTENTION=1` for the HTTP binary. Unrelated batches use the existing
kernel, and prefill always uses the existing path.

## Computation and ownership

One common physical prefix is supported per decode batch. A borrowed plan checks
block identity at the same logical position, stops at the first divergence and
excludes every query's current write block. The common interval is intersected
with all query windows. Each query's earlier window positions and trailing
private history are computed separately. Unmapped positions remain masked.

The plan is rebuilt for every batched forward pass after scheduling and COW have
settled the mappings. Its table borrows live only through that forward pass;
the plan can be reused across its layers, but never across scheduler steps.
No block identities are stored in scratch. Cancellation, allocation recycling
and speculative rollback therefore cannot leave a stale grouping behind.

The SIMD score primitive loads one K vector for several queries. The value
primitive shares V loads and keeps each output vector in registers across a
physical block, preserving per-coordinate token order. Private histories keep
GQA heads together so grouping across requests retains the existing locality.
Task width narrows when needed to retain parallel work for the configured threads.

Every query retains its complete score window and its own softmax. Scores are
written to their original logical positions, and values accumulate in increasing
token order: private leading interval, common interval, private trailing interval.
Zero-probability updates are skipped exactly as in the fallback, including when
underflow, signed zero or masked nonfinite values are involved.

This is a CPU adaptation of inter-sequence KV reuse, described in
[Hydragen](https://arxiv.org/html/2402.05099v2). Unlike its partitioned attention
approach, this prototype keeps full-window normalization, so it needs no merge
of independently normalized prefix/suffix outputs. This preserves the existing
arithmetic on the same backend at the cost of retaining full-window score storage.
It makes no claim of identical arithmetic across different SIMD architectures.

## Controls and evidence

```bash
cargo run --release --bin workload_replay -- \
  --workload workloads/shared-prefix.json --threads 4 \
  --config workloads/configs/baseline.json \
  --config workloads/configs/shared-attention.json \
  --repeats 3 --verify --output /tmp/shared-prefix-replay.jsonl
```

Replay reports and HTTP `/stats` expose `shared_attention_layer_calls` and
`shared_attention_query_tokens`. The latter counts common attended positions
times query entries, summed across executed layers; it is not a measured count
of memory loads or model output tokens. Prometheus adds a `paged_infer_` prefix
and `_total` suffix to these counters.

`shared_attention_scratch_bytes` reports retained additional packed scratch
capacity, including tile padding. Replay places it in `memory`; HTTP exposes it
as a gauge. It excludes other model scratch, weights, KV, allocator overhead and
process RSS. Scratch capacity survives reset, while execution counters reset.
Old reports load with zero defaults for these fields, meaning unavailable
historical measurements rather than proof that the path was unused.

## Reproduce performance measurements

The kernel sweep includes batches 1/4/8/16, contexts 256/1,024/4,096 and physical
sharing fractions 0/50/90%. It checks every output bit and retains all paired
timings, rotating baseline/candidate order. Candidate timing includes planning;
input construction and warmup are outside the timer.

```bash
RAYON_NUM_THREADS=4 SHARED_ATTN_REPS=21 \
  cargo run --release --bin shared_attention_benchmark > /tmp/shared-kernel.jsonl
```

For full-model steady-state decode, the second benchmark computes genuine prompt
KV once, shares or copies whole blocks to control physical sharing, then appends
a distinct private token for each sequence. Its inputs are recorded deterministic
token IDs, not a language-quality evaluation. Every run restarts from the same
prepared history and overwrites future KV causally. Complete greedy outputs and
final logit hashes must match across all configurations and repeats.

```bash
MODEL_PATH=models/llama3-reference/checkpoint/model.safetensors QUANT=int8 \
RAYON_NUM_THREADS=4 SHARED_DECODE_CONTEXT=4096 SHARED_DECODE_BATCH=8 \
SHARED_DECODE_STEPS=8 SHARED_DECODE_REPS=3 SHARED_DECODE_PERCENTAGE=90 \
  cargo run --release --bin shared_decode_benchmark > /tmp/shared-decode.jsonl
```

Repeat with `SHARED_DECODE_PERCENTAGE=0` for the physical no-sharing control.
Every transformer layer, the vocabulary projection and greedy selection are
timed. Loading, preparation, allocation warmup, admission and HTTP are excluded.
EOS is treated as an ordinary token to hold the decode budget fixed. Reported
step latency is model-batch latency, not client-observed streaming latency.
The manifest records these limits and the input, model and build fingerprints.

The activation gate remains at least 15% full-model throughput improvement in a
declared long-context shared-prefix workload, with no more than 5% regression in
the no-sharing fallback. Kernel speedups alone do not satisfy it. Keep the option
off unless measurements justify enabling it for the intended model and traffic.
