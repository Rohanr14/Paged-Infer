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

The merge and activation gate remains at least 15% full-model throughput improvement in a
declared long-context shared-prefix workload, with no more than 5% regression in
the no-sharing fallback. Kernel speedups alone do not satisfy it. Keep the option
off unless measurements justify enabling it for the intended model and traffic.


## Apple M2 measurements, 2026-09-15

The declared real-model case is Llama 3.2 1B with int8 projections, 4,096 prompt
positions, eight sequences, eight measured steps and four Rayon workers. Each
of three repetitions measures both schedules in rotating order after separate
full-range warmup. The physically shared portion is 3,680 positions (requested
90%, rounded to 16-position blocks). All 64 generated tokens and the final logits
match exactly in every run. Inputs are synthetic token IDs with genuine computed
KV, and EOS does not shorten the fixed budget.

These are developer-workstation measurements on Apple M2 / 16 GiB, NEON and Rust
1.93.1 release builds. No other project build or benchmark ran alongside timing;
other system activity was uncontrolled. The raw reports retain all timed samples,
per-step latencies, full outputs, model/input hashes and build source hashes in
[the measurement directory](measurements/shared-prefix-m2/README.md). The measured
source snapshot matches the implementation commit; later edits only move or
clarify tests and document the evidence.

An earlier value kernel measured 0.788x median throughput. Keeping value sums in
registers across each physical block raised the next run to 1.106x. Review then
found that cloning empty output vectors discarded their reserved capacity; the
final harness allocates each buffer separately before timing. These earlier
reports are retained as development history, not pooled with the final build.

The final kernel-only sweep covers all 36 requested cases with 21 timed pairs
per case and finite bit-identical outputs throughout. At context 4,096 and 90%
sharing it measured 1.129x / 1.232x / 1.193x for batches 4 / 8 / 16. Those figures
exclude model projections and cannot establish the full-model gate.

| Final harness, physical sharing | Baseline tokens/s | Candidate tokens/s | Ratio of medians | Baseline / candidate pooled step p95 |
|---|---:|---:|---:|---:|
| 90% requested | 22.66 | 27.29 | 1.205x | 507 / 345 ms |
| 0% control | 18.82 | 19.34 | 1.028x | 1,186 / 471 ms |

The shared run's three paired speedups are 1.216x, 1.114x and 1.097x: the median
paired improvement is 11.4%, below the target. The no-sharing control ranges from
0.812x to 2.014x despite both variants executing the same fallback kernel. Its
large variation makes the apparent 2.8% improvement and tail difference evidence
of timing noise, not a fallback optimization. No timed samples were discarded.
These small samples do not establish either a repeatable 15% gain or a reliable
5% upper bound on fallback regression. The feature stays off and the PR remains
a draft; the gate is **not yet established**.

The final shared runs retain 8.25 MiB of additional packed scratch. Actual KV
storage is 446 MiB for 90% sharing and 2,056 MiB for the no-sharing control; these
are storage-sharing effects present in both attention variants. Process maximum
RSS was 3.11 GiB / 3.96 GiB respectively, measured by macOS `time -l` across each
complete process, including checkpoint loading, preparation, warmup and both
variants. It is not a per-variant RSS comparison. The control's additional packed
scratch and shared layer-call count are zero. All twelve final real-model runs
match the same complete token output and final-logit hash across both sharing
fractions.

## Focused next step

1. Profile the declared eight-sequence, 4,096-position case to separate shared
   score calculation, value accumulation, packing/scatter and model projections.
   Use that breakdown to choose one further optimization.
2. Repeat longer paired runs on a quiet machine, retaining every sample and
   reporting paired ratios as well as ratios of medians. Establish both the
   shared-workload gain and the no-sharing bound before merging or enabling.
3. If the model-level result clears that gate, verify the actual request workload
   with replay, including prompt preparation, throughput, delivery tails and
   memory pressure. The current steady-state benchmark excludes those effects.
