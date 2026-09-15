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

On NEON, the measured Llama shape (32 query heads, eight KV heads, dimension 64)
also pairs adjacent shared keys when the attended common prefix is at least
2,048 tokens long. A two-query by two-key tile reuses query vectors while keeping
four independent dot products, each with the existing accumulator and reduction
order. Pairs stay inside physical blocks; odd keys and single-query tile tails
use the established score loop. Other shapes and shorter common prefixes keep
that loop because broader score tiling regressed those measured cases. This
restriction is experimental, not a speed guarantee for every qualifying input.

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

The default shape has 32 query heads, four KV heads and head dimension 64.
Set `SHARED_ATTN_KV_HEADS=8` for the Llama 3.2 1B attention shape. This control
accepts one positive divisor of 32; the manifest records the actual shape.
Use both shapes when evaluating changes so a gain for one does not hide a
regression for the other.

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

## Profiling and paired statistics

Add `--features profiling` to the release Cargo command above to collect
diagnostic stage timings. Shared-decode, kernel-sweep and replay manifests mark
these builds with `profiling_enabled: true` and
`performance_gate_eligible: false`. Timer and counter overhead can change the
result, so collect performance evidence separately with profiling disabled.
Ordinary builds report `profiling_enabled: false`; their shared-decode `profile`
arrays are empty. Older artifacts without these fields must be interpreted using
their recorded source and build fingerprints.

The shared-decode benchmark resets profiling counters after each full-range
warmup and captures a `profile` snapshot after the measured decode loop.
`model_main` spans cover model stages such as projections and attention.
`attention_main` setup/scatter spans and `attention_workers` lane spans sit
inside model attention. Worker totals sum overlapping elapsed spans, including
scheduling delay; they are not CPU time or exclusive wall time. Do not add these
scopes together. Counters are process-global, so resetting or interpreting them
also requires that other inference work is absent and workers have joined.

Shared-decode summaries retain the existing ratio of separate elapsed-time
medians and add `paired_elapsed_time_speedup`: every same-repeat
baseline/candidate elapsed-time ratio, plus its median, minimum and maximum.
Ratios above one favor the candidate. The paired median uses the midpoint of
both middle ratios for even sample counts; the legacy marginal p50 fields keep
their nearest-rank convention.

The reported 95% interval uses 10,000 deterministic paired bootstrap resamples.
Each draws the original number of complete repeat pairs with replacement and
recomputes the median ratio; nearest-rank 2.5th and 97.5th percentiles provide
the endpoints. The artifact records the fixed seed, generator and estimator.
One pair produces no interval, and fewer than ten pairs trigger a caution;
ten is not a validity threshold. This assumes independent, exchangeable repeat
pairs and does not correct thermal drift, serial dependence or order effects.
Small samples can under-cover, and narrow or zero-width intervals do not
guarantee future gains. Increase `SHARED_DECODE_STEPS` for longer timed runs and
`SHARED_DECODE_REPS` for more pairs; steps within one run are not independent
bootstrap observations. Retain every sample and compare equivalent workloads
and build fingerprints.

## Cold-prefill follow-up, 2026-09-15

Per-run replay profiling now covers the instrumented batched CPU stages,
including the final prefill vocabulary projection. Counters exclude warmup
and reset between configurations and repeats; ordinary builds emit empty
profiles. Older version-1 reports still compare normally. These diagnostic
timers are excluded from performance-gate builds.

A cold 2,049-token Llama 3.2 1B int8 request spent 70.72% of recorded model
elapsed time in feed-forward projections, and 87.56% in all transformer
projections. That selected a bounded NEON experiment: reuse converted weights
across six activation vectors instead of four, preserving exact arithmetic.
Three fixed ordinary comparison pairs produced four/six elapsed ratios of
0.987x, 1.030x and 0.953x. The candidate slowed two pairs and failed the
predeclared consistency check, so it was reverted. The default remains four.

The [complete cold-prefill evidence](measurements/cold-prefill-m2-tile6/README.md)
preserves every run, source patches, exact output checks, the fixed protocol
and a reproducible analysis. These single-request cold measurements neither
replace the shared-decode evidence below nor establish its merge gate.

## Latest Apple M2 decode measurements, 2026-09-15 (`1ad58be`)

The score-tile follow-up adds query reuse for the narrow NEON shape and prefix
range described above. The retained value-loop build was profiled again: shared
scores occupied about 54% of summed attention-worker elapsed time and shared
values about 19%. These overlapping diagnostic spans motivated the experiment;
they do not predict its model-level gain.

Three alternating source comparisons were run for each of two attention shapes,
first with unrestricted score tiling, then with shape-only dispatch, then with
the final shape-and-prefix restriction. Every sweep includes all 36 kernel cases
and 21 timing pairs per case. Short contexts and four-KV-head cases exposed
regressions, which motivated retaining the previous score loop there. In the
final primary eight-KV-head/context-4,096/batch-eight/90% case, candidate shared
attention was faster than the retained build in two comparisons and roughly
tied in the third. Fallback timing also varied. All samples and intermediate
patches remain in [the score-tile evidence directory](measurements/shared-prefix-m2-score-tile/README.md).

The final ordinary build then used the unchanged real-model protocol: Llama 3.2
1B int8, context 4,096, batch eight, four workers, sixteen decode steps and twelve
paired repeats for each physical-sharing fraction.

| Physical sharing | Median paired speedup | Descriptive bootstrap 95% interval | Baseline / candidate pooled batch-step p95 |
|---|---:|---:|---:|
| 90% requested | 1.182x | [1.040, 1.230] | 293 / 401 ms |
| 0% control | 0.996x | [0.939, 1.032] | 431 / 434 ms |

All 48 timed runs have the same complete 128-token output and final-logit hash.
The shared point estimate exceeds the 15% target, but the interval includes much
smaller gains and the candidate's p95 worsens. The control's median slowdown is
about 0.4%; its interval still includes regressions beyond 5%. Other desktop CPU
activity was present, with no concurrent project build or benchmark. No samples
were discarded. These measurements do not yet establish a repeatable gain and
regression bound sufficient to merge or enable the feature.

The separate real-model request replay completes four runs of two eight-request
bursts, with a 2,048-token common prefix, private suffixes and one cancellation.
Every run has identical tokens and finish reasons, 15 successful requests and
one cancellation after eight emitted tokens. Each reaches eight active sequences
and reuses 30,720 prompt tokens. The shared variant executes 608 shared-attention
layer calls; the baseline executes none.

Across its two repeat pairs, overall useful throughput changes by about -0.9%
and +0.3%. Cold prefill dominates the run. The warm burst's useful throughput
improves by about 4.9% and 16.6%, and overall observed inter-token p95 improves from
207 / 217 ms to 191 / 192 ms. These are descriptive results from two pairs, not
a reliable speed bound. Warm-burst throughput includes queueing and private
prefill: successful warm output tokens divided by the interval from the first
warm submission to the last warm completion. No prefill time is subtracted.

All runs peak at 142 of 192 KV blocks and retain 128 cached prefix blocks after
completion. The shared variant retains about 4.25 MiB of additional scratch.
There are no OOMs, rejections, preemptions or COW copies. This validates the
bounded-memory lifecycle and cancellation path; the pool does not force memory
pressure. Arrivals follow scheduler steps, not wall-clock deadlines, and model
loading and initial warmup remain outside the request timer. This 2,048-token
request case is distinct from the 4,096-position steady-state model benchmark.

## Earlier Apple M2 measurements, 2026-09-15 (`cf7bbe1`)

Profiling the declared Llama 3.2 1B int8 case put attention at 41.4% of
candidate model wall time. Shared scores and shared value accumulation accounted
for 46.3% and 28.5% of summed attention-worker elapsed spans. These overlapping
diagnostic scopes motivated a value-loop change: retain sixteen coordinates per
output row in SIMD registers and reuse each weight load, mask check and broadcast
across them. Token-order FMA arithmetic remains unchanged.

The ordinary build then measured twelve paired repeats of sixteen decode steps
at context 4,096, batch eight and four workers. Both sharing fractions use the
same model and input histories; the shared case has 3,680 common positions.

| Physical sharing | Median paired speedup | Descriptive bootstrap 95% interval | Baseline / candidate pooled batch-step p95 |
|---|---:|---:|---:|
| 90% requested | 1.135x | [1.107, 1.158] | 332 / 377 ms |
| 0% control | 0.990x | [0.950, 1.027] | 410 / 478 ms |

All 48 timed runs pass complete output and final-logit checks. All 36 kernel
cases also retain finite bit-identical outputs; kernel speedups at context 4,096
and 90% sharing are 1.580x / 1.582x / 1.538x for batches 4 / 8 / 16.

The shared workload's paired median gain of 13.5% is below the 15% target.
The control's median slowdown is about 1%, but its interval and the worsened
tails do not establish a reliable regression bound. Other desktop processes
were consuming CPU during these measurements. No concurrent project builds or
benchmarks ran, and no samples were discarded. The PR remains a draft and the
feature remains off; full request-lifecycle validation is still pending.

A second experiment scanned each weight block once and skipped repeated zero
checks only when every weight was nonzero. It preserved all numerical checks,
but its kernel sweep showed no consistent improvement. That specialization was
reverted before full-model testing. Its patch and all measurements remain in
[the follow-up evidence directory](measurements/shared-prefix-m2-followup/README.md),
alongside exact source fingerprints, process memory reports and reproduction
instructions for both retained and rejected implementations.

## Historical Apple M2 measurements, 2026-09-15 (`017c3e8`)

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
source snapshot matches implementation commit `017c3e8`. Subsequent profiling
and optimization changes require their own measured builds and source
fingerprints; the historical figures below do not describe those revisions.

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

1. Establish the current build's throughput and tail behavior on a quiet host,
   using the same declared workload, twelve paired repeats and sixteen steps.
   Retain every sample. Do not change the workload or select a favorable run to
   satisfy the gate; the no-sharing control must also provide a stable baseline.
2. The cold-prefill profile now identifies feed-forward projections as 70.72%
   of recorded model time (87.56% for all transformer projections). Before a
   further kernel change, separate gate/up/down projection compute from output
   transposes and inspect the rejected six-wide tile for register spills. A
   possible next experiment reuses activations across two weight rows while
   retaining each output's arithmetic; the current profile does not establish
   that activation traffic is the bottleneck. Keep cold-start, warm-cache and
   steady-state results separate.
3. Extend lifecycle evidence to timed mixed arrivals and a pool that actually
   forces eviction/recompute. Preserve complete outputs, cancellation behavior
   and competing-stream latency. The completed 192-block trace has spare capacity
   and cannot establish behavior under pressure.
4. Keep the feature off and the PR in draft until repeatable model improvement,
   the no-sharing bound and request-latency evidence justify merging. The source
   and measurement record are ready for review; the performance decision remains
   unresolved.
