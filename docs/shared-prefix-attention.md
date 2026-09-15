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

Every warmup now hashes the complete logits at every decode position and checks
that digest across variants and repeats. The timed specialization performs no
logit hashing. Each measured run retains its warmup duration, loop-start UNIX
timestamp, complete tokens, final-logit digest and individual step times. On
Unix, process CPU time, page-fault and context-switch deltas are sampled just
outside the timed loop; unavailable or decreasing counters produce `null`.
These process-wide deltas include all threads and a small amount of boundary
work. They cannot identify a particular worker or prove that outside activity
caused a stall.

Shared-decode summaries retain the existing ratio of separate elapsed-time
medians and add `paired_elapsed_time_speedup`: every same-repeat
baseline/candidate elapsed-time ratio, plus its median, minimum and maximum.
Ratios above one favor the candidate. The paired median uses the midpoint of
both middle ratios for even sample counts; the legacy marginal p50 fields keep
their nearest-rank convention.

The `performance_diagnostics` summary additionally reports aggregate throughput
(equal total token budgets divided by total elapsed time) and paired medians
split by which variant ran first. These views preserve the primary estimator
while exposing costly tails and order sensitivity; the strata are not separate
independent experiments. Compare pooled p95/p99/max with the per-run step
distributions rather than treating each decode step as an independent repeat.

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

## Current confirmation, 2026-09-15

The fixed two-session confirmation does **not establish the merge gate**:
session B's no-sharing control has a descriptive interval lower bound of
**0.908615**, below the required 0.95. Every other prescribed check passes,
but both sessions must satisfy every predeclared check. The feature remains
disabled by default.

The confirmation uses one frozen ordinary build: Llama 3.2 1B int8, context
4,096, batch eight, sixteen decode steps, twelve pairs per sharing fraction,
four Rayon workers, matmul tile four and two attention lanes per thread. Session
A runs 90% requested sharing (3,680 common positions) then 0%; session B
reverses that order. The control copies the same token history into independent
physical blocks. Within each case, the existing alternating baseline/candidate
order is unchanged. There are no
numerical or kernel changes in this confirmation. All observations are retained;
results are neither pooled across sessions nor replaced by additional runs.

| Session / physical sharing | Median paired speedup | Descriptive bootstrap 95% interval | Aggregate throughput ratio | Baseline / candidate pooled step p95 |
|---|---:|---:|---:|---:|
| A / 90% | 1.196x | [1.186, 1.203] | 1.197x | 280.8 / 240.8 ms |
| A / 0% | 1.006x | [0.980, 1.036] | 1.003x | 366.6 / 372.2 ms |
| B / 0% | 0.985x | [0.909, 1.072] | 0.969x | 2269.0 / 2289.9 ms |
| B / 90% | 1.574x | [1.489, 1.662] | 1.583x | 1639.3 / 1088.9 ms |

Both sessions require paired-median and descriptive-interval lower bounds of
at least 1.15 for sharing and 0.95 for the control. Additional safeguards require
the same aggregate-throughput bounds, candidate pooled p95 no more than 1.05
times baseline, and order-stratum medians of at least 1.0 for sharing and 0.95
for the control. These checks preserve the original improvement and regression
targets; the descriptive intervals retain the statistical limitations above.

The [complete confirmation archive](measurements/shared-prefix-m2-confirmation/README.md)
retains every report, frozen input, host snapshot and executable/source digest.
Its independent analyzer validates all 96 timed runs, complete tokens, final
logits, every-step warmup logits, path counts and both lifecycle traces. Eleven
negative checks verify that invalid evidence is rejected. The only failed
criterion is the second control's interval lower bound.

The measured source is `fdd76e3`; source reconstruction verifies its compiled
fingerprint. The later `160426f` change only replaces equivalent four-byte
chunk iteration inside the untimed hash helper. It leaves timed decode source
unchanged; the archive distinguishes these revisions and does not claim
identical executable bytes.

No-sharing whole-process time rises from 395 to 1,614 seconds between sessions,
with roughly the same retired instruction count; B's recorded one-minute load
average rises from 4.63 to 24.36. Full per-run p95/p99/max, order-stratum medians
and process counters are retained. These observations do not establish a cause.
No project builds, tests or other benchmarks overlap the measurements. Desktop
applications remain running; host/resource observations cannot establish the
cause of a slow sample or justify excluding it.

The separate lifecycle cases test deadline-based arrivals and actual memory
pressure. Their acceptance rules cover correctness and latency accounting
within the fixed traces; they contain no lifecycle throughput or latency
regression bound.

| Lifecycle case | Fixed scope | Final evidence |
|---|---|---|
| Mixed arrivals | Two alternating baseline/shared repeats; millisecond arrivals; four requests, eight sequences and 336 output tokens; no OOM, rejection or preemption | All four runs pass output and latency-accounting checks; peak 162/192 blocks, no preemption; 784 shared layer calls in each enabled run |
| Memory pressure | One repeat of roomy-192 baseline, tight-132 baseline and tight-132 shared; two sequences and 128 output tokens; both tight runs must reach capacity and exercise preemption, recomputation and COW | Both tight runs match the oracle, peak at 132 blocks, and record one preemption, 33 recomputed tokens and one COW copy; enabled run executes 496 shared layer calls |

These traces explicitly use token 0 as EOS and clear additional EOS IDs. An
early EOS or missed pressure invariant fails the declared observation; prompts
are not replaced after inspecting the results. Cancellation evidence remains
in the earlier replay archive below.

## Measurement history

These records explain the implementation's development. They retain every
sample, source fingerprint and rejected experiment. Historical timings belong
to their recorded builds and are not pooled with the current confirmation.

| Source / experiment | Main result | Evidence |
|---|---|---|
| `017c3e8`: initial real-model benchmark, final harness | Shared paired median 1.114x from three pairs, below 1.15; no-sharing pairs ranged 0.812–2.014x. All complete tokens and final logits matched. | [Initial M2 measurements](measurements/shared-prefix-m2/README.md) |
| Earlier value kernels and harness correction | Development measurements moved from 0.788x to 1.106x after retaining value sums in registers. Review then corrected discarded output-buffer capacity. These reports were superseded by the final harness. | [Initial experiment history](measurements/shared-prefix-m2/README.md) |
| `cf7bbe1`: retained value-loop optimization | Twelve-pair shared median 1.135x, interval [1.107, 1.158]; control 0.990x, interval [0.950, 1.027]. Shared and control tails worsened; the gate was not established. | [Value-loop follow-up](measurements/shared-prefix-m2-followup/README.md) |
| Rejected nonzero-weight specialization | Exact numerical checks passed, but kernel sweeps showed no consistent benefit; the specialization was reverted before model timing. | [Retained and rejected follow-up experiments](measurements/shared-prefix-m2-followup/README.md) |
| `1ad58be`: guarded long-prefix score tile | Twelve-pair shared median 1.182x, interval [1.040, 1.230], with pooled p95 worsening 293→401 ms. Control 0.996x, interval [0.939, 1.032]. All 48 timed runs matched outputs; the gate was not established. | [Score-tile evidence](measurements/shared-prefix-m2-score-tile/README.md) |
| Rejected unrestricted and shape-only score tiles | Three alternating source comparisons per shape exposed regressions for four KV heads and shorter prefixes. The retained tile is restricted to the measured Llama shape and at least 2,048 attended shared tokens. | [Score-tile development and source patches](measurements/shared-prefix-m2-score-tile/README.md) |
| `1ad58be`: earlier request lifecycle | Two replay pairs matched outputs, cancellation and prefix reuse. Overall throughput was nearly flat; warm-burst gains were 4.9% and 16.6%. Peak allocation was 142/192 blocks, so the trace did not force pressure. | [Replay results and limitations](measurements/shared-prefix-m2-score-tile/README.md#bounded-memory-request-lifecycle) |
| Cold-prefill six-wide projection tile, reverted | A cold 2,049-token profile attributed 70.72% of recorded model time to feed-forward projections. Fixed four/six elapsed ratios of 0.987x, 1.030x and 0.953x failed the consistency check; the default remained four. This separate cold-start experiment did not establish the shared-decode gate. | [Cold-prefill experiment](measurements/cold-prefill-m2-tile6/README.md) |

## Focused next step

Keep the feature disabled by default and the PR in draft. The source and
correctness evidence are ready for review; the performance gate is unresolved.
Both shared cases clear their bounds, so another speculative kernel rewrite is
not the next step supported by this record.

Establish a stable evaluation environment before a new confirmation: use a
quiet, dedicated M2 host with sufficient memory for the declared no-sharing
case, record CPU/memory conditions, and fix the workload, run count and failure
rules before timing. Preserve this failed observation and every new sample.
Do not repeat the existing session until a favorable result appears or pool
across changed sources or host conditions. If the control remains variable,
investigate that variation before making a model-level performance claim.

The completed mixed-arrival and forced-pressure traces establish bounded
correctness and latency accounting. They do not establish a serving throughput
or tail-latency guarantee; measure representative traffic separately before
enabling the option for a deployment.
