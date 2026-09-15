# M2 shared-prefix score-tile experiments

This archive records the retained implementation, two discarded dispatch policies, and the final narrow score-tile candidate. It preserves the diagnostic profile, all kernel samples and process timing reports, source patches, the original protocols, completed full-model sharing and control measurements, and a bounded-memory request replay. The 90%-sharing model run's paired median speedup is 18.2%, with a broad descriptive interval and worse tails. The physical no-sharing control's median is about 0.4% slower, with uncertainty extending beyond 5% regression. Lifecycle outputs match and warm-burst throughput improves, while overall lifecycle throughput is essentially flat. This evidence does not establish the performance gate.

The final candidate uses the NEON two-query/two-key score primitive only for **32 query heads, eight KV heads, head dimension 64, and at least 2,048 attended shared tokens**. Other shapes and shorter attended shared prefixes retain the previous score schedule. Shared-prefix attention remains experimental and disabled by default. This dispatch reflects the sampled M2 workloads; it does not establish an optimal threshold or performance benefit for other CPUs, shapes, batch sizes, block sizes, or attention windows.

## Evidence and source identity

All source patches below are alternatives relative to clean commit `1b221fa4747c9cd6c4a10c0a9fb40e303a344adb`. Apply exactly one patch to that baseline; do not stack the candidate patches. The baseline includes the benchmark's optional KV-head shape setting.

Each of the three patches was independently applied to a clean archive of that baseline, and its reconstructed source fingerprint matched the corresponding measured digest below. The current final source also matches the final candidate digest.

| Evidence | Compiled source SHA-256 | Reproduction from the baseline |
| --- | --- | --- |
| Retained kernel, all three experiment stages | `ebf1817e8e5212926ac0073bcb419464e88e196a418d94846e467563a1ef5442` | Unmodified baseline, ordinary release build |
| [Retained diagnostic profile](profile/retained-profile.jsonl) | `ebf1817e8e5212926ac0073bcb419464e88e196a418d94846e467563a1ef5442` | Unmodified baseline, release build with `--features profiling` |
| [Unrestricted candidate](unrestricted/) | `e76381eafff7f04ec9491b733ab1c19f085a0603941a9bff95ca2539bf31f7b9` | [unrestricted-candidate.patch](unrestricted/unrestricted-candidate.patch) |
| [Shape-only candidate](shape-only/) | `5a800c235a8b942342ee1c264947a1d1e1beab3b1562f8295fd175dceacc7b9b` | [shape-only-candidate.patch](shape-only/shape-only-candidate.patch) |
| [Final long-prefix candidate](long-prefix/) | `e67c81a4a31e0a03f79d8102bb2eb1e543964fd43e94db4d270db8ef8b499a88` | [final-candidate.patch](long-prefix/final-candidate.patch) |

The raw manifests report the build-time Git head and dirty flag. Those fields identify the checkout state, while `source_sha256` identifies the measured compiled inputs. In particular, candidate reports bearing a dirty `1b221fa` head do not describe the unmodified baseline. The source digest does not encode feature selection: the retained diagnostic and ordinary binaries have the same source digest but different `profiling_enabled` flags.

The final candidate was subsequently committed as `1ad58be`. Its completed kernel and ordinary full-model artifacts share source digest `e67c81a4a31e0a03f79d8102bb2eb1e543964fd43e94db4d270db8ef8b499a88`; their original build metadata remains intact.

Measurements used Apple M2, macOS/Darwin 25.6.0, NEON, four Rayon workers, Rust 1.93.1, and ordinary release optimization without additional Rust flags. The exact environment is in each manifest. Kernel manifests have `profiling_enabled: false`; their `performance_gate_eligible: true` means instrumentation is absent, not that the performance gate has passed.

The directory layout preserves the original experiment boundaries:

- [profile/](profile/): one diagnostic real-model run and its process timing report.
- [unrestricted/](unrestricted/): protocol, three alternating source pairs for each of two KV-head shapes, patches, and assembly review.
- [shape-only/](shape-only/): the same complete sweep with only the exact head-shape guard.
- [long-prefix/](long-prefix/): the same complete sweep with both the head-shape and attended-prefix guards, plus subsequent full-model evidence for that source.
- [lifecycle/](lifecycle/): the fixed replay protocol, both configurations, complete input trace, four-run report, stdout, and process timing report.

Each kernel stage contains 12 completed JSONL reports and 12 matching `.time` files. Each report has 36 cases, 21 retained timing pairs per case, and a successful verification footer. Every output in all three stages was finite and bit-identical to ordinary attention, with `max_abs_error: 0`. Copies in this archive were checked byte-for-byte against the completed measurement files. Executables, build logs, and working checkouts are not included.

## Diagnostic motivation and assembly inspection

The [retained profile](profile/retained-profile.jsonl) uses a real Llama 3.2 1B checkpoint quantized to int8, 4,096 history tokens, batch eight, 90% physically shared history, 16 decode steps, and one repeat. It prepares genuine model KV outside the timer, warms the complete measured position range, then resets profiler counters and tokens before timed decode. The manifest records all input IDs and model, configuration, and input fingerprints. The checkpoint digest is `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`; the input digest is `906bef4118a465579dad78a63b05a7ad03fc63cf06d72b590407db94b73a8e81`.

For the retained shared-prefix path, attention accounts for 36.65% of the sum of recorded `model_main` spans. Within the sum of recorded `attention_workers` spans, shared scores account for **53.99%**, shared values 19.16%, softmax 10.91%, private scores 8.16%, private values 7.69%, and packing 0.09%. This made query reuse in the score loop a bounded next experiment after the prior value-loop work.

These are diagnostic elapsed spans. Main-thread attention contains the nested attention setup and parallel work; worker totals include scheduling delay and are not CPU time. Do not add scopes or use these fractions as an exact prediction of end-to-end speedup. Timer/counter overhead makes the profile unsuitable for performance gating. With only one repeat, its bootstrap interval is explicitly unavailable.

The [assembly review](unrestricted/score-tile-assembly-review.md) and [vector-loop excerpt](unrestricted/score-tile-vector-loop.s) belong to the **unrestricted candidate**, before dispatch guards were added. The inlined two-query/two-key loop keeps 16 accumulators in vector registers, reuses each query load across two adjacent keys, and has no stack accesses or helper calls inside the vector accumulation loop. Covering four queries and two keys uses 32 vector loads instead of 40, with the same FMA count and reduction arithmetic. This confirms the intended load reduction, not a latency improvement; metadata handling, reduction/scatter overhead, and instruction footprint can offset it. No separate assembly-review claim is made for the later guarded binaries.

## Kernel protocol and interpretation

The [unrestricted](unrestricted/protocol.json), [shape-only](shape-only/protocol.json), and [long-prefix](long-prefix/protocol.json) protocols retain the same primary case: 32 query heads, eight KV heads, head dimension 64, context 4,096, batch eight, and 90% requested sharing. Four KV heads are a secondary shape. Each shape sweeps batches 1/4/8/16, contexts 256/1,024/4,096, and sharing 0/50/90%. Context denotes history tokens; attention also includes one current token. Sharing rounds down to complete 16-token blocks, so the primary case attends 3,680 shared tokens.

Each case has two untimed warmup pairs and 21 timed pairs. Ordinary attention and shared-plan construction plus execution rotate order within each source sweep. The plan borrows entries without heap allocation; entry vectors are outside both timers, and scratch capacity is reused after warmup. The harness uses distinct queries, nonzero KV, and scattered physical blocks. This is an isolated one-layer attention benchmark, not full-model or request throughput.

Across source comparisons, pair 1 runs retained then candidate, pair 2 candidate then retained, and pair 3 retained then candidate. KV-head order is 8/4, 4/8, and 8/4 respectively. Every sample remains in the archive, including outliers and discarded experiments.

Two descriptive ratios are reported below; values above one favor the new source:

- **Raw ratio:** retained-source shared-path p50 divided by candidate-source shared-path p50.
- **Fallback-normalized ratio:** candidate-source `(ordinary p50 / shared p50)` divided by retained-source `(ordinary p50 / shared p50)`.

The second ratio uses each source sweep's ordinary-attention timings as a drift indicator. It does not causally remove host drift or establish a regression bound. These source sweeps run successively; individual samples from different binaries are not matched repeat pairs. Ratios of p50s are not the median of individual paired ratios, and the three source comparisons do not constitute a confidence interval. With 21 observations, p95 is also sensitive to individual slow samples.

## Discarded dispatch policies

All ratio lists below are in source-pair order 1/2/3.

| Candidate policy and case | Raw ratios | Fallback-normalized ratios |
| --- | --- | --- |
| Unrestricted, primary 8 KV / batch 8 / context 4,096 / 90% | 1.298 / 1.239 / 1.086 | 1.215 / 1.131 / 1.109 |
| Unrestricted, 4 KV / batch 4 / context 4,096 / 90% | 0.882 / 0.940 / 0.836 | 0.878 / 0.962 / 0.722 |
| Unrestricted, 8 KV / batch 8 / context 1,024 / 90% | 0.910 / 0.876 / 0.900 | 0.925 / 0.914 / 0.968 |
| Shape-only, primary 8 KV / batch 8 / context 4,096 / 90% | 1.090 / 1.097 / 1.084 | 1.110 / 1.157 / 1.126 |
| Shape-only, 8 KV / batch 8 / context 1,024 / 90% | 0.911 / 0.911 / 0.820 | 0.904 / 0.904 / 0.867 |

The unrestricted policy repeatedly regressed the four-KV-head case despite improving the primary eight-KV-head case. Restricting to the exact model head shape preserved the primary benefit but still repeatedly regressed shorter prefixes. The shape-only batch-eight/context-256/90% raw ratios were 0.972/0.964/0.496, including a large third-pair outlier; all samples are retained rather than treating that outlier as a precise effect size.

The final guard therefore also requires at least 2,048 attended shared tokens. This is the smallest shared interval sampled in the promising context-4,096/50% cells, and excludes all tested context-256 and context-1,024 cells. The threshold is an experimental restriction supported by this grid, not a claim that every longer interval benefits.

## Final guarded kernel results

The primary case shows two clear raw median improvements and a third comparison that is roughly flat. Fallback-normalized ratios favor the candidate in all three. Tail behavior is mixed: the third comparison's candidate p95 is worse.

| Source pair | Retained shared p50, µs | Candidate shared p50, µs | Raw ratio | Fallback-normalized ratio | Retained / candidate shared p95, µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| [1 retained](long-prefix/pair1-kv8-retained.jsonl) / [candidate](long-prefix/pair1-kv8-candidate.jsonl) | 3,841.833 | 3,334.709 | 1.152 | 1.307 | 4,971.417 / 4,520.750 |
| [2 retained](long-prefix/pair2-kv8-retained.jsonl) / [candidate](long-prefix/pair2-kv8-candidate.jsonl) | 3,884.208 | 3,385.458 | 1.147 | 1.107 | 7,070.583 / 4,382.875 |
| [3 retained](long-prefix/pair3-kv8-retained.jsonl) / [candidate](long-prefix/pair3-kv8-candidate.jsonl) | 3,671.625 | 3,689.667 | 0.995 | 1.141 | 5,134.042 / 5,518.041 |

At 50% sharing in the same shape, context, and batch, raw ratios are **0.920/1.159/0.904**, while fallback-normalized ratios are **1.003/1.185/1.036**. Two raw comparisons regress. Their retained/candidate p95s are 7,430.833/7,595.709, 8,325.375/7,696.417, and 7,664.666/8,682.417 µs. This is weaker evidence than the primary case, and normalization does not erase those observed regressions.

The physical no-sharing control also shows host variation. At eight KV heads, batch eight, context 4,096, and 0% sharing, both benchmark variants select ordinary attention. Across sources, raw candidate-path ratios are **0.900/0.932/0.943**, while fallback-normalized ratios are **0.976/0.996/1.029**. The ordinary-path p50s in the retained/candidate sources are 11,263.541/12,216.584, 11,217.375/11,991.375, and 11,080.083/12,090.750 µs. Both unchanged fallback schedules slowed in the later source runs; this is evidence of drift in the measurements, not proof that the candidate has a bounded no-sharing cost. The batch-16 no-sharing cell likewise has raw ratios 0.904/0.903/0.916 and normalized ratios 1.004/1.005/1.042.

Across the 72 distinct shape/grid cells, none has all three fallback-normalized ratios below 0.95. That descriptive screen is not a statistical assurance of less than 5% regression. It also does not establish behavior outside the sampled grid.

The primary cell uses 29,229,056 physical KV bytes, 4,195,328 bytes of ordinary score scratch, and 4,326,400 bytes of shared scratch in both retained and final candidates. The shared plan itself allocates no heap memory. The harness keeps ordinary scratch while timing the shared path, so these figures describe separate scratch allocations rather than a net memory saving. Matching `.time` reports preserve whole-process maximum RSS, including the full sweep and allocations; they do not isolate one case's memory use.

These results justify measuring the guarded candidate in the real model: two primary raw median wins, one flat comparison, and consistent normalized primary gains amid variable controls. They do not establish universal kernel improvement or the full-model performance gate.

## Full-model steady-state decode

The completed [90%-sharing run](long-prefix/shared90-model12.jsonl) and [process report](long-prefix/shared90-model12.time) use the final guarded source without profiling: real Llama 3.2 1B int8 weights, context 4,096, batch eight, 16 decode steps, 12 repeat pairs, four Rayon workers, and 3,680 physically shared history tokens. Model, configuration, and input fingerprints match the diagnostic profile. Each variant starts from identical genuine prepared KV and held tokens, performs a complete untimed warmup, resets, and times all model layers, LM head, finite-logit scan, argmax, and token recording. Loading, preparation, warmup, and output comparison are excluded. EOS does not terminate this fixed-budget benchmark.

| Measurement | Ordinary attention | Shared-prefix candidate |
| --- | ---: | ---: |
| Reported median throughput, tokens/s | 30.232 | 35.625 |
| Elapsed p50 per 16-step run, ms | 4,233.871 | 3,592.967 |
| Pooled decode-step p50, ms | 262.201 | 217.601 |
| Pooled decode-step p95, ms | 292.922 | 400.591 |
| Pooled decode-step p99, ms | 365.593 | 948.887 |
| Maximum decode-step time, ms | 380.308 | 1,073.573 |

The **median of the 12 paired elapsed-time ratios is 1.182249** (18.2% more tokens per unit time for an equal token budget). Its observed minimum/maximum are **0.671940/1.253544**. The deterministic paired-bootstrap descriptive 95% interval is **[1.040278, 1.229775]**. The separate legacy ratio of p50 elapsed times is 1.178377; it is a different estimator and is not the headline paired result. Repeats 7 and 8 regress materially and repeat 9 is approximately flat; all remain in the summary and raw data.

The bootstrap takes 10,000 resamples of the 12 complete baseline/candidate repeat pairs, then computes each resample's median elapsed-time ratio. It uses SplitMix64 with rejection-sampled indices and seed `0x7061697265644349`, midpoint medians for even counts, and nearest-rank 2.5th/97.5th percentile endpoints. It assumes independent, exchangeable repeat pairs. Twelve pairs remain a small empirical sample; the interval does not correct serial dependence, thermal drift, order effects, or unobserved conditions, and can under-cover. The 192 steps per variant are dependent within runs and do not create 192 independent bootstrap observations. The interval extends well below a 15% speedup, and the pooled tail latency is worse, so the nominal median improvement is insufficient to establish the gate.

All 24 runs have identical complete greedy token arrays, finite logits, the same `fixed_step_budget` finish reason, and final-logit digest `c601cab3040fcc87e17402eda8509653a7507014a7890ca79b8b1b561f8e5758`. Each shared run records 256 shared layer calls; ordinary runs record zero. The manifest reports 476,053,504 physical KV bytes and 2,025,259,008 packed projection bytes; the shared scratch reports 8,654,848 extra allocated bytes. Whole-process maximum RSS is 3,268,575,232 bytes and includes loading, preparation, warmup, and both variants. It cannot attribute a memory delta to one variant.

The completed [physical no-sharing control](long-prefix/shared0-model12.jsonl) and [process report](long-prefix/shared0-model12.time) use the same model, inputs, source, steps, and repeat count, with genuine common-content KV copied to distinct physical blocks. Both variants select fallback on every run, with zero shared layer calls and zero extra shared scratch. All 24 control outputs match, including the complete greedy arrays, fixed-budget finish reason, finite-logit checks, and the same final-logit digest as the sharing run.

| No-sharing control measurement | Ordinary attention | Shared-prefix option enabled, selecting fallback |
| --- | ---: | ---: |
| Reported median throughput, tokens/s | 24.233 | 24.011 |
| Elapsed p50 per 16-step run, ms | 5,282.134 | 5,330.815 |
| Pooled decode-step p50, ms | 317.676 | 318.330 |
| Pooled decode-step p95, ms | 430.599 | 434.052 |
| Pooled decode-step p99, ms | 755.072 | 1,035.347 |

The control's paired median ratio is **0.995985**, its observed minimum/maximum are **0.841564/1.272467**, and its descriptive bootstrap interval is **[0.938505, 1.031578]**. The central observation is approximately 0.4% lower throughput, but the interval extends beyond 5% regression and cannot establish the required bound. Physical KV allocation is 2,164,260,864 bytes; whole-process maximum RSS is 5,029,068,800 bytes. The control contains more physical KV than the sharing case, so their throughputs and RSS should not be compared as if they were two variants of the same physical layout. Within each case, both timed variants use the same layout.

Across the two ordinary model artifacts, all 48 timed runs verify. These are full-model **steady-state decode** measurements, not request-lifecycle or HTTP throughput. The separate lifecycle trace below inspects scheduling, cancellation, prefix-cache reuse, request latency, and memory.

## Bounded-memory request lifecycle

The [completed replay](lifecycle/replay.jsonl), [protocol](lifecycle/protocol.json), [workload](lifecycle/workload.json), [ordinary configuration](lifecycle/baseline.json), and [shared configuration](lifecycle/shared.json) use the same real int8 checkpoint and final source digest as the model runs, without profiling. The workload digest is `cfa95add2a85cccffab8c1d2db71a80b90df9d355a73e2a8a68354f7666ef7b8`. The report preserves both the binary's dirty `1b221fa` build metadata and the run-time `1ad58be` checkout identity.

Each run starts with an empty prefix cache. Eight requests arrive at scheduler step zero with 2,049-token prompts: a common 2,048-token content prefix and a private final token. The cold burst requests up to 24 generated tokens; `cold-7` is cancelled at step 72. Eight further requests arrive at step 88, request up to 16 tokens, and reuse the populated prefix cache. Prompts use deterministic synthetic token IDs with real model computation; this is not a language-quality test. EOS handling is unchanged and no run ends on EOS.

Both configurations use prefix caching, a maximum batch of eight, 32-token prefill chunks and per-step prefill budget, streaming single-token delivery, no speculation, and 192 blocks of 16 tokens. Their shared-attention toggle is the only engine difference. Two repeats interleave ordinary/shared then shared/ordinary. The 256-step and 300-second per-run guards were fixed before the first run. This is direct-engine lifecycle timing: loading, fingerprinting, allocation, and initial engine warmup are excluded, while actual cold request prefill, scheduling, decode, cancellation, and delivery are included. Step arrivals describe deterministic scheduler traffic, not wall-clock arrival deadlines or HTTP traffic.

All four runs pass exact token-and-outcome verification, independently confirmed from the recorded requests. Each settles 16 requests once: 15 successes and `cold-7` cancelled after eight delivered tokens. Each produces 296 useful tokens from successful sequences and 304 total delivered tokens, finishes in 104 engine steps, reaches eight active sequences, reuses 30,720 prompt tokens, and computes 2,064 prompt tokens in 80 prefill chunks. Shared runs record 608 shared-attention layer calls and 9,437,184 shared query tokens; ordinary runs record zero.

| Repeat / configuration | Whole-run useful tokens/s | Warm-burst useful tokens/s | Whole-run inter-token p95, ms |
| --- | ---: | ---: | ---: |
| 1 ordinary | 5.009674 | 38.197090 | 207.214 |
| 1 shared | 4.962820 | 40.085031 | 191.166 |
| 2 ordinary | 4.823550 | 34.334449 | 217.479 |
| 2 shared | 4.840112 | 40.027769 | 191.842 |

| Repeat / configuration | Cold TTFT p50 / p95, ms | Warm TTFT p50 / p95, ms |
| --- | ---: | ---: |
| 1 ordinary | 51,272.229 / 51,272.293 | 389.791 / 389.823 |
| 1 shared | 52,195.113 / 52,195.140 | 396.563 / 396.597 |
| 2 ordinary | 52,783.103 / 52,783.133 | 409.946 / 409.991 |
| 2 shared | 53,683.847 / 53,683.877 | 393.981 / 394.013 |

Definitions use the recorded timestamps without additional normalization:

- Whole-run useful throughput is 296 successful-sequence tokens divided by the report's complete measured elapsed seconds; cancelled output is excluded from its numerator.
- Warm-burst useful throughput is 128 successful warm tokens divided by `(latest warm request terminal time − earliest warm request actual submission time)` in seconds. These intervals are 3,351.041/3,193.212 ms for ordinary/shared in repeat 1 and 3,728.034/3,197.780 ms in repeat 2. They include warm request queueing, private-token prefill, and all warm decoding; no prefill time is subtracted.
- Each request's TTFT is its first delivery timestamp minus its own actual submission timestamp. Cold and warm distributions each contain eight requests, including the cold request later cancelled. Percentiles use nearest rank, `ceil(p × count)`, so each burst's p95 is its maximum.
- Inter-token intervals come from successive delivery timestamps within each sequence. All deliveries contain one token here, so this matches the report's inter-delivery distribution. The whole-run p95 pools 288 intervals, including intervals delivered before cancellation; it excludes TTFT and cross-sequence gaps.

Overall lifecycle throughput is essentially flat: one shared run is slightly slower and the other slightly faster. Warm-burst throughput and whole-run inter-token p95 improve in both repeats, but warm TTFT is mixed. Cold prefill dominates, with recorded total prefill time ranging from 51.7 to 54.1 seconds out of 59.1 to 61.4 measured seconds. The steady-state model gain therefore does not translate to a similar whole-lifecycle gain in this trace. Two repeats provide descriptive validation rather than a reliable speed bound.

All runs peak at 142 of 192 allocated blocks, or 148,897,792 occupied KV bytes out of 201,326,592 reserved bytes. They retain 128 cached blocks at completion. Shared runs report 4,460,544 bytes of shared-attention scratch, versus zero in ordinary runs. There are no OOM outcomes, rejected requests, preemptions, recomputation, or copy-on-write copies. The 192-block limit bounds memory but does **not** force pressure or preemption; this trace is not a pressure test. The [process report](lifecycle/replay.time) records maximum RSS of 4,796,399,616 bytes across loading and all four runs, which does not isolate a per-configuration memory delta.

## Reproduction

Use a clean checkout of the baseline and, for a candidate, apply only its baseline-relative patch. Preserve a separate binary per source so later builds cannot replace an executable under measurement. On the measured Mac, Cargo builds need `DEVELOPER_DIR=/Library/Developer/CommandLineTools` to use the installed command-line tools; bare Cargo otherwise encounters the unaccepted Xcode license. The following shows one source build and one eight-KV-head sweep; repeat it for four KV heads and use the source/shape orders recorded in the protocol.

```sh
# In the selected clean baseline checkout, optionally apply ONE candidate patch.
git apply /absolute/path/to/archive/long-prefix/final-candidate.patch
DEVELOPER_DIR=/Library/Developer/CommandLineTools cargo build --release --bin shared_attention_benchmark
RAYON_NUM_THREADS=4 SHARED_ATTN_KV_HEADS=8 \
  SHARED_ATTN_BATCHES="1 4 8 16" SHARED_ATTN_CONTEXTS="256 1024 4096" \
  SHARED_ATTN_PERCENTAGES="0 50 90" SHARED_ATTN_REPS=21 SHARED_ATTN_WARMUP=2 \
  /usr/bin/time -l target/release/shared_attention_benchmark > sweep.jsonl 2> sweep.time
```

Omit `git apply` for the retained source. Substitute the unrestricted or shape-only patch to reproduce those historical alternatives. Check the emitted source digest, head shape, thread count, feature flag, and successful verification footer against the corresponding archived manifest. Build and test outside measurement periods; concurrent CPU work changes these comparisons.

The retained diagnostic profile is reproduced from the unmodified baseline, using the same checkpoint and configuration digests recorded in its manifest:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cargo build --release --features profiling --bin shared_decode_benchmark
MODEL_PATH=models/llama3-reference/checkpoint/model.safetensors QUANT=int8 \
  RAYON_NUM_THREADS=4 SHARED_DECODE_CONTEXT=4096 SHARED_DECODE_BATCH=8 \
  SHARED_DECODE_PERCENTAGE=90 SHARED_DECODE_STEPS=16 SHARED_DECODE_REPS=1 \
  /usr/bin/time -l target/release/shared_decode_benchmark > retained-profile.jsonl 2> retained-profile.time
```

The profile command is diagnostic only. To reproduce the completed ordinary model run, use the final candidate source, build `shared_decode_benchmark` without `--features profiling`, set `SHARED_DECODE_REPS=12`, and retain the other model settings above. For the physical no-sharing control, additionally set `SHARED_DECODE_PERCENTAGE=0`. Check `profiling_enabled: false` and the expected source, model, configuration, and input digests. Keep complete per-repeat timings, paired speedup statistics, output checks, scratch figures, and process memory reports.

To reproduce the lifecycle trace from the final source, use the archived configurations and workload without changing the timeout, arrivals, or EOS handling:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cargo build --release --bin workload_replay
score_tile_archive=docs/measurements/shared-prefix-m2-score-tile/lifecycle
/usr/bin/time -l target/release/workload_replay \
  --model models/llama3-reference/checkpoint/model.safetensors --quant int8 --threads 4 \
  --workload "$score_tile_archive/workload.json" \
  --config "$score_tile_archive/baseline.json" --config "$score_tile_archive/shared.json" \
  --repeats 2 --max-steps 256 --timeout-secs 300 --verify --output replay.jsonl \
  > replay.stdout 2> replay.time
```

## Remaining performance uncertainty

The recorded protocol requires at least 15% paired median full-model throughput improvement versus ordinary attention in the Llama 3.2 1B int8, context-4,096, batch-eight, 90%-sharing case, plus the no-sharing control, output parity, latency, and lifecycle checks. The sharing run exceeds the nominal median threshold but has substantial uncertainty and worse tails; the control does not establish a less-than-5% regression bound. Lifecycle output, cancellation, cache reuse, shared-path activation, and bounded-memory behavior verify, with essentially flat overall useful throughput. These observations leave the performance gate unestablished and the feature experimental and off by default. Earlier value-tile measurements belong to their separate [follow-up archive](../shared-prefix-m2-followup/); they must not be relabeled as measurements of this score-tile source.
