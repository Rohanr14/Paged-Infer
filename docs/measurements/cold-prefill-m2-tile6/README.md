# M2 cold-prefill projection experiment

Cold prompt preparation dominated the preceding request-lifecycle replay. This
experiment adds isolated replay profiles, measures one cold prompt, and tests a
bounded change selected from that profile: reuse converted int8 weights across
six activation vectors instead of four on NEON. **The candidate was rejected
and reverted** after slowing two of three pairs. The retained changes add
per-run replay profiles and the missing final-prefill vocabulary-projection
span. The projection default remains four, and shared-prefix attention remains
disabled by default with its merge gate unresolved.

The diagnostic attributes 70.72% of recorded `model_main` elapsed spans to
feed-forward projections, 10.63% to QKV projections and 6.21% to output
projections: 87.56% together. Attention contributes 9.86%; the final vocabulary
projection contributes 0.30%. These elapsed spans are diagnostic, not CPU time
or performance-gate evidence. The instrumented request took 48,388.483 ms.

## Ordinary timing results and decision

| Pair / process order | Four-wide elapsed, s | Six-wide elapsed, s | Four / six elapsed ratio | Six-wide latency change |
| --- | ---: | ---: | ---: | ---: |
| 1: four, six | 48.164 | 48.818 | 0.987x | +1.36% |
| 2: six, four | 50.741 | 49.267 | 1.030x | -2.91% |
| 3: four, six | 51.055 | 53.565 | 0.953x | +4.92% |

The median paired elapsed ratio is **0.986602x**, with observed range
**0.953143–1.029919x**. Median paired latency is 1.36% worse. These are three
exploratory pairs, not a precise estimate of a universal slowdown. Host drift
is visible in the four-wide control's 48.164–51.055 s range. No run was dropped
or replaced, and the predeclared all-three-pairs improvement condition failed.

Every diagnostic and ordinary run computes all 2,049 prompt tokens in 65
scheduler/model chunks, emits token `[13]` with finish reason `length`, and
settles one successful request without shared-attention calls. There is no
subsequent decode forward. Exact primitive and matrix checks also passed for
the candidate before timing, including int8 extrema, dimensions 2,048/8,192,
ragged batches, scaled rows and both output layouts. The rejected kernel and
those candidate-specific tests are preserved in `candidate-source.patch`.

The source hashes below identify the measured diagnostic and candidate builds;
the final retained source removes the six-wide branch and its dedicated tests.
The ordinary four-wide observations therefore describe the candidate binary
with that branch disabled, not a fresh performance measurement of the final
retained source. Neither the discarded tile nor its timing is evidence that
the shared-attention gate has passed.

The next useful diagnostic separates gate/up/down projection compute from
their transposes, then inspects the discarded six-wide assembly for spills.
A possible next kernel would reuse activations across two weight rows while
keeping 16 independent accumulators and the same per-output arithmetic. That
trades fewer activation loads for more repeated widening. The current profile
and noisy tile comparison do not establish which cost dominates, so they do
not yet justify implementing that alternative.

## Workload and protocol

The input is the first cold request from the preceding
[shared-attention lifecycle trace](../shared-prefix-m2-score-tile/lifecycle/):
2,049 deterministic token IDs, with its output budget reduced to one token.
That token comes from the final prefill vocabulary projection, so no decode
forward is needed. This is real Llama 3.2 1B model computation with synthetic
input IDs, not a language-quality evaluation. The checkpoint SHA-256 is
`68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.

Both variants use int8 transformer projections, the existing f32 LM head,
four Rayon workers, 32-token prefill chunks and per-step budget, a maximum
batch of eight, and 192 blocks of 16 tokens. Prefix caching and shared-prefix
attention are disabled. Every run starts with a fresh engine and empty cache.
Model loading, fingerprinting, allocation and warmup are excluded from replay
time. Prompt processing, scheduling, final sampling and delivery are included.

The diagnostic uses `--features profiling`; counters reset after warmup and
engine reset. It contains 65 model batches, 1,040 calls each to QKV, attention
and output projection, 2,080 feed-forward projection spans, and one LM-head
projection. All shared-attention stages have zero calls. Profiles deliberately
omit some wrapper/scheduler work and nested scopes must not be summed together.

The ordinary comparison was fixed before candidate timing: three pairs, in
order **4/6, 6/4, 4/6**, from one non-instrumented binary. The environment knob
is set before each fresh process starts and recorded in its manifest. Every
run is retained, with an exact token/outcome comparison against the first
four-wide run. Each process has a 128-step and 180-second guard, checked between
steps. The primary comparison is whole replay elapsed time; engine prefill time
and TTFT corroborate it. The six-activation tile is eligible to remain
an opt-in experiment only if all three pairs improve latency. Defaults stay
unchanged regardless; this is not a shared-attention merge-gate experiment.

Measurements use Apple M2, macOS/Darwin 25.6.0, NEON and Rust 1.93.1 release
builds. No other project build, test or benchmark runs alongside timing.
Desktop applications, scheduling and thermal conditions remain uncontrolled.
Three serial pairs cannot establish a reliable confidence bound or request-tail
improvement. A one-request p95 equals that request's observation.

## Source identity and reproduction

Both source patches are alternatives relative to commit
`a0849f4b69acdc14c33621a85c94382e42351c41`; apply one, not both. The diagnostic
patch contains replay profiles and the missing final-prefill LM-head span.
The candidate patch additionally records the tile environment, adds the NEON
six-wide branch and preserves its exact-arithmetic tests. Production source
fingerprints exclude documentation and integration tests.

Both patches were independently applied to clean archives of that base commit;
the reconstructed source fingerprints match the recorded build fingerprints
exactly. `source-verification.json` preserves that check.

| Measured build | Source SHA-256 |
| --- | --- |
| Diagnostic with profiling | `9683cfbedf0bb2afec62ed8f1af0a95349cfb4d3867d5907367c18ea8e39fe45` |
| Ordinary candidate binary, both tile settings | `70f9795a937084955d1cc8f1977b743cc85d53bfa7b44b789ea42ca8a57e1b77` |

The raw manifests retain the build-time dirty Git state, compiled source hash,
checkpoint/input/configuration hashes, environment, and runtime checkout state.
Use the compiled source digest to distinguish the measured changes from the
unchanged base commit. `performance_gate_eligible: true` only indicates that
profiling is absent; it does not indicate that any acceptance gate passed.

To reproduce in a clean checkout, apply the selected source patch, build the
release `workload_replay` binary with the corresponding profiling feature
selection, and use the archived workload and configuration. On this host the
build selected `/Library/Developer/CommandLineTools` via `DEVELOPER_DIR`.
`run_comparison.py BINARY CHECKPOINT` runs the fixed ordinary comparison from
the repository root, using files beside the script. Output paths must be new;
copy the protocol, workload, config and scripts to a fresh output directory.
`analyze.py` validates all completed reports, exact outputs, workload identities,
source identity, feature flags and the fixed pair ordering before summarizing.

Whole-process `.time` files include loading and warmup. Their RSS and process
elapsed values are not isolated request or variant memory measurements.
Executables are excluded from this archive; the ordinary binary digest is kept.
