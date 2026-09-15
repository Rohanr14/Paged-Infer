# Shared-prefix profiling and dimension-tile follow-up

These measurements follow the initial `017c3e8` experiment. All use Apple M2,
NEON and four Rayon workers. Source fingerprints and feature flags distinguish
instrumented diagnostics from ordinary timing builds. Other desktop processes
were consuming substantial CPU; no other project build or benchmark ran beside
the measured process. Every timing sample is retained.

## Diagnostic profile before the dimension tile

[Profile](profile-before.jsonl) and [process report](profile-before.time):
Llama 3.2 1B int8, context 4,096, batch 8, sixteen decode steps, one paired repeat.
The profile source SHA-256 is
`549b8f597bdffdf05f7b22a0a9111e6d1af2b9eb6f936f62a1fae48492ead0a3`.
To reproduce its exact compiled source, check out `cf7bbe1` in a separate worktree
and apply [profile-before.patch](profile-before.patch), then build with
`--features profiling`. The reconstructed source hash was checked against the
report. The patch restores the earlier value loop and the then-current metadata
in the two other timing binaries; the profiled decode manifest already identifies
its diagnostic status.

Candidate attention took 41.4% of measured model wall time. Within the summed
attention-worker spans, shared score calculation took 46.3% and shared value
accumulation 28.5%. Worker times overlap model attention and include scheduling
delay; these scopes cannot be added. These diagnostic numbers motivated reducing
repeated value-weight work. They are not performance-gate evidence.

## First dimension-tile candidate (`cf7bbe1`)

Source SHA-256:
`89520473bf25ed47447e21660ae7386b6359e52c620cef15a0aecd023849c719`.

The ordinary-build manifests report Git head
`799e91575ed85d883c3a9c4bef7de98c14a55477` and `git_dirty: true`: the binaries were
built before those source edits were committed as `cf7bbe1`. Their compiled
source fingerprint matches that commit. Use the source hash to identify the
measured implementation; the recorded base revision alone does not reproduce it.

[Kernel sweep](kernel.jsonl): all 36 cases retain 21 timing pairs and pass finite,
bit-identical output checks. At context 4,096 and 90% sharing, batch 4/8/16 speedups
against the fallback were 1.580x / 1.582x / 1.538x. The kernel uses 32 query heads and
4 KV heads; the real Llama model uses 8 KV heads.

[Real-model shared run](shared90-tile12.jsonl) and [process report](shared90-tile12.time):
twelve paired repeats, sixteen decode steps per sequence, 4,096 prompt positions,
eight sequences, 90% requested physical sharing (3,680 common positions).
The [no-sharing control](shared0-tile12.jsonl) and its
[process report](shared0-tile12.time) use the same settings and source fingerprint,
with zero physically shared blocks. Both attention variants execute the existing
fallback in that control. All 48 timed real-model runs pass finite-logit checks
and have the same complete 128-token output and final-logit hash across both
sharing fractions. Checkpoint, config and input fingerprints also match.

| Physical sharing | Median paired speedup | Paired-bootstrap 95% interval | Paired min / max | Ratio of marginal medians |
|---|---:|---:|---:|---:|
| 90% requested | 1.135096x | [1.107372, 1.158449] | 0.933373x / 1.205629x | 1.122533x |
| 0% control | 0.989555x | [0.949928, 1.026792] | 0.803669x / 1.064546x | 0.989785x |

Each interval uses 10,000 paired resamples and fixed seed
`0x7061697265644349`. It assumes independent, exchangeable repeat pairs; desktop
load, thermal drift and order effects can violate that assumption. These are
descriptive intervals, not guaranteed bounds on future performance. See the
[methodology](../../shared-prefix-attention.md#profiling-and-paired-statistics).

| Physical sharing / variant | Median tokens/s | Pooled step p95 (ms) | Pooled step p99 (ms) | Maximum step (ms) |
|---|---:|---:|---:|---:|
| 90% / baseline | 27.841312 | 331.687500 | 476.831834 | 515.405250 |
| 90% / candidate | 31.252804 | 377.191459 | 613.610667 | 623.086500 |
| 0% / baseline | 22.286506 | 409.661250 | 747.868291 | 973.842000 |
| 0% / candidate | 22.058845 | 478.182083 | 908.779250 | 1451.897333 |

Each latency population contains 192 measured batch steps. These are model-batch
latencies, not client streaming latencies. The candidate tails worsen in both
samples. The shared paired median improves throughput by 13.5%, below the 15%
target. The control's paired median is about 1.0% slower; its interval crosses
one and extends just below 0.95, so it does not provide a strong 5% regression
bound. The performance gate remains unestablished. No request-lifecycle replay
was promoted from these results.

The shared candidate reports 256 executed shared-attention layer calls and
7,536,640 shared query positions per timed run. It retains 8,654,848 bytes
(8.254 MiB) of additional packed scratch; both counters and extra scratch are
zero for the control. Actual KV backing storage is 454 MiB for 90% sharing and
2,064 MiB for zero sharing. These storage differences are present in both
attention variants and are not a kernel memory saving.

Process maximum RSS is 2,666,364,928 bytes (2.483 GiB) for the shared process and
4,852,760,576 bytes (4.519 GiB) for the control. The macOS `time -l` reports cover
loading, preparation, warmup and both variants; they cannot assign RSS to one
attention variant. The final-logit SHA-256 shared by all 48 runs is
`c601cab3040fcc87e17402eda8509653a7507014a7890ca79b8b1b561f8e5758`.

## Rejected all-nonzero value specialization

[The separate kernel sweep](kernel-nonzero-rejected.jsonl) records an additional
specialization for value weights that are all nonzero. It passes all 36 cases
with finite, bit-identical outputs and retains 21 timing pairs per case.
To reconstruct its source, check out `cf7bbe1` in a separate worktree and apply
[nonzero-rejected.patch](nonzero-rejected.patch). Its source SHA-256 is
`8e154e334666baaed18a7148f131afccf4c2722c3966957184a4ad9bde3deb03`.

At context 4,096 and 90% sharing, batch 4/8/16 ratios against the fallback are
1.539x / 1.541x / 1.555x, compared with 1.580x / 1.582x / 1.538x for the first
dimension tile. This shows no consistent improvement; the separate noisy sweeps
do not establish a regression either. The specialization was not promoted to
real-model runs and was reverted. The retained implementation remains `cf7bbe1`
with source hash `89520473bf25ed47447e21660ae7386b6359e52c620cef15a0aecd023849c719`.
Do not pool these candidate histories across implementation fingerprints.

## Reproduction and validation

On this workstation, Cargo commands require
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`. Build the retained `cf7bbe1`
source without profiling for the timed comparisons:

```bash
DEVELOPER_DIR=/Library/Developer/CommandLineTools cargo build --release \
  --bin shared_attention_benchmark --bin shared_decode_benchmark
RAYON_NUM_THREADS=4 SHARED_ATTN_REPS=21 \
  target/release/shared_attention_benchmark > kernel.jsonl
MODEL_PATH=models/llama3-reference/checkpoint/model.safetensors QUANT=int8 \
RAYON_NUM_THREADS=4 SHARED_DECODE_CONTEXT=4096 SHARED_DECODE_BATCH=8 \
SHARED_DECODE_STEPS=16 SHARED_DECODE_REPS=12 SHARED_DECODE_PERCENTAGE=90 \
  /usr/bin/time -l target/release/shared_decode_benchmark \
  > shared90-tile12.jsonl 2> shared90-tile12.time
```

Run the model command again with percentage `0` and separate `shared0-tile12`
output names. Loading, prompt preparation and full-range warmup are excluded
from the benchmark timer. The `.time` reports include them. Run each process
serially without project builds alongside it. The diagnostic profile instead
uses its reconstruction patch, `--features profiling`, and one repeat; the
rejected experiment uses its own patch and only the ordinary kernel command.

The retained source hash and both reconstruction patches were verified. With
the additional all-nonzero numerical regression test retained after rejecting
the specialization, local validation passed 293 release tests, with zero
failures and one optional checkpoint test ignored. Formatting and strict
all-target, all-feature lint checks also passed. The optional real-checkpoint
test was not rerun; the separate real-model timing verification described above
compares the two attention schedules, not an independent Transformers oracle.
