# Apple M2 shared-prefix confirmation

**Status: performance gate not established.** Session B's no-sharing
descriptive interval has a lower endpoint of 0.908615, below the fixed 0.95
requirement. Its paired median (0.984985), aggregate throughput (0.968996),
pooled p95 ratio (1.009246) and both run-order medians meet their limits.
The production option remains disabled by default and the PR remains a draft.

This fixed confirmation changes measurement coverage, not attention arithmetic.
`protocol.json` declares two sessions using Llama 3.2 1B int8, context 4,096,
batch eight, four workers, 16 decode steps and 12 baseline/shared pairs per case.
Session A runs 90% requested physical sharing then 0%; B reverses that case
order. The requested 90% maps 3,680 common positions in whole blocks; the
control copies the same token history into separate physical blocks. Within
each case, baseline/shared order alternates. All four cases use
the same ordinary benchmark executable, with profiling disabled. Genuine
prompt KV is prepared from deterministic token IDs; this is not a language-quality
evaluation.

## Results and decision

| Session / sharing | Median paired elapsed ratio | Descriptive bootstrap 95% interval | Aggregate throughput ratio | Shared / baseline pooled step p95 |
| --- | ---: | --- | ---: | ---: |
| A / 90% | 1.196x | [1.186, 1.203] | 1.197x | 0.858x |
| A / 0% | 1.006x | [0.980, 1.036] | 1.003x | 1.015x |
| B / 0% | 0.985x | [0.909, 1.072] | 0.969x | 1.009x |
| B / 90% | 1.574x | [1.489, 1.662] | 1.583x | 0.664x |

Elapsed ratios are baseline/shared; aggregate ratios are shared/baseline
throughput. Values above one favor shared attention. The p95 column is a
latency ratio, so values below one favor it. These are model-batch step
latencies, not HTTP or request-streaming latencies.

Each session must meet the protocol's paired-median and interval-lower-bound
thresholds of 1.15 for sharing and 0.95 for the control, plus aggregate throughput,
pooled p95 and run-order safeguards. Inspect each session separately. Preserve
every completed or partial run, including failures; do not pool sessions,
drop outliers, replace prompts or rerun until passing.

- **Mixed arrivals:** all four runs match complete outputs and outcomes: four
  successful requests, eight length completions and 336 tokens per run, with no
  OOM, rejection or preemption. Every run peaks at 162/192 blocks and retains
  146 cached blocks. Enabled runs execute 784 shared-attention layer calls;
  disabled runs execute zero. The trace uses fixed 0/500/1,500/3,000 ms arrivals
  and two alternating repeats per configuration.
- **Pressure:** all three runs match the roomy oracle: two length completions
  and 128 tokens per run. Both tight runs reach 132 blocks and record one
  preemption, 33 recomputed tokens and one COW copy; the shared run executes
  496 shared-attention layer calls. The roomy run peaks at 136/192 blocks with
  no preemption or recomputation. For this prompt, admission needs 130 blocks,
  one sequence can finish in 132, and simultaneous siblings would need 136.

`trace-notes.md` specifies the inputs and storage arithmetic. Both replays use
prefix caching, 32-token prefill chunks/budget and no speculation. Their explicit
EOS override is valid token 0 with no extra EOS IDs. Any early EOS invalidates
the observation and remains reported. These traces contain no active
wall-clock cancellation; deterministic cancellation evidence remains in the
earlier [lifecycle archive](../shared-prefix-m2-score-tile/lifecycle/).

The independent audit validates all 96 timed decode runs, full token/final-logit
and every-step warmup-logit identity across all cases, every declared path count,
all seven lifecycle runs and deadline/delivery-derived latency summaries.
The only failed check is the B / 0% interval lower endpoint. Its baseline-first
and shared-first paired medians are 0.984985 and 0.958897, respectively.
Both shared cases clear every prescribed performance check; neither repairs
the failed control bound.

| Mixed run | Elapsed (s) | Deadline-based TTFT p95 (s) | Observed inter-token p95 (ms) | Dispatch lag p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| mixed-baseline / 1 | 217.000 | 187.140 | 2736.765 | 2289.991 |
| mixed-shared / 1 | 209.667 | 184.438 | 2795.630 | 2151.086 |
| mixed-shared / 2 | 216.844 | 192.091 | 2960.148 | 2342.492 |
| mixed-baseline / 2 | 217.501 | 187.583 | 2950.525 | 3134.377 |

These four mixed runs have matching outputs, but their latency observations
are not a regression bound. The pressure runs take 180.733, 107.827 and 60.271
seconds in fixed roomy/baseline/shared order. One unpaired observation per
configuration on this changing host cannot support a pressure-speedup claim.
The tight runs retain 130 cached blocks; the roomy run retains 128. Unit tests
also check that reset releases every KV reference.

Host conditions changed substantially. Whole-process no-sharing time rose from
395.38 seconds in A to 1,613.53 in B; retired instructions were approximately
13.93 and 14.05 trillion. B's before/after one-minute load averages were 4.63
and 24.36. Timed process counters, every per-run tail, pooled p99/max and order
strata remain in `analysis.json` and the raw reports. These observations do not
identify the cause or excuse the failed bound. No observations were discarded.

## Source and measurement boundaries

The frozen source is commit `fdd76e3b799056292539f8edaa3129e65c5cd0b5`, with
compiled source SHA-256
`a1fc6917c6009b763766e3cfbc67387ddcd6bd43d2623b0306d9ff3e2bdf39af`.
`compiled-build.json` retains its build-time metadata; `build.json` records
both executable digests, checkpoint/config digests and frozen input hashes.
The checkpoint SHA-256 is
`68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.

`source-verification.json` independently reconstructs the source hash from Git
blobs. Later commit `160426fbd4561bf09295b927da97fba2ea762ce7` changes only the
untimed logit serializer from `chunks_exact_mut(4)` to `as_chunks_mut::<4>().0`.
Both traverse the same 256 four-byte chunks of a 1,024-byte buffer, in the same
order. Its source hash differs; timed decode source is unchanged. This is a
source-level equivalence check, not a new performance measurement or proof of
identical executable bytes.

Every warmup hashes every complete logit row at every decode position outside
timing. The measured loop includes model computation, vocabulary projection,
finite-logit scanning, greedy selection and token recording. Complete timed
tokens, final-logit digests, all-step warmup digests and step times are retained.
Loading, prepared KV, warmup, validation and reporting are excluded. Replay
timing instead includes prompt processing, scheduling and delivery after initial
loading/warmup. Neither path measures HTTP transport.

`run_readiness.py` sets four workers, the four-wide projection setting and
`PAGED_INFER_ATTN_LANES_PER_THREAD=2` before every process. Benchmark manifests
record the lane setting. Replay manifests omit it: replay lane provenance comes
from the archived runner and fixed protocol, not an independent manifest field.

Executables are deliberately absent from the archive. Their recorded hashes
remain available; rebuilding creates a new binary identity. The analyzer reports
missing executables as unavailable for independent rehash, rather than claiming
to have verified their bytes.

## Independent analysis

After every process has finished, run the standard-library analyzer:

```sh
python3 analyze_readiness.py --directory . --output analysis-recheck.json
```

It requires all four decode cases and both lifecycle reports, checks input/build
identity and ordering, compares complete outputs, reconstructs paired ratios,
bootstrap intervals, aggregate throughput, order strata and latency summaries,
and evaluates the predeclared gates. Output files must be new. A nonzero exit
means missing/invalid evidence or a failed measurement gate. Add
`--check-binaries` only when the frozen executables are available for rehash.
The analyzer does not check CI or decide whether to merge. The archived
`analysis.json` was generated with `--check-binaries` while both frozen
executables were available and matched their recorded digests.

`analyzer-validation.json` records eleven negative checks, including truncated
reports, incorrect path counts or warmup hashes, equally shortened/EOS outputs,
absent pressure and incorrect deadline/inter-token statistics. Every mutation
is rejected for its expected reason; all 28 original evidence files remain
unchanged. Recheck this separately with:

```sh
python3 validate_analyzer_negatives.py --directory . --output validation-recheck.json
```

`local-validation-results.json` and its three logs record the post-measurement
formatting, strict lint and 17 targeted release tests on `160426f`; all pass.
The earlier full release suite passed 306 tests, with one optional checkpoint
test ignored, and both feature-enabled profiling tests passed.

`SHA256SUMS` covers every archived file, including the runner. Verify it from
this directory with `shasum -a 256 -c SHA256SUMS`.

## Fresh-directory reproduction

Use Apple M2/macOS, Rust 1.93.1 and the matching checkpoint with its adjacent
configuration files. Start in the repository root. This creates a detached
checkout and new evidence directory, regenerates build metadata from the new
binary, then runs the fixed sequence exactly once. Do not copy archived output
reports or `build.json`/`compiled-build.json` into the new evidence directory.

```sh
confirmation_archive="$PWD/docs/measurements/shared-prefix-m2-confirmation"
confirmation_model="$PWD/models/llama3-reference/checkpoint/model.safetensors"
confirmation_dir=$(mktemp -d /tmp/paged-infer-confirmation.XXXXXX)
git worktree add --detach "$confirmation_dir/source" fdd76e3b799056292539f8edaa3129e65c5cd0b5
mkdir "$confirmation_dir/evidence"
cp "$confirmation_archive"/protocol.json "$confirmation_archive"/run_readiness.py \
  "$confirmation_archive"/analyze_readiness.py "$confirmation_archive"/mixed-*.json \
  "$confirmation_archive"/pressure-*.json "$confirmation_dir/evidence/"
cd "$confirmation_dir/source"
env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
  DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  CARGO_TARGET_DIR="$PWD/target" cargo +1.93.1 build --release --no-default-features \
  --bin shared_decode_benchmark --bin workload_replay
target/release/workload_replay --threads 4 --repeats 1 --verify \
  --output "$confirmation_dir/evidence/fixture-build-probe.jsonl"
python3 - "$confirmation_dir/evidence" <<'PY'
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
manifest = json.loads((folder / 'fixture-build-probe.jsonl').read_text().splitlines()[0])
build = manifest['environment']['build']
assert manifest['profiling_enabled'] is False
assert build['source_sha256'] == 'a1fc6917c6009b763766e3cfbc67387ddcd6bd43d2623b0306d9ff3e2bdf39af'
with (folder / 'compiled-build.json').open('x') as out:
    json.dump({'compiled_source_sha256': build['source_sha256'], 'build': build,
               'profiling_enabled': False}, out, indent=2)
    out.write('\n')
PY
python3 "$confirmation_dir/evidence/run_readiness.py" \
  "$PWD/target/release/shared_decode_benchmark" "$PWD/target/release/workload_replay" \
  "$confirmation_model"
python3 "$confirmation_dir/evidence/analyze_readiness.py" \
  --directory "$confirmation_dir/evidence" --output "$confirmation_dir/evidence/analysis.json"
```

Finish all builds/tests before timing and run no concurrent project workload.
The original host kept desktop applications running; before/after load and
thermal snapshots do not establish isolation. Process CPU/fault/context-switch
deltas are diagnostic and cannot identify the cause of a stall. Whole-process
`.time` elapsed/RSS values include preparation and warmup, not per-variant memory.
The deterministic 10,000-resample paired bootstrap remains conditional on its
sampling assumptions; it does not correct serial drift, thermal effects or
order dependence. Retain every new observation and report failures explicitly.
