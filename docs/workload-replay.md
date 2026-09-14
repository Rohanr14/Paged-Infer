# Workload replay

`workload_replay` runs versioned request traces directly against the engine and
writes a JSONL report. It establishes a baseline for scheduler changes: record
arrivals, full output tokens, terminal outcomes, latency distributions, useful
completed throughput, prefix reuse, recomputation and exact KV allocation peaks.

## Run the bundled fixture

No downloads or Python dependencies are required:

```bash
cargo run --release --bin workload_replay -- \
  --workload workloads/mixed.json --threads 4 \
  --config workloads/configs/baseline.json \
  --config workloads/configs/no-prefix.json \
  --repeats 3 --verify --output /tmp/mixed-replay.jsonl
```

The output path must be new; an existing report is never overwritten. Omit
`--output` to write JSONL to stdout; progress goes to stderr. All repeats are
retained. Configurations run interleaved, rotating the first configuration each
round, and each run receives a fresh engine with a warmup followed by reset.
Only one loaded, converted weight set and one KV pool exist at a time. Model
loading, hashing, engine allocation and warmup are outside the run timer.

The default model is `tests/fixtures/tiny_llama.safetensors`, with architecture
read from its adjacent `config.json`. This is a synthetic correctness fixture,
not a representative language model performance result. It declares no special
tokens, so this default mode explicitly disables EOS (`u32::MAX`) and BOS. The
manifest records `synthetic_fixture: true` and the effective settings.

For real weights, pass the safetensors file explicitly:

```bash
cargo run --release --bin workload_replay -- \
  --model models/tinyllama-1.1b/model.safetensors \
  --workload workloads/mixed.json --threads 4 \
  --config workloads/configs/baseline.json \
  --repeats 3 --verify --output /tmp/tinyllama-replay.jsonl
```

An explicit missing model or missing/malformed `config.json` fails immediately.
Real checkpoint BOS, every EOS ID, and context limits come from the model
profile, including `generation_config.json`. Engine overrides start from those
settings, and the manifest contains the complete effective configuration.
Workload token IDs are already complete model inputs: no tokenizer or BOS
insertion runs in this driver. Bundled IDs are synthetic patterns; prepare
model-tokenized traces to represent your actual traffic. `--quant int8` is an
explicit alternative to default f32 projection storage.

## Workload format

```json
{
  "version": 1,
  "clock": "steps",
  "events": [
    {"type": "submit", "at": 0, "id": "a", "tokens": [1, 3, 4],
     "max_tokens": 16, "samples": 2, "seed": 42},
    {"type": "cancel", "at": 3, "id": "a"}
  ]
}
```

Events must be in nondecreasing `at` order; array order breaks ties. Request IDs
must be unique, cancellations must follow the corresponding submission in the
trace, prompts and generation budgets must be nonempty/positive, and every
submission needs a seed. `samples` defaults to 1. Sampling overrides are
`temperature` (default 0), `top_p` (default 1), and `top_k` (default 0). A request
seed controls its own sample streams, so unrelated request IDs do not change
its RNG stream. Unknown schema fields fail validation. Requests outside model
vocabulary, context or possible pool capacity are recorded as rejected requests.

`clock: "steps"` dispatches arrivals before the numbered engine step. It makes
arrival/cancellation boundaries deterministic and skips empty ranges without
sleeping. The first step is 0. There is no wall-time deadline for a step index,
so dispatch lag is null and latency starts at actual driver submission.
Changing the scheduler can change how much work finishes by the same step.

`clock: "milliseconds"` makes `at` a deadline relative to the measured run's
start. The driver sleeps only when idle and dispatches due events between
engine steps. Since a step is synchronous, a long prefill can delay later
arrivals. The report preserves scheduled time, actual submission time and
that dispatch lag; TTFT/end-to-end latency include the lag. This exposes
blocking rather than silently shifting the offered arrival schedule. A future
scheduler can use the same wall-time trace to measure the improvement.

`--max-steps` (default 100000) and `--timeout-secs` (default 60) bound each run.
Checks happen between steps, so a forward pass cannot be interrupted. Hitting
a limit exits with an error, and the partial report lacks a completion record.

Bundled scenarios:

| Workload | Purpose |
| --- | --- |
| `burst.json` | Eight simultaneous independent requests |
| `mixed.json` | Existing decode followed by long and short prefills |
| `shared-prefix.json` | Repeated 32-token prefixes and a later arrival |
| `forks.json` | Multiple continuations of prompts ending inside a block |
| `cancellation.json` | Cancellation before admission and during generation |
| `pressure.json` | Forks and a waiting request under a constrained KV pool |
| `wallclock.json` | Millisecond arrivals plus a timed cancellation |
| `tinyllama-long-prefill.json` | TinyLlama-tokenized 11-token stream prompt followed by a 256-token prompt; requires the real TinyLlama checkpoint |

Use the last trace with `workloads/configs/latency-prefill.json`. That configuration
overrides EOS to ID 0 to keep the test streams running for their output budgets;
it is a controlled latency experiment, not production stopping behavior. The
[chunked prefill guide](chunked-prefill.md) records its inputs and measured tradeoffs.

```bash
cargo run --release --bin workload_replay -- \
  --workload workloads/pressure.json \
  --config workloads/configs/tight-pool.json --threads 4 \
  --repeats 3 --output /tmp/pressure-replay.jsonl
```

The tight-pool configuration is intentionally small. Use `pressure.json` to
exercise it; pairing it with a prompt larger than its capacity measures request
rejection instead. Preemptions, deferred steps, recomputed tokens and OOM
terminal outcomes distinguish pressure from successful useful throughput.

Configuration files contain a unique name plus partial `EngineConfig`
overrides, for example:

```json
{"name": "small-batch", "engine": {"max_batch_size": 2, "total_blocks": 64}}
```

Use repeated `--config` arguments to compare variants. With no config files,
`baseline` uses model-aware engine defaults. Streaming must stay enabled.
Dimensions, allocation arithmetic, context bounds and sampling values are
validated before engine allocation. The driver respects `RAYON_NUM_THREADS`;
`--threads` explicitly sets the Rayon worker count.

## Read and compare results

Each line is one JSON object:

1. `manifest`: schema version, trace, effective configurations, SHA-256 of
   checkpoint/config/generation-config/workload bytes and configuration values,
   quantization, repeats/limits and environment information.
2. `run`: configuration name, repeat, order, weight storage bytes, and the full
   replay report. Requests include every generated token and terminal reason,
   submission/cancellation timing, per-sequence timing and delivery batches.
3. `config_summary`: nearest-rank p50/p95/p99/max across all run values, plus
   pooled latency distributions using all observations across all repeats.
4. `verification`: a completion marker and exact output verification result.
   `passed` is null when verification was not requested.

The environment records OS/architecture, CPU when available, actual Rayon
threads, selected SIMD backend, and debug-assertion/build mode. The nested
`build` object embeds the build's actual `RUSTC -Vv`, target, profile,
optimization/debug settings, Rust flags, Git revision/dirty state, and a SHA-256
of `src`, Cargo files, `build.rs`, and `.cargo` configuration. Runtime
`rustc_on_path`, `git_head_at_run`, and `rustflags_env_at_run` remain separate:
they can differ when a previously built executable runs in another checkout.
Keep the build command and checkout with published benchmark results.

Every sequence's first observable token yields one TTFT sample. Engine queue
time is the engine's own admission wait; for wall-clock traces, driver dispatch
lag is reported separately. End-to-end samples include terminal sequences,
including cancellations and failures. Rejections have no sequence latencies.
Successful-request throughput counts requests whose sequences all finish with
EOS or length; useful-token throughput counts tokens from successful sequences
and excludes cancelled/OOM sequences. All rates use the full measured duration,
including idle time required by the trace and scheduled cancellations whose
requests have already finished. The complete event horizon is intentional;
remove trailing no-op cancellations when they do not belong to offered traffic.
These are completion-based rates,
not deadline/SLO-qualified goodput.

A delivery is the token batch observable when the driver drains after a step or
cancellation. Tokens within one delivery share a timestamp, so observed
inter-token latency has zero gaps within that batch. It does not estimate
hidden per-token production times. `inter_delivery_ms` contains only gaps
between nonempty deliveries. This distinction matters for speculative decoding,
which can emit several tokens in one step.

`kv_reserved_bytes` is the preallocated KV buffer. `peak_allocated_blocks` and
`peak_occupied_kv_bytes` track the allocator's exact high water, including blocks
allocated and freed within a step and retained prefix-cache blocks. They are
not process RSS: weights, scratch, allocator overhead and OS page residency are
outside that value. `weight_bytes` is weight storage accounting, also not RSS.
No subprocess memory sampler runs inside the measurement.

Use `--verify` to compare exact tokens and terminal outcomes across every
configuration/repeat. It fails the command when an output differs, while keeping
the measured report for inspection. Timing is excluded from equality checks.
Verification is opt-in: cancellations at fixed wall time or at a scheduler step
can legitimately retain different token prefixes across configurations.
Similarly, a small pool may produce OOM while a larger pool completes; this is
a measured outcome, not a throughput win. Different numerical arithmetic or
quantization can also change sampled tokens.

A later build can compare against a completed prior report with matching input
fingerprints and configuration names:

```bash
cargo run --release --bin workload_replay -- \
  --workload workloads/mixed.json --threads 4 \
  --config workloads/configs/baseline.json --repeats 3 \
  --compare-to /tmp/mixed-replay.jsonl --output /tmp/mixed-replay-next.jsonl
```

`--compare-to` checks every matching prior run, refuses changed model/workload
fingerprints, and verifies outcomes rather than enforcing a noisy latency
threshold. Inspect all repeats and pooled tails when evaluating performance.
This is a direct-engine harness; HTTP/socket backpressure and slow readers need
the separate server probe and are not simulated by these timing numbers.

## Exercise a client that stops reading

```bash
cargo run --release --bin stalled_reader_probe
```

This starts the fixture server on ephemeral loopback and emits a JSON evidence
report. An identical, fully drained 512-token streaming request must complete
first. A second client reads one token frame, then keeps its socket open without
reading. A healthy follower must complete, the stalled generation must stop
before its token budget, and every KV block must be returned before the held
socket is closed. A deadline or insufficient pressure makes the command fail.

The probe deliberately pads the model name by 64 KiB per SSE frame to fill OS
socket buffers with a small model. It is a synthetic transport stress test;
its latency is not representative model-serving performance. Reports include
padding, read-ahead, draining-control results and all observed counters. The
server does not expose per-cause cancellation counters, so the report identifies
queue saturation or socket write failure as inferred causes and does not claim
to have measured a write-timeout error directly. `--help` lists pressure controls.
For example, zero padding may let the entire response fit in socket buffers;
the probe then correctly reports insufficient cancellation evidence.

CI runs replay parity and the socket probe on Linux and macOS, retains their
JSON reports as artifacts, and makes no performance-speedup assertion from
shared-runner timings.
