# Decode-priority chunked prefill

The scheduler advances existing decoders before processing a bounded number of
prompt positions. A long prompt retains its position and KV mapping across
`Engine::step()` calls, so active streams regain control between slices. The
first output token is sampled when the final prompt slice completes; ordinary
decode for that new sequence starts on the following step.

One unfinished prefill is retained at a time. It stays ahead of later arrivals,
including shorter prompts. The scheduler maps its complete prompt, but publishes
new prefix-cache blocks only after every prompt position has completed every
layer. Intermediate slices skip the vocabulary projection; only the final slice
needs next-token logits.

## Controls

| Setting | Default | Scope |
|---|---:|---|
| `EngineConfig.max_prefill_tokens_per_step` | 32 | Maximum prompt or resumed-sequence positions processed per scheduler step |
| `EngineConfig.prefill_chunk_size` | 32 | Positions grouped in each model matrix batch |
| `ServeConfig.max_jobs_per_step` | 32 | Jobs received before a server scheduler step, including rejected and cancelled jobs |

The prefill allowance must be positive. It bounds prefill work only: existing
decoders retain priority and are outside that allowance. It is not a total-token
or wall-time limit. A server delivers this step's output after the engine returns,
so the prefill slice still contributes to observed inter-token latency. A smaller
allowance offers more frequent scheduler returns and can reduce batching
efficiency; measure the tradeoff on the intended model and workload.

The HTTP binary exposes the settings independently:

```bash
PREFILL_TOKENS_PER_STEP=16 PREFILL_CHUNK_SIZE=8 MAX_JOBS_PER_STEP=16 \
  cargo run --release --bin http_server
```

The job cap includes the job that wakes an idle engine thread. Continuous arrivals
therefore cannot extend intake indefinitely before an existing stream gets its
next scheduler step. Cancellation flags are checked between steps. A synchronous
model call cannot be interrupted halfway through a slice.

## Memory pressure and accounting

An unfinished prompt cannot keep its allocation at the expense of mandatory
decode. When necessary, it releases its mapping and returns to the front of the
waiting queue for recomputation. Sequence identity and sampling state survive
that retry. New arrivals remain behind it; active requests have finite output
budgets, so arrivals cannot continually overtake the retained prompt.

Cancellation and reset release retained mappings. Completed immutable prefix
blocks may remain cached under the existing eviction policy. Partially computed
new blocks are never published.

Once a request has been admitted, cancellation emits one terminal event and
completion per reserved sample, even when no output token exists yet. Cancellation
before the first admission keeps the existing request-level outcome with no
sample IDs. A cancelled partial prefill has no measured time to first token.

HTTP `/stats` and Prometheus `/metrics` expose:

- `requests_prefilling` and `pending_prefill_tokens`: the retained prompt and
  its remaining positions.
- `last_prefill_tokens`: positions processed in the last scheduler step.
- `prefill_chunks` and `prefill_preemptions`: scheduler slices and discarded
  unfinished mappings.
- `prompt_tokens_reused`: actual prefix positions skipped at initial admission.
  Uncomputed positions from cancelled prompts do not count as cache reuse.

The older `prefix_cache_tokens_saved` counter counts positions covered by reused
blocks across admissions, including replayed boundary tokens. Use
`prompt_tokens_reused` for actual skipped positions on initial admission.

Prometheus counters have a `paged_infer_` prefix and `_total` suffix; gauges omit
the suffix. `requests_queued` includes the retained unfinished prefill, while
`sequences_active` counts decoders. `prompt_tokens_prefilled` counts actual fresh
request input processed, including retries. `recomputed_tokens` includes resumed
sequence input and repeated positions after prefill eviction; these counters can
overlap and must not be added as independent token populations.

Replay reports include the new counters, plus the maximum observed number of
prefilling requests and pending positions at event/step boundaries. Those sampled
maxima can miss a prefill that starts and finishes in a single step. Allocator KV
high-water tracking still records transient allocations exactly. Older JSONL
reports deserialize with zero defaults for newly introduced counters; those
defaults do not constitute historical measurements.

## Compare budgets with replay

The bundled mixed workload starts an existing decoder, then introduces a long
prompt and shorter requests. Run two budgets against the same synthetic weights:

```bash
cargo run --release --bin workload_replay -- \
  --workload workloads/mixed.json \
  --config workloads/configs/baseline.json \
  --config workloads/configs/chunked-prefill.json \
  --threads 4 --repeats 5 --verify --output /tmp/paged-prefill-comparison.jsonl
```

The baseline configuration inherits the 32-position allowance; the chunked
configuration uses 8. Both use the current decode-priority scheduler. Compare
existing-stream inter-delivery latency, TTFT, completion throughput, prefill
preemptions and recomputed tokens. Inspect per-request records as well as pooled
tails: a long prompt and an existing decoder experience different tradeoffs.

This step-clock trace supports repeatable scheduling and exact token/finish-reason
comparison. Millisecond traces measure dispatch delay from fixed arrival times.
Changing the budget can change how many tokens precede a scheduled cancellation;
exact comparison correctly reports such a difference. See the
[replay guide](workload-replay.md) for clock and measurement semantics.

## TinyLlama latency experiment

Measured on 2026-09-14 with TinyLlama 1.1B, f32 projections, Apple M2 / 16 GiB,
four Rayon workers, NEON and Rust 1.93.1 release builds. The old scheduler is
the clean merged `1ba8e2d` baseline. The candidate uses matrix batches of 32
in both budget variants. All nine runs, build/input fingerprints and complete
outputs are recorded in [the measurement file](measurements/chunked-prefill-m2.json).
The candidate fingerprint identifies the measured source snapshot; subsequent
source edits only clarified API comments and metric help text.

The trace submits an 11-token prompt requesting 40 output tokens at step 0, then
a 256-token prompt requesting four tokens at step 2. These are TinyLlama-tokenized
inputs, with BOS already included, greedy sampling and no prefix reuse. The
latency configuration overrides EOS to ID 0 so the streams reach their length
budgets; this tests controlled traffic rather than production stopping behavior.
All configurations produced identical 44-token outputs and length finishes.

Each table entry is the median of three per-run measurements. Variants ran in
rotating order, each in a separate process, with loading, hashing, allocation and
warmup outside the timer. This was a developer workstation with uncontrolled
system load; no other benchmark or build ran alongside these measurements.

| Scheduler | Existing stream p95 delivery gap | Existing stream worst gap | Long prompt TTFT | Useful output tokens/s |
|---|---:|---:|---:|---:|
| Previous, whole-prompt prefill | 236 ms | 5,978 ms | 5,978 ms | 4.60 |
| Decode priority, budget 32 (default) | 839 ms | 978 ms | 6,658 ms | 4.73 |
| Decode priority, budget 8 | 227 ms | 305 ms | 6,769 ms | 5.75 |

Chunking distributes prefill cost across more stream deliveries: the default
shortened the worst pause but increased the stream's p95 gap and the long
prompt's TTFT. The 32-token runs ranged from 8.77 to 14.18 seconds total, so the
small throughput difference against the old scheduler is not evidence of a
general speedup. Budget 8 performed well on this trace, but one model and trace
do not establish a universal optimum. The default remains 32, matching the
existing matrix batch size; tune the independent allowance for actual traffic.

Reproduce the current default using the committed real-model trace:

```bash
cargo run --release --bin workload_replay -- \
  --model models/tinyllama-1.1b/model.safetensors \
  --workload workloads/tinyllama-long-prefill.json \
  --config workloads/configs/latency-prefill.json \
  --threads 4 --repeats 3 --verify --output /tmp/prefill-latency-32.jsonl
```

For budget 8, copy that configuration and add
`"max_prefill_tokens_per_step": 8` inside `engine`, keeping the `latency` name.
Run it separately with a new output path and
`--compare-to /tmp/prefill-latency-32.jsonl` to verify outputs. To compare with
the old scheduler, build `workload_replay` at `1ba8e2d` in a separate checkout,
pass the current trace and unchanged latency configuration by absolute path,
and use its completed report as `--compare-to` for the candidate. Rotate the
order of separate old/default/8-token runs when collecting timing evidence.
