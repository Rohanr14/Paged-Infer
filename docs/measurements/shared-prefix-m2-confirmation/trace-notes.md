# Bounded PR 16 request traces

These traces use the same first 2,048 token IDs as the earlier real-model
lifecycle workload. The initial 2,049-token prompt is copied exactly. Other
inputs are deterministic synthetic token IDs; this is a model-execution and
request-lifecycle check, not a language-quality evaluation.

Both traces use int8 Llama 3.2 1B, four workers, 16-token KV blocks, batch cap 8,
32-token prefill chunks and per-step prefill budget, prefix caching enabled,
streaming enabled and speculation disabled. Each request explicitly selects
seed 42, greedy temperature 0, top-p 1 and top-k 0. Use ordinary release builds
and a fresh engine/cache per run.

The controlled EOS override is `eos_token: 0, extra_eos_tokens: []`. This does
not make early EOS impossible. Require all specified output lengths and
`length` finish reasons; any emitted EOS/early termination invalidates that
observation and must be retained and reported, not silently replaced. The
real-model replay validator rejects the out-of-vocabulary 4294967295 sentinel.

## Mixed millisecond arrivals

Use `mixed-workload.json` with `mixed-baseline.json` and `mixed-shared.json`.
Arrivals are fixed open-loop deadlines, not scheduler steps:

| Deadline | Prompt | Samples | Output budget per sample |
| --- | --- | ---: | ---: |
| 0 ms | Original 2,049-token prompt | 4 | 64 |
| 500 ms | Unrelated 17-token prompt | 1 | 8 |
| 1,500 ms | Original 2,048-token prefix + 17 private tokens | 2 | 32 |
| 3,000 ms | Unrelated 257-token prompt | 1 | 8 |

There are four requests, eight sequences and 336 expected output tokens. The
conservative simultaneous mapped-storage bound is 128 shared full blocks +
4 * 4 private blocks + 2 * 3 private blocks + 2 short-request blocks + 17
unrelated-request blocks = 169, below the 192-block pool. Some private full
blocks may also be shared within the second fork, making this an upper bound.
This is a mixed-arrival/queueing trace, not a pressure test.

Require complete baseline/shared output and outcome parity, no rejection or
OOM, positive shared-attention layer calls only in the shared configuration,
and every request settling once. Report actual occupancy and arrival dispatch
lags. TTFT and end-to-end time include lateness relative to scheduled deadlines.
Do not require exactly eight simultaneous active sequences or assume that a
particular deadline is dispatched during decode: timing and admission determine
that. Active cancellation remains covered by the separate deterministic
step-clock lifecycle trace; this trace has no timing-dependent cancellation.

## Forced pressure

Use `pressure-workload.json` with `pressure-roomy.json`,
`pressure-baseline.json` and `pressure-shared.json`. Compare both tight runs
against the roomy baseline's complete output and outcome oracle.

The 2,049-token prompt forks into two samples of 64 outputs each. Initial
admission needs ceil(2049 / 16) + 1 = 130 blocks. A single sequence requires at
most ceil((2049 + 64 - 1) / 16) = 132 mapped blocks; the final emitted token does
not need its KV written. Together the siblings would require 128 shared full
blocks + 2 * 4 private blocks = 136, exceeding the 132-block tight pool while
either sequence can finish alone. The shared interval includes 2,048 positions,
so the Llama shared score path is eligible before pressure changes scheduling.

Require two 64-token `length` completions, exact full outputs against the
192-block oracle, positive preemption/recompute/COW counters in both tight
runs, peak 132 blocks, and positive shared-attention calls in the shared run.
The roomy run must have no preemption. Do not interpret admission deferral alone
as sufficient proof of pressure. Cache retention can leave allocated blocks
after completion; engine reset must release all KV references. These bounded
correctness observations do not establish the throughput or tail-latency gate.
