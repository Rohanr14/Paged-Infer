# Shared-prefix CPU attention evidence

Measured 2026-09-15 on Apple M2 / 16 GiB, four Rayon workers, NEON, macOS,
Rust 1.93.1 release builds. Builds selected the installed Apple Command Line
Tools with `DEVELOPER_DIR=/Library/Developer/CommandLineTools`. No other project
build or benchmark ran concurrently with timing; background system load was not
controlled. See [the design and decision](../../shared-prefix-attention.md).

## Final source snapshot

`017c3e8` implementation source SHA-256:
`51922b002c6a1930a0ad96c97cbd9fbdd1d7f062660c9000c187a72871efd40a`.
The reports were generated before committing that source, so their Git head is
its base and `git_dirty` is true. The complete compiled-source hash was verified
against the committed implementation. Later changes only move/clarify tests and
document measurements; those files are outside the compiled-source hash.

- [Kernel sweep](kernel-m2-verified.jsonl): 36 cases, 21 pairs each, every finite
  output checked bitwise. TinyLlama attention shape (32 query / 4 KV heads).
- [Real-model 90% sharing](llama3-int8-4096-shared90-verified.jsonl) and
  [process memory/timing](llama3-int8-4096-shared90-verified.time).
- [Real-model zero-sharing control](llama3-int8-4096-shared0-verified.jsonl) and
  [process memory/timing](llama3-int8-4096-shared0-verified.time).

The real model is Llama 3.2 1B, int8 projections, 16 layers, 32 query / 8 KV heads,
4,096 prompt positions, eight sequences and eight steps per sequence. Each
variant has three full-range-warmed timed runs. Whole prompt KV is computed from
weights once, then shared or copied into separate physical blocks before adding
distinct private tokens. Fixed token budgets ignore EOS. These are synthetic
content IDs with genuine KV, not a language-quality evaluation. The twelve final
runs share identical complete outputs and final-logit hashes.

The JSONL manifests record full inputs, checkpoint/config/input hashes, model
shape, timing scope, memory quantities and build fingerprints. Each run retains
all step timings and output tokens. Summary `median_speedup` is the ratio of
median elapsed times, which differs from the median of paired speedups. The
separate `.time` files retain macOS process maximum RSS including loading,
preparation and both variants; they cannot assign peak RSS to one variant.

## Development history

These earlier complete real-model runs are retained and not pooled with the
final source snapshot:

- [Per-token value accumulation](llama3-int8-4096-shared90.jsonl),
  [process report](llama3-int8-4096-shared90.time): source `cf2648297632d238ffe44eb65430df2e31173ec91fa5ffe40da0f5da4b26a06e`,
  ratio of medians 0.788x.
- [Blockwise register accumulation](llama3-int8-4096-shared90-block-sums.jsonl),
  [process report](llama3-int8-4096-shared90-block-sums.time): source
  `5ab6f75391494f5a33bee547ee1eea0abd5a2a5321e779dcae7c7211442b9f1f`,
  ratio of medians 1.106x. The final implementation uses the same kernel;
  its harness preallocates each output vector separately to remove in-timer
  allocations caused by cloning empty vectors.

The point estimate for the final shared run is 1.205x, but its median paired
speedup is 1.114x and the fallback control is noisy. The declared performance
gate is not yet established. Keep the experiment in draft and disabled.
