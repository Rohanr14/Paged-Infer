**Llama 3 reference validation.** This closes the short-context real-checkpoint
validation gap left by PR #10. It also adds small, independent numerical
fixtures that run without Python or model downloads in the ordinary Rust suite.

The oracle executes Transformers' own `LlamaForCausalLM`,
`LlamaRotaryEmbedding`, and `apply_rotary_pos_emb`. It does not transcribe the
Rust implementation. Both the oracle and Rust use float32 CPU arithmetic on
the original BF16 checkpoint values, with full causal attention. Transformers
uses eager attention, evaluation mode, and no remote model code.

| Reference | Pinned identity |
|---|---|
| Transformers | `4.52.4`, commit `51f94ea06d19a6308c61bbb4dc97c40aabd12bad` |
| PyTorch | `2.7.1` (Linux CPU wheel `2.7.1+cpu` also accepted) |
| Public checkpoint | `unsloth/Llama-3.2-1B` at `9535bd9b1d1dea6acafbdc4813b728796aeb28da` |
| Official checkpoint | `meta-llama/Llama-3.2-1B` at `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Weight SHA256, identical for both | `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a` |
| Weight file size | 2,471,645,608 bytes |

The checkpoint has 16 layers, head dimension 64, theta 500000, scaling factor
32, frequency factors 1/4, and original context length 8192. Source links:
[pinned model/config](https://huggingface.co/unsloth/Llama-3.2-1B/tree/9535bd9b1d1dea6acafbdc4813b728796aeb28da),
[Transformers rotary implementation](https://github.com/huggingface/transformers/blob/51f94ea06d19a6308c61bbb4dc97c40aabd12bad/src/transformers/modeling_rope_utils.py),
[Transformers Llama implementation](https://github.com/huggingface/transformers/blob/51f94ea06d19a6308c61bbb4dc97c40aabd12bad/src/transformers/models/llama/modeling_llama.py).
The public copy retains its publisher's license; weights are not committed to
this repository.

**Measured September 8, 2026, on Apple Silicon.** The real sequence contains
52 prompt tokens and eight greedy continuation tokens, spanning four 16-token
KV blocks. Every vocabulary logit is checked at all 60 incremental positions.
Whole and resumed batched prefill additionally check their final rows and the
16-token priming boundary.

| Check | Maximum absolute logit difference | Greedy agreement |
|---|---:|---|
| Real checkpoint, Rust vs Transformers | `8.070469e-5` | 60/60 incremental rows; all prefill checks |
| Tiny scaled fixture, Rust vs Transformers | `3.8146973e-6` | 40/40 incremental rows; all prefill checks |
| Transformers cached decode vs full prefill, real checkpoint | `6.1035156e-5` | 60/60 |
| Disable scaling, real checkpoint reference | about `0.391` | Negative control measures changed logits |
| Disable scaling, tiny reference | about `0.00419` | Rust comparison rejects this change |

These measurements validate this checkpoint, dtype policy, and short context;
they do not prove every Llama checkpoint or full 128K-context generation is
equivalent. Separate rotary fixtures test positions 0, 1, 7, 255, 8191, 8192,
8193, 32767, 65535, and 131071. They cover the real head-64/factor-32 case,
default and linear scaling controls, and a derived head-128/factor-8 stress
shape. The derived shape is not another real-checkpoint validation.

**Tolerance policy.** Tiny logits use `atol=5e-5`, `rtol=1e-5`; real logits use
absolute `5e-4`. Both reject nonfinite values and require greedy agreement.
These bounds leave room for CPU reduction-order differences above the measured
errors while remaining below the demonstrated missing-scaling error. The tiny
negative-control test executes Rust with scaling disabled and proves its actual
comparison rejects the result.

Inverse frequencies have zero absolute tolerance and relative tolerance of
eight float32 epsilons. RoPE tables use a per-frequency phase-error bound: a
one-ULP frequency difference accumulates at large positions, so a flat `1e-6`
table tolerance is not portable. Sine/cosine errors are bounded by the actual
phase difference plus `4e-7`; the total bound is capped at `0.02`. Rotation
bounds account for input magnitudes and float32 arithmetic. This permits known
phase rounding differences, not arbitrary drift. The independent analytic
frequency tests also exercise both cutoff boundaries and the transition band.

All reference manifests pin source-file hashes, package versions, model
identity, configuration, token inputs, and checkpoint SHA256. Rust hashes the
actual checkpoint bytes and checks the manifest; the real test also requires
the pinned official weight digest. Regeneration compares metadata and inputs
exactly, logits numerically, and rotary outputs with the phase-aware policy.
Six Python safeguard tests ensure metadata drift, lost cases, zero frequencies,
nonfinite values, and corrupted tables/rotations cannot pass that checker.

**Run the ordinary tests.** These consume the committed small fixtures and
the existing synthetic BF16 weights; Python is unnecessary:

```bash
cargo test --release --test rope_reference_tests --test transformers_reference_tests
```

**Regenerate the small fixtures.** Use Python 3.11 and an isolated environment:

```bash
python3.11 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r scripts/requirements-reference.txt
.venv-reference/bin/python scripts/gen_llama3_reference.py --check
.venv-reference/bin/python -m unittest discover -s scripts -p 'test_llama3_reference.py'
```

On Linux, install the CPU PyTorch wheel before the requirements file to avoid
unneeded CUDA packages:

```bash
.venv-reference/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.venv-reference/bin/python -m pip install -r scripts/requirements-reference.txt
```

Omit `--check` only when intentionally regenerating and reviewing the fixture
changes. CI runs this pinned CPU oracle separately; the normal ARM64/x86 Rust
jobs consume the same committed fixtures. It never downloads the real weights.

**Run full real-checkpoint parity.** Downloading uses the pinned public snapshot
and checks its checksum before reference inference. It downloads approximately
2.5 GB; float32 inference needs several additional GB of RAM. Run the Python
oracle and Rust consumer sequentially to avoid holding two models in memory:

```bash
.venv-reference/bin/python scripts/gen_llama3_reference.py --download
LLAMA3_CHECKPOINT=models/llama3-reference/checkpoint/model.safetensors \
LLAMA3_REFERENCE_DIR=models/llama3-reference/reference \
cargo test --release --test transformers_reference_tests -- --ignored --nocapture
```

For an existing copy of that exact checkpoint:

```bash
.venv-reference/bin/python scripts/gen_llama3_reference.py \
  --checkpoint-dir models/llama3-reference/checkpoint
```

The checkpoint and the larger real-model logits stay under ignored `models/`.
The generator accepts only the pinned weight digest and config; changing the
model is a deliberate reference-version change. It also records hashes of the
local tokenizer/config files used to construct the real input.

The added config regression fix rejects nonfinite theta, epsilon, and scaling
parameters, including finite JSON numbers that overflow when converted to
float32. Parsed and manually constructed configs share the same validation.
No inference arithmetic needed to change for the measured checkpoint parity.
