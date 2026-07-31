#!/usr/bin/env python3
"""Generate the golden-parity fixture used by tests/golden_parity_tests.rs.

Emits three files under tests/fixtures/:

  tiny_llama.safetensors  -- a tiny synthetic Llama checkpoint in the exact
                             HuggingFace tensor layout (bf16, same key names as
                             a real `model.safetensors`), so `ModelLoader` reads
                             it through the same code path as TinyLlama.
  tiny_llama_golden.bin   -- reference logits, (seq_len x vocab_size) f32 LE.
  tiny_llama_meta.txt     -- config + token ids, `key=value` lines.

The reference forward pass below is an independent NumPy transcription of
`transformers/models/llama/modeling_llama.py`. It is deliberately written
against the HuggingFace semantics rather than against the Rust code, so that
agreement between the two is evidence and not a tautology. In particular it
uses the NeoX/`rotate_half` rotary convention, which is the one HF checkpoints
are permuted for at conversion time.

Weights are rounded to bf16 *before* the reference runs, so the only remaining
difference against the engine is floating-point summation order.

Usage:  python3 scripts/gen_golden_fixture.py
"""

import json
import os
import struct

import numpy as np

# ── fixture configuration ────────────────────────────────────────────────────

HIDDEN = 96
LAYERS = 2
HEADS = 4
KV_HEADS = 2  # exercises grouped-query attention (kv_group = 2)
INTERMEDIATE = 192
VOCAB = 112
EPS = 1e-5
THETA = 10000.0
# 40 tokens spans three 16-token KV blocks, so the paged block table is
# exercised across block boundaries rather than living in a single block.
SEQ_LEN = 40
HEAD_DIM = HIDDEN // HEADS

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")


# ── bf16 helpers ─────────────────────────────────────────────────────────────


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round f32 -> bf16 with round-half-to-even, returning raw uint16 bits."""
    u = x.astype(np.float32).view(np.uint32)
    # Add 0x7FFF plus the lsb of the surviving mantissa: standard RNE for a
    # 16-bit truncation.
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Widen bf16 bits back to f32 (exact — bf16 is a truncated f32)."""
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def quantize_bf16(x: np.ndarray) -> np.ndarray:
    """Snap an f32 array onto the bf16 grid, staying in f32."""
    return bf16_bits_to_f32(f32_to_bf16_bits(x))


# ── safetensors writer (the format is small enough not to need a dependency) ──


def write_safetensors(path: str, tensors: dict) -> None:
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        raw = f32_to_bf16_bits(arr).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors requires the data buffer to start 8-byte aligned.
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


# ── reference forward pass (NumPy transcription of HF modeling_llama) ─────────


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(variance + eps))) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rope_tables(seq_len: int, head_dim: int, theta: float):
    """HF LlamaRotaryEmbedding: inv_freq over even indices, then cat(f, f)."""
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.arange(seq_len, dtype=np.float64)[:, None]
    freqs = pos * inv_freq[None, :]  # (T, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (T, head_dim)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def rotate_interleaved(x: np.ndarray) -> np.ndarray:
    """The GPT-J / original-Meta convention, kept only to show it differs."""
    out = np.empty_like(x)
    out[..., 0::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, interleaved: bool):
    """x: (T, n_heads, head_dim); cos/sin: (T, head_dim)."""
    if interleaved:
        # Interleaved pairs (2i, 2i+1) consume angle theta^(-2i/d) at index 2i.
        half = cos.shape[-1] // 2
        c = np.repeat(cos[:, :half], 2, axis=-1)
        s = np.repeat(sin[:, :half], 2, axis=-1)
        return x * c[:, None, :] + rotate_interleaved(x) * s[:, None, :]
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def reference_forward(w: dict, tokens: np.ndarray, interleaved: bool = False):
    """Full-sequence causal forward; returns logits of shape (T, VOCAB)."""
    seq_len = len(tokens)
    kv_group = HEADS // KV_HEADS
    cos, sin = rope_tables(seq_len, HEAD_DIM, THETA)

    h = w["model.embed_tokens.weight"][tokens]  # (T, HIDDEN)

    causal = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)

    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        hn = rms_norm(h, w[f"{p}.input_layernorm.weight"], EPS)

        # HF stores projections as (out_features, in_features), so x @ W.T.
        q = hn @ w[f"{p}.self_attn.q_proj.weight"].T
        k = hn @ w[f"{p}.self_attn.k_proj.weight"].T
        v = hn @ w[f"{p}.self_attn.v_proj.weight"].T

        q = q.reshape(seq_len, HEADS, HEAD_DIM)
        k = k.reshape(seq_len, KV_HEADS, HEAD_DIM)
        v = v.reshape(seq_len, KV_HEADS, HEAD_DIM)

        q = apply_rope(q, cos, sin, interleaved)
        k = apply_rope(k, cos, sin, interleaved)

        # repeat_kv: query head i reads kv head i // kv_group.
        k = np.repeat(k, kv_group, axis=1)
        v = np.repeat(v, kv_group, axis=1)

        # (H, T, D)
        qh, kh, vh = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
        scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(HEAD_DIM, dtype=np.float32)
        scores = scores + causal[None, :, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        attn = probs @ vh  # (H, T, D)

        attn = attn.transpose(1, 0, 2).reshape(seq_len, HIDDEN)
        h = h + attn @ w[f"{p}.self_attn.o_proj.weight"].T

        hn = rms_norm(h, w[f"{p}.post_attention_layernorm.weight"], EPS)
        gate = hn @ w[f"{p}.mlp.gate_proj.weight"].T
        up = hn @ w[f"{p}.mlp.up_proj.weight"].T
        h = h + (silu(gate) * up) @ w[f"{p}.mlp.down_proj.weight"].T

    h = rms_norm(h, w["model.norm.weight"], EPS)
    return (h @ w["lm_head.weight"].T).astype(np.float32)


# ── entry point ──────────────────────────────────────────────────────────────


def build_weights(rng) -> dict:
    def w(*shape, scale=0.05):
        return quantize_bf16(rng.normal(0.0, scale, size=shape).astype(np.float32))

    def norm(*shape):
        # Norm weights sit near 1.0 in real checkpoints.
        return quantize_bf16(
            (1.0 + rng.normal(0.0, 0.1, size=shape)).astype(np.float32)
        )

    tensors = {"model.embed_tokens.weight": w(VOCAB, HIDDEN, scale=0.2)}
    kv_dim = KV_HEADS * HEAD_DIM
    for layer in range(LAYERS):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = norm(HIDDEN)
        tensors[f"{p}.self_attn.q_proj.weight"] = w(HIDDEN, HIDDEN)
        tensors[f"{p}.self_attn.k_proj.weight"] = w(kv_dim, HIDDEN)
        tensors[f"{p}.self_attn.v_proj.weight"] = w(kv_dim, HIDDEN)
        tensors[f"{p}.self_attn.o_proj.weight"] = w(HIDDEN, HIDDEN)
        tensors[f"{p}.post_attention_layernorm.weight"] = norm(HIDDEN)
        tensors[f"{p}.mlp.gate_proj.weight"] = w(INTERMEDIATE, HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = w(INTERMEDIATE, HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = w(HIDDEN, INTERMEDIATE)
    tensors["model.norm.weight"] = norm(HIDDEN)
    tensors["lm_head.weight"] = w(VOCAB, HIDDEN, scale=0.2)
    return tensors


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    rng = np.random.default_rng(20240517)

    tensors = build_weights(rng)
    tokens = rng.integers(0, VOCAB, size=SEQ_LEN, dtype=np.uint32)

    logits = reference_forward(tensors, tokens, interleaved=False)
    interleaved = reference_forward(tensors, tokens, interleaved=True)

    # The fixture is only useful if it can tell the two rotary conventions
    # apart; assert that up front so a future regression can't pass silently.
    spread = np.abs(logits - interleaved).max()
    assert spread > 1e-2, f"fixture cannot discriminate rope conventions ({spread})"

    write_safetensors(os.path.join(FIXTURE_DIR, "tiny_llama.safetensors"), tensors)

    with open(os.path.join(FIXTURE_DIR, "tiny_llama_golden.bin"), "wb") as f:
        f.write(logits.astype("<f4").tobytes())

    meta = [
        f"hidden_size={HIDDEN}",
        f"num_hidden_layers={LAYERS}",
        f"num_attention_heads={HEADS}",
        f"num_key_value_heads={KV_HEADS}",
        f"intermediate_size={INTERMEDIATE}",
        f"vocab_size={VOCAB}",
        f"rms_norm_eps={EPS}",
        f"rope_theta={THETA}",
        f"seq_len={SEQ_LEN}",
        "tokens=" + ",".join(str(int(t)) for t in tokens),
    ]
    with open(os.path.join(FIXTURE_DIR, "tiny_llama_meta.txt"), "w") as f:
        f.write("\n".join(meta) + "\n")

    print(f"wrote fixtures to {os.path.normpath(FIXTURE_DIR)}")
    print(f"  tokens          : {list(int(t) for t in tokens)}")
    print(f"  logits          : {logits.shape}, range [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  argmax per pos  : {list(int(i) for i in logits.argmax(axis=-1))}")
    print(f"  rope convention discriminated by max|delta| = {spread:.4f}")


if __name__ == "__main__":
    main()
