#!/usr/bin/env python3
"""Pinned Transformers oracle, independent of the Rust RoPE/forward code.

Install requirements-reference.txt in a separate venv. With no arguments this
uses the existing tiny BF16 weights with Llama 3 scaling and writes small CI
fixtures. --check regenerates them in memory and checks numeric reproducibility.
--download verifies a public copy of the official Llama 3.2 1B weights, then
produces a local full-checkpoint reference. Large files remain under models/.
No model Python code or pickle checkpoints are downloaded/executed.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SMALL_OUTPUT = FIXTURES / "llama3_reference"
MODEL_ID = "unsloth/Llama-3.2-1B"
MODEL_REVISION = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
OFFICIAL_ID = "meta-llama/Llama-3.2-1B"
OFFICIAL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
WEIGHT_SHA256 = "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a"
TRANSFORMERS_COMMIT = "51f94ea06d19a6308c61bbb4dc97c40aabd12bad"
PINS = {
    "torch": "2.7.1", "transformers": "4.52.4", "numpy": "2.2.6",
    "safetensors": "0.5.3", "huggingface-hub": "0.32.4", "tokenizers": "0.21.2",
}
POSITIONS = [0, 1, 7, 255, 8191, 8192, 8193, 32767, 65535, 131071]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup(threads):
    for package, expected in PINS.items():
        actual = importlib.metadata.version(package)
        # Linux's official CPU wheel appends +cpu to the same release version.
        if actual not in {expected, expected + "+cpu"}:
            raise ValueError(f"{package}=={expected} required, found {actual}; "
                             "install scripts/requirements-reference.txt")
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def provenance():
    import transformers.modeling_rope_utils as rope
    import transformers.models.llama.modeling_llama as llama
    return {
        "transformers_version": PINS["transformers"],
        "transformers_commit": TRANSFORMERS_COMMIT,
        "packages": PINS,
        "device": "cpu", "compute_dtype": "float32", "attention": "eager",
        "source_sha256": {"modeling_llama.py": sha256(llama.__file__),
                          "modeling_rope_utils.py": sha256(rope.__file__)},
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "official_model_id": OFFICIAL_ID, "official_revision": OFFICIAL_REVISION,
        "official_weight_sha256": WEIGHT_SHA256,
    }


def rope_reference(config):
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding, apply_rotary_pos_emb,
    )

    cases = []
    for name in ["llama3_2_1b", "default", "linear", "head128_factor8"]:
        raw = copy.deepcopy(config)
        if name == "default":
            raw["rope_scaling"] = None
        elif name == "linear":
            raw["rope_scaling"] = {"rope_type": "linear", "factor": 4.0}
        elif name == "head128_factor8":
            # A derived stress shape, not a claim to have loaded another model.
            raw["head_dim"] = 128
            raw["hidden_size"] = 128 * raw["num_attention_heads"]
            raw["rope_scaling"]["factor"] = 8.0
        hf_config = LlamaConfig.from_dict(raw)
        rotary = LlamaRotaryEmbedding(hf_config, device="cpu")
        dim = hf_config.head_dim
        # Dyadic inputs are exact across platforms; only oracle outputs may vary.
        q = torch.arange(dim, dtype=torch.float32) / 64.0 - 0.75
        k = (torch.arange(dim, dtype=torch.float32) % 13 - 6.0) / 8.0
        position_ids = torch.tensor([POSITIONS], dtype=torch.long)
        with torch.inference_mode():
            cos, sin = rotary(torch.zeros(1, len(POSITIONS), dim), position_ids)
            qr, kr = apply_rotary_pos_emb(
                q.reshape(1, 1, 1, dim), k.reshape(1, 1, 1, dim), cos, sin,
            )
        cases.append({
            "name": name, "config": raw, "q": q.tolist(), "k": k.tolist(),
            "inv_freq": rotary.inv_freq.tolist(),
            "positions": [{
                "position": position,
                "cos": cos[0, i, :dim // 2].tolist(),
                "sin": sin[0, i, :dim // 2].tolist(),
                "q_rotated": qr[0, 0, i].tolist(),
                "k_rotated": kr[0, 0, i].tolist(),
            } for i, position in enumerate(POSITIONS)],
        })
    return {"schema_version": 1, "provenance": provenance(), "cases": cases}


def forward_reference(model, config, tokens, checkpoint):
    import numpy as np
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    model.eval()
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits[0].float().numpy().copy()
        # Also check the reference's cache path, independently of the Rust test.
        cached = []
        past = None
        for token in tokens:
            step = model(torch.tensor([[token]]), past_key_values=past, use_cache=True)
            past = step.past_key_values
            cached.append(step.logits[0, -1].float().numpy().copy())
        cached = np.stack(cached)
        delta = float(np.max(np.abs(expected - cached)))
        if not np.allclose(expected, cached, atol=2e-4, rtol=2e-5):
            raise ValueError(f"reference prefill/cached decode disagree: max delta {delta}")
        if not np.array_equal(expected.argmax(-1), cached.argmax(-1)):
            raise ValueError("reference prefill/cached greedy choices differ")

        # Demonstrate that the fixture detects losing scaling, not just shapes.
        saved = model.model.rotary_emb
        unscaled = copy.deepcopy(config)
        unscaled["rope_scaling"] = None
        model.model.rotary_emb = LlamaRotaryEmbedding(LlamaConfig.from_dict(unscaled))
        control = model(ids, use_cache=False).logits[0].float().numpy()
        negative_gap = float(np.max(np.abs(expected - control)))
        model.model.rotary_emb = saved
    if not np.isfinite(expected).all() or negative_gap <= 5e-4:
        raise ValueError(f"fixture is nonfinite or fails to detect missing scaling: {negative_gap}")
    manifest = {
        "schema_version": 1, "provenance": provenance(), "config": config,
        "tokens": tokens, "logits_file": "logits.bin", "logit_rows": len(tokens),
        "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "reference_self_check_max_abs": delta,
        "negative_control_max_abs": negative_gap,
        "greedy_tokens": expected.argmax(-1).tolist(),
    }
    print(f"Forward oracle: {len(tokens)} positions, cache delta={delta:.3g}, "
          f"unscaled delta={negative_gap:.3g}", flush=True)
    return manifest, expected.astype("<f4").tobytes()


def tiny_reference(real_config):
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    raw = json.loads((FIXTURES / "config.json").read_text())
    raw.update(rope_theta=real_config["rope_theta"],
               rope_scaling=copy.deepcopy(real_config["rope_scaling"]),
               max_position_embeddings=real_config["max_position_embeddings"],
               tie_word_embeddings=False)
    config = LlamaConfig.from_dict(raw)
    config._attn_implementation = "eager"
    checkpoint = FIXTURES / "tiny_llama.safetensors"
    model = LlamaForCausalLM(config)
    # Copy the already-rounded BF16 values into float32 parameters, as Rust does.
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    meta = dict(line.split("=", 1) for line in
                (FIXTURES / "tiny_llama_meta.txt").read_text().splitlines() if "=" in line)
    tokens = [int(x) for x in meta["tokens"].split(",")]
    manifest, logits = forward_reference(model, raw, tokens, checkpoint)
    manifest["kind"] = "synthetic_existing_weights_with_llama3_scaling"
    manifest["checkpoint"] = "tests/fixtures/tiny_llama.safetensors"
    return manifest, logits


def download():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, token=False,
        local_dir=ROOT / "models" / "llama3-reference" / "checkpoint",
        allow_patterns=["model.safetensors", "config.json", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json",
                        "generation_config.json", "LICENSE", "USE_POLICY.md"],
        max_workers=3,
    ))


def checkpoint_reference(directory):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = directory / "model.safetensors"
    if sha256(checkpoint) != WEIGHT_SHA256:
        raise ValueError("checkpoint SHA256 does not match the pinned Llama 3.2 1B release")
    raw = json.loads((directory / "config.json").read_text())
    pinned = json.loads((SMALL_OUTPUT / "llama3_config.json").read_text())
    if raw != pinned:
        raise ValueError("config.json differs from the pinned checkpoint configuration")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    prompt = ("A library stores books on numbered shelves. Each shelf holds sixteen books. "
              "When two readers borrow the same collection, they can share the catalog "
              "without copying every entry. Explain how this saves memory and why a "
              "private copy is needed when one reader changes an entry.")
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    # Select argmax directly from the independent model. generate() can merge
    # checkpoint generation defaults (this checkpoint defaults to sampling),
    # which must never turn a deterministic reference into a sampled sequence.
    sequence = ids[0].tolist()
    with torch.inference_mode():
        step = model(ids, attention_mask=torch.ones_like(ids), use_cache=True)
        for index in range(8):
            token = int(step.logits[0, -1].argmax())
            sequence.append(token)
            if token == raw["eos_token_id"] or index == 7:
                break
            step = model(torch.tensor([[token]]), past_key_values=step.past_key_values,
                         use_cache=True)
    manifest, logits = forward_reference(model, raw, sequence, checkpoint)
    if any(sequence[i] != manifest["greedy_tokens"][i-1]
           for i in range(ids.shape[1], len(sequence))):
        raise ValueError("cached continuation is not greedy under full-sequence reference logits")
    manifest.update(kind="real_llama3_2_1b", prompt=prompt, prompt_length=ids.shape[1],
                    continuation_tokens=sequence[ids.shape[1]:])
    manifest["input_sha256"] = {
        name: sha256(directory / name) for name in
        ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]
        if (directory / name).is_file()
    }
    return manifest, logits


def check_exact(actual, expected, path="root"):
    """Configurations, provenance and inputs must not drift, even as floats."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{path}: keys differ")
        for key in actual:
            check_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f"{path}: lengths differ")
        for i, (a, e) in enumerate(zip(actual, expected)):
            check_exact(a, e, f"{path}[{i}]")
    elif actual != expected:
        raise ValueError(f"{path}: {actual!r} != {expected!r}")


def check_manifest(actual, expected):
    import math
    measurements = {"reference_self_check_max_abs", "negative_control_max_abs"}
    check_exact({k: v for k, v in actual.items() if k not in measurements},
                {k: v for k, v in expected.items() if k not in measurements}, "manifest")
    for key in measurements:
        if not math.isclose(actual[key], expected[key], abs_tol=1e-4, rel_tol=1e-5):
            raise ValueError(f"manifest.{key} drifted")


def check_rope(actual, expected):
    """Check last-bit CPU differences using their phase error, not a flat epsilon."""
    import numpy as np
    check_exact({k: v for k, v in actual.items() if k != "cases"},
                {k: v for k, v in expected.items() if k != "cases"}, "rope")
    if len(actual["cases"]) != len(expected["cases"]):
        raise ValueError("rope: case count drifted")

    def finite_vector(value, length):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (length,) or not np.isfinite(vector).all():
            raise ValueError("rope: invalid numeric vector")
        return vector

    for a, e in zip(actual["cases"], expected["cases"]):
        outputs = {"inv_freq", "positions"}
        check_exact({k: v for k, v in a.items() if k not in outputs},
                    {k: v for k, v in e.items() if k not in outputs}, a["name"])
        half = a["config"]["head_dim"] // 2
        af = finite_vector(a["inv_freq"], half)
        ef = finite_vector(e["inv_freq"], half)
        if np.any(ef <= 0) or not np.allclose(af, ef, atol=0, rtol=8*np.finfo(np.float32).eps):
            raise ValueError(f"{a['name']}: inverse frequencies drifted")
        if len(a["positions"]) != len(e["positions"]):
            raise ValueError("rope: position count drifted")
        for ap, ep in zip(a["positions"], e["positions"]):
            numeric = {"cos", "sin", "q_rotated", "k_rotated"}
            check_exact({k: v for k, v in ap.items() if k not in numeric},
                        {k: v for k, v in ep.items() if k not in numeric}, a["name"])
            pos = np.float32(ap["position"])
            rounded = np.abs((pos * af.astype(np.float32)).astype(np.float64)
                             - (pos * ef.astype(np.float32)).astype(np.float64))
            bound = np.maximum(float(pos) * np.abs(af-ef), rounded) + 4e-7
            if np.any(bound > 0.02):
                raise ValueError("rope: excessive phase drift")
            for key in ["cos", "sin", "q_rotated", "k_rotated"]:
                tolerance = bound
                if key.endswith("_rotated"):
                    inp = np.abs(np.asarray(a[key[0]], dtype=np.float64))
                    magnitude = inp[:half] + inp[half:]
                    tolerance = np.tile(magnitude * (bound + 4*np.finfo(np.float32).eps) + 1e-7, 2)
                av = finite_vector(ap[key], len(tolerance))
                ev = finite_vector(ep[key], len(tolerance))
                if np.any(np.abs(av-ev) > tolerance):
                    raise ValueError(f"{a['name']} position {pos} {key}: phase-aware bound exceeded")


def emit(output, manifest, logits, rope, check):
    import numpy as np
    documents = {"manifest.json": manifest}
    if rope is not None:
        documents["rope.json"] = rope
    if check:
        check_manifest(manifest, json.loads((output / "manifest.json").read_text()))
        if rope is not None:
            check_rope(rope, json.loads((output / "rope.json").read_text()))
        actual = np.frombuffer(logits, dtype="<f4")
        expected = np.frombuffer((output / "logits.bin").read_bytes(), dtype="<f4")
        if actual.shape != expected.shape or not np.isfinite(expected).all():
            raise ValueError("committed logits have an invalid shape or nonfinite values")
        if not np.allclose(actual, expected, atol=1e-4, rtol=1e-5):
            raise ValueError(f"logits drift: {float(np.max(np.abs(actual-expected)))}")
        vocab = manifest["config"]["vocab_size"]
        if not np.array_equal(actual.reshape(-1, vocab).argmax(-1),
                              expected.reshape(-1, vocab).argmax(-1)):
            raise ValueError("committed greedy choices drifted")
        print(f"Reference fixtures verified: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, document in documents.items():
            (output / name).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        (output / "logits.bin").write_bytes(logits)
        print(f"Reference fixtures written: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--download", action="store_true", help="download pinned real weights (~2.5 GB)")
    source.add_argument("--checkpoint-dir", type=Path, help="use already-downloaded pinned weights")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="verify instead of overwriting")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    setup(args.threads)
    directory = download() if args.download else args.checkpoint_dir
    if directory is not None:
        manifest, logits = checkpoint_reference(directory)
        output = args.output or ROOT / "models" / "llama3-reference" / "reference"
        emit(output, manifest, logits, None, args.check)
    else:
        config = json.loads((SMALL_OUTPUT / "llama3_config.json").read_text())
        manifest, logits = tiny_reference(config)
        emit(args.output or SMALL_OUTPUT, manifest, logits, rope_reference(config), args.check)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Reference generation failed: {error}", file=sys.stderr)
        sys.exit(1)
