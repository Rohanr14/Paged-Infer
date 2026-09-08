#!/usr/bin/env python3
"""Generate the synthetic tokenizer fixture under tests/fixtures/tokenizer/.

A tiny tokenizer with the *structure* of a Llama SentencePiece-BPE tokenizer as
`tokenizers` serialises it, so the text layer can be tested in CI without
shipping a real model's vocabulary:

  * `<unk>`, `<s>`, `</s>` special tokens, with a TemplateProcessing
    post-processor that prepends `<s>` — so `encode(text, True)` adds BOS.
  * 256 byte-fallback tokens `<0x00>`..`<0xFF>`, so any character outside the
    vocabulary round-trips through bytes, and a multi-byte character arriving
    one token at a time decodes to replacement characters until complete.
  * The `▁` metaspace convention: a prepended `▁`, spaces replaced by `▁`, and
    the standard Replace / ByteFallback / Fuse / Strip decoder chain.
  * A few dozen whole-word pieces, with merges that build each from characters.

Alongside it, a `tokenizer_config.json` carrying TinyLlama-1.1B-Chat's exact
chat template and special-token strings, and a `generation_config.json` with
the ids, so the profile loader's fallback chain is exercised.

Stdlib only. Usage:  python3 scripts/gen_tokenizer_fixture.py
"""

import json
import os

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "tests", "fixtures", "tokenizer"
)

# TinyLlama-1.1B-Chat-v1.0's chat_template, verbatim.
TINYLLAMA_CHAT_TEMPLATE = (
    "{% for message in messages %}\n{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n"
    "{% endif %}\n{% endfor %}"
)

WORDS = [
    "the", "cat", "sat", "on", "mat", "hello", "world", "Hello", "you", "are",
    "a", "bot", "user", "assistant", "system", "How", "can", "I", "help",
    "today", "is", "an", "and", "to", "of", "it", "in", "be", "brief", "You",
]

CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?:;'\"-()[]<>|/"
)


def build():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    vocab["▁"] = len(vocab)
    for c in CHARS:
        vocab[c] = len(vocab)

    merges = []
    seen = set()
    for word in WORDS:
        piece = "▁" + word
        # Build the piece left to right: (▁, t) -> ▁t, (▁t, h) -> ▁th, ...
        # Every intermediate is itself a vocabulary entry, as BPE requires.
        left = "▁"
        for ch in word:
            merged = left + ch
            if merged not in vocab:
                vocab[merged] = len(vocab)
            merge = f"{left} {ch}"
            if merge not in seen:
                seen.add(merge)
                merges.append(merge)
            left = merged

    added_tokens = [
        {"id": 0, "content": "<unk>", "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True},
        {"id": 1, "content": "<s>", "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True},
        {"id": 2, "content": "</s>", "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True},
    ]
    # The chat markers are ordinary (non-special) added tokens: one id each,
    # kept in decoded text, exactly like a template marker a model was trained
    # to emit.
    for marker in ["<|system|>", "<|user|>", "<|assistant|>"]:
        vocab[marker] = len(vocab)
        added_tokens.append(
            {"id": vocab[marker], "content": marker, "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False,
             "special": False}
        )

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
            ],
        },
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
            },
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }

    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "model_max_length": 2048,
        "tokenizer_class": "LlamaTokenizer",
        "chat_template": TINYLLAMA_CHAT_TEMPLATE,
    }
    generation_config = {"bos_token_id": 1, "eos_token_id": 2}
    return tokenizer, tokenizer_config, generation_config


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    tokenizer, tokenizer_config, generation_config = build()
    for name, doc in [
        ("tokenizer.json", tokenizer),
        ("tokenizer_config.json", tokenizer_config),
        ("generation_config.json", generation_config),
    ]:
        with open(os.path.join(FIXTURE_DIR, name), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
            f.write("\n")
    print(f"wrote tokenizer fixture to {os.path.normpath(FIXTURE_DIR)}")
    print(f"  vocab size : {len(tokenizer['model']['vocab'])}")
    print(f"  merges     : {len(tokenizer['model']['merges'])}")


if __name__ == "__main__":
    main()
