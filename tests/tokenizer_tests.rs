//! The text layer, against a real `tokenizers` tokenizer with byte fallback.
//!
//! The fixture under `tests/fixtures/tokenizer/` has the structure of a Llama
//! SentencePiece-BPE tokenizer (see `scripts/gen_tokenizer_fixture.py`):
//! `<s>`/`</s>` specials added by a post-processor, 256 byte-fallback tokens,
//! the `▁` whitespace convention and the standard decoder chain. That is enough
//! to exercise everything that went wrong with real TinyLlama:
//!
//! * text prompts reached the model with two BOS tokens;
//! * an emoji split across four byte tokens streamed as `���😀`;
//! * chat requests ignored the checkpoint's template.

use std::path::PathBuf;

use paged_infer::detokenizer::IncrementalDetokenizer;
use paged_infer::engine::{Engine, EngineConfig};
use paged_infer::model::{LlamaConfig, ModelLoader};
use paged_infer::profile::{render_template, ChatMessage, ModelProfile};
use tokenizers::Tokenizer;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn tokenizer_path() -> PathBuf {
    fixtures().join("tokenizer/tokenizer.json")
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_file(tokenizer_path()).expect("fixture tokenizer loads")
}

fn id(t: &Tokenizer, piece: &str) -> u32 {
    t.token_to_id(piece)
        .unwrap_or_else(|| panic!("{piece:?} is not in the fixture vocabulary"))
}

fn byte_ids(t: &Tokenizer, text: &str) -> Vec<u32> {
    text.bytes()
        .map(|b| id(t, &format!("<0x{b:02X}>")))
        .collect()
}

fn msgs(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
    pairs
        .iter()
        .map(|(r, c)| ChatMessage {
            role: r.to_string(),
            content: c.to_string(),
        })
        .collect()
}

/// The fixture model's config plus the fixture tokenizer: a complete profile.
/// The "model path" points into the tokenizer fixture directory so that its
/// `generation_config.json` is found beside it, as in a real checkpoint dir.
fn profile() -> ModelProfile {
    let config = LlamaConfig::from_hf_config(fixtures().join("config.json")).unwrap();
    ModelProfile::from_parts(
        config,
        &fixtures().join("tokenizer/model.safetensors"),
        Some(&tokenizer_path()),
    )
    .unwrap()
}

/// A word-level tokenizer whose ids fit the fixture *model's* 112-token
/// vocabulary, so text can be pushed through the engine. `with_bos` controls
/// whether its post-processor adds `<s>` itself.
fn small_tokenizer(with_bos: bool) -> Tokenizer {
    let post = if with_bos {
        serde_json::json!({
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}
        })
    } else {
        serde_json::Value::Null
    };
    let json = serde_json::json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": post,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "the": 5, "cat": 6}, "unk_token": "<unk>"}
    });
    Tokenizer::from_bytes(json.to_string().as_bytes()).expect("small tokenizer parses")
}

// ── the fixture behaves like a Llama tokenizer ───────────────────────────────

#[test]
fn test_fixture_tokenizer_adds_bos_uses_metaspace_and_falls_back_to_bytes() {
    let t = tokenizer();
    let enc = t.encode("hello world", true).unwrap();
    assert_eq!(
        enc.get_ids(),
        &[id(&t, "<s>"), id(&t, "▁hello"), id(&t, "▁world")]
    );
    let enc = t.encode("hello world", false).unwrap();
    assert_eq!(enc.get_ids(), &[id(&t, "▁hello"), id(&t, "▁world")]);

    // An emoji is four byte tokens after the metaspace marker, and decodes back.
    let enc = t.encode("😀", false).unwrap();
    let mut expected = vec![id(&t, "▁")];
    expected.extend(byte_ids(&t, "😀"));
    assert_eq!(enc.get_ids(), &expected[..]);
    assert_eq!(t.decode(enc.get_ids(), true).unwrap(), "😀");
    assert_eq!(
        t.decode(
            t.encode("the cat sat on the mat", true).unwrap().get_ids(),
            true
        )
        .unwrap(),
        "the cat sat on the mat"
    );
}

// ── streaming text ───────────────────────────────────────────────────────────

/// Every way to split `tokens` into consecutive non-empty pushes.
fn partitions(n: usize) -> Vec<Vec<usize>> {
    // Cut points as a bitmask over the n-1 gaps.
    let mut out = Vec::new();
    for mask in 0..(1u32 << (n - 1)) {
        let mut sizes = Vec::new();
        let mut run = 1;
        for gap in 0..n - 1 {
            if mask & (1 << gap) != 0 {
                sizes.push(run);
                run = 1;
            } else {
                run += 1;
            }
        }
        sizes.push(run);
        out.push(sizes);
    }
    out
}

/// Stream `tokens` in the given chunk sizes; return the concatenated deltas
/// (with the final flush) and the deltas themselves.
fn stream(t: &Tokenizer, tokens: &[u32], sizes: &[usize]) -> (String, Vec<String>) {
    let mut detok = IncrementalDetokenizer::new();
    let mut deltas = Vec::new();
    let mut start = 0;
    for &size in sizes {
        deltas.push(detok.push(t, &tokens[start..start + size]));
        start += size;
    }
    assert_eq!(start, tokens.len());
    deltas.push(detok.finish());
    (deltas.concat(), deltas)
}

#[test]
fn test_every_partition_of_multibyte_text_streams_to_the_buffered_decode() {
    let t = tokenizer();
    let cases: Vec<(&str, Vec<u32>)> = vec![
        // emoji: 4 bytes
        ("😀", t.encode("😀", false).unwrap().get_ids().to_vec()),
        // CJK: two 3-byte characters
        ("你好", t.encode("你好", false).unwrap().get_ids().to_vec()),
        // emoji between words, with the whitespace that SentencePiece folds
        // into the following token
        (
            "the 😀 cat",
            t.encode("the 😀 cat", false).unwrap().get_ids().to_vec(),
        ),
        // a legitimate replacement character in the text itself
        (
            "a\u{FFFD}b",
            t.encode("a\u{FFFD}b", false).unwrap().get_ids().to_vec(),
        ),
        // a legitimate replacement character at the very end
        (
            "hello \u{FFFD}",
            t.encode("hello \u{FFFD}", false)
                .unwrap()
                .get_ids()
                .to_vec(),
        ),
        // two spaces: a bare metaspace token between words
        (
            "the  cat",
            t.encode("the  cat", false).unwrap().get_ids().to_vec(),
        ),
        // newline through byte fallback, then a word without its space
        (
            "hello\nworld",
            t.encode("hello\nworld", false).unwrap().get_ids().to_vec(),
        ),
    ];
    for (text, tokens) in cases {
        let full = t.decode(&tokens, true).unwrap();
        assert_eq!(full, text, "fixture should round-trip {text:?}");
        assert!(tokens.len() <= 12, "keep the partition count sane");
        for sizes in partitions(tokens.len()) {
            let (joined, deltas) = stream(&t, &tokens, &sizes);
            assert_eq!(
                joined, full,
                "{text:?} streamed in chunks {sizes:?} gave {deltas:?}"
            );
            // Nothing sent is ever a fragment: no delta except the very last
            // can end in a replacement character unless the text has one there.
            for d in &deltas[..deltas.len() - 1] {
                if d.ends_with('\u{FFFD}') {
                    assert!(
                        text.contains('\u{FFFD}'),
                        "{text:?} in chunks {sizes:?} leaked a fragment: {deltas:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_an_emoji_arriving_one_token_at_a_time_is_held_until_complete() {
    // The review's exact symptom, on the fixture: one byte per push produced
    // `['�', '�', '�', '😀']` and displayed `���😀`.
    let t = tokenizer();
    let tokens = t.encode("😀", false).unwrap().get_ids().to_vec();
    let mut detok = IncrementalDetokenizer::new();
    let deltas: Vec<String> = tokens.iter().map(|&tok| detok.push(&t, &[tok])).collect();
    assert_eq!(deltas, vec!["", "", "", "", "😀"], "{deltas:?}");
    assert_eq!(detok.finish(), "");
    assert_eq!(detok.text(), "😀");
}

#[test]
fn test_a_final_replacement_character_is_flushed_not_lost() {
    let t = tokenizer();
    let tokens = t.encode("a\u{FFFD}", false).unwrap().get_ids().to_vec();
    let mut detok = IncrementalDetokenizer::new();
    let streamed = detok.push(&t, &tokens);
    assert_eq!(
        streamed, "a",
        "the trailing U+FFFD could still be a fragment"
    );
    assert_eq!(detok.finish(), "\u{FFFD}");
    assert_eq!(detok.text(), "a\u{FFFD}");
}

#[test]
fn test_whitespace_bearing_tokens_stream_with_their_space() {
    let t = tokenizer();
    let tokens = t.encode("the cat sat", false).unwrap().get_ids().to_vec();
    let mut detok = IncrementalDetokenizer::new();
    let deltas: Vec<String> = tokens.iter().map(|&tok| detok.push(&t, &[tok])).collect();
    assert_eq!(deltas, vec!["the", " cat", " sat"]);
}

// ── BOS ownership ────────────────────────────────────────────────────────────

#[test]
fn test_text_prompts_carry_exactly_one_bos_and_token_prompts_are_untouched() {
    let p = profile();
    let t = p.tokenizer().unwrap();
    let bos = id(t, "<s>");
    assert_eq!(p.bos_token, Some(bos));

    // The tokenizer adds BOS itself; the profile must not add a second one.
    let ids = p.encode_prompt("hello").unwrap();
    assert_eq!(ids, vec![bos, id(t, "▁hello")]);
    assert_eq!(ids.iter().filter(|&&x| x == bos).count(), 1);

    let engine_config = EngineConfig {
        total_blocks: 16,
        block_size: 8,
        ..p.engine_config()
    };
    assert_eq!(engine_config.bos_token, Some(bos));
    assert_eq!(engine_config.eos_token, id(t, "</s>"));

    // Through the engine's text path as well — with a tokenizer whose ids fit
    // the fixture model's vocabulary, once with a BOS-adding post-processor
    // and once without. Either way the model sees exactly one BOS.
    let bytes = std::fs::read(fixtures().join("tiny_llama.safetensors")).unwrap();
    let loader = ModelLoader::new(&bytes).unwrap();
    for tokenizer_adds_bos in [true, false] {
        let weights = loader.load_weights(&p.config).unwrap();
        let small = small_tokenizer(tokenizer_adds_bos);
        assert_eq!(
            small.encode("hello world", true).unwrap().get_ids(),
            if tokenizer_adds_bos {
                &[1, 3, 4][..]
            } else {
                &[3, 4][..]
            }
        );
        let mut engine =
            Engine::new(weights, p.config.clone(), engine_config.clone()).with_tokenizer(small);
        let text_request = engine.submit("hello world", 2, 1).unwrap();
        let token_request = engine.submit_tokens(vec![3, 4], 2, 1).unwrap();
        let out = engine.run().unwrap();
        let prompt_len = |rid| {
            out.iter()
                .find(|c| c.request_id == rid)
                .unwrap()
                .prompt_tokens
        };
        assert_eq!(prompt_len(text_request), 3, "BOS + two words");
        assert_eq!(prompt_len(token_request), 2, "ids are the complete input");
    }
}

#[test]
fn test_a_tokenizer_without_a_bos_post_processor_still_gets_one_bos() {
    // Same vocabulary, post-processor removed: the profile has to supply BOS.
    let raw = std::fs::read_to_string(tokenizer_path()).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["post_processor"] = serde_json::Value::Null;
    let dir = std::env::temp_dir().join(format!("paged-infer-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), json.to_string()).unwrap();
    std::fs::copy(
        fixtures().join("tokenizer/tokenizer_config.json"),
        dir.join("tokenizer_config.json"),
    )
    .unwrap();

    let config = LlamaConfig::from_hf_config(fixtures().join("config.json")).unwrap();
    let p = ModelProfile::from_parts(
        config,
        &fixtures().join("tiny_llama.safetensors"),
        Some(&dir.join("tokenizer.json")),
    )
    .unwrap();
    let t = p.tokenizer().unwrap();
    assert_eq!(
        t.encode("hello", true).unwrap().get_ids(),
        &[id(t, "▁hello")]
    );
    assert_eq!(
        p.encode_prompt("hello").unwrap(),
        vec![id(t, "<s>"), id(t, "▁hello")]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_profile_resolves_special_tokens_and_context_through_the_fallback_chain() {
    // The fixture config.json declares no special tokens; generation_config
    // and the tokenizer config do.
    let p = profile();
    assert_eq!(p.bos_token, Some(1));
    assert_eq!(p.eos_tokens, vec![2]);
    assert!(p.add_bos_token);
    assert!(p.has_chat_template());
    assert_eq!(p.max_context, None);
    assert_eq!(p.bos_text.as_deref(), Some("<s>"));
    assert_eq!(p.eos_text.as_deref(), Some("</s>"));

    // No generation_config.json beside the model either: the tokenizer's own
    // special-token strings are the last resort, and they resolve.
    let config = LlamaConfig::from_hf_config(fixtures().join("config.json")).unwrap();
    let p = ModelProfile::from_parts(
        config,
        &fixtures().join("tiny_llama.safetensors"),
        Some(&tokenizer_path()),
    )
    .unwrap();
    assert_eq!((p.bos_token, p.eos_tokens.clone()), (Some(1), vec![2]));

    // config.json wins over both when it speaks, and every declared EOS stops.
    let mut config = LlamaConfig::from_hf_config(fixtures().join("config.json")).unwrap();
    config.bos_token_id = Some(7);
    config.eos_token_ids = vec![9];
    config.max_position_embeddings = Some(64);
    let p = ModelProfile::from_parts(
        config,
        &fixtures().join("tokenizer/model.safetensors"),
        Some(&tokenizer_path()),
    )
    .unwrap();
    assert_eq!(p.bos_token, Some(7));
    assert_eq!(p.eos_tokens, vec![9, 2]);
    assert_eq!(p.max_context, Some(64));
    let ec = p.engine_config();
    assert_eq!((ec.eos_token, ec.extra_eos_tokens.clone()), (9, vec![2]));
    assert_eq!(ec.max_context, Some(64));

    // Without a tokenizer, token-id serving still works — provided something
    // declares an EOS. Nothing at all is refused: the model could never stop.
    let config = LlamaConfig::from_hf_config(fixtures().join("config.json")).unwrap();
    let err = ModelProfile::from_parts(
        config.clone(),
        &fixtures().join("tiny_llama.safetensors"),
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("end-of-sequence"), "{err}");
    let mut config = config;
    config.eos_token_ids = vec![2];
    let p =
        ModelProfile::from_parts(config, &fixtures().join("tiny_llama.safetensors"), None).unwrap();
    assert!(p.tokenizer.is_none());
    assert!(p.encode_prompt("x").is_err());
    assert!(!p.has_chat_template());
    assert_eq!(
        p.bos_token, None,
        "nothing declared a BOS, none is invented"
    );
}

// ── chat template ────────────────────────────────────────────────────────────

#[test]
fn test_chat_requests_render_the_checkpoint_template_and_tokenize_without_extra_bos() {
    let p = profile();
    let t = p.tokenizer().unwrap();
    let conversation = msgs(&[("system", "You are a bot."), ("user", "Hello")]);

    let rendered = p.render_chat(&conversation, true).unwrap();
    assert_eq!(
        rendered, "<|system|>\nYou are a bot.</s>\n<|user|>\nHello</s>\n<|assistant|>\n",
        "TinyLlama's template: per-message EOS and an assistant marker"
    );

    // `apply_chat_template` tokenizes the rendered text with
    // add_special_tokens=False. The ids are exactly the tokenizer's own for
    // that string: the markers are single pieces, every EOS is the real </s>
    // id, and there is no BOS because this template does not put one there.
    let ids = p.encode_chat(&conversation).unwrap();
    assert_eq!(
        ids,
        t.encode(rendered.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec()
    );
    assert_eq!(ids[0], id(t, "<|system|>"));
    assert_eq!(ids.iter().filter(|&&x| x == id(t, "</s>")).count(), 2);
    assert!(ids.contains(&id(t, "<|user|>")) && ids.contains(&id(t, "<|assistant|>")));
    assert!(!ids.contains(&id(t, "<s>")));
    assert_ne!(
        ids,
        t.encode(rendered.as_str(), true)
            .unwrap()
            .get_ids()
            .to_vec(),
        "with added specials the tokenizer would prepend a BOS the template did not ask for"
    );

    // The generic `role: content` rendering it replaced is gone.
    assert!(!rendered.contains("user: Hello"));
}

#[test]
fn test_unsupported_roles_and_missing_templates_are_refused() {
    let p = profile();
    let err = p
        .render_chat(&msgs(&[("tool", "x")]), true)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("tool") && err.contains("not supported"),
        "{err}"
    );
    assert!(p.render_chat(&[], true).is_err());

    let mut config = LlamaConfig::from_hf_config(fixtures().join("config.json")).unwrap();
    config.eos_token_ids = vec![2];
    let no_template =
        ModelProfile::from_parts(config, &fixtures().join("tiny_llama.safetensors"), None).unwrap();
    let err = no_template
        .render_chat(&msgs(&[("user", "x")]), true)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no chat_template"), "{err}");
}

#[test]
fn test_templates_with_bos_put_the_bos_in_the_ids_once() {
    // A Llama-2 style template carries `{{ bos_token }}` itself; tokenizing
    // its output without added specials yields exactly one BOS.
    let template = "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]{% else %}{{ m['content'] }}{{ eos_token }}{% endif %}{% endfor %}";
    let rendered =
        render_template(template, &msgs(&[("user", "Hello")]), "<s>", "</s>", true).unwrap();
    assert_eq!(rendered, "<s>[INST] Hello [/INST]");
    let t = tokenizer();
    let ids = t
        .encode(rendered.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert_eq!(ids.iter().filter(|&&x| x == id(&t, "<s>")).count(), 1);
    assert_eq!(ids[0], id(&t, "<s>"));
}
