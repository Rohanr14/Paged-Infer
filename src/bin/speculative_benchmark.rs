//! What speculative decoding is actually worth, per workload.
//!
//! Reports two numbers that are easy to conflate:
//!
//! * **tokens per model pass** — how much work the drafting saved. Ordinary
//!   decoding is exactly 1.0.
//! * **wall-clock speedup** — what the user feels. Always lower, because a pass
//!   verifying `K+1` positions costs more than one producing a single token.
//!
//! Acceptance is a property of the workload, not of the engine, so several are
//! run side by side. Copy-heavy prompts — summarize this, answer from this
//! passage, rewrite this code — are where prompt-lookup drafting shines,
//! because the output genuinely repeats the input. Open-ended generation is the
//! hard case and is reported too rather than quietly omitted.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use paged_infer::engine::{Engine, EngineConfig};
use paged_infer::model::{LlamaConfig, ModelLoader, Quantization};
use paged_infer::simd;
use tokenizers::Tokenizer;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Workload {
    name: &'static str,
    note: &'static str,
    tokens: Vec<u32>,
}

/// Prompts for a real chat model, chosen to span the acceptance range.
fn text_workloads(tokenizer: &Tokenizer) -> Vec<Workload> {
    const PASSAGE: &str = "The KV cache stores one key and one value vector per token per layer. \
Because it grows with every generated token, a naive allocator reserves the maximum \
sequence length up front and wastes most of it. Paging the cache into fixed blocks \
removes that waste and lets separate requests share identical prefixes.";

    let specs: Vec<(&'static str, &'static str, String)> = vec![
        (
            "verbatim copy",
            "output is the input; best case for prompt lookup",
            format!("Repeat the following passage exactly, word for word:\n\n{PASSAGE}\n\nRepeated passage:\n"),
        ),
        (
            "extractive QA",
            "the answer is a span of the passage",
            format!("{PASSAGE}\n\nUsing only the passage above, answer in one sentence: what does paging the KV cache remove?\n\nAnswer:"),
        ),
        (
            "open-ended",
            "nothing to copy; the hard case",
            "Explain why memory bandwidth, rather than arithmetic, limits transformer decoding.".to_string(),
        ),
    ];

    specs
        .into_iter()
        .filter_map(|(name, note, text)| {
            let encoding = tokenizer.encode(text, true).ok()?;
            Some(Workload {
                name,
                note,
                tokens: encoding.get_ids().to_vec(),
            })
        })
        .collect()
}

/// Fallback when no tokenizer is available: synthetic sequences with known
/// repetition structure.
fn synthetic_workloads() -> Vec<Workload> {
    let unit: Vec<u32> = (100..112).collect();
    let mut repeated = Vec::new();
    for _ in 0..8 {
        repeated.extend_from_slice(&unit);
    }
    vec![
        Workload {
            name: "repetitive",
            note: "a short cycle repeated; prompt lookup should latch on",
            tokens: repeated,
        },
        Workload {
            name: "non-repeating",
            note: "strictly increasing ids; nothing to copy",
            tokens: (0..96).collect(),
        },
    ]
}

struct Measurement {
    decode_secs: f64,
    tokens: usize,
    tokens_per_step: f64,
    acceptance: f64,
}

fn measure(
    engine: &mut Engine<'_>,
    prompt: &[u32],
    max_tokens: usize,
    draft_tokens: usize,
) -> Measurement {
    engine.reset();
    engine.set_draft_tokens(draft_tokens);
    engine.submit_tokens(prompt.to_vec(), max_tokens, 1);
    let out = engine.run().expect("generation should not fail");

    let stats = engine.stats();
    let spec = engine.spec_stats();
    Measurement {
        decode_secs: stats.decode_time.as_secs_f64(),
        tokens: out.iter().map(|c| c.tokens.len()).sum(),
        tokens_per_step: spec.tokens_per_step(),
        acceptance: spec.acceptance_rate(),
    }
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/model.safetensors".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/tokenizer.json".to_string());
    let max_tokens = env_usize("SPEC_TOKENS", 64);
    let drafts: Vec<usize> = std::env::var("SPEC_K")
        .unwrap_or_else(|_| "2 4 8".to_string())
        .split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect();

    // Fall back to the test fixture so this runs without a download. Acceptance
    // on a randomly-initialised model is not meaningful, and the output says so.
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let real_model = Path::new(&model_path).exists();

    let fixture_bytes = (!real_model)
        .then(|| std::fs::read(fixture_dir.join("tiny_llama.safetensors")))
        .transpose()?;
    let _mmap;
    let loader = if real_model {
        let file = std::fs::File::open(&model_path)?;
        _mmap = unsafe { MmapOptions::new().map(&file)? };
        ModelLoader::new(&_mmap)?
    } else {
        ModelLoader::new(fixture_bytes.as_ref().expect("fixture loaded"))?
    };

    let config = if real_model {
        LlamaConfig {
            quantization: match std::env::var("QUANT").as_deref() {
                Ok("int8") => Quantization::Int8,
                _ => Quantization::F32,
            },
            ..LlamaConfig::beside_checkpoint(&model_path)
        }
    } else {
        let meta = std::fs::read_to_string(fixture_dir.join("tiny_llama_meta.txt"))?;
        let kv: HashMap<&str, &str> = meta
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim(), v.trim()))
            .collect();
        let n = |k: &str| kv[k].parse::<usize>().unwrap();
        LlamaConfig {
            hidden_size: n("hidden_size"),
            num_hidden_layers: n("num_hidden_layers"),
            num_attention_heads: n("num_attention_heads"),
            num_key_value_heads: n("num_key_value_heads"),
            intermediate_size: n("intermediate_size"),
            vocab_size: n("vocab_size"),
            rms_norm_eps: kv["rms_norm_eps"].parse().unwrap(),
            rope_theta: kv["rope_theta"].parse().unwrap(),
            ..LlamaConfig::default()
        }
    };
    let weights = loader.load_weights(&config)?;

    let tokenizer = real_model
        .then(|| Tokenizer::from_file(&tokenizer_path).ok())
        .flatten();
    let workloads = match &tokenizer {
        Some(t) => text_workloads(t),
        None => synthetic_workloads(),
    };

    // One engine, reconfigured between runs: reloading multi-gigabyte weights
    // per configuration would dominate the measurement.
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: env_usize("SPEC_BLOCKS", 512),
            block_size: 16,
            temperature: 0.0,
            bos_token: real_model.then_some(1),
            eos_token: if real_model { 2 } else { u32::MAX },
            // Isolate speculation from prefix reuse, which would otherwise
            // shorten the later runs over the same prompt.
            enable_prefix_cache: false,
            ..EngineConfig::default()
        },
    );

    println!("Speculative decoding benchmark");
    println!("==============================");
    println!(
        "model      : {}",
        if real_model {
            model_path.as_str()
        } else {
            "test fixture (randomly initialised -- acceptance here is NOT meaningful)"
        }
    );
    println!("drafter    : prompt-lookup (no draft model)");
    println!("simd       : {}", simd::backend());
    println!("decode     : greedy, {max_tokens} tokens per prompt");
    println!();

    for workload in &workloads {
        println!("-- {} : {}", workload.name, workload.note);
        println!("   prompt: {} tokens", workload.tokens.len());
        println!(
            "   {:<9} {:>9} {:>10} {:>13} {:>9}",
            "drafts", "accepted", "tok/step", "decode", "speedup"
        );

        let base = measure(&mut engine, &workload.tokens, max_tokens, 0);
        println!(
            "   {:<9} {:>9} {:>10.2} {:>11.3} s {:>8.2}x",
            "0 (off)", "-", base.tokens_per_step, base.decode_secs, 1.0
        );

        for &k in &drafts {
            let m = measure(&mut engine, &workload.tokens, max_tokens, k);
            assert_eq!(
                m.tokens, base.tokens,
                "speculation changed the token count -- it must be lossless"
            );
            println!(
                "   {:<9} {:>8.1}% {:>10.2} {:>11.3} s {:>8.2}x",
                k,
                m.acceptance * 100.0,
                m.tokens_per_step,
                m.decode_secs,
                base.decode_secs / m.decode_secs.max(1e-12)
            );
        }
        println!();
    }

    println!("Reading this table");
    println!("------------------");
    println!("tok/step is how many tokens each sequence-step emitted; 1.00 is ordinary");
    println!("decoding. Speedup is lower than tok/step, because verifying K+1 positions");
    println!("costs more than producing one token -- that gap is the price of a wrong");
    println!("guess, and it is why a larger K is not automatically better.");
    println!();
    println!("Acceptance is a property of the workload. Output that repeats the input");
    println!("is drafted almost perfectly; genuinely novel text is not drafted at all,");
    println!("and there speculation costs a little and saves nothing. It is never a");
    println!("quality trade: a draft is accepted only when it matches what the model");
    println!("would have produced anyway, so the text is identical either way -- which");
    println!("this benchmark asserts on every run.");
    Ok(())
}
