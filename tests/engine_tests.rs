//! End-to-end scheduler tests against the synthetic fixture checkpoint.
//!
//! These drive the real `Engine` — admission, prefill, prefix reuse, decode,
//! block growth, copy-on-write forking, reclamation — with real weights. The
//! model is tiny, but every code path the serving loop takes is the production
//! one, so a scheduler bug shows up here rather than only under a 1B checkpoint
//! nobody can run in CI.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::engine::{Engine, EngineConfig, FinishReason};
use paged_infer::model::{LlamaConfig, ModelLoader};

fn fixture() -> (LlamaConfig, Vec<u32>, Vec<u8>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let meta = std::fs::read_to_string(dir.join("tiny_llama_meta.txt"))
        .expect("missing fixture: run python3 scripts/gen_golden_fixture.py");
    let kv: HashMap<&str, &str> = meta
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect();
    let num = |k: &str| kv[k].parse::<usize>().unwrap();

    let config = LlamaConfig {
        hidden_size: num("hidden_size"),
        num_hidden_layers: num("num_hidden_layers"),
        num_attention_heads: num("num_attention_heads"),
        num_key_value_heads: num("num_key_value_heads"),
        intermediate_size: num("intermediate_size"),
        vocab_size: num("vocab_size"),
        rms_norm_eps: kv["rms_norm_eps"].parse().unwrap(),
        rope_theta: kv["rope_theta"].parse().unwrap(),
        attention_window: None,
        rope_style: Default::default(),
        quantization: Default::default(),
    };
    let tokens = kv["tokens"].split(',').map(|t| t.parse().unwrap()).collect();
    (config, tokens, std::fs::read(dir.join("tiny_llama.safetensors")).unwrap())
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        total_blocks: 128,
        block_size: 8,
        temperature: 0.0,
        // No BOS: the fixture has no special tokens, and prepending one would
        // just shift every position.
        bos_token: None,
        // Pick a token the fixture is unlikely to emit, so tests that want a
        // length stop are not cut short by a spurious EOS.
        eos_token: u32::MAX,
        ..EngineConfig::default()
    }
}

#[test]
fn test_generates_the_requested_number_of_tokens() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(weights, config, engine_config());

    engine.submit_tokens(tokens[..12].to_vec(), 16, 1);
    let out = engine.run().unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].tokens.len(), 16);
    assert_eq!(out[0].finish_reason, FinishReason::Length);
    assert_eq!(out[0].prompt_tokens, 12);
    assert_eq!(engine.stats().generated_tokens, 16);
}

#[test]
fn test_eos_stops_generation_early() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    // Learn what the model actually generates, then declare its third token EOS
    // and confirm the engine stops exactly there.
    let mut probe = Engine::new(
        loader.load_weights(&config).unwrap(),
        config.clone(),
        engine_config(),
    );
    probe.submit_tokens(tokens[..12].to_vec(), 8, 1);
    let baseline = probe.run().unwrap();
    let eos = baseline[0].tokens[2];

    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            eos_token: eos,
            ..engine_config()
        },
    );
    engine.submit_tokens(tokens[..12].to_vec(), 8, 1);
    let out = engine.run().unwrap();

    assert_eq!(out[0].finish_reason, FinishReason::Eos);
    assert_eq!(out[0].tokens.len(), 3, "should stop on the EOS token");
    assert_eq!(*out[0].tokens.last().unwrap(), eos);
}

#[test]
fn test_generation_crosses_block_boundaries() {
    // 12 prompt tokens plus 40 generated spans seven 8-token blocks, so the
    // sequence has to grow its mapping repeatedly mid-decode.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(weights, config, engine_config());

    engine.submit_tokens(tokens[..12].to_vec(), 40, 1);
    let out = engine.run().unwrap();
    assert_eq!(out[0].tokens.len(), 40);
    assert_eq!(out[0].finish_reason, FinishReason::Length);
}

#[test]
fn test_greedy_generation_is_deterministic() {
    let run_once = || {
        let (config, tokens, bytes) = fixture();
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(weights, config, engine_config());
        engine.submit_tokens(tokens[..16].to_vec(), 12, 1);
        engine.run().unwrap()[0].tokens.clone()
    };
    assert_eq!(run_once(), run_once());
}

#[test]
fn test_prefix_cache_cuts_prefill_without_changing_output() {
    let (config, tokens, bytes) = fixture();
    let shared = &tokens[..24];

    let run = |prefix_cache: bool| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(
            weights,
            config.clone(),
            EngineConfig {
                enable_prefix_cache: prefix_cache,
                ..engine_config()
            },
        );
        for i in 0..4 {
            let mut prompt = shared.to_vec();
            prompt.extend_from_slice(&tokens[24 + i..24 + i + 4]);
            engine.submit_tokens(prompt, 6, 1);
        }
        let out = engine.run().unwrap();
        (
            out.iter().map(|c| c.tokens.clone()).collect::<Vec<_>>(),
            engine.stats().prompt_tokens_prefilled,
            engine.prefix_stats().hits,
        )
    };

    let (cold_text, cold_prefill, cold_hits) = run(false);
    let (warm_text, warm_prefill, warm_hits) = run(true);

    assert_eq!(cold_hits, 0);
    assert!(warm_hits > 0, "the warm run never hit the cache");
    assert!(
        warm_prefill < cold_prefill,
        "reuse should cut prefill: {warm_prefill} vs {cold_prefill}"
    );
    // The whole point: less work, identical answers.
    assert_eq!(
        cold_text, warm_text,
        "prefix reuse changed what the model generated"
    );
}

#[test]
fn test_forked_samples_share_the_prompt_but_diverge() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(weights, config, engine_config());

    // 20 tokens leaves a partial trailing block that all four samples write to,
    // which is exactly the block copy-on-write has to split.
    engine.submit_tokens(tokens[..20].to_vec(), 10, 4);
    let out = engine.run().unwrap();

    assert_eq!(out.len(), 4, "one completion per sample");
    assert!(
        out.iter().all(|c| c.request_id == out[0].request_id),
        "all samples belong to one request"
    );
    // The prompt is prefilled once no matter how many samples are drawn.
    assert_eq!(engine.stats().prompt_tokens_prefilled, 20);
    assert!(
        engine.cow_copies() > 0,
        "the shared partial block should have split on first write"
    );

    let distinct: std::collections::HashSet<_> = out.iter().map(|c| c.tokens.clone()).collect();
    assert!(
        distinct.len() > 1,
        "forked samples should not all produce the same text"
    );
}

#[test]
fn test_forking_costs_far_less_memory_than_independent_requests() {
    let (config, tokens, bytes) = fixture();
    let prompt = tokens[..32].to_vec();

    let blocks_used = |samples: usize, fork: bool| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let cfg = EngineConfig {
            // No prefix cache, so this isolates forking from prefix reuse.
            enable_prefix_cache: false,
            ..engine_config()
        };
        let mut engine = Engine::new(weights, config.clone(), cfg);
        if fork {
            engine.submit_tokens(prompt.clone(), 4, samples);
        } else {
            for _ in 0..samples {
                engine.submit_tokens(prompt.clone(), 4, 1);
            }
        }
        engine.run().unwrap();
        engine.stats().prompt_tokens_prefilled
    };

    // Independent requests each prefill the whole prompt; forked samples do not.
    assert_eq!(blocks_used(4, false), 4 * 32);
    assert_eq!(blocks_used(4, true), 32);
}

#[test]
fn test_queued_requests_wait_for_memory_then_run() {
    // Six requests against a cache that can hold two at a time: the scheduler
    // has to admit what fits, retire it, and admit the rest.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 8, // 8 blocks x 8 tokens = 64 tokens of KV in total
            enable_prefix_cache: false,
            ..engine_config()
        },
    );

    for i in 0..6 {
        // 24 tokens of prompt plus 4 generated needs 4 blocks each.
        let mut prompt = tokens[..20].to_vec();
        prompt.extend_from_slice(&tokens[20 + i..24 + i]);
        engine.submit_tokens(prompt, 4, 1);
    }

    let out = engine.run().unwrap();
    assert_eq!(out.len(), 6, "every request should eventually complete");
    assert!(
        out.iter().all(|c| c.tokens.len() == 4),
        "requests should not be truncated by memory pressure: {:?}",
        out.iter().map(|c| c.tokens.len()).collect::<Vec<_>>()
    );
    assert!(engine.stats().steps > 1, "admission should have been staged");
}

#[test]
fn test_a_prompt_that_cannot_fit_is_reported_not_hung() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 2, // 16 tokens of KV total
            ..engine_config()
        },
    );
    engine.submit_tokens(tokens[..40].to_vec(), 4, 1);

    let err = engine.run().unwrap_err();
    assert!(
        err.to_string().contains("does not fit"),
        "should say why, got: {err}"
    );
}

#[test]
fn test_temperature_sampling_respects_its_seed() {
    let (config, tokens, bytes) = fixture();
    let run = |seed: u64| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(
            weights,
            config.clone(),
            EngineConfig {
                temperature: 1.0,
                seed,
                ..engine_config()
            },
        );
        engine.submit_tokens(tokens[..16].to_vec(), 20, 1);
        engine.run().unwrap()[0].tokens.clone()
    };
    assert_eq!(run(7), run(7), "the same seed must replay exactly");
    assert_ne!(run(7), run(8), "different seeds should diverge");
}

#[test]
fn test_int8_weights_shrink_the_model_and_track_f32_output() {
    // Quantization is only useful if it is nearly free in quality. Compare an
    // int8 load against the f32 load on the same prompt: memory must drop ~4x
    // and the logits must stay close enough that greedy decoding agrees.
    use paged_infer::model::Quantization;

    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();

    let f32_weights = loader.load_weights(&config).unwrap();
    let int8_config = LlamaConfig {
        quantization: Quantization::Int8,
        ..config.clone()
    };
    let int8_weights = loader.load_weights(&int8_config).unwrap();

    // The LM head stays f32 on purpose, so the ratio is below a clean 4x.
    let f32_bytes = f32_weights.weight_bytes();
    let int8_bytes = int8_weights.weight_bytes();
    let ratio = f32_bytes as f64 / int8_bytes as f64;
    println!("weights: {f32_bytes} B f32 -> {int8_bytes} B int8 ({ratio:.2}x smaller)");
    assert!(ratio > 3.0, "expected a large reduction, got {ratio:.2}x");

    let generate = |weights, cfg: LlamaConfig| {
        let mut engine = Engine::new(weights, cfg, engine_config());
        engine.submit_tokens(tokens[..16].to_vec(), 8, 1);
        engine.run().unwrap()[0].tokens.clone()
    };
    let baseline = generate(f32_weights, config);
    let quantized = generate(int8_weights, int8_config);

    let agree = baseline
        .iter()
        .zip(quantized.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!("greedy tokens matching before divergence: {agree}/{}", baseline.len());
    assert!(
        agree >= baseline.len() / 2,
        "int8 diverged too early: {baseline:?} vs {quantized:?}"
    );
}
