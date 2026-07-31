//! Speculative decoding must be lossless, not merely close.
//!
//! The whole claim rests on one property: a draft token is accepted only when
//! it equals what the model itself would have chosen there, so the emitted
//! sequence is *exactly* the greedy sequence. These tests assert that
//! token-for-token, across draft depths, prompts, batch compositions and
//! stopping conditions. If any of them fails, the feature is a quality
//! regression dressed up as a speedup.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::engine::{Engine, EngineConfig};
use paged_infer::model::{LlamaConfig, ModelLoader};

fn fixture() -> (LlamaConfig, Vec<u32>, Vec<u8>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let meta = std::fs::read_to_string(dir.join("tiny_llama_meta.txt")).unwrap();
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
    let tokens = kv["tokens"]
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    (
        config,
        tokens,
        std::fs::read(dir.join("tiny_llama.safetensors")).unwrap(),
    )
}

fn base_config(draft_tokens: usize) -> EngineConfig {
    EngineConfig {
        total_blocks: 256,
        block_size: 8,
        temperature: 0.0,
        bos_token: None,
        eos_token: u32::MAX,
        draft_tokens,
        ..EngineConfig::default()
    }
}

/// Generate from a set of prompts at a given draft depth, returning the tokens
/// each request produced plus the speculation counters.
fn generate(
    prompts: &[Vec<u32>],
    max_tokens: usize,
    engine_config: EngineConfig,
) -> (Vec<Vec<u32>>, paged_infer::speculative::SpecStats, usize) {
    let (config, _, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(weights, config, engine_config);
    for p in prompts {
        engine.submit_tokens(p.clone(), max_tokens, 1);
    }
    let out = engine.run().unwrap();
    let mut by_request: Vec<(usize, Vec<u32>)> = out
        .iter()
        .map(|c| (c.request_id, c.tokens.clone()))
        .collect();
    by_request.sort_by_key(|(id, _)| *id);
    (
        by_request.into_iter().map(|(_, t)| t).collect(),
        engine.spec_stats(),
        engine.stats().steps,
    )
}

/// A prompt with heavy internal repetition — the regime prompt-lookup drafting
/// is built for, and where acceptance should be high.
fn repetitive_prompt(tokens: &[u32]) -> Vec<u32> {
    let unit = &tokens[..6];
    let mut p = Vec::new();
    for _ in 0..6 {
        p.extend_from_slice(unit);
    }
    p
}

#[test]
fn test_speculative_output_is_identical_to_greedy() {
    let (_, tokens, _) = fixture();
    let prompts = vec![
        tokens[..12].to_vec(),
        repetitive_prompt(&tokens),
        tokens[..20].to_vec(),
    ];

    let (baseline, base_stats, _) = generate(&prompts, 24, base_config(0));
    assert_eq!(base_stats.drafted, 0, "the baseline must not speculate");

    for k in [1usize, 2, 4, 8] {
        let (speculative, stats, _) = generate(&prompts, 24, base_config(k));
        assert_eq!(
            baseline, speculative,
            "draft depth {k} changed the generated text"
        );
        println!(
            "k={k}: drafted {} accepted {} ({:.1}%), {:.2} tokens/step",
            stats.drafted,
            stats.accepted,
            stats.acceptance_rate() * 100.0,
            stats.tokens_per_step()
        );
    }
}

#[test]
fn test_speculation_reduces_model_passes_on_repetitive_text() {
    // Losslessness is necessary but not sufficient — it also has to actually
    // save passes, or the feature is pure overhead.
    let (_, tokens, _) = fixture();
    let prompts = vec![repetitive_prompt(&tokens)];

    let (_, base_stats, _) = generate(&prompts, 32, base_config(0));
    let (_, spec_stats, _) = generate(&prompts, 32, base_config(4));

    println!(
        "passes: {} without speculation -> {} with ({:.2} tokens/step)",
        base_stats.passes,
        spec_stats.passes,
        spec_stats.tokens_per_step()
    );
    assert!(
        spec_stats.acceptance_rate() > 0.0,
        "repetitive text should produce some accepted drafts"
    );
    assert!(
        spec_stats.passes < base_stats.passes,
        "speculation did not reduce passes: {} vs {}",
        spec_stats.passes,
        base_stats.passes
    );
    assert!(spec_stats.tokens_per_step() > 1.0);
}

#[test]
fn test_every_step_emits_exactly_the_accepted_run_plus_one() {
    // The accounting identity the whole scheme rests on: a step emits the
    // drafts the model agreed with, plus the model's own next token. Nothing is
    // invented and nothing is dropped.
    //
    // Note this cannot be tested by assuming a non-repeating prompt yields zero
    // acceptance -- generated tokens join the context, so the drafter can start
    // matching against text the model just produced.
    let prompt: Vec<u32> = (0..24).collect();

    let (baseline, _, _) = generate(std::slice::from_ref(&prompt), 16, base_config(0));
    let (speculative, stats, _) = generate(&[prompt], 16, base_config(4));

    assert_eq!(baseline, speculative, "output diverged from greedy");
    assert_eq!(
        stats.emitted,
        stats.steps + stats.accepted,
        "emitted ({}) should equal steps ({}) + accepted ({})",
        stats.emitted,
        stats.steps,
        stats.accepted
    );
    assert!(
        stats.accepted <= stats.drafted,
        "cannot accept more than was drafted"
    );
}

#[test]
fn test_speculation_respects_the_token_budget() {
    // A step can emit several tokens at once, so the budget has to be enforced
    // mid-run rather than only at a step boundary.
    let (_, tokens, _) = fixture();
    let prompts = vec![repetitive_prompt(&tokens)];
    for max_tokens in [1usize, 2, 3, 5, 7, 13] {
        let (out, _, _) = generate(&prompts, max_tokens, base_config(4));
        assert_eq!(
            out[0].len(),
            max_tokens,
            "budget {max_tokens} overshot to {}",
            out[0].len()
        );
    }
}

#[test]
fn test_speculation_stops_at_eos_inside_an_accepted_run() {
    // If EOS lands among accepted drafts, generation has to stop there and drop
    // the rest, not run past the end of the sequence.
    let (config, tokens, bytes) = fixture();
    let prompts = vec![repetitive_prompt(&tokens)];

    // Find what greedy actually produces, then make its 4th token EOS.
    let (baseline, _, _) = generate(&prompts, 16, base_config(0));
    let eos = baseline[0][3];
    let _ = (config, bytes);

    let cfg = EngineConfig {
        eos_token: eos,
        ..base_config(4)
    };
    let (out, _, _) = generate(&prompts, 16, cfg.clone());
    let cfg_plain = EngineConfig {
        eos_token: eos,
        ..base_config(0)
    };
    let (plain, _, _) = generate(&prompts, 16, cfg_plain);

    assert_eq!(out, plain, "EOS handling diverged under speculation");
    assert_eq!(*out[0].last().unwrap(), eos);
    assert!(
        out[0].len() <= 4,
        "should have stopped at the EOS token, got {} tokens",
        out[0].len()
    );
}

#[test]
fn test_speculation_is_lossless_with_several_sequences_in_flight() {
    // Groups from different sequences share one batch and can be split across
    // chunks. Each sequence must still land on its own greedy output.
    let (_, tokens, _) = fixture();
    let prompts = vec![
        repetitive_prompt(&tokens),
        tokens[..10].to_vec(),
        tokens[..25].to_vec(),
        (0..18).collect(),
    ];

    let (baseline, _, _) = generate(&prompts, 20, base_config(0));
    for k in [2usize, 5] {
        let (speculative, _, _) = generate(&prompts, 20, base_config(k));
        assert_eq!(baseline, speculative, "k={k} diverged with a mixed batch");
    }
}

#[test]
fn test_speculation_is_lossless_when_the_batch_must_be_chunked() {
    // Four sequences drafting 6 tokens each is 28 batch entries; a capacity of
    // 4 forces groups to be split mid-flight.
    let (_, tokens, _) = fixture();
    let prompts = vec![
        repetitive_prompt(&tokens),
        tokens[..14].to_vec(),
        tokens[..22].to_vec(),
        repetitive_prompt(&tokens[3..]),
    ];

    let (baseline, _, _) = generate(&prompts, 18, base_config(0));
    let cramped = EngineConfig {
        max_batch_size: 4,
        prefill_chunk_size: 4,
        ..base_config(6)
    };
    let (speculative, _, _) = generate(&prompts, 18, cramped);
    assert_eq!(
        baseline, speculative,
        "splitting a speculative group across chunks changed the output"
    );
}

#[test]
fn test_speculation_is_lossless_across_kv_block_boundaries() {
    // A draft run can span the end of a block, forcing the mapping to grow by
    // more than one block in a single step.
    let (_, tokens, _) = fixture();
    let prompts = vec![repetitive_prompt(&tokens)];
    let baseline = generate(&prompts, 40, base_config(0)).0;
    for k in [3usize, 7, 15] {
        let out = generate(&prompts, 40, base_config(k)).0;
        assert_eq!(baseline, out, "k={k} diverged across block boundaries");
    }
}

#[test]
fn test_sampled_sequences_do_not_speculate() {
    // Acceptance is defined as "the model would have picked this", which is
    // greedy. A temperature-sampled sequence must take the ordinary path rather
    // than a shortcut that would quietly change its distribution.
    let (_, tokens, _) = fixture();
    let cfg = EngineConfig {
        temperature: 1.0,
        ..base_config(4)
    };
    let (_, stats, _) = generate(&[repetitive_prompt(&tokens)], 16, cfg);
    assert_eq!(
        stats.drafted, 0,
        "a sampled sequence must not have drafts verified greedily"
    );
}

#[test]
fn test_speculation_survives_memory_pressure() {
    // Under a tight pool the planner has to give up speculation rather than the
    // request. Output must still match greedy.
    let (_, tokens, _) = fixture();
    let prompts = vec![repetitive_prompt(&tokens), tokens[..16].to_vec()];

    let baseline = generate(&prompts, 12, base_config(0)).0;
    let tight = EngineConfig {
        total_blocks: 12,
        enable_prefix_cache: false,
        ..base_config(8)
    };
    let out = generate(&prompts, 12, tight).0;
    assert_eq!(baseline, out, "speculation under memory pressure diverged");
}
