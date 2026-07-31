//! Prefix-cache reuse has to be numerically invisible.
//!
//! Skipping recomputation is only sound if the KV blocks a request inherits are
//! bit-for-bit what it would have produced itself. These tests run the real
//! fixture model through `KvCacheManager`, take actual cache hits, and compare
//! the resulting logits against a run with the cache disabled. Anything less
//! and "we cache prefixes" is an unverified claim.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::memory::kv_cache_manager::KvCacheManager;
use paged_infer::memory::layout::KvLayout;
use paged_infer::model::{LlamaConfig, LlamaWeights, ModelLoader};

const BLOCK_SIZE: usize = 8;
const TOTAL_BLOCKS: usize = 64;

fn load_config() -> (LlamaConfig, Vec<u32>, Vec<u8>) {
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
    let tokens: Vec<u32> = kv["tokens"]
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    let weights = std::fs::read(dir.join("tiny_llama.safetensors")).unwrap();
    (config, tokens, weights)
}

/// Run a batch of prompts through the manager, returning each prompt's final
/// logits plus how many prompt tokens were actually recomputed.
fn run_batch(
    weights: &LlamaWeights<'_>,
    config: &LlamaConfig,
    prompts: &[Vec<u32>],
    prefix_cache: bool,
) -> (Vec<Vec<f32>>, usize, u64) {
    let layout: KvLayout = config.kv_layout(TOTAL_BLOCKS, BLOCK_SIZE);
    let mut kv_cache = vec![0.0_f32; layout.total_floats()];
    let mut mgr = KvCacheManager::new(TOTAL_BLOCKS, BLOCK_SIZE).with_prefix_cache(prefix_cache);

    let mut all_logits = Vec::new();
    let mut recomputed = 0;

    for (seq_id, prompt) in prompts.iter().enumerate() {
        let admission = mgr
            .admit(seq_id, prompt, seq_id as u64)
            .expect("fixture batch should fit");
        let resume_at = admission.cached_tokens.min(prompt.len() - 1);
        recomputed += prompt.len() - resume_at;

        let logits = weights.prefill(
            &prompt[resume_at..],
            resume_at,
            config,
            &admission.block_table,
            &mut kv_cache,
            BLOCK_SIZE,
            None,
        );
        mgr.publish_prompt_blocks(&admission.block_table);
        all_logits.push(logits);
        mgr.release_sequence(seq_id);
    }

    (all_logits, recomputed, mgr.prefix_stats().hits)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn test_prefix_reuse_does_not_change_the_logits() {
    let (config, tokens, weight_bytes) = load_config();
    let loader = ModelLoader::new(&weight_bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    // A shared 24-token preamble (three full 8-token blocks) plus divergent
    // tails, which is the shape real serving traffic has.
    let shared = &tokens[..24];
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|i| {
            let mut p = shared.to_vec();
            p.extend_from_slice(&tokens[24 + i..24 + i + 6]);
            p
        })
        .collect();

    let (cold, cold_recomputed, cold_hits) = run_batch(&weights, &config, &prompts, false);
    let (warm, warm_recomputed, warm_hits) = run_batch(&weights, &config, &prompts, true);

    assert_eq!(cold_hits, 0, "cache was supposed to be disabled");
    assert!(warm_hits > 0, "the warm run never hit the cache");
    assert!(
        warm_recomputed < cold_recomputed,
        "reuse should have skipped work: {warm_recomputed} vs {cold_recomputed}"
    );
    println!(
        "prompt tokens recomputed: {cold_recomputed} without cache -> {warm_recomputed} with ({} hits)",
        warm_hits
    );

    for (i, (c, w)) in cold.iter().zip(warm.iter()).enumerate() {
        let delta = max_abs(c, w);
        assert!(
            delta < 1e-5,
            "prompt {i}: reuse changed the logits by {delta}"
        );
    }
}

#[test]
fn test_identical_prompts_reuse_everything_and_agree() {
    let (config, tokens, weight_bytes) = load_config();
    let loader = ModelLoader::new(&weight_bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    // Length is an exact multiple of the block size, so every block is
    // cacheable and the second request should recompute only its last token.
    let prompt = tokens[..32].to_vec();
    let prompts = vec![prompt.clone(), prompt.clone(), prompt];

    let (cold, _, _) = run_batch(&weights, &config, &prompts, false);
    let (warm, warm_recomputed, _) = run_batch(&weights, &config, &prompts, true);

    // First request prefills 32; the next two recompute one token each.
    assert_eq!(warm_recomputed, 32 + 1 + 1);
    for (i, (c, w)) in cold.iter().zip(warm.iter()).enumerate() {
        assert!(max_abs(c, w) < 1e-5, "repeat {i} diverged");
    }
    // All three prompts are the same, so all three answers must be too.
    assert!(max_abs(&warm[0], &warm[2]) < 1e-6);
}

#[test]
fn test_a_colliding_suffix_with_a_different_history_is_not_reused() {
    // The soundness case the chained hash exists for: the same 8 tokens
    // preceded by different history must produce different KV, and the cache
    // must not serve one for the other.
    let (config, tokens, weight_bytes) = load_config();
    let loader = ModelLoader::new(&weight_bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    let tail = &tokens[16..24];
    let a: Vec<u32> = tokens[..8].iter().chain(tail).copied().collect();
    let b: Vec<u32> = tokens[8..16].iter().chain(tail).copied().collect();

    let (cold, _, _) = run_batch(&weights, &config, &[a.clone(), b.clone()], false);
    let (warm, _, _) = run_batch(&weights, &config, &[a, b], true);

    for (i, (c, w)) in cold.iter().zip(warm.iter()).enumerate() {
        assert!(
            max_abs(c, w) < 1e-5,
            "prompt {i} was served a block from the wrong history"
        );
    }
    assert!(
        max_abs(&warm[0], &warm[1]) > 1e-3,
        "the two histories should genuinely differ, or this test proves nothing"
    );
}

#[test]
fn test_forked_samples_do_not_corrupt_each_other() {
    // Copy-on-write correctness under the real model: two sequences fork a
    // shared prompt, then decode different tokens. Each must see exactly the KV
    // it would have seen on its own.
    let (config, tokens, weight_bytes) = load_config();
    let loader = ModelLoader::new(&weight_bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    let layout = config.kv_layout(TOTAL_BLOCKS, BLOCK_SIZE);
    // 20 tokens leaves the third block partial, so the forked pair shares a
    // block they both write to -- the case copy-on-write has to handle.
    let prompt = &tokens[..20];
    let (branch_a, branch_b) = (tokens[30], tokens[31]);
    assert_ne!(branch_a, branch_b, "branches must differ to be a real test");

    // Reference: each branch run entirely on its own, no sharing at all.
    let solo = |next: u32| -> Vec<f32> {
        let mut cache = vec![0.0_f32; layout.total_floats()];
        let mut mgr = KvCacheManager::new(TOTAL_BLOCKS, BLOCK_SIZE).with_prefix_cache(false);
        let adm = mgr.admit(0, prompt, 0).unwrap();
        let mut table = adm.block_table;
        weights.prefill(prompt, 0, &config, &table, &mut cache, BLOCK_SIZE, None);
        if prompt.len() >= table.len() * BLOCK_SIZE {
            assert!(mgr.append_block(0, &mut table, 1));
        }
        weights.forward(
            next,
            prompt.len(),
            &config,
            &table,
            &mut cache,
            BLOCK_SIZE,
            None,
        )
    };
    let ref_a = solo(branch_a);
    let ref_b = solo(branch_b);

    // Forked: one prefill, two sequences sharing its blocks.
    let mut cache = vec![0.0_f32; layout.total_floats()];
    let mut mgr = KvCacheManager::new(TOTAL_BLOCKS, BLOCK_SIZE).with_prefix_cache(false);
    let adm = mgr.admit(0, prompt, 0).unwrap();
    let mut table_a = adm.block_table;
    weights.prefill(prompt, 0, &config, &table_a, &mut cache, BLOCK_SIZE, None);

    let blocks_before = mgr.total_blocks() - mgr.available_blocks();
    let mut table_b = mgr.fork(&table_a, 1, 1);
    assert_eq!(
        mgr.total_blocks() - mgr.available_blocks(),
        blocks_before,
        "forking must not allocate"
    );

    let pos = prompt.len();
    let mut decode = |seq: usize, table: &mut _, next: u32| {
        if pos >= paged_infer::memory::block_table::BlockTable::len(table) * BLOCK_SIZE {
            assert!(mgr.append_block(seq, table, 2));
        }
        mgr.ensure_writable(seq, table, pos, &mut cache, &layout);
        weights.forward(next, pos, &config, table, &mut cache, BLOCK_SIZE, None)
    };
    let got_a = decode(0, &mut table_a, branch_a);
    let got_b = decode(1, &mut table_b, branch_b);

    assert!(
        mgr.cow_copies() > 0,
        "the shared partial block should have split"
    );
    assert!(
        max_abs(&got_a, &ref_a) < 1e-5,
        "branch A was corrupted by its sibling"
    );
    assert!(
        max_abs(&got_b, &ref_b) < 1e-5,
        "branch B was corrupted by its sibling"
    );
}
