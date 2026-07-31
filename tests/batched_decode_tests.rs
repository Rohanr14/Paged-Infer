//! Batched decode must be numerically indistinguishable from decoding each
//! sequence on its own.
//!
//! Batching changes only *when* the weights are read, never what is computed —
//! so any divergence beyond float reassociation is a bug. These tests run both
//! paths over the fixture model and compare, including the cases most likely to
//! break: sequences at different positions, different context lengths, and
//! block tables that are not laid out alike.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{BatchScratch, ForwardScratch, LlamaConfig, LlamaWeights, ModelLoader};

const BLOCK_SIZE: usize = 8;
const TOTAL_BLOCKS: usize = 256;

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

fn new_cache(config: &LlamaConfig) -> Vec<f32> {
    vec![0.0; config.kv_layout(TOTAL_BLOCKS, BLOCK_SIZE).total_floats()]
}

/// Seed a cache by prefilling `prompt` for one sequence, returning its table.
fn prefill_one(
    weights: &LlamaWeights<'_>,
    config: &LlamaConfig,
    cache: &mut [f32],
    allocator: &mut BlockAllocator,
    prompt: &[u32],
    extra_blocks: usize,
) -> BlockTable {
    let mut table = BlockTable::new();
    for _ in 0..prompt.len().div_ceil(BLOCK_SIZE) + extra_blocks {
        table.append_block(allocator.allocate().unwrap());
    }
    weights.prefill(prompt, 0, config, &table, cache, BLOCK_SIZE, None);
    table
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Build `n` sequences with deliberately different prompt lengths, decode one
/// step both ways, and compare.
fn compare_batch(prompt_lens: &[usize]) -> f32 {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let batch = prompt_lens.len();

    // Sequential reference: each sequence gets its own cache and its own
    // single-sequence forward.
    let mut sequential = Vec::new();
    for &len in prompt_lens {
        let mut cache = new_cache(&config);
        let mut allocator = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
        let table = prefill_one(
            &weights,
            &config,
            &mut cache,
            &mut allocator,
            &tokens[..len],
            2,
        );
        let mut scratch = ForwardScratch::new(&config);
        weights.forward_into(
            tokens[len],
            len,
            &config,
            &table,
            &mut cache,
            BLOCK_SIZE,
            None,
            &mut scratch,
        );
        sequential.push(scratch.logits.clone());
    }

    // Batched: all sequences share one cache, exactly as the engine has them.
    let mut cache = new_cache(&config);
    let mut allocator = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let tables: Vec<BlockTable> = prompt_lens
        .iter()
        .map(|&len| {
            prefill_one(
                &weights,
                &config,
                &mut cache,
                &mut allocator,
                &tokens[..len],
                2,
            )
        })
        .collect();
    let table_refs: Vec<&BlockTable> = tables.iter().collect();
    let step_tokens: Vec<u32> = prompt_lens.iter().map(|&len| tokens[len]).collect();
    let positions: Vec<usize> = prompt_lens.to_vec();

    let mut batch_scratch = BatchScratch::new(&config, batch);
    weights.decode_batch_into(
        &step_tokens,
        &positions,
        &table_refs,
        &config,
        &mut cache,
        BLOCK_SIZE,
        &mut batch_scratch,
    );

    (0..batch)
        .map(|b| {
            max_abs(
                batch_scratch.logits_for(b, config.vocab_size),
                &sequential[b],
            )
        })
        .fold(0.0_f32, f32::max)
}

#[test]
fn test_batch_of_one_matches_the_single_sequence_path() {
    let delta = compare_batch(&[12]);
    println!("batch=1: max|delta|={delta:.8}");
    assert!(delta < 1e-5, "batch of one diverged: {delta}");
}

#[test]
fn test_uniform_batch_matches_sequential() {
    let delta = compare_batch(&[12, 12, 12, 12]);
    println!("batch=4 uniform: max|delta|={delta:.8}");
    assert!(delta < 1e-5, "uniform batch diverged: {delta}");
}

#[test]
fn test_ragged_batch_matches_sequential() {
    // Different positions mean different rotary angles, different attention
    // spans, and different block-table depths in the same batch. This is the
    // case a naive implementation gets wrong.
    let delta = compare_batch(&[4, 9, 17, 24, 31]);
    println!("batch=5 ragged: max|delta|={delta:.8}");
    assert!(delta < 1e-5, "ragged batch diverged: {delta}");
}

#[test]
fn test_batch_matches_across_a_block_boundary() {
    // Position 15 with block_size 8 sits at the end of block 1; position 16
    // opens block 2. Both are in the same batch.
    let delta = compare_batch(&[15, 16, 8, 7]);
    println!("batch across block boundary: max|delta|={delta:.8}");
    assert!(delta < 1e-5, "block-boundary batch diverged: {delta}");
}

#[test]
fn test_batched_decode_is_deterministic() {
    let a = compare_batch(&[6, 13, 20]);
    let b = compare_batch(&[6, 13, 20]);
    assert_eq!(a, b, "batched decode is not reproducible");
}

#[test]
fn test_sliding_window_is_respected_per_sequence() {
    // With a window, each sequence attends over its own span. Sequences at
    // different positions get different spans in the same batch, and the
    // batched path pads its score lanes to the widest -- the padding must not
    // leak into the shorter sequences.
    let (mut config, tokens, bytes) = fixture();
    config.attention_window = Some(6);
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    let lens = [4_usize, 20];
    let mut sequential = Vec::new();
    for &len in &lens {
        let mut cache = new_cache(&config);
        let mut allocator = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
        let table = prefill_one(
            &weights,
            &config,
            &mut cache,
            &mut allocator,
            &tokens[..len],
            2,
        );
        let mut scratch = ForwardScratch::new(&config);
        weights.forward_into(
            tokens[len],
            len,
            &config,
            &table,
            &mut cache,
            BLOCK_SIZE,
            None,
            &mut scratch,
        );
        sequential.push(scratch.logits.clone());
    }

    let mut cache = new_cache(&config);
    let mut allocator = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let tables: Vec<BlockTable> = lens
        .iter()
        .map(|&len| {
            prefill_one(
                &weights,
                &config,
                &mut cache,
                &mut allocator,
                &tokens[..len],
                2,
            )
        })
        .collect();
    let refs: Vec<&BlockTable> = tables.iter().collect();
    let step: Vec<u32> = lens.iter().map(|&l| tokens[l]).collect();

    let mut bs = BatchScratch::new(&config, lens.len());
    weights.decode_batch_into(
        &step, &lens, &refs, &config, &mut cache, BLOCK_SIZE, &mut bs,
    );

    for (b, expected) in sequential.iter().enumerate() {
        let delta = max_abs(bs.logits_for(b, config.vocab_size), expected);
        assert!(delta < 1e-5, "windowed sequence {b} diverged: {delta}");
    }
}
