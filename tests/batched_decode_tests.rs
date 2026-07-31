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

// ── batched prefill ──────────────────────────────────────────────────────────

use paged_infer::model::LlamaConfig as Cfg;

/// Prefill a prompt both ways and compare the resulting logits.
fn compare_prefill(prompt_len: usize, chunk_size: usize, start_split: Option<usize>) -> f32 {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let prompt = &tokens[..prompt_len];
    let blocks = prompt_len.div_ceil(BLOCK_SIZE) + 2;

    let build_table = |allocator: &mut BlockAllocator| {
        let mut t = BlockTable::new();
        for _ in 0..blocks {
            t.append_block(allocator.allocate().unwrap());
        }
        t
    };

    // Reference: the position-at-a-time prefill.
    let mut ref_cache = new_cache(&config);
    let mut ref_alloc = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let ref_table = build_table(&mut ref_alloc);
    let reference = weights.prefill(
        prompt,
        0,
        &config,
        &ref_table,
        &mut ref_cache,
        BLOCK_SIZE,
        None,
    );

    // Batched, optionally resumed mid-prompt the way a prefix-cache hit does.
    let mut cache = new_cache(&config);
    let mut alloc = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let table = build_table(&mut alloc);
    let mut scratch = BatchScratch::new(&config, chunk_size.max(1));
    match start_split {
        None => weights.prefill_batched(
            prompt,
            0,
            &config,
            &table,
            &mut cache,
            BLOCK_SIZE,
            chunk_size,
            &mut scratch,
        ),
        Some(split) => {
            weights.prefill_batched(
                &prompt[..split],
                0,
                &config,
                &table,
                &mut cache,
                BLOCK_SIZE,
                chunk_size,
                &mut scratch,
            );
            weights.prefill_batched(
                &prompt[split..],
                split,
                &config,
                &table,
                &mut cache,
                BLOCK_SIZE,
                chunk_size,
                &mut scratch,
            );
        }
    }

    max_abs(scratch.logits_for(0, config.vocab_size), &reference)
}

#[test]
fn test_batched_prefill_matches_sequential_prefill() {
    for chunk in [1, 2, 4, 8, 16, 64] {
        let delta = compare_prefill(24, chunk, None);
        println!("prefill chunk={chunk}: max|delta|={delta:.8}");
        assert!(delta < 1e-5, "chunk {chunk} diverged: {delta}");
    }
}

#[test]
fn test_batched_prefill_handles_a_ragged_final_chunk() {
    // 23 tokens in chunks of 8 leaves a final chunk of 7.
    let delta = compare_prefill(23, 8, None);
    println!("ragged final chunk: max|delta|={delta:.8}");
    assert!(delta < 1e-5, "ragged tail diverged: {delta}");
}

#[test]
fn test_batched_prefill_chunk_larger_than_the_prompt() {
    let delta = compare_prefill(5, 64, None);
    assert!(delta < 1e-5, "oversized chunk diverged: {delta}");
}

#[test]
fn test_batched_prefill_can_resume_mid_prompt() {
    // What a prefix-cache hit does: replay only the uncached suffix and land on
    // exactly the state a full prefill would have produced.
    let delta = compare_prefill(24, 8, Some(16));
    println!("resumed prefill: max|delta|={delta:.8}");
    assert!(delta < 1e-5, "resumed prefill diverged: {delta}");
}

#[test]
fn test_batched_prefill_is_causal() {
    // The load-bearing property: a position inside a chunk must not see the
    // positions after it. If it did, changing a later token would change an
    // earlier one's output.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    let run = |prompt: &[u32]| -> Vec<f32> {
        let mut cache = new_cache(&config);
        let mut alloc = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
        let mut table = BlockTable::new();
        for _ in 0..prompt.len().div_ceil(BLOCK_SIZE) + 2 {
            table.append_block(alloc.allocate().unwrap());
        }
        let mut scratch = BatchScratch::new(&config, 16);
        // One chunk holds the whole prompt, so every position is in the batch
        // together -- the case where a masking bug would show.
        weights.prefill_batched(
            prompt,
            0,
            &config,
            &table,
            &mut cache,
            BLOCK_SIZE,
            16,
            &mut scratch,
        );
        scratch.logits_for(0, config.vocab_size).to_vec()
    };

    // Truncating the prompt must reproduce the shorter prompt's own answer.
    let short = run(&tokens[..8]);
    let mut altered = tokens[..8].to_vec();
    altered.extend_from_slice(&[7, 7, 7, 7]);
    // The 12-token run's *first eight* positions saw the same history, so
    // prefilling 8 alone must equal prefilling 8 of the 12-token prompt.
    let mut cache = new_cache(&config);
    let mut alloc = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let mut table = BlockTable::new();
    for _ in 0..4 {
        table.append_block(alloc.allocate().unwrap());
    }
    let mut scratch = BatchScratch::new(&config, 16);
    weights.prefill_batched(
        &altered,
        0,
        &config,
        &table,
        &mut cache,
        BLOCK_SIZE,
        16,
        &mut scratch,
    );
    // Re-derive position 7's logits from the same cache by decoding token 7
    // again at position 7 -- it must not have been contaminated by 8..11.
    let mut single = ForwardScratch::new(&config);
    weights.forward_into(
        tokens[7],
        7,
        &config,
        &table,
        &mut cache,
        BLOCK_SIZE,
        None,
        &mut single,
    );
    let delta = max_abs(&single.logits, &short);
    println!("causality check: max|delta|={delta:.8}");
    assert!(
        delta < 1e-5,
        "a position saw tokens that come after it: {delta}"
    );
}

#[test]
fn test_batched_prefill_respects_scratch_capacity() {
    // Asking for a chunk larger than the scratch was built for must clamp, not
    // overrun.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut cache = new_cache(&config);
    let mut alloc = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let mut table = BlockTable::new();
    for _ in 0..6 {
        table.append_block(alloc.allocate().unwrap());
    }
    let mut scratch = BatchScratch::new(&config, 4);
    weights.prefill_batched(
        &tokens[..20],
        0,
        &config,
        &table,
        &mut cache,
        BLOCK_SIZE,
        999,
        &mut scratch,
    );

    let mut ref_cache = new_cache(&config);
    let mut ref_alloc = BlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let mut ref_table = BlockTable::new();
    for _ in 0..6 {
        ref_table.append_block(ref_alloc.allocate().unwrap());
    }
    let reference = weights.prefill(
        &tokens[..20],
        0,
        &config,
        &ref_table,
        &mut ref_cache,
        BLOCK_SIZE,
        None,
    );
    let delta = max_abs(scratch.logits_for(0, config.vocab_size), &reference);
    assert!(delta < 1e-5, "capacity-clamped prefill diverged: {delta}");
}

// Silence the unused-import warning when only some tests reference it.
#[allow(dead_code)]
fn _cfg_marker(_: &Cfg) {}
