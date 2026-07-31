//! End-to-end numerical parity against an independent reference implementation.
//!
//! `scripts/gen_golden_fixture.py` builds a tiny synthetic Llama checkpoint in
//! the exact HuggingFace tensor layout and computes reference logits with a
//! NumPy transcription of `modeling_llama.py`. This test drives the same
//! checkpoint through the real engine — `ModelLoader`, the paged KV cache, the
//! block table — and asserts the two agree.
//!
//! It covers the whole forward pass end to end: embedding lookup, RMSNorm, the
//! rotary convention, grouped-query attention across block boundaries, SwiGLU,
//! and the LM head. Regenerate with `python3 scripts/gen_golden_fixture.py`.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{LlamaConfig, ModelLoader};

const BLOCK_SIZE: usize = 16;

struct Fixture {
    config: LlamaConfig,
    tokens: Vec<u32>,
    golden: Vec<f32>,
    weights: Vec<u8>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_fixture() -> Fixture {
    let dir = fixture_dir();

    let meta = std::fs::read_to_string(dir.join("tiny_llama_meta.txt"))
        .expect("missing fixture: run python3 scripts/gen_golden_fixture.py");
    let kv: HashMap<&str, &str> = meta
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect();

    let num = |key: &str| -> usize { kv[key].parse().expect(key) };
    let flt = |key: &str| -> f32 { kv[key].parse().expect(key) };

    let config = LlamaConfig {
        hidden_size: num("hidden_size"),
        num_hidden_layers: num("num_hidden_layers"),
        num_attention_heads: num("num_attention_heads"),
        num_key_value_heads: num("num_key_value_heads"),
        intermediate_size: num("intermediate_size"),
        vocab_size: num("vocab_size"),
        rms_norm_eps: flt("rms_norm_eps"),
        rope_theta: flt("rope_theta"),
        // The reference is full causal attention, so no sliding window.
        attention_window: None,
        // HF-format checkpoints require the rotate_half convention.
        rope_style: paged_infer::math::RopeStyle::Neox,
        quantization: Default::default(),
    };

    let tokens: Vec<u32> = kv["tokens"]
        .split(',')
        .map(|t| t.parse().expect("token"))
        .collect();
    assert_eq!(tokens.len(), num("seq_len"));

    let raw = std::fs::read(dir.join("tiny_llama_golden.bin")).expect("golden logits");
    let golden: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(golden.len(), tokens.len() * config.vocab_size);

    let weights = std::fs::read(dir.join("tiny_llama.safetensors")).expect("fixture weights");

    Fixture {
        config,
        tokens,
        golden,
        weights,
    }
}

fn alloc_kv_cache(config: &LlamaConfig, total_blocks: usize) -> Vec<f32> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    vec![
        0.0_f32;
        config.num_hidden_layers
            * total_blocks
            * BLOCK_SIZE
            * config.num_key_value_heads
            * 2
            * head_dim
    ]
}

/// Largest absolute deviation, plus the number of positions whose argmax moved.
fn compare(actual: &[f32], expected: &[f32], vocab: usize) -> (f32, usize) {
    let mut max_abs = 0.0_f32;
    let mut argmax_mismatch = 0;
    for (a, e) in actual.chunks_exact(vocab).zip(expected.chunks_exact(vocab)) {
        for (x, y) in a.iter().zip(e.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        let pick = |s: &[f32]| {
            s.iter()
                .enumerate()
                .max_by(|(_, l), (_, r)| l.total_cmp(r))
                .map(|(i, _)| i)
                .unwrap()
        };
        if pick(a) != pick(e) {
            argmax_mismatch += 1;
        }
    }
    (max_abs, argmax_mismatch)
}

/// Decode-style: one `forward` call per position, each appending to the KV
/// cache and attending over everything written so far. This is the path the
/// continuous-batching loop takes.
#[test]
fn test_incremental_decode_matches_reference() {
    let fx = load_fixture();
    let loader = ModelLoader::new(&fx.weights).expect("parse fixture safetensors");
    let weights = loader
        .load_weights(&fx.config)
        .expect("load fixture weights");

    let total_blocks = 8;
    let mut allocator = BlockAllocator::new(total_blocks, BLOCK_SIZE);
    let mut kv_cache = alloc_kv_cache(&fx.config, total_blocks);

    let mut block_table = BlockTable::new();
    let blocks_needed = fx.tokens.len().div_ceil(BLOCK_SIZE);
    assert!(blocks_needed >= 3, "fixture should span multiple KV blocks");
    for _ in 0..blocks_needed {
        block_table.append_block(allocator.allocate().unwrap());
    }

    let mut actual = Vec::with_capacity(fx.golden.len());
    for (pos, &token) in fx.tokens.iter().enumerate() {
        let logits = weights.forward(
            token,
            pos,
            &fx.config,
            &block_table,
            &mut kv_cache,
            BLOCK_SIZE,
            None,
        );
        actual.extend_from_slice(&logits);
    }

    let (max_abs, argmax_mismatch) = compare(&actual, &fx.golden, fx.config.vocab_size);
    println!("incremental decode: max|delta|={max_abs:.6}, argmax mismatches={argmax_mismatch}");
    assert_eq!(
        argmax_mismatch, 0,
        "greedy token choice diverged from reference"
    );
    assert!(
        max_abs < 5e-4,
        "logits diverged from reference: max|delta|={max_abs}"
    );
}

/// Prefill: the whole prompt is consumed in one call, and only the final
/// position's logits are produced. Must land on the same answer as decoding
/// the prompt token by token.
#[test]
fn test_prefill_matches_reference() {
    let fx = load_fixture();
    let loader = ModelLoader::new(&fx.weights).expect("parse fixture safetensors");
    let weights = loader
        .load_weights(&fx.config)
        .expect("load fixture weights");

    let total_blocks = 8;
    let mut allocator = BlockAllocator::new(total_blocks, BLOCK_SIZE);
    let mut kv_cache = alloc_kv_cache(&fx.config, total_blocks);

    let mut block_table = BlockTable::new();
    for _ in 0..fx.tokens.len().div_ceil(BLOCK_SIZE) {
        block_table.append_block(allocator.allocate().unwrap());
    }

    let logits = weights.prefill(
        &fx.tokens,
        0,
        &fx.config,
        &block_table,
        &mut kv_cache,
        BLOCK_SIZE,
        None,
    );

    let vocab = fx.config.vocab_size;
    let expected = &fx.golden[(fx.tokens.len() - 1) * vocab..];
    let (max_abs, argmax_mismatch) = compare(&logits, expected, vocab);
    println!("prefill: max|delta|={max_abs:.6}");
    assert_eq!(
        argmax_mismatch, 0,
        "greedy token choice diverged from reference"
    );
    assert!(
        max_abs < 5e-4,
        "prefill logits diverged from reference: max|delta|={max_abs}"
    );
}

/// Resuming prefill at a non-zero position must reproduce the same KV state.
/// This is what prefix-cache reuse relies on: replaying only the uncached
/// suffix has to be indistinguishable from prefilling the whole prompt.
#[test]
fn test_split_prefill_matches_whole_prefill() {
    let fx = load_fixture();
    let loader = ModelLoader::new(&fx.weights).expect("parse fixture safetensors");
    let weights = loader
        .load_weights(&fx.config)
        .expect("load fixture weights");

    let total_blocks = 8;
    let split = 16; // a block boundary, as the prefix cache would produce

    let run = |chunks: &[&[u32]]| -> Vec<f32> {
        let mut allocator = BlockAllocator::new(total_blocks, BLOCK_SIZE);
        let mut kv_cache = alloc_kv_cache(&fx.config, total_blocks);
        let mut block_table = BlockTable::new();
        for _ in 0..fx.tokens.len().div_ceil(BLOCK_SIZE) {
            block_table.append_block(allocator.allocate().unwrap());
        }
        let mut start = 0;
        let mut logits = Vec::new();
        for chunk in chunks {
            logits = weights.prefill(
                chunk,
                start,
                &fx.config,
                &block_table,
                &mut kv_cache,
                BLOCK_SIZE,
                None,
            );
            start += chunk.len();
        }
        logits
    };

    let whole = run(&[&fx.tokens]);
    let split_run = run(&[&fx.tokens[..split], &fx.tokens[split..]]);

    let (max_abs, _) = compare(&split_run, &whole, fx.config.vocab_size);
    assert!(
        max_abs < 1e-5,
        "resuming prefill mid-prompt changed the result: max|delta|={max_abs}"
    );
}
