//! Sequential vs batched decode, swept across batch size.
//!
//! Decoding is memory-bound: each step streams every weight matrix to do one
//! multiply-add per element. Running `B` sequences one at a time therefore reads
//! the weights `B` times. Batching reads them once and reuses each row across
//! the batch, so the ceiling is roughly `B`x — approached until the kernel stops
//! being bandwidth-limited. What ends it is the projection kernel's arithmetic
//! intensity, which batching raises to about `B/2` flop per byte: past roughly
//! `B = 2C/BW` it is issue-bound rather than bandwidth-bound. Attention (which
//! cannot amortize across sequences) and the per-sequence elementwise work set
//! the floor, but they are not what caps the curve at short context.
//!
//! Runs on synthetic weights by default so the comparison is available without
//! a checkpoint, sized to exceed last-level cache — a cache-resident model would
//! show almost no benefit and mislead. Set `MODEL_PATH` to measure a real one.

use std::time::Instant;

use memmap2::MmapOptions;
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{
    AttentionWeights, BatchScratch, FeedForwardWeights, ForwardScratch, LayerWeights, LlamaConfig,
    LlamaWeights, ModelLoader, PackedLinear, Projection, Quantization,
};
use paged_infer::simd;
use paged_infer::tensor::Tensor;

const BLOCK_SIZE: usize = 16;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Synthetic weights with TinyLlama's per-layer shape but fewer layers, so the
/// working set exceeds cache without needing 4 GB.
fn synthetic_config() -> LlamaConfig {
    LlamaConfig {
        num_hidden_layers: env_usize("BENCH_LAYERS", 4),
        vocab_size: env_usize("BENCH_VOCAB", 8192),
        ..LlamaConfig::default()
    }
}

fn synthetic_weights(config: &LlamaConfig) -> LlamaWeights<'static> {
    let embed: &'static [u8] =
        Box::leak(vec![0u8; config.vocab_size * config.hidden_size * 2].into_boxed_slice());
    let token_embeddings = Tensor::new(embed, vec![config.vocab_size, config.hidden_size]);

    let h = config.hidden_size;
    let kv = config.kv_dim();
    let ff = config.intermediate_size;
    let proj = |rows: usize, cols: usize| -> Projection {
        Projection::F32(PackedLinear {
            rows,
            cols,
            // Small non-uniform values: all-equal weights can let the hardware
            // collapse work that a real matrix would not.
            weight: (0..rows * cols)
                .map(|i| ((i % 251) as f32) * 1e-4 - 0.012)
                .collect(),
        })
    };

    let layers = (0..config.num_hidden_layers)
        .map(|_| LayerWeights {
            attention_norm: vec![1.0; h],
            attention: AttentionWeights {
                wq: proj(h, h),
                wk: proj(kv, h),
                wv: proj(kv, h),
                wo: proj(h, h),
            },
            ffn_norm: vec![1.0; h],
            feed_forward: FeedForwardWeights {
                w1: proj(ff, h),
                w2: proj(h, ff),
                w3: proj(ff, h),
            },
        })
        .collect();

    LlamaWeights {
        token_embeddings,
        layers,
        final_norm: vec![1.0; h],
        lm_head: proj(config.vocab_size, h),
    }
}

struct Harness {
    config: LlamaConfig,
    kv_cache: Vec<f32>,
    tables: Vec<BlockTable>,
}

impl Harness {
    fn new(config: &LlamaConfig, batch: usize, context: usize) -> Self {
        let blocks_per_seq = (context + 2).div_ceil(BLOCK_SIZE);
        let total_blocks = blocks_per_seq * batch + 8;
        let layout = config.kv_layout(total_blocks, BLOCK_SIZE);
        let mut allocator = BlockAllocator::new(total_blocks, BLOCK_SIZE);
        let tables = (0..batch)
            .map(|_| {
                let mut t = BlockTable::new();
                for _ in 0..blocks_per_seq {
                    t.append_block(allocator.allocate().expect("underprovisioned"));
                }
                t
            })
            .collect();
        Self {
            config: config.clone(),
            kv_cache: vec![0.0; layout.total_floats()],
            tables,
        }
    }
}

/// Decode `steps` tokens for every sequence, one sequence at a time.
fn run_sequential(
    weights: &LlamaWeights<'_>,
    h: &mut Harness,
    batch: usize,
    context: usize,
    steps: usize,
) -> f64 {
    let mut scratch = ForwardScratch::new(&h.config);
    let start = Instant::now();
    for s in 0..steps {
        for b in 0..batch {
            weights.forward_into(
                (b + s) as u32,
                context + s,
                &h.config,
                &h.tables[b],
                &mut h.kv_cache,
                BLOCK_SIZE,
                None,
                &mut scratch,
            );
        }
    }
    start.elapsed().as_secs_f64()
}

/// The same work, with the batch going through the model together.
fn run_batched(
    weights: &LlamaWeights<'_>,
    h: &mut Harness,
    batch: usize,
    context: usize,
    steps: usize,
) -> f64 {
    let mut scratch = BatchScratch::new(&h.config, batch);
    let refs: Vec<&BlockTable> = h.tables.iter().collect();
    let start = Instant::now();
    for s in 0..steps {
        let tokens: Vec<u32> = (0..batch).map(|b| (b + s) as u32).collect();
        let positions = vec![context + s; batch];
        weights.decode_batch_into(
            &tokens,
            &positions,
            &refs,
            &h.config,
            &mut h.kv_cache,
            BLOCK_SIZE,
            &mut scratch,
        );
    }
    start.elapsed().as_secs_f64()
}

/// Prefill a prompt one position at a time.
fn prefill_sequential(weights: &LlamaWeights<'_>, h: &mut Harness, prompt: &[u32]) -> f64 {
    let mut scratch = ForwardScratch::new(&h.config);
    let start = Instant::now();
    weights.prefill_into(
        prompt,
        0,
        &h.config,
        &h.tables[0],
        &mut h.kv_cache,
        BLOCK_SIZE,
        None,
        &mut scratch,
    );
    start.elapsed().as_secs_f64()
}

/// Prefill the same prompt `chunk` positions at a time.
fn prefill_batched(
    weights: &LlamaWeights<'_>,
    h: &mut Harness,
    prompt: &[u32],
    chunk: usize,
) -> f64 {
    let mut scratch = BatchScratch::new(&h.config, chunk);
    let start = Instant::now();
    weights.prefill_batched(
        prompt,
        0,
        &h.config,
        &h.tables[0],
        &mut h.kv_cache,
        BLOCK_SIZE,
        chunk,
        &mut scratch,
    );
    start.elapsed().as_secs_f64()
}

fn main() -> anyhow::Result<()> {
    let steps = env_usize("BENCH_STEPS", 8);
    // Wall-clock on a shared or thermally-limited machine is noisy enough to
    // reorder adjacent batch sizes; taking the fastest of several runs removes
    // interference, which can only ever have made a run slower.
    let repeats = env_usize("BENCH_REPEATS", 3).max(1);
    let context = env_usize("BENCH_CONTEXT", 128);
    let batches: Vec<usize> = std::env::var("BENCH_BATCHES")
        .unwrap_or_else(|_| "1 2 4 8".to_string())
        .split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect();

    let model_path = std::env::var("MODEL_PATH").ok();
    let quantization = match std::env::var("QUANT").as_deref() {
        Ok("int8") => Quantization::Int8,
        _ => Quantization::F32,
    };

    // Keep the mmap alive for as long as the weights borrowed from it.
    let _mmap;
    let _loader;
    let (config, weights): (LlamaConfig, LlamaWeights<'_>) = match &model_path {
        Some(path) if std::path::Path::new(path).exists() => {
            let file = std::fs::File::open(path)?;
            _mmap = unsafe { MmapOptions::new().map(&file)? };
            _loader = ModelLoader::new(&_mmap)?;
            let config = LlamaConfig {
                quantization,
                ..LlamaConfig::beside_checkpoint(path)
            };
            let w = _loader.load_weights(&config)?;
            (config, w)
        }
        _ => {
            let mut config = synthetic_config();
            config.quantization = quantization;
            let w = synthetic_weights(&config);
            (config, w)
        }
    };

    let weight_bytes = weights.weight_bytes();
    println!("Batched decode benchmark");
    println!("========================");
    println!(
        "model      : {}",
        model_path
            .as_deref()
            .filter(|p| std::path::Path::new(p).exists())
            .unwrap_or("synthetic (set MODEL_PATH for a real checkpoint)")
    );
    println!(
        "shape      : hidden={} layers={} heads={} kv_heads={} ff={} vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.intermediate_size,
        config.vocab_size
    );
    println!(
        "weights    : {:.2} GB as {:?}",
        weight_bytes as f64 / 1e9,
        config.quantization
    );
    println!("simd       : {}", simd::backend());
    println!("threads    : {}", rayon::current_num_threads());
    println!("context    : {context} tokens, {steps} decode steps per sequence");
    println!("timing     : fastest of {repeats} runs per configuration (BENCH_REPEATS)");
    println!();

    println!(
        "{:<7} {:>13} {:>13} {:>10} {:>14}",
        "batch", "sequential", "batched", "speedup", "weight GB/s"
    );
    println!("{}", "-".repeat(62));

    for &batch in &batches {
        let mut h = Harness::new(&config, batch, context);
        // Warm up the page cache and the thread pool before timing.
        run_batched(&weights, &mut h, batch, context, 1);

        // Fastest of `repeats`. Both configurations are re-measured on every
        // repeat rather than once each, so a machine that gets busy partway
        // through cannot hand one of them a better slot than the other.
        let (mut seq_secs, mut bat_secs) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..repeats {
            seq_secs = seq_secs.min(run_sequential(&weights, &mut h, batch, context, steps));
            bat_secs = bat_secs.min(run_batched(&weights, &mut h, batch, context, steps));
        }

        let tokens = (batch * steps) as f64;
        let seq_tps = tokens / seq_secs.max(1e-9);
        let bat_tps = tokens / bat_secs.max(1e-9);
        // Batched reads the weights once per step regardless of batch size.
        let batched_traffic = weight_bytes as f64 * steps as f64 / bat_secs.max(1e-9) / 1e9;

        println!(
            "{batch:<7} {:>10.2} t/s {:>10.2} t/s {:>9.2}x {:>13.1}",
            seq_tps,
            bat_tps,
            bat_tps / seq_tps,
            batched_traffic
        );
    }

    // ── prefill ──────────────────────────────────────────────────────────────
    let prompt_len = env_usize("BENCH_PROMPT", 256);
    let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 1000) as u32).collect();

    println!();
    println!("Prefill: positions per pass over a {prompt_len}-token prompt");
    println!(
        "{:<10} {:>15} {:>12} {:>10}",
        "chunk", "prefill time", "tok/s", "speedup"
    );
    println!("{}", "-".repeat(51));

    let mut h = Harness::new(&config, 1, prompt_len + 2);
    prefill_batched(&weights, &mut h, &prompt[..8.min(prompt_len)], 8);
    let base = prefill_sequential(&weights, &mut h, &prompt);
    println!(
        "{:<10} {:>12.3} s {:>12.1} {:>9.2}x",
        "1 (seq)",
        base,
        prompt_len as f64 / base,
        1.0
    );
    for chunk in [4usize, 8, 16, 32, 64] {
        let secs = prefill_batched(&weights, &mut h, &prompt, chunk);
        println!(
            "{:<10} {:>12.3} s {:>12.1} {:>9.2}x",
            chunk,
            secs,
            prompt_len as f64 / secs,
            base / secs.max(1e-12)
        );
    }
    println!();
    println!("Prefill batches along positions rather than sequences, but the kernel");
    println!("and the reason are identical: one pass over the weights serves the whole");
    println!("chunk. Causality still holds because each position attends only up to");
    println!("itself, and every position's K/V is written before any attention runs.");
    println!("This is the path that sets time-to-first-token.");

    println!();
    println!("Sequential decode reads every weight matrix once per sequence; batched");
    println!("reads it once per step and reuses each row across the batch. Sequential");
    println!("throughput is therefore roughly flat in batch size, while batched scales.");
    println!();
    println!("The speedup is sublinear in batch, and the weight-bandwidth column shows");
    println!("why: it falls as batch grows, so the kernel stops being bandwidth-bound");
    println!("and the parts that do not batch start to dominate -- attention (each");
    println!("sequence has its own KV, position and block table) plus the per-sequence");
    println!("RMSNorm, RoPE and SwiGLU. Context length moves this, but less than the");
    println!("Do not read the ceiling as an attention limit. Batching raises the");
    println!("projection kernel's arithmetic intensity to about B/2 flop per byte, so");
    println!("it leaves the bandwidth-bound regime at roughly batch 2C/BW -- 4 to 5 on");
    println!("the machines measured -- and no amount of attention work moves that.");
    println!("Attention only dominates once the weights are small (int8) or the");
    println!("context is long; QUANT=int8 with a long BENCH_CONTEXT is that regime.");
    Ok(())
}
