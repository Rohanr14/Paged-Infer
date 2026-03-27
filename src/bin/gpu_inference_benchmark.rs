//! Benchmarks a full forward pass on CPU vs GPU using synthetic weights.
//!
//! Uses a 4-layer, 8192-vocab config that fits comfortably in GPU memory
//! (~700 MB weight buffers) while exercising the exact same code paths as
//! full TinyLlama inference.  Per-layer timing extrapolates to the 22-layer model.
//!
//! Run with:
//!   cargo run --release --bin gpu_inference_benchmark
//!
//! On Apple Silicon the Metal backend is selected automatically.
//! If no GPU is available the GPU section is skipped and only CPU numbers are shown.

use std::time::Instant;

use paged_infer::{
    memory::{allocator::BlockAllocator, block_table::BlockTable},
    model::{
        AttentionWeights, FeedForwardWeights, GpuForwardContext, LayerWeights, LlamaConfig,
        LlamaWeights, PackedLinear,
    },
    tensor::Tensor,
};

// ── benchmark config ──────────────────────────────────────────────────────────

/// A reduced-scale config that keeps the same per-layer arithmetic as TinyLlama
/// (hidden=2048, heads=32, kv_heads=4, ff=5632) but uses only 4 layers and
/// 8192 vocab so the total GPU weight buffers stay under ~700 MB.
fn bench_config() -> LlamaConfig {
    LlamaConfig {
        num_hidden_layers: 4,
        vocab_size: 8192,
        ..LlamaConfig::default()
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_packed(rows: usize, cols: usize, val: f32) -> PackedLinear {
    PackedLinear { rows, cols, weight: vec![val; rows * cols] }
}

/// Build synthetic LlamaWeights for the given config.
/// Leaks the embedding byte buffer to give it a 'static lifetime.
fn synthetic_weights(config: &LlamaConfig) -> LlamaWeights<'static> {
    let embed_bytes: &'static [u8] =
        Box::leak(vec![0u8; config.vocab_size * config.hidden_size * 2].into_boxed_slice());
    let token_embeddings =
        Tensor::new(embed_bytes, vec![config.vocab_size, config.hidden_size]);

    let h  = config.hidden_size;
    let kv = config.num_key_value_heads * (h / config.num_attention_heads);
    let ff = config.intermediate_size;

    let layers = (0..config.num_hidden_layers)
        .map(|_| LayerWeights {
            attention_norm: vec![1.0f32; h],
            attention: AttentionWeights {
                wq: make_packed(h,  h,  0.001),
                wk: make_packed(kv, h,  0.001),
                wv: make_packed(kv, h,  0.001),
                wo: make_packed(h,  h,  0.001),
            },
            ffn_norm: vec![1.0f32; h],
            feed_forward: FeedForwardWeights {
                w1: make_packed(ff, h,  0.001),
                w2: make_packed(h,  ff, 0.001),
                w3: make_packed(ff, h,  0.001),
            },
        })
        .collect();

    LlamaWeights {
        token_embeddings,
        layers,
        final_norm: vec![1.0f32; h],
        lm_head: make_packed(config.vocab_size, h, 0.001),
    }
}

fn run_forward(
    weights: &LlamaWeights<'_>,
    gpu: Option<&GpuForwardContext>,
    config: &LlamaConfig,
    iters: usize,
) -> std::time::Duration {
    let block_size   = 16usize;
    let total_blocks = 64usize;
    let head_dim     = config.hidden_size / config.num_attention_heads;
    let kv_cache_size = config.num_hidden_layers
        * total_blocks * block_size
        * config.num_key_value_heads * 2 * head_dim;
    let mut kv_cache = vec![0.0f32; kv_cache_size];
    let mut allocator = BlockAllocator::new(total_blocks, block_size);
    let mut bt = BlockTable::new();
    bt.append_block(allocator.allocate().expect("block allocator empty"));

    let t = Instant::now();
    for i in 0..iters {
        let pos = i % (block_size * total_blocks - 1);
        if pos > 0 && pos % block_size == 0 {
            if let Some(pb) = allocator.allocate() { bt.append_block(pb); }
        }
        let _ = weights.forward(1u32, pos, config, &bt, &mut kv_cache, block_size, gpu);
    }
    t.elapsed()
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let config = bench_config();
    let full_layers = 22usize; // actual TinyLlama layer count for extrapolation

    println!("GPU Inference Benchmark  (full forward pass, CPU vs GPU)");
    println!("=========================================================");
    println!(
        "Benchmark scale : {} layers / {} vocab / {} hidden  (TinyLlama: {} layers / 32K vocab)",
        config.num_hidden_layers, config.vocab_size, config.hidden_size, full_layers,
    );
    println!("Weights         : synthetic (timing-representative, no model file needed)");
    println!();

    // ── estimate GPU weight footprint ─────────────────────────────────────────
    let h  = config.hidden_size;
    let kv = config.num_key_value_heads * (h / config.num_attention_heads);
    let ff = config.intermediate_size;
    let bytes_per_layer =
        (h * h + kv * h + kv * h + h * h + ff * h + h * ff + ff * h) * 4;
    let total_gpu_mb =
        (config.num_hidden_layers * bytes_per_layer + config.vocab_size * h * 4) / (1024 * 1024);
    println!("GPU weight upload : ~{total_gpu_mb} MB  (full 22-layer f32 would be ~3.9 GB)");
    println!();

    let weights = synthetic_weights(&config);

    let warmup = 2usize;
    let iters  = 8usize;

    // ── CPU baseline ──────────────────────────────────────────────────────────
    println!("Warming up CPU ({warmup} iters)...");
    run_forward(&weights, None, &config, warmup);
    println!("Timing CPU ({iters} iters)...");
    let cpu_ms = run_forward(&weights, None, &config, iters)
        .as_secs_f64() * 1000.0 / iters as f64;
    println!("CPU  : {cpu_ms:.2} ms/token  ({:.1} tok/s)", 1000.0 / cpu_ms);
    println!();

    // ── GPU ───────────────────────────────────────────────────────────────────
    println!("Uploading weights to GPU...");
    let t_upload = Instant::now();
    let gpu_ctx = match GpuForwardContext::from_weights(&weights) {
        Some(g) => g,
        None => {
            println!("No GPU adapter found — skipping GPU benchmark.");
            return;
        }
    };
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;
    println!("Weight upload : {upload_ms:.0} ms  (one-time cost at model load)");
    println!();

    println!("Warming up GPU ({warmup} iters)...");
    run_forward(&weights, Some(&gpu_ctx), &config, warmup);
    println!("Timing GPU ({iters} iters)...");
    let gpu_ms = run_forward(&weights, Some(&gpu_ctx), &config, iters)
        .as_secs_f64() * 1000.0 / iters as f64;
    println!("GPU  : {gpu_ms:.2} ms/token  ({:.1} tok/s)", 1000.0 / gpu_ms);
    println!();

    // ── summary ───────────────────────────────────────────────────────────────
    let speedup = cpu_ms / gpu_ms;
    println!("Results");
    println!("-------");
    println!("CPU  : {cpu_ms:.2} ms/token");
    println!("GPU  : {gpu_ms:.2} ms/token");
    println!("Speedup : {speedup:.2}x");
    println!();

    let dispatches = config.num_hidden_layers * 7 + 1;
    println!("Architecture");
    println!("  {} dispatches/token  ({} layers × 7 projections + 1 lm_head)", dispatches, config.num_hidden_layers);
    println!("  RMS-norm, RoPE, attention, SwiGLU remain on CPU.");
    println!("  Each dispatch: write_buffer → workgroup kernel → copy → map_async → poll.");
    println!("  On Apple Silicon all steps operate in unified physical memory.");
    println!();
    if speedup < 1.0 {
        println!("Note: per-dispatch driver overhead dominates at batch=1 / single-token.");
        println!("Production gains require fused ops (Flash Attention, CUDA graphs,");
        println!("triton kernels) that eliminate per-projection GPU round-trips.");
        println!("The GPU infrastructure here is the foundation for those optimisations.");
    }
}
