//! Benchmarks a full TinyLlama-scale forward pass on CPU vs GPU.
//!
//! Synthetic weights (constant values) are used so no model file is required.
//! Every projection matrix runs through the same code paths as real inference:
//! wq/wk/wv → RoPE → KV-cache → attention → wo → FFN (w1/w3/SwiGLU/w2) → lm_head.
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

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_packed(rows: usize, cols: usize, val: f32) -> PackedLinear {
    PackedLinear { rows, cols, weight: vec![val; rows * cols] }
}

/// Build synthetic LlamaWeights at TinyLlama 1.1B dimensions.
/// Returns the weights together with an owned byte buffer that backs the
/// token-embedding Tensor (so the lifetime is satisfied).
fn synthetic_weights(config: &LlamaConfig) -> (Vec<u8>, LlamaWeights<'static>) {
    // Token embeddings: vocab_size × hidden bf16 bytes — all zeros is fine for timing.
    let embed_bytes_count = config.vocab_size * config.hidden_size * 2;
    // We need a 'static slice.  Box::leak gives us that.
    let embed_bytes: &'static [u8] = Box::leak(vec![0u8; embed_bytes_count].into_boxed_slice());
    let token_embeddings = Tensor::new(embed_bytes, vec![config.vocab_size, config.hidden_size]);

    let h   = config.hidden_size;
    let kv  = config.num_key_value_heads * (h / config.num_attention_heads);
    let ff  = config.intermediate_size;

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

    let final_norm = vec![1.0f32; h];
    let lm_head    = make_packed(config.vocab_size, h, 0.001);

    let weights = LlamaWeights { token_embeddings, layers, final_norm, lm_head };
    // Return a dummy vec just to keep the API consistent; the real backing is leaked above.
    (Vec::new(), weights)
}

fn run_forward(
    weights: &LlamaWeights<'_>,
    gpu: Option<&GpuForwardContext>,
    config: &LlamaConfig,
    iters: usize,
) -> std::time::Duration {
    let block_size = 16usize;
    let total_blocks = 64usize;
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_cache_size = config.num_hidden_layers
        * total_blocks
        * block_size
        * config.num_key_value_heads
        * 2
        * head_dim;
    let mut kv_cache = vec![0.0f32; kv_cache_size];

    let mut allocator = BlockAllocator::new(total_blocks, block_size);
    let mut bt = BlockTable::new();
    bt.append_block(allocator.allocate().expect("block allocator empty"));

    let t = Instant::now();
    for i in 0..iters {
        let pos = i % (block_size * total_blocks - 1);
        if pos > 0 && pos % block_size == 0 {
            if let Some(pb) = allocator.allocate() {
                bt.append_block(pb);
            }
        }
        let _ = weights.forward(1u32, pos, config, &bt, &mut kv_cache, block_size, gpu);
    }
    t.elapsed()
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let config = LlamaConfig::default(); // TinyLlama: 2048 hidden, 22 layers, 32K vocab

    println!("GPU Inference Benchmark");
    println!("=======================");
    println!(
        "Model   : TinyLlama 1.1B dimensions (synthetic weights, {} layers, {} hidden, {} vocab)",
        config.num_hidden_layers, config.hidden_size, config.vocab_size
    );
    println!();

    let warmup = 2usize;
    let iters  = 8usize;

    // ── build synthetic weights ───────────────────────────────────────────────
    let (_buf, weights) = synthetic_weights(&config);

    // ── CPU baseline ─────────────────────────────────────────────────────────
    println!("Warming up CPU ({warmup} passes)...");
    run_forward(&weights, None, &config, warmup);
    println!("Timing CPU ({iters} passes)...");
    let cpu_dur = run_forward(&weights, None, &config, iters);
    let cpu_ms  = cpu_dur.as_secs_f64() * 1000.0 / iters as f64;
    println!("CPU  : {:.2} ms/token  ({:.1} tok/s)", cpu_ms, 1000.0 / cpu_ms);
    println!();

    // ── GPU path ─────────────────────────────────────────────────────────────
    println!("Uploading weights to GPU...");
    let t_upload = Instant::now();
    let gpu_ctx = match GpuForwardContext::from_weights(&weights) {
        Some(g) => g,
        None => {
            println!("No GPU adapter found — skipping GPU benchmark.");
            println!("(Run on a machine with Metal / Vulkan / DX12 to see GPU results.)");
            return;
        }
    };
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;
    println!("Weight upload : {upload_ms:.0} ms (one-time cost)");
    println!();

    println!("Warming up GPU ({warmup} passes)...");
    run_forward(&weights, Some(&gpu_ctx), &config, warmup);
    println!("Timing GPU ({iters} passes)...");
    let gpu_dur = run_forward(&weights, Some(&gpu_ctx), &config, iters);
    let gpu_ms  = gpu_dur.as_secs_f64() * 1000.0 / iters as f64;
    println!("GPU  : {:.2} ms/token  ({:.1} tok/s)", gpu_ms, 1000.0 / gpu_ms);
    println!();

    // ── summary ───────────────────────────────────────────────────────────────
    let speedup = cpu_ms / gpu_ms;
    println!("Results");
    println!("-------");
    println!("CPU  : {cpu_ms:.2} ms/token");
    println!("GPU  : {gpu_ms:.2} ms/token");
    println!("Speedup : {speedup:.2}x");
    println!();
    println!("Architecture notes:");
    println!("  All 7 projection matrices per layer ({} layers) + lm_head run on GPU.", config.num_hidden_layers);
    println!("  RMS-norm, RoPE, attention scores, SwiGLU remain on CPU.");
    println!("  Each projection incurs one GPU dispatch + synchronous readback.");
    println!("  On Apple Silicon the readback is a local copy within unified memory.");
    println!();
    println!("  Per-token dispatch count: {} layers × 7 + 1 lm_head = {} dispatches/token.",
        config.num_hidden_layers,
        config.num_hidden_layers * 7 + 1);
    println!("  Production speedup comes from fusing ops and removing per-dispatch");
    println!("  readbacks (e.g. Flash Attention, triton kernels, or full GPU forward).");
}
