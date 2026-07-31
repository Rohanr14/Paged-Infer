//! CLI front end. The scheduler itself lives in `paged_infer::engine`.

use anyhow::Result;
use memmap2::MmapOptions;
use paged_infer::engine::{Engine, EngineConfig};
use paged_infer::model::{LlamaConfig, ModelLoader, Quantization};
use paged_infer::simd;
use std::fs::File;
use std::time::Instant;
use tokenizers::Tokenizer;

const SYSTEM: &str = "You are a systems engineer who explains GPU inference clearly and concisely. Answer the question directly.";

fn main() -> Result<()> {
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/tokenizer.json".to_string());
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/model.safetensors".to_string());

    if !std::path::Path::new(&tokenizer_path).exists()
        || !std::path::Path::new(&model_path).exists()
    {
        println!("Model weights not found.");
        println!("  MODEL_PATH     = {model_path}");
        println!("  TOKENIZER_PATH = {tokenizer_path}");
        println!("Fetch them with: python3 scripts/download_model.py");
        println!();
        println!("Everything that does not need weights still runs:");
        println!("  cargo test                                  # includes golden-parity vs a reference model");
        println!("  cargo run --release --bin benchmark          # kernel attribution ladder");
        println!("  cargo run --release --bin prefix_cache_benchmark");
        return Ok(());
    }

    // Memory-map the weights; safetensors views point straight into the mapping.
    println!("Opening {model_path}...");
    let file = File::open(&model_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let loader = ModelLoader::new(&mmap)?;
    // QUANT=int8 stores projections as per-row int8, about 4x less weight
    // memory. A matvec is memory-bound, so that is a throughput win too.
    let quantization = match std::env::var("QUANT").as_deref() {
        Ok("int8") => Quantization::Int8,
        _ => Quantization::F32,
    };
    let config = LlamaConfig {
        quantization,
        ..LlamaConfig::beside_checkpoint(&model_path)
    };
    let weights = loader.load_weights(&config)?;
    println!(
        "Mapped {} layers, {:.2} GB of {:?} weights ({:.2}x vs f32). SIMD backend: {}.",
        config.num_hidden_layers,
        weights.weight_bytes() as f64 / 1e9,
        quantization,
        weights.weight_bytes_f32() as f64 / weights.weight_bytes() as f64,
        simd::backend()
    );

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    let engine_config = EngineConfig {
        total_blocks: 512,
        block_size: 16,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(weights, config, engine_config).with_tokenizer(tokenizer);
    println!(
        "KV cache: {:.2} MB across {} blocks of 16 tokens.\n",
        engine.kv_cache_bytes() as f64 / 1_048_576.0,
        512
    );

    // Two requests share a system prompt: the first pays to prefill it, the
    // second maps the same KV blocks and prefills only its own question. The
    // third draws four continuations from one prompt, forking rather than
    // re-prefilling.
    engine.submit(
        &format!("{SYSTEM}\n\nQuestion: What problem does PagedAttention solve?"),
        32,
        1,
    )?;
    engine.submit(
        &format!("{SYSTEM}\n\nQuestion: Why does continuous batching improve throughput?"),
        32,
        1,
    )?;
    engine.submit(
        &format!("{SYSTEM}\n\nQuestion: Describe grouped-query attention."),
        32,
        4,
    )?;

    let start = Instant::now();
    let completions = engine.run()?;
    let elapsed = start.elapsed();

    for c in &completions {
        let text = engine.decode_text(&c.tokens).unwrap_or_default();
        println!(
            "[req {} / seq {}] {} tokens, {:?}, ttft {:.2?}\n  {}\n",
            c.request_id,
            c.sequence_id,
            c.tokens.len(),
            c.finish_reason,
            c.time_to_first_token,
            text.trim()
        );
    }

    let s = engine.stats();
    let prefix = engine.prefix_stats();
    println!(
        "Completed {} sequences in {elapsed:.2?} over {} steps.",
        completions.len(),
        s.steps
    );
    println!(
        "  prompt tokens     : {} total, {} prefilled, {} reused from cache",
        s.prompt_tokens,
        s.prompt_tokens_prefilled,
        s.prompt_tokens_reused()
    );
    println!("  generated tokens  : {}", s.generated_tokens);
    println!(
        "  prefix cache      : {}/{} block lookups hit ({:.1}%)",
        prefix.hits,
        prefix.hits + prefix.misses,
        prefix.hit_rate() * 100.0
    );
    println!("  copy-on-write     : {} block copies", engine.cow_copies());
    println!(
        "  prefill {:.2?} / decode {:.2?} ({:.2} tok/s)",
        s.prefill_time,
        s.decode_time,
        s.decode_tokens_per_second()
    );

    Ok(())
}
