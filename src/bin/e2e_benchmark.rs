use memmap2::MmapOptions;
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{ForwardScratch, LlamaConfig, ModelLoader, Quantization};
use paged_infer::simd;
use std::cmp::Ordering;
use std::fs::File;
use std::time::Instant;

fn read_rss_kb() -> Option<usize> {
    // Linux fast path: /proc exposes VmRSS in kB.
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(line) = status.lines().find(|l| l.starts_with("VmRSS:")) {
            if let Some(val) = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<usize>().ok())
            {
                return Some(val);
            }
        }
    }

    // Cross-platform fallback (macOS + Linux): `ps` rss column in kB.
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let rss = String::from_utf8_lossy(&output.stdout);
    rss.trim().parse::<usize>().ok()
}

fn percentile_us(mut data: Vec<u128>, p: f32) -> u128 {
    if data.is_empty() {
        return 0;
    }
    data.sort_unstable();
    let idx = ((data.len() - 1) as f32 * p).round() as usize;
    data[idx]
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/model.safetensors".to_string());
    let steps: usize = std::env::var("BENCH_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let batch_size: usize = std::env::var("BENCH_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    // Unset or 0 means full causal attention, which is what the model
    // specifies. A window is an explicit accuracy-for-latency trade, so it has
    // to be asked for rather than applied by default.
    let attention_window: Option<usize> = std::env::var("BENCH_WINDOW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|w| *w > 0);
    let block_size = 16;
    let total_blocks = 1024;

    if !std::path::Path::new(&model_path).exists() {
        println!(
            "MODEL_PATH not found ({model_path}). Set MODEL_PATH to run real end-to-end benchmark."
        );
        return Ok(());
    }

    let file = File::open(&model_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let loader = ModelLoader::new(&mmap)?;
    // QUANT=int8 stores projections as per-row int8. Worth measuring here
    // rather than only in the kernel benchmark: weight memory dominates a 1B
    // model's footprint, and a matvec is memory-bound so it is a speed knob too.
    let quantization = match std::env::var("QUANT").as_deref() {
        Ok("int8") => Quantization::Int8,
        _ => Quantization::F32,
    };
    let config = LlamaConfig {
        attention_window,
        quantization,
        // Read the architecture from the checkpoint rather than assuming
        // TinyLlama's shape.
        ..LlamaConfig::beside_checkpoint(&model_path)
    };
    let weights = loader.load_weights(&config)?;

    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_cache_size = config.num_hidden_layers
        * total_blocks
        * block_size
        * config.num_key_value_heads
        * 2
        * head_dim;

    let mut allocator = BlockAllocator::new(total_blocks, block_size);
    let mut kv_cache = vec![0.0_f32; kv_cache_size];

    let mut block_tables = Vec::new();
    let mut tokens = vec![1_u32; batch_size];
    let mut positions = vec![0_usize; batch_size];
    for _ in 0..batch_size {
        let mut bt = BlockTable::new();
        bt.append_block(allocator.allocate().expect("allocator underprovisioned"));
        block_tables.push(bt);
    }

    let mut token_lat_us = Vec::with_capacity(batch_size * steps);
    let mut peak_rss_kb = read_rss_kb().unwrap_or(0);

    // Reuse one scratch across every step, as the engine does. Calling the
    // allocating `forward` here would fold ~10 allocations per token -- the
    // largest a vocab-wide logit buffer -- into the measurement.
    let mut scratch = ForwardScratch::new(&config);

    let t0 = Instant::now();
    for _ in 0..steps {
        for i in 0..batch_size {
            let start = Instant::now();
            weights.forward_into(
                tokens[i],
                positions[i],
                &config,
                &block_tables[i],
                &mut kv_cache,
                block_size,
                None,
                &mut scratch,
            );
            token_lat_us.push(start.elapsed().as_micros());

            let next = scratch
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(2);

            tokens[i] = next;
            positions[i] += 1;

            if positions[i].is_multiple_of(block_size) {
                if let Some(pb) = allocator.allocate() {
                    block_tables[i].append_block(pb);
                }
            }
        }

        if let Some(rss) = read_rss_kb() {
            peak_rss_kb = peak_rss_kb.max(rss);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_tokens = (batch_size * steps) as f64;

    println!("E2E decode benchmark");
    println!("===================");
    println!("model      : {model_path}");
    println!(
        "weights    : {:.2} GB as {:?} ({:.2}x vs f32)",
        weights.weight_bytes() as f64 / 1e9,
        quantization,
        weights.weight_bytes_f32() as f64 / weights.weight_bytes().max(1) as f64
    );
    println!(
        "kv cache   : {:.2} GB ({total_blocks} blocks x {block_size} tokens)",
        (kv_cache_size * 4) as f64 / 1e9
    );
    println!("simd       : {}", simd::backend());
    println!("threads    : {}", rayon::current_num_threads());
    println!();
    println!(
        "batch_size={batch_size}, steps={steps}, attention_window={}, total_tokens={}",
        attention_window.map_or("full".to_string(), |w| w.to_string()),
        total_tokens as usize
    );
    println!("throughput_tok_s={:.2}", total_tokens / elapsed.max(1e-9));
    println!(
        "avg_token_latency_ms={:.3}",
        (elapsed * 1000.0) / total_tokens.max(1.0)
    );
    println!(
        "p50_token_latency_us={}",
        percentile_us(token_lat_us.clone(), 0.50)
    );
    println!(
        "p95_token_latency_us={}",
        percentile_us(token_lat_us.clone(), 0.95)
    );
    println!("peak_rss_mb={:.2}", peak_rss_kb as f64 / 1024.0);

    Ok(())
}
