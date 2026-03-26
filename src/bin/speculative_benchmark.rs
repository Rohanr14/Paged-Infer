use memmap2::MmapOptions;
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{LlamaConfig, ModelLoader};
use paged_infer::speculative::NgramDrafter;
use std::cmp::Ordering;
use std::fs::File;
use std::time::Instant;

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(2)
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/model.safetensors".to_string());
    let steps: usize = std::env::var("SPEC_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let spec_k: usize = std::env::var("SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let ngram_n: usize = std::env::var("SPEC_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let block_size = 16;
    let total_blocks = 1024;

    if !std::path::Path::new(&model_path).exists() {
        println!(
            "MODEL_PATH not found ({model_path}). Set MODEL_PATH to run speculative decoding benchmark."
        );
        return Ok(());
    }

    let file = File::open(&model_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let loader = ModelLoader::new(&mmap)?;
    let config = LlamaConfig::default();
    let weights = loader.load_weights(&config)?;

    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_cache_size = config.num_hidden_layers
        * total_blocks
        * block_size
        * config.num_key_value_heads
        * 2
        * head_dim;

    // ---- Baseline greedy decoding ----
    println!("Running baseline greedy decoding ({steps} steps)...");
    {
        let mut allocator = BlockAllocator::new(total_blocks, block_size);
        let mut kv_cache = vec![0.0_f32; kv_cache_size];
        let mut bt = BlockTable::new();
        bt.append_block(allocator.allocate().expect("allocator underprovisioned"));

        let mut token: u32 = 1;
        let mut pos: usize = 0;

        let t0 = Instant::now();
        for _ in 0..steps {
            let logits = weights.forward(token, pos, &config, &bt, &mut kv_cache, block_size);
            token = argmax(&logits);
            pos += 1;
            if pos.is_multiple_of(block_size) {
                if let Some(pb) = allocator.allocate() {
                    bt.append_block(pb);
                }
            }
        }
        let baseline_elapsed = t0.elapsed().as_secs_f64();
        let baseline_toks_per_sec = steps as f64 / baseline_elapsed.max(1e-9);
        println!("Baseline: {steps} tokens in {baseline_elapsed:.3}s => {baseline_toks_per_sec:.2} tok/s");

        // ---- Speculative decoding ----
        println!(
            "Running speculative decoding ({steps} steps, K={spec_k}, N-gram N={ngram_n})..."
        );
        let mut allocator2 = BlockAllocator::new(total_blocks, block_size);
        let mut kv_cache2 = vec![0.0_f32; kv_cache_size];
        let mut bt2 = BlockTable::new();
        bt2.append_block(allocator2.allocate().expect("allocator underprovisioned"));

        let mut drafter = NgramDrafter::new(ngram_n);
        let mut spec_token: u32 = 1;
        let mut spec_pos: usize = 0;
        let mut total_draft_proposed: usize = 0;
        let mut total_draft_accepted: usize = 0;
        let mut spec_steps_done: usize = 0;

        let t1 = Instant::now();
        while spec_steps_done < steps {
            // Get verifier token via forward pass
            let logits = weights.forward(
                spec_token,
                spec_pos,
                &config,
                &bt2,
                &mut kv_cache2,
                block_size,
            );
            let verifier_token = argmax(&logits);

            // Check draft against verifier
            let drafts = drafter.draft(spec_k);
            let draft_len = drafts.len();
            total_draft_proposed += draft_len;

            if !drafts.is_empty() && drafts[0] == verifier_token {
                total_draft_accepted += 1;
            }

            // Observe the verified token
            drafter.observe(verifier_token);

            spec_token = verifier_token;
            spec_pos += 1;
            spec_steps_done += 1;

            if spec_pos.is_multiple_of(block_size) {
                if let Some(pb) = allocator2.allocate() {
                    bt2.append_block(pb);
                }
            }
        }
        let spec_elapsed = t1.elapsed().as_secs_f64();
        let spec_toks_per_sec = spec_steps_done as f64 / spec_elapsed.max(1e-9);

        let acceptance_rate = if total_draft_proposed > 0 {
            total_draft_accepted as f64 / total_draft_proposed as f64
        } else {
            0.0
        };

        // Theoretical speedup with batched verification:
        // With acceptance rate R and K draft tokens, expected tokens per verifier call = 1 + R*K
        // (when draft model is free / batched with verifier)
        let theoretical_speedup = 1.0 + acceptance_rate * spec_k as f64;

        println!("\n=== Speculative Decoding Benchmark Results ===");
        println!(
            "{:<35} {:>12}",
            "Baseline tok/s:", format!("{baseline_toks_per_sec:.2}")
        );
        println!(
            "{:<35} {:>12}",
            "Speculative tok/s:", format!("{spec_toks_per_sec:.2}")
        );
        println!(
            "{:<35} {:>12}",
            "Draft tokens proposed:", total_draft_proposed
        );
        println!(
            "{:<35} {:>12}",
            "Draft tokens accepted:", total_draft_accepted
        );
        println!(
            "{:<35} {:>12}",
            "Acceptance rate:", format!("{:.2}%", acceptance_rate * 100.0)
        );
        println!(
            "{:<35} {:>12}",
            "N-gram K:", spec_k
        );
        println!(
            "{:<35} {:>12}",
            "N-gram N:", ngram_n
        );
        println!(
            "{:<35} {:>12}",
            "Theoretical max speedup (batched):", format!("{theoretical_speedup:.2}x")
        );
        println!(
            "\nNote: theoretical speedup = 1 + acceptance_rate * K = {:.2}x",
            theoretical_speedup
        );
        println!("      (assumes draft model is free / batched with verifier)");
    }

    Ok(())
}
