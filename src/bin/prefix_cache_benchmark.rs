//! What automatic prefix caching and copy-on-write forking are worth on a
//! serving-shaped workload.
//!
//! Runs without model weights: the quantities that matter — prompt tokens that
//! must be prefilled, physical blocks allocated, peak residency — are exact
//! properties of the memory manager, not of the arithmetic. That makes this
//! deterministic and runnable in CI.
//!
//! It reports work saved, not seconds. Converting one to the other needs a
//! per-token prefill cost, which is hardware-specific — measure it with
//! `e2e_benchmark` on the machine you care about. What this benchmark
//! establishes is the multiplier that cost gets applied to.

use paged_infer::memory::kv_cache_manager::KvCacheManager;

const BLOCK_SIZE: usize = 16;

/// A synthetic but serving-shaped workload: one system prompt every request
/// carries, a handful of retrieved documents requests draw from, and a unique
/// user turn on the end.
struct Workload {
    system: Vec<u32>,
    documents: Vec<Vec<u32>>,
    requests: usize,
    user_tokens: usize,
}

impl Workload {
    fn new(system_tokens: usize, doc_tokens: usize, docs: usize, requests: usize) -> Self {
        let tok = |base: u32, n: usize| -> Vec<u32> {
            (0..n).map(|i| base + (i as u32 * 7919) % 30_000).collect()
        };
        Self {
            system: tok(1_000, system_tokens),
            documents: (0..docs).map(|d| tok(100_000 + d as u32 * 1_000, doc_tokens)).collect(),
            requests,
            user_tokens: 48,
        }
    }

    fn prompt(&self, i: usize) -> Vec<u32> {
        let mut p = self.system.clone();
        if !self.documents.is_empty() {
            p.extend_from_slice(&self.documents[i % self.documents.len()]);
        }
        // Unique tail, so no two requests are wholly identical.
        p.extend((0..self.user_tokens).map(|t| 900_000 + (i * 131 + t) as u32));
        p
    }

    fn total_prompt_tokens(&self) -> usize {
        (0..self.requests).map(|i| self.prompt(i).len()).sum()
    }
}

#[derive(Debug, Default)]
struct Outcome {
    prefilled_tokens: usize,
    blocks_allocated: usize,
    peak_blocks_in_use: usize,
    hits: u64,
    lookups: u64,
}

/// Admit every request in turn, decode a little, and retire it — the lifecycle
/// a real scheduler drives, minus the arithmetic.
fn simulate(workload: &Workload, total_blocks: usize, prefix_cache: bool) -> Outcome {
    let mut mgr = KvCacheManager::new(total_blocks, BLOCK_SIZE).with_prefix_cache(prefix_cache);
    let mut out = Outcome::default();

    for i in 0..workload.requests {
        let prompt = workload.prompt(i);
        let Some(admission) = mgr.admit(i, &prompt, i as u64) else {
            panic!("request {i} could not be admitted with {total_blocks} blocks");
        };
        // Whatever the cache did not cover still has to run through the model.
        // One token is always replayed to produce logits.
        let resume_at = admission.cached_tokens.min(prompt.len() - 1);
        out.prefilled_tokens += prompt.len() - resume_at;
        out.blocks_allocated += admission.allocated_blocks;

        mgr.publish_prompt_blocks(&admission.block_table);
        out.peak_blocks_in_use = out
            .peak_blocks_in_use
            .max(mgr.total_blocks() - mgr.available_blocks());
        mgr.release_sequence(i);
    }

    let stats = mgr.prefix_stats();
    out.hits = stats.hits;
    out.lookups = stats.hits + stats.misses;
    out
}

/// Blocks needed to draw `samples` continuations from one prompt, with and
/// without copy-on-write sharing.
fn parallel_sampling(prompt_len: usize, samples: usize, total_blocks: usize) -> (usize, usize) {
    let prompt: Vec<u32> = (0..prompt_len).map(|i| 5_000 + i as u32).collect();

    // Sharing: prefill once, fork the rest.
    let mut mgr = KvCacheManager::new(total_blocks, BLOCK_SIZE).with_prefix_cache(false);
    let admission = mgr.admit(0, &prompt, 0).expect("prompt should fit");
    for s in 1..samples {
        mgr.fork(&admission.block_table, s, s as u64);
    }
    let shared = mgr.total_blocks() - mgr.available_blocks();

    // No sharing: every sample carries its own copy of the prompt.
    let mut solo = KvCacheManager::new(total_blocks, BLOCK_SIZE).with_prefix_cache(false);
    for s in 0..samples {
        solo.admit(s, &prompt, s as u64).expect("prompt should fit");
    }
    let independent = solo.total_blocks() - solo.available_blocks();

    (independent, shared)
}

fn pct(saved: usize, of: usize) -> f64 {
    if of == 0 {
        0.0
    } else {
        100.0 * saved as f64 / of as f64
    }
}

fn main() {
    println!("Prefix cache + copy-on-write benchmark");
    println!("======================================");
    println!("block size : {BLOCK_SIZE} tokens");
    println!();

    // ── shared-prefix serving workload ───────────────────────────────────────
    let workload = Workload::new(240, 480, 3, 64);
    let total_blocks = 4096;

    println!("Workload");
    println!("--------");
    println!("  {} requests", workload.requests);
    println!("  {} token system prompt on every request", workload.system.len());
    println!(
        "  {} retrieved documents of {} tokens, round-robin",
        workload.documents.len(),
        workload.documents[0].len()
    );
    println!("  {} unique user tokens per request", workload.user_tokens);
    println!("  {} prompt tokens in total", workload.total_prompt_tokens());
    println!();

    let cold = simulate(&workload, total_blocks, false);
    let warm = simulate(&workload, total_blocks, true);

    println!(
        "{:<26} {:>14} {:>14} {:>10}",
        "metric", "no cache", "prefix cache", "saved"
    );
    println!("{}", "-".repeat(68));
    println!(
        "{:<26} {:>14} {:>14} {:>9.1}%",
        "prompt tokens prefilled",
        cold.prefilled_tokens,
        warm.prefilled_tokens,
        pct(
            cold.prefilled_tokens - warm.prefilled_tokens,
            cold.prefilled_tokens
        )
    );
    println!(
        "{:<26} {:>14} {:>14} {:>9.1}%",
        "KV blocks allocated",
        cold.blocks_allocated,
        warm.blocks_allocated,
        pct(
            cold.blocks_allocated - warm.blocks_allocated,
            cold.blocks_allocated
        )
    );
    println!(
        "{:<26} {:>14} {:>14} {:>10}",
        "peak blocks resident", cold.peak_blocks_in_use, warm.peak_blocks_in_use, "-"
    );
    println!(
        "{:<26} {:>14} {:>13.1}% {:>10}",
        "block cache hit rate",
        "-",
        100.0 * warm.hits as f64 / warm.lookups.max(1) as f64,
        "-"
    );
    println!();
    println!(
        "Prefill is O(prompt tokens), so cutting prefilled tokens by {:.1}% cuts",
        pct(
            cold.prefilled_tokens - warm.prefilled_tokens,
            cold.prefilled_tokens
        )
    );
    println!("time-to-first-token by roughly the same factor for the requests that hit.");
    println!("Peak residency *rises* with the cache: retaining hot prefixes is the");
    println!("point, and the LRU gives those blocks back the moment they are needed.");
    println!();

    // ── how the win scales with prefix sharing ───────────────────────────────
    println!("Sensitivity to how much of the prompt is shared");
    println!("-----------------------------------------------");
    println!(
        "{:<18} {:>14} {:>14} {:>10}",
        "shared prefix", "no cache", "prefix cache", "saved"
    );
    println!("{}", "-".repeat(60));
    for system_tokens in [0, 64, 240, 480, 960] {
        let w = Workload::new(system_tokens, 0, 0, 64);
        let c = simulate(&w, total_blocks, false);
        let h = simulate(&w, total_blocks, true);
        println!(
            "{:<18} {:>14} {:>14} {:>9.1}%",
            format!("{system_tokens} tokens"),
            c.prefilled_tokens,
            h.prefilled_tokens,
            pct(c.prefilled_tokens - h.prefilled_tokens, c.prefilled_tokens)
        );
    }
    println!();
    println!("With no shared prefix there is nothing to reuse and the cache costs");
    println!("nothing but the lookups -- it degrades to the uncached path.");
    println!();

    // ── parallel sampling via copy-on-write ──────────────────────────────────
    println!("Parallel sampling (n continuations from one prompt)");
    println!("---------------------------------------------------");
    println!(
        "{:<10} {:>14} {:>14} {:>10}",
        "samples", "independent", "copy-on-write", "saved"
    );
    println!("{}", "-".repeat(52));
    for samples in [1, 2, 4, 8, 16] {
        let (independent, shared) = parallel_sampling(1024, samples, total_blocks);
        println!(
            "{:<10} {:>14} {:>14} {:>9.1}%",
            samples,
            independent,
            shared,
            pct(independent - shared, independent)
        );
    }
    println!();
    println!("Forking maps the parent's blocks instead of copying them, so n samples");
    println!("cost one prompt's KV plus whatever the samples actually diverge on.");
    println!("Blocks split lazily, on first write -- a sample that stops early never");
    println!("pays for a copy at all.");
}
