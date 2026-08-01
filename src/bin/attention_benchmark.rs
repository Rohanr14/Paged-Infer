//! Paged attention on its own, with the rest of the model taken away.
//!
//! End-to-end throughput cannot answer the question this benchmark exists for.
//! A decode step is dominated by weight streaming, so an attention change shows
//! up there diluted by an order of magnitude and tangled with whatever else
//! moved. Here there are no weights: just a KV cache, a query, and the kernel.
//!
//! What it sweeps is `heads_per_lane` — how many query heads one task owns.
//! Under grouped-query attention `kv_group` query heads share one key/value
//! head, so a lane covering the whole group reads each K and V vector once,
//! while `heads_per_lane = 1` gives every query head its own task and every
//! task its own pass over the same bytes. That is a `kv_group`-fold difference
//! in cache traffic for identical arithmetic, and the columns below are the
//! price of it.
//!
//! The two ends of the sweep must agree to the last bit, and the benchmark
//! checks that on every run: this is a scheduling change, not a numerical one.

use std::time::Instant;

use paged_infer::attention::{AttnEntry, PagedAttention};
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::LlamaConfig;
use paged_infer::simd;

const BLOCK_SIZE: usize = 16;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Bench {
    config: LlamaConfig,
    kv_cache: Vec<f32>,
    tables: Vec<BlockTable>,
    q: Vec<f32>,
    out: Vec<f32>,
    scores: Vec<f32>,
    context: usize,
}

impl Bench {
    fn new(config: LlamaConfig, batch: usize, context: usize) -> Self {
        let blocks_per_seq = (context + 1).div_ceil(BLOCK_SIZE);
        let total_blocks = blocks_per_seq * batch;
        let layout = config.kv_layout(total_blocks, BLOCK_SIZE);
        let mut allocator = BlockAllocator::new(total_blocks, BLOCK_SIZE);

        // Scatter the blocks: a sequence whose blocks happen to be consecutive
        // would flatter the paged path by turning a gather into a linear scan,
        // which is exactly what paging does not give you in a real run.
        let mut handed: Vec<_> = (0..total_blocks)
            .map(|_| allocator.allocate().expect("underprovisioned"))
            .collect();
        let mut seed = 0x9E3779B9u64;
        for i in (1..handed.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            handed.swap(i, (seed >> 33) as usize % (i + 1));
        }
        let tables = handed
            .chunks(blocks_per_seq)
            .map(|chunk| {
                let mut t = BlockTable::new();
                for &b in chunk {
                    t.append_block(b);
                }
                t
            })
            .collect();

        let hidden = config.hidden_size;
        let mut kv_cache = vec![0.0f32; layout.total_floats()];
        for (i, slot) in kv_cache.iter_mut().enumerate() {
            *slot = ((i % 257) as f32) * 1e-3 - 0.12;
        }
        let q = (0..batch * hidden)
            .map(|i| ((i % 193) as f32) * 1e-3 - 0.09)
            .collect();

        Self {
            kv_cache,
            tables,
            q,
            out: vec![0.0; batch * hidden],
            scores: vec![0.0; batch * config.num_attention_heads * (context + 1)],
            config,
            context,
        }
    }

    fn entries(&self) -> Vec<AttnEntry<'_>> {
        self.tables
            .iter()
            .map(|t| AttnEntry::new(t, self.context, None))
            .collect()
    }

    fn kernel(&self, heads_per_lane: usize) -> PagedAttention {
        PagedAttention {
            layout: self
                .config
                .kv_layout_for_cache(self.kv_cache.len(), BLOCK_SIZE),
            block_size: BLOCK_SIZE,
            num_heads: self.config.num_attention_heads,
            head_dim: self.config.head_dim(),
            kv_group: self.config.kv_group(),
            score_stride: self.context + 1,
            heads_per_lane,
        }
    }

    /// Fastest of `reps` runs. Interference is one-sided, so the minimum is the
    /// best estimate of the cost with the noise taken out.
    fn time(&mut self, heads_per_lane: usize, reps: usize) -> (f64, Vec<f32>) {
        let attn = self.kernel(heads_per_lane);
        let entries: Vec<AttnEntry<'_>> = self
            .tables
            .iter()
            .map(|t| AttnEntry::new(t, self.context, None))
            .collect();

        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let start = Instant::now();
            attn.run(
                &mut self.out,
                &mut self.scores,
                &self.q,
                &self.kv_cache,
                &entries,
                0,
            );
            best = best.min(start.elapsed().as_secs_f64());
        }
        (best, self.out.clone())
    }
}

fn main() {
    let contexts: Vec<usize> = std::env::var("ATTN_CONTEXTS")
        .unwrap_or_else(|_| "256 1024 4096".to_string())
        .split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect();
    let batches: Vec<usize> = std::env::var("ATTN_BATCHES")
        .unwrap_or_else(|_| "1 4 8".to_string())
        .split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect();
    let reps = env_usize("ATTN_REPS", 20);

    let config = LlamaConfig {
        // One layer: this benchmark measures the kernel, not the model.
        num_hidden_layers: 1,
        ..LlamaConfig::default()
    };
    let kv_group = config.kv_group();
    let widths: Vec<usize> = (1..=kv_group)
        .filter(|d| kv_group.is_multiple_of(*d))
        .collect();

    println!("Paged attention kernel");
    println!("======================");
    println!(
        "shape      : heads={} kv_heads={} head_dim={} kv_group={}",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim(),
        kv_group
    );
    println!("simd       : {}", simd::backend());
    println!("threads    : {}", rayon::current_num_threads());
    println!("timing     : fastest of {reps} runs");
    println!();
    println!("heads/lane is how many query heads one task owns. 1 gives every query");
    println!("head its own task -- and its own pass over the same K and V, since the");
    println!("group shares them. {kv_group} reads each K and V exactly once.");
    println!();

    for &context in &contexts {
        println!("-- context {context}");
        print!("   {:<7} {:>10}", "batch", "KV MB");
        for d in &widths {
            print!("{:>12}", format!("d={d}"));
        }
        println!("{:>10}", "best");

        for &batch in &batches {
            let mut bench = Bench::new(config.clone(), batch, context);
            // Bytes of K and V one step must read if each is read once.
            let kv_mb =
                (batch * config.num_key_value_heads * 2 * config.head_dim() * (context + 1) * 4)
                    as f64
                    / 1_048_576.0;

            let mut times = Vec::new();
            let mut reference: Option<Vec<f32>> = None;
            for &d in &widths {
                let (secs, out) = bench.time(d, reps);
                match &reference {
                    None => reference = Some(out),
                    Some(r) => assert_eq!(
                        *r, out,
                        "heads_per_lane={d} changed the output -- this must be a \
                         scheduling change only, at context {context} batch {batch}"
                    ),
                }
                times.push(secs);
            }

            let slowest = times[0];
            print!("   {:<7} {:>10.1}", batch, kv_mb);
            for t in &times {
                print!("{:>11.2}x", slowest / t);
            }
            let best = times
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| widths[i])
                .unwrap();
            println!("{:>10}", format!("d={best}"));
            let _ = bench.entries();
        }
        println!();
    }

    println!("Every column is the same arithmetic in the same order -- the benchmark");
    println!("asserts the outputs are bit-identical across the sweep before printing a");
    println!("single number. What changes is only how many tasks share a KV read, so a");
    println!("speedup here is cache traffic removed, not work removed.");
    println!();
    println!("Wider is not always better: a lane covering the whole group reads the");
    println!("least, but produces the fewest tasks, and a schedule with fewer tasks");
    println!("than threads leaves cores idle. That is why the engine picks the width");
    println!("from the batch size rather than hardcoding one.");
}
