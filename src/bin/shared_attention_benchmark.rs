//! Shared-prefix attention in isolation. JSONL retains every timing sample.
//!
//! Candidate timing includes rebuilding its physical-prefix plan. Both paths
//! reuse warmed scratch, and validation, input construction, and reporting are
//! outside the timer. This does not measure full-model throughput.

use anyhow::{ensure, Context, Result};
use paged_infer::attention::{AttnEntry, PagedAttention, SharedPrefixPlan, SharedPrefixScratch};
use paged_infer::memory::allocator::PhysicalBlock;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::LlamaConfig;
use serde_json::{json, Value};
use std::hint::black_box;
use std::io::{self, BufWriter, Write};
use std::process::Command;
use std::time::Instant;

const BLOCK_SIZE: usize = 16;
const HELP: &str = "Shared-prefix attention kernel benchmark; writes JSONL to stdout.
Environment controls (space- or comma-separated lists):
  SHARED_ATTN_BATCHES       Default: 1 4 8 16
  SHARED_ATTN_CONTEXTS      Previous tokens per entry; default: 256 1024 4096
  SHARED_ATTN_PERCENTAGES   Requested shared prefix percentages; default: 0 50 90
  SHARED_ATTN_REPS          Timed pairs per case; default: 9
  SHARED_ATTN_WARMUP        Untimed pairs per case; default: 2
  RAYON_NUM_THREADS        Worker count, following the usual Rayon setting
The shared prefix is rounded down to whole blocks. Each entry also attends to
its current token. Baseline/candidate order alternates by case and repetition.
Every run checks finite, bit-identical output. No measured samples are discarded.";

fn values(key: &str, default: &str, allow_zero: bool) -> Result<Vec<usize>> {
    let value = std::env::var(key).unwrap_or_else(|_| default.to_owned());
    let parsed = value
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .with_context(|| format!("invalid {key}: {s}"))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!parsed.is_empty(), "{key} must contain at least one value");
    ensure!(
        allow_zero || parsed.iter().all(|&n| n > 0),
        "{key} must be positive"
    );
    Ok(parsed)
}

fn count(key: &str, default: &str) -> Result<usize> {
    let parsed = values(key, default, false)?;
    ensure!(parsed.len() == 1, "{key} must be one integer");
    Ok(parsed[0])
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

struct Case {
    attn: PagedAttention,
    tables: Vec<BlockTable>,
    kv: Vec<f32>,
    q: Vec<f32>,
    baseline_out: Vec<f32>,
    candidate_out: Vec<f32>,
    scores: Vec<f32>,
    scratch: SharedPrefixScratch,
    context: usize,
    shared_blocks: usize,
}

impl Case {
    fn new(batch: usize, context: usize, percentage: usize) -> Result<Self> {
        let config = LlamaConfig {
            num_hidden_layers: 1,
            ..LlamaConfig::default()
        };
        let used = context.checked_add(1).context("context overflow")?;
        let blocks = used.div_ceil(BLOCK_SIZE);
        let shared_blocks =
            context.checked_mul(percentage).context("prefix overflow")? / 100 / BLOCK_SIZE;
        let total_blocks = (blocks - shared_blocks)
            .checked_mul(batch)
            .and_then(|n| n.checked_add(shared_blocks))
            .context("block count overflow")?;
        let cache_floats = total_blocks
            .checked_mul(BLOCK_SIZE)
            .and_then(|n| n.checked_mul(config.num_key_value_heads * 2 * config.head_dim()))
            .context("KV size overflow")?;
        let queries = batch
            .checked_mul(config.hidden_size)
            .context("query size overflow")?;
        let score_floats = batch
            .checked_mul(config.num_attention_heads)
            .and_then(|n| n.checked_mul(used))
            .context("score size overflow")?;
        let layout = config.kv_layout(total_blocks, BLOCK_SIZE);

        // Shuffle physical IDs before assigning any mappings. Shared IDs repeat
        // only at matching logical prefix positions; suffix mappings are unique.
        let mut physical: Vec<_> = (0..total_blocks)
            .map(|index| PhysicalBlock { index })
            .collect();
        let mut seed = 0xc017_cafe_58a9_120bu64;
        for i in (1..physical.len()).rev() {
            let other = (random(&mut seed) % (i as u64 + 1)) as usize;
            physical.swap(i, other);
        }
        let tables = (0..batch)
            .map(|b| {
                let mut table = BlockTable::new();
                for &block in &physical[..shared_blocks] {
                    table.append_block(block);
                }
                let first = shared_blocks + b * (blocks - shared_blocks);
                for &block in &physical[first..first + blocks - shared_blocks] {
                    table.append_block(block);
                }
                table
            })
            .collect();
        let kv = (0..cache_floats)
            .map(|_| ((random(&mut seed) >> 40) as f32 + 0.5) / 8_388_608.0 - 1.0)
            .collect();
        let q = (0..queries)
            .map(|_| ((random(&mut seed) >> 40) as f32 + 0.5) / 8_388_608.0 - 1.0)
            .collect();
        Ok(Self {
            attn: PagedAttention {
                layout,
                block_size: BLOCK_SIZE,
                num_heads: config.num_attention_heads,
                head_dim: config.head_dim(),
                kv_group: config.kv_group(),
                score_stride: used,
                heads_per_lane: PagedAttention::lane_width(
                    config.kv_group(),
                    config.num_key_value_heads,
                    batch,
                ),
            },
            tables,
            kv,
            q,
            baseline_out: vec![0.0; queries],
            candidate_out: vec![0.0; queries],
            scores: vec![0.0; score_floats],
            scratch: SharedPrefixScratch::default(),
            context,
            shared_blocks,
        })
    }

    fn run(&mut self, candidate: bool) -> f64 {
        let entries: Vec<_> = self
            .tables
            .iter()
            .map(|table| AttnEntry::new(table, self.context, None))
            .collect();
        let out = if candidate {
            &mut self.candidate_out
        } else {
            &mut self.baseline_out
        };
        // The existing kernel also receives already-constructed AttnEntry values.
        // Only the new grouping plan is additional work and belongs in its timer.
        let start = Instant::now();
        if candidate {
            if let Some(plan) = SharedPrefixPlan::new(&entries, BLOCK_SIZE) {
                plan.run(&self.attn, out, &self.q, &self.kv, 0, &mut self.scratch);
            } else {
                self.attn
                    .run(out, &mut self.scores, &self.q, &self.kv, &entries, 0);
            }
        } else {
            self.attn
                .run(out, &mut self.scores, &self.q, &self.kv, &entries, 0);
        }
        black_box(out);
        start.elapsed().as_secs_f64() * 1_000_000.0
    }

    fn verify(&self) -> Result<f32> {
        let mut max_abs = 0.0f32;
        for (index, (&baseline, &candidate)) in self
            .baseline_out
            .iter()
            .zip(&self.candidate_out)
            .enumerate()
        {
            ensure!(
                baseline.is_finite() && candidate.is_finite(),
                "nonfinite attention output at {index}"
            );
            max_abs = max_abs.max((baseline - candidate).abs());
            ensure!(
                baseline.to_bits() == candidate.to_bits(),
                "attention mismatch at {index}: baseline={baseline}, candidate={candidate}, abs={}",
                (baseline - candidate).abs()
            );
        }
        Ok(max_abs)
    }
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program).args(args).output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_owned())
}

fn environment() -> Value {
    let cpu = command("sysctl", &["-n", "machdep.cpu.brand_string"]).or_else(|| {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()?
            .lines()
            .find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|s| s.split_once(':'))
                    .map(|(_, name)| name.trim().to_owned())
            })
    });
    json!({
        "cpu": cpu, "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "uname": command("uname", &["-srm"]), "simd": paged_infer::simd::backend(),
        "rayon_threads": rayon::current_num_threads(),
        "rayon_num_threads_env": std::env::var("RAYON_NUM_THREADS").ok(),
        "attention_lanes_per_thread_env": std::env::var("PAGED_INFER_ATTN_LANES_PER_THREAD").ok(),
        "logical_cpus": std::thread::available_parallelism().ok().map(|n| n.get()),
        "build": {
            "rustc": env!("PAGED_BUILD_RUSTC"), "target": env!("PAGED_BUILD_TARGET"),
            "profile": env!("PAGED_BUILD_PROFILE"), "opt_level": env!("PAGED_BUILD_OPT_LEVEL"),
            "debug": env!("PAGED_BUILD_DEBUG"), "rustflags": env!("PAGED_BUILD_CARGO_ENCODED_RUSTFLAGS"),
            "git_head": env!("PAGED_BUILD_GIT_HEAD"), "git_dirty": env!("PAGED_BUILD_GIT_DIRTY"),
            "source_sha256": env!("PAGED_BUILD_SOURCE_SHA256"),
        },
    })
}

fn distribution(mut samples: Vec<f64>) -> Value {
    samples.sort_by(f64::total_cmp);
    let percentile = |p: f64| samples[(samples.len() as f64 * p).ceil() as usize - 1];
    json!({"p50_us": percentile(0.5), "p95_us": percentile(0.95), "min_us": samples[0],
        "max_us": samples[samples.len() - 1]})
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--help"] || args.as_slice() == ["-h"] {
        println!("{HELP}");
        return Ok(());
    }
    ensure!(args.is_empty(), "unknown arguments; use --help");
    let batches = values("SHARED_ATTN_BATCHES", "1 4 8 16", false)?;
    let contexts = values("SHARED_ATTN_CONTEXTS", "256 1024 4096", false)?;
    let percentages = values("SHARED_ATTN_PERCENTAGES", "0 50 90", true)?;
    ensure!(
        percentages.iter().all(|&n| n <= 100),
        "SHARED_ATTN_PERCENTAGES must be at most 100"
    );
    let reps = count("SHARED_ATTN_REPS", "9")?;
    let warmup = count("SHARED_ATTN_WARMUP", "2")?;
    let mut output = BufWriter::new(io::stdout().lock());
    writeln!(
        output,
        "{}",
        json!({
            "type": "manifest", "version": 1, "benchmark": "shared_attention", "environment": environment(),
            "shape": {"layers": 1, "heads": 32, "kv_heads": 4, "head_dim": 64, "block_size": BLOCK_SIZE},
            "batches": batches, "contexts": contexts, "requested_shared_percentages": percentages,
            "repeats": reps, "warmup_pairs": warmup, "timing_unit": "microseconds",
            "candidate_includes_plan_construction": true, "scratch_allocation": "reused after warmup",
            "context_definition": "history tokens, plus one current attended token",
        })
    )?;
    let mut case_index = 0;
    for &context in &contexts {
        for &batch in &batches {
            for &percentage in &percentages {
                let mut case = Case::new(batch, context, percentage)?;
                let entries: Vec<_> = case
                    .tables
                    .iter()
                    .map(|table| AttnEntry::new(table, context, None))
                    .collect();
                let shared_tokens = SharedPrefixPlan::new(&entries, BLOCK_SIZE)
                    .map(|plan| plan.prefix_tokens())
                    .unwrap_or(0);
                for repetition in 0..warmup {
                    let first_candidate = (case_index + repetition) % 2 == 1;
                    case.run(first_candidate);
                    case.run(!first_candidate);
                    case.verify()?;
                }
                let mut samples = Vec::with_capacity(reps);
                let mut baseline_times = Vec::with_capacity(reps);
                let mut candidate_times = Vec::with_capacity(reps);
                let mut max_abs = 0.0f32;
                for repetition in 0..reps {
                    let first_candidate = (case_index + repetition) % 2 == 1;
                    let first = case.run(first_candidate);
                    let second = case.run(!first_candidate);
                    let (baseline_us, candidate_us) = if first_candidate {
                        (second, first)
                    } else {
                        (first, second)
                    };
                    max_abs = max_abs.max(case.verify()?);
                    baseline_times.push(baseline_us);
                    candidate_times.push(candidate_us);
                    samples.push(json!({"repeat": repetition + 1,
                        "order": if first_candidate { "candidate_baseline" } else { "baseline_candidate" },
                        "baseline_us": baseline_us, "candidate_us": candidate_us}));
                }
                let baseline = distribution(baseline_times);
                let candidate = distribution(candidate_times);
                writeln!(
                    output,
                    "{}",
                    json!({
                        "type": "case", "case_index": case_index, "batch": batch, "context": context,
                        "requested_shared_percentage": percentage,
                        "mapped_common_prefix_tokens": case.shared_blocks * BLOCK_SIZE,
                        "selected_path": if shared_tokens > 0 { "shared_prefix" } else { "fallback" },
                        "shared_prefix_tokens": shared_tokens, "heads_per_lane": case.attn.heads_per_lane,
                        "physical_kv_bytes": case.kv.len() * size_of::<f32>(),
                        "baseline_score_scratch_bytes": case.scores.capacity() * size_of::<f32>(),
                        "candidate_extra_scratch_bytes": case.scratch.allocated_bytes(),
                        "candidate_score_scratch_bytes": if shared_tokens > 0 { 0 } else { case.scores.capacity() * size_of::<f32>() },
                        "scratch_note": "plan has no heap allocation; entry vectors are outside both timers; harness retains baseline scores",
                        "baseline": baseline, "candidate": candidate,
                        "median_speedup": baseline["p50_us"].as_f64().unwrap() / candidate["p50_us"].as_f64().unwrap(),
                        "samples": samples, "verified_finite_bit_identical": true, "max_abs_error": max_abs,
                    })
                )?;
                output.flush()?;
                case_index += 1;
            }
        }
    }
    writeln!(
        output,
        "{}",
        json!({"type": "verification", "passed": true, "complete": true, "cases": case_index})
    )?;
    Ok(())
}
