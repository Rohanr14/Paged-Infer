//! Full-model steady-state decode with real, causally computed KV histories.
//!
//! This measures all transformer layers, vocabulary projection and greedy
//! selection. Loading, prompt preparation, admission and HTTP are outside its
//! scope. Every timed variant starts from the same prepared history.

use anyhow::{ensure, Context, Result};
use memmap2::MmapOptions;
use paged_infer::memory::allocator::PhysicalBlock;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{
    BatchScratch, ForwardScratch, LlamaConfig, LlamaWeights, ModelLoader, Quantization,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const BLOCK_SIZE: usize = 16;
const PREFILL_CHUNK: usize = 32;
const BOOTSTRAP_RESAMPLES: usize = 10_000;
const BOOTSTRAP_SEED: u64 = 0x7061_6972_6564_4349;
const HELP: &str = "Full-model shared-prefix steady-state decode; writes JSONL to stdout.
Environment controls:
  MODEL_PATH                 Safetensors file with adjacent config.json;
                             default: bundled tiny synthetic-weight fixture
  SHARED_DECODE_CONTEXT      Common content prompt length, multiple of 16; default: 64
  SHARED_DECODE_BATCH        Distinct decoding sequences; default: 4
  SHARED_DECODE_STEPS        Greedy decode steps per sequence; default: 8
  SHARED_DECODE_REPS         Interleaved baseline/candidate pairs; default: 3
  SHARED_DECODE_PERCENTAGE   Physically shared prefix percent, rounded to blocks;
                             default: 90 (use 0 for no-sharing control)
  QUANT                     f32 or int8; default: f32
  RAYON_NUM_THREADS          Rayon worker count
Every KV history is computed from model weights, including fixture runs. The
common content prompt is followed by one distinct private token per sequence.
EOS is treated as an ordinary token: every sequence runs the fixed step budget.
Prompt preparation, warmup, validation and reporting are outside timed loops;
finite-logit checking is combined with timed greedy selection. Each variant
warms its entire decode range, then restarts from identical held tokens.
Warmup hashes complete logits at every step; timed decode never hashes logits.
Process resource snapshots and the loop-start timestamp are outside timed work.
Increase SHARED_DECODE_STEPS for longer runs and SHARED_DECODE_REPS for more
measurement pairs; steps within a run are not bootstrap replicates.
The paired bootstrap interval describes observed pair variation under independent,
exchangeable-pair assumptions. Few pairs, serial drift and order effects can make
it misleading; even a narrow interval is not a performance guarantee.
Builds with the profiling feature are diagnostic and unsuitable for performance
gating because instrumentation adds timer and counter overhead.
This is steady-state model decode, not request-lifecycle or HTTP throughput.";

struct Settings {
    model: PathBuf,
    fixture: bool,
    context: usize,
    batch: usize,
    steps: usize,
    repeats: usize,
    percentage: usize,
    quantization: Quantization,
}

fn integer(key: &str, default: usize, allow_zero: bool) -> Result<usize> {
    let value = std::env::var(key).map_or(Ok(default), |s| {
        s.parse::<usize>()
            .with_context(|| format!("invalid {key}: {s}"))
    })?;
    ensure!(allow_zero || value > 0, "{key} must be positive");
    Ok(value)
}

impl Settings {
    fn from_environment() -> Result<Self> {
        let model = std::env::var_os("MODEL_PATH").map(PathBuf::from);
        let fixture = model.is_none();
        let model = model.unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_llama.safetensors")
        });
        let quantization = match std::env::var("QUANT").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("f32") => Quantization::F32,
            Ok("int8") => Quantization::Int8,
            _ => anyhow::bail!("QUANT must be f32 or int8"),
        };
        let settings = Self {
            model,
            fixture,
            context: integer("SHARED_DECODE_CONTEXT", 64, false)?,
            batch: integer("SHARED_DECODE_BATCH", 4, false)?,
            steps: integer("SHARED_DECODE_STEPS", 8, false)?,
            repeats: integer("SHARED_DECODE_REPS", 3, false)?,
            percentage: integer("SHARED_DECODE_PERCENTAGE", 90, true)?,
            quantization,
        };
        ensure!(
            settings.context.is_multiple_of(BLOCK_SIZE),
            "SHARED_DECODE_CONTEXT must be a multiple of {BLOCK_SIZE}"
        );
        ensure!(
            settings.percentage <= 100,
            "SHARED_DECODE_PERCENTAGE must be at most 100"
        );
        ensure!(
            settings.model.is_file(),
            "checkpoint not found: {}",
            settings.model.display()
        );
        Ok(settings)
    }

    fn reserved_tokens(&self) -> Result<usize> {
        self.context
            .checked_add(self.steps)
            .and_then(|n| n.checked_add(1))
            .context("context plus decode steps overflows")
    }

    fn validate_model(&self, config: &LlamaConfig) -> Result<()> {
        config.validate()?;
        ensure!(
            config.vocab_size <= u32::MAX as usize,
            "vocabulary exceeds u32 token IDs"
        );
        ensure!(
            self.batch <= config.vocab_size,
            "batch exceeds vocabulary; private prompt tokens must be distinct"
        );
        if let Some(limit) = config.max_position_embeddings {
            ensure!(
                self.reserved_tokens()? <= limit,
                "prepared history plus decode positions exceeds checkpoint context limit {limit}"
            );
        }
        checked_product(&[self.batch, self.steps], "generated token count")?;
        checked_product(
            &[
                self.batch,
                config
                    .hidden_size
                    .max(config.intermediate_size)
                    .max(config.vocab_size)
                    .max(config.kv_dim()),
                size_of::<f32>(),
            ],
            "batch scratch",
        )?;
        checked_product(
            &[
                self.batch,
                config.num_attention_heads,
                self.reserved_tokens()?,
                size_of::<f32>(),
            ],
            "attention score scratch",
        )?;
        Ok(())
    }
}

fn checked_product(factors: &[usize], description: &str) -> Result<usize> {
    let count = factors
        .iter()
        .try_fold(1usize, |n, &factor| n.checked_mul(factor))
        .with_context(|| format!("{description} size overflows"))?;
    ensure!(
        count <= isize::MAX as usize,
        "{description} exceeds addressable allocation size"
    );
    Ok(count)
}

struct Harness {
    tables: Vec<BlockTable>,
    kv: Vec<f32>,
    shared_blocks: usize,
    prompt: Vec<u32>,
    private_tokens: Vec<u32>,
    initial_tokens: Vec<u32>,
}

impl Harness {
    fn prepare(
        config: &LlamaConfig,
        settings: &Settings,
        weights: &LlamaWeights<'_>,
    ) -> Result<Self> {
        let blocks_per_sequence = settings.reserved_tokens()?.div_ceil(BLOCK_SIZE);
        let shared_blocks = settings
            .context
            .checked_mul(settings.percentage)
            .context("shared prefix length overflows")?
            / 100
            / BLOCK_SIZE;
        let private_blocks = blocks_per_sequence - shared_blocks;
        let total_blocks = private_blocks
            .checked_mul(settings.batch)
            .and_then(|n| n.checked_add(shared_blocks))
            .context("physical block count overflows")?;
        let floats = checked_product(
            &[
                config.num_hidden_layers,
                total_blocks,
                BLOCK_SIZE,
                config.num_key_value_heads,
                2,
                config.head_dim(),
            ],
            "KV cache",
        )?;
        checked_product(&[floats, size_of::<f32>()], "KV cache bytes")?;
        let layout = config.kv_layout(total_blocks, BLOCK_SIZE);
        let mut physical: Vec<_> = (0..total_blocks)
            .map(|index| PhysicalBlock { index })
            .collect();
        let mut random = 0xc4a5_89bd_1043_3927u64;
        for index in (1..physical.len()).rev() {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            physical.swap(index, (random % (index as u64 + 1)) as usize);
        }
        let tables: Vec<_> = (0..settings.batch)
            .map(|sequence| {
                let mut table = BlockTable::new();
                for &block in &physical[..shared_blocks] {
                    table.append_block(block);
                }
                let first = shared_blocks + sequence * private_blocks;
                for &block in &physical[first..first + private_blocks] {
                    table.append_block(block);
                }
                table
            })
            .collect();
        let mut kv = vec![0.0; floats];
        // These are deterministic valid IDs, not a language-quality prompt.
        // The first token uses the checkpoint BOS convention when one exists.
        let mut prompt: Vec<u32> = (0..settings.context)
            .map(|index| ((index as u64 * 17 + 3) % config.vocab_size as u64) as u32)
            .collect();
        if let Some(bos) = config.bos_token_id {
            ensure!(
                (bos as usize) < config.vocab_size,
                "BOS ID is outside the vocabulary"
            );
            prompt[0] = bos;
        }
        let private_tokens: Vec<u32> = (0..settings.batch)
            .map(|sequence| ((sequence + 17) % config.vocab_size) as u32)
            .collect();
        let mut prefill_scratch = BatchScratch::new(config, PREFILL_CHUNK);
        weights.prefill_batched(
            &prompt,
            0,
            config,
            &tables[0],
            &mut kv,
            BLOCK_SIZE,
            PREFILL_CHUNK,
            &mut prefill_scratch,
        );
        greedy(prefill_scratch.logits_for(0, config.vocab_size))
            .context("nonfinite common prompt logits")?;
        // All sequences have the same content history. Copy only complete
        // computed blocks whose physical identity is deliberately private.
        // Future writable slots stay private and are never used as history.
        for table in tables.iter().skip(1) {
            for logical in shared_blocks..settings.context / BLOCK_SIZE {
                layout.copy_block(
                    &mut kv,
                    tables[0].slots()[logical].block,
                    table.slots()[logical].block,
                );
            }
        }
        let mut initial_tokens = Vec::with_capacity(settings.batch);
        let mut scratch = ForwardScratch::new(config);
        for (table, &token) in tables.iter().zip(&private_tokens) {
            weights.forward_into(
                token,
                settings.context,
                config,
                table,
                &mut kv,
                BLOCK_SIZE,
                None,
                &mut scratch,
            );
            initial_tokens
                .push(greedy(&scratch.logits).context("nonfinite private prompt logits")?);
        }
        Ok(Self {
            tables,
            kv,
            shared_blocks,
            prompt,
            private_tokens,
            initial_tokens,
        })
    }
}

/// Finite check and deterministic first-maximum selection in one vocabulary pass.
fn greedy(logits: &[f32]) -> Result<u32> {
    ensure!(!logits.is_empty(), "empty vocabulary logits");
    let mut best = f32::NEG_INFINITY;
    let mut token = 0;
    for (index, &value) in logits.iter().enumerate() {
        ensure!(value.is_finite(), "nonfinite logit at token {index}");
        if value > best {
            best = value;
            token = index as u32;
        }
    }
    Ok(token)
}

struct DecodeRun {
    elapsed_ms: f64,
    steps_ms: Vec<f64>,
    tokens: Vec<Vec<u32>>,
    all_step_logits_sha256: Option<String>,
    timed_loop_start_unix_ms: Option<u128>,
    process_resource_usage: Option<Value>,
}

/// Cumulative process counters. Snapshot outside the timed loop; these include
/// all process threads and cannot identify which worker or outside process
/// caused a stall. A missing or non-monotonic snapshot produces no delta.
struct ProcessUsage {
    user_cpu_us: u64,
    system_cpu_us: u64,
    minor_page_faults: u64,
    major_page_faults: u64,
    voluntary_context_switches: u64,
    involuntary_context_switches: u64,
}

impl ProcessUsage {
    fn since(&self, before: &Self) -> Option<Value> {
        Some(json!({
            "user_cpu_ms": self.user_cpu_us.checked_sub(before.user_cpu_us)? as f64 / 1000.0,
            "system_cpu_ms": self.system_cpu_us.checked_sub(before.system_cpu_us)? as f64 / 1000.0,
            "minor_page_faults": self.minor_page_faults.checked_sub(before.minor_page_faults)?,
            "major_page_faults": self.major_page_faults.checked_sub(before.major_page_faults)?,
            "voluntary_context_switches": self.voluntary_context_switches.checked_sub(before.voluntary_context_switches)?,
            "involuntary_context_switches": self.involuntary_context_switches.checked_sub(before.involuntary_context_switches)?,
        }))
    }
}

#[cfg(unix)]
fn process_usage() -> Option<ProcessUsage> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage receives writable storage for exactly one rusage. Only
    // a successful call initializes the value that we read below.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful getrusage call initialized this structure.
    let usage = unsafe { usage.assume_init() };
    let micros = |time: libc::timeval| {
        u64::try_from(time.tv_sec)
            .ok()?
            .checked_mul(1_000_000)?
            .checked_add(u64::try_from(time.tv_usec).ok()?)
    };
    Some(ProcessUsage {
        user_cpu_us: micros(usage.ru_utime)?,
        system_cpu_us: micros(usage.ru_stime)?,
        minor_page_faults: u64::try_from(usage.ru_minflt).ok()?,
        major_page_faults: u64::try_from(usage.ru_majflt).ok()?,
        voluntary_context_switches: u64::try_from(usage.ru_nvcsw).ok()?,
        involuntary_context_switches: u64::try_from(usage.ru_nivcsw).ok()?,
    })
}

#[cfg(not(unix))]
fn process_usage() -> Option<ProcessUsage> {
    None
}

/// Canonical f32-bit encoding without allocating a second vocabulary buffer.
fn update_logit_digest(digest: &mut Sha256, logits: &[f32]) {
    let mut bytes = [0u8; 1024];
    for chunk in logits.chunks(bytes.len() / size_of::<f32>()) {
        for (value, encoded) in chunk.iter().zip(bytes.as_chunks_mut::<4>().0) {
            encoded.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        digest.update(&bytes[..std::mem::size_of_val(chunk)]);
    }
}

// The const parameter gives timed decode its own specialization with no
// per-step logit hashing or process-resource sampling in the model loop.
fn decode<const VERIFY_LOGITS: bool>(
    weights: &LlamaWeights<'_>,
    config: &LlamaConfig,
    settings: &Settings,
    harness: &mut Harness,
    scratch: &mut BatchScratch,
) -> Result<DecodeRun> {
    let tables: Vec<_> = harness.tables.iter().collect();
    let mut held = harness.initial_tokens.clone();
    let mut positions = vec![settings.context + 1; settings.batch];
    let mut tokens: Vec<Vec<u32>> = (0..settings.batch)
        .map(|_| Vec::with_capacity(settings.steps))
        .collect();
    let mut steps_ms = Vec::with_capacity(settings.steps);
    let mut logits_digest = VERIFY_LOGITS.then(Sha256::new);
    let resource_start = if VERIFY_LOGITS { None } else { process_usage() };
    let timed_loop_start_unix_ms = if VERIFY_LOGITS {
        None
    } else {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_millis())
    };
    let start = Instant::now();
    for _ in 0..settings.steps {
        let step_start = Instant::now();
        weights.decode_batch_into(
            &held,
            &positions,
            &tables,
            config,
            &mut harness.kv,
            BLOCK_SIZE,
            scratch,
        );
        if VERIFY_LOGITS {
            update_logit_digest(
                logits_digest.as_mut().unwrap(),
                &scratch.logits[..settings.batch * config.vocab_size],
            );
        }
        for sequence in 0..settings.batch {
            held[sequence] = greedy(scratch.logits_for(sequence, config.vocab_size))?;
            tokens[sequence].push(held[sequence]);
            positions[sequence] += 1;
        }
        steps_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let resource_end = if VERIFY_LOGITS { None } else { process_usage() };
    let process_resource_usage = resource_start
        .as_ref()
        .zip(resource_end.as_ref())
        .and_then(|(before, after)| after.since(before));
    Ok(DecodeRun {
        elapsed_ms,
        steps_ms,
        tokens,
        all_step_logits_sha256: logits_digest.map(|digest| format!("{:x}", digest.finalize())),
        timed_loop_start_unix_ms,
        process_resource_usage,
    })
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
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
        "matmul_tile_env": std::env::var("PAGED_INFER_MATMUL_TILE").ok(),
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

fn distribution(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let at = |p: f64| values[(values.len() as f64 * p).ceil() as usize - 1];
    json!({"count": values.len(), "p50": at(0.5), "p95": at(0.95), "p99": at(0.99),
        "min": values[0], "max": values[values.len() - 1]})
}

/// The conventional median; for even samples use the midpoint of both middle
/// values. This is deliberately separate from the legacy nearest-rank p50
/// fields, whose meaning remains unchanged in existing reports.
fn median(values: &mut [f64]) -> f64 {
    assert!(!values.is_empty());
    let even = values.len().is_multiple_of(2);
    let middle = values.len() / 2;
    let (lower, upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    if even {
        let lower = lower.iter().copied().max_by(f64::total_cmp).unwrap();
        lower + (*upper - lower) * 0.5
    } else {
        *upper
    }
}

/// SplitMix64 with rejection sampling rather than a biased modulo reduction.
/// All state is local to reporting, so statistics never alter model sampling.
struct BootstrapRandom(u64);

impl BootstrapRandom {
    fn index(&mut self, size: usize) -> usize {
        let bound = size as u64;
        let rejection = bound.wrapping_neg() % bound;
        loop {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            if value >= rejection {
                return (value % bound) as usize;
            }
        }
    }
}

/// Percentile bootstrap of the median paired elapsed-time ratio. Resampling a
/// ratio selects its complete original baseline/candidate pair; sampling the
/// two schedules independently would destroy pairing. The individual decode
/// steps are never treated as independent observations.
///
/// Each resample contains n pairs sampled with replacement. Endpoints are the
/// nearest-rank 2.5th/97.5th percentiles of the resampled medians. This is a
/// descriptive, conditional interval, not a correction for serial dependence,
/// order effects, tiny samples or unobserved system states. Method reference:
/// https://www.itl.nist.gov/div898/handbook/eda/section3/eda334.htm
fn paired_speedup_statistics(baseline: &[f64], candidate: &[f64]) -> Result<Value> {
    ensure!(
        !baseline.is_empty(),
        "paired statistics require at least one repeat"
    );
    ensure!(
        baseline.len() == candidate.len(),
        "paired timing counts differ"
    );
    let mut ratios = Vec::with_capacity(baseline.len());
    let mut pairs = Vec::with_capacity(baseline.len());
    for (repeat, (&baseline_ms, &candidate_ms)) in baseline.iter().zip(candidate).enumerate() {
        ensure!(
            baseline_ms.is_finite()
                && baseline_ms > 0.0
                && candidate_ms.is_finite()
                && candidate_ms > 0.0,
            "paired timings must be positive and finite"
        );
        let speedup = baseline_ms / candidate_ms;
        ensure!(
            speedup.is_finite() && speedup > 0.0,
            "paired timing ratio must be positive and finite"
        );
        ratios.push(speedup);
        pairs.push(
            json!({"repeat": repeat + 1, "baseline_elapsed_ms": baseline_ms,
            "candidate_elapsed_ms": candidate_ms, "speedup": speedup}),
        );
    }
    let minimum = ratios.iter().copied().min_by(f64::total_cmp).unwrap();
    let maximum = ratios.iter().copied().max_by(f64::total_cmp).unwrap();
    let estimate = median(&mut ratios.clone());
    let enough_pairs = ratios.len() > 1;
    let interval = if enough_pairs {
        let mut random = BootstrapRandom(BOOTSTRAP_SEED);
        let mut sample = vec![0.0; ratios.len()];
        let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
        for _ in 0..BOOTSTRAP_RESAMPLES {
            for value in &mut sample {
                *value = ratios[random.index(ratios.len())];
            }
            medians.push(median(&mut sample));
        }
        medians.sort_by(f64::total_cmp);
        let lower = medians[BOOTSTRAP_RESAMPLES / 40 - 1];
        let upper = medians[BOOTSTRAP_RESAMPLES * 39 / 40 - 1];
        Some(json!({"lower": lower, "upper": upper}))
    } else {
        // One observed pair provides no between-pair uncertainty estimate.
        // Returning [estimate, estimate] would imply unsupported precision.
        None
    };
    Ok(json!({
        "pair_count": ratios.len(), "pairs": pairs,
        "estimator": "median of baseline_elapsed_ms / candidate_elapsed_ms within the same repeat",
        "median_convention": "middle value for odd counts; midpoint of two middle values for even counts",
        "interpretation": "greater than 1 means the candidate was faster",
        "median": estimate, "min": minimum, "max": maximum,
        "bootstrap_95": {
            "method": "paired percentile bootstrap of the median elapsed-time ratio",
            "confidence_level": 0.95, "interval": interval,
            "resamples": if enough_pairs { BOOTSTRAP_RESAMPLES } else { 0 },
            "seed_hex": format!("0x{BOOTSTRAP_SEED:016x}"),
            "rng": "SplitMix64 with rejection-sampled pair indices",
            "resampling_unit": "one complete baseline/candidate repeat pair",
            "pairs_per_resample": ratios.len(),
            "endpoint_convention": "nearest-rank 2.5th and 97.5th percentiles, ceil(p * resamples), one-based",
            "status": if enough_pairs { "descriptive_interval" } else { "insufficient_pairs" },
            "small_sample_caution": ratios.len() < 10,
            "small_sample_caution_threshold_pairs": 10,
            "identical_observed_ratios": minimum == maximum,
            "assumptions": "repeat pairs are independent and exchangeable observations of the same workload and machine conditions",
            "limitations": [
                "Few pairs provide a coarse empirical distribution; 10 pairs is a caution threshold, not a validity guarantee.",
                "Serial dependence, thermal drift and systematic order effects are not corrected by this resampling method.",
                "A narrow or zero-width interval does not guarantee future speedup or measure uncertainty in unobserved system conditions.",
                "A nominal 95% percentile interval can under-cover, especially with small samples; it is not proof of a performance gate.",
                "Decode steps within one run are dependent and do not increase the bootstrap sample count."
            ],
        }
    }))
}

/// Extra views of the same retained observations, never replacements for the
/// predeclared median of paired ratios. Order follows the unchanged AB/BA loop.
fn performance_diagnostics(baseline: &[f64], candidate: &[f64], tokens_per_run: usize) -> Value {
    assert!(!baseline.is_empty() && baseline.len() == candidate.len());
    let baseline_total: f64 = baseline.iter().sum();
    let candidate_total: f64 = candidate.iter().sum();
    let tokens_per_variant = tokens_per_run as f64 * baseline.len() as f64;
    let strata: Vec<_> = ["baseline", "shared_prefix"]
        .into_iter()
        .enumerate()
        .map(|(parity, first_variant)| {
            let repeat_numbers: Vec<_> = (parity..baseline.len())
                .step_by(2)
                .map(|index| index + 1)
                .collect();
            let mut ratios: Vec<_> = repeat_numbers
                .iter()
                .map(|&repeat| baseline[repeat - 1] / candidate[repeat - 1])
                .collect();
            let estimate = (!ratios.is_empty()).then(|| median(&mut ratios));
            json!({
                "first_variant": first_variant, "pair_count": repeat_numbers.len(),
                "repeat_numbers": repeat_numbers, "median": estimate,
            })
        })
        .collect();
    json!({
        "diagnostic_only": true,
        "interpretation": "Alternative views of the same complete samples; the primary estimator remains paired_elapsed_time_speedup. Order strata are descriptive, not independent experiments.",
        "aggregate_throughput": {
            "estimator": "sum of baseline elapsed times divided by sum of candidate elapsed times, with equal total generated-token budgets",
            "baseline_elapsed_ms": baseline_total,
            "candidate_elapsed_ms": candidate_total,
            "tokens_per_variant": tokens_per_variant,
            "baseline_tokens_per_second": tokens_per_variant * 1000.0 / baseline_total,
            "candidate_tokens_per_second": tokens_per_variant * 1000.0 / candidate_total,
            "speedup": baseline_total / candidate_total,
        },
        "order_stratified_paired_speedup": strata,
    })
}

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.as_slice() == ["--help"] || arguments.as_slice() == ["-h"] {
        println!("{HELP}");
        return Ok(());
    }
    ensure!(arguments.is_empty(), "unknown arguments; use --help");
    let settings = Settings::from_environment()?;
    let config_path = settings.model.with_file_name("config.json");
    let mut config = LlamaConfig::from_hf_config(&config_path)?;
    config.quantization = settings.quantization;
    settings.validate_model(&config)?;
    let checkpoint_sha256 = hash_file(&settings.model)?;
    let config_sha256 = hash_file(&config_path)?;
    let checkpoint = File::open(&settings.model)?;
    // The checkpoint mapping outlives all loaded tensor views.
    let mmap = unsafe { MmapOptions::new().map(&checkpoint)? };
    let loader = ModelLoader::new(&mmap)?;
    let weights = loader.load_weights(&config)?;
    eprintln!(
        "Preparing computed KV for {} entries at context {}",
        settings.batch, settings.context
    );
    let preparation_start = Instant::now();
    let mut harness = Harness::prepare(&config, &settings, &weights)?;
    let preparation_ms = preparation_start.elapsed().as_secs_f64() * 1000.0;
    let inputs = json!({"common_content_prompt": harness.prompt,
        "private_prompt_tokens": harness.private_tokens, "initial_held_tokens": harness.initial_tokens,
        "initial_held_position": settings.context + 1, "measured_steps": settings.steps});
    let inputs_sha256 = hash(&serde_json::to_vec(&inputs)?);
    let mut output = BufWriter::new(io::stdout().lock());
    writeln!(
        output,
        "{}",
        json!({
            "type": "manifest", "version": 1, "benchmark": "shared_decode",
            "scope": "full-model steady-state decode, not request-lifecycle or HTTP throughput",
            "profiling_enabled": paged_infer::profiling::enabled(),
            "performance_gate_eligible": !paged_infer::profiling::enabled(),
            "profiling_note": "Instrumented builds are diagnostic; timer/counter overhead makes their timings unsuitable for performance gating. Nested stage timings are not additive; parallel lane timings sum worker time.",
            "environment": environment(), "model_path": settings.model,
            "synthetic_weight_fixture": settings.fixture, "synthetic_kv": false,
            "checkpoint_sha256": checkpoint_sha256, "config_sha256": config_sha256,
            "inputs_sha256": inputs_sha256, "inputs": inputs,
            "quantization": format!("{:?}", settings.quantization),
            "shape": {"layers": config.num_hidden_layers, "hidden": config.hidden_size,
                "heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
                "head_dim": config.head_dim(), "vocabulary": config.vocab_size},
            "context": settings.context, "batch": settings.batch, "steps": settings.steps,
            "repeats": settings.repeats, "block_size": BLOCK_SIZE,
            "requested_shared_percentage": settings.percentage,
            "mapped_common_prefix_tokens": harness.shared_blocks * BLOCK_SIZE,
            "packed_projection_bytes": weights.weight_bytes(), "checkpoint_bytes": mmap.len(),
        "physical_kv_bytes": harness.kv.len() * size_of::<f32>(),
        "preparation_ms": preparation_ms,
            "timed_work": "all layers, LM head, finite-logit scan, argmax and token recording",
            "excluded_work": "model loading, fingerprinting, prompt preparation, full-range warmup and logit hashing, process-resource snapshots, loop-start wall-clock timestamp, output comparison, reporting",
            "warmup": "fresh scratch per variant/run; one full untimed decode hashes every step's complete logits, then restarts from identical held tokens; warmup_elapsed_ms includes hashing",
            "warmup_logit_hash": {
                "algorithm": "SHA-256",
                "encoding": "complete raw f32 bits, little-endian, in step/batch-entry/vocabulary-ID order",
                "steps_per_run": settings.steps,
                "entries_per_step": settings.batch,
                "logits_per_entry": config.vocab_size,
                "timed_loop_hashing": false,
            },
            "process_resource_usage": {
                "provider": if cfg!(unix) { "getrusage(RUSAGE_SELF)" } else { "unavailable" },
                "scope": "process-wide deltas bracketing the timed loop, including all worker threads and small boundary bookkeeping; no per-step sampling",
                "unavailable": "run field is null if unsupported, either snapshot fails, or counters decrease",
            },
            "termination": "fixed step budget; EOS tokens do not terminate this benchmark",
            "physical_no_sharing_control": "percentage 0 copies genuine common-content KV into distinct physical blocks",
        })
    )?;
    output.flush()?;
    let token_count = settings.batch * settings.steps;
    let mut reference: Option<Vec<Vec<u32>>> = None;
    let mut reference_logits: Option<String> = None;
    let mut reference_warmup_logits: Option<String> = None;
    let mut times = [
        Vec::with_capacity(settings.repeats),
        Vec::with_capacity(settings.repeats),
    ];
    let mut pooled_steps = [Vec::new(), Vec::new()];
    for repetition in 0..settings.repeats {
        let order = if repetition % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for (order_index, enabled) in order.into_iter().enumerate() {
            eprintln!(
                "Decode repeat {}/{}, shared={enabled}",
                repetition + 1,
                settings.repeats
            );
            let mut scratch = BatchScratch::new(&config, settings.batch);
            scratch.set_shared_prefix_attention(enabled);
            // Populate scratch to the widest measured window before starting the
            // timer. Replaying overwrites future KV causally; the prepared prompt
            // and distinct private token at `context` are never overwritten.
            let warmup = decode::<true>(&weights, &config, &settings, &mut harness, &mut scratch)?;
            let warmup_logits_sha256 = warmup.all_step_logits_sha256.as_ref().unwrap();
            if let Some(expected) = &reference_warmup_logits {
                ensure!(
                    expected == warmup_logits_sha256,
                    "complete warmup logits differ at repeat {} shared={enabled}",
                    repetition + 1
                );
            } else {
                reference_warmup_logits = Some(warmup_logits_sha256.clone());
            }
            scratch.reset_shared_attention_stats();
            paged_infer::profiling::reset();
            let run = decode::<false>(&weights, &config, &settings, &mut harness, &mut scratch)?;
            let profile = paged_infer::profiling::snapshot();
            ensure!(
                warmup.tokens == run.tokens,
                "warmup and timed greedy outputs differ at repeat {} shared={enabled}",
                repetition + 1
            );
            match &reference {
                Some(expected) => ensure!(
                    *expected == run.tokens,
                    "complete greedy token output differs at repeat {} shared={enabled}",
                    repetition + 1
                ),
                None => reference = Some(run.tokens.clone()),
            }
            let stats = scratch.shared_attention_stats();
            let expected_calls = if enabled && settings.batch > 1 && harness.shared_blocks > 0 {
                settings.steps * config.num_hidden_layers
            } else {
                0
            };
            ensure!(
                stats.layer_calls == expected_calls,
                "unexpected attention path count"
            );
            let mut digest = Sha256::new();
            for &logit in &scratch.logits[..settings.batch * config.vocab_size] {
                digest.update(logit.to_bits().to_le_bytes());
            }
            let final_logits_sha256 = format!("{:x}", digest.finalize());
            if let Some(expected) = &reference_logits {
                ensure!(
                    *expected == final_logits_sha256,
                    "final logits changed across schedules"
                );
            } else {
                reference_logits = Some(final_logits_sha256.clone());
            }
            let variant = usize::from(enabled);
            times[variant].push(run.elapsed_ms);
            pooled_steps[variant].extend_from_slice(&run.steps_ms);
            writeln!(
                output,
                "{}",
                json!({
                    "type": "run", "repeat": repetition + 1, "order_index": order_index,
                    "variant": if enabled { "shared_prefix" } else { "baseline" },
                    "selected_path": if stats.layer_calls > 0 { "shared_prefix" } else { "fallback" },
                    "elapsed_ms": run.elapsed_ms, "tokens_per_second": token_count as f64 * 1000.0 / run.elapsed_ms,
                    "warmup_elapsed_ms": warmup.elapsed_ms,
                    "warmup_logits_sha256": warmup_logits_sha256,
                    "warmup_logit_steps_hashed": settings.steps,
                    "verified_complete_warmup_logits": true,
                    "verified_warmup_timed_greedy_output": true,
                    "timed_loop_start_unix_ms": run.timed_loop_start_unix_ms,
                    "process_resource_usage": run.process_resource_usage,
                    "step_ms": run.steps_ms, "step_distribution_ms": distribution(run.steps_ms.clone()),
                    "generated_tokens": run.tokens, "finish_reason": "fixed_step_budget",
                    "final_logits_sha256": final_logits_sha256,
                    "shared_layer_calls": stats.layer_calls, "shared_query_tokens": stats.query_tokens,
                    "candidate_extra_scratch_bytes": stats.scratch_bytes,
                    "verified_finite_logits": true, "verified_complete_greedy_output": true,
                    "profile": profile,
                })
            )?;
            output.flush()?;
        }
    }
    let baseline = distribution(times[0].clone());
    let candidate = distribution(times[1].clone());
    let baseline_ms = baseline["p50"].as_f64().unwrap();
    let candidate_ms = candidate["p50"].as_f64().unwrap();
    let paired_speedups = paired_speedup_statistics(&times[0], &times[1])?;
    writeln!(
        output,
        "{}",
        json!({
            "type": "summary", "baseline_elapsed_ms": baseline, "candidate_elapsed_ms": candidate,
            "baseline_step_ms": distribution(pooled_steps[0].clone()),
            "candidate_step_ms": distribution(pooled_steps[1].clone()),
            "baseline_median_tokens_per_second": token_count as f64 * 1000.0 / baseline_ms,
            "candidate_median_tokens_per_second": token_count as f64 * 1000.0 / candidate_ms,
            "median_speedup": baseline_ms / candidate_ms,
            "median_speedup_estimator": "legacy ratio of separate nearest-rank p50 elapsed times; paired_elapsed_time_speedup reports the paired estimator",
            "paired_elapsed_time_speedup": paired_speedups,
            "performance_diagnostics": performance_diagnostics(&times[0], &times[1], token_count),
            "timing_samples_retained": settings.repeats * 2,
        })
    )?;
    writeln!(
        output,
        "{}",
        json!({"type": "verification", "passed": true, "complete": true,
        "runs": settings.repeats * 2,
        "warmup_logits_sha256": reference_warmup_logits,
        "warmup_logit_steps_per_run": settings.steps,
        "scope": "all timed greedy tokens, finite logits, final timed logits, every warmup step's complete logits, warmup/timed greedy parity, fixed step budget"})
    )?;
    Ok(())
}

#[cfg(test)]
mod statistical_tests {
    use super::*;

    #[test]
    fn full_logit_digest_preserves_bits_across_steps_and_chunk_boundaries() {
        let mut first_step = vec![1.0f32; 300];
        first_step[0] = -0.0;
        first_step[299] = f32::from_bits(0x7fc0_0123);
        let final_step = [2.0f32, 3.0];
        let digest_steps = |first: &[f32]| {
            let mut digest = Sha256::new();
            update_logit_digest(&mut digest, first);
            update_logit_digest(&mut digest, &final_step);
            format!("{:x}", digest.finalize())
        };
        let bytes: Vec<_> = first_step
            .iter()
            .chain(&final_step)
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let expected = hash(&bytes);
        assert_eq!(digest_steps(&first_step), expected);
        // A changed earlier step must be visible even when the final step is
        // identical, and signed zero must not be normalized during hashing.
        first_step[0] = 0.0;
        assert_ne!(digest_steps(&first_step), expected);
    }

    #[test]
    fn diagnostics_keep_aggregate_and_order_effects_separate() {
        let result = performance_diagnostics(&[2.0, 12.0, 8.0, 1.0], &[1.0, 3.0, 2.0, 2.0], 9);
        assert_eq!(result["aggregate_throughput"]["speedup"], 2.875);
        assert_eq!(result["aggregate_throughput"]["tokens_per_variant"], 36.0);
        let strata = &result["order_stratified_paired_speedup"];
        assert_eq!(strata[0]["first_variant"], "baseline");
        assert_eq!(strata[0]["repeat_numbers"], json!([1, 3]));
        assert_eq!(strata[0]["median"], 3.0);
        assert_eq!(strata[1]["first_variant"], "shared_prefix");
        assert_eq!(strata[1]["repeat_numbers"], json!([2, 4]));
        assert_eq!(strata[1]["median"], 2.25);
        let single = performance_diagnostics(&[2.0], &[1.0], 9);
        assert_eq!(
            single["order_stratified_paired_speedup"][1]["pair_count"],
            0
        );
        assert!(single["order_stratified_paired_speedup"][1]["median"].is_null());
    }

    #[test]
    fn process_deltas_subtract_counters_and_refuse_decreasing_snapshots() {
        let before = ProcessUsage {
            user_cpu_us: 1000,
            system_cpu_us: 2000,
            minor_page_faults: 3,
            major_page_faults: 4,
            voluntary_context_switches: 5,
            involuntary_context_switches: 6,
        };
        let after = ProcessUsage {
            user_cpu_us: 3500,
            system_cpu_us: 6000,
            minor_page_faults: 5,
            major_page_faults: 7,
            voluntary_context_switches: 9,
            involuntary_context_switches: 11,
        };
        assert_eq!(
            after.since(&before).unwrap(),
            json!({"user_cpu_ms": 2.5, "system_cpu_ms": 4.0,
                "minor_page_faults": 2, "major_page_faults": 3,
                "voluntary_context_switches": 4, "involuntary_context_switches": 5})
        );
        assert!(before.since(&after).is_none());
    }

    #[test]
    fn paired_median_preserves_repeat_matching() {
        // The ratio of the two marginal medians is 50, but the median paired
        // effect is 1. Strong common run noise must not destroy pairing.
        let result = paired_speedup_statistics(&[1.0, 50.0, 101.0], &[1.0, 100.0, 1.0]).unwrap();
        assert_eq!(result["median"], 1.0);
        assert_eq!(result["min"], 0.5);
        assert_eq!(result["max"], 101.0);
        assert_eq!(result["pair_count"], 3);
        assert_eq!(
            result["pairs"][1],
            json!({"repeat": 2,
            "baseline_elapsed_ms": 50.0, "candidate_elapsed_ms": 100.0, "speedup": 0.5})
        );
    }

    #[test]
    fn median_uses_both_middle_values_for_even_samples() {
        assert_eq!(median(&mut [9.0, 1.0, 5.0, 3.0]), 4.0);
        assert_eq!(median(&mut [9.0, 1.0, 5.0]), 5.0);
        assert_eq!(median(&mut [f64::MAX, f64::MAX]), f64::MAX);
    }

    #[test]
    fn constant_paired_effect_does_not_invent_independent_run_noise() {
        let result = paired_speedup_statistics(&[2.0, 2000.0, 6.0], &[1.0, 1000.0, 3.0]).unwrap();
        assert_eq!(result["median"], 2.0);
        assert_eq!(
            result["bootstrap_95"]["interval"],
            json!({"lower": 2.0, "upper": 2.0})
        );
        assert_eq!(result["bootstrap_95"]["identical_observed_ratios"], true);
        assert_eq!(result["bootstrap_95"]["small_sample_caution"], true);
    }

    #[test]
    fn two_pair_interval_covers_the_exact_resampling_distribution_endpoints() {
        // Resampling two ratios [1, 3] has medians 1, 2, 2, 3 with equal
        // probability. Its 2.5th and 97.5th percentiles are therefore 1 and 3.
        let result = paired_speedup_statistics(&[1.0, 3.0], &[1.0, 1.0]).unwrap();
        assert_eq!(result["median"], 2.0);
        assert_eq!(
            result["bootstrap_95"]["interval"],
            json!({"lower": 1.0, "upper": 3.0})
        );
        assert_eq!(result["bootstrap_95"]["small_sample_caution"], true);
    }

    #[test]
    fn bootstrap_is_reproducible_and_invariant_to_time_units() {
        let baseline = [8.0, 12.0, 10.0, 40.0, 64.0];
        let candidate = [4.0, 3.0, 20.0, 32.0, 16.0];
        let result = paired_speedup_statistics(&baseline, &candidate).unwrap();
        assert_eq!(
            result,
            paired_speedup_statistics(&baseline, &candidate).unwrap()
        );
        let scaled = paired_speedup_statistics(
            &baseline.map(|value| value * 1024.0),
            &candidate.map(|value| value * 1024.0),
        )
        .unwrap();
        assert_eq!(result["median"], scaled["median"]);
        assert_eq!(result["bootstrap_95"], scaled["bootstrap_95"]);
        let interval = &result["bootstrap_95"]["interval"];
        assert!(interval["lower"].as_f64().unwrap() >= result["min"].as_f64().unwrap());
        assert!(interval["upper"].as_f64().unwrap() <= result["max"].as_f64().unwrap());
        assert!(interval["lower"].as_f64().unwrap() <= interval["upper"].as_f64().unwrap());
    }

    #[test]
    fn one_pair_has_no_claimed_uncertainty_interval() {
        let result = paired_speedup_statistics(&[9.0], &[3.0]).unwrap();
        assert_eq!(result["median"], 3.0);
        assert!(result["bootstrap_95"]["interval"].is_null());
        assert_eq!(result["bootstrap_95"]["status"], "insufficient_pairs");
        assert_eq!(result["bootstrap_95"]["resamples"], 0);
        assert_eq!(result["bootstrap_95"]["small_sample_caution"], true);
    }

    #[test]
    fn invalid_pairs_are_refused_instead_of_serializing_invalid_statistics() {
        assert!(paired_speedup_statistics(&[], &[]).is_err());
        assert!(paired_speedup_statistics(&[1.0], &[1.0, 2.0]).is_err());
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(paired_speedup_statistics(&[value], &[1.0]).is_err());
            assert!(paired_speedup_statistics(&[1.0], &[value]).is_err());
        }
        assert!(paired_speedup_statistics(&[f64::MAX], &[f64::MIN_POSITIVE]).is_err());
        assert!(paired_speedup_statistics(&[f64::MIN_POSITIVE], &[f64::MAX]).is_err());
    }
}
