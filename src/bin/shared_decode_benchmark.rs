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
use std::time::Instant;

const BLOCK_SIZE: usize = 16;
const PREFILL_CHUNK: usize = 32;
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
}

fn decode(
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
        for sequence in 0..settings.batch {
            held[sequence] = greedy(scratch.logits_for(sequence, config.vocab_size))?;
            tokens[sequence].push(held[sequence]);
            positions[sequence] += 1;
        }
        steps_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(DecodeRun {
        elapsed_ms,
        steps_ms,
        tokens,
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
            "excluded_work": "model loading, fingerprinting, prompt preparation, full-range warmup, output comparison, reporting",
            "warmup": "fresh scratch per variant/run; one full untimed decode followed by restart from identical held tokens",
            "termination": "fixed step budget; EOS tokens do not terminate this benchmark",
            "physical_no_sharing_control": "percentage 0 copies genuine common-content KV into distinct physical blocks",
        })
    )?;
    output.flush()?;
    let token_count = settings.batch * settings.steps;
    let mut reference: Option<Vec<Vec<u32>>> = None;
    let mut reference_logits: Option<String> = None;
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
            decode(&weights, &config, &settings, &mut harness, &mut scratch)?;
            scratch.reset_shared_attention_stats();
            let run = decode(&weights, &config, &settings, &mut harness, &mut scratch)?;
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
                    "step_ms": run.steps_ms, "step_distribution_ms": distribution(run.steps_ms.clone()),
                    "generated_tokens": run.tokens, "finish_reason": "fixed_step_budget",
                    "final_logits_sha256": final_logits_sha256,
                    "shared_layer_calls": stats.layer_calls, "shared_query_tokens": stats.query_tokens,
                    "candidate_extra_scratch_bytes": stats.scratch_bytes,
                    "verified_finite_logits": true, "verified_complete_greedy_output": true,
                })
            )?;
            output.flush()?;
        }
    }
    let baseline = distribution(times[0].clone());
    let candidate = distribution(times[1].clone());
    let baseline_ms = baseline["p50"].as_f64().unwrap();
    let candidate_ms = candidate["p50"].as_f64().unwrap();
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
            "timing_samples_retained": settings.repeats * 2,
        })
    )?;
    writeln!(
        output,
        "{}",
        json!({"type": "verification", "passed": true, "complete": true,
        "runs": settings.repeats * 2, "scope": "all timed greedy tokens, finite logits, fixed step budget"})
    )?;
    Ok(())
}
