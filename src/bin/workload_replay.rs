//! Reproducible direct-engine workload replay; see docs/workload-replay.md.
use anyhow::{bail, ensure, Context, Result};
use memmap2::MmapOptions;
use paged_infer::engine::{Engine, EngineConfig};
use paged_infer::model::{LlamaConfig, ModelLoader, Quantization};
use paged_infer::profile::ModelProfile;
use paged_infer::replay::{
    compare_outputs, run, Distribution, ReplayLimits, ReplayReport, Workload,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const HELP: &str = "Usage: workload_replay [OPTIONS]
  --workload PATH       Version 1 JSON trace (default: workloads/mixed.json)
  --model PATH          Safetensors checkpoint; requires adjacent config.json
                        Omitted: bundled tiny synthetic fixture, EOS disabled
  --config PATH         Named engine overrides; repeat to compare configurations
  --repeats N           Runs per configuration, interleaved (default: 3)
  --threads N           Rayon worker count (otherwise respects RAYON_NUM_THREADS)
  --quant f32|int8      Projection storage (default: f32)
  --max-steps N         Maximum engine steps per run (default: 100000)
  --timeout-secs N      Wall-time guard per run (default: 60)
  --output PATH         New JSONL report; omitted writes JSONL to stdout
  --verify              Compare exact tokens and outcomes across every run
  --compare-to PATH     Verify against matching configurations in a prior report
  --help                Show this help
Model loading, fingerprinting and warmup are outside measured run time.
Limits are checked between steps; they cannot interrupt a model forward pass.";

#[derive(Debug)]
struct Args {
    workload: PathBuf,
    model: Option<PathBuf>,
    configs: Vec<PathBuf>,
    repeats: usize,
    threads: Option<usize>,
    quantization: Quantization,
    limits: ReplayLimits,
    output: Option<PathBuf>,
    verify: bool,
    compare_to: Option<PathBuf>,
}

fn parse_args() -> Result<Option<Args>> {
    let mut parsed = Args {
        workload: Path::new(env!("CARGO_MANIFEST_DIR")).join("workloads/mixed.json"),
        model: None,
        configs: Vec::new(),
        repeats: 3,
        threads: None,
        quantization: Quantization::F32,
        limits: ReplayLimits::default(),
        output: None,
        verify: false,
        compare_to: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("{HELP}");
            return Ok(None);
        }
        if flag == "--verify" {
            parsed.verify = true;
            continue;
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--workload" => parsed.workload = value.into(),
            "--model" => parsed.model = Some(value.into()),
            "--config" => parsed.configs.push(value.into()),
            "--repeats" => parsed.repeats = value.parse().context("invalid --repeats")?,
            "--threads" => parsed.threads = Some(value.parse().context("invalid --threads")?),
            "--max-steps" => {
                parsed.limits.max_steps = value.parse().context("invalid --max-steps")?
            }
            "--timeout-secs" => {
                let seconds: f64 = value.parse().context("invalid --timeout-secs")?;
                ensure!(
                    seconds.is_finite() && seconds > 0.0,
                    "--timeout-secs must be positive and finite"
                );
                parsed.limits.timeout =
                    Duration::try_from_secs_f64(seconds).context("invalid --timeout-secs")?;
            }
            "--output" => parsed.output = Some(value.into()),
            "--compare-to" => parsed.compare_to = Some(value.into()),
            "--quant" => {
                parsed.quantization = match value.as_str() {
                    "f32" => Quantization::F32,
                    "int8" => Quantization::Int8,
                    _ => bail!("--quant must be f32 or int8"),
                }
            }
            _ => bail!("unknown argument {flag}; use --help"),
        }
    }
    ensure!(parsed.repeats > 0, "--repeats must be positive");
    ensure!(parsed.threads != Some(0), "--threads must be positive");
    ensure!(parsed.limits.max_steps > 0, "--max-steps must be positive");
    Ok(Some(parsed))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    name: String,
    #[serde(default = "empty_object")]
    engine: Value,
}
fn empty_object() -> Value {
    json!({})
}

#[derive(Serialize)]
struct NamedConfig {
    name: String,
    engine: EngineConfig,
    sha256: String,
    source: Option<PathBuf>,
    source_sha256: Option<String>,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn checked_product(values: &[usize], description: &str) -> Result<usize> {
    let value = values
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .with_context(|| format!("{description} overflows"))?;
    ensure!(
        value <= isize::MAX as usize,
        "{description} exceeds allocation limit"
    );
    Ok(value)
}

fn validate_engine(engine: &EngineConfig, model: &LlamaConfig, synthetic: bool) -> Result<()> {
    ensure!(
        engine.total_blocks > 0 && engine.block_size > 0,
        "total_blocks and block_size must be positive"
    );
    ensure!(
        engine.max_batch_size > 0 && engine.prefill_chunk_size > 0,
        "max_batch_size and prefill_chunk_size must be positive"
    );
    ensure!(
        engine.max_prefill_tokens_per_step > 0,
        "max_prefill_tokens_per_step must be positive"
    );
    ensure!(
        engine.stream_tokens,
        "workload replay requires stream_tokens=true"
    );
    ensure!(
        engine.temperature.is_finite() && engine.temperature >= 0.0,
        "temperature must be finite and nonnegative"
    );
    ensure!(
        engine.top_p.is_finite() && engine.top_p > 0.0 && engine.top_p <= 1.0,
        "top_p must be in (0, 1]"
    );
    ensure!(
        engine.max_context != Some(0),
        "max_context must be positive when present"
    );
    if let (Some(requested), Some(maximum)) = (engine.max_context, model.max_position_embeddings) {
        ensure!(
            requested <= maximum,
            "max_context exceeds checkpoint context window"
        );
    }
    if model.max_position_embeddings.is_some() {
        ensure!(
            engine.max_context.is_some(),
            "cannot remove checkpoint context window"
        );
    }
    for token in engine
        .extra_eos_tokens
        .iter()
        .copied()
        .chain(engine.bos_token)
        .chain(std::iter::once(engine.eos_token))
    {
        ensure!(
            (token as usize) < model.vocab_size || (synthetic && token == u32::MAX),
            "special token {token} outside vocabulary"
        );
    }
    let tokens = checked_product(
        &[engine.total_blocks, engine.block_size],
        "KV token capacity",
    )?;
    ensure!(
        engine.draft_tokens <= tokens,
        "draft_tokens exceeds KV token capacity"
    );
    checked_product(
        &[
            model.num_hidden_layers,
            engine.total_blocks,
            engine.block_size,
            model.num_key_value_heads,
            2,
            model.head_dim(),
            4,
        ],
        "KV cache bytes",
    )?;
    let widest = model
        .vocab_size
        .max(model.intermediate_size)
        .max(model.hidden_size)
        .max(model.kv_dim());
    checked_product(
        &[
            engine.max_batch_size.max(engine.prefill_chunk_size),
            widest,
            4,
        ],
        "batch scratch bytes",
    )?;
    Ok(())
}

fn configs(
    paths: &[PathBuf],
    defaults: &EngineConfig,
    model: &LlamaConfig,
    synthetic: bool,
) -> Result<Vec<NamedConfig>> {
    let mut result = Vec::new();
    let mut names = HashSet::new();
    let inputs = if paths.is_empty() {
        vec![None]
    } else {
        paths.iter().map(Some).collect()
    };
    for path in inputs {
        let bytes = path.map(fs::read).transpose()?;
        let config: ConfigFile = match &bytes {
            Some(bytes) => serde_json::from_slice(bytes).context("parsing engine overrides")?,
            None => ConfigFile {
                name: "baseline".into(),
                engine: empty_object(),
            },
        };
        ensure!(
            !config.name.trim().is_empty(),
            "configuration name must not be empty"
        );
        ensure!(
            names.insert(config.name.clone()),
            "duplicate configuration name {}",
            config.name
        );
        let mut merged = serde_json::to_value(defaults)?;
        let overrides = config
            .engine
            .as_object()
            .context("engine overrides must be a JSON object")?;
        merged
            .as_object_mut()
            .expect("EngineConfig is object")
            .extend(overrides.clone());
        let engine: EngineConfig =
            serde_json::from_value(merged).context("invalid engine overrides")?;
        validate_engine(&engine, model, synthetic)
            .with_context(|| format!("configuration {}", config.name))?;
        result.push(NamedConfig {
            name: config.name,
            sha256: hash(&serde_json::to_vec(&engine)?),
            engine,
            source: path.cloned(),
            source_sha256: bytes.as_deref().map(hash),
        });
    }
    Ok(result)
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn hardware() -> Value {
    let cpu = command("sysctl", &["-n", "machdep.cpu.brand_string"]).or_else(|| {
        fs::read_to_string("/proc/cpuinfo")
            .ok()?
            .lines()
            .find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|text| text.split_once(':'))
                    .map(|(_, name)| name.trim().to_owned())
            })
    });
    json!({
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "uname": command("uname", &["-srm"]), "cpu": cpu,
        "logical_cpus": std::thread::available_parallelism().ok().map(|n| n.get()),
        "rayon_threads": rayon::current_num_threads(),
        "rayon_num_threads_env": std::env::var("RAYON_NUM_THREADS").ok(),
        "simd": paged_infer::simd::backend(),
        "build": {
            "rustc": env!("PAGED_BUILD_RUSTC"),
            "target": env!("PAGED_BUILD_TARGET"),
            "profile": env!("PAGED_BUILD_PROFILE"),
            "opt_level": env!("PAGED_BUILD_OPT_LEVEL"),
            "debug": env!("PAGED_BUILD_DEBUG"),
            "rustflags": env!("PAGED_BUILD_CARGO_ENCODED_RUSTFLAGS").split('\u{1f}').filter(|s| !s.is_empty()).collect::<Vec<_>>(),
            "git_head": env!("PAGED_BUILD_GIT_HEAD"),
            "git_dirty": env!("PAGED_BUILD_GIT_DIRTY"),
            "source_sha256": env!("PAGED_BUILD_SOURCE_SHA256"),
        },
        "rustc_on_path": command("rustc", &["-Vv"]),
        "git_head_at_run": command("git", &["rev-parse", "HEAD"]),
        "git_dirty_at_run": command("git", &["status", "--porcelain"]).map(|s| !s.is_empty()),
        "build_mode": if cfg!(debug_assertions) { "debug_assertions" } else { "optimized_no_debug_assertions" },
        "rustflags_env_at_run": std::env::var("RUSTFLAGS").ok(),
    })
}

#[derive(Deserialize)]
struct PriorRun {
    config: String,
    report: ReplayReport,
}
fn read_prior(path: &Path, identity: &Value) -> Result<Vec<PriorRun>> {
    let file = BufReader::new(File::open(path)?);
    let mut runs = Vec::new();
    let mut found_manifest = false;
    let mut complete = false;
    for line in file.lines() {
        let value: Value = serde_json::from_str(&line?)?;
        match value.get("type").and_then(Value::as_str) {
            Some("manifest") => {
                ensure!(!found_manifest, "prior report has multiple manifests");
                ensure!(
                    value.get("version") == Some(&json!(1)),
                    "unsupported prior report version"
                );
                ensure!(
                    value.get("identity") == Some(identity),
                    "prior report model, workload or quantization fingerprint differs"
                );
                found_manifest = true;
            }
            Some("run") => runs.push(serde_json::from_value(value)?),
            Some("verification") => {
                ensure!(
                    value.get("passed") != Some(&json!(false)),
                    "prior report failed output verification"
                );
                complete = value.get("complete") == Some(&json!(true));
            }
            _ => {}
        }
    }
    ensure!(
        found_manifest && complete && !runs.is_empty(),
        "prior report needs a manifest, runs and successful completion record"
    );
    Ok(runs)
}

fn distribution(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let rank = |p: f64| {
        (!values.is_empty())
            .then(|| values[((values.len() as f64 * p).ceil() as usize).saturating_sub(1)])
    };
    json!({"count": values.len(), "p50": rank(0.5), "p95": rank(0.95), "p99": rank(0.99), "max": values.last()})
}
fn pooled_latencies(reports: &[ReplayReport]) -> Value {
    let mut dispatch = Vec::new();
    let mut cancellation_dispatch = Vec::new();
    let mut ttft = Vec::new();
    let mut queue = Vec::new();
    let mut end_to_end = Vec::new();
    let mut inter_token = Vec::new();
    let mut inter_delivery = Vec::new();
    for request in reports.iter().flat_map(|report| &report.requests) {
        dispatch.extend(request.dispatch_lag_ms);
        cancellation_dispatch.extend(request.cancel_dispatch_lag_ms);
        for sequence in &request.sequences {
            ttft.extend(sequence.ttft_ms);
            queue.push(sequence.engine_queue_ms);
            end_to_end.push(sequence.end_to_end_ms);
            let mut previous = None;
            for delivery in &sequence.deliveries {
                if delivery.tokens == 0 {
                    continue;
                }
                if let Some(at) = previous {
                    let gap = delivery.at_ms - at;
                    inter_token.push(gap);
                    inter_delivery.push(gap);
                }
                inter_token.extend(std::iter::repeat_n(0.0, delivery.tokens - 1));
                previous = Some(delivery.at_ms);
            }
        }
    }
    json!({
        "dispatch_lag_ms": Distribution::from_samples(dispatch),
        "cancel_dispatch_lag_ms": Distribution::from_samples(cancellation_dispatch),
        "ttft_ms": Distribution::from_samples(ttft),
        "engine_queue_ms": Distribution::from_samples(queue),
        "end_to_end_ms": Distribution::from_samples(end_to_end),
        "observed_inter_token_ms": Distribution::from_samples(inter_token),
        "inter_delivery_ms": Distribution::from_samples(inter_delivery),
    })
}
fn write_record(writer: &mut dyn Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    let synthetic = args.model.is_none();
    let model_path = args.model.clone().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_llama.safetensors")
    });
    ensure!(
        model_path.is_file(),
        "model checkpoint not found: {}",
        model_path.display()
    );
    let config_path = model_path.with_file_name("config.json");
    let mut model = LlamaConfig::from_hf_config(&config_path)?;
    model.quantization = args.quantization;
    let mut defaults = if synthetic {
        EngineConfig {
            eos_token: u32::MAX,
            extra_eos_tokens: Vec::new(),
            bos_token: None,
            ..EngineConfig::default()
        }
    } else {
        ModelProfile::from_parts(model.clone(), &model_path, None)?.engine_config()
    };
    defaults.stream_tokens = true;
    let configs = configs(&args.configs, &defaults, &model, synthetic)?;
    let workload_bytes =
        fs::read(&args.workload).with_context(|| format!("reading {}", args.workload.display()))?;
    let workload: Workload = serde_json::from_slice(&workload_bytes).context("parsing workload")?;
    workload.validate()?;
    let generation_config = model_path.with_file_name("generation_config.json");
    let identity = json!({
        "weights_sha256": hash_file(&model_path)?,
        "model_config_sha256": hash_file(&config_path)?,
        "generation_config_sha256": if generation_config.exists() { Some(hash_file(&generation_config)?) } else { None },
        "workload_sha256": hash(&workload_bytes),
        "quantization": format!("{:?}", args.quantization),
        "synthetic_fixture": synthetic,
    });
    let prior = args
        .compare_to
        .as_deref()
        .map(|path| read_prior(path, &identity))
        .transpose()?;
    if let Some(prior) = &prior {
        for config in &configs {
            ensure!(
                prior.iter().any(|run| run.config == config.name),
                "prior report has no configuration {}",
                config.name
            );
        }
    }
    let mmap_file = File::open(&model_path)?;
    // The mapping remains alive while every sequential engine borrows its tensors.
    let mmap = unsafe { MmapOptions::new().map(&mmap_file)? };
    let loader = ModelLoader::new(&mmap)?;
    let mut writer: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| format!("creating new report {}", path.display()))?,
        )),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };
    write_record(
        &mut writer,
        &json!({
            "type": "manifest", "version": 1,
            "profiling_enabled": paged_infer::profiling::enabled(),
            "performance_gate_eligible": !paged_infer::profiling::enabled(),
            "profiling_note": "Instrumented builds contain diagnostic timer/counter overhead; exclude their timings from performance gates.",
            "started_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            "identity": identity, "model_path": model_path, "workload_path": args.workload,
            "workload": workload, "configs": configs, "repeats": args.repeats,
            "max_steps": args.limits.max_steps, "timeout_ms": args.limits.timeout.as_secs_f64() * 1000.0,
            "environment": hardware(),
            "measurement": "direct engine; elapsed starts after model load, hashing, allocation and warmup",
        }),
    )?;
    let mut reports: Vec<Vec<ReplayReport>> = (0..configs.len()).map(|_| Vec::new()).collect();
    let mut verification_errors = Vec::new();
    let mut first_report = None;
    for repeat in 0..args.repeats {
        // Rotate the first configuration each round, avoiding a fixed first-run advantage.
        for slot in 0..configs.len() {
            let index = (slot + repeat) % configs.len();
            let config = &configs[index];
            eprintln!(
                "replay {} repeat {}/{}",
                config.name,
                repeat + 1,
                args.repeats
            );
            // Only one converted weight set and one KV pool exist at a time.
            // Loading and warmup are deliberately outside replay::run's timer.
            let weights = loader.load_weights(&model)?;
            let weight_bytes = weights.weight_bytes();
            let mut engine = Engine::new(weights, model.clone(), config.engine.clone());
            engine.warm_up();
            engine.reset();
            let report = run(&mut engine, &workload, &args.limits)
                .with_context(|| format!("{} repeat {}", config.name, repeat + 1))?;
            if args.verify {
                if let Some(expected) = &first_report {
                    if let Err(error) = compare_outputs(expected, &report) {
                        verification_errors.push(format!(
                            "{} repeat {}: {error:#}",
                            config.name,
                            repeat + 1
                        ));
                    }
                } else {
                    first_report = Some(report.clone());
                }
            }
            if let Some(prior) = &prior {
                for expected in prior.iter().filter(|run| run.config == config.name) {
                    if let Err(error) = compare_outputs(&expected.report, &report) {
                        verification_errors.push(format!(
                            "prior {} / repeat {}: {error:#}",
                            config.name,
                            repeat + 1
                        ));
                    }
                }
            }
            write_record(
                &mut writer,
                &json!({"type": "run", "config": config.name, "repeat": repeat + 1, "order_in_repeat": slot + 1, "weight_bytes": weight_bytes, "report": report}),
            )?;
            reports[index].push(report);
        }
    }
    for (config, reports) in configs.iter().zip(&reports) {
        let values: Vec<Value> = reports
            .iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<_, _>>()?;
        let mut metrics = serde_json::Map::new();
        for field in [
            "elapsed_ms",
            "summary/completed_requests_per_second",
            "summary/useful_tokens_per_second",
            "summary/ttft_ms/p50",
            "summary/ttft_ms/p95",
            "summary/ttft_ms/p99",
            "summary/end_to_end_ms/p95",
            "summary/observed_inter_token_ms/p95",
            "summary/dispatch_lag_ms/p95",
            "engine/prefill_chunks",
            "engine/prefill_preemptions",
            "engine/recomputed_tokens",
            "engine/max_observed_prefilling_requests",
            "engine/max_observed_pending_prefill_tokens",
        ] {
            let pointer = format!("/{field}");
            metrics.insert(
                field.replace('/', "."),
                distribution(
                    values
                        .iter()
                        .filter_map(|v| v.pointer(&pointer).and_then(Value::as_f64))
                        .collect(),
                ),
            );
        }
        write_record(
            &mut writer,
            &json!({"type": "config_summary", "config": config.name, "repeats": reports.len(), "run_metric_distributions": metrics, "pooled_latency_distributions": pooled_latencies(reports), "aggregation": "nearest-rank over every repeat; run_metric_distributions aggregates per-run values, pooled_latency_distributions pools observations from all repeats"}),
        )?;
    }
    write_record(
        &mut writer,
        &json!({"type": "verification", "enabled": args.verify || args.compare_to.is_some(), "complete": true, "passed": (args.verify || args.compare_to.is_some()).then_some(verification_errors.is_empty()), "errors": verification_errors}),
    )?;
    ensure!(
        verification_errors.is_empty(),
        "output verification failed; inspect verification record"
    );
    if let Some(path) = args.output {
        eprintln!("report: {}", path.display());
    }
    Ok(())
}
