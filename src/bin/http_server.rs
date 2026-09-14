//! OpenAI-shaped HTTP front end backed by the real engine.
//!
//! All of the serving logic lives in `paged_infer::serve`, where it is tested
//! over real sockets against the fixture model. This binary only turns the
//! environment into a [`ServeConfig`], loads the checkpoint, and runs.
//!
//! | variable          | default                                    |
//! |-------------------|--------------------------------------------|
//! | `HOST`, `PORT`    | `127.0.0.1`, `8080`                        |
//! | `MODEL_PATH`      | `models/tinyllama-1.1b/model.safetensors`  |
//! | `TOKENIZER_PATH`  | `models/tinyllama-1.1b/tokenizer.json`     |
//! | `MODEL_NAME`      | `paged-infer` (what `/v1/models` reports)  |
//! | `QUANT`           | `f32`; `int8` quantizes the projections    |
//! | `KV_BLOCKS`       | `512` blocks of `BLOCK_SIZE` (`16`) tokens |
//! | `DRAFT_TOKENS`    | `0`; `>0` enables speculative decoding     |
//! | `WARMUP`          | `1`; `0` skips the warm-up pass            |
//! | `MAX_BODY_BYTES`  | `1048576`                                  |
//! | `MAX_CONNECTIONS` | `256`                                      |
//! | `MAX_QUEUED_JOBS` | `1024`                                     |
//! | `MAX_JOBS_PER_STEP` | `32`, received jobs before scheduling    |
//! | `PREFILL_TOKENS_PER_STEP` | `32`, prefill positions per step   |
//! | `PREFILL_CHUNK_SIZE` | `32`, positions per matrix batch       |
//! | `TIMEOUT_SECS`    | `30`, socket read and write deadlines      |
//! | `MAX_TOKENS`      | `4096`, cap on a request's `max_tokens`    |

use std::path::Path;
use std::time::Duration;

use memmap2::MmapOptions;
use paged_infer::engine::EngineConfig;
use paged_infer::model::{LlamaConfig, ModelLoader, Quantization};
use paged_infer::profile::ModelProfile;
use paged_infer::serve::{self, ServeConfig};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_usize(key: &str, default: usize) -> anyhow::Result<usize> {
    match std::env::var(key) {
        Ok(v) => v
            .parse()
            .map_err(|e| anyhow::anyhow!("{key}={v:?} is not a valid integer: {e}")),
        Err(_) => Ok(default),
    }
}

fn main() -> anyhow::Result<()> {
    let host = env_or("HOST", "127.0.0.1");
    let port = env_or("PORT", "8080");
    let model_path = env_or("MODEL_PATH", "models/tinyllama-1.1b/model.safetensors");
    let tokenizer_path = env_or("TOKENIZER_PATH", "models/tinyllama-1.1b/tokenizer.json");
    let model_name = env_or("MODEL_NAME", "paged-infer");

    if !Path::new(&model_path).exists() {
        anyhow::bail!(
            "MODEL_PATH not found ({model_path}).\n\
             Fetch weights with `python3 scripts/download_model.py`, or point MODEL_PATH at \n\
             a HuggingFace-format Llama .safetensors checkpoint with its config.json beside it."
        );
    }
    let tokenizer = Path::new(&tokenizer_path).exists();
    if !tokenizer {
        eprintln!(
            "warning: no tokenizer at {tokenizer_path}; only prompt_tokens requests will work"
        );
    }

    // The mapping lives as long as the process: the engine thread borrows the
    // weights out of it for the whole run.
    let file = std::fs::File::open(&model_path)?;
    let mmap: &'static memmap2::Mmap =
        Box::leak(Box::new(unsafe { MmapOptions::new().map(&file)? }));
    let loader: &'static ModelLoader<'static> = Box::leak(Box::new(ModelLoader::new(mmap)?));

    let quantization = match std::env::var("QUANT").as_deref() {
        Ok("int8") => Quantization::Int8,
        _ => Quantization::F32,
    };
    // Architecture, special tokens, context window and chat template all come
    // from the checkpoint's own files.
    let config = LlamaConfig {
        quantization,
        ..LlamaConfig::beside_checkpoint(&model_path)?
    };
    let profile = ModelProfile::from_parts(
        config,
        Path::new(&model_path),
        tokenizer.then(|| Path::new(&tokenizer_path)),
    )?;
    let weights = loader.load_weights(&profile.config)?;

    let timeout = Duration::from_secs(env_usize("TIMEOUT_SECS", 30)? as u64);
    let engine = EngineConfig {
        total_blocks: env_usize("KV_BLOCKS", 512)?,
        block_size: env_usize("BLOCK_SIZE", 16)?,
        max_prefill_tokens_per_step: env_usize("PREFILL_TOKENS_PER_STEP", 32)?,
        prefill_chunk_size: env_usize("PREFILL_CHUNK_SIZE", 32)?,
        // Speculative decoding is off unless asked for: it only pays on
        // copy-heavy workloads, and costs a little on the rest. See
        // `speculative_benchmark` for the trade.
        draft_tokens: env_usize("DRAFT_TOKENS", 0)?,
        stream_tokens: true,
        ..profile.engine_config()
    };
    let serve_config = ServeConfig {
        addr: format!("{host}:{port}"),
        model_name: model_name.clone(),
        max_body_bytes: env_usize("MAX_BODY_BYTES", 1024 * 1024)?,
        max_connections: env_usize("MAX_CONNECTIONS", 256)?,
        max_queued_jobs: env_usize("MAX_QUEUED_JOBS", 1024)?,
        max_jobs_per_step: env_usize("MAX_JOBS_PER_STEP", 32)?,
        read_timeout: timeout,
        write_timeout: timeout,
        max_tokens_limit: env_usize("MAX_TOKENS", 4096)?,
        warm_up: std::env::var("WARMUP").as_deref() != Ok("0"),
        engine,
        ..ServeConfig::default()
    };

    println!(
        "Model: {} layers, {:.2} GB of {:?} weights; bos={:?} eos={:?} context={}; chat template: {}",
        profile.config.num_hidden_layers,
        weights.weight_bytes() as f64 / 1e9,
        quantization,
        profile.bos_token,
        profile.eos_tokens,
        profile
            .max_context
            .map_or("unlimited".to_string(), |c| c.to_string()),
        if profile.has_chat_template() {
            "yes"
        } else {
            "no (chat requests will be refused)"
        }
    );

    let server = serve::run(serve_config, profile, weights)?;
    println!(
        "Paged-Infer serving {model_name} on http://{}",
        server.addr()
    );
    println!("  GET  /health  /v1/models  /metrics (prometheus)  /stats (json)");
    println!("  POST /v1/completions  /v1/chat/completions   (add \"stream\": true for SSE)");
    server.wait();
    Ok(())
}
