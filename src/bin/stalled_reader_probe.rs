//! A real-socket slow-reader probe, using only the checked-in tiny checkpoint.
//!
//! Run `cargo run --release --bin stalled_reader_probe -- --help` for controls.
//! The client consumes one SSE frame, then keeps its socket open without
//! reading. Synthetic padding in the model name makes OS socket buffers fill
//! with hundreds of fixture tokens instead of a costly long-context run.
//! This is a transport stress probe, not a model throughput benchmark.
//!
//! No production instrumentation distinguishes a full reply queue from a
//! socket write timeout, so the JSON reports that uncertainty explicitly.
//! Passing requires early termination while the client still holds its socket,
//! a complete identical draining control and healthy follower, and all KV blocks
//! returned. An oversized OS
//! buffer or a deadline produces an inconclusive/failing result, never a false
//! claim that backpressure was exercised. Run separately from timing benchmarks.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use paged_infer::engine::EngineConfig;
use paged_infer::model::{LlamaConfig, ModelLoader};
use paged_infer::profile::ModelProfile;
use paged_infer::serve::{run, ServeConfig};
use serde_json::{json, Value};

#[derive(Debug)]
struct Options {
    tokens: usize,
    padding_bytes: usize,
    pending_events: usize,
    write_timeout_ms: u64,
    deadline_ms: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tokens: 512,
            padding_bytes: 64 * 1024,
            pending_events: 32,
            write_timeout_ms: 200,
            deadline_ms: 15_000,
        }
    }
}

fn options() -> Result<Option<Options>> {
    let mut out = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!(
                "Usage: stalled_reader_probe [--tokens N] [--padding-bytes N] \
                 [--pending-events N] [--write-timeout-ms N] [--deadline-ms N]\n\
                 Defaults: 512 tokens, 65536 padding bytes/frame, 32 pending events, \
                 200 ms write timeout, 15000 ms deadline.\n\
                 Uses the tiny fixture on ephemeral loopback. Keeps the stalled socket open \
                 until cancellation/reclamation are observed. Emits JSON; exit 0 means \
                 the invariants passed, 1 means failure or insufficient pressure evidence.\n\
                 Padding is synthetic: timings describe this transport probe only. \
                 Override RAYON_NUM_THREADS to change the default one worker.\n\
                 To exclude reply-queue overflow as the early-termination cause: \
                 --tokens 4096 --pending-events 4096 --write-timeout-ms 20.\n\
                 --padding-bytes 0 is a useful negative control; a normal completion \
                 is reported as insufficient pressure evidence, with exit 1."
            );
            return Ok(None);
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--tokens" => out.tokens = value.parse()?,
            "--padding-bytes" => out.padding_bytes = value.parse()?,
            "--pending-events" => out.pending_events = value.parse()?,
            "--write-timeout-ms" => out.write_timeout_ms = value.parse()?,
            "--deadline-ms" => out.deadline_ms = value.parse()?,
            _ => bail!("unknown option {arg}; use --help"),
        }
    }
    ensure!((64..=4096).contains(&out.tokens), "tokens must be 64..4096");
    ensure!(out.padding_bytes <= 1024 * 1024, "padding must be <= 1 MiB");
    ensure!(
        (4..=4096).contains(&out.pending_events),
        "pending events must be 4..4096"
    );
    ensure!(
        (1..=5000).contains(&out.write_timeout_ms),
        "write timeout must be 1..5000 ms"
    );
    ensure!(
        (1000..=60_000).contains(&out.deadline_ms),
        "deadline must be 1000..60000 ms"
    );
    Ok(Some(out))
}

fn timeout(deadline: Instant) -> Result<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    ensure!(!left.is_zero(), "probe deadline elapsed");
    Ok(left.min(Duration::from_secs(2)))
}

fn connect(addr: SocketAddr, deadline: Instant) -> Result<TcpStream> {
    let stream = TcpStream::connect_timeout(&addr, timeout(deadline)?)?;
    stream.set_read_timeout(Some(timeout(deadline)?))?;
    stream.set_write_timeout(Some(timeout(deadline)?))?;
    Ok(stream)
}

fn request(stream: &mut TcpStream, method: &str, path: &str, body: &str) -> Result<()> {
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: probe\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}

fn http(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: &str,
    deadline: Instant,
) -> Result<Value> {
    let mut stream = connect(addr, deadline)?;
    request(&mut stream, method, path, body)?;
    let mut response = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        stream.set_read_timeout(Some(timeout(deadline)?))?;
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        response.extend_from_slice(&chunk[..n]);
        ensure!(
            response.len() <= 2 * 1024 * 1024,
            "HTTP reply exceeded probe bound"
        );
    }
    let text = String::from_utf8(response)?;
    let (head, body) = text
        .split_once("\r\n\r\n")
        .context("missing HTTP header terminator")?;
    ensure!(
        head.starts_with("HTTP/1.1 200 "),
        "unexpected status: {}",
        head.lines().next().unwrap_or(head)
    );
    Ok(serde_json::from_str(body)?)
}

fn read_head(reader: &mut BufReader<TcpStream>, deadline: Instant) -> Result<()> {
    let mut head = String::new();
    loop {
        reader
            .get_ref()
            .set_read_timeout(Some(timeout(deadline)?))?;
        let mut line = String::new();
        ensure!(
            reader.read_line(&mut line)? > 0,
            "stream closed before response headers"
        );
        head.push_str(&line);
        ensure!(head.len() <= 16 * 1024, "oversized response headers");
        if line == "\r\n" {
            break;
        }
    }
    ensure!(head.starts_with("HTTP/1.1 200 "), "stream rejected: {head}");
    Ok(())
}

fn read_frame(
    reader: &mut BufReader<TcpStream>,
    deadline: Instant,
    bound: usize,
) -> Result<Vec<u8>> {
    let mut frame = Vec::new();
    loop {
        reader
            .get_ref()
            .set_read_timeout(Some(timeout(deadline)?))?;
        let before = frame.len();
        ensure!(
            reader.read_until(b'\n', &mut frame)? > 0,
            "stream ended before a complete frame"
        );
        ensure!(frame.len() <= bound, "frame exceeded probe bound");
        if &frame[before..] == b"\n" {
            break;
        }
    }
    Ok(frame)
}

fn read_first_frame(
    reader: &mut BufReader<TcpStream>,
    deadline: Instant,
    bound: usize,
) -> Result<usize> {
    read_head(reader, deadline)?;
    let frame = read_frame(reader, deadline, bound)?;
    let text = std::str::from_utf8(&frame)?;
    let payload = text
        .strip_prefix("data: ")
        .context("missing SSE data prefix")?;
    let value: Value = serde_json::from_str(payload.trim())?;
    ensure!(
        value["tokens"].as_array().is_some_and(|t| !t.is_empty()),
        "first frame had no generated tokens"
    );
    Ok(frame.len())
}

fn draining_control(
    addr: SocketAddr,
    body: &str,
    options: &Options,
    deadline: Instant,
) -> Result<Value> {
    let started = Instant::now();
    let mut stream = connect(addr, deadline)?;
    request(&mut stream, "POST", "/v1/completions", body)?;
    let mut reader = BufReader::with_capacity(32 * 1024, stream);
    read_head(&mut reader, deadline)?;
    let mut tokens = 0;
    let mut bytes = 0;
    let mut finish = None;
    let mut usage = None;
    loop {
        let frame = read_frame(&mut reader, deadline, options.padding_bytes + 8192)?;
        bytes += frame.len();
        let payload = std::str::from_utf8(&frame)?
            .strip_prefix("data: ")
            .context("missing SSE data prefix")?
            .trim();
        if payload == "[DONE]" {
            break;
        }
        let value: Value = serde_json::from_str(payload)?;
        ensure!(
            value.get("error").is_none(),
            "control stream failed: {value}"
        );
        tokens += value["tokens"].as_array().map_or(0, Vec::len);
        if let Some(reason) = value["choices"][0]["finish_reason"].as_str() {
            finish = Some(reason.to_string());
        }
        if let Some(count) = value["usage"]["completion_tokens"].as_u64() {
            usage = Some(count);
        }
    }
    Ok(json!({
        "complete": tokens == options.tokens && usage == Some(options.tokens as u64) && finish.as_deref() == Some("length"),
        "tokens": tokens, "usage_tokens": usage, "finish_reason": finish,
        "wire_bytes": bytes, "latency_ms": started.elapsed().as_secs_f64() * 1000.0,
        "same_request_and_server_limits": true
    }))
}

fn count(stats: &Value, name: &str) -> u64 {
    stats[name].as_u64().expect("validated by read_stats")
}

fn read_stats(addr: SocketAddr, deadline: Instant) -> Result<Value> {
    let stats = http(addr, "GET", "/stats", "", deadline)?;
    for name in [
        "generated_tokens",
        "sequences_active",
        "requests_queued",
        "kv_blocks_free",
        "kv_blocks_total",
        "connections",
    ] {
        ensure!(
            stats[name].as_u64().is_some(),
            "stats omitted unsigned metric {name}"
        );
    }
    Ok(stats)
}

fn idle(stats: &Value) -> bool {
    count(stats, "sequences_active") == 0
        && count(stats, "requests_queued") == 0
        && count(stats, "kv_blocks_free") == count(stats, "kv_blocks_total")
}

fn probe(options: &Options) -> Result<Value> {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut config = LlamaConfig::from_hf_config(fixtures.join("config.json"))?;
    config.eos_token_ids = vec![111];
    let bytes: &'static [u8] =
        Box::leak(std::fs::read(fixtures.join("tiny_llama.safetensors"))?.into_boxed_slice());
    let loader: &'static ModelLoader<'static> = Box::leak(Box::new(ModelLoader::new(bytes)?));
    let weights = loader.load_weights(&config)?;
    let profile = ModelProfile::from_parts(config, &fixtures.join("tiny_llama.safetensors"), None)?;
    // Capacity covers both entire generations: OOM cannot masquerade as
    // cancellation. Disabling EOS/context/prefix retention removes the other
    // normal reasons for an early stop or blocks remaining after completion.
    let total_blocks = (options.tokens + 32).div_ceil(8);
    let server = run(
        ServeConfig {
            addr: "127.0.0.1:0".into(),
            model_name: format!("fixture-probe-{}", "x".repeat(options.padding_bytes)),
            max_connections: 8,
            max_queued_jobs: 8,
            max_pending_events: options.pending_events,
            max_tokens_limit: options.tokens,
            read_timeout: Duration::from_secs(2),
            write_timeout: Duration::from_millis(options.write_timeout_ms),
            warm_up: false,
            engine: EngineConfig {
                total_blocks,
                block_size: 8,
                eos_token: u32::MAX,
                extra_eos_tokens: Vec::new(),
                bos_token: None,
                max_context: None,
                enable_prefix_cache: false,
                ..profile.engine_config()
            },
            ..ServeConfig::default()
        },
        profile,
        weights,
    )?;
    let started = Instant::now();
    let deadline = started + Duration::from_millis(options.deadline_ms);
    while http(server.addr(), "GET", "/health", "", deadline).is_err() {
        timeout(deadline)?;
        std::thread::sleep(Duration::from_millis(10));
    }
    let stalled_body = json!({
        "prompt_tokens": [5, 6, 7, 8], "max_tokens": options.tokens,
        "temperature": 0, "stream": true
    })
    .to_string();
    // An identical full-draining client must succeed under these very same
    // queue limits. Otherwise serializer/consumer lag alone could explain the
    // early termination, without any effect from stopping socket reads.
    let control = draining_control(server.addr(), &stalled_body, options, deadline)?;
    let baseline = loop {
        let stats = read_stats(server.addr(), deadline)?;
        if idle(&stats) && count(&stats, "connections") <= 1 {
            break stats;
        }
        timeout(deadline)?;
        std::thread::sleep(Duration::from_millis(10));
    };
    let stalled_started = Instant::now();
    let mut stalled = connect(server.addr(), deadline)?;
    request(&mut stalled, "POST", "/v1/completions", &stalled_body)?;
    // BufReader can read ahead up to its capacity; report it and retain the
    // whole reader without touching it until after the observations below.
    let mut stalled = BufReader::with_capacity(1024, stalled);
    let first_frame_bytes = read_first_frame(&mut stalled, deadline, options.padding_bytes + 8192)?;
    let stop_reading_at = Instant::now();
    let read_ahead_bytes = stalled.buffer().len();
    let at_stop = read_stats(server.addr(), deadline)?;
    let follower_started = Instant::now();
    let follower = http(
        server.addr(),
        "POST",
        "/v1/completions",
        &json!({
            "prompt_tokens": [9, 10, 11, 12], "max_tokens": 8, "temperature": 0
        })
        .to_string(),
        deadline,
    )?;
    let follower_ms = follower_started.elapsed().as_secs_f64() * 1000.0;
    let follower_tokens = follower["choices"][0]["tokens"]
        .as_array()
        .map_or(0, Vec::len);
    let follower_complete =
        follower_tokens == 8 && follower["choices"][0]["finish_reason"] == "length";
    let mut final_stats;
    loop {
        final_stats = read_stats(server.addr(), deadline)?;
        // The stats connection itself accounts for one handler. Waiting for
        // the slow handler to leave also exercises bounded socket cleanup.
        if idle(&final_stats) && count(&final_stats, "connections") <= 1 {
            break;
        }
        timeout(deadline)?;
        std::thread::sleep(Duration::from_millis(10));
    }
    let reclaimed_ms = stop_reading_at.elapsed().as_secs_f64() * 1000.0;
    let stalled_generated = count(&final_stats, "generated_tokens")
        .checked_sub(count(&baseline, "generated_tokens"))
        .and_then(|n| n.checked_sub(follower_tokens as u64))
        .context("generated-token counters do not cover the control and follower")?;
    let early_termination = stalled_generated < options.tokens as u64;
    let active_when_stopped = count(&at_stop, "sequences_active") > 0;
    // This is the first client-side close after the initial frame. All
    // cancellation and reclamation evidence was collected with the socket held.
    drop(stalled);
    let shutdown_started = Instant::now();
    server.shutdown();
    let success = control["complete"] == true
        && active_when_stopped
        && early_termination
        && follower_complete
        && idle(&final_stats);
    Ok(json!({
        "probe": "stalled_reader", "schema_version": 1, "success": success,
        "outcome": if success { "early_termination_while_socket_held" } else { "insufficient_pressure_evidence_or_follower_failure" },
        "configuration": {
            "fixture": "tiny_llama", "requested_tokens": options.tokens,
            "synthetic_model_name_padding_bytes": options.padding_bytes,
            "max_pending_events": options.pending_events, "write_timeout_ms": options.write_timeout_ms,
            "deadline_ms": options.deadline_ms, "kv_blocks": total_blocks,
            "prefix_cache": false, "eos": "disabled", "context_limit": null
        },
        "evidence": {
            "socket_held_open_until_reclaimed": true, "client_closed_before_observation": false,
            "active_when_reading_stopped": active_when_stopped,
            "first_frame_bytes": first_frame_bytes, "read_ahead_bytes": read_ahead_bytes,
            "stalled_generated_tokens": stalled_generated, "terminated_before_budget": early_termination,
            "all_kv_released": idle(&final_stats),
            "reply_queue_overflow_possible": options.pending_events < options.tokens,
            "cancellation_cause": if !early_termination { "none observed" } else if options.pending_events >= options.tokens {
                "socket write failure inferred: queue holds every generated delta; timeout error kind is not instrumented"
            } else { "inferred: reply queue saturation or socket write failure; no per-cause counters" },
            "socket_write_timeout_confirmed": false
        },
        "timing_ms": {
            "first_frame_from_request": stop_reading_at.duration_since(stalled_started).as_secs_f64() * 1000.0,
            "healthy_follower": follower_ms, "reclamation_after_stop_reading": reclaimed_ms,
            "shutdown": shutdown_started.elapsed().as_secs_f64() * 1000.0
        },
        "healthy_follower": {"complete": follower_complete, "tokens": follower_tokens,
            "finish_reason": follower["choices"][0]["finish_reason"]},
        "draining_control": control, "stats_before_stalled_request": baseline,
        "stats_at_stop_reading": at_stop, "final_stats": final_stats
    }))
}

fn main() -> Result<()> {
    let Some(options) = options()? else {
        return Ok(());
    };
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    ensure!(threads > 0, "RAYON_NUM_THREADS must be positive");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    match probe(&options) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report)?);
            ensure!(
                report["success"] == true,
                "probe invariants failed; see JSON evidence"
            );
            Ok(())
        }
        Err(error) => {
            println!(
                "{}",
                json!({"probe": "stalled_reader", "schema_version": 1,
                "success": false, "outcome": "probe_error", "error": format!("{error:#}")})
            );
            Err(error)
        }
    }
}
