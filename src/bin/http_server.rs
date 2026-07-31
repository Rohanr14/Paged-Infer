//! OpenAI-shaped HTTP front end backed by the real engine.
//!
//! The engine owns one thread and runs its scheduling loop there; handler
//! threads push jobs onto a channel and block on a reply. That layout is the
//! point: requests arriving from *different* clients land in the same batch and
//! decode together, and two clients sending the same system prompt hit the
//! prefix cache. A server that took a lock around the engine per request would
//! serialize them and get none of it.
//!
//! ```text
//!   client A ─┐
//!   client B ─┼─► job channel ─► engine thread: admit → step → complete ─┐
//!   client C ─┘                        ▲                                 │
//!                                      └───────── replies ◄──────────────┘
//! ```
//!
//! | route                       | method | purpose                          |
//! |-----------------------------|--------|----------------------------------|
//! | `/health`                   | GET    | liveness                         |
//! | `/v1/models`                | GET    | the loaded model                 |
//! | `/metrics`                  | GET    | Prometheus text exposition       |
//! | `/stats`                    | GET    | the same counters as JSON        |
//! | `/v1/completions`           | POST   | `prompt` or `prompt_tokens`      |
//! | `/v1/chat/completions`      | POST   | `messages`                       |
//!
//! `prompt_tokens` takes pre-tokenized input, so the server is usable for
//! load testing without a tokenizer on hand. Both completion routes accept
//! `"stream": true` and reply with OpenAI-shaped server-sent events.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use memmap2::MmapOptions;
use paged_infer::detokenizer::IncrementalDetokenizer;
use paged_infer::engine::{Completion, Engine, EngineConfig, FinishReason};
use paged_infer::model::{LlamaConfig, ModelLoader, Quantization};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

/// One unit of work for the engine thread.
struct Job {
    tokens: Vec<u32>,
    max_tokens: usize,
    temperature: Option<f32>,
    samples: usize,
    reply: Sender<Event>,
    /// Raised by the handler when its client stops reading.
    cancel: Arc<AtomicBool>,
}

/// What the engine thread sends back over a job's reply channel.
///
/// Both response shapes ride the same channel: a streaming handler forwards
/// each `Delta` as it arrives, and a buffered one ignores them and waits for
/// `Done`. The engine thread does not need to know which kind of client it is
/// serving.
enum Event {
    Delta {
        /// Which of an `n>1` request's choices this belongs to.
        index: usize,
        tokens: Vec<u32>,
        finish_reason: Option<FinishReason>,
    },
    Done(Vec<Completion>),
}

/// Engine counters, refreshed by the engine thread after every step.
#[derive(Default, Clone)]
struct Metrics {
    prompt_tokens: usize,
    prompt_tokens_prefilled: usize,
    generated_tokens: usize,
    prefix_hits: u64,
    prefix_lookups: u64,
    prefix_tokens_saved: u64,
    cow_copies: u64,
    decode_tok_s: f64,
    steps: usize,
    requests: usize,
    kv_blocks_total: usize,
    kv_blocks_free: usize,
}

/// A request in flight, held until every sibling sample of it has finished.
struct Pending {
    request_id: usize,
    samples: usize,
    reply: Sender<Event>,
    done: Vec<Completion>,
    /// Sequence ids in the order the engine created them, so a streaming client
    /// sees stable `index` values across chunks.
    order: Vec<usize>,
    cancel: Arc<AtomicBool>,
}

impl Pending {
    /// Position of a sequence within its request, assigned on first sight.
    fn index_of(&mut self, sequence_id: usize) -> usize {
        if let Some(i) = self.order.iter().position(|&s| s == sequence_id) {
            return i;
        }
        self.order.push(sequence_id);
        self.order.len() - 1
    }
}

fn main() -> anyhow::Result<()> {
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/model.safetensors".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/tinyllama-1.1b/tokenizer.json".to_string());
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "paged-infer".to_string());

    if !std::path::Path::new(&model_path).exists() {
        anyhow::bail!(
            "MODEL_PATH not found ({model_path}).\n\
             Fetch weights with `python3 scripts/download_model.py`, or point MODEL_PATH at \n\
             any HuggingFace-format Llama .safetensors checkpoint."
        );
    }

    // The tokenizer is immutable, so handler threads share one copy and the
    // engine thread never needs it — it works on token ids.
    let tokenizer = Tokenizer::from_file(&tokenizer_path).ok().map(Arc::new);
    if tokenizer.is_none() {
        eprintln!(
            "warning: no tokenizer at {tokenizer_path}; only prompt_tokens requests will work"
        );
    }

    let (tx, rx) = channel::<Job>();
    let metrics = Arc::new(Mutex::new(Metrics::default()));
    let ready = Arc::new(AtomicBool::new(false));

    let engine_metrics = Arc::clone(&metrics);
    let engine_ready = Arc::clone(&ready);
    let engine_thread = std::thread::Builder::new()
        .name("paged-infer-engine".into())
        .spawn(move || -> anyhow::Result<()> {
            let file = std::fs::File::open(&model_path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let loader = ModelLoader::new(&mmap)?;
            let quantization = match std::env::var("QUANT").as_deref() {
                Ok("int8") => Quantization::Int8,
                _ => Quantization::F32,
            };
            // Take the architecture from the checkpoint's own config.json when
            // it has one, so the server is not pinned to a single model shape.
            let config = LlamaConfig {
                quantization,
                ..LlamaConfig::beside_checkpoint(&model_path)
            };
            let weights = loader.load_weights(&config)?;
            let mut engine = Engine::new(
                weights,
                config,
                EngineConfig {
                    // Every job carries a streaming reply channel; whether the
                    // client reads the deltas is its own business.
                    stream_tokens: true,
                    // Speculative decoding is off unless asked for: it only
                    // pays on copy-heavy workloads, and costs a little on the
                    // rest. See `speculative_benchmark` for the trade.
                    draft_tokens: std::env::var("DRAFT_TOKENS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0),
                    ..EngineConfig::default()
                },
            );
            // Pay the one-time costs before the listener opens, so the first
            // client's time-to-first-token measures its own prefill.
            if std::env::var("WARMUP").as_deref() != Ok("0") {
                engine.warm_up();
            }
            engine_ready.store(true, Ordering::SeqCst);

            let mut pending: Vec<Pending> = Vec::new();
            loop {
                // Absorb everything queued right now, so jobs that arrived
                // while the previous step ran join this batch rather than the
                // next one.
                loop {
                    match rx.try_recv() {
                        Ok(job) => admit(&mut engine, &mut pending, job),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return Ok(()),
                    }
                }

                if !engine.has_work() {
                    // Idle: block rather than spin.
                    match rx.recv() {
                        Ok(job) => admit(&mut engine, &mut pending, job),
                        Err(_) => return Ok(()),
                    }
                    continue;
                }

                // A client that hung up stops costing tokens and, more to the
                // point, stops holding KV blocks that a live request wants.
                for p in pending.iter() {
                    if p.cancel.load(Ordering::Relaxed) {
                        engine.cancel_request(p.request_id);
                    }
                }

                engine.step()?;

                // Deltas first: a streaming client should see this step's
                // tokens before the response that closes its stream.
                for delta in engine.take_deltas() {
                    if let Some(slot) = pending
                        .iter_mut()
                        .find(|p| p.request_id == delta.request_id)
                    {
                        let index = slot.index_of(delta.sequence_id);
                        let _ = slot.reply.send(Event::Delta {
                            index,
                            tokens: delta.tokens,
                            finish_reason: delta.finish_reason,
                        });
                    }
                }

                for completion in engine.take_completed() {
                    if let Some(slot) = pending
                        .iter_mut()
                        .find(|p| p.request_id == completion.request_id)
                    {
                        slot.done.push(completion);
                    }
                }
                // Reply only once every sample of a request is in, so an n>1
                // response arrives complete.
                pending.retain(|p| {
                    if p.done.len() < p.samples {
                        return true;
                    }
                    let _ = p.reply.send(Event::Done(p.done.clone()));
                    false
                });

                let s = engine.stats();
                let prefix = engine.prefix_stats();
                *engine_metrics.lock().unwrap() = Metrics {
                    prompt_tokens: s.prompt_tokens,
                    prompt_tokens_prefilled: s.prompt_tokens_prefilled,
                    generated_tokens: s.generated_tokens,
                    prefix_hits: prefix.hits,
                    prefix_lookups: prefix.hits + prefix.misses,
                    prefix_tokens_saved: prefix.tokens_saved,
                    cow_copies: engine.cow_copies(),
                    decode_tok_s: s.decode_tokens_per_second(),
                    steps: s.steps,
                    requests: s.requests,
                    kv_blocks_total: engine.total_blocks(),
                    kv_blocks_free: engine.available_blocks(),
                };
            }
        })?;

    // A multi-GB checkpoint takes a moment to map and prepack; do not accept
    // connections until the engine can serve them.
    while !ready.load(Ordering::SeqCst) {
        if engine_thread.is_finished() {
            return match engine_thread.join() {
                Ok(Err(e)) => Err(e.context("engine failed to start")),
                _ => Err(anyhow::anyhow!("engine exited during startup")),
            };
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let addr = format!("{host}:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("Paged-Infer serving {model_name} on http://{addr}");
    println!("  GET  /health  /v1/models  /metrics (prometheus)  /stats (json)");
    println!("  POST /v1/completions  /v1/chat/completions   (add \"stream\": true for SSE)");

    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let ctx = HandlerCtx {
            tx: tx.clone(),
            metrics: Arc::clone(&metrics),
            tokenizer: tokenizer.clone(),
            model_name: model_name.clone(),
        };
        // One thread per connection, all funnelling into the single engine, so
        // concurrent clients batch instead of queueing behind each other.
        std::thread::spawn(move || handle(stream, ctx));
    }
    Ok(())
}

fn admit(engine: &mut Engine<'_>, pending: &mut Vec<Pending>, job: Job) {
    let request_id =
        engine.submit_tokens_with(job.tokens, job.max_tokens, job.samples, job.temperature);
    pending.push(Pending {
        request_id,
        samples: job.samples,
        reply: job.reply,
        done: Vec::new(),
        order: Vec::new(),
        cancel: job.cancel,
    });
}

// ── HTTP ─────────────────────────────────────────────────────────────────────

struct HandlerCtx {
    tx: Sender<Job>,
    metrics: Arc<Mutex<Metrics>>,
    tokenizer: Option<Arc<Tokenizer>>,
    model_name: String,
}

fn respond(stream: &mut TcpStream, status: &str, body: &Value) {
    respond_text(stream, status, "application/json", &body.to_string());
}

fn respond_text(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

/// Read the request line, the headers, and exactly `Content-Length` body bytes.
fn read_request(stream: &TcpStream) -> Option<(String, String, String)> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            if key.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some((method, path, String::from_utf8_lossy(&body).into_owned()))
}

fn handle(mut stream: TcpStream, ctx: HandlerCtx) {
    let Some((method, path, body)) = read_request(&stream) else {
        return;
    };

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut stream, "200 OK", &json!({"status": "ok"})),

        ("GET", "/v1/models") => respond(
            &mut stream,
            "200 OK",
            &json!({"object": "list", "data": [{"id": ctx.model_name, "object": "model"}]}),
        ),

        // `/metrics` is the path every scraper already knows, so it speaks the
        // format they expect; the JSON view lives beside it for humans.
        ("GET", "/metrics") => {
            let m = ctx.metrics.lock().unwrap().clone();
            respond_text(
                &mut stream,
                "200 OK",
                "text/plain; version=0.0.4",
                &prometheus(&m),
            )
        }

        ("GET", "/stats") => {
            let m = ctx.metrics.lock().unwrap().clone();
            respond(&mut stream, "200 OK", &metrics_body(&m))
        }

        ("POST", "/v1/completions") | ("POST", "/v1/chat/completions") => {
            let chat = path.ends_with("chat/completions");
            match dispatch(&body, chat, &ctx) {
                Ok(Response::Buffered(v)) => respond(&mut stream, "200 OK", &v),
                Ok(Response::Streaming(rx, spec)) => stream_sse(&mut stream, rx, spec, &ctx),
                Err(e) => respond(
                    &mut stream,
                    "400 Bad Request",
                    &json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}}),
                ),
            }
        }

        _ => respond(
            &mut stream,
            "404 Not Found",
            &json!({"error": {"message": format!("no route for {method} {path}")}}),
        ),
    }
}

fn metrics_body(m: &Metrics) -> Value {
    json!({
        "requests": m.requests,
        "scheduler_steps": m.steps,
        "kv_blocks_total": m.kv_blocks_total,
        "kv_blocks_free": m.kv_blocks_free,
        "prefix_cache_tokens_saved": m.prefix_tokens_saved,
        "prompt_tokens": m.prompt_tokens,
        "prompt_tokens_prefilled": m.prompt_tokens_prefilled,
        "prompt_tokens_reused": m.prompt_tokens.saturating_sub(m.prompt_tokens_prefilled),
        "generated_tokens": m.generated_tokens,
        "prefix_cache_hits": m.prefix_hits,
        "prefix_cache_lookups": m.prefix_lookups,
        "prefix_cache_hit_rate": if m.prefix_lookups == 0 {
            0.0
        } else {
            m.prefix_hits as f64 / m.prefix_lookups as f64
        },
        "copy_on_write_copies": m.cow_copies,
        "decode_tokens_per_second": m.decode_tok_s,
    })
}

/// Whether the client asked for the whole answer at once or as it is produced.
enum Response {
    Buffered(Value),
    Streaming(Receiver<Event>, StreamSpec),
}

struct StreamSpec {
    id: String,
    chat: bool,
    samples: usize,
    prompt_tokens: usize,
    /// Flipped when the client stops reading, so the engine can stop working.
    cancel: Arc<AtomicBool>,
}

fn next_id() -> String {
    use std::sync::atomic::AtomicUsize;
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("cmpl-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn dispatch(body: &str, chat: bool, ctx: &HandlerCtx) -> anyhow::Result<Response> {
    let req: Value =
        serde_json::from_str(body).map_err(|e| anyhow::anyhow!("invalid JSON: {e}"))?;

    let max_tokens = req["max_tokens"].as_u64().unwrap_or(64).clamp(1, 4096) as usize;
    let temperature = req["temperature"].as_f64().map(|t| t.max(0.0) as f32);
    let samples = req["n"].as_u64().unwrap_or(1).clamp(1, 16) as usize;
    let streaming = req["stream"].as_bool().unwrap_or(false);

    let tokens = match req["prompt_tokens"].as_array() {
        Some(arr) if !arr.is_empty() => arr
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as u32)
                    .ok_or_else(|| anyhow::anyhow!("prompt_tokens must be unsigned integers"))
            })
            .collect::<anyhow::Result<Vec<u32>>>()?,
        _ => {
            let prompt = if chat {
                let msgs = req["messages"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("chat requests need a messages array"))?;
                msgs.iter()
                    .map(|m| {
                        format!(
                            "{}: {}",
                            m["role"].as_str().unwrap_or("user"),
                            m["content"].as_str().unwrap_or("")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                req["prompt"].as_str().unwrap_or("").to_string()
            };
            anyhow::ensure!(
                !prompt.trim().is_empty(),
                "request needs a non-empty prompt, messages, or prompt_tokens"
            );
            let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("no tokenizer loaded; send prompt_tokens instead of text")
            })?;
            tokenizer
                .encode(prompt, true)
                .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?
                .get_ids()
                .to_vec()
        }
    };
    anyhow::ensure!(!tokens.is_empty(), "prompt tokenized to nothing");
    let prompt_tokens = tokens.len();

    let cancel = Arc::new(AtomicBool::new(false));
    let (reply_tx, reply_rx) = channel();
    ctx.tx
        .send(Job {
            tokens,
            max_tokens,
            temperature,
            samples,
            reply: reply_tx,
            cancel: Arc::clone(&cancel),
        })
        .map_err(|_| anyhow::anyhow!("engine is not running"))?;

    if streaming {
        return Ok(Response::Streaming(
            reply_rx,
            StreamSpec {
                id: next_id(),
                chat,
                samples,
                prompt_tokens,
                cancel,
            },
        ));
    }

    // Buffered: drop every delta on the floor and wait for the whole answer.
    let completions = loop {
        match reply_rx.recv() {
            Ok(Event::Done(c)) => break c,
            Ok(Event::Delta { .. }) => continue,
            Err(_) => anyhow::bail!("engine dropped the request"),
        }
    };

    let generated: usize = completions.iter().map(|c| c.tokens.len()).sum();
    let choices: Vec<Value> = completions
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let text = ctx
                .tokenizer
                .as_ref()
                .and_then(|t| t.decode(&c.tokens, true).ok())
                .unwrap_or_default();
            let finish = finish_reason_str(Some(c.finish_reason));
            if chat {
                json!({
                    "index": i,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                    // Raw ids too, so the endpoint is usable without a tokenizer.
                    "tokens": c.tokens,
                })
            } else {
                json!({
                    "index": i,
                    "text": text,
                    "finish_reason": finish,
                    "tokens": c.tokens,
                })
            }
        })
        .collect();

    Ok(Response::Buffered(json!({
        "id": next_id(),
        "object": if chat { "chat.completion" } else { "text_completion" },
        "model": ctx.model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated,
            "total_tokens": prompt_tokens + generated,
        },
    })))
}

/// OpenAI's vocabulary for why generation stopped. `null` while it has not.
fn finish_reason_str(reason: Option<FinishReason>) -> Option<&'static str> {
    match reason? {
        FinishReason::Eos => Some("stop"),
        FinishReason::Length | FinishReason::OutOfMemory => Some("length"),
        FinishReason::Cancelled => Some("cancelled"),
    }
}

// ── streaming ────────────────────────────────────────────────────────────────

/// Write the answer as server-sent events, one chunk per scheduler step.
///
/// The shape is OpenAI's, so an existing client works unchanged: `data: ` plus
/// one JSON object per line, a blank line between events, and a literal
/// `data: [DONE]` to close.
///
/// Every write is checked. A client that hangs up mid-generation is the normal
/// case — someone closed a tab — and the point of noticing is that the engine
/// then stops: the cancel flag is raised, the scheduler retires the sequence at
/// its next step, and its KV blocks go back to the pool for a request that is
/// still being read.
fn stream_sse(stream: &mut TcpStream, rx: Receiver<Event>, spec: StreamSpec, ctx: &HandlerCtx) {
    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\n\
                Connection: close\r\n\
                X-Accel-Buffering: no\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        spec.cancel.store(true, Ordering::SeqCst);
        return;
    }

    let object = if spec.chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let mut detok: Vec<IncrementalDetokenizer> = (0..spec.samples)
        .map(|_| IncrementalDetokenizer::new())
        .collect();
    let mut role_sent = vec![false; spec.samples];
    let mut generated = 0usize;

    let send = |stream: &mut TcpStream, payload: &Value| -> bool {
        let frame = format!("data: {payload}\n\n");
        stream.write_all(frame.as_bytes()).is_ok() && stream.flush().is_ok()
    };

    // A `recv` error means the engine thread is gone; close the stream cleanly
    // rather than leave the client waiting on a socket nobody will write to.
    while let Ok(event) = rx.recv() {
        let (index, tokens, finish) = match event {
            Event::Delta {
                index,
                tokens,
                finish_reason,
            } => (index, tokens, finish_reason),
            // Completions carry nothing the deltas did not already deliver.
            Event::Done(_) => break,
        };
        if index >= detok.len() {
            continue;
        }

        generated += tokens.len();
        let text = match ctx.tokenizer.as_ref() {
            Some(t) => detok[index].push(t, &tokens),
            None => String::new(),
        };
        // A token that only completes half a character produces no text yet.
        // Suppress that chunk rather than emit an empty delta -- but never
        // suppress the one carrying the finish reason.
        if text.is_empty() && tokens.is_empty() && finish.is_none() {
            continue;
        }

        // OpenAI announces the role once, on the opening chunk of each choice.
        let delta = if spec.chat {
            if std::mem::replace(&mut role_sent[index], true) {
                json!({ "content": text })
            } else {
                json!({"role": "assistant", "content": text})
            }
        } else {
            Value::Null
        };
        let choice = if spec.chat {
            json!({"index": index, "delta": delta, "finish_reason": finish_reason_str(finish)})
        } else {
            json!({"index": index, "text": text, "finish_reason": finish_reason_str(finish)})
        };
        let payload = json!({
            "id": spec.id,
            "object": object,
            "model": ctx.model_name,
            "choices": [choice],
            // Raw ids alongside the text, matching the buffered response.
            "tokens": tokens,
        });

        if !send(stream, &payload) {
            spec.cancel.store(true, Ordering::SeqCst);
            return;
        }
    }

    let usage = json!({
        "id": spec.id,
        "object": object,
        "model": ctx.model_name,
        "choices": [],
        "usage": {
            "prompt_tokens": spec.prompt_tokens,
            "completion_tokens": generated,
            "total_tokens": spec.prompt_tokens + generated,
        },
    });
    if send(stream, &usage) {
        let _ = stream.write_all(b"data: [DONE]\n\n");
        let _ = stream.flush();
    }
}

// ── metrics ──────────────────────────────────────────────────────────────────

/// Prometheus text exposition format.
///
/// Counters carry the `_total` suffix and gauges do not, because that is what
/// the convention is and a scraper's `rate()` depends on it.
fn prometheus(m: &Metrics) -> String {
    let mut out = String::new();
    let mut metric = |name: &str, kind: &str, help: &str, value: f64| {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
        out.push_str(&format!("{name} {value}\n"));
    };

    metric(
        "paged_infer_requests_total",
        "counter",
        "Requests admitted to the scheduler.",
        m.requests as f64,
    );
    metric(
        "paged_infer_scheduler_steps_total",
        "counter",
        "Iteration-level scheduling steps executed.",
        m.steps as f64,
    );
    metric(
        "paged_infer_prompt_tokens_total",
        "counter",
        "Prompt tokens submitted.",
        m.prompt_tokens as f64,
    );
    metric(
        "paged_infer_prompt_tokens_prefilled_total",
        "counter",
        "Prompt tokens that actually went through the model.",
        m.prompt_tokens_prefilled as f64,
    );
    metric(
        "paged_infer_generated_tokens_total",
        "counter",
        "Tokens generated.",
        m.generated_tokens as f64,
    );
    metric(
        "paged_infer_prefix_cache_lookups_total",
        "counter",
        "Prefix-cache block lookups.",
        m.prefix_lookups as f64,
    );
    metric(
        "paged_infer_prefix_cache_hits_total",
        "counter",
        "Prefix-cache block lookups that hit.",
        m.prefix_hits as f64,
    );
    metric(
        "paged_infer_prefix_cache_tokens_saved_total",
        "counter",
        "Prompt tokens whose KV was reused instead of recomputed.",
        m.prefix_tokens_saved as f64,
    );
    metric(
        "paged_infer_copy_on_write_copies_total",
        "counter",
        "Shared KV blocks copied because a forked sequence diverged.",
        m.cow_copies as f64,
    );
    metric(
        "paged_infer_kv_blocks",
        "gauge",
        "KV cache blocks in the pool.",
        m.kv_blocks_total as f64,
    );
    metric(
        "paged_infer_kv_blocks_free",
        "gauge",
        "KV cache blocks not currently mapped or cached.",
        m.kv_blocks_free as f64,
    );
    metric(
        "paged_infer_decode_tokens_per_second",
        "gauge",
        "Generated tokens divided by time spent in decode.",
        m.decode_tok_s,
    );
    out
}
