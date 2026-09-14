//! OpenAI-shaped HTTP serving on top of the engine.
//!
//! The engine owns one thread and runs its scheduling loop there; handler
//! threads push jobs onto a bounded channel and wait on a bounded reply
//! channel. That layout is the point: requests arriving from *different*
//! clients land in the same batch and decode together, and two clients sending
//! the same system prompt hit the prefix cache. A server that took a lock
//! around the engine per request would serialize them and get none of it.
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
//! | `/health`                   | GET    | readiness, reflecting the worker |
//! | `/v1/models`                | GET    | the loaded model                 |
//! | `/metrics`                  | GET    | Prometheus text exposition       |
//! | `/stats`                    | GET    | the same counters as JSON        |
//! | `/v1/completions`           | POST   | `prompt` or `prompt_tokens`      |
//! | `/v1/chat/completions`      | POST   | `messages`, via the chat template|
//!
//! `prompt_tokens` takes pre-tokenized input, so the server is usable for
//! load testing without a tokenizer on hand; those ids are the complete model
//! input. Text and chat requests go through the [`ModelProfile`], which owns
//! special tokens and the checkpoint's chat template. Both completion routes
//! accept `"stream": true` and reply with OpenAI-shaped server-sent events.
//!
//! # Every resource a client can consume is bounded
//!
//! A network service is only as safe as its least-bounded input. Here:
//! request head bytes, body bytes (checked against `Content-Length` *before*
//! anything is allocated), concurrent connections, queued jobs, per-client
//! reply events, and socket read/write time are all capped by [`ServeConfig`],
//! and exceeding any of them is an explicit HTTP response — 413, 431, 503 with
//! `Retry-After`, 408 — never a stall of the engine thread. A client that
//! stops reading its stream has its request cancelled rather than blocking the
//! one thread everybody shares.
//!
//! The split between layers is deliberate: [`read_request`] is network
//! plumbing, [`prepare_request`] is validation with no I/O at all, and the
//! engine loop drives scheduling. Each is testable on its own.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{ensure, Result};
use serde_json::{json, Value};

use crate::detokenizer::IncrementalDetokenizer;
use crate::engine::{Completion, Engine, EngineConfig, FinishReason, RequestOptions};
use crate::model::LlamaWeights;
use crate::profile::{ChatMessage, ModelProfile};

/// Limits and defaults for one server.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// `host:port`; port `0` picks a free one (see [`Server::addr`]).
    pub addr: String,
    pub model_name: String,
    /// Request line plus headers, in bytes.
    pub max_header_bytes: usize,
    /// Largest `Content-Length` accepted. Checked before the body is read.
    pub max_body_bytes: usize,
    /// Connections handled at once; the rest get 503 immediately.
    pub max_connections: usize,
    /// Jobs waiting for the engine thread; beyond this, 503.
    pub max_queued_jobs: usize,
    /// Jobs received before a scheduler step, including refused/cancelled jobs.
    /// A bounded intake lets decoding proceed under continuous arrivals.
    pub max_jobs_per_step: usize,
    /// Events buffered per client between the engine and its handler. A client
    /// further behind than this is not reading, and is cancelled.
    pub max_pending_events: usize,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    /// Cap on `max_tokens`; larger requests are clamped to it.
    pub max_tokens_limit: usize,
    /// Cap on `n`.
    pub max_samples: usize,
    /// Run the engine's warm-up pass before reporting ready.
    pub warm_up: bool,
    /// Scheduler settings. `stream_tokens` is forced on: every job streams.
    pub engine: EngineConfig,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080".to_string(),
            model_name: "paged-infer".to_string(),
            max_header_bytes: 16 * 1024,
            max_body_bytes: 1024 * 1024,
            max_connections: 256,
            max_queued_jobs: 1024,
            max_jobs_per_step: 32,
            max_pending_events: 1024,
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            max_tokens_limit: 4096,
            max_samples: 16,
            warm_up: true,
            engine: EngineConfig::default(),
        }
    }
}

/// One unit of work for the engine thread.
struct Job {
    tokens: Vec<u32>,
    max_tokens: usize,
    samples: usize,
    options: RequestOptions,
    reply: SyncSender<Event>,
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
    /// The engine refused the request at submission: the client's input was
    /// wrong (an id outside the vocabulary, a prompt that can never fit).
    Rejected(String),
}

/// Engine counters, refreshed by the engine thread after every step.
#[derive(Default, Clone, Debug)]
pub struct Metrics {
    pub prompt_tokens: usize,
    pub prompt_tokens_prefilled: usize,
    pub prompt_tokens_reused: usize,
    pub generated_tokens: usize,
    pub prefix_hits: u64,
    pub prefix_lookups: u64,
    pub prefix_tokens_saved: u64,
    pub cow_copies: u64,
    pub decode_tok_s: f64,
    pub steps: usize,
    pub requests: usize,
    pub kv_blocks_total: usize,
    pub kv_blocks_free: usize,
    pub sequences_active: usize,
    pub requests_queued: usize,
    pub requests_prefilling: usize,
    pub pending_prefill_tokens: usize,
    pub last_prefill_tokens: usize,
    pub prefill_chunks: usize,
    pub prefill_preemptions: usize,
    pub sequences_deferred: usize,
    pub preemptions: usize,
    pub recomputed_tokens: usize,
}

/// A request in flight, held until every sibling sample of it has finished.
struct Pending {
    request_id: usize,
    samples: usize,
    reply: SyncSender<Event>,
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

/// State shared by the accept loop, the handlers and the engine thread.
struct Shared {
    config: ServeConfig,
    profile: ModelProfile,
    /// Handlers clone this to submit. Taken (set to `None`) at shutdown, so
    /// the engine thread's receiver disconnects and its loop exits; handler
    /// clones are transient, held only for the duration of a send.
    jobs: Mutex<Option<SyncSender<Job>>>,
    metrics: Mutex<Metrics>,
    /// The engine finished loading (and warming up) and is taking jobs.
    ready: AtomicBool,
    /// The engine thread is still running. Cleared if it exits or panics, so
    /// `/health` and new requests say so instead of hanging.
    engine_alive: AtomicBool,
    connections: AtomicUsize,
    shutdown: AtomicBool,
}

/// A running server. Dropping it (or calling [`Server::shutdown`]) stops
/// accepting, cancels in-flight work and joins the threads.
pub struct Server {
    addr: SocketAddr,
    shared: Arc<Shared>,
    accept_thread: Option<JoinHandle<()>>,
    engine_thread: Option<JoinHandle<()>>,
}

impl Server {
    /// The bound address — meaningful when the config asked for port 0.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Block until the accept loop ends.
    pub fn wait(mut self) {
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
        if let Some(t) = self.engine_thread.take() {
            let _ = t.join();
        }
    }

    pub fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        // Wake the accept loop so it notices.
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_secs(1));
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
        // Drop the job sender: the engine loop exits when its receiver
        // disconnects, which also drops every in-flight reply sender, so
        // handlers still waiting on the engine get an error and finish.
        self.shared
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(t) = self.engine_thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Start serving `weights` under `profile`. Returns once the listener is bound;
/// the engine loads and warms up in the background and `/health` reports when
/// it is ready. Requests that arrive before then are answered 503.
pub fn run(
    config: ServeConfig,
    profile: ModelProfile,
    weights: LlamaWeights<'static>,
) -> Result<Server> {
    ensure!(
        config.max_jobs_per_step > 0,
        "max_jobs_per_step must be positive"
    );
    ensure!(
        config.engine.max_prefill_tokens_per_step > 0,
        "max_prefill_tokens_per_step must be positive"
    );
    let listener = TcpListener::bind(&config.addr)?;
    let addr = listener.local_addr()?;
    let (jobs_tx, jobs_rx) = sync_channel::<Job>(config.max_queued_jobs.max(1));

    let mut engine_config = config.engine.clone();
    engine_config.stream_tokens = true;
    let model_config = profile.config.clone();
    let warm_up = config.warm_up;

    let shared = Arc::new(Shared {
        config,
        profile,
        jobs: Mutex::new(Some(jobs_tx)),
        metrics: Mutex::new(Metrics::default()),
        ready: AtomicBool::new(false),
        engine_alive: AtomicBool::new(true),
        connections: AtomicUsize::new(0),
        shutdown: AtomicBool::new(false),
    });

    let engine_shared = Arc::clone(&shared);
    let engine_thread = std::thread::Builder::new()
        .name("paged-infer-engine".into())
        .spawn(move || {
            // Whatever ends this thread — a clean exit or a panic — the flag
            // flips, so handlers stop promising work the engine will not do.
            struct Alive<'a>(&'a AtomicBool);
            impl Drop for Alive<'_> {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::SeqCst);
                }
            }
            let _alive = Alive(&engine_shared.engine_alive);

            let mut engine = Engine::new(weights, model_config, engine_config);
            // Pay the one-time costs before taking traffic, so the first
            // client's time-to-first-token measures its own prefill.
            if warm_up {
                engine.warm_up();
            }
            engine_shared.ready.store(true, Ordering::SeqCst);
            engine_loop(&mut engine, jobs_rx, &engine_shared);
        })?;

    let accept_shared = Arc::clone(&shared);
    let accept_thread = std::thread::Builder::new()
        .name("paged-infer-accept".into())
        .spawn(move || accept_loop(listener, accept_shared))?;

    Ok(Server {
        addr,
        shared,
        accept_thread: Some(accept_thread),
        engine_thread: Some(engine_thread),
    })
}

// ── engine thread ────────────────────────────────────────────────────────────

fn engine_loop(engine: &mut Engine<'static>, rx: Receiver<Job>, shared: &Shared) {
    let mut pending: Vec<Pending> = Vec::new();
    loop {
        let mut received = 0;
        if !engine.has_work() {
            publish_metrics(engine, shared);
            // Count the wake-up job in this iteration's intake allowance too.
            match rx.recv() {
                Ok(job) => {
                    admit(engine, &mut pending, job);
                    received = 1;
                }
                Err(_) => return,
            }
        }
        // Every received job consumes the allowance, even if it is invalid or
        // its client already cancelled. Producers cannot keep this loop busy
        // indefinitely while existing streams wait for a scheduler step.
        for _ in received..shared.config.max_jobs_per_step {
            match rx.try_recv() {
                Ok(job) => admit(engine, &mut pending, job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        if !engine.has_work() {
            publish_metrics(engine, shared);
            continue;
        }

        // A client that hung up — or fell too far behind to be reading — stops
        // costing tokens and, more to the point, stops holding KV blocks that
        // a live request wants. Nobody is listening for its completion.
        pending.retain(|p| {
            if p.cancel.load(Ordering::Relaxed) {
                engine.cancel_request(p.request_id);
                false
            } else {
                true
            }
        });

        if let Err(e) = engine.step() {
            eprintln!("engine step failed: {e:#}");
            return;
        }

        // Deltas first: a streaming client should see this step's tokens
        // before the response that closes its stream.
        for delta in engine.take_deltas() {
            if let Some(slot) = pending
                .iter_mut()
                .find(|p| p.request_id == delta.request_id)
            {
                let index = slot.index_of(delta.sequence_id);
                let event = Event::Delta {
                    index,
                    tokens: delta.tokens,
                    finish_reason: delta.finish_reason,
                };
                if slot.reply.try_send(event).is_err() {
                    // Full: the handler is not draining, so the client is not
                    // reading. Disconnected: the handler is gone. Either way
                    // the request is cancelled at the next step, never waited
                    // on — this thread is shared by every other client.
                    slot.cancel.store(true, Ordering::Relaxed);
                }
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
            let _ = p.reply.try_send(Event::Done(p.done.clone()));
            false
        });

        publish_metrics(engine, shared);
    }
}

fn admit(engine: &mut Engine<'_>, pending: &mut Vec<Pending>, job: Job) {
    if job.cancel.load(Ordering::Relaxed) {
        return;
    }
    match engine.submit_tokens_with(job.tokens, job.max_tokens, job.samples, job.options) {
        Ok(request_id) => pending.push(Pending {
            request_id,
            samples: job.samples,
            reply: job.reply,
            done: Vec::new(),
            order: Vec::new(),
            cancel: job.cancel,
        }),
        Err(e) => {
            let _ = job.reply.try_send(Event::Rejected(e.to_string()));
        }
    }
}

fn publish_metrics(engine: &Engine<'_>, shared: &Shared) {
    let s = engine.stats();
    let prefix = engine.prefix_stats();
    let (active, queued) = engine.queue_depth();
    let m = Metrics {
        prompt_tokens: s.prompt_tokens,
        prompt_tokens_prefilled: s.prompt_tokens_prefilled,
        prompt_tokens_reused: s.prompt_tokens_reused(),
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
        sequences_active: active,
        requests_queued: queued,
        requests_prefilling: engine.prefilling_requests(),
        pending_prefill_tokens: engine.pending_prefill_tokens(),
        last_prefill_tokens: s.last_prefill_tokens,
        prefill_chunks: s.prefill_chunks,
        prefill_preemptions: s.prefill_preemptions,
        sequences_deferred: engine.deferred_sequences(),
        preemptions: s.preemptions,
        recomputed_tokens: s.recomputed_tokens,
    };
    *shared.metrics.lock().unwrap_or_else(|e| e.into_inner()) = m;
}

// ── accept loop and connections ──────────────────────────────────────────────

fn accept_loop(listener: TcpListener, shared: Arc<Shared>) {
    for conn in listener.incoming() {
        if shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        let Ok(stream) = conn else { continue };
        // Count before spawning, so the number of live handler threads is
        // bounded by the cap plus the rejections in flight.
        let live = shared.connections.fetch_add(1, Ordering::SeqCst) + 1;
        let shared = Arc::clone(&shared);
        let over = live > shared.config.max_connections;
        std::thread::spawn(move || {
            struct Slot<'a>(&'a AtomicUsize);
            impl Drop for Slot<'_> {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _slot = Slot(&shared.connections);
            let mut stream = stream;
            let _ = stream.set_nodelay(true);
            let _ = stream.set_read_timeout(Some(shared.config.read_timeout));
            let _ = stream.set_write_timeout(Some(shared.config.write_timeout));
            if over {
                respond_error_and_drain(
                    &mut stream,
                    "503 Service Unavailable",
                    "overloaded_error",
                    "too many open connections; retry shortly",
                    Some(1),
                );
                return;
            }
            handle_connection(stream, &shared);
        });
    }
}

/// A parsed request head plus its body.
#[derive(Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Why a request could not be read. Each maps to a specific status.
#[derive(Debug)]
pub enum HttpError {
    /// The request line and headers exceeded the limit.
    HeadersTooLarge,
    /// `Content-Length` exceeds the limit. Detected before allocating.
    PayloadTooLarge {
        declared: u64,
        limit: usize,
    },
    /// A body-bearing method without `Content-Length`.
    LengthRequired,
    BadRequest(String),
    /// The client stopped sending before the request was complete.
    Timeout,
    /// The client went away before sending a request.
    Closed,
}

impl HttpError {
    pub fn status(&self) -> &'static str {
        match self {
            HttpError::HeadersTooLarge => "431 Request Header Fields Too Large",
            HttpError::PayloadTooLarge { .. } => "413 Payload Too Large",
            HttpError::LengthRequired => "411 Length Required",
            HttpError::BadRequest(_) => "400 Bad Request",
            HttpError::Timeout => "408 Request Timeout",
            HttpError::Closed => "400 Bad Request",
        }
    }

    pub fn message(&self) -> String {
        match self {
            HttpError::HeadersTooLarge => "request headers too large".into(),
            HttpError::PayloadTooLarge { declared, limit } => {
                format!("request body of {declared} bytes exceeds the limit of {limit} bytes")
            }
            HttpError::LengthRequired => "POST requests need a Content-Length header".into(),
            HttpError::BadRequest(m) => m.clone(),
            HttpError::Timeout => "timed out waiting for the request".into(),
            HttpError::Closed => "connection closed before a request arrived".into(),
        }
    }
}

/// Read one HTTP/1.1 request within the configured limits.
///
/// The head is read line by line under a total byte cap. The body is read only
/// after its declared length has been checked against the cap — a client
/// cannot make the server allocate by announcing a size. `Expect:
/// 100-continue` is honoured, since curl sends it for bodies over a kilobyte
/// and otherwise waits a second before giving up on it.
pub fn read_request(stream: &TcpStream, config: &ServeConfig) -> Result<Request, HttpError> {
    let mut reader = BufReader::new(stream);
    let mut head = Vec::new();
    loop {
        let mut line = Vec::new();
        let budget = (config.max_header_bytes + 1).saturating_sub(head.len()) as u64;
        let n = (&mut reader)
            .take(budget)
            .read_until(b'\n', &mut line)
            .map_err(io_error)?;
        if n == 0 {
            return Err(if head.is_empty() {
                HttpError::Closed
            } else {
                HttpError::BadRequest("incomplete request head".into())
            });
        }
        head.extend_from_slice(&line);
        if head.len() > config.max_header_bytes {
            return Err(HttpError::HeadersTooLarge);
        }
        if line == b"\r\n" || line == b"\n" {
            break;
        }
        if !line.ends_with(b"\n") {
            // `take` ran out before a newline: the head is over budget.
            return Err(HttpError::HeadersTooLarge);
        }
    }

    let head = String::from_utf8_lossy(&head);
    let mut lines = head.split("\r\n").flat_map(|l| l.split('\n'));
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| HttpError::BadRequest("missing request line".into()))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| HttpError::BadRequest("missing request path".into()))?
        .to_string();

    let mut content_length: Option<u64> = None;
    let mut expect_continue = false;
    for line in lines {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if key.eq_ignore_ascii_case("content-length") {
            content_length = Some(
                value
                    .parse()
                    .map_err(|_| HttpError::BadRequest("malformed Content-Length".into()))?,
            );
        } else if key.eq_ignore_ascii_case("transfer-encoding") {
            if value.to_ascii_lowercase().contains("chunked") {
                return Err(HttpError::BadRequest(
                    "chunked transfer encoding is not supported; send Content-Length".into(),
                ));
            }
        } else if key.eq_ignore_ascii_case("expect") {
            expect_continue = value.eq_ignore_ascii_case("100-continue");
        }
    }

    let body = if method == "POST" || method == "PUT" || method == "PATCH" {
        let declared = content_length.ok_or(HttpError::LengthRequired)?;
        if declared > config.max_body_bytes as u64 {
            return Err(HttpError::PayloadTooLarge {
                declared,
                limit: config.max_body_bytes,
            });
        }
        if expect_continue {
            let mut w = stream;
            let _ = w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        }
        let mut body = vec![0u8; declared as usize];
        reader.read_exact(&mut body).map_err(io_error)?;
        body
    } else {
        Vec::new()
    };

    Ok(Request { method, path, body })
}

fn io_error(e: std::io::Error) -> HttpError {
    match e.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => HttpError::Timeout,
        std::io::ErrorKind::UnexpectedEof => {
            HttpError::BadRequest("connection closed mid-request".into())
        }
        _ => HttpError::BadRequest(format!("read failed: {e}")),
    }
}

fn handle_connection(mut stream: TcpStream, shared: &Shared) {
    let request = match read_request(&stream, &shared.config) {
        Ok(r) => r,
        Err(HttpError::Closed) => return,
        Err(e) => {
            respond_error_and_drain(
                &mut stream,
                e.status(),
                "invalid_request_error",
                &e.message(),
                None,
            );
            return;
        }
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            let (status, body) = health(shared);
            respond(&mut stream, status, &body);
        }

        ("GET", "/v1/models") => respond(
            &mut stream,
            "200 OK",
            &json!({"object": "list", "data": [{"id": shared.config.model_name, "object": "model"}]}),
        ),

        // `/metrics` is the path every scraper already knows, so it speaks the
        // format they expect; the JSON view lives beside it for humans.
        ("GET", "/metrics") => {
            let m = shared
                .metrics
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let connections = shared.connections.load(Ordering::Relaxed);
            respond_text(
                &mut stream,
                "200 OK",
                "text/plain; version=0.0.4",
                &prometheus(&m, connections),
            )
        }

        ("GET", "/stats") => {
            let m = shared
                .metrics
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let connections = shared.connections.load(Ordering::Relaxed);
            respond(&mut stream, "200 OK", &metrics_body(&m, connections))
        }

        ("POST", "/v1/completions") | ("POST", "/v1/chat/completions") => {
            let chat = request.path.ends_with("chat/completions");
            if !shared.engine_alive.load(Ordering::SeqCst) {
                respond_error(
                    &mut stream,
                    "503 Service Unavailable",
                    "engine_unavailable",
                    "the inference engine is not running",
                    None,
                );
                return;
            }
            if !shared.ready.load(Ordering::SeqCst) {
                respond_error(
                    &mut stream,
                    "503 Service Unavailable",
                    "starting",
                    "the model is still loading; retry shortly",
                    Some(2),
                );
                return;
            }
            let prepared =
                match prepare_request(&request.body, chat, &shared.profile, &shared.config) {
                    Ok(p) => p,
                    Err(e) => {
                        respond_error(
                            &mut stream,
                            "400 Bad Request",
                            e.kind(),
                            &e.to_string(),
                            None,
                        );
                        return;
                    }
                };
            complete(&mut stream, prepared, chat, shared);
        }

        (method, path) => respond_error(
            &mut stream,
            "404 Not Found",
            "not_found",
            &format!("no route for {method} {path}"),
            None,
        ),
    }
}

fn health(shared: &Shared) -> (&'static str, Value) {
    let alive = shared.engine_alive.load(Ordering::SeqCst);
    let ready = shared.ready.load(Ordering::SeqCst);
    let m = shared
        .metrics
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let status = match (alive, ready) {
        (true, true) => "ok",
        (true, false) => "starting",
        (false, _) => "down",
    };
    let body = json!({
        "status": status,
        "ready": alive && ready,
        "engine_alive": alive,
        "connections": shared.connections.load(Ordering::Relaxed),
        "sequences_active": m.sequences_active,
        "requests_queued": m.requests_queued,
        "requests_prefilling": m.requests_prefilling,
        "pending_prefill_tokens": m.pending_prefill_tokens,
        "sequences_deferred": m.sequences_deferred,
        "kv_blocks_free": m.kv_blocks_free,
        "kv_blocks_total": m.kv_blocks_total,
    });
    let http = if alive && ready {
        "200 OK"
    } else {
        "503 Service Unavailable"
    };
    (http, body)
}

// ── responses ────────────────────────────────────────────────────────────────

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

/// An OpenAI-shaped error body. `retry_after` adds the header overload
/// responses need for a well-behaved client to back off.
fn respond_error(
    stream: &mut TcpStream,
    status: &str,
    kind: &str,
    message: &str,
    retry_after: Option<u32>,
) {
    let body = json!({"error": {"message": message, "type": kind}}).to_string();
    let retry = retry_after.map_or(String::new(), |s| format!("Retry-After: {s}\r\n"));
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{retry}Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

/// Answer a request whose remaining bytes will never be read, then let the
/// client actually see the answer.
///
/// Closing a socket that still has unread input makes the kernel send RST,
/// and a reset can discard the response sitting in the client's receive
/// buffer — the client then sees "connection reset" instead of the 413 that
/// explains what it did wrong. So: write the response, half-close, then drain
/// what the client is still sending, briefly and boundedly, before dropping
/// the connection. The bound keeps a hostile sender from turning the drain
/// into a free upload.
fn respond_error_and_drain(
    stream: &mut TcpStream,
    status: &str,
    kind: &str,
    message: &str,
    retry_after: Option<u32>,
) {
    respond_error(stream, status, kind, message, retry_after);
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let mut sink = [0u8; 4096];
    let mut drained = 0usize;
    while drained < 256 * 1024 {
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(n) => drained += n,
        }
    }
}

// ── request validation (no I/O) ──────────────────────────────────────────────

/// A validated request, ready for the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    pub samples: usize,
    pub options: RequestOptions,
    pub stream: bool,
}

/// Why a request body was refused. All of these are the client's fault and
/// map to 400; the `kind` is the OpenAI-style `error.type`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestError {
    Invalid(String),
    /// A field this server advertises no support for, sent with a non-default
    /// value. Silently ignoring it would return an answer the client did not
    /// ask for.
    Unsupported(String),
}

impl RequestError {
    pub fn kind(&self) -> &'static str {
        match self {
            RequestError::Invalid(_) => "invalid_request_error",
            RequestError::Unsupported(_) => "unsupported_option",
        }
    }
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestError::Invalid(m) | RequestError::Unsupported(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RequestError {}

/// Options this server does not implement. Present with a non-default value,
/// they are refused rather than ignored.
const UNSUPPORTED_OPTIONS: &[&str] = &[
    "stop",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "suffix",
    "echo",
    "best_of",
];

/// True when `value` is the field's do-nothing default, which clients send
/// routinely and which changes nothing.
fn is_default_value(key: &str, value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(false) => true,
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => {
            o.is_empty() || (key == "response_format" && o.get("type") == Some(&json!("text")))
        }
        Value::Number(n) => {
            let zero = n.as_f64() == Some(0.0);
            zero || (key == "best_of" && n.as_u64() == Some(1))
        }
        Value::String(s) => s.is_empty(),
        _ => false,
    }
}

/// Parse and validate a completion or chat request body.
pub fn prepare_request(
    body: &[u8],
    chat: bool,
    profile: &ModelProfile,
    config: &ServeConfig,
) -> Result<Prepared, RequestError> {
    let invalid = |m: String| RequestError::Invalid(m);
    let req: Value =
        serde_json::from_slice(body).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
    let obj = req
        .as_object()
        .ok_or_else(|| invalid("request body must be a JSON object".into()))?;

    for key in UNSUPPORTED_OPTIONS {
        if let Some(v) = obj.get(*key) {
            if !is_default_value(key, v) {
                return Err(RequestError::Unsupported(format!(
                    "{key} is not supported by this server (got {v})"
                )));
            }
        }
    }

    let max_tokens = match obj
        .get("max_tokens")
        .or_else(|| obj.get("max_completion_tokens"))
    {
        None | Some(Value::Null) => 64,
        Some(v) => match v.as_u64() {
            Some(n) if n >= 1 => (n as usize).min(config.max_tokens_limit),
            _ => {
                return Err(invalid(format!(
                    "max_tokens must be a positive integer, got {v}"
                )))
            }
        },
    };
    let samples = match obj.get("n") {
        None | Some(Value::Null) => 1,
        Some(v) => match v.as_u64() {
            Some(n) if n >= 1 && n as usize <= config.max_samples => n as usize,
            _ => {
                return Err(invalid(format!(
                    "n must be an integer between 1 and {}, got {v}",
                    config.max_samples
                )))
            }
        },
    };
    let stream = match obj.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(v) => return Err(invalid(format!("stream must be a boolean, got {v}"))),
    };

    let mut options = RequestOptions::default();
    if let Some(v) = obj.get("temperature").filter(|v| !v.is_null()) {
        let t = v
            .as_f64()
            .filter(|t| t.is_finite() && *t >= 0.0)
            .ok_or_else(|| invalid(format!("temperature must be a number >= 0, got {v}")))?;
        options.temperature = Some(t as f32);
    }
    if let Some(v) = obj.get("top_p").filter(|v| !v.is_null()) {
        let p = v
            .as_f64()
            .filter(|p| *p > 0.0 && *p <= 1.0)
            .ok_or_else(|| invalid(format!("top_p must be in (0, 1], got {v}")))?;
        options.top_p = Some(p as f32);
    }
    if let Some(v) = obj.get("top_k").filter(|v| !v.is_null()) {
        let k = v
            .as_u64()
            .ok_or_else(|| invalid(format!("top_k must be a non-negative integer, got {v}")))?;
        options.top_k = Some(k as usize);
    }
    if let Some(v) = obj.get("seed").filter(|v| !v.is_null()) {
        let s = v
            .as_u64()
            .ok_or_else(|| invalid(format!("seed must be a non-negative integer, got {v}")))?;
        options.seed = Some(s);
    }

    let tokens = match obj.get("prompt_tokens") {
        Some(Value::Array(arr)) if !arr.is_empty() => arr
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        invalid(format!(
                            "prompt_tokens[{i}] must be an unsigned 32-bit integer, got {v}"
                        ))
                    })
            })
            .collect::<Result<Vec<u32>, _>>()?,
        Some(Value::Array(_)) | None | Some(Value::Null) => {
            if chat {
                let messages = parse_messages(obj.get("messages"))?;
                profile
                    .encode_chat(&messages)
                    .map_err(|e| invalid(e.to_string()))?
            } else {
                let prompt = match obj.get("prompt") {
                    Some(Value::String(s)) => s.as_str(),
                    Some(other) if !other.is_null() => {
                        return Err(invalid(format!(
                            "prompt must be a string (got {other}); batch prompts are not supported"
                        )))
                    }
                    _ => "",
                };
                if prompt.trim().is_empty() {
                    return Err(invalid(
                        "request needs a non-empty prompt or prompt_tokens".into(),
                    ));
                }
                profile
                    .encode_prompt(prompt)
                    .map_err(|e| invalid(e.to_string()))?
            }
        }
        Some(other) => {
            return Err(invalid(format!(
                "prompt_tokens must be an array of integers, got {other}"
            )))
        }
    };
    if tokens.is_empty() {
        return Err(invalid("prompt tokenized to nothing".into()));
    }

    Ok(Prepared {
        tokens,
        max_tokens,
        samples,
        options,
        stream,
    })
}

fn parse_messages(value: Option<&Value>) -> Result<Vec<ChatMessage>, RequestError> {
    let items = value
        .and_then(Value::as_array)
        .ok_or_else(|| RequestError::Invalid("chat requests need a messages array".into()))?;
    if items.is_empty() {
        return Err(RequestError::Invalid("messages must not be empty".into()));
    }
    items
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let role = m.get("role").and_then(Value::as_str).ok_or_else(|| {
                RequestError::Invalid(format!("messages[{i}].role must be a string"))
            })?;
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(_)) => {
                    return Err(RequestError::Unsupported(format!(
                        "messages[{i}].content: only string content is supported"
                    )))
                }
                _ => {
                    return Err(RequestError::Invalid(format!(
                        "messages[{i}].content must be a string"
                    )))
                }
            };
            Ok(ChatMessage {
                role: role.to_string(),
                content,
            })
        })
        .collect()
}

// ── completions ──────────────────────────────────────────────────────────────

fn next_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("cmpl-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Hand a validated request to the engine and write the response.
fn complete(stream: &mut TcpStream, prepared: Prepared, chat: bool, shared: &Shared) {
    let prompt_tokens = prepared.tokens.len();
    let samples = prepared.samples;
    let cancel = Arc::new(AtomicBool::new(false));
    let (reply_tx, reply_rx) = sync_channel(shared.config.max_pending_events.max(1));
    let job = Job {
        tokens: prepared.tokens,
        max_tokens: prepared.max_tokens,
        samples,
        options: prepared.options,
        reply: reply_tx,
        cancel: Arc::clone(&cancel),
    };
    let sender = shared
        .jobs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let Some(sender) = sender else {
        respond_error(
            stream,
            "503 Service Unavailable",
            "engine_unavailable",
            "the server is shutting down",
            None,
        );
        return;
    };
    let submitted = sender.try_send(job);
    drop(sender);
    match submitted {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            respond_error(
                stream,
                "503 Service Unavailable",
                "overloaded_error",
                "the request queue is full; retry shortly",
                Some(1),
            );
            return;
        }
        Err(TrySendError::Disconnected(_)) => {
            respond_error(
                stream,
                "503 Service Unavailable",
                "engine_unavailable",
                "the inference engine is not running",
                None,
            );
            return;
        }
    }

    if prepared.stream {
        stream_sse(
            stream,
            reply_rx,
            cancel,
            chat,
            samples,
            prompt_tokens,
            shared,
        );
        return;
    }

    // Buffered: drop every delta on the floor and wait for the whole answer.
    let completions = loop {
        match reply_rx.recv() {
            Ok(Event::Done(c)) => break c,
            Ok(Event::Delta { .. }) => continue,
            Ok(Event::Rejected(message)) => {
                respond_error(
                    stream,
                    "400 Bad Request",
                    "invalid_request_error",
                    &message,
                    None,
                );
                return;
            }
            Err(_) => {
                respond_error(
                    stream,
                    "503 Service Unavailable",
                    "engine_unavailable",
                    "the inference engine stopped before answering",
                    None,
                );
                return;
            }
        }
    };

    let generated: usize = completions.iter().map(|c| c.tokens.len()).sum();
    let choices: Vec<Value> = completions
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let text = shared.profile.decode(&c.tokens).unwrap_or_default();
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

    respond(
        stream,
        "200 OK",
        &json!({
            "id": next_id(),
            "object": if chat { "chat.completion" } else { "text_completion" },
            "model": shared.config.model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated,
                "total_tokens": prompt_tokens + generated,
            },
        }),
    );
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
/// Every write is checked and bounded by the write timeout. A client that
/// hangs up mid-generation is the normal case — someone closed a tab — and the
/// point of noticing is that the engine then stops: the cancel flag is raised,
/// the scheduler retires the sequence at its next step, and its KV blocks go
/// back to the pool for a request that is still being read. If the engine
/// itself disappears, the stream carries an error event instead of going
/// silent.
fn stream_sse(
    stream: &mut TcpStream,
    rx: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    chat: bool,
    samples: usize,
    prompt_tokens: usize,
    shared: &Shared,
) {
    // The first event decides whether this is a stream at all: a rejection
    // has to be a 400, and it is known before any token is produced.
    let first = match rx.recv() {
        Ok(Event::Rejected(message)) => {
            respond_error(
                stream,
                "400 Bad Request",
                "invalid_request_error",
                &message,
                None,
            );
            return;
        }
        Ok(event) => Some(event),
        Err(_) => {
            respond_error(
                stream,
                "503 Service Unavailable",
                "engine_unavailable",
                "the inference engine stopped before answering",
                None,
            );
            return;
        }
    };

    let head = "HTTP/1.1 200 OK\r\n\
                Content-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\n\
                Connection: close\r\n\
                X-Accel-Buffering: no\r\n\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        cancel.store(true, Ordering::SeqCst);
        return;
    }

    let id = next_id();
    let object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    let mut detok: Vec<IncrementalDetokenizer> = (0..samples)
        .map(|_| IncrementalDetokenizer::new())
        .collect();
    let mut role_sent = vec![false; samples];
    let mut generated = 0usize;

    let send = |stream: &mut TcpStream, payload: &Value| -> bool {
        let frame = format!("data: {payload}\n\n");
        stream.write_all(frame.as_bytes()).is_ok() && stream.flush().is_ok()
    };

    let mut engine_lost = false;
    let mut next = first;
    loop {
        let event = match next.take() {
            Some(e) => e,
            None => match rx.recv() {
                Ok(e) => e,
                Err(_) => {
                    // The sender is gone. Either the request completed and its
                    // slot was dropped after `Done`, or the engine died.
                    engine_lost = !shared.engine_alive.load(Ordering::SeqCst);
                    break;
                }
            },
        };
        let (index, tokens, finish) = match event {
            Event::Delta {
                index,
                tokens,
                finish_reason,
            } => (index, tokens, finish_reason),
            // Completions carry nothing the deltas did not already deliver.
            Event::Done(_) => break,
            Event::Rejected(_) => break,
        };
        if index >= detok.len() {
            continue;
        }

        generated += tokens.len();
        let text = match shared.profile.tokenizer.as_ref() {
            Some(t) => {
                let mut text = detok[index].push(t, &tokens);
                // The last chunk releases whatever the detokenizer was holding
                // back as a possible fragment: nothing more is coming.
                if finish.is_some() {
                    text.push_str(&detok[index].finish());
                }
                text
            }
            None => String::new(),
        };
        // A token that only completes half a character produces no text yet.
        // Suppress that chunk rather than emit an empty delta -- but never
        // suppress the one carrying the finish reason.
        if text.is_empty() && tokens.is_empty() && finish.is_none() {
            continue;
        }

        // OpenAI announces the role once, on the opening chunk of each choice.
        let delta = if chat {
            if std::mem::replace(&mut role_sent[index], true) {
                json!({ "content": text })
            } else {
                json!({"role": "assistant", "content": text})
            }
        } else {
            Value::Null
        };
        let choice = if chat {
            json!({"index": index, "delta": delta, "finish_reason": finish_reason_str(finish)})
        } else {
            json!({"index": index, "text": text, "finish_reason": finish_reason_str(finish)})
        };
        let payload = json!({
            "id": id,
            "object": object,
            "model": shared.config.model_name,
            "choices": [choice],
            // Raw ids alongside the text, matching the buffered response.
            "tokens": tokens,
        });

        if !send(stream, &payload) {
            cancel.store(true, Ordering::SeqCst);
            return;
        }
    }

    if engine_lost {
        let error = json!({"error": {"message": "the inference engine stopped", "type": "engine_unavailable"}});
        let _ = send(stream, &error);
        return;
    }

    let usage = json!({
        "id": id,
        "object": object,
        "model": shared.config.model_name,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated,
            "total_tokens": prompt_tokens + generated,
        },
    });
    if send(stream, &usage) {
        let _ = stream.write_all(b"data: [DONE]\n\n");
        let _ = stream.flush();
    }
}

// ── metrics ──────────────────────────────────────────────────────────────────

fn metrics_body(m: &Metrics, connections: usize) -> Value {
    json!({
        "requests": m.requests,
        "scheduler_steps": m.steps,
        "kv_blocks_total": m.kv_blocks_total,
        "kv_blocks_free": m.kv_blocks_free,
        "sequences_active": m.sequences_active,
        "requests_queued": m.requests_queued,
        "requests_prefilling": m.requests_prefilling,
        "pending_prefill_tokens": m.pending_prefill_tokens,
        "last_prefill_tokens": m.last_prefill_tokens,
        "prefill_chunks": m.prefill_chunks,
        "prefill_preemptions": m.prefill_preemptions,
        "sequences_deferred": m.sequences_deferred,
        "preemptions": m.preemptions,
        "recomputed_tokens": m.recomputed_tokens,
        "connections": connections,
        "prefix_cache_tokens_saved": m.prefix_tokens_saved,
        "prompt_tokens": m.prompt_tokens,
        "prompt_tokens_prefilled": m.prompt_tokens_prefilled,
        "prompt_tokens_reused": m.prompt_tokens_reused,
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

/// Prometheus text exposition format.
///
/// Counters carry the `_total` suffix and gauges do not, because that is what
/// the convention is and a scraper's `rate()` depends on it.
pub fn prometheus(m: &Metrics, connections: usize) -> String {
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
        "paged_infer_prompt_tokens_reused_total",
        "counter",
        "Prompt positions skipped through prefix reuse on initial admission.",
        m.prompt_tokens_reused as f64,
    );
    metric(
        "paged_infer_prefill_chunks_total",
        "counter",
        "Prefill slices processed by the scheduler.",
        m.prefill_chunks as f64,
    );
    metric(
        "paged_infer_prefill_preemptions_total",
        "counter",
        "Unfinished prefills that released their mappings for mandatory decode.",
        m.prefill_preemptions as f64,
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
        "Tokens covered by reused blocks across admissions; includes replayed boundary tokens.",
        m.prefix_tokens_saved as f64,
    );
    metric(
        "paged_infer_copy_on_write_copies_total",
        "counter",
        "Shared KV blocks copied because a forked sequence diverged.",
        m.cow_copies as f64,
    );
    metric(
        "paged_infer_preemptions_total",
        "counter",
        "Sequences that gave up their KV blocks under memory pressure to be recomputed.",
        m.preemptions as f64,
    );
    metric(
        "paged_infer_recomputed_tokens_total",
        "counter",
        "Tokens processed for resumed sequences or repeated after partial-prefill eviction.",
        m.recomputed_tokens as f64,
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
        "paged_infer_sequences_active",
        "gauge",
        "Sequences currently decoding.",
        m.sequences_active as f64,
    );
    metric(
        "paged_infer_requests_queued",
        "gauge",
        "Pending requests, including the retained unfinished prefill.",
        m.requests_queued as f64,
    );
    metric(
        "paged_infer_requests_prefilling",
        "gauge",
        "Requests holding an unfinished prompt mapping.",
        m.requests_prefilling as f64,
    );
    metric(
        "paged_infer_pending_prefill_tokens",
        "gauge",
        "Positions remaining in the retained unfinished prefill.",
        m.pending_prefill_tokens as f64,
    );
    metric(
        "paged_infer_last_prefill_tokens",
        "gauge",
        "Prompt or resumed-sequence positions processed in the last scheduler step.",
        m.last_prefill_tokens as f64,
    );
    metric(
        "paged_infer_sequences_deferred",
        "gauge",
        "Sequences that sat the last step out waiting for a KV block.",
        m.sequences_deferred as f64,
    );
    metric(
        "paged_infer_connections",
        "gauge",
        "Open client connections.",
        connections as f64,
    );
    metric(
        "paged_infer_decode_tokens_per_second",
        "gauge",
        "Generated tokens divided by time spent in decode.",
        m.decode_tok_s,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_server_parts() -> (ServeConfig, ModelProfile, LlamaWeights<'static>) {
        use crate::model::{LlamaConfig, ModelLoader};
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tiny_llama.safetensors"
        ));
        let mut model = LlamaConfig::beside_checkpoint(path).unwrap();
        model.eos_token_ids = vec![111];
        let loader =
            ModelLoader::new(include_bytes!("../tests/fixtures/tiny_llama.safetensors")).unwrap();
        let weights = loader.load_weights(&model).unwrap();
        let profile = ModelProfile::from_parts(model, path, None).unwrap();
        let config = ServeConfig {
            max_jobs_per_step: 1,
            warm_up: false,
            engine: EngineConfig {
                total_blocks: 32,
                block_size: 4,
                max_prefill_tokens_per_step: 4096,
                enable_prefix_cache: false,
                eos_token: u32::MAX,
                stream_tokens: true,
                ..profile.engine_config()
            },
            ..ServeConfig::default()
        };
        (config, profile, weights)
    }

    fn closed_job_queue(
        jobs: Vec<(Vec<u32>, usize, bool)>,
    ) -> (Receiver<Job>, Vec<Receiver<Event>>) {
        let (sender, receiver) = sync_channel(jobs.len());
        let mut replies = Vec::new();
        for (tokens, max_tokens, cancelled) in jobs {
            let (reply, response) = sync_channel(32);
            sender
                .send(Job {
                    tokens,
                    max_tokens,
                    samples: 1,
                    options: RequestOptions::default(),
                    reply,
                    cancel: Arc::new(AtomicBool::new(cancelled)),
                })
                .unwrap();
            replies.push(response);
        }
        drop(sender);
        (receiver, replies)
    }

    fn fixture_shared(config: ServeConfig, profile: ModelProfile) -> Shared {
        Shared {
            config,
            profile,
            jobs: Mutex::new(None),
            metrics: Mutex::new(Metrics::default()),
            ready: AtomicBool::new(true),
            engine_alive: AtomicBool::new(true),
            connections: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    #[test]
    fn idle_wakeup_job_counts_toward_bounded_intake_before_each_step() {
        let (config, profile, weights) = fixture_server_parts();
        let mut engine = Engine::new(weights, profile.config.clone(), config.engine.clone());
        let shared = fixture_shared(config, profile);
        let (jobs, replies) = closed_job_queue((0..5).map(|_| (vec![1], 1, false)).collect());
        engine_loop(&mut engine, jobs, &shared);

        assert_eq!(
            engine.stats().steps,
            5,
            "each wake-up job consumes the entire one-job allowance"
        );
        assert_eq!(engine.stats().requests, 5);
        assert!(!engine.has_work());
        for reply in replies {
            let events: Vec<_> = reply.try_iter().collect();
            assert_eq!(events.len(), 2);
            assert!(
                matches!(&events[0], Event::Delta { tokens, finish_reason: Some(FinishReason::Length), .. } if tokens.len() == 1)
            );
            assert!(
                matches!(&events[1], Event::Done(done) if done.len() == 1 && done[0].tokens.len() == 1)
            );
        }
    }

    #[test]
    fn invalid_and_cancelled_jobs_consume_intake_while_decode_keeps_advancing() {
        let (config, profile, weights) = fixture_server_parts();
        let mut engine = Engine::new(weights, profile.config.clone(), config.engine.clone());
        let shared = fixture_shared(config, profile);
        let mut submitted = vec![
            (vec![1], 8, false),
            (vec![u32::MAX], 1, false),
            (vec![1], 1, true),
            (vec![2], 1, false),
        ];
        submitted.extend((0..4).map(|_| (vec![u32::MAX], 1, false)));
        let (jobs, replies) = closed_job_queue(submitted);
        engine_loop(&mut engine, jobs, &shared);

        assert_eq!(engine.stats().steps, 8);
        assert_eq!(engine.stats().requests, 2);
        assert!(!engine.has_work());
        for (index, reply) in replies.into_iter().enumerate() {
            let events: Vec<_> = reply.try_iter().collect();
            match index {
                0 => {
                    assert_eq!(events.len(), 9);
                    assert!(
                        matches!(events.last(), Some(Event::Done(done)) if done.len() == 1 && done[0].tokens.len() == 8)
                    );
                }
                2 => assert!(
                    events.is_empty(),
                    "already-cancelled job must not be submitted"
                ),
                3 => assert!(
                    matches!(events.last(), Some(Event::Done(done)) if done.len() == 1 && done[0].tokens.len() == 1)
                ),
                _ => assert!(matches!(&events[..], [Event::Rejected(_)])),
            }
        }
    }

    #[test]
    fn zero_scheduler_limits_fail_before_binding_or_starting_threads() {
        for (jobs, budget, expected) in [
            (0, 32, "max_jobs_per_step must be positive"),
            (1, 0, "max_prefill_tokens_per_step must be positive"),
        ] {
            let (mut config, profile, weights) = fixture_server_parts();
            config.addr = "invalid socket address".into();
            config.max_jobs_per_step = jobs;
            config.engine.max_prefill_tokens_per_step = budget;
            let error = run(config, profile, weights)
                .err()
                .expect("zero limit must fail");
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn prefill_metrics_preserve_explicit_reuse_when_retries_exceed_prompt_length() {
        let metrics = Metrics {
            prompt_tokens: 8,
            prompt_tokens_prefilled: 12,
            prompt_tokens_reused: 3,
            requests_prefilling: 1,
            pending_prefill_tokens: 5,
            last_prefill_tokens: 2,
            prefill_chunks: 6,
            prefill_preemptions: 1,
            ..Metrics::default()
        };
        let body = metrics_body(&metrics, 0);
        assert_eq!(body["prompt_tokens_reused"], 3);
        assert_eq!(body["pending_prefill_tokens"], 5);
        let text = prometheus(&metrics, 0);
        assert!(text.contains("paged_infer_prompt_tokens_reused_total 3\n"));
        assert!(text.contains("# TYPE paged_infer_prefill_chunks_total counter\n"));
        assert!(text.contains("paged_infer_prefill_chunks_total 6\n"));
        assert!(text.contains("# TYPE paged_infer_pending_prefill_tokens gauge\n"));
        assert!(text.contains("paged_infer_pending_prefill_tokens 5\n"));
    }

    #[test]
    fn default_valued_unsupported_options_are_tolerated() {
        assert!(is_default_value("stop", &json!(null)));
        assert!(is_default_value("stop", &json!([])));
        assert!(!is_default_value("stop", &json!(["\n"])));
        assert!(is_default_value("logit_bias", &json!({})));
        assert!(!is_default_value("logit_bias", &json!({"1": 5})));
        assert!(is_default_value("presence_penalty", &json!(0)));
        assert!(is_default_value("presence_penalty", &json!(0.0)));
        assert!(!is_default_value("presence_penalty", &json!(0.5)));
        assert!(is_default_value("best_of", &json!(1)));
        assert!(!is_default_value("best_of", &json!(2)));
        assert!(is_default_value(
            "response_format",
            &json!({"type": "text"})
        ));
        assert!(!is_default_value(
            "response_format",
            &json!({"type": "json_object"})
        ));
        assert!(is_default_value("echo", &json!(false)));
        assert!(!is_default_value("echo", &json!(true)));
    }

    #[test]
    fn http_errors_map_to_their_statuses() {
        assert!(HttpError::HeadersTooLarge.status().starts_with("431"));
        assert!(HttpError::PayloadTooLarge {
            declared: 10,
            limit: 1
        }
        .status()
        .starts_with("413"));
        assert!(HttpError::LengthRequired.status().starts_with("411"));
        assert!(HttpError::Timeout.status().starts_with("408"));
        assert!(HttpError::BadRequest("x".into())
            .status()
            .starts_with("400"));
    }

    #[test]
    fn prometheus_exposition_names_every_series_once() {
        let text = prometheus(&Metrics::default(), 3);
        for name in [
            "paged_infer_requests_total",
            "paged_infer_preemptions_total",
            "paged_infer_sequences_deferred",
            "paged_infer_connections 3",
        ] {
            assert!(text.contains(name), "{name} missing from\n{text}");
        }
        assert_eq!(
            text.matches("# TYPE paged_infer_kv_blocks gauge").count(),
            1
        );
    }
}
