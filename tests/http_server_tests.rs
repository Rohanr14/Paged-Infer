//! The HTTP layer, over real sockets, against the fixture model.
//!
//! Every test starts a server on an ephemeral port with the fixture
//! checkpoint (and, where text is involved, a word-level tokenizer whose ids
//! fit its vocabulary) and speaks raw HTTP/1.1 to it. The engine's own tests
//! never touch the transport; these are the ones that prove a bad request is a
//! 400 and not a stall, that an oversized body is refused before it is read,
//! that a silent client is timed out, and that a stream ends the way OpenAI
//! clients expect.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use paged_infer::engine::EngineConfig;
use paged_infer::model::{LlamaConfig, LlamaWeights, ModelLoader};
use paged_infer::profile::ModelProfile;
use paged_infer::serve::{run, ServeConfig, Server};
use serde_json::{json, Value};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn model_config() -> LlamaConfig {
    let mut config = LlamaConfig::from_hf_config(fixtures().join("config.json")).unwrap();
    // The fixture declares no EOS; pick the last id so a spurious early stop
    // is unlikely and a profile without a tokenizer still has one.
    config.eos_token_ids = vec![111];
    config
}

/// The fixture weights, leaked: the server's threads borrow them for `'static`.
fn load_weights(config: &LlamaConfig) -> LlamaWeights<'static> {
    let bytes: &'static [u8] = Box::leak(
        std::fs::read(fixtures().join("tiny_llama.safetensors"))
            .unwrap()
            .into_boxed_slice(),
    );
    let loader: &'static ModelLoader<'static> =
        Box::leak(Box::new(ModelLoader::new(bytes).unwrap()));
    loader.load_weights(config).unwrap()
}

/// A word-level tokenizer whose ids fit the fixture model's vocabulary, with
/// TinyLlama's chat template in its config (or none), in a fresh directory.
fn small_tokenizer_dir(with_template: bool) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "paged-infer-http-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "the": 5, "cat": 6, "<|user|>": 7, "<|assistant|>": 8, "<|system|>": 9}, "unk_token": "<unk>"}
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let mut config = json!({
        "add_bos_token": true,
        "bos_token": "<s>",
        "eos_token": "</s>",
    });
    if with_template {
        let template =
            std::fs::read_to_string(fixtures().join("tokenizer/tokenizer_config.json")).unwrap();
        let template: Value = serde_json::from_str(&template).unwrap();
        config["chat_template"] = template["chat_template"].clone();
    }
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
    dir
}

fn start(tokenizer_dir: Option<&PathBuf>, tweak: impl FnOnce(&mut ServeConfig)) -> Server {
    let config = model_config();
    let weights = load_weights(&config);
    let tokenizer_path = tokenizer_dir.map(|d| d.join("tokenizer.json"));
    let profile = ModelProfile::from_parts(
        config,
        &fixtures().join("tiny_llama.safetensors"),
        tokenizer_path.as_deref(),
    )
    .unwrap();
    let mut serve_config = ServeConfig {
        addr: "127.0.0.1:0".to_string(),
        warm_up: false,
        read_timeout: Duration::from_secs(5),
        write_timeout: Duration::from_secs(5),
        engine: EngineConfig {
            total_blocks: 64,
            block_size: 8,
            ..profile.engine_config()
        },
        ..ServeConfig::default()
    };
    tweak(&mut serve_config);
    let server = run(serve_config, profile, weights).unwrap();
    wait_ready(server.addr());
    server
}

fn wait_ready(addr: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok((200, _, _)) = get(addr, "/health") {
            return;
        }
        assert!(Instant::now() < deadline, "server never became ready");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Send raw bytes, return (status, headers, body). `Connection: close` is the
/// server's default, so reading to EOF is the whole response.
fn raw(addr: SocketAddr, request: &[u8]) -> std::io::Result<(u16, String, String)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(20)))?;
    // The server may answer and close before it has read all of an oversized
    // request; the write error that causes is not the interesting part.
    let _ = stream.write_all(request);
    let mut response = Vec::new();
    if let Err(e) = stream.read_to_end(&mut response) {
        if response.is_empty() {
            return Err(e);
        }
    }
    let text = String::from_utf8_lossy(&response).into_owned();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("no header terminator in response: {text:?}"));
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no status in {head:?}"));
    Ok((status, head.to_string(), body.to_string()))
}

fn get(addr: SocketAddr, path: &str) -> std::io::Result<(u16, String, String)> {
    raw(
        addr,
        format!("GET {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n").as_bytes(),
    )
}

fn post(addr: SocketAddr, path: &str, body: &Value) -> (u16, Value) {
    let body = body.to_string();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let (status, _, text) = raw(addr, request.as_bytes()).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text}));
    (status, value)
}

fn post_raw(addr: SocketAddr, path: &str, body: &Value) -> (u16, String, String) {
    let body = body.to_string();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    raw(addr, request.as_bytes()).unwrap()
}

/// Parse an SSE body into its `data:` payloads.
fn sse_frames(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter_map(|frame| frame.strip_prefix("data: ").map(str::to_string))
        .collect()
}

fn error_message(v: &Value) -> String {
    v["error"]["message"].as_str().unwrap_or("").to_string()
}

// ── basics ───────────────────────────────────────────────────────────────────

#[test]
fn test_health_and_models_reflect_the_running_engine() {
    let server = start(None, |c| c.model_name = "fixture-llama".into());
    let (status, _, body) = get(server.addr(), "/health").unwrap();
    assert_eq!(status, 200);
    let health: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["ready"], true);
    assert_eq!(health["engine_alive"], true);
    assert_eq!(health["kv_blocks_total"], 64);

    let (status, _, body) = get(server.addr(), "/v1/models").unwrap();
    assert_eq!(status, 200);
    assert!(body.contains("fixture-llama"));

    let (status, _, _) = get(server.addr(), "/nope").unwrap();
    assert_eq!(status, 404);
    server.shutdown();
}

#[test]
fn test_prompt_tokens_are_the_complete_input_and_complete() {
    let server = start(None, |_| {});
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": [5, 6, 7, 8], "max_tokens": 4}),
    );
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["usage"]["prompt_tokens"], 4, "no BOS is added to ids");
    let tokens = v["choices"][0]["tokens"].as_array().unwrap();
    assert!(!tokens.is_empty() && tokens.len() <= 4);
    assert_eq!(v["usage"]["completion_tokens"], tokens.len());
    assert!(matches!(
        v["choices"][0]["finish_reason"].as_str(),
        Some("length") | Some("stop")
    ));
    server.shutdown();
}

// ── validation ───────────────────────────────────────────────────────────────

#[test]
fn test_invalid_token_ids_are_client_errors_and_the_server_keeps_serving() {
    let server = start(None, |_| {});
    // 2^32 + 1 used to be cast to token 1.
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": [4294967297u64], "max_tokens": 2}),
    );
    assert_eq!(status, 400, "{v}");
    assert!(error_message(&v).contains("32-bit"), "{v}");

    // In range for u32, outside the vocabulary: refused by the engine, still
    // reported as the client's mistake.
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": [1, 99999], "max_tokens": 2}),
    );
    assert_eq!(status, 400, "{v}");
    assert!(error_message(&v).contains("outside the vocabulary"), "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");

    for body in [
        json!({"prompt_tokens": [1, 2], "max_tokens": 0}),
        json!({"prompt_tokens": [1, 2], "n": 0}),
        json!({"prompt_tokens": [1, 2], "n": 999}),
        json!({"prompt_tokens": [1, 2], "temperature": -1}),
        json!({"prompt_tokens": [1, 2], "top_p": 1.5}),
        json!({"prompt_tokens": "1,2"}),
        json!({"prompt": ["a", "b"]}),
        json!([1, 2]),
    ] {
        let (status, v) = post(server.addr(), "/v1/completions", &body);
        assert_eq!(status, 400, "{body} -> {v}");
    }
    let (status, _, body) = raw(
        server.addr(),
        b"POST /v1/completions HTTP/1.1\r\nHost: t\r\nContent-Length: 5\r\n\r\n{nope",
    )
    .unwrap();
    assert_eq!(status, 400, "{body}");

    // And it still works afterwards.
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": [5, 6, 7], "max_tokens": 2}),
    );
    assert_eq!(status, 200, "{v}");
    server.shutdown();
}

#[test]
fn test_a_prompt_that_can_never_fit_is_refused_and_a_follower_completes() {
    // The review's stall: two eight-token blocks, a 17-token request, then a
    // valid one. The first is a 400; the second finishes.
    let server = start(None, |c| {
        c.engine.total_blocks = 2;
        c.engine.enable_prefix_cache = false;
    });
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": (1..=17).collect::<Vec<u32>>(), "max_tokens": 4}),
    );
    assert_eq!(status, 400, "{v}");
    assert!(error_message(&v).contains("does not fit"), "{v}");

    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": [1, 2, 3, 4], "max_tokens": 4}),
    );
    assert_eq!(status, 200, "{v}");
    assert!(!v["choices"][0]["tokens"].as_array().unwrap().is_empty());
    server.shutdown();
}

#[test]
fn test_unsupported_options_are_refused_and_defaults_are_tolerated() {
    let server = start(None, |_| {});
    for (key, value) in [
        ("stop", json!(["\n"])),
        ("logit_bias", json!({"5": 10})),
        ("logprobs", json!(true)),
        ("presence_penalty", json!(0.5)),
        ("tools", json!([{"type": "function"}])),
        ("response_format", json!({"type": "json_object"})),
        ("best_of", json!(3)),
    ] {
        let mut body = json!({"prompt_tokens": [5, 6], "max_tokens": 1});
        body[key] = value.clone();
        let (status, v) = post(server.addr(), "/v1/completions", &body);
        assert_eq!(status, 400, "{key}={value}: {v}");
        assert_eq!(v["error"]["type"], "unsupported_option", "{key}");
    }
    // The values clients send by default are not a reason to refuse.
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({
            "prompt_tokens": [5, 6], "max_tokens": 1,
            "stop": null, "logit_bias": {}, "logprobs": false,
            "presence_penalty": 0, "frequency_penalty": 0.0,
            "response_format": {"type": "text"}, "best_of": 1, "tools": [],
            "user": "someone"
        }),
    );
    assert_eq!(status, 200, "{v}");
    server.shutdown();
}

#[test]
fn test_explicit_greedy_with_n_yields_identical_choices() {
    let server = start(None, |_| {});
    let choices = |body: Value| -> Vec<Vec<u64>> {
        let (status, v) = post(server.addr(), "/v1/completions", &body);
        assert_eq!(status, 200, "{v}");
        v["choices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                c["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t.as_u64().unwrap())
                    .collect()
            })
            .collect()
    };
    let greedy = choices(
        json!({"prompt_tokens": [5, 6, 7, 8, 9], "max_tokens": 6, "n": 3, "temperature": 0}),
    );
    assert_eq!(greedy.len(), 3);
    assert!(greedy.iter().all(|c| *c == greedy[0]), "{greedy:?}");

    let sampled =
        choices(json!({"prompt_tokens": [5, 6, 7, 8, 9], "max_tokens": 6, "n": 3, "seed": 7}));
    assert_eq!(sampled.len(), 3);
    assert!(sampled.iter().any(|c| *c != sampled[0]), "{sampled:?}");
    // A seed replays.
    let again =
        choices(json!({"prompt_tokens": [5, 6, 7, 8, 9], "max_tokens": 6, "n": 3, "seed": 7}));
    assert_eq!(sampled, again);
    server.shutdown();
}

// ── streaming ────────────────────────────────────────────────────────────────

#[test]
fn test_streaming_delivers_the_buffered_tokens_and_ends_with_done() {
    let server = start(None, |_| {});
    let request = json!({"prompt_tokens": [5, 6, 7, 8], "max_tokens": 6, "temperature": 0});
    let (status, buffered) = post(server.addr(), "/v1/completions", &request);
    assert_eq!(status, 200);
    let expected: Vec<u64> = buffered["choices"][0]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap())
        .collect();

    let mut streaming = request.clone();
    streaming["stream"] = json!(true);
    let (status, head, body) = post_raw(server.addr(), "/v1/completions", &streaming);
    assert_eq!(status, 200, "{body}");
    assert!(head.contains("text/event-stream"), "{head}");
    let frames = sse_frames(&body);
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"), "{body}");

    let mut tokens = Vec::new();
    let mut finish = None;
    let mut usage = None;
    for frame in &frames[..frames.len() - 1] {
        let v: Value = serde_json::from_str(frame).unwrap();
        if let Some(u) = v.get("usage") {
            usage = Some(u.clone());
            continue;
        }
        tokens.extend(
            v["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap()),
        );
        if let Some(reason) = v["choices"][0]["finish_reason"].as_str() {
            finish = Some(reason.to_string());
        }
    }
    assert_eq!(
        tokens, expected,
        "streamed tokens differ from the buffered answer"
    );
    assert!(finish.is_some(), "the stream never carried a finish reason");
    let usage = usage.expect("usage frame before [DONE]");
    assert_eq!(usage["completion_tokens"], expected.len());
    assert_eq!(usage["prompt_tokens"], 4);
    server.shutdown();
}

#[test]
fn test_a_client_that_disconnects_mid_stream_frees_its_blocks() {
    let server = start(None, |c| {
        c.engine.enable_prefix_cache = false;
    });
    let request = json!({"prompt_tokens": [5, 6, 7, 8, 9, 10, 11, 12, 13], "max_tokens": 400, "stream": true});
    let body = request.to_string();
    let mut stream = TcpStream::connect(server.addr()).unwrap();
    stream
        .write_all(
            format!(
                "POST /v1/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .unwrap();
    // Read something, then hang up.
    let mut buf = [0u8; 256];
    let n = stream.read(&mut buf).unwrap();
    assert!(n > 0);
    drop(stream);

    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (_, _, body) = get(server.addr(), "/stats").unwrap();
        let stats: Value = serde_json::from_str(&body).unwrap();
        if stats["kv_blocks_free"] == stats["kv_blocks_total"] && stats["sequences_active"] == 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "blocks were not returned after the client left: {stats}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    server.shutdown();
}

// ── bounds ───────────────────────────────────────────────────────────────────

#[test]
fn test_an_oversized_body_is_refused_before_it_is_read() {
    let server = start(None, |c| c.max_body_bytes = 4096);
    let started = Instant::now();
    let (status, _, body) = raw(
        server.addr(),
        b"POST /v1/completions HTTP/1.1\r\nHost: t\r\nContent-Length: 10000000000\r\n\r\n",
    )
    .unwrap();
    assert_eq!(status, 413, "{body}");
    assert!(body.contains("exceeds the limit"), "{body}");
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the server waited for a body it should have refused outright"
    );

    // A missing Content-Length on a POST is 411, not a hang.
    let (status, _, _) = raw(
        server.addr(),
        b"POST /v1/completions HTTP/1.1\r\nHost: t\r\n\r\n",
    )
    .unwrap();
    assert_eq!(status, 411);
    server.shutdown();
}

#[test]
fn test_oversized_headers_are_refused() {
    let server = start(None, |c| c.max_header_bytes = 2048);
    let long_path = "a".repeat(4096);
    let (status, _, _) = raw(
        server.addr(),
        format!("GET /{long_path} HTTP/1.1\r\nHost: t\r\n\r\n").as_bytes(),
    )
    .unwrap();
    assert_eq!(status, 431);
    // A single header line that never ends is caught by the same cap.
    let mut request = b"GET /health HTTP/1.1\r\nX-Junk: ".to_vec();
    request.extend(std::iter::repeat_n(b'x', 8192));
    let (status, _, _) = raw(server.addr(), &request).unwrap();
    assert_eq!(status, 431);
    server.shutdown();
}

#[test]
fn test_a_silent_client_is_timed_out_and_does_not_block_others() {
    let server = start(None, |c| c.read_timeout = Duration::from_millis(300));
    let mut silent = TcpStream::connect(server.addr()).unwrap();
    silent
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    // Meanwhile a normal request goes through.
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": [5, 6], "max_tokens": 1}),
    );
    assert_eq!(status, 200, "{v}");
    // The silent connection is closed by the server.
    let mut buf = Vec::new();
    let started = Instant::now();
    silent.read_to_end(&mut buf).unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "silent client was not timed out"
    );
    server.shutdown();
}

#[test]
fn test_connections_beyond_the_cap_get_503_with_retry_after() {
    let server = start(None, |c| {
        c.max_connections = 1;
        c.read_timeout = Duration::from_secs(3);
    });
    let _held = TcpStream::connect(server.addr()).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let (status, head, body) = get(server.addr(), "/health").unwrap();
    assert_eq!(status, 503, "{body}");
    assert!(head.contains("Retry-After"), "{head}");
    assert!(body.contains("too many open connections"), "{body}");
    drop(_held);
    // Once the held connection times out, capacity comes back.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok((200, _, _)) = get(server.addr(), "/health") {
            break;
        }
        assert!(Instant::now() < deadline, "capacity never came back");
        std::thread::sleep(Duration::from_millis(100));
    }
    server.shutdown();
}

// ── text and chat through the profile ────────────────────────────────────────

#[test]
fn test_text_and_chat_requests_go_through_the_model_profile() {
    let dir = small_tokenizer_dir(true);
    let server = start(Some(&dir), |_| {});

    // Text: the tokenizer adds BOS, the profile adds none on top.
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt": "hello world", "max_tokens": 3}),
    );
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["usage"]["prompt_tokens"], 3, "BOS + two words, once: {v}");
    assert!(v["choices"][0]["text"].is_string());

    // Chat: rendered through TinyLlama's template. Its markers are single
    // pieces in this tokenizer, and the template puts no BOS in.
    let (status, v) = post(
        server.addr(),
        "/v1/chat/completions",
        &json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 3}),
    );
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(v["choices"][0]["message"]["content"].is_string());
    // The rendered "<|user|>\nhello</s>\n<|assistant|>\n" splits on whitespace
    // into the two markers, the word and the special </s>: four ids, no BOS.
    assert_eq!(v["usage"]["prompt_tokens"], 4, "{v}");

    let (status, v) = post(
        server.addr(),
        "/v1/chat/completions",
        &json!({"messages": [{"role": "tool", "content": "x"}]}),
    );
    assert_eq!(status, 400, "{v}");
    assert!(error_message(&v).contains("not supported"), "{v}");
    let (status, v) = post(
        server.addr(),
        "/v1/chat/completions",
        &json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}),
    );
    assert_eq!(status, 400, "{v}");
    let (status, v) = post(
        server.addr(),
        "/v1/chat/completions",
        &json!({"messages": []}),
    );
    assert_eq!(status, 400, "{v}");

    // Streaming chat announces the role once and closes with [DONE].
    let (status, _, body) = post_raw(
        server.addr(),
        "/v1/chat/completions",
        &json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 3, "stream": true}),
    );
    assert_eq!(status, 200, "{body}");
    let frames = sse_frames(&body);
    let first: Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"));
    server.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_chat_without_a_template_and_text_without_a_tokenizer_are_refused() {
    let dir = small_tokenizer_dir(false);
    let server = start(Some(&dir), |_| {});
    let (status, v) = post(
        server.addr(),
        "/v1/chat/completions",
        &json!({"messages": [{"role": "user", "content": "hello"}]}),
    );
    assert_eq!(status, 400, "{v}");
    assert!(error_message(&v).contains("no chat_template"), "{v}");
    server.shutdown();
    let _ = std::fs::remove_dir_all(&dir);

    let server = start(None, |_| {});
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt": "hello"}),
    );
    assert_eq!(status, 400, "{v}");
    assert!(error_message(&v).contains("no tokenizer"), "{v}");
    server.shutdown();
}

// ── observability ────────────────────────────────────────────────────────────

#[test]
fn test_metrics_and_stats_expose_scheduler_state() {
    let server = start(None, |_| {});
    let (status, v) = post(
        server.addr(),
        "/v1/completions",
        &json!({"prompt_tokens": [5, 6, 7], "max_tokens": 2}),
    );
    assert_eq!(status, 200, "{v}");

    let (status, _, text) = get(server.addr(), "/metrics").unwrap();
    assert_eq!(status, 200);
    for series in [
        "paged_infer_requests_total 1",
        "paged_infer_sequences_deferred",
        "paged_infer_preemptions_total",
        "paged_infer_requests_queued",
        "paged_infer_connections",
        "paged_infer_kv_blocks 64",
    ] {
        assert!(text.contains(series), "{series} missing from:\n{text}");
    }
    let (status, _, body) = get(server.addr(), "/stats").unwrap();
    assert_eq!(status, 200);
    let stats: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(stats["requests"], 1);
    assert_eq!(stats["kv_blocks_total"], 64);
    assert!(stats["sequences_deferred"].is_number());
    assert!(stats["preemptions"].is_number());
    server.shutdown();
}
