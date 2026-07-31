//! Streaming must not be a second implementation of generation.
//!
//! The risk in adding incremental output is that the streamed text and the
//! buffered text drift apart — a token dropped at a chunk boundary, an EOS that
//! closes the completion but never closes the stream, a speculative step that
//! emits three tokens where the stream reports one. These tests pin the two
//! views together: whatever a batch caller receives, a streaming caller must
//! receive the same tokens in the same order, and must be told when it ends.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::detokenizer::new_text;
use paged_infer::engine::{Engine, EngineConfig, FinishReason};
use paged_infer::model::{LlamaConfig, LlamaWeights, ModelLoader};

fn fixture() -> (LlamaConfig, Vec<u32>, Vec<u8>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let meta = std::fs::read_to_string(dir.join("tiny_llama_meta.txt")).unwrap();
    let kv: HashMap<&str, &str> = meta
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect();
    let num = |k: &str| kv[k].parse::<usize>().unwrap();
    let config = LlamaConfig {
        hidden_size: num("hidden_size"),
        num_hidden_layers: num("num_hidden_layers"),
        num_attention_heads: num("num_attention_heads"),
        num_key_value_heads: num("num_key_value_heads"),
        intermediate_size: num("intermediate_size"),
        vocab_size: num("vocab_size"),
        rms_norm_eps: kv["rms_norm_eps"].parse().unwrap(),
        rope_theta: kv["rope_theta"].parse().unwrap(),
        attention_window: None,
        rope_style: Default::default(),
        quantization: Default::default(),
    };
    let tokens = kv["tokens"]
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    (
        config,
        tokens,
        std::fs::read(dir.join("tiny_llama.safetensors")).unwrap(),
    )
}

fn base_config() -> EngineConfig {
    EngineConfig {
        total_blocks: 128,
        block_size: 8,
        temperature: 0.0,
        bos_token: None,
        eos_token: u32::MAX,
        stream_tokens: true,
        ..EngineConfig::default()
    }
}

fn with_engine<R>(engine_config: EngineConfig, body: impl FnOnce(Engine<'_>) -> R) -> R {
    let (config, _, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights: LlamaWeights<'_> = loader.load_weights(&config).unwrap();
    body(Engine::new(weights, config, engine_config))
}

/// Run to completion, collecting the streamed deltas per sequence alongside the
/// completions, so the two can be compared.
struct Streamed {
    /// sequence id -> tokens, in delta order.
    per_sequence: HashMap<usize, Vec<u32>>,
    /// sequence id -> how many deltas carried a finish reason.
    terminal_deltas: HashMap<usize, usize>,
    completions: HashMap<usize, (Vec<u32>, FinishReason)>,
}

fn drain(engine: &mut Engine<'_>) -> Streamed {
    let mut out = Streamed {
        per_sequence: HashMap::new(),
        terminal_deltas: HashMap::new(),
        completions: HashMap::new(),
    };
    while engine.has_work() {
        engine.step().unwrap();
        for d in engine.take_deltas() {
            out.per_sequence
                .entry(d.sequence_id)
                .or_default()
                .extend_from_slice(&d.tokens);
            if d.finish_reason.is_some() {
                *out.terminal_deltas.entry(d.sequence_id).or_default() += 1;
            }
        }
        for c in engine.take_completed() {
            out.completions
                .insert(c.sequence_id, (c.tokens, c.finish_reason));
        }
    }
    out
}

#[test]
fn test_streamed_tokens_reconstruct_the_completion_exactly() {
    let (_, tokens, _) = fixture();
    with_engine(base_config(), |mut engine| {
        engine.submit_tokens(tokens[..12].to_vec(), 20, 1);
        engine.submit_tokens(tokens[..25].to_vec(), 14, 1);
        let s = drain(&mut engine);

        assert_eq!(s.completions.len(), 2);
        for (seq_id, (completion, _)) in &s.completions {
            assert_eq!(
                s.per_sequence.get(seq_id),
                Some(completion),
                "streamed tokens for sequence {seq_id} do not match its completion"
            );
        }
    });
}

#[test]
fn test_every_sequence_is_told_exactly_once_that_it_ended() {
    // A stream that never closes is worse than one that never opens: the client
    // waits forever. Exactly one delta per sequence carries a finish reason.
    let (_, tokens, _) = fixture();
    with_engine(base_config(), |mut engine| {
        engine.submit_tokens(tokens[..10].to_vec(), 8, 1);
        engine.submit_tokens(tokens[..16].to_vec(), 5, 1);
        let s = drain(&mut engine);

        assert_eq!(s.completions.len(), 2);
        for seq_id in s.completions.keys() {
            assert_eq!(
                s.terminal_deltas.get(seq_id).copied().unwrap_or(0),
                1,
                "sequence {seq_id} did not get exactly one terminal delta"
            );
        }
    });
}

#[test]
fn test_a_speculative_step_streams_the_whole_accepted_run() {
    // Speculation emits several tokens in one step. Those tokens really are all
    // available at that instant, so the stream should carry them — and the
    // total must still be the greedy output, token for token.
    let (_, tokens, _) = fixture();
    let mut unit = Vec::new();
    for _ in 0..6 {
        unit.extend_from_slice(&tokens[..6]);
    }

    let plain = with_engine(base_config(), |mut engine| {
        engine.submit_tokens(unit.clone(), 24, 1);
        drain(&mut engine)
    });
    let spec = with_engine(
        EngineConfig {
            draft_tokens: 4,
            ..base_config()
        },
        |mut engine| {
            engine.submit_tokens(unit.clone(), 24, 1);
            drain(&mut engine)
        },
    );

    let plain_tokens: Vec<u32> = plain.per_sequence.values().next().unwrap().clone();
    let spec_tokens: Vec<u32> = spec.per_sequence.values().next().unwrap().clone();
    assert_eq!(
        plain_tokens, spec_tokens,
        "streaming under speculation diverged from greedy"
    );
    for (seq_id, (completion, _)) in &spec.completions {
        assert_eq!(spec.per_sequence.get(seq_id), Some(completion));
    }
}

#[test]
fn test_deltas_are_off_unless_asked_for() {
    // A batch caller never drains them, so recording deltas it will not read is
    // an unbounded buffer of the whole run's output.
    let (_, tokens, _) = fixture();
    with_engine(
        EngineConfig {
            stream_tokens: false,
            ..base_config()
        },
        |mut engine| {
            engine.submit_tokens(tokens[..12].to_vec(), 16, 1);
            let s = drain(&mut engine);
            assert!(
                s.per_sequence.is_empty(),
                "deltas leaked with streaming off"
            );
            assert_eq!(s.completions.len(), 1);
        },
    );
}

#[test]
fn test_forked_samples_stream_under_distinct_sequence_ids() {
    // n>1 shares one prompt and one request id; the stream has to keep the
    // branches apart or a client interleaves four continuations into mush.
    let (_, tokens, _) = fixture();
    with_engine(base_config(), |mut engine| {
        let request_id = engine.submit_tokens(tokens[..14].to_vec(), 12, 4);
        let s = drain(&mut engine);
        assert_eq!(request_id, 0);
        assert_eq!(s.completions.len(), 4);
        assert_eq!(s.per_sequence.len(), 4);
        for (seq_id, (completion, _)) in &s.completions {
            assert_eq!(s.per_sequence.get(seq_id), Some(completion));
        }
    });
}

#[test]
fn test_cancelling_stops_generation_and_returns_the_blocks() {
    // What a disconnected streaming client should cost: nothing further.
    let (_, tokens, _) = fixture();
    with_engine(base_config(), |mut engine| {
        let total = engine.total_blocks();
        let keep = engine.submit_tokens(tokens[..12].to_vec(), 40, 1);
        let drop_me = engine.submit_tokens(tokens[..20].to_vec(), 40, 1);

        for _ in 0..4 {
            engine.step().unwrap();
        }
        let held_before = total - engine.available_blocks();
        assert_eq!(engine.cancel_request(drop_me), 1);

        let cancelled = engine
            .take_completed()
            .into_iter()
            .find(|c| c.request_id == drop_me)
            .expect("cancelling should complete the request immediately");
        assert_eq!(cancelled.finish_reason, FinishReason::Cancelled);
        assert!(
            !cancelled.tokens.is_empty() && cancelled.tokens.len() < 40,
            "cancelled mid-run, so it should hold a partial answer: {} tokens",
            cancelled.tokens.len()
        );
        assert!(
            total - engine.available_blocks() < held_before,
            "cancelling did not return any blocks to the pool"
        );

        // The surviving request keeps running and finishes normally.
        let rest = engine.run().unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].request_id, keep);
        assert_eq!(rest[0].tokens.len(), 40);
    });
}

#[test]
fn test_cancelling_a_queued_request_never_admits_it() {
    let (_, tokens, _) = fixture();
    with_engine(base_config(), |mut engine| {
        let first = engine.submit_tokens(tokens[..12].to_vec(), 6, 1);
        let queued = engine.submit_tokens(tokens[..12].to_vec(), 6, 1);
        assert_eq!(engine.cancel_request(queued), 1);

        let out = engine.run().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].request_id, first);
    });
}

#[test]
fn test_warm_up_leaves_no_trace() {
    // A warmed engine has to be indistinguishable from a fresh one: same
    // output, and counters that describe the workload rather than the warm-up.
    let (_, tokens, _) = fixture();
    let cold = with_engine(base_config(), |mut engine| {
        engine.submit_tokens(tokens[..18].to_vec(), 16, 1);
        engine.run().unwrap()
    });

    with_engine(base_config(), |mut engine| {
        engine.warm_up();
        assert_eq!(engine.stats().requests, 0);
        assert_eq!(engine.stats().prompt_tokens, 0);
        assert_eq!(engine.prefix_stats(), Default::default());
        assert_eq!(engine.available_blocks(), engine.total_blocks());

        engine.submit_tokens(tokens[..18].to_vec(), 16, 1);
        let warm = engine.run().unwrap();
        assert_eq!(warm.len(), cold.len());
        assert_eq!(
            warm[0].tokens, cold[0].tokens,
            "warming up changed the generated text"
        );
        assert_eq!(engine.stats().prompt_tokens, cold[0].prompt_tokens);
    });
}

#[test]
fn test_warm_up_still_lets_the_prefix_cache_work() {
    // The warm-up runs a synthetic prompt through the model. If its blocks were
    // published, a later request could match them and reuse garbage KV.
    let (_, tokens, _) = fixture();
    with_engine(base_config(), |mut engine| {
        engine.warm_up();
        let baseline = {
            engine.submit_tokens(tokens[..20].to_vec(), 8, 1);
            engine.run().unwrap()[0].tokens.clone()
        };
        engine.submit_tokens(tokens[..20].to_vec(), 8, 1);
        let reused = engine.run().unwrap()[0].tokens.clone();

        assert_eq!(baseline, reused, "prefix reuse changed the answer");
        assert!(
            engine.prefix_stats().hits > 0,
            "the second identical prompt should have hit the cache"
        );
    });
}

#[test]
fn test_incremental_text_matches_a_single_decode() {
    // The property the SSE handler depends on: concatenating the per-chunk
    // deltas reproduces exactly what decoding the whole token list produces.
    let full = "the quick brown fox";
    let checkpoints = ["", "the", "the quick", "the quick brown", full];
    let mut emitted = String::new();
    let mut joined = String::new();
    for c in checkpoints {
        joined.push_str(new_text(&emitted, c));
        emitted = c.to_string();
    }
    assert_eq!(joined, full);
}
