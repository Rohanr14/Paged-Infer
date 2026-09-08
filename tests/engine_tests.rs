//! End-to-end scheduler tests against the synthetic fixture checkpoint.
//!
//! These drive the real `Engine` — admission, prefill, prefix reuse, decode,
//! block growth, copy-on-write forking, reclamation — with real weights. The
//! model is tiny, but every code path the serving loop takes is the production
//! one, so a scheduler bug shows up here rather than only under a 1B checkpoint
//! nobody can run in CI.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::engine::{Engine, EngineConfig, FinishReason, SubmitError};
use paged_infer::model::{LlamaConfig, ModelLoader};

fn fixture() -> (LlamaConfig, Vec<u32>, Vec<u8>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let meta = std::fs::read_to_string(dir.join("tiny_llama_meta.txt"))
        .expect("missing fixture: run python3 scripts/gen_golden_fixture.py");
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
        ..LlamaConfig::default()
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

fn engine_config() -> EngineConfig {
    EngineConfig {
        total_blocks: 128,
        block_size: 8,
        temperature: 0.0,
        // No BOS: the fixture has no special tokens, and prepending one would
        // just shift every position.
        bos_token: None,
        // Pick a token the fixture is unlikely to emit, so tests that want a
        // length stop are not cut short by a spurious EOS.
        eos_token: u32::MAX,
        ..EngineConfig::default()
    }
}

#[test]
fn test_generates_the_requested_number_of_tokens() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(weights, config, engine_config());

    engine.submit_tokens(tokens[..12].to_vec(), 16, 1).unwrap();
    let out = engine.run().unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].tokens.len(), 16);
    assert_eq!(out[0].finish_reason, FinishReason::Length);
    assert_eq!(out[0].prompt_tokens, 12);
    assert_eq!(engine.stats().generated_tokens, 16);
}

#[test]
fn test_eos_stops_generation_early() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();

    // Learn what the model actually generates, then declare its third token EOS
    // and confirm the engine stops exactly there.
    let mut probe = Engine::new(
        loader.load_weights(&config).unwrap(),
        config.clone(),
        engine_config(),
    );
    probe.submit_tokens(tokens[..12].to_vec(), 8, 1).unwrap();
    let baseline = probe.run().unwrap();
    let eos = baseline[0].tokens[2];

    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            eos_token: eos,
            ..engine_config()
        },
    );
    engine.submit_tokens(tokens[..12].to_vec(), 8, 1).unwrap();
    let out = engine.run().unwrap();

    assert_eq!(out[0].finish_reason, FinishReason::Eos);
    assert_eq!(out[0].tokens.len(), 3, "should stop on the EOS token");
    assert_eq!(*out[0].tokens.last().unwrap(), eos);
}

#[test]
fn test_generation_crosses_block_boundaries() {
    // 12 prompt tokens plus 40 generated spans seven 8-token blocks, so the
    // sequence has to grow its mapping repeatedly mid-decode.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(weights, config, engine_config());

    engine.submit_tokens(tokens[..12].to_vec(), 40, 1).unwrap();
    let out = engine.run().unwrap();
    assert_eq!(out[0].tokens.len(), 40);
    assert_eq!(out[0].finish_reason, FinishReason::Length);
}

#[test]
fn test_greedy_generation_is_deterministic() {
    let run_once = || {
        let (config, tokens, bytes) = fixture();
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(weights, config, engine_config());
        engine.submit_tokens(tokens[..16].to_vec(), 12, 1).unwrap();
        engine.run().unwrap()[0].tokens.clone()
    };
    assert_eq!(run_once(), run_once());
}

#[test]
fn test_prefix_cache_cuts_prefill_without_changing_output() {
    let (config, tokens, bytes) = fixture();
    let shared = &tokens[..24];

    let run = |prefix_cache: bool| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(
            weights,
            config.clone(),
            EngineConfig {
                enable_prefix_cache: prefix_cache,
                ..engine_config()
            },
        );
        for i in 0..4 {
            let mut prompt = shared.to_vec();
            prompt.extend_from_slice(&tokens[24 + i..24 + i + 4]);
            engine.submit_tokens(prompt, 6, 1).unwrap();
        }
        let out = engine.run().unwrap();
        (
            out.iter().map(|c| c.tokens.clone()).collect::<Vec<_>>(),
            engine.stats().prompt_tokens_prefilled,
            engine.prefix_stats().hits,
        )
    };

    let (cold_text, cold_prefill, cold_hits) = run(false);
    let (warm_text, warm_prefill, warm_hits) = run(true);

    assert_eq!(cold_hits, 0);
    assert!(warm_hits > 0, "the warm run never hit the cache");
    assert!(
        warm_prefill < cold_prefill,
        "reuse should cut prefill: {warm_prefill} vs {cold_prefill}"
    );
    // The whole point: less work, identical answers.
    assert_eq!(
        cold_text, warm_text,
        "prefix reuse changed what the model generated"
    );
}

#[test]
fn test_forked_samples_share_the_prompt_but_diverge() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(weights, config, engine_config());

    // 20 tokens leaves a partial trailing block that all four samples write to,
    // which is exactly the block copy-on-write has to split.
    engine.submit_tokens(tokens[..20].to_vec(), 10, 4).unwrap();
    let out = engine.run().unwrap();

    assert_eq!(out.len(), 4, "one completion per sample");
    assert!(
        out.iter().all(|c| c.request_id == out[0].request_id),
        "all samples belong to one request"
    );
    // The prompt is prefilled once no matter how many samples are drawn.
    assert_eq!(engine.stats().prompt_tokens_prefilled, 20);
    assert!(
        engine.cow_copies() > 0,
        "the shared partial block should have split on first write"
    );

    let distinct: std::collections::HashSet<_> = out.iter().map(|c| c.tokens.clone()).collect();
    assert!(
        distinct.len() > 1,
        "forked samples should not all produce the same text"
    );
}

#[test]
fn test_forking_costs_far_less_memory_than_independent_requests() {
    let (config, tokens, bytes) = fixture();
    let prompt = tokens[..32].to_vec();

    let blocks_used = |samples: usize, fork: bool| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let cfg = EngineConfig {
            // No prefix cache, so this isolates forking from prefix reuse.
            enable_prefix_cache: false,
            ..engine_config()
        };
        let mut engine = Engine::new(weights, config.clone(), cfg);
        if fork {
            engine.submit_tokens(prompt.clone(), 4, samples).unwrap();
        } else {
            for _ in 0..samples {
                engine.submit_tokens(prompt.clone(), 4, 1).unwrap();
            }
        }
        engine.run().unwrap();
        engine.stats().prompt_tokens_prefilled
    };

    // Independent requests each prefill the whole prompt; forked samples do not.
    assert_eq!(blocks_used(4, false), 4 * 32);
    assert_eq!(blocks_used(4, true), 32);
}

#[test]
fn test_queued_requests_wait_for_memory_then_run() {
    // Six requests against a cache that can hold two at a time: the scheduler
    // has to admit what fits, retire it, and admit the rest.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 8, // 8 blocks x 8 tokens = 64 tokens of KV in total
            enable_prefix_cache: false,
            ..engine_config()
        },
    );

    for i in 0..6 {
        // 24 tokens of prompt plus 4 generated needs 4 blocks each.
        let mut prompt = tokens[..20].to_vec();
        prompt.extend_from_slice(&tokens[20 + i..24 + i]);
        engine.submit_tokens(prompt, 4, 1).unwrap();
    }

    let out = engine.run().unwrap();
    assert_eq!(out.len(), 6, "every request should eventually complete");
    assert!(
        out.iter().all(|c| c.tokens.len() == 4),
        "requests should not be truncated by memory pressure: {:?}",
        out.iter().map(|c| c.tokens.len()).collect::<Vec<_>>()
    );
    assert!(
        engine.stats().steps > 1,
        "admission should have been staged"
    );
}

#[test]
fn test_a_prompt_that_cannot_fit_is_refused_at_submission_not_hung() {
    // The review's HTTP stall: a queued request that can never be admitted
    // used to sit at the head of the queue forever, and `step()` never admitted
    // anything behind it. Submission refuses it instead, with the reason.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 2, // 16 tokens of KV total
            ..engine_config()
        },
    );
    let err = engine
        .submit_tokens(tokens[..40].to_vec(), 4, 1)
        .unwrap_err();
    assert!(
        matches!(err, SubmitError::DoesNotFit { .. }),
        "expected DoesNotFit, got {err:?}"
    );
    assert!(
        err.to_string().contains("does not fit"),
        "should say why, got: {err}"
    );
    assert_eq!(
        engine.queue_depth(),
        (0, 0),
        "a refused request must not be queued"
    );

    // A prompt that exactly fills the pool is admissible for one token only:
    // there is no room for a decode step, so asking for more is refused too.
    assert!(engine.submit_tokens(tokens[..16].to_vec(), 1, 1).is_ok());
    assert!(matches!(
        engine.submit_tokens(tokens[..16].to_vec(), 2, 1),
        Err(SubmitError::DoesNotFit { .. })
    ));
    let out = engine.run().unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].tokens.len(), 1);
    assert_eq!(out[0].finish_reason, FinishReason::Length);
}

#[test]
fn test_a_valid_request_still_completes_behind_a_refused_one_under_step() {
    // Exactly the server's loop: `step()` driven by hand, no `run()`. Two
    // eight-token blocks; a 17-token prompt can never fit, a 4-token one can.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 2,
            block_size: 8,
            enable_prefix_cache: false,
            ..engine_config()
        },
    );
    assert!(engine.submit_tokens(tokens[..17].to_vec(), 4, 1).is_err());
    let follower = engine.submit_tokens(tokens[..4].to_vec(), 4, 1).unwrap();

    let mut completions = Vec::new();
    for _ in 0..20 {
        engine.step().unwrap();
        completions.extend(engine.take_completed());
        if !completions.is_empty() {
            break;
        }
    }
    assert_eq!(completions.len(), 1, "the valid follower never completed");
    assert_eq!(completions[0].request_id, follower);
    assert_eq!(completions[0].tokens.len(), 4);
    assert_eq!(engine.queue_depth(), (0, 0));
}

#[test]
fn test_temperature_sampling_respects_its_seed() {
    let (config, tokens, bytes) = fixture();
    let run = |seed: u64| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(
            weights,
            config.clone(),
            EngineConfig {
                temperature: 1.0,
                seed,
                ..engine_config()
            },
        );
        engine.submit_tokens(tokens[..16].to_vec(), 20, 1).unwrap();
        engine.run().unwrap()[0].tokens.clone()
    };
    assert_eq!(run(7), run(7), "the same seed must replay exactly");
    assert_ne!(run(7), run(8), "different seeds should diverge");
}

#[test]
fn test_int8_weights_shrink_the_model_and_track_f32_output() {
    // Quantization is only useful if it is nearly free in quality. Compare an
    // int8 load against the f32 load on the same prompt: memory must drop ~4x
    // and the logits must stay close enough that greedy decoding agrees.
    use paged_infer::model::Quantization;

    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();

    let f32_weights = loader.load_weights(&config).unwrap();
    let int8_config = LlamaConfig {
        quantization: Quantization::Int8,
        ..config.clone()
    };
    let int8_weights = loader.load_weights(&int8_config).unwrap();

    // The LM head stays f32 on purpose, so the ratio is below a clean 4x.
    let f32_bytes = f32_weights.weight_bytes();
    let int8_bytes = int8_weights.weight_bytes();
    let ratio = f32_bytes as f64 / int8_bytes as f64;
    println!("weights: {f32_bytes} B f32 -> {int8_bytes} B int8 ({ratio:.2}x smaller)");
    assert!(ratio > 3.0, "expected a large reduction, got {ratio:.2}x");

    let generate = |weights, cfg: LlamaConfig| {
        let mut engine = Engine::new(weights, cfg, engine_config());
        engine.submit_tokens(tokens[..16].to_vec(), 8, 1).unwrap();
        engine.run().unwrap()[0].tokens.clone()
    };
    let baseline = generate(f32_weights, config);
    let quantized = generate(int8_weights, int8_config);

    let agree = baseline
        .iter()
        .zip(quantized.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "greedy tokens matching before divergence: {agree}/{}",
        baseline.len()
    );
    assert!(
        agree >= baseline.len() / 2,
        "int8 diverged too early: {baseline:?} vs {quantized:?}"
    );
}

// ── memory safety and request progress ───────────────────────────────────────

/// Every completion keyed by (request id, sequence id), so two runs of the same
/// submission order can be compared sequence by sequence.
fn by_sequence(
    out: &[paged_infer::engine::Completion],
) -> HashMap<(usize, usize), (Vec<u32>, FinishReason)> {
    out.iter()
        .map(|c| {
            (
                (c.request_id, c.sequence_id),
                (c.tokens.clone(), c.finish_reason),
            )
        })
        .collect()
}

#[test]
fn test_forked_siblings_under_memory_pressure_never_share_a_written_block() {
    // The corruption the review found: with the pool exhausted, a failed
    // copy-on-write used to look exactly like "already private", and several
    // siblings then wrote into one physical block. The check here is the
    // strongest one available — a pressured run must produce, for every
    // sequence, exactly the tokens an unpressured run produces — and the
    // counters prove the pressure was real.
    let (config, tokens, bytes) = fixture();
    let run = |total_blocks: usize| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(
            weights,
            config.clone(),
            EngineConfig {
                total_blocks,
                block_size: 4,
                enable_prefix_cache: false,
                ..engine_config()
            },
        );
        // A single sequence that keeps growing, then a fork whose siblings
        // share a partial block they both have to write.
        engine.submit_tokens(tokens[..3].to_vec(), 10, 1).unwrap();
        engine.submit_tokens(tokens[3..9].to_vec(), 8, 2).unwrap();
        let out = engine.run().unwrap();
        assert_eq!(
            engine.available_blocks(),
            engine.total_blocks(),
            "every block must come back once the run is over"
        );
        (
            by_sequence(&out),
            engine.stats().clone(),
            engine.cow_copies(),
        )
    };

    let (roomy, roomy_stats, _) = run(64);
    let (tight, tight_stats, tight_cow) = run(4);

    assert_eq!(
        roomy_stats.deferred_steps, 0,
        "64 blocks should never stall"
    );
    assert!(
        tight_stats.deferred_steps > 0 || tight_stats.preemptions > 0,
        "four blocks should have produced memory pressure: {tight_stats:?}"
    );
    assert!(
        tight_cow > 0,
        "the siblings should have split their shared block"
    );
    assert_eq!(roomy.len(), 3);
    assert_eq!(
        tight, roomy,
        "memory pressure changed what a sequence generated"
    );
    for (tokens, reason) in tight.values() {
        assert_eq!(*reason, FinishReason::Length);
        assert!(tokens.len() == 10 || tokens.len() == 8);
    }
}

#[test]
fn test_two_requests_that_fit_one_at_a_time_complete_through_queuing() {
    // Two eight-token blocks, two eight-token prompts, four tokens each. Each
    // finishes alone; admitted together they used to strand each other with one
    // token and OutOfMemory, because admission counted only the prompt.
    let (config, tokens, bytes) = fixture();
    let make = || {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        Engine::new(
            weights,
            config.clone(),
            EngineConfig {
                total_blocks: 2,
                block_size: 8,
                enable_prefix_cache: false,
                ..engine_config()
            },
        )
    };
    let prompts = [tokens[..8].to_vec(), tokens[8..16].to_vec()];

    let mut serial = Vec::new();
    for p in &prompts {
        let mut engine = make();
        engine.submit_tokens(p.clone(), 4, 1).unwrap();
        serial.push(engine.run().unwrap().remove(0).tokens);
    }

    let mut engine = make();
    for p in &prompts {
        engine.submit_tokens(p.clone(), 4, 1).unwrap();
    }
    let mut concurrent = engine.run().unwrap();
    concurrent.sort_by_key(|c| c.request_id);

    assert_eq!(concurrent.len(), 2);
    for (c, expected) in concurrent.iter().zip(&serial) {
        assert_eq!(c.finish_reason, FinishReason::Length, "{c:?}");
        assert_eq!(
            &c.tokens, expected,
            "request {} was cut short",
            c.request_id
        );
    }
    assert!(
        engine.stats().steps > 4,
        "the second request should have waited"
    );
    assert_eq!(engine.available_blocks(), engine.total_blocks());
}

#[test]
fn test_a_sequence_the_pool_can_never_hold_ends_with_out_of_memory_once() {
    // Eight tokens of prompt in a sixteen-token pool asking for a hundred: the
    // pool fills, nothing else can free it, and the sequence cannot be
    // recomputed into an empty pool either. It ends with OutOfMemory, keeps
    // what it produced, and gives every block back.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 2,
            block_size: 8,
            enable_prefix_cache: false,
            stream_tokens: true,
            ..engine_config()
        },
    );
    engine.submit_tokens(tokens[..8].to_vec(), 100, 1).unwrap();

    let mut terminal = 0;
    let mut streamed = Vec::new();
    while engine.has_work() {
        engine.step().unwrap();
        for d in engine.take_deltas() {
            streamed.extend_from_slice(&d.tokens);
            terminal += usize::from(d.finish_reason.is_some());
            if let Some(r) = d.finish_reason {
                assert_eq!(r, FinishReason::OutOfMemory);
            }
        }
    }
    let out = engine.take_completed();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].finish_reason, FinishReason::OutOfMemory);
    // Positions 8..=15 fit: the token from prefill plus eight decode steps.
    assert_eq!(out[0].tokens.len(), 9, "{:?}", out[0].tokens);
    assert_eq!(streamed, out[0].tokens);
    assert_eq!(terminal, 1, "exactly one terminal delta");
    assert_eq!(engine.stats().preemptions, 1);
    assert_eq!(engine.available_blocks(), engine.total_blocks());
}

#[test]
fn test_explicitly_greedy_samples_are_identical_and_default_samples_are_not() {
    use paged_infer::engine::RequestOptions;
    let (config, tokens, bytes) = fixture();
    let run = |options: RequestOptions| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(weights, config.clone(), engine_config());
        engine
            .submit_tokens_with(tokens[..14].to_vec(), 8, 3, options)
            .unwrap();
        let out = engine.run().unwrap();
        assert_eq!(out.len(), 3);
        out.into_iter()
            .map(|c| c.tokens)
            .collect::<std::collections::HashSet<_>>()
            .len()
    };
    // Asked for greedy: three identical branches is exactly what was ordered.
    assert_eq!(
        run(RequestOptions {
            temperature: Some(0.0),
            ..RequestOptions::default()
        }),
        1
    );
    // Said nothing: the engine picks a temperature so the fork is useful.
    assert!(run(RequestOptions::default()) > 1);
}

#[test]
fn test_out_of_vocabulary_ids_are_refused_not_wrapped() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let vocab = config.vocab_size as u32;
    let mut engine = Engine::new(weights, config, engine_config());

    let mut bad = tokens[..6].to_vec();
    bad.push(vocab);
    assert_eq!(
        engine.submit_tokens(bad, 4, 1),
        Err(SubmitError::InvalidToken {
            index: 6,
            token: vocab,
            vocab_size: vocab as usize
        })
    );
    // `vocab + 1` used to alias token 1 through a modulo; it must not run.
    assert!(matches!(
        engine.submit_tokens(vec![vocab + 1], 4, 1),
        Err(SubmitError::InvalidToken { .. })
    ));
    assert_eq!(
        engine.submit_tokens(vec![], 4, 1),
        Err(SubmitError::EmptyPrompt)
    );
    assert_eq!(
        engine.submit_tokens(tokens[..4].to_vec(), 0, 1),
        Err(SubmitError::ZeroMaxTokens)
    );
    assert_eq!(engine.queue_depth(), (0, 0), "nothing invalid was queued");
}

#[test]
fn test_a_request_seed_replays_regardless_of_what_ran_before() {
    use paged_infer::engine::RequestOptions;
    let (config, tokens, bytes) = fixture();
    let seeded = RequestOptions {
        temperature: Some(1.0),
        seed: Some(4242),
        ..RequestOptions::default()
    };
    let run = |warm_up_traffic: usize| {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&config).unwrap();
        let mut engine = Engine::new(weights, config.clone(), engine_config());
        for i in 0..warm_up_traffic {
            engine
                .submit_tokens(tokens[i..i + 6].to_vec(), 3, 2)
                .unwrap();
        }
        let id = engine
            .submit_tokens_with(tokens[..10].to_vec(), 12, 1, seeded.clone())
            .unwrap();
        engine
            .run()
            .unwrap()
            .into_iter()
            .find(|c| c.request_id == id)
            .unwrap()
            .tokens
    };
    assert_eq!(
        run(0),
        run(3),
        "a client seed must not depend on sequence ids"
    );
}

#[test]
fn test_the_context_window_caps_generation_and_refuses_longer_prompts() {
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            max_context: Some(20),
            ..engine_config()
        },
    );
    assert!(matches!(
        engine.submit_tokens(tokens[..21].to_vec(), 4, 1),
        Err(SubmitError::ExceedsContext {
            prompt_tokens: 21,
            max_context: 20
        })
    ));
    engine.submit_tokens(tokens[..16].to_vec(), 10, 1).unwrap();
    let out = engine.run().unwrap();
    // Positions 0..=19 exist: 16 of prompt, then the prefill token and three
    // decode steps before the next token would need position 20.
    assert_eq!(out[0].tokens.len(), 5, "{:?}", out[0].tokens);
    assert_eq!(out[0].finish_reason, FinishReason::Length);
}

#[test]
fn test_time_to_first_token_includes_the_queue_wait() {
    // Six requests on a two-request pool: the later ones queue. Their
    // time-to-first-token is measured from submission, so it must cover the
    // wait, and the queue time is the part of it spent waiting.
    let (config, tokens, bytes) = fixture();
    let loader = ModelLoader::new(&bytes).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let mut engine = Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 8,
            enable_prefix_cache: false,
            ..engine_config()
        },
    );
    for i in 0..6 {
        let mut prompt = tokens[..20].to_vec();
        prompt.extend_from_slice(&tokens[20 + i..24 + i]);
        engine.submit_tokens(prompt, 4, 1).unwrap();
    }
    let mut out = engine.run().unwrap();
    out.sort_by_key(|c| c.request_id);
    for c in &out {
        assert!(
            c.time_to_first_token >= c.queue_time,
            "request {}: ttft {:?} < queue {:?}",
            c.request_id,
            c.time_to_first_token,
            c.queue_time
        );
    }
    assert!(
        out.last().unwrap().queue_time > out[0].queue_time,
        "the last request queued behind the first: {:?} vs {:?}",
        out.last().unwrap().queue_time,
        out[0].queue_time
    );
}
