//! The optional attention schedule must survive real scheduler ownership changes.
use paged_infer::engine::{Engine, EngineConfig, FinishReason, RequestOptions};
use paged_infer::model::{LlamaConfig, ModelLoader};
use paged_infer::replay::{self, ReplayClock, ReplayEvent, ReplayLimits, Workload};

fn engine(enabled: bool, blocks: usize, draft_tokens: usize) -> Engine<'static> {
    let config = LlamaConfig::from_hf_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/config.json"
    ))
    .unwrap();
    let loader = ModelLoader::new(include_bytes!("fixtures/tiny_llama.safetensors")).unwrap();
    Engine::new(
        loader.load_weights(&config).unwrap(),
        config,
        EngineConfig {
            total_blocks: blocks,
            block_size: 4,
            max_batch_size: 8,
            prefill_chunk_size: 8,
            max_prefill_tokens_per_step: 32,
            eos_token: u32::MAX,
            bos_token: None,
            stream_tokens: true,
            shared_prefix_attention: enabled,
            draft_tokens,
            ..EngineConfig::default()
        },
    )
}

#[test]
fn replay_parity_covers_prefix_forks_cancellation_and_pressure() {
    let mut used_shared = false;
    for (trace, blocks) in [
        (include_str!("../workloads/shared-prefix.json"), 64),
        (include_str!("../workloads/forks.json"), 64),
        (include_str!("../workloads/cancellation.json"), 64),
        (include_str!("../workloads/pressure.json"), 6),
    ] {
        let workload: Workload = serde_json::from_str(trace).unwrap();
        let run = |enabled| {
            let mut e = engine(enabled, blocks, 0);
            let r = replay::run(&mut e, &workload, &ReplayLimits::default()).unwrap();
            assert!(!e.has_work());
            assert_eq!(e.queue_depth(), (0, 0));
            r
        };
        let baseline = run(false);
        let candidate = run(true);
        replay::compare_outputs(&baseline, &candidate).unwrap();
        assert_eq!(baseline.engine.shared_attention_layer_calls, 0);
        assert_eq!(baseline.memory.shared_attention_scratch_bytes, 0);
        assert_eq!(
            baseline.memory.final_allocated_blocks,
            candidate.memory.final_allocated_blocks
        );
        used_shared |= candidate.engine.shared_attention_layer_calls > 0;
    }
    assert!(used_shared, "parity must exercise the optional kernel");
}

#[test]
fn prefill_stays_on_default_and_cancellation_reset_releases_mappings() {
    assert!(!EngineConfig::default().shared_prefix_attention);
    let mut e = engine(true, 64, 0);
    e.warm_up();
    assert_eq!(e.shared_attention_stats().layer_calls, 0);
    let id = e.submit_tokens((1..18).collect(), 12, 3).unwrap();
    e.step().unwrap();
    assert_eq!(
        e.shared_attention_stats().layer_calls,
        0,
        "prefill must not use shared decode"
    );
    e.step().unwrap();
    let stats = e.shared_attention_stats();
    assert!(stats.layer_calls > 0 && stats.query_tokens > 0 && stats.scratch_bytes > 0);
    e.take_deltas();
    assert_eq!(e.cancel_request(id), 3);
    assert_eq!(e.take_completed().len(), 3);
    assert_eq!(e.take_deltas().len(), 3);
    assert!(!e.has_work());
    e.reset();
    assert_eq!(e.available_blocks(), e.total_blocks());
    assert_eq!(e.shared_attention_stats().layer_calls, 0);
    assert_eq!(e.shared_attention_stats().query_tokens, 0);
    assert_eq!(
        e.shared_attention_stats().scratch_bytes,
        stats.scratch_bytes,
        "only scratch capacity survives reset"
    );
}

#[test]
fn seeded_samples_match_after_copy_on_write_and_preemption() {
    let run = |enabled, blocks| {
        let mut e = engine(enabled, blocks, 0);
        e.submit_tokens_with(
            (1..8).collect(),
            22,
            2,
            RequestOptions {
                temperature: Some(0.8),
                seed: Some(772),
                ..RequestOptions::default()
            },
        )
        .unwrap();
        let mut done = e.run().unwrap();
        done.sort_by_key(|c| c.sequence_id);
        assert_eq!(done.len(), 2);
        for completion in &done {
            assert_eq!(completion.tokens.len(), 22);
            assert_eq!(completion.finish_reason, FinishReason::Length);
        }
        assert!(!e.has_work());
        assert_eq!(e.queue_depth(), (0, 0));
        if blocks == 8 {
            assert_eq!(e.peak_allocated_blocks(), blocks);
        }
        let outputs: Vec<_> = done
            .into_iter()
            .map(|c| (c.sequence_id, c.tokens, c.finish_reason))
            .collect();
        let result = (
            outputs,
            e.stats().clone(),
            e.cow_copies(),
            e.shared_attention_stats(),
        );
        e.reset();
        assert_eq!(e.available_blocks(), e.total_blocks());
        result
    };
    let roomy = run(false, 64);
    let baseline = run(false, 8);
    let candidate = run(true, 8);
    assert_eq!(roomy.0, baseline.0);
    assert_eq!(roomy.0, candidate.0);
    assert_eq!(roomy.1.preemptions, 0);
    assert_eq!(roomy.1.recomputed_tokens, 0);
    for tight in [&baseline, &candidate] {
        assert!(
            tight.1.preemptions > 0 && tight.1.recomputed_tokens > 0 && tight.2 > 0,
            "must exercise preemption, recompute and COW"
        );
    }
    assert_eq!(baseline.3.layer_calls, 0);
    assert!(candidate.3.layer_calls > 0);
}

#[test]
fn millisecond_arrivals_preserve_outputs_and_deadline_accounting() {
    let submit = |id: &str, at, tokens, max_tokens, samples| ReplayEvent::Submit {
        id: id.into(),
        at,
        tokens,
        max_tokens,
        samples,
        seed: 42,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    let mut shared_prompt: Vec<u32> = (1..17).collect();
    shared_prompt.push(31);
    let workload = Workload {
        version: 1,
        clock: ReplayClock::Milliseconds,
        events: vec![
            submit("fork", 0, (1..18).collect(), 24, 2),
            submit("short", 1, vec![21, 22, 23], 6, 1),
            submit("shared", 2, shared_prompt, 10, 1),
        ],
    };
    let run = |enabled| {
        let mut e = engine(enabled, 64, 0);
        let report = replay::run(&mut e, &workload, &ReplayLimits::default()).unwrap();
        assert!(!e.has_work());
        assert_eq!(e.queue_depth(), (0, 0));
        assert_eq!(report.summary.successful_requests, 3);
        assert_eq!(report.summary.completed_sequences, 4);
        assert_eq!(report.summary.useful_output_tokens, 64);
        assert_eq!(report.summary.dispatch_lag_ms.count, 3);
        assert_eq!(report.summary.ttft_ms.count, 4);
        for request in &report.requests {
            let scheduled_ms = request.scheduled_at as f64;
            assert!(request.submitted_at_ms >= scheduled_ms);
            assert!(
                (request.dispatch_lag_ms.unwrap() - (request.submitted_at_ms - scheduled_ms)).abs()
                    < 1e-8
            );
            for sequence in &request.sequences {
                assert_eq!(sequence.finish_reason, FinishReason::Length);
                assert!(
                    (sequence.ttft_ms.unwrap() - (sequence.deliveries[0].at_ms - scheduled_ms))
                        .abs()
                        < 1e-8
                );
                assert!(
                    (sequence.end_to_end_ms - (sequence.finished_at_ms - scheduled_ms)).abs()
                        < 1e-8
                );
                assert_eq!(
                    sequence.tokens.len(),
                    sequence.deliveries.iter().map(|d| d.tokens).sum::<usize>()
                );
            }
        }
        e.reset();
        assert_eq!(e.available_blocks(), e.total_blocks());
        report
    };
    let baseline = run(false);
    let candidate = run(true);
    replay::compare_outputs(&baseline, &candidate).unwrap();
    assert_eq!(baseline.engine.shared_attention_layer_calls, 0);
    assert_eq!(baseline.memory.shared_attention_scratch_bytes, 0);
    assert!(candidate.engine.shared_attention_layer_calls > 0);
    assert!(candidate.memory.shared_attention_scratch_bytes > 0);
}

#[test]
fn speculative_verification_and_rollback_keep_greedy_outputs() {
    let run = |enabled, drafts| {
        let mut e = engine(enabled, 128, drafts);
        let prompt: Vec<u32> = [1, 3, 7, 9, 1, 3, 7, 9].repeat(4);
        for suffix in [vec![], vec![3], vec![7, 9]] {
            let mut p = prompt.clone();
            p.extend(suffix);
            e.submit_tokens_with(
                p,
                32,
                1,
                RequestOptions {
                    temperature: Some(0.0),
                    seed: Some(42),
                    ..RequestOptions::default()
                },
            )
            .unwrap();
        }
        let mut done = e.run().unwrap();
        done.sort_by_key(|c| c.request_id);
        let outputs: Vec<_> = done
            .into_iter()
            .map(|c| (c.request_id, c.tokens, c.finish_reason))
            .collect();
        (outputs, e.spec_stats(), e.shared_attention_stats())
    };
    let baseline = run(false, 0);
    for depth in [1, 4, 8] {
        let candidate = run(true, depth);
        assert_eq!(candidate.0, baseline.0, "draft depth {depth}");
        assert!(candidate.1.drafted > 0, "verification must receive drafts");
        assert!(
            candidate.1.accepted < candidate.1.drafted,
            "must reject some guesses"
        );
        assert!(candidate.2.layer_calls > 0);
    }
}
