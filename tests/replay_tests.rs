//! Drive the real fixture engine through the reusable workload runner.
use std::time::Duration;

use paged_infer::engine::{Engine, EngineConfig, FinishReason};
use paged_infer::model::{LlamaConfig, ModelLoader};
use paged_infer::replay::{self, ReplayClock, ReplayEvent, ReplayLimits, RequestStatus, Workload};

fn engine(blocks: usize, drafts: usize) -> Engine<'static> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny_llama.safetensors"
    );
    let config = LlamaConfig::beside_checkpoint(path).unwrap();
    let loader = ModelLoader::new(include_bytes!("fixtures/tiny_llama.safetensors")).unwrap();
    Engine::new(
        loader.load_weights(&config).unwrap(),
        config,
        EngineConfig {
            total_blocks: blocks,
            block_size: 4,
            eos_token: u32::MAX,
            bos_token: None,
            enable_prefix_cache: false,
            draft_tokens: drafts,
            stream_tokens: true,
            ..EngineConfig::default()
        },
    )
}

fn submit(id: &str, at: u64, prompt: Vec<u32>, budget: usize, samples: usize) -> ReplayEvent {
    ReplayEvent::Submit {
        id: id.into(),
        at,
        tokens: prompt,
        max_tokens: budget,
        samples,
        seed: 42,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    }
}

fn workload(events: Vec<ReplayEvent>) -> Workload {
    Workload {
        version: 1,
        clock: ReplayClock::Steps,
        events,
    }
}

#[test]
fn seeded_forks_replay_after_reset_with_stable_sample_identity() {
    let mut event = submit("fork", 0, vec![2, 4, 6], 12, 3);
    if let ReplayEvent::Submit { temperature, .. } = &mut event {
        *temperature = 0.8;
    }
    let work = workload(vec![event, submit("later", 3, vec![7, 8, 9], 4, 1)]);
    let mut engine = engine(64, 0);
    engine.warm_up();
    let first = replay::run(&mut engine, &work, &ReplayLimits::default()).unwrap();
    engine.reset();
    let second = replay::run(&mut engine, &work, &ReplayLimits::default()).unwrap();
    replay::compare_outputs(&first, &second).unwrap();
    assert_eq!(first.summary.successful_requests, 2);
    assert_eq!(first.summary.completed_sequences, 4);
    assert_eq!(first.summary.useful_output_tokens, 40);
    assert_eq!(first.summary.observed_inter_token_ms.count, 36);
    for request in &first.requests {
        assert_eq!(request.dispatch_lag_ms, None);
        for (index, seq) in request.sequences.iter().enumerate() {
            assert_eq!(seq.sample_index, index);
            assert_eq!(
                seq.tokens.len(),
                seq.deliveries.iter().map(|d| d.tokens).sum::<usize>()
            );
            assert!(seq.ttft_ms.unwrap() <= seq.end_to_end_ms);
            assert!(seq.engine_queue_ms <= seq.engine_ttft_ms.unwrap());
        }
    }
    assert_eq!(first.memory.final_allocated_blocks, 0);
    assert!(first.memory.peak_allocated_blocks > 0);
}

#[test]
fn queued_and_active_cancellation_are_distinct_and_do_not_inflate_goodput() {
    let work = workload(vec![
        submit("never_admitted", 0, vec![1, 2, 3], 20, 2),
        ReplayEvent::Cancel {
            at: 0,
            id: "never_admitted".into(),
        },
        submit("active", 0, vec![3, 4, 5], 20, 2),
        submit("healthy", 0, vec![4, 5, 6], 6, 1),
        ReplayEvent::Cancel {
            at: 1,
            id: "active".into(),
        },
        // Cancellation after completion is observed but does not rewrite its outcome.
        ReplayEvent::Cancel {
            at: 30,
            id: "healthy".into(),
        },
    ]);
    let report = replay::run(&mut engine(64, 0), &work, &ReplayLimits::default()).unwrap();
    assert_eq!(
        report.requests[0].status,
        RequestStatus::CancelledBeforeAdmission
    );
    assert!(report.requests[0].sequences.is_empty());
    assert!(report.requests[1]
        .sequences
        .iter()
        .all(|s| s.finish_reason == FinishReason::Cancelled));
    assert!(report.requests[1]
        .sequences
        .iter()
        .all(|s| !s.tokens.is_empty() && s.tokens.len() < 20));
    assert_eq!(
        report.requests[2].sequences[0].finish_reason,
        FinishReason::Length
    );
    assert_eq!(report.summary.cancelled_requests, 2);
    assert_eq!(report.summary.cancelled_sequences, 2);
    assert_eq!(report.summary.successful_requests, 1);
    assert_eq!(report.summary.useful_output_tokens, 6);
    assert!(report.summary.output_tokens > 6);
    assert_eq!(report.memory.final_allocated_blocks, 0);
}

#[test]
fn pressure_preserves_full_outputs_and_records_preemption_or_deferral() {
    let work = workload(vec![
        submit("growing", 0, vec![1, 2, 3], 10, 1),
        submit("fork", 0, vec![4, 5, 6, 7, 8, 9], 8, 2),
    ]);
    let roomy = replay::run(&mut engine(64, 0), &work, &ReplayLimits::default()).unwrap();
    let tight = replay::run(&mut engine(4, 0), &work, &ReplayLimits::default()).unwrap();
    replay::compare_outputs(&roomy, &tight).unwrap();
    assert!(tight.engine.preemptions > 0 || tight.engine.deferred_steps > 0);
    assert!(tight.engine.cow_copies > 0);
    assert_eq!(tight.memory.peak_allocated_blocks, 4);
    assert_eq!(tight.memory.final_allocated_blocks, 0);
}

#[test]
fn rejections_and_oom_are_explicit_and_followers_complete() {
    let work = workload(vec![
        submit("too_big", 0, vec![1; 9], 1, 1),
        submit("invalid_id", 0, vec![u32::MAX], 1, 1),
        submit("will_oom", 0, vec![1; 4], 100, 1),
        submit("healthy", 0, vec![2, 3], 2, 1),
    ]);
    let report = replay::run(&mut engine(2, 0), &work, &ReplayLimits::default()).unwrap();
    assert_eq!(report.summary.rejected_requests, 2);
    assert_eq!(report.summary.oom_sequences, 1);
    assert_eq!(report.summary.failed_requests, 1);
    assert_eq!(report.summary.completed_sequences, 1);
    assert_eq!(report.summary.useful_output_tokens, 2);
    assert_eq!(
        report.requests[2].sequences[0].finish_reason,
        FinishReason::OutOfMemory
    );
    assert_eq!(report.memory.final_allocated_blocks, 0);
}

#[test]
fn wall_clock_measures_from_deadline_and_never_submits_before_arrival() {
    let mut work = workload(vec![
        submit("long", 0, vec![1; 128], 3, 1),
        submit("arrival", 1, vec![2, 3, 4], 3, 1),
        submit("idle_arrival", 15, vec![3, 4, 5], 1, 1),
    ]);
    work.clock = ReplayClock::Milliseconds;
    let report = replay::run(&mut engine(128, 0), &work, &ReplayLimits::default()).unwrap();
    assert!(report.elapsed_ms >= 15.0);
    for req in &report.requests {
        let scheduled_ms = req.scheduled_at as f64;
        assert!(req.submitted_at_ms >= scheduled_ms);
        assert!((req.dispatch_lag_ms.unwrap() - (req.submitted_at_ms - scheduled_ms)).abs() < 1e-8);
        for seq in &req.sequences {
            assert!((seq.ttft_ms.unwrap() - (seq.deliveries[0].at_ms - scheduled_ms)).abs() < 1e-8);
            assert!((seq.end_to_end_ms - (seq.finished_at_ms - scheduled_ms)).abs() < 1e-8);
        }
    }
}

#[test]
fn speculative_delivery_batches_keep_observed_zero_gaps_and_exact_tokens() {
    let work = workload(vec![submit("repeat", 0, vec![1; 32], 24, 1)]);
    let base = replay::run(&mut engine(64, 0), &work, &ReplayLimits::default()).unwrap();
    let spec = replay::run(&mut engine(64, 4), &work, &ReplayLimits::default()).unwrap();
    replay::compare_outputs(&base, &spec).unwrap();
    let seq = &spec.requests[0].sequences[0];
    assert!(seq.deliveries.iter().any(|d| d.tokens > 1));
    assert_eq!(spec.summary.observed_inter_token_ms.count, 23);
    assert_eq!(
        spec.summary.inter_delivery_ms.count,
        seq.deliveries.len() - 1
    );
    // Mutating a middle token with identical lengths must fail the comparator.
    let mut corrupted = spec.clone();
    corrupted.requests[0].sequences[0].tokens[10] ^= 1;
    assert!(replay::compare_outputs(&base, &corrupted)
        .unwrap_err()
        .to_string()
        .contains("offset 10"));
    let mut wrong_reason = spec;
    wrong_reason.requests[0].sequences[0].finish_reason = FinishReason::Eos;
    assert!(replay::compare_outputs(&base, &wrong_reason).is_err());
}

#[test]
fn replay_limits_bound_execution_and_idle_waits() {
    let work = workload(vec![submit("long", 0, vec![1, 2, 3], 20, 1)]);
    let limits = ReplayLimits {
        max_steps: 1,
        timeout: Duration::from_secs(10),
    };
    assert!(replay::run(&mut engine(64, 0), &work, &limits)
        .unwrap_err()
        .to_string()
        .contains("engine steps"));
    let mut future = workload(vec![submit("future", 1000, vec![1], 1, 1)]);
    future.clock = ReplayClock::Milliseconds;
    let limits = ReplayLimits {
        max_steps: 100,
        timeout: Duration::from_millis(1),
    };
    assert!(replay::run(&mut engine(64, 0), &future, &limits)
        .unwrap_err()
        .to_string()
        .contains("timeout"));
}

#[test]
fn all_cancelled_has_no_fabricated_latency_or_sequence() {
    let work = workload(vec![
        submit("cancelled", 0, vec![1], 4, 3),
        ReplayEvent::Cancel {
            at: 0,
            id: "cancelled".into(),
        },
    ]);
    let report = replay::run(&mut engine(64, 0), &work, &ReplayLimits::default()).unwrap();
    assert_eq!(report.engine.steps, 0);
    assert_eq!(report.summary.ttft_ms.count, 0);
    assert_eq!(report.summary.ttft_ms.p50, None);
    assert_eq!(report.summary.completed_requests_per_second, 0.0);
    let encoded = serde_json::to_string(&report).unwrap();
    let decoded = serde_json::from_str(&encoded).unwrap();
    replay::compare_outputs(&report, &decoded).unwrap();
}
