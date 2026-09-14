//! Scheduler slices must yield without losing KV ownership, identity or output.
use paged_infer::engine::{Completion, Engine, EngineConfig, FinishReason, RequestOptions};
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{BatchScratch, LlamaConfig, ModelLoader};
use paged_infer::replay::{self, ReplayLimits, RequestStatus, Workload};
use std::collections::BTreeMap;

fn config() -> LlamaConfig {
    LlamaConfig::from_hf_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/config.json"
    ))
    .unwrap()
}
fn engine(tokens_per_step: usize, blocks: usize, prefix: bool) -> Engine<'static> {
    let config = config();
    let loader = ModelLoader::new(include_bytes!("fixtures/tiny_llama.safetensors")).unwrap();
    Engine::new(
        loader.load_weights(&config).unwrap(),
        config,
        EngineConfig {
            total_blocks: blocks,
            block_size: 4,
            eos_token: u32::MAX,
            bos_token: None,
            stream_tokens: true,
            enable_prefix_cache: prefix,
            prefill_chunk_size: 8,
            max_prefill_tokens_per_step: tokens_per_step,
            ..EngineConfig::default()
        },
    )
}
fn prompt(len: usize) -> Vec<u32> {
    (0..len).map(|i| (i % 109 + 1) as u32).collect()
}
fn outputs(done: Vec<Completion>) -> BTreeMap<(usize, usize), (Vec<u32>, FinishReason)> {
    done.into_iter()
        .map(|c| ((c.request_id, c.sequence_id), (c.tokens, c.finish_reason)))
        .collect()
}

#[test]
fn long_prefill_yields_to_existing_decode_and_stays_within_its_allowance() {
    let mut e = engine(4, 128, false);
    let decoding = e.submit_tokens(prompt(3), 20, 1).unwrap();
    e.step().unwrap();
    assert_eq!(e.take_deltas()[0].tokens.len(), 1);
    let long = e.submit_tokens(prompt(20), 2, 1).unwrap();
    for step in 0..5 {
        e.step().unwrap();
        let deltas = e.take_deltas();
        assert_eq!(
            deltas[0].request_id, decoding,
            "decode must run before prompt work"
        );
        assert_eq!(deltas[0].tokens.len(), 1);
        assert_eq!(e.stats().last_prefill_tokens, 4);
        if step < 4 {
            assert!(deltas.iter().all(|d| d.request_id != long));
            assert_eq!(e.prefilling_requests(), 1);
            assert_eq!(e.pending_prefill_tokens(), 20 - (step + 1) * 4);
        } else {
            assert_eq!(deltas.iter().filter(|d| d.request_id == long).count(), 1);
            assert_eq!(e.prefilling_requests(), 0);
        }
    }
    e.run().unwrap();
    assert_eq!(e.available_blocks(), e.total_blocks());
}

#[test]
fn partial_cancellation_terminates_every_reserved_sample_and_never_publishes_kv() {
    let mut e = engine(3, 32, true);
    let id = e.submit_tokens(prompt(12), 6, 3).unwrap();
    e.step().unwrap();
    assert!(e.take_deltas().is_empty());
    assert_eq!(e.queue_depth(), (0, 1));
    assert_eq!(e.stats().prompt_tokens, 12);
    assert_eq!(e.stats().prompt_tokens_prefilled, 3);
    assert_eq!(e.stats().prompt_tokens_reused(), 0);
    assert_eq!(
        e.prefix_stats().inserts,
        0,
        "future hashed blocks must remain unpublished"
    );
    assert_eq!(e.cancel_request(id), 3);
    let deltas = e.take_deltas();
    let done = e.take_completed();
    assert_eq!(deltas.len(), 3);
    assert_eq!(done.len(), 3);
    assert!(deltas
        .iter()
        .all(|d| d.tokens.is_empty() && d.finish_reason == Some(FinishReason::Cancelled)));
    assert!(done
        .iter()
        .all(|c| c.tokens.is_empty() && c.finish_reason == FinishReason::Cancelled));
    assert_eq!(e.cancel_request(id), 0);
    assert!(e.take_deltas().is_empty());
    assert!(!e.has_work());
    assert_eq!(e.available_blocks(), e.total_blocks());
    assert_eq!(e.prefix_stats().inserts, 0);
    e.submit_tokens(prompt(12), 1, 1).unwrap();
    e.run().unwrap();
    assert_eq!(
        e.stats().prompt_tokens_reused(),
        0,
        "cancelled prompt must not seed a cache hit"
    );
    assert_eq!(e.stats().prompt_tokens_prefilled, 15);
    assert_eq!(e.prefix_stats().inserts, 3);
}

#[test]
fn seeded_outputs_match_across_ragged_budgets_prefix_hits_and_forks() {
    let run = |budget| {
        let mut e = engine(budget, 128, true);
        for len in [12, 15, 12, 27] {
            e.submit_tokens_with(
                prompt(len),
                9,
                2,
                RequestOptions {
                    temperature: Some(0.8),
                    seed: Some(991),
                    ..RequestOptions::default()
                },
            )
            .unwrap();
        }
        outputs(e.run().unwrap())
    };
    let reference = run(4096);
    for budget in [1, 3, 4, 7, 8, 11, 32] {
        assert_eq!(run(budget), reference, "prefill allowance {budget}");
    }
}

#[test]
fn unfinished_prompt_is_evicted_before_decode_and_keeps_default_sample_ids() {
    let run = |blocks| {
        let mut e = engine(1, blocks, false);
        e.submit_tokens(prompt(3), 28, 1).unwrap();
        for _ in 0..3 {
            e.step().unwrap();
        }
        e.submit_tokens(prompt(21), 6, 2).unwrap();
        let done = e.run().unwrap();
        (outputs(done), e.stats().clone(), e.available_blocks())
    };
    let (reference, _, _) = run(128);
    let (pressured, stats, free) = run(10);
    assert_eq!(
        pressured, reference,
        "eviction must preserve implicit RNG seeds and siblings"
    );
    assert!(
        stats.prefill_preemptions > 0,
        "fixture must actually evict partial prefill: {stats:?}"
    );
    assert_eq!(
        stats.preemptions, 0,
        "unfinished prompt should give memory back before the decoder is preempted"
    );
    assert!(stats.recomputed_tokens > 0);
    assert!(stats.prompt_tokens_prefilled > stats.prompt_tokens);
    assert_eq!(stats.prompt_tokens_reused(), 0);
    assert_eq!(free, 10);
}

#[test]
fn cancelling_an_evicted_partial_request_still_terminates_its_samples() {
    let mut e = engine(1, 10, false);
    e.submit_tokens(prompt(3), 28, 1).unwrap();
    for _ in 0..3 {
        e.step().unwrap();
    }
    let id = e.submit_tokens(prompt(21), 6, 2).unwrap();
    for _ in 0..50 {
        e.step().unwrap();
        e.take_deltas();
        if e.stats().prefill_preemptions > 0 {
            break;
        }
    }
    assert!(e.stats().prefill_preemptions > 0);
    assert_eq!(e.prefilling_requests(), 0);
    assert_eq!(e.cancel_request(id), 2);
    let done = e.take_completed();
    assert_eq!(done.iter().filter(|c| c.request_id == id).count(), 2);
    assert!(done
        .iter()
        .filter(|c| c.request_id == id)
        .all(|c| c.tokens.is_empty() && c.finish_reason == FinishReason::Cancelled));
    assert_eq!(e.take_deltas().len(), 2);
    e.run().unwrap();
    assert_eq!(e.available_blocks(), e.total_blocks());
}

#[test]
fn continuous_new_arrivals_cannot_overtake_the_oldest_partial_prompt() {
    let mut e = engine(4, 128, false);
    let oldest = e.submit_tokens(prompt(32), 1, 1).unwrap();
    for step in 0..8 {
        e.submit_tokens(vec![99], 1, 1).unwrap();
        e.step().unwrap();
        let deltas = e.take_deltas();
        if step < 7 {
            assert!(
                deltas.is_empty(),
                "new arrivals cannot consume the unfinished prompt's lane"
            );
        } else {
            assert_eq!(deltas.len(), 1);
            assert_eq!(deltas[0].request_id, oldest);
        }
        assert!(e.stats().last_prefill_tokens <= 4);
    }
    let done = e.run().unwrap();
    assert_eq!(done.len(), 9);
}

#[test]
fn cache_hit_replays_only_final_token_and_reset_clears_partial_state() {
    let mut e = engine(3, 32, true);
    e.submit_tokens(prompt(12), 1, 1).unwrap();
    let first = e.run().unwrap()[0].tokens.clone();
    e.submit_tokens(prompt(12), 1, 1).unwrap();
    e.step().unwrap();
    assert_eq!(e.stats().last_prefill_tokens, 1);
    assert_eq!(e.stats().prompt_tokens_reused(), 11);
    assert_eq!(e.take_completed()[0].tokens, first);
    e.submit_tokens(prompt(27), 4, 1).unwrap();
    e.step().unwrap();
    assert_eq!(e.prefilling_requests(), 1);
    e.reset();
    assert!(!e.has_work());
    assert_eq!(e.queue_depth(), (0, 0));
    assert_eq!(e.pending_prefill_tokens(), 0);
    assert_eq!(e.available_blocks(), e.total_blocks());
    assert_eq!(e.stats().prefill_chunks, 0);
    e.warm_up();
    assert_eq!(e.stats().last_prefill_tokens, 0);
    assert_eq!(e.peak_allocated_blocks(), 0);
}

#[test]
fn replay_partial_cancellation_has_empty_terminal_sequences_and_no_ttft() {
    let work: Workload = serde_json::from_value(serde_json::json!({
        "version":1,"clock":"steps","events":[
            {"type":"submit","at":0,"id":"partial","tokens":prompt(20),"max_tokens":5,"samples":2,"seed":42},
            {"type":"cancel","at":1,"id":"partial"}
        ]
    })).unwrap();
    let report = replay::run(&mut engine(3, 32, true), &work, &ReplayLimits::default()).unwrap();
    let request = &report.requests[0];
    assert_eq!(request.status, RequestStatus::Finished);
    assert_eq!(request.sequences.len(), 2);
    assert!(request
        .sequences
        .iter()
        .all(|s| s.tokens.is_empty() && s.ttft_ms.is_none() && s.engine_ttft_ms.is_none()));
    assert_eq!(report.summary.ttft_ms.count, 0);
    assert_eq!(report.summary.cancelled_sequences, 2);
    assert_eq!(report.summary.useful_output_tokens, 0);
    assert_eq!(report.memory.final_allocated_blocks, 0);
}

#[test]
fn model_ranges_need_only_kv_and_skip_intermediate_vocabulary_projection() {
    let config = config();
    let loader = ModelLoader::new(include_bytes!("fixtures/tiny_llama.safetensors")).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    let tokens = prompt(23);
    let mut allocator = BlockAllocator::new(8, 4);
    let mut table = BlockTable::new();
    for _ in 0..6 {
        table.append_block(allocator.allocate().unwrap());
    }
    let layout = config.kv_layout(8, 4);
    let mut ref_cache = vec![0.0; layout.total_floats()];
    let mut reference = BatchScratch::new(&config, 8);
    weights.prefill_batched(
        &tokens,
        0,
        &config,
        &table,
        &mut ref_cache,
        4,
        8,
        &mut reference,
    );
    let mut cache = vec![0.0; layout.total_floats()];
    let mut start = 0;
    for end in [3, 8, 9, 16, 23] {
        // New scratch per slice proves no hidden activation survives a yield.
        let mut scratch = BatchScratch::new(&config, 8);
        scratch.logits_for_mut(0, config.vocab_size).fill(123.0);
        weights.prefill_range(
            &tokens[start..end],
            start,
            &config,
            &table,
            &mut cache,
            4,
            8,
            &mut scratch,
            end == 23,
        );
        if end < 23 {
            assert!(scratch
                .logits_for(0, config.vocab_size)
                .iter()
                .all(|&v| v == 123.0));
        } else {
            for (&actual, &expected) in scratch
                .logits_for(0, config.vocab_size)
                .iter()
                .zip(reference.logits_for(0, config.vocab_size))
            {
                assert!(actual.is_finite() && (actual - expected).abs() < 1e-5);
            }
        }
        start = end;
    }
}

#[test]
fn preempted_decoder_resumes_in_slices_with_its_sampler_and_can_be_cancelled() {
    let run = |blocks| {
        let mut e = engine(1, blocks, false);
        e.submit_tokens_with(
            prompt(3),
            10,
            2,
            RequestOptions {
                temperature: Some(0.8),
                seed: Some(772),
                ..RequestOptions::default()
            },
        )
        .unwrap();
        let completed = e.run().unwrap();
        (outputs(completed), e.stats().clone())
    };
    let (reference, _) = run(64);
    let (tight, stats) = run(3);
    assert_eq!(tight, reference, "chunked resumption must retain RNG state");
    assert!(stats.preemptions > 0 && stats.recomputed_tokens > 0);

    let mut e = engine(1, 3, false);
    let request = e
        .submit_tokens_with(
            prompt(3),
            10,
            2,
            RequestOptions {
                temperature: Some(0.8),
                seed: Some(772),
                ..RequestOptions::default()
            },
        )
        .unwrap();
    let mut streamed: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut terminals: BTreeMap<usize, Vec<FinishReason>> = BTreeMap::new();
    let mut reached_resumption = false;
    for _ in 0..100 {
        e.step().unwrap();
        for d in e.take_deltas() {
            streamed.entry(d.sequence_id).or_default().extend(d.tokens);
            if let Some(reason) = d.finish_reason {
                terminals.entry(d.sequence_id).or_default().push(reason);
            }
        }
        if e.stats().preemptions > 0
            && e.stats().recomputed_tokens > 0
            && e.prefilling_requests() == 1
        {
            reached_resumption = true;
            break;
        }
    }
    assert!(
        reached_resumption,
        "fixture must pause during a decoder's chunked resumption"
    );
    assert_eq!(
        e.cancel_request(request),
        1,
        "one sibling already completed"
    );
    for d in e.take_deltas() {
        assert!(d.tokens.is_empty());
        terminals
            .entry(d.sequence_id)
            .or_default()
            .push(d.finish_reason.unwrap());
    }
    let done = e.take_completed();
    assert_eq!(done.len(), 2);
    assert_eq!(
        done.iter()
            .filter(|c| c.finish_reason == FinishReason::Cancelled)
            .count(),
        1
    );
    for completion in done {
        assert_eq!(completion.tokens, streamed[&completion.sequence_id]);
        assert!(!completion.tokens.is_empty());
        assert_eq!(
            terminals[&completion.sequence_id],
            vec![completion.finish_reason]
        );
    }
    assert!(!e.has_work());
    assert_eq!(e.available_blocks(), e.total_blocks());
    assert_eq!(e.cancel_request(request), 0);
    assert!(e.take_deltas().is_empty());
}
