//! Memory reports must retain transient allocations and exclude warmup history.

use paged_infer::engine::{Engine, EngineConfig, FinishReason};
use paged_infer::model::{LlamaConfig, ModelLoader};

fn engine(prefix_cache: bool, streaming: bool) -> Engine<'static> {
    let checkpoint = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tiny_llama.safetensors"
    );
    let config = LlamaConfig::beside_checkpoint(checkpoint).unwrap();
    let loader = ModelLoader::new(include_bytes!("fixtures/tiny_llama.safetensors")).unwrap();
    let weights = loader.load_weights(&config).unwrap();
    Engine::new(
        weights,
        config,
        EngineConfig {
            total_blocks: 8,
            block_size: 4,
            prefill_chunk_size: 8,
            eos_token: u32::MAX,
            bos_token: None,
            enable_prefix_cache: prefix_cache,
            stream_tokens: streaming,
            ..EngineConfig::default()
        },
    )
}

#[test]
fn peak_preserves_blocks_allocated_and_freed_within_one_step() {
    let mut engine = engine(false, true);
    assert!(engine.is_streaming());
    assert_eq!(engine.peak_allocated_blocks(), 0);
    engine.submit_tokens(vec![1; 12], 1, 1).unwrap();
    engine.step().unwrap();

    assert!(!engine.has_work());
    assert_eq!(engine.available_blocks(), engine.total_blocks());
    assert_eq!(engine.peak_allocated_blocks(), 3);
    let completed = engine.take_completed();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].finish_reason, FinishReason::Length);
    assert_eq!(engine.take_deltas().len(), 1);

    engine.reset();
    assert_eq!(engine.peak_allocated_blocks(), 0);
    engine.submit_tokens(vec![1; 3], 1, 1).unwrap();
    engine.step().unwrap();
    assert_eq!(engine.peak_allocated_blocks(), 1);
}

#[test]
fn warmup_and_reset_exclude_prior_allocations_and_cached_blocks() {
    let mut engine = engine(true, false);
    assert!(!engine.is_streaming());
    engine.warm_up();
    assert_eq!(engine.peak_allocated_blocks(), 0);
    assert_eq!(engine.available_blocks(), engine.total_blocks());

    engine.submit_tokens(vec![1; 12], 1, 1).unwrap();
    engine.step().unwrap();
    assert_eq!(engine.peak_allocated_blocks(), 3);
    assert_eq!(engine.total_blocks() - engine.available_blocks(), 3);

    engine.reset();
    assert_eq!(engine.peak_allocated_blocks(), 0);
    assert_eq!(engine.available_blocks(), engine.total_blocks());
    engine.warm_up();
    assert_eq!(engine.peak_allocated_blocks(), 0);
}

#[test]
fn shared_cached_blocks_count_once_in_peak() {
    let mut engine = engine(true, false);
    engine.submit_tokens(vec![1; 8], 1, 1).unwrap();
    engine.submit_tokens(vec![1; 8], 1, 1).unwrap();
    engine.step().unwrap();

    assert_eq!(engine.take_completed().len(), 2);
    assert_eq!(engine.peak_allocated_blocks(), 2);
    assert_eq!(engine.total_blocks() - engine.available_blocks(), 2);
    assert!(engine.prefix_stats().hits > 0);
}
