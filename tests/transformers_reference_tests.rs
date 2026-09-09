//! Numerical compatibility with a pinned, independently executed Transformers.
//!
//! Regenerate the checked-in fixtures with `scripts/gen_llama3_reference.py`.
//! The tiny model is synthetic; the ignored test uses a full real checkpoint.
//! Both compare logits, not merely token choices, and use float32 inference on
//! the original BF16 weights so precision policy matches the Rust engine.

use std::path::{Path, PathBuf};

use paged_infer::math::{rope_inv_freq, rope_rotate, rope_table_from, RopeScaling};
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{BatchScratch, ForwardScratch, LlamaConfig, LlamaWeights, ModelLoader};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const BLOCK_SIZE: usize = 16;
// Observed max absolute error is 3.82e-6 over the tiny fixture's 40 positions.
// Allow CPU arithmetic variation while retaining a clear scaling regression.
const TINY_ABS_TOL: f32 = 5e-5;
const TINY_REL_TOL: f32 = 1e-5;
// Observed max error is 8.08e-5 for this checkpoint's short 60-token sequence.
// This bound validates that sequence; it is not a long-context accuracy claim.
const REAL_ABS_TOL: f32 = 5e-4;
const TRANSFORMERS_VERSION: &str = "4.52.4";
const LLAMA3_WEIGHT_SHA256: &str =
    "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a";
const REQUIRED_ROPE_POSITIONS: [usize; 10] = [0, 1, 7, 255, 8191, 8192, 8193, 32767, 65535, 131071];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/llama3_reference")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    serde_json::from_slice(&std::fs::read(path).unwrap_or_else(|err| {
        panic!(
            "read {}: {err}; regenerate the Transformers fixtures",
            path.display()
        )
    }))
    .unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

fn check_provenance(schema: usize, provenance: &Value) {
    assert_eq!(schema, 1, "unsupported Transformers fixture schema");
    assert_eq!(
        provenance["transformers_version"].as_str(),
        Some(TRANSFORMERS_VERSION),
        "fixture reference version changed; review numerical tolerances"
    );
}

#[derive(Deserialize)]
struct RopeFixture {
    schema_version: usize,
    provenance: Value,
    cases: Vec<RopeCase>,
}

#[derive(Deserialize)]
struct RopeCase {
    name: String,
    config: Value,
    q: Vec<f32>,
    k: Vec<f32>,
    inv_freq: Vec<f32>,
    positions: Vec<RopePosition>,
}

#[derive(Deserialize)]
struct RopePosition {
    position: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    q_rotated: Vec<f32>,
    k_rotated: Vec<f32>,
}

#[test]
fn checkpoint_rope_matches_transformers_frequencies_tables_and_rotations() {
    let fixture: RopeFixture = read_json(&fixture_dir().join("rope.json"));
    check_provenance(fixture.schema_version, &fixture.provenance);
    let mut covered_cases = Vec::new();
    for case in fixture.cases {
        let config = LlamaConfig::from_hf_json(&case.config).expect("reference HF config");
        covered_cases.push((config.head_dim(), config.rope_scaling));
        for required in REQUIRED_ROPE_POSITIONS {
            assert!(
                case.positions.iter().any(|row| row.position == required),
                "{} is missing required position {required}",
                case.name
            );
        }
        let half = config.head_dim() / 2;
        assert_eq!(case.inv_freq.len(), half);
        assert_eq!(case.q.len(), config.head_dim());
        assert_eq!(case.k.len(), config.head_dim());
        assert!(case.q.iter().chain(&case.k).all(|x| x.is_finite()));
        assert!(!case.positions.is_empty());
        let actual_freq = rope_inv_freq(config.head_dim(), config.rope_theta, config.rope_scaling);
        for (pair, (&actual, &expected)) in actual_freq.iter().zip(&case.inv_freq).enumerate() {
            // Relative error matters for the slow band: an absolute epsilon
            // alone can accept a frequency that has disappeared entirely.
            assert!(actual.is_finite() && expected.is_finite() && expected > 0.0);
            let relative = (f64::from(actual) / f64::from(expected) - 1.0).abs();
            assert!(
                relative <= 8.0 * f64::from(f32::EPSILON),
                "{} pair {pair}: frequency {actual:e} vs {expected:e}, relative error {relative:e}",
                case.name
            );
        }
        for reference in case.positions {
            assert_eq!(reference.cos.len(), half);
            assert_eq!(reference.sin.len(), half);
            assert_eq!(reference.q_rotated.len(), 2 * half);
            assert_eq!(reference.k_rotated.len(), 2 * half);
            let mut cos = vec![0.0; half];
            let mut sin = vec![0.0; half];
            rope_table_from(reference.position, &actual_freq, &mut cos, &mut sin);
            let mut table_bounds = Vec::with_capacity(half);
            let mut max_table_error = 0.0_f64;
            for pair in 0..half {
                // A one-ulp frequency difference becomes a measurable phase
                // difference at 128k context. sin/cos are 1-Lipschitz; account
                // for both accumulated frequency error and f32 phase rounding.
                // The independent frequency check above stays strict. The
                // 0.02 cap rejects excessive drift even at the longest fixture
                // position; this is not a flat 0.02 allowance at short context.
                let pos = reference.position as f32;
                let accumulated = f64::from(pos)
                    * (f64::from(actual_freq[pair]) - f64::from(case.inv_freq[pair])).abs();
                let rounded = (f64::from(pos * actual_freq[pair])
                    - f64::from(pos * case.inv_freq[pair]))
                .abs();
                let bound = accumulated.max(rounded) + 4e-7;
                assert!(
                    bound <= 0.02,
                    "{} position {} pair {pair}: excessive phase drift {bound:e}",
                    case.name,
                    reference.position
                );
                table_bounds.push(bound);
                for (kind, actual, expected) in [
                    ("cos", cos[pair], reference.cos[pair]),
                    ("sin", sin[pair], reference.sin[pair]),
                ] {
                    assert!(actual.is_finite() && expected.is_finite());
                    let error = (f64::from(actual) - f64::from(expected)).abs();
                    max_table_error = max_table_error.max(error);
                    assert!(error <= bound, "{} position {} pair {pair} {kind}: {actual:e} vs {expected:e}, error {error:e}, phase-aware bound {bound:e}", case.name, reference.position);
                }
            }
            for (kind, input, expected) in [
                ("q", &case.q, &reference.q_rotated),
                ("k", &case.k, &reference.k_rotated),
            ] {
                let mut actual = input.clone();
                rope_rotate(&mut actual, &cos, &sin, config.rope_style);
                for dim in 0..2 * half {
                    let pair = dim % half;
                    let magnitude =
                        f64::from(input[pair].abs()) + f64::from(input[pair + half].abs());
                    let bound =
                        magnitude * (table_bounds[pair] + 4.0 * f64::from(f32::EPSILON)) + 1e-7;
                    assert!(actual[dim].is_finite() && expected[dim].is_finite());
                    let error = (f64::from(actual[dim]) - f64::from(expected[dim])).abs();
                    assert!(
                        error <= bound,
                        "{} position {} {kind} dimension {dim}: error {error:e}, bound {bound:e}",
                        case.name,
                        reference.position
                    );
                }
            }
            println!(
                "{} position {}: max table error {max_table_error:e}",
                case.name, reference.position
            );
        }
    }
    for required in [
        (64, RopeScaling::None),
        (64, RopeScaling::Linear { factor: 4.0 }),
        (
            64,
            RopeScaling::Llama3 {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            },
        ),
        (
            128,
            RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            },
        ),
    ] {
        assert!(
            covered_cases.contains(&required),
            "missing head dimension / scaling coverage: {required:?}"
        );
    }
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: usize,
    provenance: Value,
    config: Value,
    tokens: Vec<u32>,
    logits_file: String,
    logit_rows: usize,
    negative_control_max_abs: f32,
    checkpoint_sha256: String,
    checkpoint_bytes: usize,
}

struct ForwardFixture {
    manifest: Manifest,
    config: LlamaConfig,
    logits: Vec<f32>,
}

impl ForwardFixture {
    fn read(dir: &Path, checkpoint: &[u8]) -> Self {
        let manifest: Manifest = read_json(&dir.join("manifest.json"));
        check_provenance(manifest.schema_version, &manifest.provenance);
        let config = LlamaConfig::from_hf_json(&manifest.config).expect("reference HF config");
        assert!(matches!(config.rope_scaling, RopeScaling::Llama3 { .. }));
        assert_eq!(
            manifest.checkpoint_bytes,
            checkpoint.len(),
            "checkpoint size changed"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(checkpoint)),
            manifest.checkpoint_sha256,
            "engine checkpoint bytes differ from the reference checkpoint"
        );
        assert!(
            manifest.tokens.len() > BLOCK_SIZE,
            "reference must cross a KV block boundary"
        );
        assert!(manifest
            .tokens
            .iter()
            .all(|&token| (token as usize) < config.vocab_size));
        assert_eq!(manifest.logit_rows, manifest.tokens.len());
        assert_eq!(manifest.logits_file, "logits.bin");
        let bytes = std::fs::read(dir.join(&manifest.logits_file)).expect("reference logits");
        let (words, remainder) = bytes.as_chunks::<4>();
        assert!(remainder.is_empty(), "truncated float32 logits");
        let logits: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
        assert_eq!(logits.len(), manifest.logit_rows * config.vocab_size);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "non-finite reference logits"
        );
        assert!(manifest.negative_control_max_abs.is_finite());
        Self {
            manifest,
            config,
            logits,
        }
    }

    fn row(&self, position: usize) -> &[f32] {
        let start = position * self.config.vocab_size;
        &self.logits[start..start + self.config.vocab_size]
    }
}

fn cache_and_table(config: &LlamaConfig, token_count: usize) -> (Vec<f32>, BlockTable) {
    let blocks = token_count.div_ceil(BLOCK_SIZE);
    let mut allocator = BlockAllocator::new(blocks, BLOCK_SIZE);
    let mut table = BlockTable::new();
    for _ in 0..blocks {
        table.append_block(allocator.allocate().expect("fixture KV block"));
    }
    (
        vec![0.0; config.kv_layout(blocks, BLOCK_SIZE).total_floats()],
        table,
    )
}

fn assert_logits(
    actual: &[f32],
    expected: &[f32],
    path: &str,
    position: usize,
    abs_tol: f32,
    rel_tol: f32,
) {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let mut max_abs = 0.0_f32;
    let mut max_index = 0;
    let mut violations = 0;
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "{path} position {position} logit {index}: non-finite {a} vs {e}"
        );
        let delta = (a - e).abs();
        if delta > max_abs {
            max_abs = delta;
            max_index = index;
        }
        if delta > abs_tol + rel_tol * e.abs() {
            violations += 1;
        }
    }
    println!("{path} position {position}: max|delta|={max_abs:e}, logit={max_index}, tolerance violations={violations}");
    assert_eq!(violations, 0, "{path} position {position}: max error {max_abs:e} at logit {max_index} ({} vs {}), atol={abs_tol:e}, rtol={rel_tol:e}", actual[max_index], expected[max_index]);
    let argmax = |values: &[f32]| {
        values.iter().enumerate().fold(
            0,
            |best, (index, value)| {
                if *value > values[best] {
                    index
                } else {
                    best
                }
            },
        )
    };
    assert_eq!(
        paged_infer::sampling::argmax(actual),
        argmax(expected),
        "{path} position {position}: greedy token mismatch"
    );
}

fn check_incremental(
    weights: &LlamaWeights<'_>,
    fixture: &ForwardFixture,
    abs_tol: f32,
    rel_tol: f32,
) {
    let (mut cache, table) = cache_and_table(&fixture.config, fixture.manifest.tokens.len());
    let mut scratch = ForwardScratch::new(&fixture.config);
    for (position, &token) in fixture.manifest.tokens.iter().enumerate() {
        weights.forward_into(
            token,
            position,
            &fixture.config,
            &table,
            &mut cache,
            BLOCK_SIZE,
            None,
            &mut scratch,
        );
        assert_logits(
            &scratch.logits,
            fixture.row(position),
            "incremental",
            position,
            abs_tol,
            rel_tol,
        );
    }
}

fn check_prefill(
    weights: &LlamaWeights<'_>,
    fixture: &ForwardFixture,
    chunk_size: usize,
    resumed: bool,
    abs_tol: f32,
    rel_tol: f32,
) {
    let tokens = &fixture.manifest.tokens;
    let (mut cache, table) = cache_and_table(&fixture.config, tokens.len());
    let mut scratch = BatchScratch::new(&fixture.config, chunk_size);
    let start = if resumed {
        weights.prefill_batched(
            &tokens[..BLOCK_SIZE],
            0,
            &fixture.config,
            &table,
            &mut cache,
            BLOCK_SIZE,
            chunk_size,
            &mut scratch,
        );
        assert_logits(
            scratch.logits_for(0, fixture.config.vocab_size),
            fixture.row(BLOCK_SIZE - 1),
            "prefill prefix",
            BLOCK_SIZE - 1,
            abs_tol,
            rel_tol,
        );
        BLOCK_SIZE
    } else {
        0
    };
    weights.prefill_batched(
        &tokens[start..],
        start,
        &fixture.config,
        &table,
        &mut cache,
        BLOCK_SIZE,
        chunk_size,
        &mut scratch,
    );
    let label = format!(
        "{} chunk={chunk_size}",
        if resumed {
            "resumed prefill"
        } else {
            "whole prefill"
        }
    );
    assert_logits(
        scratch.logits_for(0, fixture.config.vocab_size),
        fixture.row(tokens.len() - 1),
        &label,
        tokens.len() - 1,
        abs_tol,
        rel_tol,
    );
}

#[test]
fn tiny_llama3_forward_and_prefill_match_transformers() {
    let bytes = std::fs::read(
        fixture_dir()
            .parent()
            .unwrap()
            .join("tiny_llama.safetensors"),
    )
    .expect("tiny checkpoint");
    let fixture = ForwardFixture::read(&fixture_dir(), &bytes);
    let loader = ModelLoader::new(&bytes).expect("parse tiny checkpoint");
    let weights = loader
        .load_weights(&fixture.config)
        .expect("load tiny weights");
    check_incremental(&weights, &fixture, TINY_ABS_TOL, TINY_REL_TOL);
    for chunk_size in [7, 16] {
        for resumed in [false, true] {
            check_prefill(
                &weights,
                &fixture,
                chunk_size,
                resumed,
                TINY_ABS_TOL,
                TINY_REL_TOL,
            );
        }
    }
}

#[test]
fn tiny_reference_detects_disabling_llama3_scaling() {
    let bytes = std::fs::read(
        fixture_dir()
            .parent()
            .unwrap()
            .join("tiny_llama.safetensors"),
    )
    .expect("tiny checkpoint");
    let fixture = ForwardFixture::read(&fixture_dir(), &bytes);
    let mut unscaled = fixture.config.clone();
    unscaled.rope_scaling = RopeScaling::None;
    let loader = ModelLoader::new(&bytes).expect("parse tiny checkpoint");
    let weights = loader.load_weights(&unscaled).expect("load tiny weights");
    let (mut cache, table) = cache_and_table(&unscaled, fixture.manifest.tokens.len());
    let mut scratch = ForwardScratch::new(&unscaled);
    let mut rejected = false;
    let mut max_abs = 0.0_f32;
    for (position, &token) in fixture.manifest.tokens.iter().enumerate() {
        weights.forward_into(
            token,
            position,
            &unscaled,
            &table,
            &mut cache,
            BLOCK_SIZE,
            None,
            &mut scratch,
        );
        for (&actual, &expected) in scratch.logits.iter().zip(fixture.row(position)) {
            assert!(actual.is_finite());
            let delta = (actual - expected).abs();
            max_abs = max_abs.max(delta);
            rejected |= delta > TINY_ABS_TOL + TINY_REL_TOL * expected.abs();
        }
    }
    println!(
        "unscaled negative control: Rust max|delta|={max_abs:e}, Transformers max|delta|={:e}",
        fixture.manifest.negative_control_max_abs
    );
    assert!(
        fixture.manifest.negative_control_max_abs > TINY_ABS_TOL,
        "reference is insensitive to disabling scaling"
    );
    assert!(
        rejected,
        "the actual comparison tolerance must reject an unscaled model"
    );
}

fn assert_same_model(actual: &LlamaConfig, expected: &LlamaConfig) {
    // Compare semantic fields, allowing irrelevant HF metadata to differ.
    assert_eq!(actual.hidden_size, expected.hidden_size);
    assert_eq!(actual.num_hidden_layers, expected.num_hidden_layers);
    assert_eq!(actual.num_attention_heads, expected.num_attention_heads);
    assert_eq!(actual.num_key_value_heads, expected.num_key_value_heads);
    assert_eq!(actual.intermediate_size, expected.intermediate_size);
    assert_eq!(actual.vocab_size, expected.vocab_size);
    assert_eq!(actual.rms_norm_eps, expected.rms_norm_eps);
    assert_eq!(actual.rope_theta, expected.rope_theta);
    assert_eq!(actual.rope_scaling, expected.rope_scaling);
    assert_eq!(actual.rope_style, expected.rope_style);
    assert_eq!(
        actual.max_position_embeddings,
        expected.max_position_embeddings
    );
    assert_eq!(actual.bos_token_id, expected.bos_token_id);
    assert_eq!(actual.eos_token_ids, expected.eos_token_ids);
}

#[test]
#[ignore = "requires LLAMA3_CHECKPOINT and LLAMA3_REFERENCE_DIR; several GB of weights and CPU inference"]
fn real_llama3_checkpoint_matches_transformers() {
    let checkpoint = PathBuf::from(
        std::env::var_os("LLAMA3_CHECKPOINT").expect("set LLAMA3_CHECKPOINT to model.safetensors"),
    );
    let dir = PathBuf::from(
        std::env::var_os("LLAMA3_REFERENCE_DIR")
            .expect("set LLAMA3_REFERENCE_DIR to generated manifest.json/logits.bin directory"),
    );
    let file = std::fs::File::open(&checkpoint).expect("open real checkpoint");
    // SAFETY: This read-only test never modifies the checkpoint. The caller
    // must keep the file unchanged while the mapping is alive.
    let bytes = unsafe { memmap2::MmapOptions::new().map(&file) }.expect("map real checkpoint");
    let fixture = ForwardFixture::read(&dir, &bytes);
    assert_eq!(
        fixture.manifest.checkpoint_sha256, LLAMA3_WEIGHT_SHA256,
        "real-checkpoint validation requires the pinned official Llama 3.2 1B weights"
    );
    let checkpoint_config = LlamaConfig::from_hf_config(checkpoint.with_file_name("config.json"))
        .expect("real checkpoint config.json is required");
    assert_same_model(&checkpoint_config, &fixture.config);
    let loader = ModelLoader::new(&bytes).expect("parse real checkpoint");
    let weights = loader
        .load_weights(&checkpoint_config)
        .expect("load real checkpoint weights");
    check_incremental(&weights, &fixture, REAL_ABS_TOL, 0.0);
    check_prefill(&weights, &fixture, 16, false, REAL_ABS_TOL, 0.0);
    check_prefill(&weights, &fixture, 16, true, REAL_ABS_TOL, 0.0);
}
