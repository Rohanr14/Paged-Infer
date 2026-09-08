//! The loader has to be faithful to the checkpoint, or fail before anything
//! runs.
//!
//! Three things a checkpoint can be that the engine must not misread: a dtype
//! other than bf16 (an f16 or f32 file used to load "successfully" and run on
//! reinterpreted bytes), a shape that disagrees with `config.json`, and a
//! config feature the kernels do not implement (rotary scaling, biases, a
//! different architecture). Each is either handled exactly or refused with a
//! message that names the problem.

use std::collections::HashMap;
use std::path::PathBuf;

use paged_infer::math::{rope_inv_freq, rope_table, RopeScaling};
use paged_infer::memory::allocator::BlockAllocator;
use paged_infer::memory::block_table::BlockTable;
use paged_infer::model::{LlamaConfig, ModelLoader};
use paged_infer::tensor::DType;
use serde_json::{json, Value};

const BLOCK_SIZE: usize = 16;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

struct Fixture {
    config: LlamaConfig,
    tokens: Vec<u32>,
    golden: Vec<f32>,
    bytes: Vec<u8>,
}

fn fixture() -> Fixture {
    let dir = fixture_dir();
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
        ..LlamaConfig::default()
    };
    let tokens = kv["tokens"]
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    let golden = std::fs::read(dir.join("tiny_llama_golden.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Fixture {
        config,
        tokens,
        golden,
        bytes: std::fs::read(dir.join("tiny_llama.safetensors")).unwrap(),
    }
}

/// Re-encode every tensor of a bf16 safetensors file in another dtype. The
/// values are numerically the same (bf16 widens exactly into f32 and f64, and
/// into f16 except for tiny magnitudes below f16's normal range).
fn convert(bytes: &[u8], to: &str) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];

    let mut new_header = serde_json::Map::new();
    let mut blob = Vec::new();
    for (name, info) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        assert_eq!(info["dtype"], "BF16", "fixture is bf16");
        let offsets = info["data_offsets"].as_array().unwrap();
        let (s, e) = (
            offsets[0].as_u64().unwrap() as usize,
            offsets[1].as_u64().unwrap() as usize,
        );
        let values: Vec<f32> = data[s..e]
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        let out: Vec<u8> = match to {
            "F16" => values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect(),
            "F32" => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            "F64" => values
                .iter()
                .flat_map(|v| (*v as f64).to_le_bytes())
                .collect(),
            other => panic!("no converter for {other}"),
        };
        new_header.insert(
            name.clone(),
            json!({
                "dtype": to,
                "shape": info["shape"],
                "data_offsets": [blob.len(), blob.len() + out.len()],
            }),
        );
        blob.extend_from_slice(&out);
    }

    let mut header_bytes = serde_json::to_vec(&Value::Object(new_header)).unwrap();
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let mut file = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header_bytes);
    file.extend_from_slice(&blob);
    file
}

/// Logits at every position, decode-style, as the golden test computes them.
fn all_logits(bytes: &[u8], config: &LlamaConfig, tokens: &[u32]) -> Vec<f32> {
    let loader = ModelLoader::new(bytes).unwrap();
    let weights = loader.load_weights(config).unwrap();
    let total_blocks = 8;
    let mut allocator = BlockAllocator::new(total_blocks, BLOCK_SIZE);
    let mut kv_cache = vec![0.0f32; config.kv_layout(total_blocks, BLOCK_SIZE).total_floats()];
    let mut table = BlockTable::new();
    for _ in 0..tokens.len().div_ceil(BLOCK_SIZE) {
        table.append_block(allocator.allocate().unwrap());
    }
    let mut out = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        out.extend(weights.forward(t, pos, config, &table, &mut kv_cache, BLOCK_SIZE, None));
    }
    out
}

fn compare(actual: &[f32], expected: &[f32], vocab: usize) -> (f32, usize) {
    let mut max_abs = 0.0f32;
    let mut argmax_mismatch = 0;
    for (a, e) in actual.chunks_exact(vocab).zip(expected.chunks_exact(vocab)) {
        for (x, y) in a.iter().zip(e) {
            max_abs = max_abs.max((x - y).abs());
        }
        let pick = |s: &[f32]| {
            s.iter()
                .enumerate()
                .max_by(|(_, l), (_, r)| l.total_cmp(r))
                .map(|(i, _)| i)
                .unwrap()
        };
        if pick(a) != pick(e) {
            argmax_mismatch += 1;
        }
    }
    (max_abs, argmax_mismatch)
}

#[test]
fn test_bf16_f16_and_f32_checkpoints_all_match_the_reference() {
    // The same numbers in three encodings must give the same answer, to the
    // golden test's own tolerance. Before dtype was tracked, the f16 and f32
    // files loaded without complaint and produced noise.
    let fx = fixture();
    let vocab = fx.config.vocab_size;
    for (dtype, bytes) in [
        ("BF16", fx.bytes.clone()),
        ("F16", convert(&fx.bytes, "F16")),
        ("F32", convert(&fx.bytes, "F32")),
    ] {
        let loader = ModelLoader::new(&bytes).unwrap();
        let weights = loader.load_weights(&fx.config).unwrap();
        assert_eq!(
            format!("{:?}", weights.token_embeddings.dtype()),
            dtype,
            "the loader must record the checkpoint's dtype"
        );
        let logits = all_logits(&bytes, &fx.config, &fx.tokens);
        let (max_abs, mismatches) = compare(&logits, &fx.golden, vocab);
        println!("{dtype}: max|delta| vs golden = {max_abs:.6}, argmax mismatches = {mismatches}");
        assert_eq!(
            mismatches, 0,
            "{dtype}: greedy choice diverged from the reference"
        );
        assert!(max_abs < 5e-4, "{dtype}: max|delta| = {max_abs}");
    }

    // bf16 -> f32 is exact, so the f32 file must reproduce the bf16 run bit
    // for bit — the two decoders feed identical numbers into identical code.
    let bf16 = all_logits(&fx.bytes, &fx.config, &fx.tokens);
    let f32 = all_logits(&convert(&fx.bytes, "F32"), &fx.config, &fx.tokens);
    assert_eq!(
        bf16, f32,
        "an f32 copy of the bf16 weights must load identically"
    );
}

#[test]
fn test_norm_weights_are_decoded_in_their_own_dtype() {
    // The review's concrete symptom: a norm weight of 1.0234375 read as
    // 0.009277344 once the file was f16. Check the decoded value directly.
    let fx = fixture();
    let reference = ModelLoader::new(&fx.bytes)
        .unwrap()
        .load_weights(&fx.config)
        .unwrap()
        .final_norm
        .clone();
    for to in ["F16", "F32"] {
        let bytes = convert(&fx.bytes, to);
        let norm = ModelLoader::new(&bytes)
            .unwrap()
            .load_weights(&fx.config)
            .unwrap()
            .final_norm
            .clone();
        assert_eq!(norm, reference, "{to} norm weights decoded differently");
    }
    assert!(reference.iter().all(|w| (0.5..1.5).contains(w)));
}

#[test]
fn test_an_unsupported_dtype_is_refused_by_name() {
    let fx = fixture();
    let bytes = convert(&fx.bytes, "F64");
    let err = ModelLoader::new(&bytes)
        .unwrap()
        .load_weights(&fx.config)
        .expect_err("an f64 checkpoint must not load");
    let msg = format!("{err:#}");
    assert!(msg.contains("F64"), "should name the dtype: {msg}");
    assert!(
        msg.contains("model.embed_tokens.weight"),
        "should name the tensor: {msg}"
    );
}

#[test]
fn test_a_shape_that_disagrees_with_the_config_is_refused_before_running() {
    let fx = fixture();
    let loader = ModelLoader::new(&fx.bytes).unwrap();

    let narrow = LlamaConfig {
        hidden_size: 64,
        ..fx.config.clone()
    };
    let msg = format!("{:#}", loader.load_weights(&narrow).unwrap_err());
    assert!(msg.contains("model.embed_tokens.weight"), "{msg}");
    assert!(
        msg.contains("[112, 96]") && msg.contains("[112, 64]"),
        "{msg}"
    );

    let wrong_kv = LlamaConfig {
        num_key_value_heads: 4,
        ..fx.config.clone()
    };
    let msg = format!("{:#}", loader.load_weights(&wrong_kv).unwrap_err());
    assert!(msg.contains("k_proj"), "{msg}");

    let wrong_ffn = LlamaConfig {
        intermediate_size: 128,
        ..fx.config.clone()
    };
    let msg = format!("{:#}", loader.load_weights(&wrong_ffn).unwrap_err());
    assert!(msg.contains("gate_proj"), "{msg}");

    // Fewer layers than the file holds would load a truncated model that runs.
    let truncated = LlamaConfig {
        num_hidden_layers: 1,
        ..fx.config.clone()
    };
    let msg = format!("{:#}", loader.load_weights(&truncated).unwrap_err());
    assert!(msg.contains("more than the 1 layers"), "{msg}");

    // More layers than the file holds is a missing tensor.
    let extended = LlamaConfig {
        num_hidden_layers: 3,
        ..fx.config.clone()
    };
    let msg = format!("{:#}", loader.load_weights(&extended).unwrap_err());
    assert!(msg.contains("model.layers.2"), "{msg}");
}

#[test]
fn test_an_unrunnable_shape_is_refused_by_validation() {
    let fx = fixture();
    let loader = ModelLoader::new(&fx.bytes).unwrap();
    let bad = LlamaConfig {
        num_attention_heads: 5,
        ..fx.config.clone()
    };
    let msg = format!("{:#}", loader.load_weights(&bad).unwrap_err());
    assert!(msg.contains("not a multiple"), "{msg}");
    let bad = LlamaConfig {
        num_key_value_heads: 3,
        ..fx.config.clone()
    };
    assert!(loader.load_weights(&bad).is_err());
    let bad = LlamaConfig {
        attention_window: Some(0),
        ..fx.config.clone()
    };
    assert!(loader.load_weights(&bad).is_err());
}

// ── config.json ──────────────────────────────────────────────────────────────

fn parse(json: Value) -> anyhow::Result<LlamaConfig> {
    LlamaConfig::from_hf_json(&json)
}

fn base_config_json() -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("config.json")).unwrap())
        .unwrap()
}

#[test]
fn test_the_fixture_config_parses_to_the_fixture_shape() {
    let fx = fixture();
    let parsed = LlamaConfig::from_hf_config(fixture_dir().join("config.json")).unwrap();
    assert_eq!(parsed.hidden_size, fx.config.hidden_size);
    assert_eq!(parsed.num_hidden_layers, fx.config.num_hidden_layers);
    assert_eq!(parsed.num_attention_heads, fx.config.num_attention_heads);
    assert_eq!(parsed.num_key_value_heads, fx.config.num_key_value_heads);
    assert_eq!(parsed.intermediate_size, fx.config.intermediate_size);
    assert_eq!(parsed.vocab_size, fx.config.vocab_size);
    assert_eq!(parsed.rope_theta, fx.config.rope_theta);
    assert_eq!(parsed.rope_scaling, RopeScaling::None);
    assert_eq!(parsed.max_position_embeddings, None);
    // And the parsed config loads the checkpoint.
    ModelLoader::new(&fx.bytes)
        .unwrap()
        .load_weights(&parsed)
        .unwrap();
}

#[test]
fn test_rope_scaling_is_parsed_not_ignored() {
    let mut json = base_config_json();
    json["rope_scaling"] = json!({
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    });
    let parsed = parse(json).unwrap();
    assert_eq!(
        parsed.rope_scaling,
        RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192
        }
    );

    let mut json = base_config_json();
    json["rope_scaling"] = json!({"type": "linear", "factor": 2.0});
    assert_eq!(
        parse(json).unwrap().rope_scaling,
        RopeScaling::Linear { factor: 2.0 }
    );

    // The newer nested form, which also carries the theta.
    let mut json = base_config_json();
    json.as_object_mut().unwrap().remove("rope_theta");
    json["rope_parameters"] = json!({
        "rope_type": "llama3",
        "rope_theta": 500000.0,
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    });
    let parsed = parse(json).unwrap();
    assert_eq!(parsed.rope_theta, 500000.0);
    assert!(matches!(
        parsed.rope_scaling,
        RopeScaling::Llama3 { factor, .. } if factor == 8.0
    ));

    // Explicitly default scaling is fine.
    let mut json = base_config_json();
    json["rope_scaling"] = json!({"rope_type": "default"});
    assert_eq!(parse(json).unwrap().rope_scaling, RopeScaling::None);
    let mut json = base_config_json();
    json["rope_scaling"] = Value::Null;
    assert_eq!(parse(json).unwrap().rope_scaling, RopeScaling::None);
}

#[test]
fn test_unsupported_or_malformed_rope_scaling_is_an_error_not_a_default() {
    for scaling in [
        json!({"rope_type": "yarn", "factor": 4.0}),
        json!({"rope_type": "dynamic", "factor": 2.0}),
        json!({"rope_type": "longrope"}),
        json!({"rope_type": "linear"}),
        json!({"rope_type": "linear", "factor": "2"}),
        json!({"rope_type": "llama3", "factor": 8.0}),
        json!("linear"),
    ] {
        let mut json = base_config_json();
        json["rope_scaling"] = scaling.clone();
        assert!(
            parse(json).is_err(),
            "rope_scaling {scaling} must be refused, not silently ignored"
        );
    }
    // Two representations that disagree are refused too.
    let mut json = base_config_json();
    json["rope_scaling"] = json!({"rope_type": "linear", "factor": 2.0});
    json["rope_parameters"] = json!({"rope_type": "linear", "factor": 4.0});
    assert!(parse(json).is_err());
}

#[test]
fn test_features_the_engine_does_not_implement_are_refused() {
    let cases: Vec<(&str, Value)> = vec![
        ("model_type", json!("qwen2")),
        ("attention_bias", json!(true)),
        ("mlp_bias", json!(true)),
        ("hidden_act", json!("gelu")),
        ("head_dim", json!(32)),
        ("num_attention_heads", json!(5)),
        ("hidden_size", json!("96")),
    ];
    for (key, value) in cases {
        let mut json = base_config_json();
        json[key] = value.clone();
        let err = parse(json).expect_err(&format!("{key} = {value} must be refused"));
        println!("{key} = {value}: {err:#}");
    }
    // A consistent explicit head_dim is fine; so are the documented defaults.
    let mut json = base_config_json();
    json["head_dim"] = json!(24);
    json["attention_bias"] = json!(false);
    json["hidden_act"] = json!("silu");
    parse(json).unwrap();
}

#[test]
fn test_missing_required_fields_and_malformed_files_are_errors() {
    let mut json = base_config_json();
    json.as_object_mut().unwrap().remove("hidden_size");
    let msg = format!("{:#}", parse(json).unwrap_err());
    assert!(msg.contains("hidden_size"), "{msg}");

    assert!(parse(json!([1, 2, 3])).is_err());

    let dir = std::env::temp_dir().join(format!("paged-infer-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let broken = dir.join("config.json");
    std::fs::write(&broken, "{ not json").unwrap();
    assert!(LlamaConfig::from_hf_config(&broken).is_err());
    // A checkpoint beside a broken config must not fall back to defaults.
    assert!(LlamaConfig::beside_checkpoint(dir.join("model.safetensors")).is_err());
    std::fs::remove_file(&broken).unwrap();
    // No config at all: the documented default shape.
    let cfg = LlamaConfig::beside_checkpoint(dir.join("model.safetensors")).unwrap();
    assert_eq!(cfg.hidden_size, LlamaConfig::default().hidden_size);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_special_tokens_and_context_are_read_from_the_config() {
    let mut json = base_config_json();
    json["bos_token_id"] = json!(128000);
    json["eos_token_id"] = json!([128001, 128008, 128009]);
    json["max_position_embeddings"] = json!(4096);
    let parsed = parse(json).unwrap();
    assert_eq!(parsed.bos_token_id, Some(128000));
    assert_eq!(parsed.eos_token_ids, vec![128001, 128008, 128009]);
    assert_eq!(parsed.max_position_embeddings, Some(4096));

    let mut json = base_config_json();
    json["eos_token_id"] = json!(2);
    assert_eq!(parse(json).unwrap().eos_token_ids, vec![2]);

    // The fixture config declares neither; nothing is invented.
    let parsed = parse(base_config_json()).unwrap();
    assert_eq!(parsed.bos_token_id, None);
    assert!(parsed.eos_token_ids.is_empty());
}

// ── rotary scaling arithmetic ────────────────────────────────────────────────

#[test]
fn test_unscaled_inverse_frequencies_reproduce_the_legacy_table_exactly() {
    // The golden parity tests depend on the default path being bit-identical
    // to what `rope_table` always computed.
    let (head_dim, theta) = (24usize, 10_000.0f32);
    let inv = rope_inv_freq(head_dim, theta, RopeScaling::None);
    for pos in [0usize, 1, 7, 39, 1000] {
        let mut cos = vec![0.0; head_dim / 2];
        let mut sin = vec![0.0; head_dim / 2];
        rope_table(pos, head_dim, theta, &mut cos, &mut sin);
        let mut cos2 = vec![0.0; head_dim / 2];
        let mut sin2 = vec![0.0; head_dim / 2];
        paged_infer::math::rope_table_from(pos, &inv, &mut cos2, &mut sin2);
        assert_eq!(cos, cos2, "pos {pos}");
        assert_eq!(sin, sin2, "pos {pos}");
    }
}

#[test]
fn test_linear_scaling_divides_every_frequency() {
    let base = rope_inv_freq(64, 10_000.0, RopeScaling::None);
    let scaled = rope_inv_freq(64, 10_000.0, RopeScaling::Linear { factor: 4.0 });
    for (b, s) in base.iter().zip(&scaled) {
        assert!(
            (s - b / 4.0).abs() <= f32::EPSILON * b,
            "{s} vs {}",
            b / 4.0
        );
    }
}

#[test]
fn test_llama3_scaling_leaves_high_frequencies_and_divides_low_ones() {
    // Llama 3.1's published parameters. The highest frequency has a wavelength
    // of 2π positions — far below the high-frequency cutoff — so it is
    // untouched; the lowest has a wavelength of 2π·θ^(126/128) ≈ 2.7M
    // positions, far above the low cutoff, so it is divided by the factor.
    let (head_dim, theta) = (128usize, 500_000.0f32);
    let scaling = RopeScaling::Llama3 {
        factor: 8.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192,
    };
    let base = rope_inv_freq(head_dim, theta, RopeScaling::None);
    let scaled = rope_inv_freq(head_dim, theta, scaling);

    assert_eq!(
        scaled[0], base[0],
        "the highest frequency must be untouched"
    );
    let last = head_dim / 2 - 1;
    assert!(
        (scaled[last] - base[last] / 8.0).abs() <= f32::EPSILON * base[last],
        "the lowest frequency must be divided by the factor"
    );
    // Monotone, and every scaled frequency lies between full and divided.
    for j in 0..=last {
        assert!(scaled[j] <= base[j] * (1.0 + 1e-6) && scaled[j] >= base[j] / 8.0 * (1.0 - 1e-6));
        if j > 0 {
            assert!(
                scaled[j] < scaled[j - 1],
                "frequencies must stay decreasing"
            );
        }
    }
    // Something in the middle is genuinely interpolated, not just clamped.
    let interpolated = (1..last).any(|j| {
        let r = scaled[j] / base[j];
        r > 1.0 / 8.0 + 1e-3 && r < 1.0 - 1e-3
    });
    assert!(
        interpolated,
        "the smooth band between the cutoffs is missing"
    );
}

#[test]
fn test_scaled_and_unscaled_tables_run_the_same_weights_differently() {
    // Scaling reaches the forward pass: the same checkpoint with a scaling
    // declaration produces different logits at a non-zero position.
    let fx = fixture();
    let scaled = LlamaConfig {
        rope_scaling: RopeScaling::Linear { factor: 4.0 },
        ..fx.config.clone()
    };
    let plain = all_logits(&fx.bytes, &fx.config, &fx.tokens[..8]);
    let with_scaling = all_logits(&fx.bytes, &scaled, &fx.tokens[..8]);
    let vocab = fx.config.vocab_size;
    // Position 0 is unaffected: every angle there is zero.
    assert_eq!(&plain[..vocab], &with_scaling[..vocab]);
    let (max_abs, _) = compare(&plain[vocab..], &with_scaling[vocab..], vocab);
    assert!(max_abs > 1e-3, "scaling had no effect on later positions");
}

#[test]
fn test_dtype_helper_sizes() {
    assert_eq!(DType::BF16.size(), 2);
    assert_eq!(DType::F16.size(), 2);
    assert_eq!(DType::F32.size(), 4);
}
