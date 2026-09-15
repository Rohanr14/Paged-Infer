//! The kernel benchmark must execute and report its selected attention shape.
use serde_json::{json, Value};
use std::process::{Command, Output};

fn invoke(kv_heads: Option<&str>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_shared_attention_benchmark"));
    command
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("RAYON_NUM_THREADS", "2")
        .env("SHARED_ATTN_BATCHES", "4")
        .env("SHARED_ATTN_CONTEXTS", "32")
        .env("SHARED_ATTN_PERCENTAGES", "50")
        .env("SHARED_ATTN_REPS", "1")
        .env("SHARED_ATTN_WARMUP", "1")
        .env_remove("SHARED_ATTN_KV_HEADS");
    if let Some(value) = kv_heads {
        command.env("SHARED_ATTN_KV_HEADS", value);
    }
    command.output().unwrap()
}

fn verify_shape(output: Output, kv_heads: usize) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records[0]["type"], "manifest");
    assert_eq!(
        records[0]["shape"],
        json!({"layers": 1, "heads": 32, "kv_heads": kv_heads,
            "head_dim": 64, "block_size": 16})
    );
    let cases: Vec<_> = records
        .iter()
        .filter(|record| record["type"] == "case")
        .collect();
    assert_eq!(cases.len(), 1);
    let case = cases[0];
    assert_eq!(case["selected_path"], "shared_prefix");
    assert_eq!(case["shared_prefix_tokens"], 16);
    assert_eq!(case["verified_finite_bit_identical"], true);
    assert_eq!(case["max_abs_error"], 0.0);
    assert_eq!(case["samples"].as_array().unwrap().len(), 1);
    // One shared block plus two private blocks per sequence. This checks the
    // actual KV layout, not just a manifest field disconnected from execution.
    assert_eq!(
        case["physical_kv_bytes"],
        json!(9 * 16 * kv_heads * 2 * 64 * 4)
    );
    assert_eq!(records.last().unwrap()["passed"], true);
    assert_eq!(records.last().unwrap()["complete"], true);
    assert_eq!(records.last().unwrap()["cases"], 1);
}

#[test]
fn eight_kv_heads_use_the_llama_shape_and_verify_every_output() {
    verify_shape(invoke(Some("8")), 8);
}

#[test]
fn omitted_kv_heads_preserves_the_default_tinyllama_shape() {
    verify_shape(invoke(None), 4);
}

#[test]
fn invalid_kv_head_counts_fail_before_emitting_measurements() {
    for value in ["0", "3", "33", "4 8"] {
        let output = invoke(Some(value));
        assert!(!output.status.success(), "accepted KV head count {value}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("SHARED_ATTN_KV_HEADS"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.stdout.is_empty(),
            "reported measurements for invalid shape {value}"
        );
    }
}
