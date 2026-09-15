use serde_json::Value;
use std::process::Command;

fn command(percentage: usize, context: usize) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_shared_decode_benchmark"));
    command
        .env_remove("MODEL_PATH")
        .env("QUANT", "f32")
        .env("RAYON_NUM_THREADS", "2")
        .env("SHARED_DECODE_BATCH", "3")
        .env("SHARED_DECODE_CONTEXT", context.to_string())
        .env("SHARED_DECODE_STEPS", "3")
        .env("SHARED_DECODE_REPS", "2")
        .env("SHARED_DECODE_PERCENTAGE", percentage.to_string());
    command
}

#[test]
fn genuine_kv_and_outputs_are_invariant_to_physical_sharing() {
    let mut reference = None;
    for percent in [0, 50, 90] {
        let output = command(percent, 64).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let records: Vec<Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        assert_eq!(records.last().unwrap()["passed"], true);
        assert_eq!(records[0]["synthetic_kv"], false);
        for run in records.iter().filter(|r| r["type"] == "run") {
            let key = (
                run["generated_tokens"].clone(),
                run["final_logits_sha256"].clone(),
            );
            if let Some(expected) = &reference {
                assert_eq!(expected, &key);
            } else {
                reference = Some(key);
            }
            let shared = run["variant"] == "shared_prefix" && percent > 0;
            assert_eq!(run["shared_layer_calls"].as_u64().unwrap() > 0, shared);
            assert_eq!(
                run["candidate_extra_scratch_bytes"].as_u64().unwrap() > 0,
                shared
            );
        }
    }
}

#[test]
fn nonaligned_context_is_refused_before_preparing_weights() {
    let output = command(90, 13).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("multiple of 16"));
    assert!(output.stdout.is_empty());
}
