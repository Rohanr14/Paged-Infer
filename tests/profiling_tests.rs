//! Timed-run profiles must exclude preparation, warmup and previous variants.
use serde_json::Value;
use std::process::Command;

#[test]
fn benchmark_profiles_only_the_measured_decode_and_selected_path() {
    let output = Command::new(env!("CARGO_BIN_EXE_shared_decode_benchmark"))
        .env_remove("MODEL_PATH")
        .env("QUANT", "f32")
        .env("RAYON_NUM_THREADS", "2")
        .env("SHARED_DECODE_BATCH", "4")
        .env("SHARED_DECODE_CONTEXT", "64")
        .env("SHARED_DECODE_STEPS", "3")
        .env("SHARED_DECODE_REPS", "2")
        .env("SHARED_DECODE_PERCENTAGE", "90")
        .output()
        .unwrap();
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
    let enabled = cfg!(feature = "profiling");
    assert_eq!(records[0]["profiling_enabled"], enabled);
    let layers = records[0]["shape"]["layers"].as_u64().unwrap();
    for run in records.iter().filter(|record| record["type"] == "run") {
        let profile = run["profile"].as_array().unwrap();
        if !enabled {
            assert!(profile.is_empty());
            continue;
        }
        let calls = |stage: &str| {
            profile
                .iter()
                .find(|entry| entry["stage"] == stage)
                .unwrap()["calls"]
                .as_u64()
                .unwrap()
        };
        assert_eq!(calls("qkv_projection"), layers * 3);
        assert_eq!(calls("attention"), layers * 3);
        assert_eq!(calls("lm_head"), 3);
        let shared = run["selected_path"] == "shared_prefix";
        assert_eq!(calls("shared_scores") > 0, shared);
        assert_eq!(calls("shared_values") > 0, shared);
        for entry in profile {
            let elapsed = entry["elapsed_ms"].as_f64().unwrap();
            assert!(elapsed.is_finite() && elapsed >= 0.0);
        }
    }
    assert_eq!(records.last().unwrap()["passed"], true);
}
