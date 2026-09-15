//! Timed-run profiles must exclude preparation, warmup and previous variants.
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

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

struct ReplayTemp(PathBuf);

impl ReplayTemp {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("paged-replay-profiling-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for ReplayTemp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn replay_records(output: Output) -> Vec<Value> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn replay_profiles_exclude_warmup_and_reset_between_configs_and_repeats() {
    let temp = ReplayTemp::new();
    let workload = temp.path("workload.json");
    fs::write(
        &workload,
        json!({"version": 1, "clock": "steps", "events": [
            {"type": "submit", "at": 0, "id": "prefill-only", "tokens": [1, 3, 4, 5, 6], "max_tokens": 1, "seed": 99}
        ]})
        .to_string(),
    )
    .unwrap();
    let config_paths = [temp.path("chunk2.json"), temp.path("chunk3.json")];
    for (path, chunk) in config_paths.iter().zip([2, 3]) {
        fs::write(
            path,
            json!({"name": format!("chunk{chunk}"), "engine": {
                "total_blocks": 8, "block_size": 16, "max_batch_size": 2,
                "prefill_chunk_size": chunk,
                "max_prefill_tokens_per_step": if chunk == 2 { 2 } else { 6 },
                "enable_prefix_cache": false, "draft_tokens": 0,
                "shared_prefix_attention": chunk == 3
            }})
            .to_string(),
        )
        .unwrap();
    }
    let invoke = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_workload_replay"));
        command
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args(["--threads", "1", "--workload"])
            .arg(&workload)
            .arg("--config")
            .arg(&config_paths[0])
            .arg("--config")
            .arg(&config_paths[1]);
        command
    };
    let mut records = replay_records(
        invoke()
            .args(["--repeats", "2", "--verify"])
            .output()
            .unwrap(),
    );
    let enabled = cfg!(feature = "profiling");
    assert_eq!(records[0]["version"], 1);
    assert_eq!(records[0]["profiling_enabled"], enabled);
    assert_eq!(records[0]["performance_gate_eligible"], !enabled);
    let fixture: Value = serde_json::from_str(include_str!("fixtures/config.json")).unwrap();
    let layers = fixture["num_hidden_layers"].as_u64().unwrap();
    let runs: Vec<_> = records.iter().filter(|row| row["type"] == "run").collect();
    let order: Vec<_> = runs
        .iter()
        .map(|row| {
            (
                row["config"].as_str().unwrap(),
                row["repeat"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        order,
        [("chunk2", 1), ("chunk3", 1), ("chunk3", 2), ("chunk2", 2)]
    );
    for run in runs {
        // The first generated token comes from prefill's final LM head; a
        // one-token budget finishes without a subsequent decode forward.
        let (model_batches, scheduler_ranges) = if run["config"] == "chunk2" {
            (3, 3)
        } else {
            (2, 1)
        };
        assert_eq!(run["report"]["engine"]["prefill_chunks"], scheduler_ranges);
        assert_eq!(run["report"]["engine"]["prompt_tokens_prefilled"], 5);
        assert_eq!(run["report"]["engine"]["generated_tokens"], 1);
        assert_eq!(run["report"]["summary"]["successful_requests"], 1);
        assert_eq!(
            run["report"]["requests"][0]["sequences"][0]["finish_reason"],
            "length"
        );
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
        // Each configuration warms one extra batch before replay. Exact
        // per-run counts catch both warmup leakage and accumulated prior runs.
        // Chunk3's single scheduler range contains two model batches (3+2),
        // so engine prefill_chunks is not the model-stage call count.
        assert_eq!(calls("model_setup"), model_batches);
        for stage in [
            "qkv_projection",
            "rope_kv_write",
            "attention",
            "output_projection",
        ] {
            assert_eq!(calls(stage), layers * model_batches, "{stage}");
        }
        assert_eq!(calls("feed_forward_projection"), 2 * layers * model_batches);
        assert_eq!(calls("lm_head"), 1);
        for entry in profile {
            if entry["scope"] != "model_main" {
                assert_eq!(entry["calls"], 0, "prefill must not use the shared path");
            }
            let elapsed = entry["elapsed_ms"].as_f64().unwrap();
            assert!(elapsed.is_finite() && elapsed >= 0.0);
        }
    }
    assert_eq!(records.last().unwrap()["passed"], true);

    // Older version-1 reports have no run-level profile. Comparing their
    // outputs must continue to work independently of diagnostic fields.
    for run in records.iter_mut().filter(|row| row["type"] == "run") {
        run.as_object_mut().unwrap().remove("profile");
    }
    let prior = temp.path("prior-without-profiles.jsonl");
    fs::write(
        &prior,
        records
            .iter()
            .map(|row| format!("{row}\n"))
            .collect::<String>(),
    )
    .unwrap();
    let compared = replay_records(
        invoke()
            .args(["--repeats", "1", "--compare-to"])
            .arg(prior)
            .output()
            .unwrap(),
    );
    assert_eq!(compared.last().unwrap()["passed"], true);
}
