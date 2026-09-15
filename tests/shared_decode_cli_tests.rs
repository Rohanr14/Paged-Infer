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
        .env("PAGED_INFER_MATMUL_TILE", "4")
        .env("PAGED_INFER_ATTN_LANES_PER_THREAD", "2")
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
        assert_eq!(records[0]["environment"]["matmul_tile_env"], "4");
        assert_eq!(records[0]["warmup_logit_hash"]["timed_loop_hashing"], false);
        assert_eq!(records[0]["warmup_logit_hash"]["steps_per_run"], 3);
        assert_eq!(records.last().unwrap()["warmup_logit_steps_per_run"], 3);
        assert_eq!(records.iter().filter(|r| r["type"] == "run").count(), 4);
        let mut total_elapsed = [0.0; 2];
        for run in records.iter().filter(|r| r["type"] == "run") {
            let key = (
                run["generated_tokens"].clone(),
                run["final_logits_sha256"].clone(),
                run["warmup_logits_sha256"].clone(),
            );
            if let Some(expected) = &reference {
                assert_eq!(expected, &key);
            } else {
                reference = Some(key);
            }
            assert_eq!(run["warmup_logit_steps_hashed"], 3);
            assert_eq!(run["verified_complete_warmup_logits"], true);
            assert_eq!(run["verified_warmup_timed_greedy_output"], true);
            assert_eq!(
                run["warmup_logits_sha256"],
                records.last().unwrap()["warmup_logits_sha256"]
            );
            let digest = run["warmup_logits_sha256"].as_str().unwrap();
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert!(run["warmup_elapsed_ms"].as_f64().unwrap() >= 0.0);
            assert!(run.get("timed_loop_start_unix_ms").is_some());
            assert!(run.get("process_resource_usage").is_some());
            if !run["timed_loop_start_unix_ms"].is_null() {
                assert!(run["timed_loop_start_unix_ms"].as_u64().is_some());
            }
            let usage = &run["process_resource_usage"];
            if !usage.is_null() {
                for field in ["user_cpu_ms", "system_cpu_ms"] {
                    assert!(usage[field].as_f64().unwrap() >= 0.0);
                }
                for field in [
                    "minor_page_faults",
                    "major_page_faults",
                    "voluntary_context_switches",
                    "involuntary_context_switches",
                ] {
                    assert!(usage[field].as_u64().is_some());
                }
            }
            let variant = usize::from(run["variant"] == "shared_prefix");
            total_elapsed[variant] += run["elapsed_ms"].as_f64().unwrap();
            let shared = run["variant"] == "shared_prefix" && percent > 0;
            assert_eq!(run["shared_layer_calls"].as_u64().unwrap() > 0, shared);
            assert_eq!(
                run["candidate_extra_scratch_bytes"].as_u64().unwrap() > 0,
                shared
            );
        }
        let summary = records.iter().find(|r| r["type"] == "summary").unwrap();
        assert_eq!(summary["paired_elapsed_time_speedup"]["pair_count"], 2);
        let diagnostics = &summary["performance_diagnostics"];
        assert_eq!(diagnostics["diagnostic_only"], true);
        let aggregate = diagnostics["aggregate_throughput"]["speedup"]
            .as_f64()
            .unwrap();
        let expected_aggregate = total_elapsed[0] / total_elapsed[1];
        // Allow float serialization roundoff, without asserting a speed target.
        assert!((aggregate - expected_aggregate).abs() <= expected_aggregate.abs() * 1e-12);
        assert_eq!(
            diagnostics["aggregate_throughput"]["tokens_per_variant"],
            18.0
        );
        let strata = diagnostics["order_stratified_paired_speedup"]
            .as_array()
            .unwrap();
        assert_eq!(strata.len(), 2);
        for (index, stratum) in strata.iter().enumerate() {
            assert_eq!(stratum["pair_count"], 1);
            assert_eq!(stratum["repeat_numbers"][0], index + 1);
            assert_eq!(
                stratum["median"],
                summary["paired_elapsed_time_speedup"]["pairs"][index]["speedup"]
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
