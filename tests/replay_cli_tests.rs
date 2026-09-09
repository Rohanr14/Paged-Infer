//! Exercise the public replay CLI, including its failure and reproducibility contract.
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "paged-replay-cli-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn invoke(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_workload_replay"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("RAYON_NUM_THREADS", "1")
        .args(args)
        .output()
        .unwrap()
}
fn trace(path: &Path) {
    fs::write(
        path,
        serde_json::to_vec(&json!({"version":1,"clock":"steps","events":[
            {"type":"submit","at":0,"id":"first","tokens":[1,3,4],"max_tokens":3,"seed":99},
            {"type":"submit","at":1,"id":"second","tokens":[1,5,6],"max_tokens":3,"seed":98}
        ]}))
        .unwrap(),
    )
    .unwrap();
}
fn records(output: &Output) -> Vec<Value> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
#[test]
fn reports_interleaved_repeats_and_can_compare_prior_outputs() {
    let temp = Temp::new();
    let workload = temp.path("trace.json");
    trace(&workload);
    let prior = temp.path("prior.jsonl");
    let first = invoke(&[
        "--workload",
        workload.to_str().unwrap(),
        "--repeats",
        "2",
        "--config",
        "workloads/configs/baseline.json",
        "--config",
        "workloads/configs/no-prefix.json",
        "--verify",
    ]);
    let rows = records(&first);
    let manifest = &rows[0];
    assert_eq!(manifest["type"], "manifest");
    assert_eq!(manifest["identity"]["synthetic_fixture"], true);
    assert_eq!(
        manifest["identity"]["weights_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(manifest["environment"]["rayon_threads"], 1);
    assert_eq!(manifest["configs"][0]["engine"]["eos_token"], u32::MAX);
    let runs: Vec<_> = rows.iter().filter(|row| row["type"] == "run").collect();
    let order: Vec<_> = runs
        .iter()
        .map(|row| row["config"].as_str().unwrap())
        .collect();
    assert_eq!(order, ["baseline", "no-prefix", "no-prefix", "baseline"]);
    assert_eq!(
        rows.iter()
            .filter(|row| row["type"] == "config_summary")
            .count(),
        2
    );
    assert_eq!(rows.last().unwrap()["passed"], true);
    fs::write(&prior, &first.stdout).unwrap();
    let second = invoke(&[
        "--workload",
        workload.to_str().unwrap(),
        "--repeats",
        "1",
        "--config",
        "workloads/configs/baseline.json",
        "--compare-to",
        prior.to_str().unwrap(),
    ]);
    assert_eq!(records(&second).last().unwrap()["passed"], true);
    let mut altered = rows;
    altered[0]["identity"]["weights_sha256"] = json!("wrong");
    fs::write(
        &prior,
        altered
            .iter()
            .map(|row| format!("{row}\n"))
            .collect::<String>(),
    )
    .unwrap();
    let mismatch = invoke(&[
        "--workload",
        workload.to_str().unwrap(),
        "--repeats",
        "1",
        "--config",
        "workloads/configs/baseline.json",
        "--compare-to",
        prior.to_str().unwrap(),
    ]);
    assert!(!mismatch.status.success());
    assert!(String::from_utf8_lossy(&mismatch.stderr).contains("fingerprint differs"));
}
#[test]
fn explicit_missing_model_and_invalid_overrides_fail_without_fallback() {
    let missing = invoke(&["--model", "/definitely/missing/replay-model.safetensors"]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("model checkpoint not found"));
    let temp = Temp::new();
    let config = temp.path("bad.json");
    for engine in [
        json!({"total_blocks":0}),
        json!({"block_size":u64::MAX}),
        json!({"stream_tokens":false}),
        json!({"prefill_chunk_size":0}),
        json!({"top_p":2}),
        json!({"typo":1}),
    ] {
        fs::write(&config, json!({"name":"bad","engine":engine}).to_string()).unwrap();
        let result = invoke(&["--config", config.to_str().unwrap(), "--repeats", "1"]);
        assert!(!result.status.success(), "accepted {engine}");
    }
}
#[test]
fn output_never_overwrites_an_existing_file_and_limits_fail() {
    let temp = Temp::new();
    let workload = temp.path("trace.json");
    trace(&workload);
    let report = temp.path("report.jsonl");
    fs::write(&report, "preserve me").unwrap();
    let output = invoke(&[
        "--workload",
        workload.to_str().unwrap(),
        "--output",
        report.to_str().unwrap(),
        "--repeats",
        "1",
    ]);
    assert!(!output.status.success());
    assert_eq!(fs::read_to_string(&report).unwrap(), "preserve me");
    let limited = invoke(&[
        "--workload",
        workload.to_str().unwrap(),
        "--max-steps",
        "1",
        "--repeats",
        "1",
    ]);
    assert!(!limited.status.success());
}
#[test]
fn explicit_checkpoint_retains_special_tokens_when_overrides_are_partial() {
    let temp = Temp::new();
    let model = temp.path("model.safetensors");
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_llama.safetensors"),
        &model,
    )
    .unwrap();
    let mut config: Value = serde_json::from_slice(
        &fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/config.json"))
            .unwrap(),
    )
    .unwrap();
    config["bos_token_id"] = json!(7);
    config["eos_token_id"] = json!([8, 9]);
    config["max_position_embeddings"] = json!(64);
    fs::write(temp.path("config.json"), config.to_string()).unwrap();
    fs::write(
        temp.path("generation_config.json"),
        json!({"eos_token_id":[9,10]}).to_string(),
    )
    .unwrap();
    let workload = temp.path("trace.json");
    trace(&workload);
    let output = invoke(&[
        "--model",
        model.to_str().unwrap(),
        "--workload",
        workload.to_str().unwrap(),
        "--config",
        "workloads/configs/baseline.json",
        "--repeats",
        "1",
    ]);
    let rows = records(&output);
    let engine = &rows[0]["configs"][0]["engine"];
    assert_eq!(engine["bos_token"], 7);
    assert_eq!(engine["eos_token"], 8);
    assert_eq!(engine["extra_eos_tokens"], json!([9, 10]));
    assert_eq!(engine["max_context"], 64);
    assert_eq!(rows[0]["identity"]["synthetic_fixture"], false);
}
