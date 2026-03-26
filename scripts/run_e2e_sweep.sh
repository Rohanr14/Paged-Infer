#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/tinyllama-1.1b/model.safetensors}"
OUT_CSV="${OUT_CSV:-e2e_sweep.csv}"
STEPS_LIST="${STEPS_LIST:-64 128 256}"
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"
WINDOW="${WINDOW:-256}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "MODEL_PATH not found: $MODEL_PATH"
  echo "Set MODEL_PATH=/absolute/path/to/model.safetensors"
  exit 1
fi

echo "batch,steps,total_tokens,throughput_tok_s,avg_token_latency_ms,p50_token_latency_us,p95_token_latency_us,peak_rss_mb" > "$OUT_CSV"

for b in $BATCH_LIST; do
  for s in $STEPS_LIST; do
    echo "Running batch=$b steps=$s"
    out=$(MODEL_PATH="$MODEL_PATH" BENCH_BATCH="$b" BENCH_STEPS="$s" BENCH_WINDOW="$WINDOW" cargo run --release --bin e2e_benchmark)

    total_tokens=$(echo "$out" | awk -F'=' '/total_tokens=/{print $NF}')
    throughput=$(echo "$out" | awk -F'=' '/throughput_tok_s=/{print $2}')
    avg_ms=$(echo "$out" | awk -F'=' '/avg_token_latency_ms=/{print $2}')
    p50=$(echo "$out" | awk -F'=' '/p50_token_latency_us=/{print $2}')
    p95=$(echo "$out" | awk -F'=' '/p95_token_latency_us=/{print $2}')
    rss=$(echo "$out" | awk -F'=' '/peak_rss_mb=/{print $2}')

    echo "$b,$s,$total_tokens,$throughput,$avg_ms,$p50,$p95,$rss" >> "$OUT_CSV"
  done
done

echo "Saved sweep results to $OUT_CSV"
