#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: analyze_sweep.py <e2e_sweep.csv>")
    sys.exit(1)

rows = []
with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) if k not in ("batch", "steps", "total_tokens") else int(v) for k,v in r.items()})

by_batch = defaultdict(list)
by_steps = defaultdict(list)
for r in rows:
    by_batch[r["batch"]].append(r)
    by_steps[r["steps"]].append(r)

print("## 1→32 Batch Scaling (avg over step points)")
print("| batch | throughput_tok_s | avg_latency_ms | p95_us | peak_rss_mb |")
print("|---:|---:|---:|---:|---:|")
for b in sorted(by_batch):
    g = by_batch[b]
    print(f"| {b} | {statistics.mean(x['throughput_tok_s'] for x in g):.2f} | {statistics.mean(x['avg_token_latency_ms'] for x in g):.2f} | {statistics.mean(x['p95_token_latency_us'] for x in g):.0f} | {statistics.mean(x['peak_rss_mb'] for x in g):.2f} |")

print("\n```mermaid\nxychart-beta")
print('    title "Throughput vs Batch"')
print('    x-axis "batch" [' + ", ".join(str(b) for b in sorted(by_batch)) + "]")
print('    y-axis "tok/s" 0 --> ' + str(round(max(statistics.mean(x['throughput_tok_s'] for x in g) for g in by_batch.values()) + 1, 1)))
print('    line [' + ", ".join(f"{statistics.mean(x['throughput_tok_s'] for x in by_batch[b]):.2f}" for b in sorted(by_batch)) + "]")
print("```")

print("\n## Throughput vs Context Length (avg over batch points)")
print("| steps | throughput_tok_s | avg_latency_ms | p95_us |")
print("|---:|---:|---:|---:|")
for s in sorted(by_steps):
    g = by_steps[s]
    print(f"| {s} | {statistics.mean(x['throughput_tok_s'] for x in g):.2f} | {statistics.mean(x['avg_token_latency_ms'] for x in g):.2f} | {statistics.mean(x['p95_token_latency_us'] for x in g):.0f} |")

print("\n## Memory vs Sequence Length (avg over batch points)")
print("| steps | peak_rss_mb |")
print("|---:|---:|")
for s in sorted(by_steps):
    g = by_steps[s]
    print(f"| {s} | {statistics.mean(x['peak_rss_mb'] for x in g):.2f} |")
