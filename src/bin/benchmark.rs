//! Kernel microbenchmarks for the decode hot path. No model weights required.
//!
//! Structured as an attribution ladder rather than a list of timings: each rung
//! adds exactly one change to the one above it, so the speedup column says what
//! each optimization is actually worth instead of only reporting a final
//! number.

use paged_infer::math::{
    matvec_bf16_weight_transposed, matvec_f32_weight_transposed_parallel,
    matvec_i8_weight_parallel, pack_bf16_to_f32, quantize_rows_i8,
};
use paged_infer::simd;
use std::time::Instant;

const WARMUP: usize = 3;

/// Run `f` a few times to warm caches and settle frequency, then time it.
fn bench<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64()
}

/// Baseline: re-widen the bf16 weights to f32 on every call, then a serial
/// single-accumulator matvec. This is the naive thing to write.
fn convert_then_matvec(out: &mut [f32], x: &[f32], w_bf16: &[u8], cols: usize) {
    let w: Vec<f32> = w_bf16
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    for (r, o) in out.iter_mut().enumerate() {
        *o = simd::dot_naive(&w[r * cols..(r + 1) * cols], x);
    }
}

fn matvec_serial<F: Fn(&[f32], &[f32]) -> f32>(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    cols: usize,
    dot: F,
) {
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot(&w[r * cols..(r + 1) * cols], x);
    }
}

/// `bytes` is the traffic one iteration actually moves, which differs per
/// kernel — the bf16 baseline also materializes and re-reads a full f32 copy,
/// and int8 moves a quarter of what f32 does. Reporting each against its own
/// traffic is what makes the throughput column comparable.
fn row(label: &str, secs: f64, iters: usize, baseline: f64, prev: f64, bytes: f64) {
    let per_iter_ms = secs * 1000.0 / iters as f64;
    let gib_s = bytes * iters as f64 / secs / (1024.0 * 1024.0 * 1024.0);
    println!(
        "{label:<34} {per_iter_ms:>8.3} ms {:>8.2}x {:>8.2}x {gib_s:>9.1}",
        baseline / secs.max(1e-12),
        prev / secs.max(1e-12),
    );
}

/// Last-level cache size in bytes, if the OS will tell us. Used only to warn
/// when the benchmark is measuring cache bandwidth rather than DRAM.
fn last_level_cache_bytes() -> Option<usize> {
    let mut best = None;
    for idx in 0..8 {
        let path = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}/size");
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let raw = raw.trim();
        let (digits, mult) = match raw.chars().last() {
            Some('K') => (&raw[..raw.len() - 1], 1024),
            Some('M') => (&raw[..raw.len() - 1], 1024 * 1024),
            Some('G') => (&raw[..raw.len() - 1], 1024 * 1024 * 1024),
            _ => (raw, 1),
        };
        if let Ok(v) = digits.parse::<usize>() {
            best = Some(best.map_or(v * mult, |b: usize| b.max(v * mult)));
        }
    }
    best
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let rows = env_usize("BENCH_ROWS", 2048);
    let cols = env_usize("BENCH_COLS", 2048);
    let iters = env_usize("BENCH_ITERS", 20);
    let threads = rayon::current_num_threads();
    let working_set = rows * cols * 4;

    println!("Paged-Infer kernel microbenchmark");
    println!("=================================");
    println!("matrix     : {rows}x{cols} f32, {iters} timed iters after {WARMUP} warm-up");
    println!("simd       : {}", simd::backend());
    println!("rayon      : {threads} threads");
    println!(
        "working set: {:.1} MiB of f32 weights",
        working_set as f64 / 1_048_576.0
    );
    if let Some(llc) = last_level_cache_bytes() {
        println!(
            "last-level cache: {:.1} MiB -- weights are {}",
            llc as f64 / 1_048_576.0,
            if working_set < llc {
                "CACHE-RESIDENT, so this measures cache bandwidth, not DRAM. \n             \
                 Real 1B+ models do not fit; re-run with BENCH_ROWS/BENCH_COLS large \n             \
                 enough to exceed it for a DRAM-bound picture."
            } else {
                "larger than cache, so this is DRAM-bound like a real model"
            }
        );
    }
    println!();

    let x: Vec<f32> = (0..cols).map(|i| ((i % 31) as f32) * 0.01 - 0.15).collect();
    let mut out = vec![0.0f32; rows];

    let mut w_bf16 = vec![0u8; rows * cols * 2];
    for i in 0..rows * cols {
        let b = half::bf16::from_f32(((i % 97) as f32) * 0.001).to_le_bytes();
        w_bf16[i * 2] = b[0];
        w_bf16[i * 2 + 1] = b[1];
    }
    let packed = pack_bf16_to_f32(&w_bf16);
    let (quant, scales) = quantize_rows_i8(&packed, rows, cols);

    // Traffic per iteration, per kernel. The bf16 baseline reads the packed
    // bf16 matrix, writes a full f32 copy, then reads that copy back.
    let bf16_bytes = (rows * cols * 2) as f64;
    let f32_bytes = (rows * cols * 4) as f64;
    let convert_bytes = bf16_bytes + 2.0 * f32_bytes;
    let i8_bytes = (rows * cols + rows * 4) as f64;

    println!(
        "{:<34} {:>11} {:>8} {:>8} {:>9}",
        "kernel", "per iter", "vs base", "vs prev", "GiB/s"
    );
    println!("{}", "-".repeat(74));

    let baseline = bench(iters, || convert_then_matvec(&mut out, &x, &w_bf16, cols));
    row(
        "bf16 convert + serial scalar",
        baseline,
        iters,
        baseline,
        baseline,
        convert_bytes,
    );

    let stream = bench(iters, || {
        matvec_bf16_weight_transposed(&mut out, &x, &w_bf16, rows, cols)
    });
    row(
        "+ stream bf16 (no realloc)",
        stream,
        iters,
        baseline,
        baseline,
        bf16_bytes,
    );

    let serial_naive = bench(iters, || {
        matvec_serial(&mut out, &x, &packed, cols, simd::dot_naive)
    });
    row(
        "+ prepack f32, serial scalar",
        serial_naive,
        iters,
        baseline,
        stream,
        f32_bytes,
    );

    let serial_unrolled = bench(iters, || {
        matvec_serial(&mut out, &x, &packed, cols, simd::dot_scalar)
    });
    row(
        "+ 4 accumulators (serial)",
        serial_unrolled,
        iters,
        baseline,
        serial_naive,
        f32_bytes,
    );

    let serial_simd = bench(iters, || {
        matvec_serial(&mut out, &x, &packed, cols, simd::dot)
    });
    row(
        &format!("+ {} (serial)", simd::backend()),
        serial_simd,
        iters,
        baseline,
        serial_unrolled,
        f32_bytes,
    );

    let parallel_simd = bench(iters, || {
        matvec_f32_weight_transposed_parallel(&mut out, &x, &packed, rows, cols)
    });
    row(
        &format!("+ rayon across rows ({threads}t)"),
        parallel_simd,
        iters,
        baseline,
        serial_simd,
        f32_bytes,
    );

    let parallel_i8 = bench(iters, || {
        matvec_i8_weight_parallel(&mut out, &x, &quant, &scales, rows, cols)
    });
    row(
        "+ int8 weights",
        parallel_i8,
        iters,
        baseline,
        parallel_simd,
        i8_bytes,
    );

    println!();
    println!("Attribution");
    println!("-----------");
    println!(
        "  one-time prepack        : {:>6.2}x  (stop re-widening bf16 every call)",
        baseline / serial_naive.max(1e-12)
    );
    // Expect roughly 1.00x here. Unrolling accumulators only pays once the
    // loop is issuing wide FMAs; a scalar loop is limited by load throughput
    // long before the FMA dependency chain becomes the bottleneck.
    println!(
        "  accumulator unrolling   : {:>6.2}x  (little effect until the loop is vectorized)",
        serial_naive / serial_unrolled.max(1e-12)
    );
    println!(
        "  hand-written {:<11}: {:>6.2}x  (vs the 4-accumulator scalar loop)",
        simd::backend(),
        serial_unrolled / serial_simd.max(1e-12)
    );
    println!(
        "  rayon across rows       : {:>6.2}x  ({threads} threads)",
        serial_simd / parallel_simd.max(1e-12)
    );
    println!(
        "  int8 weights            : {:>6.2}x  (4x less weight traffic)",
        parallel_simd / parallel_i8.max(1e-12)
    );
    println!(
        "  total                   : {:>6.2}x",
        baseline / parallel_i8.max(1e-12)
    );

    println!();
    println!("Weight memory");
    println!("-------------");
    let f32_mem = rows * cols * 4;
    let i8_mem = rows * cols + rows * 4;
    println!(
        "  f32  : {:>8.2} MB\n  int8 : {:>8.2} MB  ({:.2}x smaller)",
        f32_mem as f64 / 1_048_576.0,
        i8_mem as f64 / 1_048_576.0,
        f32_mem as f64 / i8_mem as f64
    );

    println!();
    println!("Note: a matvec streams the whole matrix for one pass of multiply-adds,");
    println!("so it is memory-bound. Past the point where the kernel saturates memory");
    println!("bandwidth, further arithmetic tuning buys nothing -- which is why int8,");
    println!("a 4x cut in bytes moved, still helps after the kernel is fully vectorized.");
}
