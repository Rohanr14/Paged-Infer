//! Every vector kernel must agree with the portable scalar reference.
//!
//! The interesting failures in hand-written SIMD are not in the main loop but
//! at the edges: lengths that do not fill a vector, lengths that fill some
//! accumulators and not others, and the scalar tail. So these tests sweep every
//! length across the unroll boundaries rather than checking a few sizes.

use paged_infer::math::{
    matvec_f32_weight_transposed, matvec_f32_weight_transposed_parallel, matvec_i8_weight_parallel,
    quantize_rows_i8,
};
use paged_infer::simd;

/// Deterministic pseudo-random values with mixed signs and magnitudes.
fn seq(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((t % 2003) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn seq_i8(n: usize, seed: u32) -> Vec<i8> {
    (0..n)
        .map(|i| {
            let t = (i as u32).wrapping_mul(2246822519).wrapping_add(seed);
            ((t % 255) as i32 - 127) as i8
        })
        .collect()
}

/// Relative tolerance: reordering an inner product changes rounding, so the
/// kernels are not expected to be bit-identical to a sequential sum.
fn assert_close(actual: f32, expected: f32, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() / scale < 1e-5,
        "{what}: got {actual}, reference {expected}"
    );
}

#[test]
fn test_reports_a_backend() {
    // Not an assertion about which backend, just that dispatch resolves and the
    // benchmarks can name the code path they measured.
    let name = simd::backend();
    assert!(!name.is_empty());
    println!("simd backend: {name} (vectorized={})", simd::vectorized());
}

#[test]
fn test_dot_matches_scalar_at_every_length() {
    // 0..80 covers below one vector, across the 4x unroll (32 on AVX2, 16 on
    // NEON), and every tail length in between.
    for n in 0..80 {
        let a = seq(n, 1);
        let b = seq(n, 7);
        assert_close(
            simd::dot(&a, &b),
            simd::dot_scalar(&a, &b),
            &format!("dot n={n}"),
        );
    }
}

#[test]
fn test_dot_i8_matches_scalar_at_every_length() {
    for n in 0..80 {
        let w = seq_i8(n, 3);
        let x = seq(n, 11);
        assert_close(
            simd::dot_i8(&w, &x),
            simd::dot_i8_scalar(&w, &x),
            &format!("dot_i8 n={n}"),
        );
    }
}

#[test]
fn test_axpy_matches_scalar_at_every_length() {
    for n in 0..80 {
        let v = seq(n, 5);
        let base = seq(n, 13);
        let mut got = base.clone();
        let mut want = base;
        simd::axpy(&mut got, 0.375, &v);
        simd::axpy_scalar(&mut want, 0.375, &v);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_close(*g, *w, &format!("axpy n={n} i={i}"));
        }
    }
}

#[test]
fn test_dot_handles_extreme_magnitudes() {
    // Tiny and huge values in the same vector: a kernel that reassociates
    // wrongly, or drops the tail, shows up here.
    let a = vec![1e-8, 1e8, -1e-8, -1e8, 3.0, 0.0, -7.5, 2.25, 1.0];
    let b = vec![1e8, 1e-8, 1e8, 1e-8, 2.0, 5.0, 4.0, -1.0, 6.0];
    assert_close(simd::dot(&a, &b), simd::dot_scalar(&a, &b), "extremes");
}

#[test]
fn test_dot_i8_saturates_correctly_at_the_range_edges() {
    // i8::MIN must sign-extend to -128, not to 128.
    let w = vec![i8::MIN, i8::MAX, -1, 0, 1, i8::MIN, i8::MAX, -1, 5];
    let x = vec![1.0_f32; 9];
    let expected: f32 = w.iter().map(|v| *v as f32).sum();
    assert_close(simd::dot_i8(&w, &x), expected, "i8 range");
}

#[test]
fn test_parallel_matvec_matches_the_sequential_reference() {
    for (rows, cols) in [(1, 1), (3, 7), (8, 32), (17, 33), (64, 129), (129, 64)] {
        let weight = seq(rows * cols, 21);
        let x = seq(cols, 42);
        let mut fast = vec![0.0; rows];
        let mut reference = vec![0.0; rows];

        matvec_f32_weight_transposed_parallel(&mut fast, &x, &weight, rows, cols);
        matvec_f32_weight_transposed(&mut reference, &x, &weight, rows, cols);

        for (r, (f, s)) in fast.iter().zip(reference.iter()).enumerate() {
            assert_close(*f, *s, &format!("matvec {rows}x{cols} row {r}"));
        }
    }
}

#[test]
fn test_int8_matvec_tracks_the_f32_matvec() {
    // Quantization error is the only difference that should remain.
    let (rows, cols) = (48, 96);
    let weight = seq(rows * cols, 77);
    let x = seq(cols, 99);
    let (quant, scales) = quantize_rows_i8(&weight, rows, cols);

    let mut out_f32 = vec![0.0; rows];
    let mut out_i8 = vec![0.0; rows];
    matvec_f32_weight_transposed_parallel(&mut out_f32, &x, &weight, rows, cols);
    matvec_i8_weight_parallel(&mut out_i8, &x, &quant, &scales, rows, cols);

    for (r, (f, q)) in out_f32.iter().zip(out_i8.iter()).enumerate() {
        let denom = f.abs().max(1e-3);
        assert!(
            (f - q).abs() / denom < 0.02,
            "row {r}: f32={f}, int8={q} (>2% relative error)"
        );
    }
}

#[test]
fn test_matvec_is_row_independent() {
    // Guards against a parallel kernel writing outside its own row.
    let (rows, cols) = (33, 40);
    let weight = seq(rows * cols, 5);
    let x = seq(cols, 6);
    let mut out = vec![f32::NAN; rows];
    matvec_f32_weight_transposed_parallel(&mut out, &x, &weight, rows, cols);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "some rows were not written"
    );

    // Zeroing one weight row must change exactly that output.
    let mut zeroed = weight.clone();
    zeroed[10 * cols..11 * cols].fill(0.0);
    let mut out2 = vec![0.0; rows];
    matvec_f32_weight_transposed_parallel(&mut out2, &x, &zeroed, rows, cols);
    for r in 0..rows {
        if r == 10 {
            assert_eq!(out2[r], 0.0);
        } else {
            assert_close(out2[r], out[r], &format!("row {r} disturbed"));
        }
    }
}
