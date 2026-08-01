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

/// `dot_multi` exists to load a weight row once instead of once per batch
/// entry. It must be a pure scheduling change: the batched matmul is claimed
/// bit-identical to the single-sequence path, and that claim dies the moment
/// the tiled kernel sums in a different order. So it is not enough for it to be
/// close to `dot` — it has to *be* `dot`, at every length, including the ragged
/// tails where the vector loop stops and scalar arithmetic takes over.
#[test]
fn test_dot_multi_is_dot_at_every_length_and_tile_width() {
    let mk = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 101) % 251) as f32) * 0.013 - 1.6)
            .collect()
    };

    // Lengths spanning both vector loops and every scalar-tail remainder:
    // AVX2 chunks 32 then 8, NEON chunks 16 then 4.
    for n in [
        0usize, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 63, 64, 65, 96, 129,
    ] {
        let w = mk(0, n);
        for bt in 1..=6usize {
            let mut block = Vec::with_capacity(bt * n);
            for b in 0..bt {
                block.extend(mk(b + 1, n));
            }
            let expect: Vec<f32> = (0..bt)
                .map(|b| paged_infer::simd::dot(&w, &block[b * n..(b + 1) * n]))
                .collect();

            let got: Vec<f32> = match bt {
                1 => paged_infer::simd::dot_multi::<1>(&w, &block, n).to_vec(),
                2 => paged_infer::simd::dot_multi::<2>(&w, &block, n).to_vec(),
                3 => paged_infer::simd::dot_multi::<3>(&w, &block, n).to_vec(),
                4 => paged_infer::simd::dot_multi::<4>(&w, &block, n).to_vec(),
                5 => paged_infer::simd::dot_multi::<5>(&w, &block, n).to_vec(),
                _ => paged_infer::simd::dot_multi::<6>(&w, &block, n).to_vec(),
            };
            assert_eq!(got, expect, "n={n} tile={bt} diverged from dot");
        }
    }
}

#[test]
fn test_dot_multi_matches_the_scalar_reference() {
    // The vectorized kernels are checked against the portable one, which is
    // just `BT` calls to `dot_scalar`. On a machine with SIMD this compares two
    // different reduction trees, so it is a tolerance check rather than an
    // equality one — unlike the test above, which compares like with like.
    for n in [5usize, 32, 33, 64, 100] {
        let w: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.05 - 0.4).collect();
        let block: Vec<f32> = (0..4 * n).map(|i| ((i % 23) as f32) * 0.03 - 0.3).collect();
        let got = paged_infer::simd::dot_multi::<4>(&w, &block, n);
        let expect = paged_infer::simd::dot_multi_scalar::<4>(&w, &block, n);
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-4, "n={n}: {g} vs {e}");
        }
    }
}

/// Same contract for the int8 tile: it hoists the widening out of the batch
/// loop, which must not move a single bit of the result.
#[test]
fn test_dot_i8_multi_is_dot_i8_at_every_length_and_tile_width() {
    for n in [
        0usize, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 64, 65, 96, 129,
    ] {
        // Include the range edges: i8::MIN sign-extends wrongly if the widening
        // is done with a zero-extend, and that bug hides at small magnitudes.
        let w: Vec<i8> = (0..n)
            .map(|i| match i % 5 {
                0 => i8::MIN,
                1 => i8::MAX,
                2 => -1,
                3 => 0,
                _ => ((i * 31) % 127) as i8,
            })
            .collect();
        for bt in [1usize, 2, 3, 4] {
            let block: Vec<f32> = (0..bt * n)
                .map(|i| (((i * 17) % 211) as f32) * 0.011 - 1.2)
                .collect();
            let expect: Vec<f32> = (0..bt)
                .map(|b| paged_infer::simd::dot_i8(&w, &block[b * n..(b + 1) * n]))
                .collect();
            let got: Vec<f32> = match bt {
                1 => paged_infer::simd::dot_i8_multi::<1>(&w, &block, n).to_vec(),
                2 => paged_infer::simd::dot_i8_multi::<2>(&w, &block, n).to_vec(),
                3 => paged_infer::simd::dot_i8_multi::<3>(&w, &block, n).to_vec(),
                _ => paged_infer::simd::dot_i8_multi::<4>(&w, &block, n).to_vec(),
            };
            assert_eq!(got, expect, "n={n} tile={bt} diverged from dot_i8");
        }
    }
}
