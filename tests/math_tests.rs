use paged_infer::math::{
    apply_rope, matmul, matvec_f32_weight_transposed_parallel, matvec_i8_weight_parallel,
    quantize_rows_i8, rms_norm, silu, swiglu,
};

// Helper function to compare f32 slices with a small tolerance
fn assert_f32_slice_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "Slice lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < epsilon,
            "Mismatch at index {}: expected {}, got {}",
            i,
            e,
            a
        );
    }
}

#[test]
fn test_matmul() {
    // 2x3 matrix A
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // 3x2 matrix B
    let b = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0];
    // Expected 2x2 matrix C
    let mut c = vec![0.0; 4];

    matmul(&mut c, &a, &b, 2, 3, 2);

    let expected = vec![31.0, 19.0, 85.0, 55.0];
    assert_f32_slice_eq(&c, &expected, 1e-5);
}

#[test]
fn test_rms_norm() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.5, 0.5, 0.5, 0.5];
    let epsilon = 1e-5;

    // Mean of squares: (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // Inverse root: 1 / sqrt(7.5) ≈ 0.365148
    // Expected x[i] = x[i] * 0.365148 * 0.5

    rms_norm(&mut x, &weight, epsilon);

    let expected = vec![0.182574, 0.365148, 0.547722, 0.730296];
    assert_f32_slice_eq(&x, &expected, 1e-5);
}

#[test]
fn test_silu() {
    let x = 2.0;
    let result = silu(x);
    // 2.0 / (1.0 + e^(-2.0)) ≈ 1.761594
    assert!((result - 1.761594).abs() < 1e-5);
}

#[test]
fn test_swiglu() {
    let mut x = vec![1.0, -1.0, 2.0];
    let w3 = vec![2.0, 0.5, -1.0];

    // silu(1.0) ≈ 0.73105 * 2.0 = 1.4621
    // silu(-1.0) ≈ -0.26894 * 0.5 = -0.13447
    // silu(2.0) ≈ 1.76159 * -1.0 = -1.76159

    swiglu(&mut x, &w3);

    let expected = vec![1.462117, -0.134470, -1.761594];
    assert_f32_slice_eq(&x, &expected, 1e-5);
}

#[test]
fn test_apply_rope() {
    // 4-dimensional embeddings (head_dim = 4)
    let mut q = vec![1.0, 0.0, 1.0, 0.0];
    let mut k = vec![1.0, 0.0, 1.0, 0.0];

    let pos = 1;
    let head_dim = 4;
    let rope_theta = 10000.0;

    apply_rope(&mut q, &mut k, pos, head_dim, rope_theta);

    // For i=0 (dim 0,1): freq = 1.0. val = 1.0. sin=0.84147, cos=0.54030
    // q[0] = 1*cos - 0*sin = 0.54030
    // q[1] = 1*sin + 0*cos = 0.84147
    // For i=2 (dim 2,3): freq = 1/sqrt(10000) = 0.01. val = 0.01. sin=0.0099998, cos=0.99995
    // q[2] = 1*cos - 0*sin = 0.99995
    // q[3] = 1*sin + 0*cos = 0.0099998

    let expected = vec![0.540302, 0.841470, 0.999950, 0.0099998];
    assert_f32_slice_eq(&q, &expected, 1e-5);
    assert_f32_slice_eq(&k, &expected, 1e-5); // K should match Q in this exact scenario
}

#[test]
fn test_softmax_in_place() {
    let mut logits = vec![1.0, 2.0, 3.0];
    paged_infer::math::softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(logits[2] > logits[1] && logits[1] > logits[0]);
}

#[test]
fn test_paged_attention_matches_manual() {
    let q = vec![1.0, 0.0];
    let k1 = vec![1.0, 0.0];
    let k2 = vec![0.0, 1.0];
    let v1 = vec![2.0, 0.0];
    let v2 = vec![0.0, 4.0];

    let mut scores = vec![0.0; 2];
    let mut out = vec![0.0; 2];
    paged_infer::math::paged_attention(&q, &[&k1, &k2], &[&v1, &v2], &mut scores, &mut out);

    // Score 0 should dominate score 1 because q·k1 > q·k2
    assert!(scores[0] > scores[1]);
    // Output should be a weighted blend biased toward v1.
    assert!(out[0] > out[1]);
}

#[test]
fn test_matvec_bf16_weight_transposed_matches_reference() {
    let x = vec![1.0_f32, 2.0, 3.0];
    let w_f32 = [
        1.0_f32, 0.0, 1.0, // row 0
        0.5, 1.0, -1.0, // row 1
    ];
    let mut w_bf16 = Vec::new();
    for &v in &w_f32 {
        w_bf16.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }

    let mut out = vec![0.0; 2];
    paged_infer::math::matvec_bf16_weight_transposed(&mut out, &x, &w_bf16, 2, 3);

    assert!((out[0] - 4.0).abs() < 1e-3);
    assert!((out[1] - (-0.5)).abs() < 1e-3);
}

#[test]
fn test_quantize_rows_i8_roundtrip() {
    let rows = 4;
    let cols = 8;
    // Create a known weight matrix with values in [-1.0, 1.0]
    let weight: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) / (rows * cols) as f32) * 2.0 - 1.0)
        .collect();

    let max_abs = weight
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f32, f32::max);

    let (quant, scales) = quantize_rows_i8(&weight, rows, cols);

    // Dequantize manually and check round-trip error
    let epsilon = 0.01 * max_abs;
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let dequant = quant[idx] as f32 * scales[r];
            assert!(
                (dequant - weight[idx]).abs() < epsilon.max(1e-4),
                "Roundtrip error too large at [{r},{c}]: original={}, dequant={}, diff={}",
                weight[idx],
                dequant,
                (dequant - weight[idx]).abs()
            );
        }
    }
}

#[test]
fn test_matvec_i8_matches_f32() {
    let rows = 16;
    let cols = 32;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.15).collect();
    let weight: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 37) as f32) * 0.01 - 0.18)
        .collect();

    let (quant, scales) = quantize_rows_i8(&weight, rows, cols);

    let mut out_f32 = vec![0.0_f32; rows];
    let mut out_i8 = vec![0.0_f32; rows];

    matvec_f32_weight_transposed_parallel(&mut out_f32, &x, &weight, rows, cols);
    matvec_i8_weight_parallel(&mut out_i8, &x, &quant, &scales, rows, cols);

    // Allow up to 2% relative error (int8 per-row symmetric quantization)
    for (r, (&f, &i)) in out_f32.iter().zip(out_i8.iter()).enumerate() {
        let rel_err = if f.abs() > 1e-6 {
            (f - i).abs() / f.abs()
        } else {
            (f - i).abs()
        };
        assert!(
            rel_err < 0.02,
            "Relative error too large at row {r}: f32={f}, i8={i}, rel_err={rel_err}"
        );
    }
}

#[test]
fn test_rms_norm_bf16_weight_close_to_f32_weight() {
    let mut a = vec![1.0_f32, 2.0, 3.0, 4.0];
    let mut b = a.clone();
    let w = vec![0.5_f32, 0.5, 0.5, 0.5];
    let mut w_bf16 = Vec::new();
    for &v in &w {
        w_bf16.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }

    paged_infer::math::rms_norm(&mut a, &w, 1e-5);
    paged_infer::math::rms_norm_bf16_weight(&mut b, &w_bf16, 1e-5);

    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 2e-3);
    }
}

#[test]
fn test_packed_parallel_matvec_matches_streaming_bf16() {
    let rows = 4;
    let cols = 3;
    let x = vec![1.0_f32, -2.0, 0.5];
    let w_f32 = [
        1.0_f32, 0.0, -1.0, 0.5, 2.0, 1.5, -0.25, 0.75, 1.25, 2.0, -1.0, 0.0,
    ];
    let mut w_bf16 = Vec::new();
    for &v in &w_f32 {
        w_bf16.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }

    let packed = paged_infer::math::pack_bf16_to_f32(&w_bf16);
    let mut out_stream = vec![0.0; rows];
    let mut out_parallel = vec![0.0; rows];
    paged_infer::math::matvec_bf16_weight_transposed(&mut out_stream, &x, &w_bf16, rows, cols);
    paged_infer::math::matvec_f32_weight_transposed_parallel(
        &mut out_parallel,
        &x,
        &packed,
        rows,
        cols,
    );

    for (a, b) in out_stream.iter().zip(out_parallel.iter()) {
        assert!((a - b).abs() < 2e-3);
    }
}
