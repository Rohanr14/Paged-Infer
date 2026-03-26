use paged_infer::math::{apply_rope, matmul, rms_norm, silu, swiglu};

// Helper function to compare f32 slices with a small tolerance
fn assert_f32_slice_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "Slice lengths differ");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < epsilon,
            "Mismatch at index {}: expected {}, got {}",
            i, e, a
        );
    }
}

#[test]
fn test_matmul() {
    // 2x3 matrix A
    let a = vec![
        1.0, 2.0, 3.0, 
        4.0, 5.0, 6.0
    ];
    // 3x2 matrix B
    let b = vec![
        7.0, 8.0, 
        9.0, 1.0, 
        2.0, 3.0
    ];
    // Expected 2x2 matrix C
    let mut c = vec![0.0; 4];

    matmul(&mut c, &a, &b, 2, 3, 2);

    let expected = vec![
        31.0, 19.0, 
        85.0, 55.0
    ];
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
    
    let expected = vec![
        0.182574, 
        0.365148, 
        0.547722, 
        0.730296
    ];
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