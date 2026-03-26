use rayon::prelude::*;

/// Root Mean Square Normalization (RMSNorm)
pub fn rms_norm(x: &mut [f32], weight: &[f32], epsilon: f32) {
    let size = x.len();
    assert_eq!(
        size,
        weight.len(),
        "Input and weight size mismatch for RMSNorm"
    );

    let mut ss = 0.0;
    for &val in x.iter() {
        ss += val * val;
    }
    ss /= size as f32;

    let inv_norm = 1.0 / (ss + epsilon).sqrt();
    for i in 0..size {
        x[i] = x[i] * inv_norm * weight[i];
    }
}

pub fn rms_norm_bf16_weight(x: &mut [f32], weight_bf16: &[u8], epsilon: f32) {
    assert_eq!(
        weight_bf16.len(),
        x.len() * 2,
        "RMSNorm bf16 weight size mismatch"
    );

    let mut ss = 0.0;
    for &val in x.iter() {
        ss += val * val;
    }
    let inv_norm = 1.0 / ((ss / x.len() as f32) + epsilon).sqrt();

    for (i, xi) in x.iter_mut().enumerate() {
        let lo = weight_bf16[i * 2];
        let hi = weight_bf16[i * 2 + 1];
        let w = half::bf16::from_le_bytes([lo, hi]).to_f32();
        *xi = *xi * inv_norm * w;
    }
}

pub fn matvec_bf16_weight_transposed(
    out: &mut [f32],
    x: &[f32],
    weight_bf16: &[u8],
    rows: usize,
    cols: usize,
) {
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(weight_bf16.len(), rows * cols * 2);

    for r in 0..rows {
        let mut acc = 0.0;
        let row_offset = r * cols * 2;
        for c in 0..cols {
            let idx = row_offset + c * 2;
            let w = half::bf16::from_le_bytes([weight_bf16[idx], weight_bf16[idx + 1]]).to_f32();
            acc += x[c] * w;
        }
        out[r] = acc;
    }
}

pub fn pack_bf16_to_f32(weight_bf16: &[u8]) -> Vec<f32> {
    weight_bf16
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

pub fn matvec_f32_weight_transposed(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(weight.len(), rows * cols);
    for r in 0..rows {
        let mut acc = 0.0;
        let base = r * cols;
        for c in 0..cols {
            acc += x[c] * weight[base + c];
        }
        out[r] = acc;
    }
}

pub fn matvec_f32_weight_transposed_parallel(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(weight.len(), rows * cols);

    out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
        let mut acc = 0.0;
        let base = r * cols;
        for c in 0..cols {
            acc += x[c] * weight[base + c];
        }
        *out_r = acc;
    });
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn swiglu(x: &mut [f32], x_w3: &[f32]) {
    assert_eq!(x.len(), x_w3.len(), "SwiGLU input sizes must match");
    for i in 0..x.len() {
        x[i] = silu(x[i]) * x_w3[i];
    }
}

pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, rope_theta: f32) {
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / rope_theta.powf((i as f32) / (head_dim as f32));
        let val = (pos as f32) * freq;
        let (sin, cos) = val.sin_cos();

        let q0 = q[i];
        let q1 = q[i + 1];
        q[i] = q0 * cos - q1 * sin;
        q[i + 1] = q0 * sin + q1 * cos;

        let k0 = k[i];
        let k1 = k[i + 1];
        k[i] = k0 * cos - k1 * sin;
        k[i + 1] = k0 * sin + k1 * cos;
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn softmax_in_place(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_v = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum.max(1e-12);
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

pub fn paged_attention(
    q: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    scores: &mut [f32],
    out: &mut [f32],
) {
    assert_eq!(keys.len(), values.len());
    assert_eq!(keys.len(), scores.len());

    out.fill(0.0);
    let scale = 1.0 / (q.len() as f32).sqrt();

    for (i, k) in keys.iter().enumerate() {
        scores[i] = dot(q, k) * scale;
    }
    softmax_in_place(scores);

    for (i, v) in values.iter().enumerate() {
        let weight = scores[i];
        for (o, vv) in out.iter_mut().zip(v.iter()) {
            *o += weight * vv;
        }
    }
}

pub fn matmul(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    c.fill(0.0);

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                matmul_avx2(c, a, b, m, k, n);
            }
            return;
        }
    }

    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_avx2(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::*;

    let lanes = 8;
    for i in 0..m {
        for p in 0..k {
            let a_broadcast = _mm256_set1_ps(a[i * k + p]);
            let mut j = 0;
            while j + lanes <= n {
                let c_ptr = c.as_mut_ptr().add(i * n + j);
                let b_ptr = b.as_ptr().add(p * n + j);

                let c_vec = _mm256_loadu_ps(c_ptr);
                let b_vec = _mm256_loadu_ps(b_ptr);
                let prod = _mm256_mul_ps(a_broadcast, b_vec);
                let sum = _mm256_add_ps(c_vec, prod);
                _mm256_storeu_ps(c_ptr, sum);
                j += lanes;
            }
            while j < n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
                j += 1;
            }
        }
    }
}
