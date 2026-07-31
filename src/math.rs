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

/// Which pairs of dimensions a rotary embedding rotates together.
///
/// The two conventions are numerically different and are *not* interchangeable:
/// a checkpoint stores its Q/K projections already arranged for one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeStyle {
    /// HuggingFace / GPT-NeoX `rotate_half`: dimension `j` rotates against
    /// `j + head_dim/2`.
    ///
    /// This is the convention every HF `.safetensors` Llama checkpoint needs.
    /// `convert_llama_weights_to_hf.py` permutes `q_proj` and `k_proj` at
    /// conversion time precisely so that `rotate_half` reproduces what the
    /// original interleaved kernel computed. Feeding HF weights through the
    /// interleaved path silently corrupts every attention score.
    #[default]
    Neox,
    /// GPT-J / original Meta layout: adjacent dimensions `(2i, 2i+1)` rotate
    /// together. Correct only for un-permuted original-format checkpoints.
    Interleaved,
}

/// Precompute `(cos, sin)` for one position: `head_dim / 2` entries, indexed by
/// rotation pair rather than by dimension.
///
/// Worth hoisting — a naive implementation recomputes these inside the head
/// loop of every layer, which for TinyLlama is ~25k transcendental calls per
/// token instead of 32.
pub fn rope_table(pos: usize, head_dim: usize, rope_theta: f32, cos: &mut [f32], sin: &mut [f32]) {
    let half = head_dim / 2;
    assert_eq!(cos.len(), half);
    assert_eq!(sin.len(), half);
    for j in 0..half {
        let freq = 1.0 / rope_theta.powf((2 * j) as f32 / head_dim as f32);
        let (s, c) = ((pos as f32) * freq).sin_cos();
        cos[j] = c;
        sin[j] = s;
    }
}

/// Rotate a single head's vector in place using a precomputed table.
pub fn rope_rotate(x: &mut [f32], cos: &[f32], sin: &[f32], style: RopeStyle) {
    let half = x.len() / 2;
    debug_assert_eq!(cos.len(), half);
    match style {
        RopeStyle::Neox => {
            for j in 0..half {
                let (c, s) = (cos[j], sin[j]);
                let a = x[j];
                let b = x[j + half];
                x[j] = a * c - b * s;
                x[j + half] = b * c + a * s;
            }
        }
        RopeStyle::Interleaved => {
            for j in 0..half {
                let (c, s) = (cos[j], sin[j]);
                let a = x[2 * j];
                let b = x[2 * j + 1];
                x[2 * j] = a * c - b * s;
                x[2 * j + 1] = b * c + a * s;
            }
        }
    }
}

/// Rotate a query head and a key head that share the same rotation table.
///
/// Under grouped-query attention several query heads map onto one key head, so
/// callers must not drive this from the query-head loop — that would rotate the
/// shared key head once per group member. Rotate queries and keys separately
/// with [`rope_rotate`]; see `LlamaWeights::forward`.
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    rope_theta: f32,
    style: RopeStyle,
) {
    let half = head_dim / 2;
    let mut cos = vec![0.0_f32; half];
    let mut sin = vec![0.0_f32; half];
    rope_table(pos, head_dim, rope_theta, &mut cos, &mut sin);
    rope_rotate(q, &cos, &sin, style);
    rope_rotate(k, &cos, &sin, style);
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `out += weight * v`, the attention value accumulation.
#[inline]
pub fn axpy(out: &mut [f32], weight: f32, v: &[f32]) {
    debug_assert_eq!(out.len(), v.len());
    for (o, x) in out.iter_mut().zip(v.iter()) {
        *o += weight * x;
    }
}

pub fn softmax_in_place(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_v = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Every entry masked out: `exp(-inf - -inf)` is NaN, so bail to all-zero
    // weights instead. Contributes nothing to the attention output, which is
    // the right answer when there is nothing to attend to.
    if !max_v.is_finite() {
        x.fill(0.0);
        return;
    }
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

/// Quantize a row-major f32 weight matrix to per-row symmetric int8.
/// Returns (quantized_weights: Vec<i8>, scales: Vec<f32>) where scales[r] = max_abs_of_row / 127.0
pub fn quantize_rows_i8(weight: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut quant = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &weight[r * cols..(r + 1) * cols];
        let max_abs = row.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.push(scale);
        for &w in row {
            quant.push((w / scale).round().clamp(-127.0, 127.0) as i8);
        }
    }
    (quant, scales)
}

/// Parallel matvec with int8 weights (weight-only quantization, f32 activations).
/// weight[r * cols + c] is i8, dequantized on-the-fly: w_f32 = weight[r][c] as f32 * scales[r]
pub fn matvec_i8_weight_parallel(
    out: &mut [f32],
    x: &[f32],
    weight: &[i8],
    scales: &[f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(weight.len(), rows * cols);
    assert_eq!(scales.len(), rows);

    out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
        let base = r * cols;
        let scale = scales[r];
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += weight[base + c] as f32 * x[c];
        }
        *out_r = acc * scale;
    });
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
