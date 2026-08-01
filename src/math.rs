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

    for (r, out_r) in out.iter_mut().enumerate() {
        let row = &weight_bf16[r * cols * 2..(r + 1) * cols * 2];
        *out_r = row
            .chunks_exact(2)
            .zip(x.iter())
            .map(|(b, xv)| half::bf16::from_le_bytes([b[0], b[1]]).to_f32() * xv)
            .sum();
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
    for (r, out_r) in out.iter_mut().enumerate() {
        *out_r = crate::simd::dot_naive(&weight[r * cols..(r + 1) * cols], x);
    }
}

/// Row-parallel matvec against a pre-packed f32 weight matrix.
///
/// Two levels of parallelism: Rayon splits the output rows across cores, and
/// each row's inner product runs on a vector kernel (AVX2+FMA / NEON, see
/// [`crate::simd`]).
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

    // Resolve the kernel once, outside the row loop, rather than re-checking
    // CPU features per row.
    #[cfg(target_arch = "x86_64")]
    if crate::simd::x86::available() {
        out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
            let base = r * cols;
            *out_r = unsafe { crate::simd::x86::dot(&weight[base..base + cols], x) };
        });
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
            let base = r * cols;
            *out_r = unsafe { crate::simd::neon::dot(&weight[base..base + cols], x) };
        });
        return;
    }

    #[allow(unreachable_code)]
    out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
        let base = r * cols;
        *out_r = crate::simd::dot_scalar(&weight[base..base + cols], x);
    });
}

/// Activation vectors fed through one weight load, in the batched matmul.
///
/// Read once. Larger tiles issue fewer loads per fused multiply-add but keep
/// `4 * tile` accumulators live, and spilling those hands the saved loads back
/// as stores. `PAGED_INFER_MATMUL_TILE` exists so the ceiling can be found by
/// measurement on a given machine; `1` turns tiling off.
fn matmul_tile() -> usize {
    static TILE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TILE.get_or_init(|| {
        std::env::var("PAGED_INFER_MATMUL_TILE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0)
            .unwrap_or(4)
    })
}

/// Batched matvec: project `batch` activation vectors through one weight matrix.
///
/// This is the whole point of batching decode. A matvec is memory-bound — it
/// streams the entire weight matrix to do one multiply-add per element — so
/// running `batch` sequences one at a time reads the weights `batch` times from
/// DRAM. Here each weight row is fetched once and reused across every sequence
/// in the batch, cutting weight traffic by a factor of `batch` while doing the
/// same arithmetic. For a 1.1B model that is ~4.2 GB per token that no longer
/// has to move.
///
/// The `batch` loop sits *inside* the row loop deliberately: `w_row` is
/// streamed from DRAM on the first sequence and served from L1 for the rest.
///
/// Output is `[row][batch]`-major, which is what lets Rayon hand each task a
/// disjoint `&mut` slice. Use [`transpose_rb_to_br`] to get back to
/// per-sequence contiguous vectors.
pub fn matmat_f32_weight_transposed_parallel(
    out_rb: &mut [f32],
    x_br: &[f32],
    weight: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) {
    assert_eq!(x_br.len(), batch * cols);
    assert_eq!(out_rb.len(), rows * batch);
    assert_eq!(weight.len(), rows * cols);

    // Each weight row is loaded once and multiplied into `TILE` activation
    // vectors at a time, instead of once per batch entry. The naive nesting
    // issues two loads to feed every fused multiply-add — one for the weight,
    // one for the activation — and a core that retires two loads and two FMAs
    // per cycle is then waiting on its load ports, not its arithmetic. That is
    // what caps batched decode around batch four or five.
    //
    // `TILE` is bounded by the register file: `dot_multi` keeps `4 * TILE`
    // accumulators live, and spilling them would hand back the loads it saved.
    // Four fits aarch64's 32 vector registers comfortably and AVX2's 16 tightly;
    // the remainder of a ragged batch falls through to narrower tiles.
    // `PAGED_INFER_MATMUL_TILE=1` disables tiling entirely, reproducing the
    // one-dot-per-batch-entry nesting this replaced — which is what makes an
    // honest A/B possible inside a single binary.
    let tile = matmul_tile();

    out_rb
        .par_chunks_mut(batch)
        .enumerate()
        .for_each(|(r, out_row)| {
            let w_row = &weight[r * cols..(r + 1) * cols];
            let mut b = 0;
            if tile >= 4 {
                while b + 4 <= batch {
                    let block = &x_br[b * cols..(b + 4) * cols];
                    out_row[b..b + 4]
                        .copy_from_slice(&crate::simd::dot_multi::<4>(w_row, block, cols));
                    b += 4;
                }
            }
            if tile >= 2 {
                while b + 2 <= batch {
                    let block = &x_br[b * cols..(b + 2) * cols];
                    out_row[b..b + 2]
                        .copy_from_slice(&crate::simd::dot_multi::<2>(w_row, block, cols));
                    b += 2;
                }
            }
            // Ragged remainder, and the whole batch when tiling is off.
            while b < batch {
                out_row[b] = crate::simd::dot(w_row, &x_br[b * cols..(b + 1) * cols]);
                b += 1;
            }
        })
}

/// [`matmat_f32_weight_transposed_parallel`] against int8 weights.
pub fn matmat_i8_weight_parallel(
    out_rb: &mut [f32],
    x_br: &[f32],
    weight: &[i8],
    scales: &[f32],
    batch: usize,
    rows: usize,
    cols: usize,
) {
    assert_eq!(x_br.len(), batch * cols);
    assert_eq!(out_rb.len(), rows * batch);
    assert_eq!(weight.len(), rows * cols);
    assert_eq!(scales.len(), rows);

    // Same tiling as the f32 path, and worth more here: an int8 weight element
    // must be loaded, sign-extended and converted to f32 before it can be
    // multiplied, and the untiled nesting repeats all three once per batch
    // entry. Tiling hoists the whole widening, not just the load.
    let tile = matmul_tile();

    out_rb
        .par_chunks_mut(batch)
        .enumerate()
        .for_each(|(r, out_row)| {
            let w_row = &weight[r * cols..(r + 1) * cols];
            let scale = scales[r];
            let mut b = 0;
            if tile >= 4 {
                while b + 4 <= batch {
                    let block = &x_br[b * cols..(b + 4) * cols];
                    let got = crate::simd::dot_i8_multi::<4>(w_row, block, cols);
                    for (slot, v) in out_row[b..b + 4].iter_mut().zip(got) {
                        *slot = v * scale;
                    }
                    b += 4;
                }
            }
            if tile >= 2 {
                while b + 2 <= batch {
                    let block = &x_br[b * cols..(b + 2) * cols];
                    let got = crate::simd::dot_i8_multi::<2>(w_row, block, cols);
                    for (slot, v) in out_row[b..b + 2].iter_mut().zip(got) {
                        *slot = v * scale;
                    }
                    b += 2;
                }
            }
            while b < batch {
                out_row[b] = crate::simd::dot_i8(w_row, &x_br[b * cols..(b + 1) * cols]) * scale;
                b += 1;
            }
        })
}

/// `[row][batch]` -> `[batch][row]`.
///
/// Costs `rows * batch` moves against the projection's `rows * cols * batch`
/// multiply-adds, so it is noise at any real hidden size.
pub fn transpose_rb_to_br(src_rb: &[f32], dst_br: &mut [f32], rows: usize, batch: usize) {
    assert_eq!(src_rb.len(), rows * batch);
    assert_eq!(dst_br.len(), rows * batch);
    for r in 0..rows {
        let row = &src_rb[r * batch..(r + 1) * batch];
        for (b, v) in row.iter().enumerate() {
            dst_br[b * rows + r] = *v;
        }
    }
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

/// Inner product, dispatched to the best kernel for this CPU.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot product dimensions must match");
    crate::simd::dot(a, b)
}

/// `out += weight * v`, the attention value accumulation.
#[inline]
pub fn axpy(out: &mut [f32], weight: f32, v: &[f32]) {
    debug_assert_eq!(out.len(), v.len());
    crate::simd::axpy(out, weight, v);
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

    // Dequantization folds into the per-row scale applied once at the end,
    // rather than per element inside the accumulation loop.
    #[cfg(target_arch = "x86_64")]
    if crate::simd::x86::available() {
        out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
            let base = r * cols;
            *out_r = unsafe { crate::simd::x86::dot_i8(&weight[base..base + cols], x) } * scales[r];
        });
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
            let base = r * cols;
            *out_r =
                unsafe { crate::simd::neon::dot_i8(&weight[base..base + cols], x) } * scales[r];
        });
        return;
    }

    #[allow(unreachable_code)]
    out.par_iter_mut().enumerate().for_each(|(r, out_r)| {
        let base = r * cols;
        *out_r = crate::simd::dot_i8_scalar(&weight[base..base + cols], x) * scales[r];
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
