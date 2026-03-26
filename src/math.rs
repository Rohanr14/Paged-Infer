use crate::tensor::Tensor;

/// Root Mean Square Normalization (RMSNorm)
/// Llama 3 uses RMSNorm instead of standard LayerNorm.
pub fn rms_norm(x: &mut [f32], weight: &[f32], epsilon: f32) {
    let size = x.len();
    assert_eq!(size, weight.len(), "Input and weight size mismatch for RMSNorm");

    // Calculate the sum of squares
    let mut ss = 0.0;
    for &val in x.iter() {
        ss += val * val;
    }

    // Calculate the mean of squares
    ss /= size as f32;

    // Calculate the inverse square root
    let inv_norm = 1.0 / (ss + epsilon).sqrt();

    // Normalize and scale
    for i in 0..size {
        x[i] = x[i] * inv_norm * weight[i];
    }
}

/// SiLU (Sigmoid Linear Unit) Activation Function
/// Also known as Swish. Llama uses this within the SwiGLU block.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU Activation
/// Llama 3's feed-forward network uses SwiGLU: (xW1 * SiLU(xW1)) * xW3
/// Assuming x is already the result of xW1, and we have xW3.
pub fn swiglu(x: &mut [f32], x_w3: &[f32]) {
    assert_eq!(x.len(), x_w3.len(), "SwiGLU input sizes must match");
    for i in 0..x.len() {
        x[i] = silu(x[i]) * x_w3[i];
    }
}

/// Rotary Positional Embeddings (RoPE)
/// Applies rotation to the Query and Key vectors based on their token position.
pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, rope_theta: f32) {
    // RoPE applies to pairs of dimensions
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / rope_theta.powf((i as f32) / (head_dim as f32));
        let val = (pos as f32) * freq;
        let (sin, cos) = val.sin_cos();

        // Apply to Query
        let q0 = q[i];
        let q1 = q[i + 1];
        q[i] = q0 * cos - q1 * sin;
        q[i + 1] = q0 * sin + q1 * cos;

        // Apply to Key
        let k0 = k[i];
        let k1 = k[i + 1];
        k[i] = k0 * cos - k1 * sin;
        k[i + 1] = k0 * sin + k1 * cos;
    }
}

/// Basic Matrix Multiplication (GEMM): C = A * B
/// In inference, this is mostly Vector-Matrix multiplication (e.g., [1, d] * [d, h] = [1, h])
/// A: input vector/matrix (m x k)
/// B: weight matrix (k x n)
/// C: output vector/matrix (m x n)
pub fn matmul(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    // Initialize C to zero
    c.fill(0.0);

    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                // b is assumed to be stored in row-major order: [k, n]
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}