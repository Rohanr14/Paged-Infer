use paged_infer::math::{
    matvec_bf16_weight_transposed, matvec_f32_weight_transposed_parallel, pack_bf16_to_f32,
    softmax_in_place,
};
use std::time::Instant;

fn baseline_convert_then_matvec(
    out: &mut [f32],
    x: &[f32],
    w_bf16: &[u8],
    rows: usize,
    cols: usize,
) {
    let w: Vec<f32> = w_bf16
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();
    for r in 0..rows {
        let mut acc = 0.0;
        let base = r * cols;
        for c in 0..cols {
            acc += x[c] * w[base + c];
        }
        out[r] = acc;
    }
}

fn baseline_attention_alloc(head_dim: usize, seq_len: usize, iters: usize) -> f64 {
    let q = vec![0.1f32; head_dim];
    let k = vec![0.1f32; head_dim];
    let start = Instant::now();
    for _ in 0..iters {
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let mut s = 0.0;
            for d in 0..head_dim {
                s += q[d] * k[d];
            }
            scores[t] = s;
        }
        softmax_in_place(&mut scores);
    }
    start.elapsed().as_secs_f64()
}

fn optimized_attention_reuse(head_dim: usize, seq_len: usize, iters: usize) -> f64 {
    let q = vec![0.1f32; head_dim];
    let k = vec![0.1f32; head_dim];
    let mut scores = vec![0.0f32; seq_len];
    let start = Instant::now();
    for _ in 0..iters {
        scores.fill(0.0);
        for t in 0..seq_len {
            let mut s = 0.0;
            for d in 0..head_dim {
                s += q[d] * k[d];
            }
            scores[t] = s;
        }
        softmax_in_place(&mut scores);
    }
    start.elapsed().as_secs_f64()
}

fn main() {
    let cols = 2048;
    let rows = 2048;
    let iters = 20;

    let x = vec![0.01f32; cols];
    let mut out = vec![0.0f32; rows];
    let mut w_bf16 = vec![0u8; rows * cols * 2];
    for i in 0..rows * cols {
        let b = half::bf16::from_f32(((i % 97) as f32) * 0.001).to_le_bytes();
        w_bf16[i * 2] = b[0];
        w_bf16[i * 2 + 1] = b[1];
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        baseline_convert_then_matvec(&mut out, &x, &w_bf16, rows, cols);
    }
    let baseline = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..iters {
        matvec_bf16_weight_transposed(&mut out, &x, &w_bf16, rows, cols);
    }
    let stream_bf16 = t1.elapsed().as_secs_f64();

    let packed = pack_bf16_to_f32(&w_bf16);
    let t2 = Instant::now();
    for _ in 0..iters {
        matvec_f32_weight_transposed_parallel(&mut out, &x, &packed, rows, cols);
    }
    let packed_parallel = t2.elapsed().as_secs_f64();

    let attn_alloc = baseline_attention_alloc(64, 1024, 200);
    let attn_reuse = optimized_attention_reuse(64, 1024, 200);

    println!("MATVEC baseline(convert+matvec): {:.4}s", baseline);
    println!("MATVEC optimized(stream bf16):  {:.4}s", stream_bf16);
    println!("MATVEC optimized(packed+parallel): {:.4}s", packed_parallel);
    println!(
        "MATVEC speedup (stream bf16): {:.2}x",
        baseline / stream_bf16.max(1e-9)
    );
    println!(
        "MATVEC speedup (packed+parallel): {:.2}x",
        baseline / packed_parallel.max(1e-9)
    );
    println!("ATTN baseline(per-iter alloc):  {:.4}s", attn_alloc);
    println!("ATTN optimized(buffer reuse):   {:.4}s", attn_reuse);
    println!("ATTN speedup: {:.2}x", attn_alloc / attn_reuse.max(1e-9));
}
