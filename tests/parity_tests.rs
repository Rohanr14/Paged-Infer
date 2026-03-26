use paged_infer::math::paged_attention;

fn naive_attention(q: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<f32> {
    let scale = 1.0 / (q.len() as f32).sqrt();
    let mut scores = vec![0.0; keys.len()];
    for (i, k) in keys.iter().enumerate() {
        scores[i] = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for s in &mut scores {
        *s = (*s - max).exp();
        sum += *s;
    }
    for s in &mut scores {
        *s /= sum;
    }

    let mut out = vec![0.0; q.len()];
    for (t, v) in values.iter().enumerate() {
        for d in 0..q.len() {
            out[d] += scores[t] * v[d];
        }
    }
    out
}

#[test]
fn test_paged_attention_parity_randomized() {
    let head_dim = 16;
    let seq_len = 32;

    let q: Vec<f32> = (0..head_dim)
        .map(|i| ((i * 17) as f32).sin() * 0.3)
        .collect();
    let keys: Vec<Vec<f32>> = (0..seq_len)
        .map(|t| {
            (0..head_dim)
                .map(|d| (((t * 31 + d * 13) as f32).cos()) * 0.2)
                .collect()
        })
        .collect();
    let values: Vec<Vec<f32>> = (0..seq_len)
        .map(|t| {
            (0..head_dim)
                .map(|d| (((t * 7 + d * 19) as f32).sin()) * 0.4)
                .collect()
        })
        .collect();

    let key_refs: Vec<&[f32]> = keys.iter().map(|k| k.as_slice()).collect();
    let value_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

    let mut scores = vec![0.0; seq_len];
    let mut out = vec![0.0; head_dim];
    paged_attention(&q, &key_refs, &value_refs, &mut scores, &mut out);

    let naive = naive_attention(&q, &keys, &values);
    for (a, b) in out.iter().zip(naive.iter()) {
        assert!((a - b).abs() < 1e-5, "paged and naive attention diverged");
    }
}
