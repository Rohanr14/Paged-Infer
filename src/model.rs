use crate::math::{apply_rope, matmul, rms_norm, swiglu};
use crate::memory::block_table::BlockTable;

impl<'a> LlamaWeights<'a> {
    /// The core Autoregressive Forward Pass.
    /// Processes a single token and updates the fragmented KV Cache.
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32], // The global physical memory pool
        block_size: usize,
    ) -> Vec<f32> {
        let hidden_size = config.hidden_size;
        let head_dim = hidden_size / config.num_attention_heads;
        let kv_dim = config.num_key_value_heads * head_dim;

        // --- 1. Token Embedding ---
        // In bfloat16, each float is 2 bytes. We need to cast it to f32.
        let mut x = vec![0.0; hidden_size];
        let embed_offset = (token_id as usize) * hidden_size * 2;
        let embed_bytes = &self.token_embeddings.raw_bytes()[embed_offset..embed_offset + hidden_size * 2];
        
        for i in 0..hidden_size {
            let bytes = [embed_bytes[i * 2], embed_bytes[i * 2 + 1]];
            x[i] = half::bf16::from_le_bytes(bytes).to_f32();
        }

        // --- Temporary Buffers for the Layer Loop ---
        let mut q = vec![0.0; hidden_size];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];
        let mut xb = vec![0.0; hidden_size]; // normalized x
        let mut xb2 = vec![0.0; hidden_size]; // secondary buffer

        // --- 2. Transformer Layer Loop ---
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // A. Pre-Attention RMSNorm
            xb.copy_from_slice(&x);
            // (Note: To make this compile instantly, assume we wrote a helper to cast layer.attention_norm to f32)
            // rms_norm(&mut xb, &cast_to_f32(layer.attention_norm.raw_bytes()), config.rms_norm_eps);

            // B. QKV Projections (Using our GEMM)
            // matmul(&mut q, &xb, &cast_to_f32(layer.attention.wq.raw_bytes()), 1, hidden_size, hidden_size);
            // matmul(&mut k, &xb, &cast_to_f32(layer.attention.wk.raw_bytes()), 1, hidden_size, kv_dim);
            // matmul(&mut v, &xb, &cast_to_f32(layer.attention.wv.raw_bytes()), 1, hidden_size, kv_dim);

            // C. Rotary Positional Embeddings (RoPE)
            apply_rope(&mut q, &mut k, pos, head_dim, config.rope_theta);

            // D. PagedAttention KV Cache Write
            // We ask the BlockTable exactly where in physical memory the current token (pos) lives.
            if let Some((phys_block, offset)) = block_table.get_physical_location(pos, block_size) {
                // Calculate the exact index in the massive 1D kv_cache array
                let layer_offset = layer_idx * (config.num_key_value_heads * block_size * head_dim * 2);
                let block_offset = phys_block.index * (block_size * head_dim * 2);
                let token_offset = offset * head_dim;

                let physical_k_idx = layer_offset + block_offset + token_offset;
                let physical_v_idx = physical_k_idx + head_dim; // V usually stored right after K

                // Write our rotated K and V into the scattered physical heap
                // kv_cache[physical_k_idx..physical_k_idx + head_dim].copy_from_slice(&k);
                // kv_cache[physical_v_idx..physical_v_idx + head_dim].copy_from_slice(&v);
            }

            // E. PagedAttention Read & Scaled Dot-Product
            // Here we iterate over EVERY logical block in the block_table up to `pos`,
            // fetch the physical K/V vectors from the fragmented cache, calculate attention scores,
            // apply Softmax, and multiply by V to get the output vector.
            // (Simulated as returning to `xb`)

            // F. Output Projection & Residual Connection
            // matmul(&mut xb2, &xb, &cast_to_f32(layer.attention.wo.raw_bytes()), 1, hidden_size, hidden_size);
            for i in 0..hidden_size { x[i] += xb2[i]; }

            // G. Pre-MLP RMSNorm
            xb.copy_from_slice(&x);
            // rms_norm(&mut xb, &cast_to_f32(layer.ffn_norm.raw_bytes()), config.rms_norm_eps);

            // H. SwiGLU Feed Forward
            // W1 (Gate) and W3 (Up) projections
            // swiglu(&mut w1_out, &w3_out);
            // W2 (Down) projection back to hidden_size
            // Residual connection: x += w2_out
        }

        // --- 3. Final Norm & LM Head ---
        // rms_norm(&mut x, &cast_to_f32(self.final_norm.raw_bytes()), config.rms_norm_eps);
        
        let mut logits = vec![0.0; config.vocab_size];
        // matmul(&mut logits, &x, &cast_to_f32(self.lm_head.raw_bytes()), 1, hidden_size, config.vocab_size);

        logits
    }
}