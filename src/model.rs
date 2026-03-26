use anyhow::{Context, Result};
use safetensors::SafeTensors;

use crate::math::{
    apply_rope, matmul, matvec_f32_weight_transposed_parallel, pack_bf16_to_f32, rms_norm, swiglu,
};
use crate::memory::block_table::BlockTable;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub attention_window: Option<usize>,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            intermediate_size: 5632,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            attention_window: Some(256),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackedLinear {
    pub rows: usize,
    pub cols: usize,
    pub weight: Vec<f32>,
}

impl PackedLinear {
    fn from_tensor(t: &Tensor<'_>) -> Self {
        let shape = t.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self {
            rows,
            cols,
            weight: pack_bf16_to_f32(t.raw_bytes()),
        }
    }

    fn apply_parallel(&self, out: &mut [f32], x: &[f32]) {
        matvec_f32_weight_transposed_parallel(out, x, &self.weight, self.rows, self.cols);
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    pub rows: usize,
    pub cols: usize,
    pub weight: Vec<i8>,
    pub scales: Vec<f32>,
}

impl QuantizedLinear {
    pub fn from_packed(packed: &PackedLinear) -> Self {
        let (weight, scales) =
            crate::math::quantize_rows_i8(&packed.weight, packed.rows, packed.cols);
        Self {
            rows: packed.rows,
            cols: packed.cols,
            weight,
            scales,
        }
    }

    pub fn apply_parallel(&self, out: &mut [f32], x: &[f32]) {
        crate::math::matvec_i8_weight_parallel(
            out,
            x,
            &self.weight,
            &self.scales,
            self.rows,
            self.cols,
        );
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight.len() * std::mem::size_of::<i8>()
            + self.scales.len() * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
pub struct AttentionWeights {
    pub wq: PackedLinear,
    pub wk: PackedLinear,
    pub wv: PackedLinear,
    pub wo: PackedLinear,
}

#[derive(Debug, Clone)]
pub struct FeedForwardWeights {
    pub w1: PackedLinear,
    pub w2: PackedLinear,
    pub w3: PackedLinear,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attention_norm: Vec<f32>,
    pub attention: AttentionWeights,
    pub ffn_norm: Vec<f32>,
    pub feed_forward: FeedForwardWeights,
}

#[derive(Debug, Clone)]
pub struct LlamaWeights<'a> {
    pub token_embeddings: Tensor<'a>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: PackedLinear,
}

pub struct ModelLoader<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> ModelLoader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes).context("failed to parse safetensors")?;
        Ok(Self { tensors })
    }

    pub fn load_weights(&self, config: &LlamaConfig) -> Result<LlamaWeights<'a>> {
        let token_embeddings = self.tensor("model.embed_tokens.weight")?;
        let final_norm = pack_bf16_to_f32(self.tensor("model.norm.weight")?.raw_bytes());
        let lm_head_t = self
            .tensor("lm_head.weight")
            .or_else(|_| self.tensor("model.embed_tokens.weight"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let attn_norm = self.tensor(&format!("{prefix}.input_layernorm.weight"))?;
            let ffn_norm = self.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let wq = self.tensor(&format!("{prefix}.self_attn.q_proj.weight"))?;
            let wk = self.tensor(&format!("{prefix}.self_attn.k_proj.weight"))?;
            let wv = self.tensor(&format!("{prefix}.self_attn.v_proj.weight"))?;
            let wo = self.tensor(&format!("{prefix}.self_attn.o_proj.weight"))?;
            let w1 = self.tensor(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let w2 = self.tensor(&format!("{prefix}.mlp.down_proj.weight"))?;
            let w3 = self.tensor(&format!("{prefix}.mlp.up_proj.weight"))?;

            layers.push(LayerWeights {
                attention_norm: pack_bf16_to_f32(attn_norm.raw_bytes()),
                attention: AttentionWeights {
                    wq: PackedLinear::from_tensor(&wq),
                    wk: PackedLinear::from_tensor(&wk),
                    wv: PackedLinear::from_tensor(&wv),
                    wo: PackedLinear::from_tensor(&wo),
                },
                ffn_norm: pack_bf16_to_f32(ffn_norm.raw_bytes()),
                feed_forward: FeedForwardWeights {
                    w1: PackedLinear::from_tensor(&w1),
                    w2: PackedLinear::from_tensor(&w2),
                    w3: PackedLinear::from_tensor(&w3),
                },
            });
        }

        Ok(LlamaWeights {
            token_embeddings,
            layers,
            final_norm,
            lm_head: PackedLinear::from_tensor(&lm_head_t),
        })
    }

    fn tensor(&self, name: &str) -> Result<Tensor<'a>> {
        let view = self
            .tensors
            .tensor(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        Ok(Tensor::new(view.data(), view.shape().to_vec()))
    }
}

fn bf16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

fn kv_cache_index(
    layer: usize,
    phys_block: usize,
    token_offset: usize,
    kv_head: usize,
    is_value: bool,
    block_size: usize,
    head_dim: usize,
    num_kv_heads: usize,
    total_blocks: usize,
) -> usize {
    let kv_or_v = if is_value { 1 } else { 0 };
    ((((layer * total_blocks + phys_block) * block_size + token_offset) * num_kv_heads + kv_head)
        * 2
        + kv_or_v)
        * head_dim
}

impl<'a> LlamaWeights<'a> {
    pub fn weight_bytes_f32(&self) -> usize {
        let mut total = 0;
        for layer in &self.layers {
            total += layer.attention.wq.weight.len() * 4;
            total += layer.attention.wk.weight.len() * 4;
            total += layer.attention.wv.weight.len() * 4;
            total += layer.attention.wo.weight.len() * 4;
            total += layer.feed_forward.w1.weight.len() * 4;
            total += layer.feed_forward.w2.weight.len() * 4;
            total += layer.feed_forward.w3.weight.len() * 4;
        }
        total += self.lm_head.weight.len() * 4;
        total
    }

    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32],
        block_size: usize,
    ) -> Vec<f32> {
        let hidden = config.hidden_size;
        let head_dim = hidden / config.num_attention_heads;
        let kv_dim = config.num_key_value_heads * head_dim;
        let num_heads = config.num_attention_heads;
        let kv_group = num_heads / config.num_key_value_heads;
        let total_blocks = kv_cache.len()
            / (config.num_hidden_layers * block_size * config.num_key_value_heads * 2 * head_dim);
        let attn_window = config.attention_window.unwrap_or(pos + 1);

        let token = (token_id as usize) % config.vocab_size;
        let embed_bytes =
            &self.token_embeddings.raw_bytes()[token * hidden * 2..(token + 1) * hidden * 2];
        let mut x = bf16_bytes_to_f32(embed_bytes);

        let mut xb = vec![0.0; hidden];
        let mut attn_out = vec![0.0; hidden];
        let mut proj_out = vec![0.0; hidden];

        let mut q = vec![0.0; hidden];
        let mut k = vec![0.0; kv_dim];
        let mut v = vec![0.0; kv_dim];

        let mut ff_gate = vec![0.0; config.intermediate_size];
        let mut ff_up = vec![0.0; config.intermediate_size];
        let mut ff_down = vec![0.0; hidden];
        let mut attn_scores = vec![0.0; pos + 1];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xb.copy_from_slice(&x);
            rms_norm(&mut xb, &layer.attention_norm, config.rms_norm_eps);

            layer.attention.wq.apply_parallel(&mut q, &xb);
            layer.attention.wk.apply_parallel(&mut k, &xb);
            layer.attention.wv.apply_parallel(&mut v, &xb);

            for h in 0..num_heads {
                let q_start = h * head_dim;
                let kv_h = h / kv_group;
                let k_start = kv_h * head_dim;
                apply_rope(
                    &mut q[q_start..q_start + head_dim],
                    &mut k[k_start..k_start + head_dim],
                    pos,
                    head_dim,
                    config.rope_theta,
                );
            }

            if let Some((phys_block, offset)) = block_table.get_physical_location(pos, block_size) {
                for kv_h in 0..config.num_key_value_heads {
                    let k_start = kv_h * head_dim;
                    let k_cache_idx = kv_cache_index(
                        layer_idx,
                        phys_block.index,
                        offset,
                        kv_h,
                        false,
                        block_size,
                        head_dim,
                        config.num_key_value_heads,
                        total_blocks,
                    );
                    let v_cache_idx = kv_cache_index(
                        layer_idx,
                        phys_block.index,
                        offset,
                        kv_h,
                        true,
                        block_size,
                        head_dim,
                        config.num_key_value_heads,
                        total_blocks,
                    );
                    kv_cache[k_cache_idx..k_cache_idx + head_dim]
                        .copy_from_slice(&k[k_start..k_start + head_dim]);
                    kv_cache[v_cache_idx..v_cache_idx + head_dim]
                        .copy_from_slice(&v[k_start..k_start + head_dim]);
                }
            }

            attn_out.fill(0.0);
            for h in 0..num_heads {
                let q_start = h * head_dim;
                let kv_h = h / kv_group;
                let out_slice = &mut attn_out[q_start..q_start + head_dim];
                attn_scores.fill(0.0);

                let start_t = (pos + 1).saturating_sub(attn_window);
                for t in start_t..=pos {
                    if let Some((phys_block, token_offset)) =
                        block_table.get_physical_location(t, block_size)
                    {
                        let k_idx = kv_cache_index(
                            layer_idx,
                            phys_block.index,
                            token_offset,
                            kv_h,
                            false,
                            block_size,
                            head_dim,
                            config.num_key_value_heads,
                            total_blocks,
                        );
                        let k_cached = &kv_cache[k_idx..k_idx + head_dim];
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += q[q_start + d] * k_cached[d];
                        }
                        attn_scores[t] = score / (head_dim as f32).sqrt();
                    }
                }

                crate::math::softmax_in_place(&mut attn_scores[start_t..=pos]);

                for t in start_t..=pos {
                    if let Some((phys_block, token_offset)) =
                        block_table.get_physical_location(t, block_size)
                    {
                        let v_idx = kv_cache_index(
                            layer_idx,
                            phys_block.index,
                            token_offset,
                            kv_h,
                            true,
                            block_size,
                            head_dim,
                            config.num_key_value_heads,
                            total_blocks,
                        );
                        let v_cached = &kv_cache[v_idx..v_idx + head_dim];
                        let wt = attn_scores[t];
                        for d in 0..head_dim {
                            out_slice[d] += wt * v_cached[d];
                        }
                    }
                }
            }

            layer.attention.wo.apply_parallel(&mut proj_out, &attn_out);
            for i in 0..hidden {
                x[i] += proj_out[i];
            }

            xb.copy_from_slice(&x);
            rms_norm(&mut xb, &layer.ffn_norm, config.rms_norm_eps);

            layer.feed_forward.w1.apply_parallel(&mut ff_gate, &xb);
            layer.feed_forward.w3.apply_parallel(&mut ff_up, &xb);
            swiglu(&mut ff_gate, &ff_up);
            layer.feed_forward.w2.apply_parallel(&mut ff_down, &ff_gate);
            for i in 0..hidden {
                x[i] += ff_down[i];
            }
        }

        rms_norm(&mut x, &self.final_norm, config.rms_norm_eps);

        let mut logits = vec![0.0; config.vocab_size];
        self.lm_head.apply_parallel(&mut logits, &x);

        let _ = matmul as fn(&mut [f32], &[f32], &[f32], usize, usize, usize);
        logits
    }
}
