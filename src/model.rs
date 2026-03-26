use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Defaulting to TinyLlama 1.1B dimensions
        Self {
            hidden_size: 2048,
            intermediate_size: 5632,
            vocab_size: 32000,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4, 
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

pub struct AttentionWeights<'a> {
    pub wq: Tensor<'a>,
    pub wk: Tensor<'a>,
    pub wv: Tensor<'a>,
    pub wo: Tensor<'a>,
}

pub struct MlpWeights<'a> {
    pub w1: Tensor<'a>, // Gate projection
    pub w2: Tensor<'a>, // Down projection
    pub w3: Tensor<'a>, // Up projection
}

pub struct TransformerBlockWeights<'a> {
    pub attention: AttentionWeights<'a>,
    pub mlp: MlpWeights<'a>,
    pub attention_norm: Tensor<'a>,
    pub ffn_norm: Tensor<'a>,
}

pub struct LlamaWeights<'a> {
    pub token_embeddings: Tensor<'a>,
    pub layers: Vec<TransformerBlockWeights<'a>>,
    pub final_norm: Tensor<'a>,
    pub lm_head: Tensor<'a>,
}

/// Manages the memory-mapped file and extracts tensor views
pub struct ModelLoader<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> ModelLoader<'a> {
    pub fn new(mmap_buffer: &'a [u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(mmap_buffer)
            .map_err(|e| anyhow!("Failed to parse safetensors: {:?}", e))?;
        Ok(Self { tensors })
    }

    fn get_tensor(&self, name: &str) -> Result<Tensor<'a>> {
        let view = self.tensors.tensor(name)
            .map_err(|_| anyhow!("Tensor {} not found", name))?;
        
        Ok(Tensor::new(view.data(), view.shape().to_vec()))
    }

    pub fn load_weights(&self, config: &LlamaConfig) -> Result<LlamaWeights<'a>> {
        println!("Mapping token embeddings...");
        let token_embeddings = self.get_tensor("model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        
        println!("Mapping {} transformer layers...", config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let attention = AttentionWeights {
                wq: self.get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i))?,
                wk: self.get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i))?,
                wv: self.get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i))?,
                wo: self.get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i))?,
            };

            let mlp = MlpWeights {
                w1: self.get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i))?,
                w2: self.get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i))?,
                w3: self.get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i))?,
            };

            layers.push(TransformerBlockWeights {
                attention,
                mlp,
                attention_norm: self.get_tensor(&format!("model.layers.{}.input_layernorm.weight", i))?,
                ffn_norm: self.get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i))?,
            });
        }

        println!("Mapping final norms and LM head...");
        let final_norm = self.get_tensor("model.norm.weight")?;
        let lm_head = self.get_tensor("lm_head.weight")?;

        Ok(LlamaWeights {
            token_embeddings,
            layers,
            final_norm,
            lm_head,
        })
    }
}