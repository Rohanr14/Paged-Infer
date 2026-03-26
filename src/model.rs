use crate::tensor::Tensor;

/// Configuration parameters for the Llama 3.2 1B model.
/// These dictate the shapes of our tensors during memory mapping.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize, // Size of the SwiGLU hidden layer
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // GQA: Usually less than attention_heads
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // These are the specific architectural dimensions for Llama 3.2 1B.
        // Once we parse the model's config.json, we will populate this dynamically.
        Self {
            hidden_size: 2048,
            intermediate_size: 8192,
            vocab_size: 128256,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8, 
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            max_position_embeddings: 131072,
        }
    }
}

/// Attention mechanism weights mapping.
pub struct AttentionWeights {
    pub wq: Tensor<f32>, // Query projection
    pub wk: Tensor<f32>, // Key projection
    pub wv: Tensor<f32>, // Value projection
    pub wo: Tensor<f32>, // Output projection
}

/// Feed-forward network weights mapping.
/// Llama uses a SwiGLU architecture, requiring three weight matrices.
pub struct MlpWeights {
    pub w1: Tensor<f32>, // Gate projection
    pub w2: Tensor<f32>, // Down projection
    pub w3: Tensor<f32>, // Up projection
}

/// A single Transformer layer block containing Attention and MLP.
pub struct TransformerBlockWeights {
    pub attention: AttentionWeights,
    pub mlp: MlpWeights,
    pub attention_norm: Tensor<f32>, // RMSNorm weight before Attention
    pub ffn_norm: Tensor<f32>,       // RMSNorm weight before MLP
}

/// The root structure holding the entire Llama 3.2 Model's weights.
pub struct LlamaWeights {
    pub token_embeddings: Tensor<f32>,
    pub layers: Vec<TransformerBlockWeights>,
    pub final_norm: Tensor<f32>,
    pub lm_head: Tensor<f32>, // Vocabulary projection for next-token prediction
}