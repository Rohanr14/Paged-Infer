use anyhow::{Context, Result};
use safetensors::SafeTensors;

use rayon::prelude::*;

use crate::gpu::{GpuContext, GpuLinear};
use crate::math::{
    matvec_f32_weight_transposed_parallel, pack_bf16_to_f32, rms_norm, rope_rotate, rope_table,
    swiglu, RopeStyle,
};
use crate::memory::block_table::BlockTable;
use crate::memory::layout::KvLayout;
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
    /// Optional sliding-window attention.
    ///
    /// `None` is full causal attention, which is what Llama actually specifies.
    /// Setting a window is a *deliberate* accuracy-for-latency trade: it caps
    /// per-token attention cost at O(window) instead of O(context), at the cost
    /// of the model no longer seeing anything older. Off by default so results
    /// are the model's, not the engine's.
    pub attention_window: Option<usize>,
    pub rope_style: RopeStyle,
    /// How to store projection weights. See [`Quantization`].
    pub quantization: Quantization,
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
            attention_window: None,
            rope_style: RopeStyle::Neox,
            quantization: Quantization::F32,
        }
    }
}

impl LlamaConfig {
    /// Read the architecture from a HuggingFace `config.json`.
    ///
    /// Every HF checkpoint ships one next to `model.safetensors`, so the shape
    /// of the model should come from the checkpoint rather than from a
    /// hardcoded default that silently mismatches — the failure mode otherwise
    /// is a missing-tensor error at load, or worse, a wrong-shaped load that
    /// runs and produces noise.
    ///
    /// Fields absent from the file keep their [`Default`] value.
    pub fn from_hf_config(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;

        let default = Self::default();
        let usize_at = |key: &str, fallback: usize| -> usize {
            json[key].as_u64().map_or(fallback, |v| v as usize)
        };
        let f32_at =
            |key: &str, fallback: f32| -> f32 { json[key].as_f64().map_or(fallback, |v| v as f32) };

        let num_attention_heads = usize_at("num_attention_heads", default.num_attention_heads);
        Ok(Self {
            hidden_size: usize_at("hidden_size", default.hidden_size),
            num_hidden_layers: usize_at("num_hidden_layers", default.num_hidden_layers),
            num_attention_heads,
            // Multi-head checkpoints omit this; it then equals the query heads.
            num_key_value_heads: usize_at("num_key_value_heads", num_attention_heads),
            intermediate_size: usize_at("intermediate_size", default.intermediate_size),
            vocab_size: usize_at("vocab_size", default.vocab_size),
            rms_norm_eps: f32_at("rms_norm_eps", default.rms_norm_eps),
            rope_theta: f32_at("rope_theta", default.rope_theta),
            // Not architecture: these are engine policy, so they keep defaults
            // and are set by the caller.
            attention_window: default.attention_window,
            rope_style: default.rope_style,
            quantization: default.quantization,
        })
    }

    /// Load the config sitting beside a checkpoint, or fall back to defaults
    /// when the checkpoint ships without one.
    pub fn beside_checkpoint(model_path: impl AsRef<std::path::Path>) -> Self {
        let candidate = model_path.as_ref().with_file_name("config.json");
        Self::from_hf_config(&candidate).unwrap_or_default()
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// How many query heads share one key/value head.
    #[inline]
    pub fn kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// Derive the physical cache layout for a given block count.
    pub fn kv_layout(&self, num_blocks: usize, block_size: usize) -> KvLayout {
        KvLayout::new(
            self.num_hidden_layers,
            num_blocks,
            block_size,
            self.num_key_value_heads,
            self.head_dim(),
        )
    }

    /// Recover the layout implied by an already-allocated cache.
    pub fn kv_layout_for_cache(&self, cache_len: usize, block_size: usize) -> KvLayout {
        let per_block =
            self.num_hidden_layers * block_size * self.num_key_value_heads * 2 * self.head_dim();
        self.kv_layout(cache_len / per_block.max(1), block_size)
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

/// How projection weights are stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quantization {
    /// Widen bf16 to f32 once at load. Fastest per element, but 2x the
    /// checkpoint's size on disk and 4x an int8 copy.
    #[default]
    F32,
    /// Per-row symmetric int8, with an f32 scale per row.
    ///
    /// A matvec is memory-bound, so cutting weight traffic 4x is a throughput
    /// win as well as a footprint win — TinyLlama's projections drop from about
    /// 4.2 GB to 1.1 GB, which is the difference between fitting in a laptop's
    /// RAM and not.
    Int8,
}

/// One projection matrix, in whichever representation was requested at load.
#[derive(Debug, Clone)]
pub enum Projection {
    F32(PackedLinear),
    Int8(QuantizedLinear),
}

impl Projection {
    fn from_tensor(t: &Tensor<'_>, quantization: Quantization) -> Self {
        let packed = PackedLinear::from_tensor(t);
        match quantization {
            Quantization::F32 => Projection::F32(packed),
            Quantization::Int8 => Projection::Int8(QuantizedLinear::from_packed(&packed)),
        }
    }

    #[inline]
    pub fn apply_parallel(&self, out: &mut [f32], x: &[f32]) {
        match self {
            Projection::F32(w) => w.apply_parallel(out, x),
            Projection::Int8(w) => w.apply_parallel(out, x),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Projection::F32(w) => w.rows,
            Projection::Int8(w) => w.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Projection::F32(w) => w.cols,
            Projection::Int8(w) => w.cols,
        }
    }

    pub fn weight_bytes(&self) -> usize {
        match self {
            Projection::F32(w) => w.weight.len() * std::mem::size_of::<f32>(),
            Projection::Int8(w) => w.weight_bytes(),
        }
    }

    /// The f32 weights, if this projection kept them. The GPU path needs them:
    /// the WGSL kernel reads `array<vec4<f32>>`, so int8 weights would have to
    /// be dequantized before upload, defeating the point.
    pub fn f32_weights(&self) -> Option<&[f32]> {
        match self {
            Projection::F32(w) => Some(&w.weight),
            Projection::Int8(_) => None,
        }
    }
}

// ── GPU-resident projection weights ──────────────────────────────────────────

/// GPU copies of all seven projection matrices for one transformer layer.
pub struct GpuLayerWeights {
    pub wq: GpuLinear,
    pub wk: GpuLinear,
    pub wv: GpuLinear,
    pub wo: GpuLinear,
    pub w1: GpuLinear,
    pub w2: GpuLinear,
    pub w3: GpuLinear,
}

/// Holds a `GpuContext` plus all GPU-resident projection weights for a full
/// forward pass.  Create once via `GpuForwardContext::from_weights`, then pass
/// as `Some(&ctx)` to `LlamaWeights::forward`.
pub struct GpuForwardContext {
    pub ctx: GpuContext,
    pub layers: Vec<GpuLayerWeights>,
    pub lm_head: GpuLinear,
}

impl GpuForwardContext {
    /// Upload every projection weight to the GPU.
    ///
    /// Returns `None` when there is no GPU adapter, or when the weights were
    /// loaded as int8 — the WGSL kernel reads `array<vec4<f32>>`, so int8
    /// weights would have to be dequantized before upload, which would cost more
    /// host memory than the f32 path it was meant to replace. Load with
    /// [`Quantization::F32`] to use the GPU.
    pub fn from_weights(weights: &LlamaWeights<'_>) -> Option<Self> {
        let ctx = GpuContext::new()?;

        let upload = |p: &Projection| -> Option<GpuLinear> {
            Some(GpuLinear::new(&ctx, p.rows(), p.cols(), p.f32_weights()?))
        };

        let layers = weights
            .layers
            .iter()
            .map(|l| {
                Some(GpuLayerWeights {
                    wq: upload(&l.attention.wq)?,
                    wk: upload(&l.attention.wk)?,
                    wv: upload(&l.attention.wv)?,
                    wo: upload(&l.attention.wo)?,
                    w1: upload(&l.feed_forward.w1)?,
                    w2: upload(&l.feed_forward.w2)?,
                    w3: upload(&l.feed_forward.w3)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        let lm_head = upload(&weights.lm_head)?;
        Some(Self {
            ctx,
            layers,
            lm_head,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AttentionWeights {
    pub wq: Projection,
    pub wk: Projection,
    pub wv: Projection,
    pub wo: Projection,
}

#[derive(Debug, Clone)]
pub struct FeedForwardWeights {
    pub w1: Projection,
    pub w2: Projection,
    pub w3: Projection,
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
    pub lm_head: Projection,
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

            let q = config.quantization;
            layers.push(LayerWeights {
                attention_norm: pack_bf16_to_f32(attn_norm.raw_bytes()),
                attention: AttentionWeights {
                    wq: Projection::from_tensor(&wq, q),
                    wk: Projection::from_tensor(&wk, q),
                    wv: Projection::from_tensor(&wv, q),
                    wo: Projection::from_tensor(&wo, q),
                },
                ffn_norm: pack_bf16_to_f32(ffn_norm.raw_bytes()),
                feed_forward: FeedForwardWeights {
                    w1: Projection::from_tensor(&w1, q),
                    w2: Projection::from_tensor(&w2, q),
                    w3: Projection::from_tensor(&w3, q),
                },
            });
        }

        Ok(LlamaWeights {
            token_embeddings,
            layers,
            final_norm,
            // The LM head stays f32: it is the last projection before the
            // softmax, so its quantization error lands directly on token
            // choice, and it is one matrix rather than seven per layer.
            lm_head: Projection::F32(PackedLinear::from_tensor(&lm_head_t)),
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

fn bf16_bytes_into_f32(raw: &[u8], out: &mut [f32]) {
    debug_assert_eq!(raw.len(), out.len() * 2);
    for (o, b) in out.iter_mut().zip(raw.chunks_exact(2)) {
        *o = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
    }
}

/// Per-sequence working memory for a forward pass.
///
/// Every buffer here used to be a fresh `vec![]` on each call. At TinyLlama
/// scale that is ~10 allocations per token, the largest of them a
/// `vocab_size`-wide logit buffer (128 KB), all on the critical path. Hoisting
/// them into a reusable scratch keeps decode steps allocation-free.
pub struct ForwardScratch {
    x: Vec<f32>,
    xb: Vec<f32>,
    attn_out: Vec<f32>,
    proj_out: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ff_gate: Vec<f32>,
    ff_up: Vec<f32>,
    ff_down: Vec<f32>,
    /// Attention scores, `num_attention_heads` lanes wide, grown on demand.
    scores: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    pub logits: Vec<f32>,
}

impl ForwardScratch {
    pub fn new(config: &LlamaConfig) -> Self {
        let hidden = config.hidden_size;
        let half = config.head_dim() / 2;
        Self {
            x: vec![0.0; hidden],
            xb: vec![0.0; hidden],
            attn_out: vec![0.0; hidden],
            proj_out: vec![0.0; hidden],
            q: vec![0.0; hidden],
            k: vec![0.0; config.kv_dim()],
            v: vec![0.0; config.kv_dim()],
            ff_gate: vec![0.0; config.intermediate_size],
            ff_up: vec![0.0; config.intermediate_size],
            ff_down: vec![0.0; hidden],
            scores: Vec::new(),
            rope_cos: vec![0.0; half],
            rope_sin: vec![0.0; half],
            logits: vec![0.0; config.vocab_size],
        }
    }
}

impl<'a> LlamaWeights<'a> {
    /// Bytes the projection weights occupy as stored.
    pub fn weight_bytes(&self) -> usize {
        let mut total = 0;
        for layer in &self.layers {
            for p in [
                &layer.attention.wq,
                &layer.attention.wk,
                &layer.attention.wv,
                &layer.attention.wo,
                &layer.feed_forward.w1,
                &layer.feed_forward.w2,
                &layer.feed_forward.w3,
            ] {
                total += p.weight_bytes();
            }
        }
        total + self.lm_head.weight_bytes()
    }

    /// What the same weights would occupy as f32, for a quantization ratio.
    pub fn weight_bytes_f32(&self) -> usize {
        let mut total = 0;
        for layer in &self.layers {
            for p in [
                &layer.attention.wq,
                &layer.attention.wk,
                &layer.attention.wv,
                &layer.attention.wo,
                &layer.feed_forward.w1,
                &layer.feed_forward.w2,
                &layer.feed_forward.w3,
            ] {
                total += p.rows() * p.cols() * 4;
            }
        }
        total + self.lm_head.rows() * self.lm_head.cols() * 4
    }

    /// One decode step: consume `token_id` at `pos`, append its K/V to the paged
    /// cache, and return logits over the vocabulary.
    ///
    /// Allocates a fresh [`ForwardScratch`] per call; hot loops should hold one
    /// and use [`LlamaWeights::forward_into`] instead.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32],
        block_size: usize,
        gpu: Option<&GpuForwardContext>,
    ) -> Vec<f32> {
        let mut scratch = ForwardScratch::new(config);
        self.forward_into(
            token_id,
            pos,
            config,
            block_table,
            kv_cache,
            block_size,
            gpu,
            &mut scratch,
        );
        std::mem::take(&mut scratch.logits)
    }

    /// [`LlamaWeights::forward`] against caller-owned scratch. Leaves the logits
    /// in `scratch.logits`.
    // Wide by design: weights, config, block table, cache, and scratch are
    // separate borrows so the caller controls their lifetimes independently.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into(
        &self,
        token_id: u32,
        pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32],
        block_size: usize,
        gpu: Option<&GpuForwardContext>,
        scratch: &mut ForwardScratch,
    ) {
        self.run_layers(
            token_id,
            pos,
            config,
            block_table,
            kv_cache,
            block_size,
            gpu,
            scratch,
        );
        self.project_logits(config, gpu, scratch);
    }

    /// Consume a whole prompt, populating the KV cache for every position, and
    /// return logits for the **final** token only.
    ///
    /// This is the step the engine was missing: running only the last prompt
    /// token leaves positions `0..n-1` of the cache zeroed, so attention scores
    /// against them are meaningless and the model never sees the prompt.
    ///
    /// `start_pos` is where `tokens[0]` lands in the sequence. Passing a
    /// non-zero value resumes a partially-populated cache, which is what prefix
    /// reuse does — the tokens covered by cached blocks are skipped and only the
    /// suffix is replayed.
    ///
    /// Only the last position pays for the LM head. That projection is
    /// `vocab_size x hidden_size` — for TinyLlama the single largest matvec in
    /// the model — so skipping it for the other `n-1` tokens is most of what
    /// makes prefill cheaper than decoding the prompt token by token.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32],
        block_size: usize,
        gpu: Option<&GpuForwardContext>,
    ) -> Vec<f32> {
        let mut scratch = ForwardScratch::new(config);
        self.prefill_into(
            tokens,
            start_pos,
            config,
            block_table,
            kv_cache,
            block_size,
            gpu,
            &mut scratch,
        );
        std::mem::take(&mut scratch.logits)
    }

    /// [`LlamaWeights::prefill`] against caller-owned scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_into(
        &self,
        tokens: &[u32],
        start_pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32],
        block_size: usize,
        gpu: Option<&GpuForwardContext>,
        scratch: &mut ForwardScratch,
    ) {
        assert!(!tokens.is_empty(), "prefill needs at least one token");
        for (i, &token) in tokens.iter().enumerate() {
            self.run_layers(
                token,
                start_pos + i,
                config,
                block_table,
                kv_cache,
                block_size,
                gpu,
                scratch,
            );
        }
        self.project_logits(config, gpu, scratch);
    }

    /// Project the final hidden state through the LM head.
    fn project_logits(
        &self,
        config: &LlamaConfig,
        gpu: Option<&GpuForwardContext>,
        scratch: &mut ForwardScratch,
    ) {
        debug_assert_eq!(scratch.logits.len(), config.vocab_size);
        match gpu {
            Some(g) => g.lm_head.apply(&g.ctx, &mut scratch.logits, &scratch.x),
            None => self.lm_head.apply_parallel(&mut scratch.logits, &scratch.x),
        }
    }

    /// All transformer layers for one token. Writes the token's K/V into the
    /// paged cache and leaves the final normalized hidden state in `scratch.x`.
    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &self,
        token_id: u32,
        pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32],
        block_size: usize,
        gpu: Option<&GpuForwardContext>,
        scratch: &mut ForwardScratch,
    ) {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let kv_group = config.kv_group();
        let layout = config.kv_layout_for_cache(kv_cache.len(), block_size);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Full causal attention unless a sliding window was explicitly asked for.
        let attn_window = config.attention_window.unwrap_or(usize::MAX);
        let start_t = (pos + 1).saturating_sub(attn_window);
        let window_len = pos + 1 - start_t;

        // One rotary table per token, shared by every head of every layer.
        rope_table(
            pos,
            head_dim,
            config.rope_theta,
            &mut scratch.rope_cos,
            &mut scratch.rope_sin,
        );

        let token = (token_id as usize) % config.vocab_size;
        let embed_bytes =
            &self.token_embeddings.raw_bytes()[token * hidden * 2..(token + 1) * hidden * 2];
        bf16_bytes_into_f32(embed_bytes, &mut scratch.x);

        let need = num_heads * window_len;
        if scratch.scores.len() < need {
            scratch.scores.resize(need, 0.0);
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            scratch.xb.copy_from_slice(&scratch.x);
            rms_norm(&mut scratch.xb, &layer.attention_norm, config.rms_norm_eps);

            if let Some(g) = gpu {
                let gl = &g.layers[layer_idx];
                gl.wq.apply(&g.ctx, &mut scratch.q, &scratch.xb);
                gl.wk.apply(&g.ctx, &mut scratch.k, &scratch.xb);
                gl.wv.apply(&g.ctx, &mut scratch.v, &scratch.xb);
            } else {
                layer
                    .attention
                    .wq
                    .apply_parallel(&mut scratch.q, &scratch.xb);
                layer
                    .attention
                    .wk
                    .apply_parallel(&mut scratch.k, &scratch.xb);
                layer
                    .attention
                    .wv
                    .apply_parallel(&mut scratch.v, &scratch.xb);
            }

            // Queries and keys are rotated in separate loops on purpose. Under
            // grouped-query attention `kv_group` query heads share one key head,
            // so driving the key rotation from the query-head loop would rotate
            // each key head `kv_group` times — an 8x over-rotation on TinyLlama.
            for h in 0..num_heads {
                let off = h * head_dim;
                rope_rotate(
                    &mut scratch.q[off..off + head_dim],
                    &scratch.rope_cos,
                    &scratch.rope_sin,
                    config.rope_style,
                );
            }
            for kv_h in 0..config.num_key_value_heads {
                let off = kv_h * head_dim;
                rope_rotate(
                    &mut scratch.k[off..off + head_dim],
                    &scratch.rope_cos,
                    &scratch.rope_sin,
                    config.rope_style,
                );
            }

            if let Some((phys_block, offset)) = block_table.get_physical_location(pos, block_size) {
                for kv_h in 0..config.num_key_value_heads {
                    let src = kv_h * head_dim;
                    let k_idx = layout.index(layer_idx, phys_block.index, offset, kv_h, false);
                    let v_idx = layout.index(layer_idx, phys_block.index, offset, kv_h, true);
                    kv_cache[k_idx..k_idx + head_dim]
                        .copy_from_slice(&scratch.k[src..src + head_dim]);
                    kv_cache[v_idx..v_idx + head_dim]
                        .copy_from_slice(&scratch.v[src..src + head_dim]);
                }
            }

            // Attention runs one head per Rayon task. Every K/V write for this
            // token completed above, so the cache is read-only here and the
            // output slices are disjoint — no synchronization needed.
            let kv_ref: &[f32] = &*kv_cache;
            let q_ref: &[f32] = &scratch.q;
            let attn_out = &mut scratch.attn_out;
            let scores = &mut scratch.scores[..need];

            attn_out
                .par_chunks_mut(head_dim)
                .zip(scores.par_chunks_mut(window_len))
                .enumerate()
                .for_each(|(h, (out_slice, score_slice))| {
                    let kv_h = h / kv_group;
                    let q_start = h * head_dim;
                    let q_head = &q_ref[q_start..q_start + head_dim];

                    for (si, t) in (start_t..=pos).enumerate() {
                        score_slice[si] = match block_table.get_physical_location(t, block_size) {
                            Some((pb, off)) => {
                                let k_idx = layout.index(layer_idx, pb.index, off, kv_h, false);
                                crate::math::dot(q_head, &kv_ref[k_idx..k_idx + head_dim]) * scale
                            }
                            // No physical block backs this position, so there is
                            // nothing to attend to. It has to be masked out with
                            // -inf: a score of 0.0 would instead survive the
                            // softmax and take a full share of the weight.
                            None => f32::NEG_INFINITY,
                        };
                    }

                    crate::math::softmax_in_place(score_slice);

                    out_slice.fill(0.0);
                    for (si, t) in (start_t..=pos).enumerate() {
                        let weight = score_slice[si];
                        if weight == 0.0 {
                            continue;
                        }
                        if let Some((pb, off)) = block_table.get_physical_location(t, block_size) {
                            let v_idx = layout.index(layer_idx, pb.index, off, kv_h, true);
                            crate::math::axpy(out_slice, weight, &kv_ref[v_idx..v_idx + head_dim]);
                        }
                    }
                });

            if let Some(g) = gpu {
                g.layers[layer_idx]
                    .wo
                    .apply(&g.ctx, &mut scratch.proj_out, &scratch.attn_out);
            } else {
                layer
                    .attention
                    .wo
                    .apply_parallel(&mut scratch.proj_out, &scratch.attn_out);
            }
            for i in 0..hidden {
                scratch.x[i] += scratch.proj_out[i];
            }

            scratch.xb.copy_from_slice(&scratch.x);
            rms_norm(&mut scratch.xb, &layer.ffn_norm, config.rms_norm_eps);

            if let Some(g) = gpu {
                let gl = &g.layers[layer_idx];
                gl.w1.apply(&g.ctx, &mut scratch.ff_gate, &scratch.xb);
                gl.w3.apply(&g.ctx, &mut scratch.ff_up, &scratch.xb);
            } else {
                layer
                    .feed_forward
                    .w1
                    .apply_parallel(&mut scratch.ff_gate, &scratch.xb);
                layer
                    .feed_forward
                    .w3
                    .apply_parallel(&mut scratch.ff_up, &scratch.xb);
            }
            swiglu(&mut scratch.ff_gate, &scratch.ff_up);
            if let Some(g) = gpu {
                g.layers[layer_idx]
                    .w2
                    .apply(&g.ctx, &mut scratch.ff_down, &scratch.ff_gate);
            } else {
                layer
                    .feed_forward
                    .w2
                    .apply_parallel(&mut scratch.ff_down, &scratch.ff_gate);
            }
            for i in 0..hidden {
                scratch.x[i] += scratch.ff_down[i];
            }
        }

        rms_norm(&mut scratch.x, &self.final_norm, config.rms_norm_eps);
    }
}
