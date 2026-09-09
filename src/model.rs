use anyhow::{Context, Result};
use safetensors::SafeTensors;

use crate::attention::{AttnEntry, PagedAttention};
use crate::gpu::{GpuContext, GpuLinear};
use crate::math::{
    matvec_f32_weight_transposed_parallel, rms_norm, rope_inv_freq, rope_rotate, rope_table_from,
    swiglu, RopeScaling, RopeStyle,
};
use crate::memory::block_table::BlockTable;
use crate::memory::layout::KvLayout;
use crate::tensor::{DType, Tensor};

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
    /// Rotary scaling the checkpoint was trained with. See [`RopeScaling`].
    pub rope_scaling: RopeScaling,
    /// The context window the checkpoint declares, when it does.
    pub max_position_embeddings: Option<usize>,
    /// Special tokens as `config.json` states them. A serving layer should
    /// take these over any hardcoded default; the engine is told them through
    /// its own config.
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
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
    /// TinyLlama 1.1B's architecture.
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
            rope_scaling: RopeScaling::None,
            max_position_embeddings: None,
            bos_token_id: Some(1),
            eos_token_ids: vec![2],
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
    /// Strict on purpose. A file that is present but malformed, or that
    /// declares a feature this engine does not implement (a non-Llama
    /// `model_type`, projection biases, a rotary scaling variant other than
    /// `default`, `linear` or `llama3`, an activation other than SiLU), is an
    /// error — never a silent fallback to defaults, which would run a
    /// different model than the one on disk.
    pub fn from_hf_config(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        Self::from_hf_json(&json).with_context(|| format!("in {}", path.display()))
    }

    /// [`LlamaConfig::from_hf_config`] on an already-parsed document.
    pub fn from_hf_json(json: &serde_json::Value) -> Result<Self> {
        use serde_json::Value;
        anyhow::ensure!(json.is_object(), "config.json is not a JSON object");

        if let Some(model_type) = json.get("model_type").and_then(Value::as_str) {
            anyhow::ensure!(
                model_type == "llama",
                "model_type {model_type:?} is not supported; this engine implements the Llama \
                 architecture (model_type \"llama\")"
            );
        }
        if let Some(act) = json.get("hidden_act").and_then(Value::as_str) {
            anyhow::ensure!(
                act == "silu",
                "hidden_act {act:?} is not supported; the feed-forward path is SwiGLU"
            );
        }
        for key in ["attention_bias", "mlp_bias"] {
            anyhow::ensure!(
                json.get(key).and_then(Value::as_bool) != Some(true),
                "{key} = true is not supported: the loader has no bias tensors"
            );
        }

        let required = |key: &str| -> Result<usize> {
            match json.get(key) {
                Some(v) => v.as_u64().map(|v| v as usize).ok_or_else(|| {
                    anyhow::anyhow!("{key} must be a non-negative integer, got {v}")
                }),
                None => anyhow::bail!("config.json is missing {key}"),
            }
        };
        let optional = |key: &str| -> Result<Option<usize>> {
            match json.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => v.as_u64().map(|v| Some(v as usize)).ok_or_else(|| {
                    anyhow::anyhow!("{key} must be a non-negative integer, got {v}")
                }),
            }
        };
        let optional_f32 = |key: &str| -> Result<Option<f32>> {
            match json.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(v) => v
                    .as_f64()
                    .map(|v| Some(v as f32))
                    .ok_or_else(|| anyhow::anyhow!("{key} must be a number, got {v}")),
            }
        };

        let hidden_size = required("hidden_size")?;
        let num_hidden_layers = required("num_hidden_layers")?;
        let num_attention_heads = required("num_attention_heads")?;
        let intermediate_size = required("intermediate_size")?;
        let vocab_size = required("vocab_size")?;
        // Multi-head checkpoints omit this; it then equals the query heads.
        let num_key_value_heads = optional("num_key_value_heads")?.unwrap_or(num_attention_heads);
        // transformers' own defaults for the fields a config may omit.
        let rms_norm_eps = optional_f32("rms_norm_eps")?.unwrap_or(1e-6);

        if let Some(head_dim) = optional("head_dim")? {
            anyhow::ensure!(
                num_attention_heads > 0 && head_dim * num_attention_heads == hidden_size,
                "head_dim {head_dim} is not hidden_size / num_attention_heads = {}; a \
                 decoupled head dimension is not supported",
                hidden_size.checked_div(num_attention_heads).unwrap_or(0)
            );
        }

        let (rope_theta, rope_scaling) = parse_rope(json)?;
        let max_position_embeddings = optional("max_position_embeddings")?;
        let bos_token_id = optional("bos_token_id")?
            .map(|v| u32::try_from(v).map_err(|_| anyhow::anyhow!("bos_token_id {v} overflows")))
            .transpose()?;
        let eos_token_ids = parse_token_ids(json.get("eos_token_id"), "eos_token_id")?;

        let config = Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            rope_scaling,
            max_position_embeddings,
            bos_token_id,
            eos_token_ids,
            // Not architecture: these are engine policy, so they keep defaults
            // and are set by the caller.
            attention_window: None,
            rope_style: RopeStyle::Neox,
            quantization: Quantization::F32,
        };
        config.validate()?;
        Ok(config)
    }

    /// Load the config sitting beside a checkpoint.
    ///
    /// A checkpoint that ships without one gets the [`Default`] (TinyLlama)
    /// shape, which the loader then checks against every tensor in the file —
    /// so a wrong guess fails at load, not at inference. A `config.json` that is
    /// present but cannot be used is an error.
    pub fn beside_checkpoint(model_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let candidate = model_path.as_ref().with_file_name("config.json");
        if !candidate.exists() {
            return Ok(Self::default());
        }
        Self::from_hf_config(&candidate)
    }

    /// Check the shape is one the kernels can run. Called by the loader; a
    /// hand-built config that skips it fails later with a worse message.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.hidden_size > 0
                && self.num_hidden_layers > 0
                && self.num_attention_heads > 0
                && self.num_key_value_heads > 0
                && self.intermediate_size > 0
                && self.vocab_size > 0,
            "every model dimension must be positive: {self:?}"
        );
        anyhow::ensure!(
            self.hidden_size.is_multiple_of(self.num_attention_heads),
            "hidden_size {} is not a multiple of num_attention_heads {}",
            self.hidden_size,
            self.num_attention_heads
        );
        anyhow::ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads),
            "num_attention_heads {} is not a multiple of num_key_value_heads {}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        anyhow::ensure!(
            self.head_dim().is_multiple_of(2),
            "head_dim {} must be even for rotary embeddings",
            self.head_dim()
        );
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.rope_theta),
        ] {
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "{name} must be finite and positive, got {value}"
            );
        }
        validate_rope_scaling(self.rope_scaling, "rope_scaling")?;
        anyhow::ensure!(
            self.attention_window != Some(0),
            "attention_window must be at least 1; use None for full attention"
        );
        anyhow::ensure!(
            self.max_position_embeddings != Some(0),
            "max_position_embeddings must be positive"
        );
        Ok(())
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

/// `rope_theta` and the scaling variant, from either the older `rope_scaling`
/// object or the newer `rope_parameters` one (which also carries the theta).
fn parse_rope(json: &serde_json::Value) -> Result<(f32, RopeScaling)> {
    use serde_json::Value;
    let object = |key: &str| -> Result<Option<&serde_json::Map<String, Value>>> {
        match json.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Object(m)) => Ok(Some(m)),
            Some(other) => anyhow::bail!("{key} must be an object, got {other}"),
        }
    };

    let mut theta = match json.get("rope_theta") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .ok_or_else(|| anyhow::anyhow!("rope_theta must be a number, got {v}"))?
                as f32,
        ),
    };
    let mut scaling = None;

    if let Some(params) = object("rope_parameters")? {
        if let Some(t) = params.get("rope_theta").filter(|v| !v.is_null()) {
            theta = Some(
                t.as_f64()
                    .ok_or_else(|| anyhow::anyhow!("rope_parameters.rope_theta must be a number"))?
                    as f32,
            );
        }
        scaling = Some(parse_rope_scaling(params, "rope_parameters")?);
    }
    if let Some(legacy) = object("rope_scaling")? {
        let parsed = parse_rope_scaling(legacy, "rope_scaling")?;
        match scaling {
            Some(modern) => anyhow::ensure!(
                modern == parsed,
                "rope_scaling ({parsed:?}) and rope_parameters ({modern:?}) disagree"
            ),
            None => scaling = Some(parsed),
        }
    }
    Ok((theta.unwrap_or(10_000.0), scaling.unwrap_or_default()))
}

fn parse_rope_scaling(
    obj: &serde_json::Map<String, serde_json::Value>,
    what: &str,
) -> Result<RopeScaling> {
    let kind = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .map(|v| {
            v.as_str()
                .ok_or_else(|| anyhow::anyhow!("{what}.rope_type must be a string, got {v}"))
        })
        .transpose()?
        .unwrap_or("default");
    let number = |key: &str| -> Result<f32> {
        obj.get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .ok_or_else(|| anyhow::anyhow!("{what} of type {kind:?} needs a numeric {key}"))
    };
    let scaling = match kind {
        "default" => RopeScaling::None,
        "linear" => {
            let factor = number("factor")?;
            RopeScaling::Linear { factor }
        }
        "llama3" => {
            let factor = number("factor")?;
            let low_freq_factor = number("low_freq_factor")?;
            let high_freq_factor = number("high_freq_factor")?;
            let original = obj
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{what} of type \"llama3\" needs an integer original_max_position_embeddings"
                    )
                })? as usize;
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings: original,
            }
        }
        other => anyhow::bail!(
            "{what} type {other:?} is not supported (supported: default, linear, llama3); \
             running this checkpoint with default rotary tables would be a different model"
        ),
    };
    validate_rope_scaling(scaling, what)?;
    Ok(scaling)
}

/// Shared by parsed and hand-built configs so neither can reach the rotary
/// kernels with invalid parameters. A finite JSON f64 can overflow on the f32
/// conversion; a positivity check alone would accept the resulting infinity.
fn validate_rope_scaling(scaling: RopeScaling, what: &str) -> Result<()> {
    match scaling {
        RopeScaling::None => {}
        RopeScaling::Linear { factor } => anyhow::ensure!(
            factor.is_finite() && factor > 0.0,
            "{what}.factor must be finite and positive, got {factor}"
        ),
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings: original,
        } => anyhow::ensure!(
            factor.is_finite()
                && factor >= 1.0
                && low_freq_factor.is_finite()
                && low_freq_factor > 0.0
                && high_freq_factor.is_finite()
                && high_freq_factor > low_freq_factor
                && original > 0,
            "{what} llama3 parameters must be finite and in range: factor {factor}, \
             low {low_freq_factor}, high {high_freq_factor}, original {original}"
        ),
    }
    Ok(())
}

/// `eos_token_id` may be a single id, a list of them, or absent.
fn parse_token_ids(value: Option<&serde_json::Value>, what: &str) -> Result<Vec<u32>> {
    use serde_json::Value;
    let one = |v: &Value| -> Result<u32> {
        v.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| anyhow::anyhow!("{what} entries must be token ids, got {v}"))
    };
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items.iter().map(one).collect(),
        Some(v) => Ok(vec![one(v)?]),
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
            weight: t.to_f32_vec(),
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

    /// Project `batch` activation vectors at once, streaming the weights a
    /// single time instead of once per sequence.
    ///
    /// `x_br` and `out_br` are `[batch][feature]`-major. `stage` is scratch of
    /// at least `rows * batch`, used for the kernel's `[row][batch]` output
    /// before it is transposed back.
    pub fn apply_batched(&self, out_br: &mut [f32], x_br: &[f32], batch: usize, stage: &mut [f32]) {
        let (rows, cols) = (self.rows(), self.cols());
        let stage = &mut stage[..rows * batch];
        match self {
            Projection::F32(w) => crate::math::matmat_f32_weight_transposed_parallel(
                stage, x_br, &w.weight, batch, rows, cols,
            ),
            Projection::Int8(w) => crate::math::matmat_i8_weight_parallel(
                stage, x_br, &w.weight, &w.scales, batch, rows, cols,
            ),
        }
        crate::math::transpose_rb_to_br(stage, &mut out_br[..rows * batch], rows, batch);
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

    /// Decode every tensor the architecture needs, checking each one's dtype
    /// and shape against `config` before anything is computed with it.
    ///
    /// The checkpoint is the ground truth for what the weights *are*; the
    /// config is the ground truth for what the engine will *do* with them. The
    /// two have to agree exactly, or the failure is silent: a shape mismatch
    /// would index the wrong elements, a dtype mismatch would decode the wrong
    /// numbers, and either produces a model that runs and emits noise.
    pub fn load_weights(&self, config: &LlamaConfig) -> Result<LlamaWeights<'a>> {
        config.validate().context("model config is not usable")?;
        let hidden = config.hidden_size;
        let kv_dim = config.kv_dim();
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        let token_embeddings = self.tensor("model.embed_tokens.weight", &[vocab, hidden])?;
        let final_norm = self.tensor("model.norm.weight", &[hidden])?.to_f32_vec();
        // Tied embeddings ship no separate LM head; a present-but-wrong one is
        // still an error rather than a fallback.
        let lm_head_t = if self.has("lm_head.weight") {
            self.tensor("lm_head.weight", &[vocab, hidden])?
        } else {
            token_embeddings.clone()
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let t = |name: &str, shape: &[usize]| self.tensor(&format!("{prefix}.{name}"), shape);

            let attn_norm = t("input_layernorm.weight", &[hidden])?;
            let ffn_norm = t("post_attention_layernorm.weight", &[hidden])?;
            let wq = t("self_attn.q_proj.weight", &[hidden, hidden])?;
            let wk = t("self_attn.k_proj.weight", &[kv_dim, hidden])?;
            let wv = t("self_attn.v_proj.weight", &[kv_dim, hidden])?;
            let wo = t("self_attn.o_proj.weight", &[hidden, hidden])?;
            let w1 = t("mlp.gate_proj.weight", &[inter, hidden])?;
            let w2 = t("mlp.down_proj.weight", &[hidden, inter])?;
            let w3 = t("mlp.up_proj.weight", &[inter, hidden])?;

            let q = config.quantization;
            layers.push(LayerWeights {
                attention_norm: attn_norm.to_f32_vec(),
                attention: AttentionWeights {
                    wq: Projection::from_tensor(&wq, q),
                    wk: Projection::from_tensor(&wk, q),
                    wv: Projection::from_tensor(&wv, q),
                    wo: Projection::from_tensor(&wo, q),
                },
                ffn_norm: ffn_norm.to_f32_vec(),
                feed_forward: FeedForwardWeights {
                    w1: Projection::from_tensor(&w1, q),
                    w2: Projection::from_tensor(&w2, q),
                    w3: Projection::from_tensor(&w3, q),
                },
            });
        }

        // A config that under-counts the layers would load a truncated model
        // that runs. Refuse it.
        let beyond = format!(
            "model.layers.{}.input_layernorm.weight",
            config.num_hidden_layers
        );
        anyhow::ensure!(
            !self.has(&beyond),
            "checkpoint has more than the {} layers config.json declares",
            config.num_hidden_layers
        );

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

    fn has(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    /// A named tensor, with its dtype decoded and its shape checked.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Tensor<'a>> {
        let view = self
            .tensors
            .tensor(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        let dtype = DType::from_safetensors(view.dtype()).ok_or_else(|| {
            anyhow::anyhow!(
                "tensor {name} is stored as {:?}; only BF16, F16 and F32 checkpoints are supported",
                view.dtype()
            )
        })?;
        anyhow::ensure!(
            view.shape() == shape,
            "tensor {name} has shape {:?} but config.json implies {:?}",
            view.shape(),
            shape
        );
        Ok(Tensor::new(view.data(), view.shape().to_vec(), dtype))
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
    /// Per-pair rotary frequencies, scaled as the config declares. Fixed for
    /// the model, so computed once here rather than per token.
    rope_inv_freq: Vec<f32>,
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
            rope_inv_freq: rope_inv_freq(config.head_dim(), config.rope_theta, config.rope_scaling),
            logits: vec![0.0; config.vocab_size],
        }
    }
}

/// Working memory for decoding several sequences in one pass.
///
/// Every per-sequence buffer is `[batch][feature]`-major, so sequence `b`'s
/// slice is contiguous and the per-sequence steps (RMSNorm, RoPE, SwiGLU) index
/// it directly.
pub struct BatchScratch {
    capacity: usize,
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
    /// `[row][batch]` output from the projection kernel, before transposing.
    stage: Vec<f32>,
    /// Rotary tables, one per sequence: batched sequences sit at different
    /// positions, so they cannot share a table the way one sequence's heads do.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    rope_inv_freq: Vec<f32>,
    scores: Vec<f32>,
    pub logits: Vec<f32>,
}

impl BatchScratch {
    /// Allocate for up to `capacity` concurrent sequences.
    pub fn new(config: &LlamaConfig, capacity: usize) -> Self {
        let hidden = config.hidden_size;
        let half = config.head_dim() / 2;
        // The staging buffer has to hold the widest projection, which is the LM
        // head at vocab_size rows.
        let widest = config
            .vocab_size
            .max(config.intermediate_size)
            .max(hidden)
            .max(config.kv_dim());
        Self {
            capacity,
            x: vec![0.0; capacity * hidden],
            xb: vec![0.0; capacity * hidden],
            attn_out: vec![0.0; capacity * hidden],
            proj_out: vec![0.0; capacity * hidden],
            q: vec![0.0; capacity * hidden],
            k: vec![0.0; capacity * config.kv_dim()],
            v: vec![0.0; capacity * config.kv_dim()],
            ff_gate: vec![0.0; capacity * config.intermediate_size],
            ff_up: vec![0.0; capacity * config.intermediate_size],
            ff_down: vec![0.0; capacity * hidden],
            stage: vec![0.0; capacity * widest],
            rope_cos: vec![0.0; capacity * half],
            rope_sin: vec![0.0; capacity * half],
            rope_inv_freq: rope_inv_freq(config.head_dim(), config.rope_theta, config.rope_scaling),
            scores: Vec::new(),
            logits: vec![0.0; capacity * config.vocab_size],
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Logits for sequence `b` of a batch of `batch`.
    pub fn logits_for(&self, b: usize, vocab_size: usize) -> &[f32] {
        &self.logits[b * vocab_size..(b + 1) * vocab_size]
    }

    /// Mutable logits for sequence `b`, for sampling in place.
    pub fn logits_for_mut(&mut self, b: usize, vocab_size: usize) -> &mut [f32] {
        &mut self.logits[b * vocab_size..(b + 1) * vocab_size]
    }
}

impl<'a> LlamaWeights<'a> {
    /// The embedding of `token`, decoded from the checkpoint's own dtype.
    ///
    /// Out-of-vocabulary ids are a panic, not a wrap-around: the engine refuses
    /// them at submission, so one arriving here is a bug, and silently reading
    /// `token % vocab` would turn that bug into plausible-looking output.
    pub fn embed_into(&self, token: u32, out: &mut [f32]) {
        self.token_embeddings.row_into(token as usize, out);
    }

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

        // Full causal attention unless a sliding window was explicitly asked for.
        let entry = AttnEntry::new(block_table, pos, config.attention_window);
        // At least one slot per lane even for a zero-width window, so the score
        // buffer and the kernel's stride always agree.
        let window_len = entry.window_len().max(1);
        let attn = PagedAttention {
            layout,
            block_size,
            num_heads,
            head_dim,
            kv_group,
            score_stride: window_len,
            heads_per_lane: PagedAttention::lane_width(kv_group, config.num_key_value_heads, 1),
        };

        // One rotary table per token, shared by every head of every layer.
        rope_table_from(
            pos,
            &scratch.rope_inv_freq,
            &mut scratch.rope_cos,
            &mut scratch.rope_sin,
        );

        self.embed_into(token_id, &mut scratch.x);

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

            // The same kernel the batched path uses, with a batch of one. Every
            // K/V write for this token completed above, so the cache is
            // read-only here and the output slices are disjoint.
            attn.run(
                &mut scratch.attn_out,
                &mut scratch.scores[..need],
                &scratch.q,
                kv_cache,
                std::slice::from_ref(&entry),
                layer_idx,
            );

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

    // ── batched decode ───────────────────────────────────────────────────────

    /// Advance `tokens.len()` sequences by one token each, in a single pass.
    ///
    /// The projections are the reason this exists. Decoding sequences one at a
    /// time re-reads every weight matrix per sequence, and a matvec is
    /// memory-bound, so `batch` sequences cost `batch` times the DRAM traffic
    /// for identical arithmetic. Batching streams each matrix once.
    ///
    /// Attention does *not* batch: each sequence has its own block table, its
    /// own position, and its own KV history, so there is no shared operand.
    /// It is instead parallelized across every (sequence, head) pair at once,
    /// which gives Rayon more independent work than one sequence's heads would.
    ///
    /// Run every transformer layer for a batch of (token, position, table)
    /// triples, leaving each entry's final normalized hidden state in
    /// `scratch.x`. Shared by batched decode and batched prefill.
    ///
    /// The two differ only in what the batch *means*. Decode passes one token
    /// from each of several sequences; prefill passes consecutive tokens of one
    /// sequence, with the same block table repeated. Both are correct here for
    /// the same reason: every entry's K/V is written to the cache before any
    /// attention runs, and an entry's attention loop ends at its own position —
    /// so a prefill entry sees the entries before it in the chunk and none
    /// after, which is exactly causal masking.
    #[allow(clippy::too_many_arguments)]
    fn run_layers_batch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        block_tables: &[&BlockTable],
        config: &LlamaConfig,
        kv_cache: &mut [f32],
        block_size: usize,
        scratch: &mut BatchScratch,
    ) {
        let batch = tokens.len();
        assert_eq!(positions.len(), batch);
        assert_eq!(block_tables.len(), batch);
        assert!(batch > 0, "decode_batch_into needs at least one sequence");
        assert!(
            batch <= scratch.capacity,
            "batch {batch} exceeds scratch capacity {}",
            scratch.capacity
        );

        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let kv_dim = config.kv_dim();
        let inter = config.intermediate_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let kv_group = config.kv_group();
        let half = head_dim / 2;
        let layout = config.kv_layout_for_cache(kv_cache.len(), block_size);

        // Sequences sit at different positions, so each needs its own rotary
        // table and its own attention window.
        let mut entries = Vec::with_capacity(batch);
        let mut widest_window = 0;
        for (b, &pos) in positions.iter().enumerate() {
            rope_table_from(
                pos,
                &scratch.rope_inv_freq,
                &mut scratch.rope_cos[b * half..(b + 1) * half],
                &mut scratch.rope_sin[b * half..(b + 1) * half],
            );
            let entry = AttnEntry::new(block_tables[b], pos, config.attention_window);
            widest_window = widest_window.max(entry.window_len());
            entries.push(entry);
        }

        for (b, &token_id) in tokens.iter().enumerate() {
            self.embed_into(token_id, &mut scratch.x[b * hidden..(b + 1) * hidden]);
        }

        // One score lane per (sequence, head), sized to the widest window in the
        // batch so every lane has the same stride.
        let lanes = batch * num_heads;
        let widest_window = widest_window.max(1);
        if scratch.scores.len() < lanes * widest_window {
            scratch.scores.resize(lanes * widest_window, 0.0);
        }

        let attn = PagedAttention {
            layout,
            block_size,
            num_heads,
            head_dim,
            kv_group,
            score_stride: widest_window,
            heads_per_lane: PagedAttention::lane_width(kv_group, num_kv_heads, batch),
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for b in 0..batch {
                let span = b * hidden..(b + 1) * hidden;
                scratch.xb[span.clone()].copy_from_slice(&scratch.x[span]);
                rms_norm(
                    &mut scratch.xb[b * hidden..(b + 1) * hidden],
                    &layer.attention_norm,
                    config.rms_norm_eps,
                );
            }

            let xb = &scratch.xb[..batch * hidden];
            layer
                .attention
                .wq
                .apply_batched(&mut scratch.q, xb, batch, &mut scratch.stage);
            layer
                .attention
                .wk
                .apply_batched(&mut scratch.k, xb, batch, &mut scratch.stage);
            layer
                .attention
                .wv
                .apply_batched(&mut scratch.v, xb, batch, &mut scratch.stage);

            // Rotate queries and keys separately: under GQA several query heads
            // share one key head, so driving keys from the query loop would
            // rotate them kv_group times over.
            for b in 0..batch {
                let (cos, sin) = (
                    &scratch.rope_cos[b * half..(b + 1) * half],
                    &scratch.rope_sin[b * half..(b + 1) * half],
                );
                for h in 0..num_heads {
                    let off = b * hidden + h * head_dim;
                    rope_rotate(
                        &mut scratch.q[off..off + head_dim],
                        cos,
                        sin,
                        config.rope_style,
                    );
                }
                for kv_h in 0..num_kv_heads {
                    let off = b * kv_dim + kv_h * head_dim;
                    rope_rotate(
                        &mut scratch.k[off..off + head_dim],
                        cos,
                        sin,
                        config.rope_style,
                    );
                }
            }

            for b in 0..batch {
                if let Some((pb, offset)) =
                    block_tables[b].get_physical_location(positions[b], block_size)
                {
                    for kv_h in 0..num_kv_heads {
                        let src = b * kv_dim + kv_h * head_dim;
                        let k_idx = layout.index(layer_idx, pb.index, offset, kv_h, false);
                        let v_idx = layout.index(layer_idx, pb.index, offset, kv_h, true);
                        kv_cache[k_idx..k_idx + head_dim]
                            .copy_from_slice(&scratch.k[src..src + head_dim]);
                        kv_cache[v_idx..v_idx + head_dim]
                            .copy_from_slice(&scratch.v[src..src + head_dim]);
                    }
                }
            }

            // Every KV write for this step is done, so the cache is read-only
            // below and the output slices are disjoint.
            attn.run(
                &mut scratch.attn_out[..batch * hidden],
                &mut scratch.scores[..lanes * widest_window],
                &scratch.q[..batch * hidden],
                kv_cache,
                &entries,
                layer_idx,
            );

            let attn_in = &scratch.attn_out[..batch * hidden];
            layer.attention.wo.apply_batched(
                &mut scratch.proj_out,
                attn_in,
                batch,
                &mut scratch.stage,
            );
            for i in 0..batch * hidden {
                scratch.x[i] += scratch.proj_out[i];
            }

            for b in 0..batch {
                let span = b * hidden..(b + 1) * hidden;
                scratch.xb[span.clone()].copy_from_slice(&scratch.x[span]);
                rms_norm(
                    &mut scratch.xb[b * hidden..(b + 1) * hidden],
                    &layer.ffn_norm,
                    config.rms_norm_eps,
                );
            }

            let xb = &scratch.xb[..batch * hidden];
            layer.feed_forward.w1.apply_batched(
                &mut scratch.ff_gate,
                xb,
                batch,
                &mut scratch.stage,
            );
            layer
                .feed_forward
                .w3
                .apply_batched(&mut scratch.ff_up, xb, batch, &mut scratch.stage);
            // SwiGLU is elementwise, so the whole batch goes through in one
            // call — no per-sequence loop, and the two buffers are disjoint
            // fields so both borrows coexist.
            let gate = &mut scratch.ff_gate[..batch * inter];
            let up = &scratch.ff_up[..batch * inter];
            swiglu(gate, up);
            let ff = &scratch.ff_gate[..batch * inter];
            layer.feed_forward.w2.apply_batched(
                &mut scratch.ff_down,
                ff,
                batch,
                &mut scratch.stage,
            );
            for i in 0..batch * hidden {
                scratch.x[i] += scratch.ff_down[i];
            }
        }

        for b in 0..batch {
            rms_norm(
                &mut scratch.x[b * hidden..(b + 1) * hidden],
                &self.final_norm,
                config.rms_norm_eps,
            );
        }
    }

    /// Advance `tokens.len()` sequences by one token each, in a single pass.
    ///
    /// Leaves sequence `b`'s logits at `scratch.logits_for(b, vocab_size)`.
    /// `positions[b]` is where `tokens[b]` lands in its own sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_batch_into(
        &self,
        tokens: &[u32],
        positions: &[usize],
        block_tables: &[&BlockTable],
        config: &LlamaConfig,
        kv_cache: &mut [f32],
        block_size: usize,
        scratch: &mut BatchScratch,
    ) {
        let batch = tokens.len();
        self.run_layers_batch(
            tokens,
            positions,
            block_tables,
            config,
            kv_cache,
            block_size,
            scratch,
        );
        let x = &scratch.x[..batch * config.hidden_size];
        self.lm_head
            .apply_batched(&mut scratch.logits, x, batch, &mut scratch.stage);
    }

    /// Consume a whole prompt, processing `chunk_size` positions at a time.
    ///
    /// Prefill was the last part of the engine still walking one token at a
    /// time, which meant re-reading every weight matrix per prompt token — the
    /// same waste batched decode removed, on the path that sets
    /// time-to-first-token. Positions batch exactly like sequences do: their
    /// projections are independent, and causality is already enforced by each
    /// entry attending only up to its own position.
    ///
    /// Only the final position pays for the LM head, and its logits land at
    /// `scratch.logits_for(0, vocab_size)`. `start_pos` is where `tokens[0]`
    /// sits, so a prefix-cache hit can replay just the uncached suffix.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_batched(
        &self,
        tokens: &[u32],
        start_pos: usize,
        config: &LlamaConfig,
        block_table: &BlockTable,
        kv_cache: &mut [f32],
        block_size: usize,
        chunk_size: usize,
        scratch: &mut BatchScratch,
    ) {
        assert!(!tokens.is_empty(), "prefill needs at least one token");
        let hidden = config.hidden_size;
        let chunk_size = chunk_size.clamp(1, scratch.capacity());

        let mut last_hidden_at = 0;
        for (c, chunk) in tokens.chunks(chunk_size).enumerate() {
            let base = start_pos + c * chunk_size;
            let positions: Vec<usize> = (0..chunk.len()).map(|i| base + i).collect();
            // Every position of one sequence shares that sequence's mapping.
            let tables = vec![block_table; chunk.len()];
            self.run_layers_batch(
                chunk, &positions, &tables, config, kv_cache, block_size, scratch,
            );
            last_hidden_at = chunk.len() - 1;
        }

        // Only the last position needs logits. The LM head is `vocab_size x
        // hidden_size` — the largest matrix in the model — so projecting every
        // prompt position through it would cost more than the layers did.
        let x = &scratch.x[last_hidden_at * hidden..(last_hidden_at + 1) * hidden];
        let logits = &mut scratch.logits[..config.vocab_size];
        self.lm_head.apply_parallel(logits, x);
    }
}
