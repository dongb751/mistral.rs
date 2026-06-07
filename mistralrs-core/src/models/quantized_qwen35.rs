#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Embedding;
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::{
    attention::{AttentionMask, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    gguf::Content,
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{CausalMaskConfig, CausalMasker, QRmsNorm, Qwen3VLRotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    models::gdn::{gated_delta_rule_recurrence, l2_norm, softplus, GdnLayerCache},
    paged_attention::AttentionImplementation,
    pipeline::{extract_logits, EitherCache, KvCache},
    utils::{
        gguf_metadata::{ContentMetadata, TryValueInto},
        model_config as ModelConfig,
        progress::{new_multi_progress, NiceProgressBar},
    },
};

const DEFAULT_MAX_SEQ_LEN: u64 = 4096;
const DEFAULT_FULL_ATTENTION_INTERVAL: usize = 4;
const DEFAULT_MROPE_SECTION: [usize; 4] = [11, 11, 10, 0];
const DEFAULT_PARTIAL_ROTARY_FACTOR: f32 = 0.25;
const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000_000.0;

#[derive(Clone, Copy, PartialEq)]
enum LayerType {
    FullAttention,
    LinearAttention,
}

struct Mlp {
    gate: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        let y = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.down.forward(&y)
    }
}

struct FullAttention {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    o: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<Qwen3VLRotaryEmbedding>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl FullAttention {
    fn forward(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q_gate = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;

        let q_gate = q_gate.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        (q, k) = self.rotary.forward_qk_norm(
            cos_sin,
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
        )?;
        let (k, v) = cache.append(&k.to_dtype(self.dtype)?, &v.to_dtype(self.dtype)?)?;
        let mut y = Sdpa.run_attention(
            &q.to_dtype(self.dtype)?,
            &k,
            &v,
            mask,
            None,
            &self.sdpa_params,
        )?;
        y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        let y = y.broadcast_mul(&gate)?;
        let y = self.o.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        x.broadcast_div(&(variance + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&gate)?
            .to_dtype(dtype)
    }
}

struct LinearAttention {
    qkv: Arc<dyn QuantMethod>,
    z: Arc<dyn QuantMethod>,
    beta: Arc<dyn QuantMethod>,
    alpha: Arc<dyn QuantMethod>,
    conv1d_weight: Tensor,
    dt_bias: Tensor,
    a: Tensor,
    norm: RmsNormGated,
    out: Arc<dyn QuantMethod>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_kernel_size: usize,
    key_dim: usize,
    value_dim: usize,
}

impl LinearAttention {
    fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = cache.conv_state.dtype();
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // --- 5 quant matmuls ---
        let mixed_qkv = self.qkv.forward(x)?;
        let z =
            self.z
                .forward(x)?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let b = self
            .beta
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_v_heads))?;
        let a = self
            .alpha
            .forward(x)?
            .reshape((batch_size, seq_len, self.num_v_heads))?;

        // --- conv1d ---
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v_flat = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
        let mixed_qkv = Tensor::cat(&[&q, &k, &v_flat], D::Minus1)?;
        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        // --- reshape + narrow ---
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?.reshape((
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let k = mixed_qkv
            .narrow(D::Minus1, self.key_dim, self.key_dim)?
            .reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = mixed_qkv
            .narrow(D::Minus1, self.key_dim * 2, self.value_dim)?
            .reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // --- gating (fused_gdn_gating_metal) ---
        let (beta, g) = {
            #[cfg(feature = "metal")]
            {
                if b.device().is_metal() {
                    let b_flat = b.to_dtype(dtype)?.contiguous()?.flatten_all()?;
                    let a_flat = a.to_dtype(dtype)?.contiguous()?.flatten_all()?;
                    let (beta_flat, g_flat) = crate::metal::gdn::fused_gdn_gating_metal(
                        &b_flat,
                        &a_flat,
                        &self.a,
                        &self.dt_bias,
                    )?;
                    let shape = b.shape();
                    (beta_flat.reshape(shape)?, g_flat.reshape(shape)?)
                } else {
                    self.compute_beta_g_cpu(&b, &a, dtype)?
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                self.compute_beta_g_cpu(&b, &a, dtype)?
            }
        };

        // --- v_per_group repeat ---
        let (q, k) = if v_per_group > 1 {
            let q = q
                .unsqueeze(2)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            let k = k
                .unsqueeze(2)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // --- l2_norm (2x, fused Metal kernel) ---
        let q = {
            #[cfg(feature = "metal")]
            {
                if q.device().is_metal() {
                    crate::metal::gdn::l2_norm_metal(&q, 1e-6)?
                } else {
                    l2_norm(&q, 1e-6)?
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                l2_norm(&q, 1e-6)?
            }
        };
        let k = {
            #[cfg(feature = "metal")]
            {
                if k.device().is_metal() {
                    crate::metal::gdn::l2_norm_metal(&k, 1e-6)?
                } else {
                    l2_norm(&k, 1e-6)?
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                l2_norm(&k, 1e-6)?
            }
        };

        // --- recurrence ---
        let y = {
            #[cfg(feature = "metal")]
            {
                if q.device().is_metal() {
                    self.recurrence_metal(&q, &k, &v, &g, &beta, batch_size, seq_len, cache, dtype)
                } else {
                    gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut cache.recurrent_state)
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut cache.recurrent_state)
            }
        }?;
        cache.seqlen_offset += seq_len;

        // --- RmsNormGated ---
        let z_shape = z.shape().clone();
        let y = self.norm.forward(
            &y.reshape(((), self.head_v_dim))?,
            &z.reshape(((), self.head_v_dim))?,
        )?;
        let y = y
            .reshape(z_shape)?
            .reshape((batch_size, seq_len, self.value_dim))?;

        // --- out matmul ---
        self.out.forward(&y)
    }

    fn compute_beta_g_cpu(&self, b: &Tensor, a: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let beta = candle_nn::ops::sigmoid(b)?;
        let g = self
            .a
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&softplus(
                &a.to_dtype(DType::F32)?.broadcast_add(
                    &self
                        .dt_bias
                        .to_dtype(DType::F32)?
                        .unsqueeze(0)?
                        .unsqueeze(0)?,
                )?,
            )?)?
            .to_dtype(dtype)?;
        Ok((beta, g))
    }

    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    fn recurrence_metal(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        batch_size: usize,
        seq_len: usize,
        cache: &mut GdnLayerCache,
        dtype: DType,
    ) -> Result<Tensor> {
        let num_heads = self.num_v_heads;
        let k_head = self.head_k_dim;
        let v_head = self.head_v_dim;
        let scale = 1.0 / (k_head as f64).sqrt();

        if seq_len == 1 {
            // Fast decode path: skip transpose+contiguous.
            // For S=1, reshape from [B, S, H, D] to [B*H, S, D] is a valid view
            // because the S dimension (size 1) doesn't break contiguity between B and H.
            let flat = q.reshape((batch_size * num_heads, k_head))?;
            let q_bh = (flat.unsqueeze(1)?.to_dtype(DType::F32)? * scale)?;
            let k_bh = k
                .reshape((batch_size * num_heads, k_head))?
                .unsqueeze(1)?
                .to_dtype(DType::F32)?;
            let v_bh = v
                .reshape((batch_size * num_heads, v_head))?
                .unsqueeze(1)?
                .to_dtype(DType::F32)?;
            let g_bh = g
                .reshape((batch_size * num_heads,))?
                .unsqueeze(1)?
                .to_dtype(DType::F32)?;
            let beta_bh = beta
                .reshape((batch_size * num_heads,))?
                .unsqueeze(1)?
                .to_dtype(DType::F32)?;

            let mut state_flat = cache.recurrent_state.to_dtype(DType::F32)?.reshape((
                batch_size * num_heads,
                k_head,
                v_head,
            ))?;

            let out_bh = crate::metal::gdn::gated_delta_rule_recurrence_metal(
                &q_bh,
                &k_bh,
                &v_bh,
                &g_bh,
                &beta_bh,
                &mut state_flat,
            )?;

            cache.recurrent_state = state_flat
                .reshape((batch_size, num_heads, k_head, v_head))?
                .to_dtype(cache.recurrent_state.dtype())?;

            // For S=1, reshape from [B*H, S, V] to [B, S, H, V] is a valid view,
            // no transpose needed (H immediately follows B in memory).
            return out_bh
                .reshape((batch_size, seq_len, num_heads, v_head))?
                .to_dtype(dtype);
        }

        let q_bh = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?.reshape((
            batch_size * num_heads,
            seq_len,
            k_head,
        ))?;
        let k_bh = k
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .reshape((batch_size * num_heads, seq_len, k_head))?;
        let v_bh = v
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .reshape((batch_size * num_heads, seq_len, v_head))?;
        let g_bh = g
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size * num_heads, seq_len))?;
        let beta_bh = beta
            .to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size * num_heads, seq_len))?;

        let mut state_flat = cache.recurrent_state.to_dtype(DType::F32)?.reshape((
            batch_size * num_heads,
            k_head,
            v_head,
        ))?;

        let out_bh = if seq_len >= 64 {
            crate::metal::gdn::chunked_gated_delta_rule_recurrence_metal(
                &q_bh,
                &k_bh,
                &v_bh,
                &g_bh,
                &beta_bh,
                &mut state_flat,
            )?
        } else {
            crate::metal::gdn::gated_delta_rule_recurrence_metal(
                &q_bh,
                &k_bh,
                &v_bh,
                &g_bh,
                &beta_bh,
                &mut state_flat,
            )?
        };

        cache.recurrent_state = state_flat
            .reshape((batch_size, num_heads, k_head, v_head))?
            .to_dtype(cache.recurrent_state.dtype())?;

        out_bh
            .reshape((batch_size, num_heads, seq_len, v_head))?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(dtype)
    }

    fn causal_conv1d_update(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (_batch, seq_len, _conv_dim) = x.dims3()?;
        let x_t = x
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(cache.conv_state.dtype())?;

        #[cfg(feature = "metal")]
        if x_t.device().is_metal() {
            let conv_state = cache.conv_state.contiguous()?;
            let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
                &x_t,
                &self.conv1d_weight,
                &conv_state,
                true,
                self.conv_kernel_size,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        let state_len = cache.conv_state.dim(2)?;
        let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;
        let weight = self.conv1d_weight.to_dtype(hidden_new.dtype())?;
        let mut outs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - self.conv_kernel_size, self.conv_kernel_size)?;
            outs.push((window * weight.unsqueeze(0)?)?.sum(D::Minus1)?);
        }
        candle_nn::ops::silu(&Tensor::stack(&outs, 2)?)?.transpose(1, 2)
    }

    fn causal_conv1d_full(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let x_t = x
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(cache.conv_state.dtype())?;

        #[cfg(feature = "metal")]
        if x_t.device().is_metal() {
            let (output, new_conv_state) = crate::metal::gdn::causal_conv1d_metal(
                &x_t,
                &self.conv1d_weight,
                &cache.conv_state,
                false,
                self.conv_kernel_size,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        let pad_width = self.conv_kernel_size.saturating_sub(seq_len);
        cache.conv_state = if pad_width > 0 {
            let zeros =
                Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
            Tensor::cat(&[zeros, x_t.clone()], 2)?
        } else {
            x_t.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)?
        };
        let padded_t = Tensor::cat(
            &[
                Tensor::zeros(
                    (batch_size, conv_dim, self.conv_kernel_size - 1),
                    x_t.dtype(),
                    x_t.device(),
                )?,
                x_t,
            ],
            2,
        )?;
        let weight = self.conv1d_weight.to_dtype(padded_t.dtype())?;
        let mut outs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let window = padded_t.narrow(2, i, self.conv_kernel_size)?;
            outs.push((window * weight.unsqueeze(0)?)?.sum(D::Minus1)?);
        }
        candle_nn::ops::silu(&Tensor::stack(&outs, 2)?)?.transpose(1, 2)
    }
}

enum LayerImpl {
    Full(FullAttention),
    Linear(LinearAttention),
}

struct LayerWeights {
    layer: LayerImpl,
    attn_norm: QRmsNorm,
    ffn_norm: QRmsNorm,
    mlp: Mlp,
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
    mrope: Arc<Qwen3VLRotaryEmbedding>,
}

struct PropsGGUF {
    head_count: usize,
    head_count_kv: usize,
    block_count: usize,
    embedding_length: usize,
    rms_norm_eps: f32,
    max_seq_len: usize,
    rope_freq_base: f32,
    rope_dimension_count: usize,
    rope_dimension_sections: Vec<usize>,
    key_length: usize,
    full_attention_interval: usize,
    conv_kernel: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
}

fn verify_qwen35_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;
    if actual_arch != "qwen35" {
        candle_core::bail!("Expected `qwen35` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen35_arch(c.metadata)?;
        c.has_required_keys(&[
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "feed_forward_length",
            "attention.layer_norm_rms_epsilon",
            "ssm.conv_kernel",
            "ssm.state_size",
            "ssm.group_count",
            "ssm.time_step_rank",
            "ssm.inner_size",
        ])?;
        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;
        let linear_num_value_heads = c.get_value::<u32>("ssm.time_step_rank")? as usize;
        let linear_value_head_dim =
            c.get_value::<u32>("ssm.inner_size")? as usize / linear_num_value_heads;
        Ok(Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN) as usize,
            rope_freq_base: c
                .get_value("rope.freq_base")
                .ok()
                .unwrap_or(DEFAULT_ROPE_FREQ_BASE),
            rope_dimension_count: c
                .get_value::<u32>("rope.dimension_count")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(
                    ((embed_len / head_count) as f32 * DEFAULT_PARTIAL_ROTARY_FACTOR) as usize,
                ),
            rope_dimension_sections: c
                .get_value::<Vec<u32>>("rope.dimension_sections")
                .ok()
                .map(|xs| xs.into_iter().map(|x| x as usize).collect())
                .unwrap_or_else(|| DEFAULT_MROPE_SECTION.to_vec()),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            full_attention_interval: c
                .get_value::<u32>("full_attention_interval")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(DEFAULT_FULL_ATTENTION_INTERVAL),
            conv_kernel: c.get_value::<u32>("ssm.conv_kernel")? as usize,
            linear_key_head_dim: c.get_value::<u32>("ssm.state_size")? as usize,
            linear_value_head_dim,
            linear_num_key_heads: c.get_value::<u32>("ssm.group_count")? as usize,
            linear_num_value_heads,
        })
    }
}

fn qmethod(tensor: candle_core::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(tensor),
        b: None,
    })?))
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        if matches!(attention_mechanism, AttentionImplementation::PagedAttention) {
            candle_core::bail!("Qwen3.5 GGUF does not support PagedAttention yet.");
        }
        let actual_arch = verify_qwen35_arch(ct.get_metadata())?;
        let props = PropsGGUF::try_from(ContentMetadata {
            path_prefix: &actual_arch,
            metadata: ct.get_metadata(),
        })
        .or_else(|err| candle_core::bail!("{err}"))?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, props.rms_norm_eps)?;
        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };
        let layer_types = (0..props.block_count)
            .map(|i| {
                if (i + 1) % props.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect::<Vec<_>>();

        let mrope = Arc::new(Qwen3VLRotaryEmbedding::new(
            props.rope_freq_base,
            props.rope_dimension_count,
            device,
            props.rope_dimension_sections.clone(),
        )?);
        let mut layers = Vec::with_capacity(props.block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..props.block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let layer = match layer_types[layer_idx] {
                LayerType::FullAttention => LayerImpl::Full(FullAttention {
                    q: qmethod(ct.tensor(&format!("{prefix}.attn_q.weight"), device)?)?,
                    k: qmethod(ct.tensor(&format!("{prefix}.attn_k.weight"), device)?)?,
                    v: qmethod(ct.tensor(&format!("{prefix}.attn_v.weight"), device)?)?,
                    o: qmethod(ct.tensor(&format!("{prefix}.attn_output.weight"), device)?)?,
                    q_norm: QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                        props.rms_norm_eps,
                    )?,
                    k_norm: QRmsNorm::new(
                        ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                        props.rms_norm_eps,
                    )?,
                    n_head: props.head_count,
                    n_kv_head: props.head_count_kv,
                    head_dim: props.key_length,
                    rotary: mrope.clone(),
                    sdpa_params: SdpaParams {
                        n_kv_groups: props.head_count / props.head_count_kv,
                        softcap: None,
                        softmax_scale: 1.0 / (props.key_length as f32).sqrt(),
                        sliding_window: None,
                        sinks: None,
                    },
                    dtype,
                }),
                LayerType::LinearAttention => {
                    let conv_dim = props.linear_key_head_dim * props.linear_num_key_heads * 2
                        + props.linear_value_head_dim * props.linear_num_value_heads;
                    LayerImpl::Linear(LinearAttention {
                        qkv: qmethod(ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?)?,
                        z: qmethod(ct.tensor(&format!("{prefix}.attn_gate.weight"), device)?)?,
                        beta: qmethod(ct.tensor(&format!("{prefix}.ssm_beta.weight"), device)?)?,
                        alpha: qmethod(ct.tensor(&format!("{prefix}.ssm_alpha.weight"), device)?)?,
                        conv1d_weight: ct
                            .tensor(&format!("{prefix}.ssm_conv1d.weight"), device)?
                            .dequantize(device)?
                            .reshape((conv_dim, props.conv_kernel))?
                            .to_dtype(dtype)?
                            .contiguous()?,
                        dt_bias: ct
                            .tensor(&format!("{prefix}.ssm_dt.bias"), device)?
                            .dequantize(device)?,
                        a: ct
                            .tensor(&format!("{prefix}.ssm_a"), device)?
                            .dequantize(device)?,
                        norm: RmsNormGated {
                            weight: ct
                                .tensor(&format!("{prefix}.ssm_norm.weight"), device)?
                                .dequantize(device)?,
                            eps: props.rms_norm_eps as f64,
                        },
                        out: qmethod(ct.tensor(&format!("{prefix}.ssm_out.weight"), device)?)?,
                        num_k_heads: props.linear_num_key_heads,
                        num_v_heads: props.linear_num_value_heads,
                        head_k_dim: props.linear_key_head_dim,
                        head_v_dim: props.linear_value_head_dim,
                        conv_kernel_size: props.conv_kernel,
                        key_dim: props.linear_key_head_dim * props.linear_num_key_heads,
                        value_dim: props.linear_value_head_dim * props.linear_num_value_heads,
                    })
                }
            };
            layers.push(LayerWeights {
                layer,
                attn_norm: QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?,
                    props.rms_norm_eps,
                )?,
                ffn_norm: QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?,
                    props.rms_norm_eps,
                )?,
                mlp: Mlp {
                    gate: qmethod(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
                    down: qmethod(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
                    up: qmethod(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
                },
            });
        }

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: layer_types
                .iter()
                .map(|lt| match lt {
                    LayerType::FullAttention => HybridLayerType::Attention,
                    LayerType::LinearAttention => HybridLayerType::Recurrent,
                })
                .collect(),
            max_seq_len: props.max_seq_len,
            recurrent: RecurrentLayerConfig {
                conv_dim: props.linear_key_head_dim * props.linear_num_key_heads * 2
                    + props.linear_value_head_dim * props.linear_num_value_heads,
                conv_width: props.conv_kernel,
                state_dims: vec![
                    props.linear_num_value_heads,
                    props.linear_key_head_dim,
                    props.linear_value_head_dim,
                ],
            },
        };
        let cache = EitherCache::Hybrid(Arc::new(std::sync::Mutex::new(HybridCache::new(
            hybrid_cache_config,
            dtype,
            device,
        )?)));

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, props.embedding_length),
            layers,
            norm,
            output: qmethod(output)?,
            device: device.clone(),
            cache,
            max_seq_len: props.max_seq_len,
            mapper: Some(mapper),
            dtype,
            mrope,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(
            Vec<(Tensor, Tensor)>,
            &crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
        )>,
    ) -> Result<Tensor> {
        if metadata.is_some() {
            candle_core::bail!("Qwen3.5 GGUF does not support PagedAttention yet.");
        }
        let mut layer_in = self.tok_embeddings.forward(x)?;
        // Workaround: ensure stale private buffer cache from previous command buffer
        // doesn't contaminate this forward pass before the layer loop begins.
        let _ = layer_in.device().synchronize();
        let (batch, seq_len) = x.dims2()?;
        let mut positions = Vec::with_capacity(3 * batch * seq_len);
        for _axis in 0..3 {
            for &offset in start_offsets {
                for pos in offset..offset + seq_len {
                    positions.push(pos as u32);
                }
            }
        }
        let position_ids = Tensor::from_vec(positions, (3, batch, seq_len), x.device())?;
        let cos_sin = self.mrope.compute_cos_sin(&position_ids, DType::F32)?;

        let mask = CausalMasker.make_causal_mask(
            x,
            &start_offsets as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        let mut hybrid_cache = self.cache.hybrid();
        let single_idx = Tensor::new(&[0u32], x.device())?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }

            // attn_norm
            let residual_pre = &layer_in;
            let x_norm = layer.attn_norm.forward(&layer_in)?;

            let attn = match (&layer.layer, hybrid_cache.get_mut(i)) {
                (LayerImpl::Full(attn), Some(HybridLayerCache::Attention(kv_cache))) => {
                    attn.forward(&x_norm, &mask.get(x_norm.device()), &cos_sin, kv_cache)?
                }
                (LayerImpl::Linear(attn), Some(HybridLayerCache::Recurrent(pool))) => {
                    let conv_state = pool.gather_conv_state(&single_idx)?;
                    let recurrent_state = pool.gather_recurrent_state(&single_idx)?;
                    let conv_dtype = conv_state.dtype();
                    let recurrent_dtype = recurrent_state.dtype();
                    let mut gdn_cache = GdnLayerCache {
                        conv_state,
                        recurrent_state,
                        seqlen_offset: pool.get_seqlen_offset(0),
                    };
                    let out = attn.forward(&x_norm, &mut gdn_cache)?;
                    // Use slot-based scatter (no to_vec1 sync) to avoid GPU pipeline drain.
                    // The slot index is always 0 for single-sequence (batch=1) decode.
                    let slot_idx = 0;
                    pool.scatter_conv_state_slot(
                        slot_idx,
                        &gdn_cache.conv_state.to_dtype(conv_dtype)?,
                    )?;
                    pool.scatter_recurrent_state_slot(
                        slot_idx,
                        &gdn_cache.recurrent_state.to_dtype(recurrent_dtype)?,
                    )?;
                    pool.set_seqlen_offset(slot_idx, gdn_cache.seqlen_offset);
                    out
                }
                _ => candle_core::bail!("Qwen3.5 GGUF layer/cache type mismatch at layer {i}"),
            };

            // residual add
            let x = (attn.to_dtype(residual_pre.dtype())? + residual_pre)?;
            let residual_post = &x;

            // ffn_norm
            let x = layer.ffn_norm.forward(&x)?;

            // MLP + residual
            layer_in = {
                let mlp_out = layer.mlp.forward(&x)?;
                (mlp_out.to_dtype(residual_post.dtype())? + residual_post)?
            };
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        let logits = self.output.forward(&x.contiguous()?)?;
        Ok(logits)
    }
}
