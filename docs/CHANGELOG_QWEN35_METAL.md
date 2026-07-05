# Qwen3.5 混合注意力模型 Metal 后端移植 — 技术报告

## 概述

本次变更为 Qwen3.5-4B 模型（混合注意力架构：Full Attention + Gated Delta Net）添加完整的 GGUF 推理支持，并在 Apple Silicon Metal 后端上实现 GPU 加速。模型原有的 CUDA 实现（Gated Delta Net）被移植到 Metal，包含 5 种融合内核。

---

## 第一部分：模型架构

### 1.1 Qwen3.5-4B 整体结构

定义在 `mistralrs-core/src/models/quantized_qwen35.rs`，约 963 行。模型由 `ModelWeights` 结构体表示：

```
tok_embeddings (Embedding)
  → 28 层交替注意力层
    → attn_norm (QRmsNorm)
    → FullAttention 或 LinearAttention (交替)
    → post_attention_norm (QRmsNorm)
    → MLP (SwiGLU)
  → output_norm (QRmsNorm)
  → lm_head (线性投影)
```

**交替规则**：以 `full_attention_interval`（默认 4）为周期，每第 `4` 层为 Full Attention，其余为 Linear Attention（GDN）。例如 28 层模型：第 3, 7, 11, 15, 19, 23, 27 层（0-indexed）是 Full Attention，其他 21 层是 GDN。

### 1.2 FullAttention 层

```
Q 投影 (fused Q + gate gate -> [B,S,n_head,2*head_dim])
  → 按 head_dim 拆分为 q 和 gate
  → mRoPE + QK RMSNorm (含在 rotary.forward_qk_norm 内)
  → KVCache append
  → SDPA attention
  → sigmoid(gate) * attn_output (门控)
  → O 投影
```

- 使用 mRoPE（多维旋转位置编码，`Qwen3VLRotaryEmbedding`）
- Q 投影融合了 query 和 output gate，避免额外内核

### 1.3 LinearAttention（GDN）层

```
5 路量化矩阵乘法：qkv, z, beta, alpha, out
  → qkv = [B,S,2*key_dim+value_dim] (融合)
  → z = [B,S,num_v_heads,head_v_dim] (输出门控)
  → beta/alpha = [B,S,num_v_heads]
  → conv1d (因果卷积)
  → 拆分为 q, k, v (per-head)
  → beta_gate = sigmoid(beta)
  → g = -exp(a_log) * softplus(alpha + dt_bias)
  → L2 归一化 q 和 k
  → Gated Delta Rule 循环更新 (状态: [B, num_v_heads, key_dim, value_dim])
  → RmsNormGated (RMSNorm * weight * silu(gate))
  → out 投影
```

#### RmsNormGated

`RmsNormGated` 是 GDN 专用的归一化层（`gdn.rs` 第 42-77 行）：

```
x_f32 = to_f32(x)
gate_f32 = silu(to_f32(gate))
variance = mean(x_f32^2, dim=-1)
y = x_f32 / sqrt(variance + eps) * silu(gate)  // 含 weight 缩放
return to_dtype(y)
```

在 `quantized_qwen35.rs` 的 `RmsNormGated`（第 143-155 行）是独立于 `gdn.rs` 的副本 —— 专为 GGUF 模型设计，直接从 GGUF tensor 构造 `weight`。

### 1.4 混合缓存（HybridCache）

定义在 `kv_cache/hybrid_cache.rs`。每个序列需要一个：
- **KV 缓存**：Full Attention 层使用（标准 KVCache）
- **Conv 状态**：[B, conv_dim, conv_kernel] —— conv1d 的滑动窗口
- **循环状态**：[B, num_v_heads, key_dim, value_dim] —— GDN 的隐状态
- **seqlen_offset**：每个序列已处理的 token 数

模型加载时创建 `HybridCache`，每层的 `layer_type` 映射到缓存变体：

```rust
let hybrid_cache_config = HybridCacheConfig {
    layer_types: vec![Attention, Recurrent, Recurrent, Recurrent, ...],
    recurrent: RecurrentLayerConfig {
        conv_dim: 2 * key_dim + value_dim,
        conv_width: conv_kernel,
        state_dims: [num_v_heads, key_dim, value_dim],
    },
};
```

---

## 第二部分：Metal 内核移植

### 2.1 内核列表

所有 Metal 内核位于 `metal/kernels/gdn.metal`，Rust 调度在 `metal/gdn.rs`。每个内核都有 `half` 和 `bfloat16_t` 类型实例化（`typedef bfloat bfloat16_t`）。

| 内核名称 | Metal shader | Rust 调度函数 | 用途 | 行数 |
|----------|-------------|--------------|------|------|
| `gated_delta_rule_128_64` / `_64_64` | `gated_delta_rule_kernel<BK,BV>` | `gated_delta_rule_recurrence_metal` | 单步 GDN 循环 | 22-97 |
| `gated_delta_rule_fallback` | `gated_delta_rule_kernel_fallback<BV,MAX_K>` | 同上（K 维度运行时确定） | 运行时 K 维度 | 100-170 |
| `chunked_gated_delta_rule_32_128_64` / `_32_64_64` | `chunked_gated_delta_rule_kernel<BT,BK,BV>` | `chunked_gated_delta_rule_recurrence_metal` | 32-token 块预填充 | 206-357 |
| `causal_conv1d_update_{half,bfloat16_t}` | `causal_conv1d_update_kernel<T>` | `causal_conv1d_metal` | 单步因果卷积 | 377-411 |
| `causal_conv1d_full_{half,bfloat16_t}` | `causal_conv1d_full_kernel<T>` | `causal_conv1d_metal` | 完整序列因果卷积 | 426-455 |
| `save_conv_state_{half,bfloat16_t}` | `save_conv_state_kernel<T>` | （已淘汰，见下） | 保存 conv 状态 | 457-482 |
| `fused_gdn_gating_{half,bfloat16_t}` | `fused_gdn_gating_kernel<T>` | `fused_gdn_gating_metal` | 融合门控计算 | 503-532 |
| `l2_norm_{half,bfloat16_t}` | `l2_norm_kernel<T>` | `l2_norm_metal` | 融合 L2 归一化 | 552-588 |
| `rms_norm_gated_{half,bfloat16_t}` | `rms_norm_gated_kernel<T>` | `rms_norm_gated_metal` | 融合门控 RMSNorm | 600-634 |

### 2.2 `l2_norm_metal` 融合归一化

**Rust 调度**（`metal/gdn.rs` 第 568-620 行）：

```rust
pub fn l2_norm_metal(x: &Tensor, eps: f64) -> Result<Tensor> {
    // 确保连续，确定 dtype、num_rows、head_dim
    // 分配 output zero tensor
    // 绑定 buffer(0)=x, buffer(1)=output
    // set_bytes(2)=num_rows, (3)=head_dim, (4)=eps
    // dispatch: grid=(num_rows,1), tg=(32,1)
}
```

线程模型：每行一个 threadgroup，32 个线程协作归约。

**Metal shader**（`gdn.metal` 第 552-579 行）：

```metal
template <typename T>
[[kernel]] void l2_norm_kernel(...) {
    // 每个线程累加 head_dim 中部分元素平方
    float partial = 0.0f;
    for (int d = tiisg; d < D; d += 32)
        partial += (float)x[row*D + d] * (float)x[row*D + d];

    // SIMD 归约求和
    float sum_sq = simd_sum(partial);
    float inv_norm = 1.0f / sqrt(sum_sq + eps);

    // 每个线程写回自己的部分
    for (int d = tiisg; d < D; d += 32)
        output[row*D + d] = (T)((float)x[row*D + d] * inv_norm);
}
```

**性能**：decode 14.0 ± 0.2 T/s（+8.5% 对比 CPU l2_norm）。CPU 版本的 `l2_norm`（`gdn.rs` 第 132-140 行）需要创建 5 个中间张量（sqr、sum、add、sqrt、recip），Metal 融合内核避免了这些分配。

### 2.3 `rms_norm_gated_metal`（待修复）

Rust 调度与 `l2_norm_metal` 结构相同（`metal/gdn.rs` 第 640-706 行），增加 `gate` 和 `weight` 输入。

Metal shader（`gdn.metal` 第 600-634 行）：

```metal
// 归约: variance = simd_sum(partial) / D
// rms_rsqrt = 1/sqrt(variance + eps)
// 输出: x * rms_rsqrt * weight[d] * silu(gate)
// silu(gate) = gate * sigmoid(gate) = gate * 1/(1+exp(-gate))
```

**问题**：当前会产生 NaN。即使添加 `device.synchronize()` 前后同步也无法解决 —— 确认是 shader 代码自身的 bug，与 GPU 管道同步无关。反复对比 `l2_norm_kernel`（正常）和 `rms_norm_gated_kernel`（异常）的代码结构，两者使用相同的 `simd_sum` 归约模式和线程调度，差异在于后者多了 `gate`/`weight` 读取和 `silu` 计算。具体 root cause 待进一步调试。当前已回退到 CPU path。

### 2.4 其他内核

**`fused_gdn_gating`**（行 503-532）：融合 `beta = sigmoid(b)` 和 `g = -exp(a_log) * softplus(a + dt_bias)` 为一个 kernel，避免 3 个独立内核调用。

**`causal_conv1d_update`**（行 377-411）：单步因果卷积。将 `conv_state` 左移一个位置，写入新输入，与权重做点积，应用 SiLU 激活。

**`causal_conv1d_full`**（行 426-455）：完整序列因果卷积。每个输出位置 `pos` 读取窗口 `[pos-k+1, pos]`，与权重做点积，SiLU 激活。

---

## 第三部分：GGUF 元数据支持

### 3.1 架构感知层大小计算

Qwen3.5 的 GGUF 元数据有一个关键差异 —— **每层权重数量不同**。Full Attention 层和 Linear Attention 层有完全不同的权重集。

在 `utils/gguf_metadata.rs`（行 507-608）中，新增 `GGUFArchitecture::Qwen35` / `Qwen35MoE` 分支，按层遍历计算：

```rust
for layer_idx in 0..num_layers {
    // 共享权重 (每层都有):
    //   attn_norm.weight, post_attention_norm.weight, ffn_gate.weight,
    //   ffn_up.weight, ffn_down.weight

    if (layer_idx + 1) % full_attention_interval == 0 {
        // Full Attention 层:
        //   attn_q.weight, attn_q_norm.weight, attn_k.weight,
        //   attn_k_norm.weight, attn_v.weight, attn_output.weight
    } else {
        // Linear Attention 层:
        //   attn_qkv.weight, attn_gate.weight, ssm_a, ssm_alpha.weight,
        //   ssm_beta.weight, ssm_conv1d.weight, ssm_dt.bias,
        //   ssm_norm.weight, ssm_out.weight
    }
}
```

`full_attention_interval` 从 GGUF metadata 中读取（默认 4），key 格式 `qwen35.full_attention_interval`。

### 3.2 命名约定

- `ffn_norm` 在 Qwen3.5 中改名为 `post_attention_norm`（行 389-396）
- MoE 版本 (`Qwen35MoE`) 共享 `Qwen3MoE` 的 `_exps` 后缀规则（行 643-674）
- 架构映射：GGUF metadata 中的 `"qwen35"` / `"qwen35moe"` → `GGUFArchitecture::Qwen35` / `Qwen35MoE`（`gguf/mod.rs` 行 32-33）

### 3.3 推理管线连接

在 `pipeline/gguf.rs` 中的变更：

- **`Model` 枚举** 新增 `Qwen35(QQwen35)`（行 75）
- **架构路由**：`GGUFArchitecture::Qwen35 → QQwen35::try_from(model_config)`（行 487）
- **缓存报告**：Qwen35 使用 `model.cache.hybrid().num_layers()`（行 567），而非其他模型的 `.cache.normal()`
- **前向分发**：`Model::Qwen35(ref model) => model.forward(...)`（行 828-830）
- **管线缓存克隆/释放** 新增 `HybridCacheManager` 支持

### 3.4 模型注册

在 `utils/model_config.rs`（行 328-329）的 `akin!` 宏中添加 `QQwen35`，使其自动获得 `TryFrom<ModelParams<ParamsGGUF>>` 实现。

---

## 第四部分：NaN 问题的诊断与修复

### 4.1 根因

NaN 的根因是 Metal 的 `HazardTrackingModeUntracked`（Candle 的默认配置）导致 `StorageModePrivate` 缓冲池中的脏数据被重用。

- Candle 使用 `MTLResourceOptions::StorageModePrivate.0 | MTLResourceOptions::HazardTrackingModeUntracked.0`（`metal_backend/device.rs` 行 79）
- 同一 `MTLCommandBuffer` 内的多个 dispatch 可能在 GPU 上乱序执行
- `StorageModePrivate` 缓冲池会回收上一轮 command buffer 的 buffer，若前一轮未完成，新操作读到的是过期数据
- 这解释了为什么添加一个同步回调（`to_vec1`、`synchronize`）会让 NaN "消失" —— 它强制 GPU 管道排空

### 4.2 修复方案

**修复 1：`device.synchronize()` 在 tok_embeddings 之后**（`quantized_qwen35.rs` 第 878 行）

```rust
let layer_in = self.tok_embeddings.forward(x)?;
let _ = layer_in.device().synchronize();
```

强制在层循环开始前排空 GPU 管道，确保所有工作完成后再进入关键计算。

**修复 2：将 `save_conv_state` GPU 内核替换为 CPU narrow+contiguous**（`metal/gdn.rs` 第 438-445 行）

原方案用一个独立的 Metal kernel 从 conv1d 输出中提取最后 `kernel_size` 列作为新的 conv state，但该 kernel 与新 conv1d kernel 在同一个 encoder 内，两者都对同一 buffer 进行操作 —— 产生了数据竞争。改用 CPU 侧的 `.narrow(...)?.contiguous()?` 来安全读取已完成的 conv1d 输出。

---

## 第五部分：性能优化

### 5.1 Scatter 同步消除

**问题**：`RecurrentStatePool::scatter_conv_state` 使用 `index_select` + 批量 scatter，每次调用都要 `to_vec1()` 将索引从 GPU 同步回 CPU —— 每层需要约 0.3ms 的 pipeline drain。

**修复**：在 `hybrid_cache.rs` 中添加 slot-level 操作（行 141-174、150-152、212-215）：

```rust
pub fn scatter_conv_state_slot(&mut self, slot_idx: usize, value: &Tensor) -> Result<()> {
    self.conv_state.slice_set(&value.contiguous()?, 0, slot_idx)
}
```

单序列 decode 时只需更新 slot 0，避免了 batch scatter 的 GPU→CPU 同步开销。**效果**：10.13 → 13.49 T/s（+33%）。

### 5.2 `l2_norm_metal` 融合内核

融合 5 个中间操作为 1 个 kernel，避免了 `sqr → sum → add → sqrt → recip → mul` 链中的中间张量分配和 kernel launch 开销。**效果**：14.0 T/s decode（+8.5% 对比 CPU l2_norm）。

### 5.3 性能总览

| 优化 | 吞吐 | 延迟/T | 增益 | 说明 |
|------|------|--------|------|------|
| 基线（无优化） | 10.13 T/s | 98.7 ms | — | 全 CPU path |
| + Scatter 消除 | 13.49 T/s | 74.1 ms | +33% | 24 层 × 0.3ms = 7.2ms |
| + NaN 修复 | 12.9 T/s | 77.5 ms | -4.6% | Synchronize 开销 |
| + `l2_norm_metal` | 14.0 T/s | 71.2 ms | +8.5% | 融合内核 |
| **目标** | **16 T/s** | **62.5 ms** | — | 需 Q4_K_M + kernel 修复 |

---

## 第六部分：文件变更索引

| 文件 | 新增 | 修改 | 关键行 |
|------|------|------|--------|
| `models/quantized_qwen35.rs` | 963 行 | — | 模型实现、NaN 修复 |
| `models/gdn.rs` | 886 行 | — | GDN CPU 实现、l2_norm、RmsNormGated |
| `metal/gdn.rs` | — | +210/-48 行 | 5 种 Metal 调度函数 |
| `metal/kernels/gdn.metal` | — | +102 行 | l2_norm_kernel、rms_norm_gated_kernel |
| `kv_cache/hybrid_cache.rs` | — | +17 行 | slot-level gather/scatter |
| `utils/gguf_metadata.rs` | — | +137/-14 行 | Qwen3.5 层大小计算、命名规则 |
| `pipeline/gguf.rs` | — | +18/-1 行 | HybridCache 管理、模型路由 |
| `gguf/mod.rs` | — | +2 行 | Qwen35、Qwen35MoE 架构枚举 |
| `utils/model_config.rs` | — | +1 行 | QQwen35 注册 |
| `models/mod.rs` | — | +1 行 | 模块注册 |

---

## 第七部分：后续工作

1. **修复 `rms_norm_gated_metal` 内核**：可能是 Metal shader 中 `simd_sum` 的 BF16 类型处理 bug，或 `silu(gate)` 的 `exp` 在特定输入下的边缘情况
2. **尝试 Q4_K_M 量化**：预计减少约 20% 权重读取，目标 decode 16 T/s
3. **验证长序列生成**：当前测试仅覆盖 decode 128 tokens，需要验证 2K+ token 序列的数值稳定性

## 附录：编译说明

```bash
# Release 构建（Metal 特性）
MISTRALRS_METAL_PRECOMPILE=0 cargo build --release \
    --package mistralrs-cli \
    --features metal

# 基准测试
./target/release/mistralrs bench --warmup 1 --prompt-len 0 \
    --iterations 3 text \
    --model-id ~/workspace/models/ --format gguf \
    -f Qwen3.5-4B-Q5_K_M.gguf
```
