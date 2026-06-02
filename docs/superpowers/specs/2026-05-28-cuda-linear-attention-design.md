# CUDA Linear Attention (GatedDeltaNet) GPU Implementation

**Date:** 2026-05-28
**Model:** Qwen3.5-35B-A3B-GPTQ-Int4
**Approach:** 移植 metal_infer kernel 10-14，GPU 持久状态

## Goal

将 30 层 Linear Attention (GatedDeltaNet) 从 CPU pass-through stub 替换为完整的 GPU 实现。

## Architecture

5 个 CUDA kernel 从 metal_infer `shaders.metal` kernel 10-14 移植。GPU 持久化分配所有层的状态 buffer（conv_state + ssm_state），CPU 仅做权重加载和 kernel launch 调度。

### Dimension Constants (35B model)

```
HIDDEN_DIM          = 2048
LINEAR_NUM_V_HEADS  = 64
LINEAR_NUM_K_HEADS  = 16
LINEAR_KEY_DIM      = 128
LINEAR_VALUE_DIM    = 128
LINEAR_TOTAL_KEY    = LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM   = 2048
LINEAR_TOTAL_VALUE  = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM = 8192
LINEAR_CONV_DIM     = LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE = 12288
CONV_KERNEL_SIZE    = 4
```

### Persistent GPU State (per layer, 30 layers)

```
d_conv_state[3 * 12288] * sizeof(float) = 144KB
d_ssm_state[64 * 128 * 128] * sizeof(float) = 4MB
Total per layer ≈ 4.14MB
Total 30 layers ≈ 124MB GPU memory
```

## Kernels

### Kernel 1: conv1d_step

移植自 metal `conv1d_step`。12288 threads，每 channel 独立。

```
对于每个 channel c:
  acc = state[0,c] * w[c,0] + state[1,c] * w[c,1] + state[2,c] * w[c,2] + input[c] * w[c,3]
  output[c] = silu(acc)
  state[0,c] = state[1,c]; state[1,c] = state[2,c]; state[2,c] = input[c]
```

### Kernel 2: compute_decay_beta

移植自 metal `compute_decay_beta`。64 threads (one per v-head)。

```
softplus_val = log(1 + exp(alpha + dt_bias))
g_decay = exp(-exp(A_log) * softplus_val)
beta_gate = sigmoid(beta)
```

### Kernel 3: rms_norm_qk

移植自 metal `rms_norm_qk`。16 threadgroups × 128 threads。

```
对每个 k-head h:
  q[h][:] = rms_norm(q[h][:]) * (1/128)    — inv_scale²
  k[h][:] = rms_norm(k[h][:]) * (1/sqrt(128)) — inv_scale
```

### Kernel 4: gated_delta_net_step

移植自 metal `gated_delta_net_step`。64 threadgroups × 128 threads。每个线程处理一行 `S[vh][vi][:]`。

```
对于每个 v-head vh, 每个 vi:
  kh = vh / 4
  Step 1: S[vi][ki] *= g_decay[vh]         (128元素, decay)
  Step 2: kv_mem = dot(S[vi][:], k[kh*128:])  (warp reduce)
  Step 3: delta = (v[vh*128+vi] - kv_mem) * beta_gate[vh]
  Step 4: S[vi][ki] += k[ki] * delta        (128元素, update)
  Step 5: out[vh*128+vi] = dot(S[vi][:], q[kh*128:])  (warp reduce)
```

### Kernel 5: gated_rms_norm

移植自 metal `gated_rms_norm`。64 threadgroups × 128 threads。

```
对于每个 v-head h:
  rms = rsqrt(sum(values[h:128]^2) / 128 + eps)
  out[h:128] = rms_norm * silu(z[h:128]) * weight (BF16, shared across heads)
```

## 数据流（单层）

```
d_hidden [2048]
  │
  ├─ cuda_rms_norm(input_layernorm.weight) → d_normed [2048]
  │
  ├─ gpu_bf16_matvec_cpu_io(in_proj_qkv.weight) → qkv [12288]
  ├─ gpu_bf16_matvec_cpu_io(in_proj_z.weight)   → z   [8192]
  ├─ gpu_bf16_matvec_cpu_io(in_proj_b.weight)   → beta [64]
  └─ gpu_bf16_matvec_cpu_io(in_proj_a.weight)   → alpha [64]
  │
  ├─ conv1d_step → conv_out [12288] (原地更新 conv_state)
  │
  ├─ 切分: q=conv_out[0:2048], k=[2048:4096], v=[4096:12288]
  │
  ├─ rms_norm_qk → q/k 归一化+缩放
  ├─ compute_decay_beta → g_decay[64], beta_gate[64]
  ├─ gated_delta_net_step → out_values [8192]
  ├─ gated_rms_norm → gated_out [8192]
  │
  └─ gpu_bf16_matvec_cpu_io(out_proj.weight, [8192]→[2048]) → d_attn_out [2048]
```

## Weight Format

线性注意力权重均为 BF16（非 GPTQ 量化），每层：

| Tensor | Shape | Size |
|--------|-------|------|
| `in_proj_qkv.weight` | [12288, 2048] | 48MB BF16 |
| `in_proj_z.weight` | [8192, 2048] | 32MB |
| `in_proj_b.weight` | [64, 2048] | 256KB |
| `in_proj_a.weight` | [64, 2048] | 256KB |
| `conv1d.weight` | [12288, 4] | 96KB |
| `A_log` | [64] | 256B |
| `dt_bias` | [64] | 128B |
| `norm.weight` | [128] | 256B |
| `out_proj.weight` | [2048, 8192] | 32MB |

## 验证策略

每个 kernel 用 PyTorch 参考实现独立验证，测试文件放 `cuda_infer/tests/`。

## 文件变更

| File | Change |
|------|--------|
| `kernels.cu` | 新增 5 个 kernel + extern "C" wrapper |
| `kernels.h` | 新增 5 个声明 |
| `infer.cu` | 重写 `forward_linear_attention`，GPU 状态管理，权重加载 |
| `tests/` | 新增 test_conv1d_step.py, test_decay_beta.py, test_rms_norm_qk.py, test_delta_net.py, test_gated_rms_norm.py |

## Success Criteria

1. `make` 编译成功
2. 5 个新 test 通过
3. 全部 17 个 test（12 旧 + 5 新）通过
4. `./infer` 40 层跑完，Linear attention 层无 NaN
5. 逐层 hidden state 与 PyTorch 参考差异 < 1e-3
