# CUDA Full Attention GPU Implementation

**Date:** 2026-05-28
**Model:** Qwen3.5-35B-A3B-GPTQ-Int4
**Approach:** 移植 metal_infer kernel 6-9（方案 A）

## Goal

将 Full Attention 的 attention compute（scores/softmax/context/gate）从有 bug 的 CPU 实现替换为正确的 CUDA kernel。Q/K/V/O 投影已在 GPU（BF16 matvec），RoPE 已有 CUDA kernel。

## Architecture

4 个 CUDA kernel，从 metal_infer `shaders.metal` kernel 6-9 移植，用 CUDA warp shuffle 替代 Metal `simd_sum`。

### GPU Memory

**持久化分配（KV cache）：**
- `d_K_cache`: [MAX_SEQ, NUM_KV_HEADS * HEAD_DIM] float = [8192, 512] — 16MB
- `d_V_cache`: [MAX_SEQ, NUM_KV_HEADS * HEAD_DIM] float = [8192, 512] — 16MB

**每层临时分配：**
- `d_q`: [NUM_ATTN_HEADS * HEAD_DIM] = [4096] — Q 投影输出
- `d_q_gate`: [4096] — Q gate 投影输出
- `d_k`: [NUM_KV_HEADS * HEAD_DIM] = [512] — K 投影输出
- `d_v`: [512] — V 投影输出
- `d_q_normed`: [4096] — Q per-head norm 后
- `d_k_normed`: [512] — K per-head norm 后
- `d_q_rope`: [4096] — Q RoPE 后
- `d_k_rope`: [512] — K RoPE 后
- `d_scores`: [NUM_ATTN_HEADS * MAX_SEQ] = [16 * 8192] — attention scores
- `d_context`: [4096] — attention 输出

## Kernels

### Kernel 1: attn_scores（Q @ K^T / √d）

移植自 metal `attn_scores_batched`。

```
Grid: (seq_len, num_heads) threadgroups
Block: 256 threads
GQA: kv_h = head / heads_per_kv (8:1)

每个 TG 计算一个 (position, head) 对的 dot product:
  acc = sum_d(Q[head, d] * K_cache[pos, kv_h * head_dim + d])
  Warp shuffle reduction → shared memory → 单线程写入
  scores[head * MAX_SEQ + pos] = acc * scale
```

### Kernel 2: attn_softmax（in-place）

移植自 metal `attn_softmax_batched`。

```
Grid: num_heads threadgroups
Block: 256 threads

三趟:
  Pass 1: warp reduce 找 max
  Pass 2: exp(x - max)，warp reduce 累加 sum
  Pass 3: 归一化 x /= sum
```

### Kernel 3: attn_values（softmax @ V）

移植自 metal `attn_values_batched`。

```
Grid: num_heads * head_dim threads (= 4096)
每个线程一个 (head, dim):
  acc = sum_p(scores[head * seq_stride + p] * V_cache[p * kv_dim + kv_h * head_dim + d])
  context[head * head_dim + d] = acc
```

### Kernel 4: sigmoid_gate（in-place）

移植自 metal `sigmoid_gate`。

```
Grid: (num_heads * head_dim + 255) / 256 blocks
逐元素: context[i] *= sigmoid(q_gate[i])
```

## Q/K Per-Head RMS Norm

复用已有 `cuda_rms_norm`，对每头单独调用：
- Q: 16 heads × 256 dim → 16 次 kernel launch（每次 dim=256）
- K: 2 heads × 256 dim → 2 次 kernel launch
- 权重 `q_norm.weight [256]` / `k_norm.weight [256]`，所有头共享

单 token decode 时 18 次 launch 开销可忽略（后续可 fuse 优化）。

## 数据流（单层 Full Attention）

```
d_hidden [2048]
  │
  ├─ cuda_rms_norm → d_normed [2048]
  │
  ├─ cuda_bf16_matvec (Q_proj weight) → d_q_proj [8192]
  │   split: d_q [0:4096], d_q_gate [4096:8192]
  ├─ cuda_bf16_matvec (K_proj weight) → d_k [512]
  └─ cuda_bf16_matvec (V_proj weight) → d_v [512]
  │
  ├─ per-head cuda_rms_norm (16 calls, dim=256) → d_q_normed [4096]
  └─ per-head cuda_rms_norm (2 calls, dim=256) → d_k_normed [512]
  │
  ├─ cuda_rope(d_q_normed) → d_q_rope [4096]
  └─ cuda_rope(d_k_normed) → d_k_rope [512]
  │
  ├─ cudaMemcpy: Q/K → CPU for KV cache → 更新 CPU KV cache
  ├─ 将 updated K/V cache 拷贝到 GPU d_K_cache, d_V_cache
  │
  ├─ attn_scores(d_q_rope, d_K_cache) → d_scores [16 * seq_len]
  ├─ attn_softmax(d_scores, seq_len, MAX_SEQ)
  ├─ attn_values(d_scores, d_V_cache) → d_context [4096]
  └─ sigmoid_gate(d_context, d_q_gate)
  │
  └─ cuda_bf16_matvec (O_proj weight) → d_attn_out [2048]
```

## KV Cache 策略

将 KV cache 从 CPU `malloc` 迁移到 GPU 持久化分配：
- `d_K_cache`, `d_V_cache` 在初始化时分配（16MB 各），不再释放
- 每 token 追加：`cudaMemcpy(d_K_cache + kv->len * kv_dim, d_k, ...)`
- 消除 attention compute 前的 KV cache CPU→GPU 拷贝

## 验证策略

每个 kernel 用 PyTorch 参考实现独立验证，测试文件放 `cuda_infer/tests/`：

| CUDA Kernel | PyTorch 参考 |
|---|---|
| `attn_scores` | `torch.einsum('hd,phd->hp', Q, K) / sqrt(d)` |
| `attn_softmax` | `F.softmax(scores, dim=-1)` |
| `attn_values` | `torch.einsum('hp,phd->hd', attn_w, V)` |
| `sigmoid_gate` | `context * torch.sigmoid(gate)` |

## 文件变更

| File | Change |
|------|--------|
| `kernels.cu` | 新增 4 个 kernel + extern "C" wrapper |
| `kernels.h` | 新增 4 个 kernel 声明 |
| `infer.cu` | 重写 `forward_full_attention_cpu` → `forward_full_attention_gpu`，GPU KV cache |
| `tests/` | 新增大 `test_attn_scores.py`, `test_attn_softmax.py`, `test_attn_values.py`, `test_sigmoid_gate.py` |

## Success Criteria

1. `make` 编译成功
2. 4 个新 test 全部通过
3. 全部 12 个 test（8 旧 + 4 新）通过
4. `./infer --prompt "Hello" --tokens 10` 40 层跑完，无 CUDA error
5. Full attention 层 hidden state 无 NaN
