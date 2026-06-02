# CUDA MoE Routing + Expert Forward Design

**Date:** 2026-05-29
**Model:** Qwen3.5-35B-A3B-GPTQ-Int4

## Goal

实现 MoE routing（gate projection + softmax + topK）和 expert forward（GPTQ-Int4 dequant matvec），替换当前的占位 stub。

## Weight Formats

| 组件 | 格式 | 维度 | 大小 |
|------|------|------|------|
| Routing gate `mlp.gate.weight` | BF16 | [256, 2048] | 1MB |
| Expert gate_proj | GPTQ-Int4 | [512, 2048] | ~565KB |
| Expert up_proj | GPTQ-Int4 | [512, 2048] | ~565KB |
| Expert down_proj | GPTQ-Int4 | [2048, 512] | ~565KB |
| Shared gate_proj | BF16 | [512, 2048] | 2MB |
| Shared up_proj | BF16 | [512, 2048] | 2MB |
| Shared down_proj | BF16 | [2048, 512] | 2MB |
| Shared expert gate | BF16 | [2048] | 4KB |

## Data Flow (per layer)

```
post_attn_normed [2048] on GPU
  │
  ├─ GPU: Routing gate (BF16 matvec) → scores [256]
  ├─ CPU: cudaMemcpy scores → softmax + topK → indices[8], weights[8]
  │
  ├─ CPU: Load K=8 expert GPTQ weights from mmap → GPU (cudaMemcpy)
  │       gate_proj qweight+scales+qzeros+g_idx
  │       up_proj   qweight+scales+qzeros+g_idx
  │       down_proj qweight+scales+qzeros+g_idx
  │
  ├─ GPU: For each expert k: gate@x, up@x → SwiGLU → down@intermediate
  ├─ GPU: weighted_sum → moe_out [2048]
  │
  ├─ GPU: Shared gate/up BF16 matvec → SwiGLU → down BF16 matvec
  ├─ GPU: sigmoid(shared_gate @ post_normed) * shared_down → shared_out
  │
  └─ GPU: hidden += moe_out + shared_out
```

## New/Modified Components

### 1. Routing gate matvec (GPU)
已有 `cuda_bf16_matvec`，直接调用。

### 2. CPU softmax + topK
已有 `cpu_topk` 函数（infer.cu line 152），需添加 softmax 步骤。

### 3. Expert weight loading (CPU → GPU)
复用 `load_tensor_to_gpu`，按 expert index 构造 tensor name：
`layers.{N}.mlp.experts.{M}.{proj}.{qweight,scales,qzeros,g_idx}`

### 4. Expert forward (GPU)
已有 `cuda_dequant_matvec_gptq` 和 `cuda_swiglu`。

### 5. Shared expert forward (GPU)
已有 `cuda_bf16_matvec` 和 `cuda_swiglu`。

### 6. Shared gate (GPU)
新增一个小 kernel：`dot(shared_gate[2048], x[2048]) → sigmoid`，或者复用 BF16 matvec（shared_gate 当作 [1,2048] 的矩阵）。

## 验证策略

端到端验证：PyTorch + auto-gptq 加载模型，逐层对比 MoE 输出 + hidden state。

## File Changes

| File | Change |
|------|--------|
| `infer.cu` | 替换 MoE stub，实现完整 routing + expert forward + shared expert |

## Success Criteria

1. `make` 编译成功
2. 全部 17 个 test 通过
3. `./infer` 40 层跑完，MoE 路径不再使用占位
4. 逐层 hidden state 与 PyTorch 参考差异 < 1e-3
