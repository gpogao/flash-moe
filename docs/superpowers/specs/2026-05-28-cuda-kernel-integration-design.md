# CUDA Kernel Integration Design

**Date:** 2026-05-28
**Model:** Qwen3.5-35B-A3B-GPTQ-Int4
**Target:** NVIDIA GB10 (compute 12.1, CUDA 13.0)

## Goal

Wire existing CUDA kernels (`kernels.cu`) into the inference loop (`infer.cu`) and fix/rewrite them to match the GPTQ-Int4 quantization format. Current state: kernels compile but are never called, and the CPU path has stub implementations (passthrough) that produce invalid output.

## Approach: Hybrid CPU-GPU (Option C)

- **GPU**: Heavy compute kernels (dequant matvec, SwiGLU, RMS norm, weighted sum, RoPE)
- **CPU**: Control flow, attention KV cache management, MoE routing (softmax + topK), weight loading

## Quantization Format Mismatch

metal_infer uses MLX affine: `dequant = nibble * scale + bias`, group_size=64.
This model uses GPTQ-Int4: `dequant = (nibble - qzero) * scale`, group_size=128.

The dequant kernel cannot be directly ported — it must be rewritten for GPTQ format.
Other kernels (SwiGLU, RMS norm, weighted sum, RoPE) are math-identical and port directly.

## Kernels to Fix/Rewrite

### 1. `dequant_matvec_gptq` (rewrite)

- Reference metal_infer `dequant_matvec_4bit_v3` structure (tiled, shared input cache, SIMD reduction)
- Adapt to GPTQ: `float w = (float)(nibble - qzero) * scale` instead of `nibble * scale + bias`
- Per-row scales/qzeros indexing: GPTQ stores scales per input group (group_size=128)
- CUDA warp shuffle reduction instead of Metal `simd_sum`
- Shared memory: `x_shared[2048]` (HIDDEN_DIM) or `x_shared[512]` (MOE_INTERMEDIATE)

### 2. `rms_norm` (fix)

- Current CUDA kernel does `cudaMemcpy` host sync for reduction — eliminate this
- Two-pass GPU-only: kernel1 reduce → sum_sq in device memory, kernel2 apply
- Reference metal_infer `rms_norm_sum_sq` + `rms_norm_apply` pattern

### 3. `swiglu` (minor fix)

- Already correct formula: `silu(gate) * up`
- Ensure launch config matches MOE_INTERMEDIATE=512

### 4. `weighted_sum` (direct port)

- Already correct. Expert outputs [K, HIDDEN_DIM] weighted by routing weights [K].

### 5. `rope` (fix)

- Current kernel has index confusion. Fix to match metal_infer pattern.
- Apply to rotary_dim=64 (HEAD_DIM * 0.25), not full head_dim.

### 6. `residual_add` (new, port from metal)

- Simple element-wise: `out[i] = a[i] + b[i]`

## Forward Pass Flow (per layer)

```
1. [GPU] RMS norm(hidden) → normed
2. [CPU] Attention:
   a. [GPU] Q/K/V = dequant_matvec_gptq(normed)
   b. [GPU] Q/K per-head RMS norm
   c. [GPU] RoPE on Q/K (rotary_dim=64)
   d. [CPU] Update KV cache
   e. [GPU] attn_scores = Q @ K^T / sqrt(head_dim)
   f. [GPU] softmax
   g. [GPU] context = softmax @ V
   h. [CPU] Apply sigmoid gate to context (or GPU kernel)
   i. [GPU] O_proj = dequant_matvec_gptq(context)
3. [GPU] Residual add: hidden += attn_output
4. [GPU] RMS norm(hidden) → post_normed
5. [CPU] MoE routing: gate matvec → softmax → topK
6. For each topK expert:
   a. [GPU] gate = dequant_matvec_gptq(post_normed, expert.gate_proj)
   b. [GPU] up   = dequant_matvec_gptq(post_normed, expert.up_proj)
   c. [GPU] SwiGLU(gate, up) → intermediate
   d. [GPU] expert_out = dequant_matvec_gptq(intermediate, expert.down_proj)
7. [GPU] Weighted sum of expert outputs
8. [GPU] Shared expert (same as step 6 but with shared weights)
9. [GPU] Residual add: hidden += moe_output + shared_output
```

## Weight Loading Strategy

- `model_weights.bin` is already mmap'd via `open_weights()`
- `load_tensor()` copies from mmap to host buffer
- New: `load_tensor_to_gpu()` that loads directly to device memory via `cudaMemcpy`
- Layer buffers: pre-allocated GPU memory, reused across layers
- Expert weights: loaded per-layer, per-token (K=8 experts, ~6.75MB each at 4-bit)

## GPU Memory Layout

Pre-allocated (persistent across layers):
- `d_hidden`: [HIDDEN_DIM] float — current hidden state
- `d_normed`: [HIDDEN_DIM] float — after RMS norm
- `d_gate`: [MOE_INTERMEDIATE] float — gate projection output
- `d_up`: [MOE_INTERMEDIATE] float — up projection output
- `d_swiglu_out`: [MOE_INTERMEDIATE] float — SwiGLU result
- `d_expert_out`: [K * HIDDEN_DIM] float — all expert outputs
- `d_combined`: [HIDDEN_DIM] float — weighted sum result
- `d_rms_sum_sq`: [1] float — RMS norm reduction temp

Per-attention (full attention layers only, 10 of 40):
- `d_Q`: [NUM_ATTN_HEADS * HEAD_DIM] float
- `d_K`: [NUM_KV_HEADS * HEAD_DIM] float
- `d_V`: [NUM_KV_HEADS * HEAD_DIM] float
- `d_K_cache`: [MAX_SEQ * NUM_KV_HEADS * HEAD_DIM] float
- `d_V_cache`: [MAX_SEQ * NUM_KV_HEADS * HEAD_DIM] float

## File Changes

| File | Change |
|------|--------|
| `kernels.cu` | Rewrite dequant for GPTQ, fix rms_norm, fix rope, add residual_add |
| `infer.cu` | Replace CPU stubs with GPU kernel calls, add GPU weight loading helpers |
| `Makefile` | No changes needed (already links both) |

## Attention Strategy

### Full Attention (10 layers, every 4th starting at 3) — Phase 1 GPU

Q/K/V/O 投影都是 dequant matvec（已修正），RoPE 已有 kernel。每层计算：

```
GPU: Q/K/V 投影 (dequant matvec GPTQ) → Q per-head norm → K per-head norm → RoPE
CPU: KV cache 管理 → attention scores (Q@K^T/√d) → softmax → context (softmax@V)
CPU: sigmoid gate (context * sigmoid(q_gate))
GPU: O 投影 (dequant matvec GPTQ)
```

Attention compute（scores + softmax + context）：16 heads × 256 dim，计算量小（~65K flops），CPU 足够。
将来可以移植 metal_infer kernel 6-9 到 CUDA 实现全 GPU attention。

### Linear Attention (30 layers) — Phase 1 CPU, Phase 2 GPU

GatedDeltaNet 涉及 5 个子操作（详见 metal_infer kernel 10-14）：

| Sub-operation | Metal Kernel | CUDA Status |
|---|---|---|
| Conv1d depthwise step | `conv1d_step` | Phase 2 |
| Gated delta recurrence | `gated_delta_net_step` | Phase 2 |
| Decay/beta gate compute | `compute_decay_beta` | Phase 2 |
| Gated RMS norm | `gated_rms_norm` | Phase 2 |
| Q/K per-head RMS norm | `rms_norm_qk` | Phase 2 |

Phase 1：参考 metal_infer 逻辑，在 CPU 上实现正确的 GatedDeltaNet（使用已有的 CPU BLAS）。
Phase 2：逐一移植上述 5 个 Metal kernel 到 CUDA。

## Validation Strategy: PyTorch Golden Reference

每个 GPU kernel 用 PyTorch 参考实现独立验证。测试脚本保存在 `cuda_infer/tests/`，可随时回归。

### Kernel 单元验证

| CUDA Kernel | PyTorch 等价 |
|---|---|
| `dequant_matvec_gptq` | 从 `qweight` 解包 4-bit → `torch.matmul(W_dequant, x)` |
| `swiglu` | `F.silu(gate) * up` |
| `rms_norm` | `x * torch.rsqrt((x**2).mean(-1, keepdim=True) + eps) * weight` |
| `weighted_sum` | `(weights.unsqueeze(-1) * expert_outs).sum(dim=0)` |
| `rope` | 手写旋转逻辑（复数乘法），等价 `apply_rotary_pos_emb` |
| `residual_add` | `a + b` |

### 验证流程

```
1. 生成随机输入（shape/dtype 匹配模型实际尺寸）
2. Python ctypes 调用 .so 中的 CUDA wrapper 函数
3. CPU 接收 GPU 输出
4. PyTorch 做同样计算 → 参考输出
5. assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-3)
```

### 端到端逐层验证

PyTorch + auto-gptq 加载完整模型，hook 每层 hidden_states 与 cuda_infer 逐层对比：

```
Max diff < 1e-3 则该层通过
```

### 测试文件结构

```
cuda_infer/
├── tests/
│   ├── test_dequant.py       # GPTQ dequant 4-bit
│   ├── test_swiglu.py        # SwiGLU
│   ├── test_rms_norm.py      # RMS norm
│   ├── test_weighted_sum.py  # Weighted sum
│   ├── test_rope.py          # RoPE
│   ├── test_residual_add.py  # Residual add
│   ├── test_layer_e2e.py     # 端到端逐层对比
│   └── conftest.py           # 共享 ctypes helper、随机输入生成
```

## Success Criteria

1. `make` compiles without errors
2. All `cuda_infer/tests/` unit tests pass: `uv run pytest cuda_infer/tests/ -v`
3. Single layer forward produces non-NaN, finite values
4. Full 40-layer forward pass completes without CUDA errors
5. Output tokens decode to valid UTF-8
