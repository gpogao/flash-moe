# CUDA MoE Inference Engine Design

**Date:** 2026-05-27
**Model:** Qwen3.5-35B-A3B-GPTQ-Int4
**Target:** NVIDIA GB10 (compute 12.1, CUDA 13.0)

## Goals

Quick validation: Get Qwen3.5-35B MoE inference working on CUDA with minimal complexity.

## Architecture

**Directory:** `cuda_infer/`
```
cuda_infer/
├── infer.cu           # Main program + CUDA kernels
├── tokenizer.h         # From metal_infer (no changes)
├── tokenizer.bin       # From metal_infer
├── extract_weights.py  # safetensors → model_weights.bin
├── Makefile            # nvcc compilation
└── model_weights.bin   # Flat binary (mmap'd)
```

## Model Specs (from config.json)

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| vocab_size | 248320 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |
| shared_expert_intermediate_size | 512 |
| layer_types | 30 linear_attention + 10 full_attention (every 4th) |

## Quantization (GPTQ-Int4)

- bits: 4
- group_size: 128
- Format per expert layer: qweight + scales + qzeros + g_idx
- dynamic: lm_head, embed_tokens, attn, shared_expert excluded from quantization

## Implementation Approach

### Reuse from metal_infer

| Component | Reuse Strategy |
|-----------|----------------|
| tokenizer.h/tokenizer.bin | Direct copy |
| Model constants (HIDDEN_DIM, etc.) | Copy definitions |
| MoE routing (top-K softmax) | Copy CPU calculation code |
| SwiGLU activation | Copy computation logic |
| RMS norm | Copy CPU implementation |
| RoPE | Copy computation (CPU initially) |
| Program structure & CLI | Copy and adapt |

### Rewrite for CUDA

| Component | Implementation |
|-----------|----------------|
| 4-bit dequant kernel | Simple CUDA kernel (naive matvec) |
| Expert weight loading | mmap safetensors directly |
| Attention (linear) | cuBLAS + CPU fallback for GatedDeltaNet |
| Attention (full) | cuBLAS GEMV |
| Memory management | CUDA device malloc |

### Not Implemented (quick validation phase)

- SSD expert streaming (use mmap)
- FMA-optimized kernel
- GPU pipeline overlap
- Deferred CMD3 execution
- LZ4 compression

## Key Files

### infer.cu (~800 lines expected)

1. **Initialization**: CUDA context, device memory allocation, mmap weights
2. **Embedding**: Lookup + RMS norm
3. **Layer loop (40 layers)**:
   - Attention: linear (GatedDeltaNet) or full (RoPE + GEMV)
   - MoE routing: top-K selection
   - Expert forward: dequant kernel for active experts
   - Combine + residual + norm
4. **Output**: RMS norm → lm_head GEMV → sampling

### kernels.cu

```cuda
// Core kernels
__global__ void dequant_matvec_4bit_kernel(...)  // Naive 4-bit matvec
__global__ void swiglu_kernel(...)               // SwiGLU activation
__global__ void rms_norm_kernel(...)             // RMS normalization
__global__ void weighted_sum_kernel(...)         // MoE combine
```

### extract_weights.py

- Read safetensors index (model.safetensors.index.json)
- Map weight names to file offsets
- Reorganize into flat binary:
  - Non-expert weights: embed, all layers (input_layernorm, attention, mlp), norm, lm_head
  - Expert weights: per-layer, per-expert gate/up/down projections

## Data Layout

### model_weights.bin structure

```
[offset]  [size]  [description]
0x0000   9MB    embed_tokens.weight (248320 x 2048, BF16)
...      ...   layer 0 weights (input_layernorm, attn, linear_attn, etc.)
...      ...   layer 1..39
...      ...   final_layer_norm
...      ...   lm_head.weight
```

### Expert layout per layer

For 256 experts with gate/up/down projections:
```
gate_proj: [512, 2048]  quantized 4-bit → qweight + scales + qzeros + g_idx
up_proj:   [512, 2048]  quantized 4-bit
down_proj: [2048, 512]  quantized 4-bit
```

## Build & Run

```bash
cd cuda_infer
# Extract weights from safetensors
python extract_weights.py /data/ai/hf/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/ ./weights

# Build
make

# Run
./infer --prompt "Hello world" --tokens 50
```

## Success Criteria

1. Compilation succeeds with nvcc
2. Model loads weights from safetensors
3. Forward pass completes without crashes
4. Output is valid text (token decoding works)
5. Timing: not optimized yet, focus on correctness

## Phase 2 (future, not in scope)

- FMA-optimized kernels
- SSD expert streaming
- GPU pipeline overlap
- Performance tuning