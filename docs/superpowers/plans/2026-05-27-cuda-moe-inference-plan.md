# CUDA MoE Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working CUDA inference engine for Qwen3.5-35B-A3B-GPTQ-Int4 on NVIDIA GB10 (compute 12.1)

**Architecture:** Minimal viable implementation focusing on correctness over performance. Reuse metal_infer code where possible. Single CUDA stream, CPU fallback for complex operations.

**Tech Stack:** CUDA 13.0, nvcc, cuBLAS, safetensors

---

## File Structure

```
cuda_infer/
├── infer.cu           # Main program + CUDA kernels (~800 lines)
├── kernels.cu         # CUDA kernel implementations (~400 lines)
├── tokenizer.h        # From metal_infer (copy, no changes)
├── tokenizer.bin      # From metal_infer (copy)
├── extract_weights.py # safetensors → model_weights.bin
├── Makefile           # nvcc compilation
└── model_weights.bin   # Flat binary weights (generated)
```

---

## Task 1: Project Setup and Tokenizer

**Files:**
- Create: `cuda_infer/Makefile`
- Create: `cuda_infer/tokenizer.h` (copy from metal_infer)
- Create: `cuda_infer/tokenizer.bin` (copy from metal_infer)

- [ ] **Step 1: Copy tokenizer files**

Run:
```bash
cp /home/perryshan/wlk/source/flash-moe/metal_infer/tokenizer.h /home/perryshan/wlk/source/flash-moe/cuda_infer/tokenizer.h
cp /home/perryshan/wlk/source/flash-moe/metal_infer/tokenizer.bin /home/perryshan/wlk/source/flash-moe/cuda_infer/tokenizer.bin
```

- [ ] **Step 2: Create Makefile**

Create: `/home/perryshan/wlk/source/flash-moe/cuda_infer/Makefile`

```makefile
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90 -std=c++17 -Xcompiler -Wall
CUDA_LIBS = -lcublas -lcudart

all: infer

infer: infer.cu kernels.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ infer.cu kernels.cu $(CUDA_LIBS)

clean:
	rm -f infer *.o

.PHONY: all clean
```

- [ ] **Step 3: Create empty placeholder files**

Run:
```bash
touch /home/perryshan/wlk/source/flash-moe/cuda_infer/infer.cu
touch /home/perryshan/wlk/source/flash-moe/cuda_infer/kernels.cu
touch /home/perryshan/wlk/source/flash-moe/cuda_infer/extract_weights.py
```

- [ ] **Step 4: Create basic stub that compiles**

Run:
```bash
cd /home/perryshan/wlk/source/flash-moe/cuda_infer
echo 'int main() { return 0; }' > infer.cu
echo '' > kernels.cu
make
```
Expected: Compilation succeeds (empty program links)

- [ ] **Step 5: Commit**

```bash
git add Makefile tokenizer.h tokenizer.bin infer.cu kernels.cu extract_weights.py
git commit -m "feat(cuda_infer): initial project setup"
```

---

## Task 2: extract_weights.py — Safetensors to Binary

**Files:**
- Create: `cuda_infer/extract_weights.py`

- [ ] **Step 1: Write extract_weights.py**

Create: `/home/perryshan/wlk/source/flash-moe/cuda_infer/extract_weights.py`

```python
#!/usr/bin/env python3
"""
Extract weights from safetensors to flat binary for CUDA inference.
Qwen3.5-35B-A3B-GPTQ-Int4 format.
"""

import json
import os
import struct
import numpy as np
from safetensors import safe_open

def main():
    if len(sys.argv) < 3:
        print("Usage: extract_weights.py <model_dir> <output_bin>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load model index
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    
    # Map weight files
    weight_map = index["weight_map"]
    unique_files = list(set(weight_map.values()))
    
    print(f"Found {len(unique_files)} safetensor files")
    print(f"Total tensors: {len(weight_map)}")
    
    # Categorize tensors
    embed_tokens = None
    lm_head = None
    layers = {}  # layer_idx -> list of (name, tensor)
    all_tensors = {}
    
    for name, file_path in weight_map.items():
        if "embed_tokens" in name:
            embed_tokens = (name, file_path)
        elif "lm_head" in name:
            lm_head = (name, file_path)
        elif "layers" in name:
            # Extract layer index
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        if layer_idx not in layers:
                            layers[layer_idx] = []
                        layers[layer_idx].append((name, file_path))
                        break
                    except ValueError:
                        continue
    
    print(f"Embedding: {embed_tokens[0] if embed_tokens else 'None'}")
    print(f"LM head: {lm_head[0] if lm_head else 'None'}")
    print(f"Layers: {len(layers)}")
    
    # Open all safetensor files
    file_handles = {}
    for fp in unique_files:
        full_path = os.path.join(model_dir, fp)
        file_handles[fp] = safe_open(full_path, framework="numpy")
    
    # Write binary
    with open(output_path, "wb") as out:
        offset = 0
        
        # Layout: header first (json manifest), then data
        # We'll write a simple layout:
        # [header_size: uint32]
        # [json manifest]
        # [tensor data]
        
        # First pass: collect all tensor info
        manifest = {
            "embed_tokens": None,
            "lm_head": None,
            "layers": {},
            "final_norm": None
        }
        
        tensor_data = []
        
        # Embedding
        if embed_tokens:
            name, fp = embed_tokens
            tensor = file_handles[fp].get_tensor(name)
            print(f"embed_tokens: shape={tensor.shape}, dtype={tensor.dtype}")
            # BF16 storage
            manifest["embed_tokens"] = {
                "offset": offset,
                "size": tensor.nbytes,
                "shape": list(tensor.shape)
            }
            tensor_data.append(("embed_tokens", tensor))
            offset += tensor.nbytes
        
        # Layers
        for layer_idx in sorted(layers.keys()):
            layer_tensors = layers[layer_idx]
            layer_data = {}
            for name, fp in layer_tensors:
                tensor = file_handles[fp].get_tensor(name)
                short_name = name.split(".")[-1]  # e.g., "input_layernorm.weight"
                print(f"  Layer {layer_idx}: {short_name} -> shape={tensor.shape}")
                layer_data[short_name] = {
                    "offset": offset,
                    "size": tensor.nbytes,
                    "shape": list(tensor.shape)
                }
                tensor_data.append((f"layer{layer_idx}.{short_name}", tensor))
                offset += tensor.nbytes
            manifest["layers"][layer_idx] = layer_data
        
        # Final norm
        # (will be added based on what's in the model)
        
        # LM head
        if lm_head:
            name, fp = lm_head
            tensor = file_handles[fp].get_tensor(name)
            print(f"lm_head: shape={tensor.shape}, dtype={tensor.dtype}")
            manifest["lm_head"] = {
                "offset": offset,
                "size": tensor.nbytes,
                "shape": list(tensor.shape)
            }
            tensor_data.append(("lm_head", tensor))
            offset += tensor.nbytes
        
        # Write manifest size, manifest, then data
        manifest_json = json.dumps(manifest, indent=2).encode("utf-8")
        out.write(struct.pack("I", len(manifest_json)))
        out.write(manifest_json)
        
        for name, tensor in tensor_data:
            tensor.tofile(out)
    
    print(f"\nWrote {offset} bytes to {output_path}")

if __name__ == "__main__":
    import sys
    main()
```

- [ ] **Step 2: Run test extraction**

Run:
```bash
cd /home/perryshan/wlk/source/flash-moe/cuda_infer
python extract_weights.py /data/ai/hf/hub/models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/33f4e5e615e1f29a7b218906555ea6fe2d09c741 ./test_weights.bin
```
Expected: Should create test_weights.bin with tensor data. Check output for any errors.

- [ ] **Step 3: Inspect output structure**

Run:
```bash
python3 -c "
import struct
with open('./test_weights.bin', 'rb') as f:
    header_size = struct.unpack('I', f.read(4))[0]
    print(f'Header size: {header_size}')
    import json
    header = json.loads(f.read(header_size))
    print(json.dumps(header, indent=2)[:2000])
"
```

- [ ] **Step 4: Commit**

```bash
git add extract_weights.py
git commit -m "feat(cuda_infer): add extract_weights.py for safetensors conversion"
```

---

## Task 3: kernels.cu — CUDA Kernels

**Files:**
- Create: `cuda_infer/kernels.cu`

- [ ] **Step 1: Write CUDA kernels**

Create: `/home/perryshan/wlk/source/flash-moe/cuda_infer/kernels.cu`

```cuda
/*
 * kernels.cu — CUDA kernels for 4-bit quantized MoE inference
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

// ============================================================================
// Helper functions
// ============================================================================

__device__ float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    return __uint_as_float(bits);
}

__device__ uint16_t f32_to_bf16(float f) {
    uint32_t bits = __float_as_uint(f);
    return (uint16_t)(bits >> 16);
}

// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply (NAIVE)
// ============================================================================

__global__ void dequant_matvec_4bit_kernel(
    const uint32_t *W_packed,
    const uint16_t *scales,
    const uint16_t *biases,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;

    float acc = 0.0f;
    const uint32_t *w_row = W_packed + row * packed_cols;
    const uint16_t *s_row = scales + row * num_groups;
    const uint16_t *b_row = biases + row * num_groups;

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);
        int base_packed = g * packed_per_group;
        int base_x = g * group_size;

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            int x_base = base_x + p * 8;

            #pragma unroll
            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                acc += ((float)nibble * scale + bias) * x[x_base + n];
            }
        }
    }
    out[row] = acc;
}

// Launch wrapper
void cuda_dequant_matvec(
    const uint32_t *d_W, const uint16_t *d_scales, const uint16_t *d_biases,
    const float *d_x, float *d_out,
    int out_dim, int in_dim, int group_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;
    dequant_matvec_4bit_kernel<<<blocks, threads, 0, stream>>>(
        d_W, d_scales, d_biases, d_x, d_out, out_dim, in_dim, group_size
    );
}

// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================

__global__ void swiglu_kernel(
    const float *gate,
    const float *up,
    float *out,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    float g = gate[i];
    float silu_g = g / (1.0f + expf(-g));  // sigmoid
    out[i] = silu_g * up[i];
}

void cuda_swiglu(const float *d_gate, const float *d_up, float *d_out, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_out, dim);
}

// ============================================================================
// Kernel 3: RMS Normalization
// ============================================================================

__global__ void rms_norm_kernel(
    const float *x,
    const float *weight,
    float *out,
    int dim,
    float eps
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_sq = 0.0f;
    if (i < dim) {
        sum_sq = x[i] * x[i];
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < dim) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / dim + eps);
    float inv_rms = 1.0f / rms;

    if (i < dim) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void cuda_rms_norm(const float *d_x, const float *d_weight, float *d_out, int dim, float eps, cudaStream_t stream) {
    int threads = 256;
    int blocks = 1;
    rms_norm_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(d_x, d_weight, d_out, dim, eps);
}

// ============================================================================
// Kernel 4: Weighted sum (MoE combine)
// ============================================================================

__global__ void weighted_sum_kernel(
    const float *expert_outputs,  // [num_experts, hidden_dim]
    const float *weights,         // [num_experts]
    float *out,
    int num_experts,
    int hidden_dim,
    int top_k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim) return;

    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int expert_idx = k;  // weights[k] corresponds to expert at indices[k]
        sum += expert_outputs[k * hidden_dim + i] * weights[k];
    }
    out[i] = sum;
}

void cuda_weighted_sum(const float *d_expert_outputs, const float *d_weights, float *d_out,
                       int num_experts, int hidden_dim, int top_k, cudaStream_t stream) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    weighted_sum_kernel<<<blocks, threads, 0, stream>>>(d_expert_outputs, d_weights, d_out, num_experts, hidden_dim, top_k);
}

// ============================================================================
// RoPE: Rotary Position Embedding
// ============================================================================

__global__ void rope_kernel(
    float *q, float *k,
    int seq_len, int head_dim, int num_heads, int num_kv_heads,
    float theta
) {
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int pos = threadIdx.x;

    if (pos >= seq_len || head >= num_heads) return;

    for (int i = 0; i < head_dim / 2; i++) {
        float freq = pos / powf(theta, (2.0f * i) / head_dim);
        float cos_val = cosf(freq);
        float sin_val = sinf(freq);

        int idx = (batch * num_heads + head) * seq_len * head_dim + pos * head_dim;
        int rot_idx = idx + i;

        float q0 = q[rot_idx];
        float q1 = q[rot_idx + head_dim / 2];
        q[rot_idx] = q0 * cos_val - q1 * sin_val;
        q[rot_idx + head_dim / 2] = q0 * sin_val + q1 * cos_val;

        if (head < num_kv_heads) {
            int k_idx = (batch * num_kv_heads + head) * seq_len * head_dim + pos * head_dim;
            int k_rot_idx = k_idx + i;
            float k0 = k[k_rot_idx];
            float k1 = k[k_rot_idx + head_dim / 2];
            k[k_rot_idx] = k0 * cos_val - k1 * sin_val;
            k[k_rot_idx + head_dim / 2] = k0 * sin_val + k1 * cos_val;
        }
    }
}

void cuda_rope(float *d_q, float *d_k, int seq_len, int head_dim, int num_heads,
               int num_kv_heads, float theta, cudaStream_t stream) {
    // Simple implementation for single sequence
    int threads = seq_len;
    dim3 blocks(1, num_heads);
    rope_kernel<<<blocks, threads, 0, stream>>>(d_q, d_k, seq_len, head_dim, num_heads, num_kv_heads, theta);
}

// ============================================================================
// CUDA error check
// ============================================================================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void check_cuda(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(1);
    }
}

#define CHECK_CUDA_ERROR() check_cuda(__FILE__, __LINE__)
```

- [ ] **Step 2: Verify compilation**

Run:
```bash
cd /home/perryshan/wlk/source/flash-moe/cuda_infer
nvcc -O3 -arch=sm_90 -std=c++17 -c kernels.cu -o kernels.o 2>&1
```
Expected: No errors (just warnings if any)

- [ ] **Step 3: Commit**

```bash
git add kernels.cu
git commit -m "feat(cuda_infer): add CUDA kernels for 4-bit dequant, SwiGLU, RMS norm, weighted sum, RoPE"
```

---

## Task 4: infer.cu — Main Inference Program

**Files:**
- Create: `cuda_infer/infer.cu`

- [ ] **Step 1: Write main program structure**

Create: `/home/perryshan/wlk/source/flash-moe/cuda_infer/infer.cu`

```cuda
/*
 * infer.cu — CUDA inference engine for Qwen3.5-35B-A3B-GPTQ-Int4
 *
 * Minimal viable implementation focusing on correctness.
 * Performance optimization deferred to future iteration.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <getopt.h>

#include "tokenizer.h"

// ============================================================================
// Model Constants (from config.json)
// ============================================================================

#define HIDDEN_DIM              2048
#define NUM_LAYERS             40
#define NUM_ATTN_HEADS          16
#define NUM_KV_HEADS            2
#define HEAD_DIM                256
#define VOCAB_SIZE              248320
#define RMS_NORM_EPS            1e-6f
#define NUM_EXPERTS             256
#define NUM_EXPERTS_PER_TOK     8
#define MOE_INTERMEDIATE        512
#define SHARED_INTERMEDIATE     512
#define FULL_ATTN_INTERVAL      4
#define GROUP_SIZE              128
#define BITS                    4

#define EOS_TOKEN_1            248044
#define EOS_TOKEN_2             248046

// ============================================================================
// CUDA Globals
// ============================================================================

static cudaStream_t g_stream;
static cublasHandle_t g_cublas;

// ============================================================================
// Helper Functions
// ============================================================================

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    return __uint_as_float(bits);
}

// ============================================================================
// Weights Binary Layout
// ============================================================================

typedef struct {
    void *data;
    size_t size;
    // Manifest offsets will be loaded from header
} WeightFile;

static WeightFile *open_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        return NULL;
    }

    uint32_t header_size;
    fread(&header_size, 4, 1, f);

    char *header_json = malloc(header_size + 1);
    fread(header_json, 1, header_size, f);
    header_json[header_size] = '\0';

    // Get file size
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // mmap the data portion
    fclose(f);

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (data == MAP_FAILED) {
        free(header_json);
        return NULL;
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = file_size;

    printf("[weights] mmap'd %.2f GB from %s\n", file_size / 1e9, path);
    return wf;
}

// ============================================================================
// 4-bit Dequant MatVec (CPU fallback for complex layout)
// ============================================================================

static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;

    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int x_base = base_x + p * 8;

                for (int n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    acc += ((float)nibble * scale + bias) * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

// ============================================================================
// RMS Norm
// ============================================================================

static void cpu_rms_norm(const float *x, const float *w, float *out, int dim) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / dim + RMS_NORM_EPS);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

// ============================================================================
// SwiGLU
// ============================================================================

static void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// ============================================================================
// Softmax + Top-K
// ============================================================================

static void cpu_topk(const float *scores, int dim, int K, int *indices, float *weights) {
    // Initialize with -inf
    for (int k = 0; k < K; k++) {
        weights[k] = -1e30f;
        indices[k] = 0;
    }

    for (int i = 0; i < dim; i++) {
        int min_k = 0;
        for (int k = 1; k < K; k++) {
            if (weights[k] < weights[min_k]) min_k = k;
        }
        if (scores[i] > weights[min_k]) {
            weights[min_k] = scores[i];
            indices[min_k] = i;
        }
    }

    // Softmax on selected weights
    float max_val = weights[0];
    for (int k = 1; k < K; k++) {
        if (weights[k] > max_val) max_val = weights[k];
    }
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        weights[k] = expf(weights[k] - max_val);
        sum += weights[k];
    }
    float inv_sum = 1.0f / sum;
    for (int k = 0; k < K; k++) {
        weights[k] *= inv_sum;
    }
}

// ============================================================================
// Forward Pass Structures
// ============================================================================

typedef struct {
    float *hidden;      // [hidden_dim]
    float *gate_output; // [moe_intermediate]
    float *up_output;   // [moe_intermediate]
    float *expert_out;  // [num_experts, hidden_dim] temp
    float *routing_scores; // [num_experts]
    int topk_indices[NUM_EXPERTS_PER_TOK];
    float topk_weights[NUM_EXPERTS_PER_TOK];
} LayerBuffers;

static LayerBuffers *create_layer_buffers(void) {
    LayerBuffers *buf = calloc(1, sizeof(LayerBuffers));
    cudaMalloc(&buf->hidden, HIDDEN_DIM * sizeof(float));
    cudaMalloc(&buf->gate_output, MOE_INTERMEDIATE * sizeof(float));
    cudaMalloc(&buf->up_output, MOE_INTERMEDIATE * sizeof(float));
    cudaMalloc(&buf->expert_out, NUM_EXPERTS_PER_TOK * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&buf->routing_scores, NUM_EXPERTS * sizeof(float));
    return buf;
}

// ============================================================================
// Main Forward Pass
// ============================================================================

static int forward_layer(
    const float *input,           // [hidden_dim]
    const void *layer_weights,    // Layer weight pointer
    LayerBuffers *buf,
    float *output,                // [hidden_dim]
    int layer_idx,
    int is_full_attention
) {
    (void)layer_weights;
    (void)buf;
    (void)input;
    (void)output;
    (void)layer_idx;
    (void)is_full_attention;

    // TODO: Implement layer forward
    // For now, just copy input to output
    memcpy(output, input, HIDDEN_DIM * sizeof(float));

    return 0;
}

// ============================================================================
// Main Program
// ============================================================================

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --prompt TEXT    Input prompt\n");
    fprintf(stderr, "  --tokens N       Max tokens to generate (default: 100)\n");
    fprintf(stderr, "  --weights PATH   Path to model_weights.bin\n");
    fprintf(stderr, "  --help           Show this help\n");
}

int main(int argc, char **argv) {
    const char *prompt = "Hello";
    int max_tokens = 100;
    const char *weights_path = "model_weights.bin";

    // Parse arguments
    static struct option long_options[] = {
        {"prompt", required_argument, 0, 'p'},
        {"tokens", required_argument, 0, 't'},
        {"weights", required_argument, 0, 'w'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while (1) {
        int c = getopt_long(argc, argv, "p:t:w:h", long_options, NULL);
        if (c == -1) break;
        switch (c) {
            case 'p': prompt = optarg; break;
            case 't': max_tokens = atoi(optarg); break;
            case 'w': weights_path = optarg; break;
            case 'h':
                usage(argv[0]);
                return 0;
        }
    }

    printf("=== CUDA MoE Inference ===\n");
    printf("Prompt: %s\n", prompt);
    printf("Max tokens: %d\n", max_tokens);
    printf("Weights: %s\n", weights_path);

    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamCreate(&g_stream));
    CHECK_CUDA(cublasCreate(&g_cublas));

    // Load tokenizer
    bpe_tokenizer tokenizer;
    if (bpe_load(&tokenizer, "tokenizer.bin") != 0) {
        fprintf(stderr, "ERROR: Failed to load tokenizer\n");
        return 1;
    }
    printf("[tokenizer] Loaded successfully\n");

    // Encode prompt
    uint32_t prompt_ids[4096];
    int num_prompt_tokens = bpe_encode(&tokenizer, prompt, prompt_ids, 4096);
    if (num_prompt_tokens < 0) {
        fprintf(stderr, "ERROR: Failed to encode prompt\n");
        return 1;
    }
    printf("[tokens] Encoded %d tokens\n", num_prompt_tokens);

    // Load weights
    WeightFile *wf = open_weights(weights_path);
    if (!wf) {
        fprintf(stderr, "ERROR: Failed to load weights from %s\n", weights_path);
        return 1;
    }

    // Create layer buffers
    LayerBuffers *layer_buf = create_layer_buffers();

    // TODO: Build actual forward pass
    // For now, just print tokenization worked

    printf("[inference] Forward pass not yet implemented\n");

    // Cleanup
    cudaFree(layer_buf->hidden);
    cudaFree(layer_buf->gate_output);
    cudaFree(layer_buf->up_output);
    cudaFree(layer_buf->expert_out);
    cudaFree(layer_buf->routing_scores);
    free(layer_buf);

    bpe_free(&tokenizer);
    munmap(wf->data, wf->size);
    free(wf);

    CHECK_CUDA(cublasDestroy(g_cublas));
    CHECK_CUDA(cudaStreamDestroy(g_stream));

    printf("[done] Inference complete\n");
    return 0;
}
```

- [ ] **Step 2: Try compilation**

Run:
```bash
cd /home/perryshan/wlk/source/flash-moe/cuda_infer
nvcc -O3 -arch=sm_90 -std=c++17 -o infer infer.cu kernels.cu -lcublas -lcudart -Xcompiler -Wall 2>&1
```
Expected: Compilation warnings/errors. Fix as needed.

- [ ] **Step 3: Fix compilation errors**

If there are errors, fix them inline. Common issues:
- Missing include for getopt
- CUDA kernel launch configuration
- cublas header

- [ ] **Step 4: Run basic test**

Run:
```bash
cd /home/perryshan/wlk/source/flash-moe/cuda_infer
./infer --help
```
Expected: Usage message

- [ ] **Step 5: Commit**

```bash
git add infer.cu
git commit -m "feat(cuda_infer): add main inference program stub with CUDA initialization"
```

---

## Task 5: Implement Full Forward Pass

**Files:**
- Modify: `cuda_infer/infer.cu`

- [ ] **Step 1: Understand safetensors weight layout**

The GPTQ-Int4 format stores quantized weights per expert. Each expert has:
- qweight: uint32 array (4-bit values packed)
- scales: bfloat16 array
- qzeros: uint32 array (zero points)
- g_idx: uint32 array (group indices)

From the weight_map in model.safetensors.index.json, we can see tensors like:
- `model.language_model.layers.N.mlp.experts.M.gate_proj.qweight`
- `model.language_model.layers.N.mlp.experts.M.gate_proj.scales`
- etc.

- [ ] **Step 2: Update extract_weights.py to handle GPTQ format**

The GPTQ format is different from the simple packed binary. We need to handle:
1. Per-expert quantization (256 experts per layer)
2. gate/up/down projections per expert
3. g_idx and qzeros

Update extract_weights.py to:
```python
# For each expert, extract:
# - gate_proj: [512, 2048] qweight + scales + qzeros + g_idx
# - up_proj: [512, 2048] qweight + scales + qzeros + g_idx
# - down_proj: [2048, 512] qweight + scales + qzeros + g_idx
```

- [ ] **Step 3: Write complete forward pass**

Update infer.cu with full implementation:

```cuda
// Forward layer implementation outline:

int forward_layer(float *hidden, int layer_idx, int is_full_attention) {
    // 1. Input RMS norm
    cpu_rms_norm(hidden, input_layernorm_weight, buf->hidden, HIDDEN_DIM);

    // 2. Attention (simplified for now)
    // - Compute QKV projections (CPU matvec or cuBLAS)
    // - Apply RoPE
    // - Compute attention scores
    // - Weighted sum with V

    // 3. MoE routing
    // - Compute routing scores (gate_proj matvec)
    // - Top-K selection
    // - Load top-K expert weights

    // 4. Expert forward pass
    // For each top-K expert:
    //   - cpu_dequant_matvec for gate+up (SwiGLU)
    //   - cpu_dequant_matvec for down
    //   - Combine with routing weights

    // 5. Residual + output norm
    // Hidden = hidden + moe_output
    // cpu_rms_norm(hidden, final_layernorm_weight, hidden, HIDDEN_DIM)

    return 0;
}
```

- [ ] **Step 4: Add proper weight loading**

Need to parse the safetensors and create proper offsets for each layer's weights.

- [ ] **Step 5: Test end-to-end**

Run:
```bash
cd /home/perryshan/wlk/source/flash-moe/cuda_infer
./infer --prompt "Hello world" --tokens 20 --weights ./model_weights.bin 2>&1
```

Expected: Text output that looks like valid tokens (may not be coherent yet).

- [ ] **Step 6: Debug and fix**

Address any segmentation faults, cuda errors, or incorrect outputs.

- [ ] **Step 7: Commit**

```bash
git add infer.cu extract_weights.py
git commit -m "feat(cuda_infer): implement full forward pass"
```

---

## Task 6: Integration Testing

**Files:**
- Create: `cuda_infer/test_inference.py` (optional)

- [ ] **Step 1: Verify tokenization matches reference**

Compare tokenization output with metal_infer on same input.

- [ ] **Step 2: Verify forward pass runs to completion**

Check that all 40 layers execute without error.

- [ ] **Step 3: Verify output is valid UTF-8 text**

Print decoded tokens and verify they're valid.

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat(cuda_infer): complete working inference for Qwen3.5-35B"
```

---

## Success Criteria

1. [ ] `make` compiles without errors
2. [ ] `./infer --prompt "Hello" --tokens 10` produces text output
3. [ ] No CUDA errors during execution
4. [ ] Output is valid UTF-8 tokens (not garbage)
5. [ ] 40 layers all execute

---

## Execution Options

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?