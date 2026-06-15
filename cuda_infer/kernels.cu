#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "kernels.h"

// Model constants
#define HIDDEN_DIM 2048
#define MOE_INTERMEDIATE 512
#define NUM_EXPERTS_PER_TOK 8
#define GROUP_SIZE 128
#define HEAD_DIM 256
#define NUM_ATTN_HEADS 16
#define NUM_KV_HEADS 2

// bf16 conversion
__device__ float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    return __uint_as_float(bits);
}

__device__ uint16_t f32_to_bf16(float f) {
    uint32_t bits = __float_as_uint(f);
    bits += 0x7fff; // rounding
    return (uint16_t)(bits >> 16);
}

// Error check
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Kernel 1: GPTQ 4-bit dequantized matrix-vector multiply
// ============================================================================
// GPTQ-Int4 format (auto-gptq / HuggingFace layout):
//   qweight: uint32 [in_dim//8, out_dim] — 8 nibbles per uint32 along in_dim
//   scales:  bfloat16 [in_dim//group_size, out_dim] — per-group scale per row
//   qzeros:  uint32 [in_dim//group_size, out_dim//8] — packed 4-bit zeros per row
//
// Dequant formula: weight = (nibble - zero) * scale
// where zero is unpacked from qzeros nibble at (g, row)
//
// For gate_proj/up_proj: out_dim=512, in_dim=2048, group_size=128
//   qweight [256, 512], scales [16, 512], qzeros [16, 64]
// For down_proj: out_dim=2048, in_dim=512
//   qweight [64, 2048], scales [4, 2048], qzeros [4, 256]

__global__ void dequant_matvec_gptq_kernel(
    const uint32_t *qweight,
    const float *scales,
    const uint32_t *qzeros,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    int packed_cols = in_dim / 8;           // chunks along input dim
    int qz_packed_cols = out_dim / 8;       // chunks along output dim (for qzeros)
    int group_len = group_size / 8;         // packed cols per group

    float result = 0.0f;

    for (int col = 0; col < packed_cols; col++) {
        // qweight stored as [in_dim/8, out_dim] → index at [col][row]
        int weight_idx = col * out_dim + row;
        uint32_t packed = qweight[weight_idx];

        int g = col / group_len;
        // scales stored as [num_groups, out_dim] float32 → index at [g][row]
        float scale = scales[g * out_dim + row];

        // qzeros stored as [num_groups, out_dim/8] int32, zero at nibble (row%8)
        int qz_nibble = row % 8;
        uint32_t qz_packed = qzeros[g * qz_packed_cols + (row / 8)];
        float zero = (float)((qz_packed >> (qz_nibble * 4)) & 0xF);

        for (int n = 0; n < 8; n++) {
            float w = ((float)((packed >> (n * 4)) & 0xF) - zero) * scale;
            result += w * x[col * 8 + n];
        }
    }
    out[row] = result;
}

extern "C" {
void cuda_dequant_matvec_gptq(
    const uint32_t *d_qweight, const float *d_scales,
    const uint32_t *d_qzeros, const float *d_x,
    float *d_out, int out_dim, int in_dim, int group_size,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((out_dim + blockDim.x - 1) / blockDim.x);
    dequant_matvec_gptq_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_qweight, d_scales, d_qzeros, d_x, d_out, out_dim, in_dim, group_size);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================
// SwiGLU: out = silu(gate) * up
// silu(x) = x / (1 + exp(-x))
// Gate and up are separate input vectors of size [intermediate]

__global__ void swiglu_kernel(
    const float *gate,  // [intermediate]
    const float *up,    // [intermediate]
    float *out,
    int intermediate  // MOE_INTERMEDIATE
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= intermediate) return;

    float g = gate[idx];
    float u = up[idx];

    // silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
    float sigmoid_gate = 1.0f / (1.0f + expf(-g));

    out[idx] = g * sigmoid_gate * u;
}

extern "C" {
void cuda_swiglu(
    const float *gate,
    const float *up,
    float *out,
    int intermediate,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((intermediate + blockDim.x - 1) / blockDim.x);

    swiglu_kernel<<<gridDim, blockDim, 0, stream>>>(
        gate, up, out, intermediate);

    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 3: RMS Normalization (two-pass GPU-only)
// ============================================================================

static float *d_norm_sum_sq = NULL;

__global__ void rms_reduce_kernel(const float *x, float *sum_sq, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    float local_sum = 0.0f;

    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        local_sum += x[i] * x[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_sq, sdata[0]);
    }
}

__global__ void rms_apply_kernel(
    const float *x, const float *weight,
    float *out, const float *sum_sq_ptr, int n, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float inv_rms = rsqrtf(sum_sq_ptr[0] / (float)n + eps);
    out[idx] = x[idx] * inv_rms * (1.0f + weight[idx]);
}

extern "C" {
void cuda_rms_norm(
    const float *d_x, const float *d_weight,
    float *d_out, int dim, float eps, cudaStream_t stream
) {
    if (!d_norm_sum_sq) {
        CHECK_CUDA(cudaMalloc(&d_norm_sum_sq, sizeof(float)));
    }
    CHECK_CUDA(cudaMemsetAsync(d_norm_sum_sq, 0, sizeof(float), stream));

    int blockSize = 256;
    int gridSize = min((dim + blockSize - 1) / blockSize, 256);

    rms_reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(float), stream>>>(
        d_x, d_norm_sum_sq, dim);

    int applyBlocks = (dim + 255) / 256;
    rms_apply_kernel<<<applyBlocks, 256, 0, stream>>>(
        d_x, d_weight, d_out, d_norm_sum_sq, dim, eps);

    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 4: Weighted sum of expert outputs
// ============================================================================
// Each token selects NUM_EXPERTS_PER_TOK experts
// out = sum_i(weight_i * expert_i_output)

__global__ void weighted_sum_kernel(
    const float *expert_outputs,  // [num_experts, num_selected, hidden]
    const float *weights,        // [num_selected] routing weights
    float *out,
    int num_selected,  // NUM_EXPERTS_PER_TOK
    int hidden         // HIDDEN_DIM
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hidden) return;

    float sum = 0.0f;
    for (int i = 0; i < num_selected; i++) {
        sum += weights[i] * expert_outputs[i * hidden + idx];
    }
    out[idx] = sum;
}

extern "C" {
void cuda_weighted_sum(
    const float *expert_outputs,
    const float *weights,
    float *out,
    int num_selected,
    int hidden,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((hidden + blockDim.x - 1) / blockDim.x);

    weighted_sum_kernel<<<gridDim, blockDim, 0, stream>>>(
        expert_outputs, weights, out, num_selected, hidden);

    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 5: Rotary Position Embedding (RoPE)
// ============================================================================
// Qwen3 RoPE: rotates only the first rotary_dim=64 dimensions.
// For each head h and each of the first (rotary_dim/2)=32 dims d:
//   x[base + d], x[base + d + half_head] are rotated by angle(d, position)

__global__ void rope_kernel(
    const float *x, float *out,
    int num_heads, int head_dim, int position, float base
) {
    int rotary_dim = 64;
    int half_head = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_heads * (rotary_dim / 2);
    if (idx >= total_pairs) return;

    int h = idx / (rotary_dim / 2);
    int d = idx % (rotary_dim / 2);

    float angle = position / powf(base, 2.0f * d / (float)rotary_dim);
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int base_idx = h * head_dim;
    float x0 = x[base_idx + d];
    float x1 = x[base_idx + d + half_head];

    out[base_idx + d] = x0 * cos_val - x1 * sin_val;
    out[base_idx + d + half_head] = x0 * sin_val + x1 * cos_val;
}

extern "C" {
void cuda_rope(
    const float *d_x, float *d_out,
    int num_heads, int head_dim, int position, float base,
    cudaStream_t stream
) {
    int total = num_heads * head_dim;
    int rotary_dim = 64;
    int total_pairs = num_heads * (rotary_dim / 2);

    // First copy x to out
    CHECK_CUDA(cudaMemcpyAsync(d_out, d_x, total * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));

    // Then apply rotation only to rotary_dim/2 pairs per head
    if (total_pairs > 0) {
        dim3 blockDim(256);
        dim3 gridDim((total_pairs + 255) / 256);
        rope_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_x, d_out, num_heads, head_dim, position, base);
    }

    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 6: Residual add
// ============================================================================

__global__ void residual_add_kernel(
    const float *a, const float *b,
    float *out, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    out[idx] = a[idx] + b[idx];
}

extern "C" {
void cuda_residual_add(
    const float *d_a, const float *d_b,
    float *d_out, int dim, cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((dim + 255) / 256);
    residual_add_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_a, d_b, d_out, dim);
    CHECK_CUDA(cudaGetLastError());
}
}
// ============================================================================
// Kernel 7: BF16 matrix-vector multiply
// ============================================================================
// out[row] = sum_{col} weight[row][col] * x[col]
// where weight is BF16, x is float32

__global__ void bf16_matvec_kernel(
    const uint16_t *weight,  // [out_dim, in_dim] BF16
    const float *x,          // [in_dim] float32
    float *out,              // [out_dim] float32
    int out_dim,
    int in_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    float sum = 0.0f;
    const uint16_t *w_row = weight + (size_t)row * in_dim;
    for (int col = 0; col < in_dim; col++) {
        sum += bf16_to_f32(w_row[col]) * x[col];
    }
    out[row] = sum;
}

extern "C" {
void cuda_bf16_matvec(
    const uint16_t *d_weight,
    const float *d_x, float *d_out,
    int out_dim, int in_dim,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((out_dim + blockDim.x - 1) / blockDim.x);
    bf16_matvec_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_weight, d_x, d_out, out_dim, in_dim);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 8: Attention scores (Q @ K^T / sqrt(d)) -- batched over (pos, head)
// ============================================================================

__global__ void attn_scores_kernel(
    const float *q,
    const float *k_cache,
    float *scores,
    int head_dim,
    int kv_dim,
    int seq_len,
    int seq_stride,
    float scale,
    int heads_per_kv
) {
    int pos = blockIdx.x;
    int h = blockIdx.y;
    if (pos >= seq_len) return;

    int kv_h = h / heads_per_kv;
    const float *qh = q + h * head_dim;
    const float *kp = k_cache + pos * kv_dim + kv_h * head_dim;

    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        acc += qh[d] * kp[d];
    }

    __shared__ float sdata[256];
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        scores[h * seq_stride + pos] = sdata[0] * scale;
    }
}

extern "C" {
void cuda_attn_scores(
    const float *d_q, const float *d_k_cache,
    float *d_scores, int head_dim, int kv_dim,
    int seq_len, int seq_stride, float scale,
    int heads_per_kv, int num_seq_tgs,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim(num_seq_tgs, 16);
    attn_scores_kernel<<<gridDim, blockDim, blockDim.x * sizeof(float), stream>>>(
        d_q, d_k_cache, d_scores, head_dim, kv_dim,
        seq_len, seq_stride, scale, heads_per_kv);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 9: Attention softmax (in-place, one threadgroup per head)
// ============================================================================
__global__ void attn_softmax_kernel(
    float *scores, int seq_len, int seq_stride
) {
    int h = blockIdx.x;
    float *s = scores + h * seq_stride;

    __shared__ float shared[256];
    int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float v = s[i];
        if (v > local_max) local_max = v;
    }

    shared[tid] = local_max;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2 && shared[tid + s2] > shared[tid])
            shared[tid] = shared[tid + s2];
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = expf(s[i] - max_val);
        s[i] = val;
        local_sum += val;
    }

    shared[tid] = local_sum;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) shared[tid] += shared[tid + s2];
        __syncthreads();
    }
    float inv_sum = 1.0f / shared[0];

    for (int i = tid; i < seq_len; i += blockDim.x) {
        s[i] *= inv_sum;
    }
}

extern "C" {
void cuda_attn_softmax(
    float *d_scores, int seq_len, int seq_stride,
    cudaStream_t stream
) {
    attn_softmax_kernel<<<16, 256, 0, stream>>>(
        d_scores, seq_len, seq_stride);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 10: Attention value aggregation (softmax @ V)
// ============================================================================
__global__ void attn_values_kernel(
    const float *scores,
    const float *v_cache,
    float *out,
    int head_dim,
    int kv_dim,
    int seq_len,
    int seq_stride,
    int heads_per_kv
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 16 * head_dim;
    if (d >= total) return;

    int h = d / head_dim;
    int dim = d % head_dim;
    int kv_h = h / heads_per_kv;

    const float *s = scores + h * seq_stride;
    float acc = 0.0f;
    for (int p = 0; p < seq_len; p++) {
        acc += s[p] * v_cache[p * kv_dim + kv_h * head_dim + dim];
    }
    out[d] = acc;
}

extern "C" {
void cuda_attn_values(
    const float *d_scores, const float *d_v_cache,
    float *d_out, int head_dim, int kv_dim,
    int seq_len, int seq_stride, int heads_per_kv,
    cudaStream_t stream
) {
    int total = 16 * head_dim;
    dim3 blockDim(256);
    dim3 gridDim((total + 255) / 256);
    attn_values_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_scores, d_v_cache, d_out, head_dim, kv_dim,
        seq_len, seq_stride, heads_per_kv);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 11: Sigmoid gate (in-place): out[i] *= sigmoid(gate[i])
// ============================================================================
__global__ void sigmoid_gate_kernel(
    float *x_out, const float *gate, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    float g = 1.0f / (1.0f + expf(-gate[idx]));
    x_out[idx] *= g;
}

extern "C" {
void cuda_sigmoid_gate(
    float *d_x_out, const float *d_gate, int dim,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((dim + 255) / 256);
    sigmoid_gate_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_x_out, d_gate, dim);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 12: Conv1d depthwise step with SiLU activation
// ============================================================================
__global__ void conv1d_step_kernel(
    float *conv_state, const float *input,
    const uint16_t *weight, float *output, int conv_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_dim) return;

    int w_base = idx * 4;
    float acc = conv_state[0 * conv_dim + idx] * bf16_to_f32(weight[w_base + 0]) +
                conv_state[1 * conv_dim + idx] * bf16_to_f32(weight[w_base + 1]) +
                conv_state[2 * conv_dim + idx] * bf16_to_f32(weight[w_base + 2]) +
                input[idx] * bf16_to_f32(weight[w_base + 3]);
    float silu_out = acc / (1.0f + expf(-acc));
    output[idx] = silu_out;

    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = input[idx];
}

extern "C" {
void cuda_conv1d_step(
    float *d_conv_state, const float *d_input,
    const uint16_t *d_weight, float *d_output,
    int conv_dim, cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((conv_dim + 255) / 256);
    conv1d_step_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_conv_state, d_input, d_weight, d_output, conv_dim);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
__global__ void compute_decay_beta_kernel(
    const float *alpha, const float *beta,
    const float *A_log, const uint16_t *dt_bias,
    float *g_decay, float *beta_gate, int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads) return;

    float a_val = alpha[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = expf(A_log[idx]);
    float softplus_val = logf(1.0f + expf(a_val + dt_b));
    g_decay[idx] = expf(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + expf(-beta[idx]));
}

extern "C" {
void cuda_compute_decay_beta(
    const float *d_alpha, const float *d_beta,
    const float *d_A_log, const uint16_t *d_dt_bias,
    float *d_g_decay, float *d_beta_gate, int num_heads,
    cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((num_heads + 255) / 256);
    compute_decay_beta_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_alpha, d_beta, d_A_log, d_dt_bias,
        d_g_decay, d_beta_gate, num_heads);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel: Per-head RMS norm for Q and K (bare norm, no weights)
// ============================================================================
__global__ void rms_norm_qk_kernel(
    float *q, float *k, int num_k_heads,
    int key_dim, float inv_scale
) {
    int h = blockIdx.x;
    if (h >= num_k_heads) return;

    int base = h * key_dim;
    int tid = threadIdx.x;

    __shared__ float q_partial[256];
    __shared__ float k_partial[256];
    __shared__ float q_sum_sq, k_sum_sq;

    float q_val = (tid < key_dim) ? q[base + tid] : 0.0f;
    float k_val = (tid < key_dim) ? k[base + tid] : 0.0f;

    q_partial[tid] = q_val * q_val;
    k_partial[tid] = k_val * k_val;
    __syncthreads();

    if (tid == 0) {
        float qs = 0.0f, ks = 0.0f;
        for (int i = 0; i < key_dim; i++) {
            qs += q_partial[i];
            ks += k_partial[i];
        }
        q_sum_sq = qs;
        k_sum_sq = ks;
    }
    __syncthreads();

    float q_l2norm = rsqrtf(q_sum_sq + 1e-6f);
    float k_l2norm = rsqrtf(k_sum_sq + 1e-6f);

    if (tid < key_dim) {
        q[base + tid] = q_val * q_l2norm * inv_scale;
        k[base + tid] = k_val * k_l2norm;
    }
}

extern "C" {
void cuda_rms_norm_qk(
    float *d_q, float *d_k, int num_k_heads,
    int key_dim, float inv_scale, cudaStream_t stream
) {
    rms_norm_qk_kernel<<<num_k_heads, 256, 0, stream>>>(
        d_q, d_k, num_k_heads, key_dim, inv_scale);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel: GatedDeltaNet recurrence step
// ============================================================================
// Each v-head is a threadgroup, each thread handles one vi (value index)
// State: [num_v_heads * value_dim * key_dim] = [64 * 128 * 128]
// Steps: decay -> kv_mem=dot(S,k) -> delta=(v-kv_mem)*beta -> S+=k*delta -> out=dot(S,q)
__global__ void gated_delta_net_step_kernel(
    float *state, const float *q, const float *k,
    const float *v, const float *g_decay,
    const float *beta_gate, float *output,
    int num_v_heads, int value_dim, int key_dim,
    int k_heads_per_v
) {
    int vh = blockIdx.x;
    int vi = threadIdx.x;
    if (vh >= num_v_heads || vi >= value_dim) return;

    int kh = vh / k_heads_per_v;
    float g = g_decay[vh];
    float beta = beta_gate[vh];

    int state_base = vh * value_dim * key_dim + vi * key_dim;
    int k_base = kh * key_dim;
    int v_base = vh * value_dim;
    int q_base = kh * key_dim;

    // Step 1: Decay state row S[vi][:] *= g
    for (int ki = 0; ki < key_dim; ki++) {
        state[state_base + ki] *= g;
    }

    // Step 2: kv_mem = sum(S[vi][ki] * k[ki])
    float kv_mem = 0.0f;
    for (int ki = 0; ki < key_dim; ki++) {
        kv_mem += state[state_base + ki] * k[k_base + ki];
    }

    // Step 3-4: delta = (v[vi] - kv_mem) * beta; S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (int ki = 0; ki < key_dim; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output = sum(S[vi][ki] * q[ki])
    float out_val = 0.0f;
    for (int ki = 0; ki < key_dim; ki++) {
        out_val += state[state_base + ki] * q[q_base + ki];
    }
    output[v_base + vi] = out_val;
}

extern "C" {
void cuda_gated_delta_net_step(
    float *d_state, const float *d_q, const float *d_k,
    const float *d_v, const float *d_g_decay,
    const float *d_beta_gate, float *d_output,
    int num_v_heads, int value_dim, int key_dim,
    int k_heads_per_v, cudaStream_t stream
) {
    gated_delta_net_step_kernel<<<num_v_heads, value_dim, 0, stream>>>(
        d_state, d_q, d_k, d_v, d_g_decay, d_beta_gate, d_output,
        num_v_heads, value_dim, key_dim, k_heads_per_v);
    CHECK_CUDA(cudaGetLastError());
}
}

// ============================================================================
// Kernel: Gated RMS norm (RMS Norm + SiLU gate + BF16 weight)
// ============================================================================
// output[head][dim] = rms_norm(values[head])[dim] * silu(z[head][dim]) * weight[dim]
// Weight is BF16 and shared across all heads
__global__ void gated_rms_norm_kernel(
    const float *values, const float *z,
    const uint16_t *weight, float *output,
    int num_heads, int value_dim, float eps
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= num_heads || tid >= value_dim) return;

    int base = h * value_dim;

    __shared__ float partial[256];
    float val = values[base + tid];
    partial[tid] = val * val;
    __syncthreads();

    if (tid == 0) {
        float ss = 0.0f;
        for (int i = 0; i < value_dim; i++) ss += partial[i];
        partial[0] = ss;
    }
    __syncthreads();

    float inv_rms = rsqrtf(partial[0] / (float)value_dim + eps);
    float normed = val * inv_rms;
    float zv = z[base + tid];
    float gate = zv / (1.0f + expf(-zv));
    float w = bf16_to_f32(weight[tid]);
    output[base + tid] = normed * gate * w;
}

extern "C" {
void cuda_gated_rms_norm(
    const float *d_values, const float *d_z,
    const uint16_t *d_weight, float *d_output,
    int num_heads, int value_dim, float eps,
    cudaStream_t stream
) {
    gated_rms_norm_kernel<<<num_heads, value_dim, 0, stream>>>(
        d_values, d_z, d_weight, d_output, num_heads, value_dim, eps);
    CHECK_CUDA(cudaGetLastError());
}
}
