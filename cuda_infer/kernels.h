#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_dequant_matvec_gptq(
    const uint32_t *d_qweight, const float *d_scales,
    const uint32_t *d_qzeros, const float *d_x,
    float *d_out, int out_dim, int in_dim, int group_size,
    cudaStream_t stream);

void cuda_swiglu(
    const float *d_gate, const float *d_up,
    float *d_out, int dim, cudaStream_t stream);

void cuda_rms_norm(
    const float *d_x, const float *d_weight,
    float *d_out, int dim, float eps, cudaStream_t stream);

void cuda_weighted_sum(
    const float *d_expert_outputs, const float *d_weights,
    float *d_out, int num_experts, int hidden, cudaStream_t stream);

void cuda_rope(
    const float *d_x, float *d_out,
    int num_heads, int head_dim, int position,
    float base, cudaStream_t stream);

void cuda_residual_add(
    const float *d_a, const float *d_b,
    float *d_out, int dim, cudaStream_t stream);

void cuda_bf16_matvec(
    const uint16_t *d_weight,
    const float *d_x, float *d_out,
    int out_dim, int in_dim,
    cudaStream_t stream);

void cuda_attn_scores(
    const float *d_q, const float *d_k_cache,
    float *d_scores, int head_dim, int kv_dim,
    int seq_len, int seq_stride, float scale,
    int heads_per_kv, int num_seq_tgs,
    cudaStream_t stream);

void cuda_attn_softmax(
    float *d_scores, int seq_len, int seq_stride,
    cudaStream_t stream);

void cuda_attn_values(
    const float *d_scores, const float *d_v_cache,
    float *d_out, int head_dim, int kv_dim,
    int seq_len, int seq_stride, int heads_per_kv,
    cudaStream_t stream);

void cuda_sigmoid_gate(
    float *d_x_out, const float *d_gate, int dim,
    cudaStream_t stream);

void cuda_conv1d_step(
    float *d_conv_state, const float *d_input,
    const uint16_t *d_weight, float *d_output,
    int conv_dim, cudaStream_t stream);

void cuda_compute_decay_beta(
    const float *d_alpha, const float *d_beta,
    const float *d_A_log, const uint16_t *d_dt_bias,
    float *d_g_decay, float *d_beta_gate, int num_heads,
    cudaStream_t stream);

void cuda_rms_norm_qk(
    float *d_q, float *d_k, int num_k_heads,
    int key_dim, float inv_scale, cudaStream_t stream);

void cuda_gated_delta_net_step(
    float *d_state, const float *d_q, const float *d_k,
    const float *d_v, const float *d_g_decay,
    const float *d_beta_gate, float *d_output,
    int num_v_heads, int value_dim, int key_dim,
    int k_heads_per_v, cudaStream_t stream);

void cuda_gated_rms_norm(
    const float *d_values, const float *d_z,
    const uint16_t *d_weight, float *d_output,
    int num_heads, int value_dim, float eps,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
