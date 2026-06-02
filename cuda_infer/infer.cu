/*
 * infer.cu — CUDA inference engine for Qwen3-MoE
 *
 * Main program: loads weights, tokenizes input, runs forward pass
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <getopt.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "kernels.h"

// Tokenizer declarations (compiled separately as C++)
extern "C" {
typedef struct {
    char *vocab;
    uint32_t vocab_size;
    char *merges;
    uint32_t num_merges;
    char *added;
    uint32_t num_added;
    uint32_t ht_mask, *ht_ids;
    char **ht_keys;
    uint16_t *ht_klens;
    uint32_t mt_mask, *mt_prio;
    char **mt_keys;
    uint16_t *mt_klens;
    uint32_t byte_char[256];
    uint8_t char_byte[512];
} bpe_tokenizer;

extern int bpe_load(bpe_tokenizer *tok, const char *path);
extern int bpe_encode(const bpe_tokenizer *tok, const char *text, uint32_t *out_ids, int max_ids);
extern void bpe_free(bpe_tokenizer *tok);
extern int bpe_decode_token(const bpe_tokenizer *tok, uint32_t token_id, char *buf, int buf_size);
}

// Model constants
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
#define FULL_ATTN_INTERVAL      4
#define GROUP_SIZE              128
#define EOS_TOKEN_1             151643
#define EOS_TOKEN_2             151645
#define LM_CHUNK                8192

// Linear attention (GatedDeltaNet) constants
#define LINEAR_NUM_V_HEADS      32
#define LINEAR_NUM_K_HEADS      16
#define LINEAR_KEY_DIM          128
#define LINEAR_VALUE_DIM        128
#define LINEAR_TOTAL_KEY        (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)   // 2048
#define LINEAR_TOTAL_VALUE      (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) // 8192
#define LINEAR_CONV_DIM         (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) // 12288
#define CONV_KERNEL_SIZE        4

// ============================================================================
// Error checking
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

#define CHECK_NULL(ptr) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "Out of memory at %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Bfloat16 conversion (host-only)
// ============================================================================

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

__attribute__((unused))
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFFU; // rounding
    return (uint16_t)(bits >> 16);
}

// ============================================================================
// CPU fallback kernels
// ============================================================================

__attribute__((unused))
static void cpu_rms_norm(
    const float *x,
    const float *weight,
    float *out,
    int n
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + RMS_NORM_EPS);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * weight[i] * inv_rms;
    }
}

__attribute__((unused))
static void cpu_swiglu(
    const float *gate,
    const float *up,
    float *out,
    int n
) {
    for (int i = 0; i < n; i++) {
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate[i]));
        out[i] = sigmoid_gate * up[i];
    }
}

__attribute__((unused))
static void cpu_silu(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

__attribute__((unused))
static void cpu_topk(
    const float *scores,
    int *indices,
    float *weights,
    int n,
    int k
) {
    // Simple selection: find top-k by value
    for (int i = 0; i < k; i++) {
        int max_idx = i;
        float max_val = scores[i];
        for (int j = i + 1; j < n; j++) {
            if (scores[j] > max_val) {
                max_val = scores[j];
                max_idx = j;
            }
        }
        indices[i] = max_idx;
        weights[i] = max_val;
    }

    // Softmax over top-k
    float max_val = weights[0];
    for (int i = 1; i < k; i++) {
        if (weights[i] > max_val) max_val = weights[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        weights[i] = expf(weights[i] - max_val);
        sum += weights[i];
    }
    for (int i = 0; i < k; i++) {
        weights[i] /= sum;
    }
}

// ============================================================================
// Weight file loading
// ============================================================================

typedef struct {
    void *data;
    size_t size;
    int fd;
} MappedFile;

typedef struct {
    uint64_t offset;
    uint64_t size;
    int shape[4];
    int num_dims;
} TensorInfo;

typedef struct {
    void *base;
    size_t header_size;
    TensorInfo *tensors;
    int num_tensors;
    char **tensor_names;
    int current_layer;
} WeightData;

static int open_weights(const char *path, MappedFile *mf, WeightData *wd) {
    memset(mf, 0, sizeof(*mf));
    memset(wd, 0, sizeof(*wd));
    wd->current_layer = -1;

    mf->fd = open(path, O_RDONLY);
    if (mf->fd < 0) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    struct stat st;
    if (fstat(mf->fd, &st) < 0) {
        close(mf->fd);
        return -1;
    }
    mf->size = st.st_size;

    mf->data = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->data == MAP_FAILED) {
        close(mf->fd);
        return -1;
    }

    // Read header: 4 bytes header_size, then JSON manifest
    uint32_t header_size;
    memcpy(&header_size, mf->data, 4);
    wd->header_size = header_size;
    wd->base = (uint8_t *)mf->data + ((header_size + 63) & ~63UL);

    // Try binary index first for fast loading
    FILE *idx_f = fopen("tensor_index.bin", "rb");
    if (idx_f) {
        uint32_t magic, version, num_tensors_idx;
        uint64_t data_start_offset_read;
        if (fread(&magic, 4, 1, idx_f) == 1 &&
            fread(&version, 4, 1, idx_f) == 1 &&
            fread(&num_tensors_idx, 4, 1, idx_f) == 1 &&
            fread(&data_start_offset_read, 8, 1, idx_f) == 1 &&
            magic == 0x54504549 && version == 1) {
            fprintf(stderr, "Using binary tensor_index.bin (%d tensors), data_start=%lu\n",
                    num_tensors_idx, (unsigned long)data_start_offset_read);

            wd->num_tensors = num_tensors_idx;
            wd->tensors = (TensorInfo *)calloc(wd->num_tensors, sizeof(TensorInfo));
            wd->tensor_names = (char **)calloc(wd->num_tensors, sizeof(char *));
            CHECK_NULL(wd->tensors);
            CHECK_NULL(wd->tensor_names);

            for (uint32_t i = 0; i < num_tensors_idx; i++) {
                uint32_t name_len;
                if (fread(&name_len, 4, 1, idx_f) != 1) break;
                wd->tensor_names[i] = (char *)malloc(name_len + 1);
                if (fread(wd->tensor_names[i], 1, name_len, idx_f) != name_len) break;
                wd->tensor_names[i][name_len] = '\0';
                uint64_t offset, size;
                if (fread(&offset, 8, 1, idx_f) != 1) break;
                if (fread(&size, 8, 1, idx_f) != 1) break;
                wd->tensors[i].offset = offset;
                wd->tensors[i].size = size;
            }

            fclose(idx_f);
            fprintf(stderr, "Binary index loaded.\n");
            fprintf(stderr, "First 5 tensors:\n");
            for (int i = 0; i < 5 && i < wd->num_tensors; i++) {
                fprintf(stderr, "  [%d] %s: offset=%lu, size=%lu\n",
                        i, wd->tensor_names[i], (unsigned long)wd->tensors[i].offset,
                        (unsigned long)wd->tensors[i].size);
            }
            fprintf(stderr, "Tensor loading setup complete.\n");
            return 0;
        }
        fclose(idx_f);
    }

    fprintf(stderr, "tensor_index.bin not found, cannot load weights efficiently\n");
    return -1;
}

static int find_tensor(WeightData *wd, const char *name) {
    for (int i = 0; i < wd->num_tensors; i++) {
        if (wd->tensor_names[i] && strcmp(wd->tensor_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

static void load_tensor(WeightData *wd, const char *name, void *dest) {
    int idx = find_tensor(wd, name);
    if (idx < 0) {
        fprintf(stderr, "Warning: tensor '%s' not found\n", name);
        return;
    }

    TensorInfo *t = &wd->tensors[idx];
    // Data starts at absolute file offset computed as:
    //   data_start = 4 + header_size
    //   pad = (64 - (data_start % 64)) % 64
    //   DATA_START = data_start + pad = ((header_size + 63) & ~63) + 4
    size_t data_start = 4 + wd->header_size;
    size_t data_start_aligned = (data_start + 63) & ~63ULL;
    void *src = (uint8_t *)wd->base + (t->offset - data_start_aligned);
    memcpy(dest, src, t->size);
}

static void load_tensor_to_gpu(WeightData *wd, const char *name, void *d_ptr, cudaStream_t stream) {
    int idx = find_tensor(wd, name);
    if (idx < 0) {
        fprintf(stderr, "Warning: tensor '%s' not found\n", name);
        return;
    }
    TensorInfo *t = &wd->tensors[idx];
    size_t data_start = 4 + wd->header_size;
    size_t data_start_aligned = (data_start + 63) & ~63ULL;
    void *src = (uint8_t *)wd->base + (t->offset - data_start_aligned);
    CHECK_CUDA(cudaMemcpyAsync(d_ptr, src, t->size, cudaMemcpyHostToDevice, stream));
}

static void load_tensor_bf16_to_f32(WeightData *wd, const char *name, float *dest) {
    int idx = find_tensor(wd, name);
    if (idx < 0) {
        fprintf(stderr, "Warning: tensor '%s' not found\n", name);
        return;
    }
    TensorInfo *t = &wd->tensors[idx];
    size_t data_start = 4 + wd->header_size;
    size_t data_start_aligned = (data_start + 63) & ~63ULL;
    uint16_t *src = (uint16_t *)((uint8_t *)wd->base + (t->offset - data_start_aligned));
    int n = (int)(t->size / sizeof(uint16_t));
    for (int i = 0; i < n; i++) {
        dest[i] = bf16_to_f32(src[i]);
    }
}

static float *d_scratch_w = NULL;
static size_t d_scratch_w_size = 0;

static void ensure_scratch(size_t bytes) {
    if (d_scratch_w_size >= bytes) return;
    if (d_scratch_w) cudaFree(d_scratch_w);
    CHECK_CUDA(cudaMalloc(&d_scratch_w, bytes));
    d_scratch_w_size = bytes;
}

static void close_weights(MappedFile *mf, WeightData *wd) {
    if (wd->tensors) free(wd->tensors);
    if (wd->tensor_names) {
        for (int i = 0; i < wd->num_tensors; i++) {
            if (wd->tensor_names[i]) free(wd->tensor_names[i]);
        }
        free(wd->tensor_names);
    }
    if (mf->data && mf->data != MAP_FAILED) {
        munmap(mf->data, mf->size);
    }
    if (mf->fd >= 0) close(mf->fd);
    memset(mf, 0, sizeof(*mf));
    memset(wd, 0, sizeof(*wd));
}

// ============================================================================
// CUDA buffers
// ============================================================================

typedef struct {
    float *d_input;
    float *d_output;
    float *d_gate;
    float *d_up;
    float *d_swiglu;
    float *d_rms_out;
    float *d_expert_out;
    float *d_weights;
    float *d_combined;
} LayerBuffers;

static void init_layer_buffers(LayerBuffers *b) {
    CHECK_CUDA(cudaMalloc(&b->d_input, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_output, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_gate, MOE_INTERMEDIATE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_up, MOE_INTERMEDIATE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_swiglu, MOE_INTERMEDIATE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_rms_out, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_expert_out, NUM_EXPERTS_PER_TOK * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_weights, NUM_EXPERTS_PER_TOK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b->d_combined, HIDDEN_DIM * sizeof(float)));
}

static void free_layer_buffers(LayerBuffers *b) {
    if (b->d_input) cudaFree(b->d_input);
    if (b->d_output) cudaFree(b->d_output);
    if (b->d_gate) cudaFree(b->d_gate);
    if (b->d_up) cudaFree(b->d_up);
    if (b->d_swiglu) cudaFree(b->d_swiglu);
    if (b->d_rms_out) cudaFree(b->d_rms_out);
    if (b->d_expert_out) cudaFree(b->d_expert_out);
    if (b->d_weights) cudaFree(b->d_weights);
    if (b->d_combined) cudaFree(b->d_combined);
    memset(b, 0, sizeof(*b));
}

// ============================================================================
// Check if layer is full attention (every 4th layer starting at 3)
// ============================================================================
static inline int is_full_attention(int layer_idx) {
    return (layer_idx >= 3 && (layer_idx - 3) % 4 == 0);
}

// ============================================================================
// KV Cache for full attention layers
// ============================================================================
#define MAX_SEQ_LEN 8192

typedef struct {
    float *k_cache;   // [MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] (CPU)
    float *v_cache;   // [MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] (CPU)
    float *d_k_cache; // GPU copy
    float *d_v_cache; // GPU copy
    int len;
} KVCache;

static KVCache *create_kv_cache(void) {
    KVCache *kv = (KVCache *)calloc(1, sizeof(KVCache));
    size_t size = MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    kv->k_cache = (float *)calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    kv->v_cache = (float *)calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    CHECK_CUDA(cudaMalloc(&kv->d_k_cache, size));
    CHECK_CUDA(cudaMalloc(&kv->d_v_cache, size));
    kv->len = 0;
    return kv;
}

static void free_kv_cache(KVCache *kv) {
    if (!kv) return;
    if (kv->d_k_cache) cudaFree(kv->d_k_cache);
    if (kv->d_v_cache) cudaFree(kv->d_v_cache);
    free(kv->k_cache);
    free(kv->v_cache);
    free(kv);
}

// ============================================================================
// Linear attention state for GatedDeltaNet
// ============================================================================
typedef struct {
    float *conv_state;    // CPU fallback: [(CONV_KERNEL_SIZE-1) * LINEAR_CONV_DIM]
    float *ssm_state;     // CPU fallback: [LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM]
    float *d_conv_state;  // GPU conv state
    float *d_ssm_state;   // GPU ssm state
} LinearAttnState;

static LinearAttnState *create_linear_state(void) {
    LinearAttnState *s = (LinearAttnState *)calloc(1, sizeof(LinearAttnState));
    s->conv_state = (float *)calloc((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, sizeof(float));
    s->ssm_state = (float *)calloc(LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM, sizeof(float));
    CHECK_CUDA(cudaMalloc(&s->d_conv_state, (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s->d_ssm_state, LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(s->d_conv_state, 0, (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemset(s->d_ssm_state, 0, LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float)));
    return s;
}

static void free_linear_state(LinearAttnState *s) {
    if (!s) return;
    free(s->conv_state); free(s->ssm_state);
    if (s->d_conv_state) cudaFree(s->d_conv_state);
    if (s->d_ssm_state) cudaFree(s->d_ssm_state);
    free(s);
}

// ============================================================================
// CPU dequantized matvec for GPTQ-4bit with qzeros format
// ============================================================================
__attribute__((unused))
static void cpu_matvec_bf16(
    const uint16_t *weight_bf16,  // [out_dim, in_dim] BF16 weights
    const float *x,
    float *out,
    int out_dim,
    int in_dim
) {
    // BF16 matrix-vector multiply: out = W @ x where W is BF16
    for (int row = 0; row < out_dim; row++) {
        float sum = 0.0f;
        const uint16_t *w_row = weight_bf16 + row * in_dim;
        for (int col = 0; col < in_dim; col++) {
            sum += bf16_to_f32(w_row[col]) * x[col];
        }
        out[row] = sum;
    }
}

__attribute__((unused))
static void cpu_dequant_matvec_gptq(
    const uint32_t *qweight,
    const uint16_t *scales,
    const uint16_t *qzeros,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int group_size
) {
    int num_groups = out_dim / group_size;
    int packed_cols = in_dim / 8;  // 8 nibbles per uint32

    // For each output row
    for (int row = 0; row < out_dim; row++) {
        int row_group = row / group_size;
        float result = 0.0f;

        for (int col = 0; col < packed_cols; col++) {
            int weight_idx = row * packed_cols + col;
            uint32_t packed_weight = qweight[weight_idx];

            // Get scale and zero for this group
            float scale = bf16_to_f32(scales[row_group * packed_cols + col]);
            int8_t qzero = (int8_t)((qzeros[row_group * packed_cols + col] >> 4) & 0xF);
            if (row_group > 0 && (row_group * packed_cols + col) >= (num_groups * packed_cols)) {
                // Handle out of bounds - use last valid group
                scale = bf16_to_f32(scales[num_groups * packed_cols - packed_cols + col]);
                qzero = (int8_t)((qzeros[num_groups * packed_cols - packed_cols + col] >> 4) & 0xF);
            }

            // Extract 8 4-bit values
            for (int nibble = 0; nibble < 8; nibble++) {
                uint32_t raw_val = (packed_weight >> (nibble * 4)) & 0xF;
                float w = (float)(raw_val - qzero);  // Dequantize with zero point

                int x_idx = col * 8 + nibble;
                if (x_idx < in_dim) {
                    result += w * scale * x[x_idx];
                }
            }
        }

        out[row] = result;
    }
}

// ============================================================================
// CPU RMS norm (full implementation)
// ============================================================================
static void cpu_rms_norm_full(
    const float *x,
    const float *weight,
    float *out,
    int n
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + RMS_NORM_EPS);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * weight[i] * inv_rms;
    }
}

// ============================================================================
// RMSNormGated: out = rms_norm(x) * silu(z) * weight
// ============================================================================
__attribute__((unused))
static void cpu_rms_norm_gated(
    const float *x,
    const float *z,
    const float *weight,
    float *out,
    int dim
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + RMS_NORM_EPS);
    for (int i = 0; i < dim; i++) {
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * weight[i] * silu_z;
    }
}

// ============================================================================
// CPU Conv1D step (for linear attention conv state)
// ============================================================================
__attribute__((unused))
static void cpu_conv1d_step(
    const float *conv_state,
    const float *input,
    const uint16_t *conv_w,
    float *output,
    int conv_dim,
    int kernel_size
) {
    // conv_state holds last (kernel_size-1) inputs
    // output = sum_{i=0}^{kernel_size-1} conv_w[i] * input_i
    // where input_0 = conv_state[(kernel_size-2)*conv_dim], input_{kernel_size-1} = input
    for (int i = 0; i < kernel_size - 1; i++) {
        float w = bf16_to_f32(conv_w[i]);
        const float *src = conv_state + i * conv_dim;
        for (int j = 0; j < conv_dim; j++) {
            output[j] += w * src[j];
        }
    }
    {
        float w = bf16_to_f32(conv_w[kernel_size - 1]);
        for (int j = 0; j < conv_dim; j++) {
            output[j] += w * input[j];
        }
    }
}

// ============================================================================
// Apply RoPE (rotary position embedding) to Q and K
// ============================================================================
static void apply_rotary_emb(
    float *q,
    float *k,
    int position,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim
) {
    float base = 10000000.0f;
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < rotary_dim / 2; d++) {
            float angle = position * powf(base, -2.0f * d / rotary_dim);
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            int idx = h * head_dim + d;
            int idx_rot = h * head_dim + d + head_dim / 2;
            float q0 = q[idx];
            float q1 = q[idx_rot];
            q[idx] = q0 * cos_val - q1 * sin_val;
            q[idx_rot] = q0 * sin_val + q1 * cos_val;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        for (int d = 0; d < rotary_dim / 2; d++) {
            float angle = position * powf(base, -2.0f * d / rotary_dim);
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            int idx = h * head_dim + d;
            int idx_rot = h * head_dim + d + head_dim / 2;
            float k0 = k[idx];
            float k1 = k[idx_rot];
            k[idx] = k0 * cos_val - k1 * sin_val;
            k[idx_rot] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// ============================================================================
// GPU BF16 matvec helper: input on CPU, output on CPU
// ============================================================================
static void gpu_bf16_matvec_cpu_io(
    const float *cpu_x,
    float *cpu_out,
    int out_dim, int in_dim,
    WeightData *wd,
    const char *weight_name,
    cudaStream_t stream
) {
    size_t w_bytes = (size_t)out_dim * in_dim * sizeof(uint16_t);
    ensure_scratch(w_bytes);

    float *d_x, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, in_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, out_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, cpu_x, in_dim * sizeof(float), cudaMemcpyHostToDevice));

    load_tensor_to_gpu(wd, weight_name, d_scratch_w, stream);

    cuda_bf16_matvec((const uint16_t *)d_scratch_w, d_x, d_out, out_dim, in_dim, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(cpu_out, d_out, out_dim * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_out);
}

// ============================================================================
// GPU BF16 matvec helper: input/output on GPU, no CPU roundtrip
// ============================================================================
static void gpu_bf16_matvec_direct(
    const float *d_x, float *d_out,
    int out_dim, int in_dim,
    WeightData *wd, const char *weight_name,
    cudaStream_t stream
) {
    size_t w_bytes = (size_t)out_dim * in_dim * sizeof(uint16_t);
    ensure_scratch(w_bytes + 4096);
    load_tensor_to_gpu(wd, weight_name, d_scratch_w, stream);
    cuda_bf16_matvec((uint16_t *)d_scratch_w, d_x, d_out, out_dim, in_dim, stream);
}

// ============================================================================
// Full attention forward (GPU implementation)
// ============================================================================
static void forward_full_attention_gpu(
    const float *d_normed,     // [HIDDEN_DIM] on GPU
    float *d_attn_out,         // [HIDDEN_DIM] on GPU output
    WeightData *wd,
    int layer_idx,
    KVCache *kv,
    int position,
    cudaStream_t stream
) {
    char tname[256];
    int q_dim = NUM_ATTN_HEADS * HEAD_DIM;          // 4096
    int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2; // 8192
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;           // 512

    // Copy d_normed to CPU for BF16 matvec (which uses cpu_io helper)
    float *cpu_normed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    CHECK_CUDA(cudaMemcpy(cpu_normed, d_normed, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Q projection via GPU BF16 matvec
    float *q_proj = (float *)malloc(q_proj_dim * sizeof(float));
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.q_proj.weight", layer_idx);
    gpu_bf16_matvec_cpu_io(cpu_normed, q_proj, q_proj_dim, HIDDEN_DIM, wd, tname, stream);

    // K projection
    float *k_out = (float *)malloc(kv_dim * sizeof(float));
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.k_proj.weight", layer_idx);
    gpu_bf16_matvec_cpu_io(cpu_normed, k_out, kv_dim, HIDDEN_DIM, wd, tname, stream);

    // V projection
    float *v_out = (float *)malloc(kv_dim * sizeof(float));
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.v_proj.weight", layer_idx);
    gpu_bf16_matvec_cpu_io(cpu_normed, v_out, kv_dim, HIDDEN_DIM, wd, tname, stream);

    free(cpu_normed);

    // Split Q and q_gate on CPU
    float *q = (float *)malloc(q_dim * sizeof(float));
    float *q_gate = (float *)malloc(q_dim * sizeof(float));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        memcpy(q + h * HEAD_DIM, q_proj + h * 2 * HEAD_DIM, HEAD_DIM * sizeof(float));
        memcpy(q_gate + h * HEAD_DIM, q_proj + h * 2 * HEAD_DIM + HEAD_DIM,
               HEAD_DIM * sizeof(float));
    }
    free(q_proj);

    // Load per-head Q/K norm weights
    float *q_norm_w = (float *)malloc(HEAD_DIM * sizeof(float));
    float *k_norm_w = (float *)malloc(HEAD_DIM * sizeof(float));
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.q_norm.weight", layer_idx);
    load_tensor_bf16_to_f32(wd, tname, q_norm_w);
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.k_norm.weight", layer_idx);
    load_tensor_bf16_to_f32(wd, tname, k_norm_w);

    // Upload Q, K and norm weights to GPU
    float *d_q, *d_k, *d_v, *d_q_gate, *d_norm_w;
    CHECK_CUDA(cudaMalloc(&d_q, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q_gate, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, kv_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, kv_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_norm_w, HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_q, q, q_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_gate, q_gate, q_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, k_out, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, v_out, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_norm_w, q_norm_w, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
    free(q); free(q_gate); free(k_out); free(v_out);

    // Per-head Q RMS norm on GPU (16 heads, each dim=256, same norm weight)
    float *d_q_normed, *d_k_normed;
    CHECK_CUDA(cudaMalloc(&d_q_normed, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_normed, kv_dim * sizeof(float)));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        cuda_rms_norm(d_q + h * HEAD_DIM, d_norm_w, d_q_normed + h * HEAD_DIM,
                       HEAD_DIM, RMS_NORM_EPS, stream);
    }
    // Per-head K RMS norm (2 heads)
    CHECK_CUDA(cudaMemcpy(d_norm_w, k_norm_w, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        cuda_rms_norm(d_k + h * HEAD_DIM, d_norm_w, d_k_normed + h * HEAD_DIM,
                       HEAD_DIM, RMS_NORM_EPS, stream);
    }
    cudaFree(d_norm_w);
    free(q_norm_w); free(k_norm_w);

    // RoPE on GPU
    float *d_q_rope, *d_k_rope;
    CHECK_CUDA(cudaMalloc(&d_q_rope, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_rope, kv_dim * sizeof(float)));
    cuda_rope(d_q_normed, d_q_rope, NUM_ATTN_HEADS, HEAD_DIM, position, 10000000.0f, stream);
    cuda_rope(d_k_normed, d_k_rope, NUM_KV_HEADS, HEAD_DIM, position, 10000000.0f, stream);
    cudaFree(d_q_normed); cudaFree(d_k_normed);

    // Update KV cache on GPU (must use cudaMemcpyAsync on same stream!)
    int cache_pos = kv->len;
    CHECK_CUDA(cudaMemcpyAsync(kv->d_k_cache + cache_pos * kv_dim, d_k_rope,
                               kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(kv->d_v_cache + cache_pos * kv_dim, d_v,
                               kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    // Also update CPU cache (backward compat)
    CHECK_CUDA(cudaMemcpy(kv->k_cache + cache_pos * kv_dim, d_k_rope,
                          kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(kv->v_cache + cache_pos * kv_dim, d_v,
                          kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
    kv->len++;
    int seq_len = kv->len;

    // Attention scores on GPU
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_scores, NUM_ATTN_HEADS * MAX_SEQ_LEN * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_scores, 0, NUM_ATTN_HEADS * MAX_SEQ_LEN * sizeof(float)));
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    cuda_attn_scores(d_q_rope, kv->d_k_cache, d_scores, HEAD_DIM, kv_dim,
                      seq_len, MAX_SEQ_LEN, scale, NUM_ATTN_HEADS / NUM_KV_HEADS,
                      seq_len, stream);

    // Softmax on GPU
    cuda_attn_softmax(d_scores, seq_len, MAX_SEQ_LEN, stream);

    // Values on GPU
    float *d_context;
    CHECK_CUDA(cudaMalloc(&d_context, q_dim * sizeof(float)));
    cuda_attn_values(d_scores, kv->d_v_cache, d_context, HEAD_DIM, kv_dim,
                      seq_len, MAX_SEQ_LEN, NUM_ATTN_HEADS / NUM_KV_HEADS, stream);

    // Sigmoid gate on GPU (in-place)
    cuda_sigmoid_gate(d_context, d_q_gate, q_dim, stream);

    // O projection via BF16 matvec (cpu_io helper)
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float *cpu_context = (float *)malloc(q_dim * sizeof(float));
    CHECK_CUDA(cudaMemcpy(cpu_context, d_context, q_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.o_proj.weight", layer_idx);
    float *attn_out_cpu = (float *)malloc(HIDDEN_DIM * sizeof(float));
    gpu_bf16_matvec_cpu_io(cpu_context, attn_out_cpu, HIDDEN_DIM, q_dim, wd, tname, stream);
    CHECK_CUDA(cudaMemcpy(d_attn_out, attn_out_cpu, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Cleanup
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_q_gate); cudaFree(d_q_rope); cudaFree(d_k_rope);
    cudaFree(d_scores); cudaFree(d_context);
    free(cpu_context); free(attn_out_cpu);
}

// ============================================================================
// Full attention forward (CPU implementation)
// ============================================================================
static void forward_full_attention_cpu(
    const float *input,
    float *output,
    WeightData *wd,
    int layer_idx,
    KVCache *kv,
    int position,
    cudaStream_t stream
) {
    char tensor_name[512];

    // Working buffers
    float *normed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float *residual = (float *)malloc(HIDDEN_DIM * sizeof(float));
    memcpy(residual, input, HIDDEN_DIM * sizeof(float));

    // ===== Step 1: Input RMS norm =====
    float *input_ln_weight = (float *)malloc(HIDDEN_DIM * sizeof(float));
    snprintf(tensor_name, sizeof(tensor_name), "layers.%d.input_layernorm.weight", layer_idx);
    load_tensor_bf16_to_f32(wd, tensor_name, input_ln_weight);
    cpu_rms_norm_full(input, input_ln_weight, normed, HIDDEN_DIM);
    free(input_ln_weight);

    // ===== Step 2: QKV Projection =====
    // Q projection outputs num_heads * head_dim * 2 = 8192 (the second half is a sigmoid gate)
    int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;  // 16 * 256 * 2 = 8192
    int q_dim = NUM_ATTN_HEADS * HEAD_DIM;            // 4096
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;             // 512

    float *q_proj_out = (float *)calloc(q_proj_dim, sizeof(float));
    float *k = (float *)calloc(kv_dim, sizeof(float));
    float *v = (float *)calloc(kv_dim, sizeof(float));

    // Q projection via GPU BF16 matvec
    {
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.self_attn.q_proj.weight", layer_idx);
        gpu_bf16_matvec_cpu_io(normed, q_proj_out, q_proj_dim, HIDDEN_DIM, wd, tensor_name, stream);
    }

    // K projection via GPU BF16 matvec
    {
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.self_attn.k_proj.weight", layer_idx);
        gpu_bf16_matvec_cpu_io(normed, k, kv_dim, HIDDEN_DIM, wd, tensor_name, stream);
    }

    // V projection via GPU BF16 matvec
    {
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.self_attn.v_proj.weight", layer_idx);
        gpu_bf16_matvec_cpu_io(normed, v, kv_dim, HIDDEN_DIM, wd, tensor_name, stream);
    }

    // Split q_proj_out into Q and gate
    float *q = (float *)malloc(q_dim * sizeof(float));
    float *q_gate = (float *)malloc(q_dim * sizeof(float));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        float *src = q_proj_out + h * (2 * HEAD_DIM);
        memcpy(q + h * HEAD_DIM, src, HEAD_DIM * sizeof(float));
        memcpy(q_gate + h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
    }
    free(q_proj_out);

    // ===== Step 3: Q and K RMS norm per head =====
    float *q_norm_weight = (float *)malloc(HEAD_DIM * sizeof(float));
    float *k_norm_weight = (float *)malloc(HEAD_DIM * sizeof(float));

    snprintf(tensor_name, sizeof(tensor_name), "layers.%d.self_attn.q_norm.weight", layer_idx);
    load_tensor_bf16_to_f32(wd, tensor_name, q_norm_weight);
    snprintf(tensor_name, sizeof(tensor_name), "layers.%d.self_attn.k_norm.weight", layer_idx);
    load_tensor_bf16_to_f32(wd, tensor_name, k_norm_weight);

    // Apply per-head Q RMS norm
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        float *qh = q + h * HEAD_DIM;
        float sum_sq = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++) sum_sq += qh[i] * qh[i];
        float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
        for (int i = 0; i < HEAD_DIM; i++) {
            qh[i] = qh[i] * inv_rms * q_norm_weight[i];
        }
    }
    // Apply per-head K RMS norm
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        float *kh = k + h * HEAD_DIM;
        float sum_sq = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++) sum_sq += kh[i] * kh[i];
        float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
        for (int i = 0; i < HEAD_DIM; i++) {
            kh[i] = kh[i] * inv_rms * k_norm_weight[i];
        }
    }
    free(q_norm_weight);
    free(k_norm_weight);

    // ===== Step 4: Apply RoPE =====
    int rotary_dim = 64;  // HEAD_DIM * 0.25 = 256 * 0.25 = 64
    apply_rotary_emb(q, k, position, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, rotary_dim);

    // ===== Step 5: Update KV cache =====
    int cache_pos = kv->len;
    memcpy(kv->k_cache + cache_pos * kv_dim, k, kv_dim * sizeof(float));
    memcpy(kv->v_cache + cache_pos * kv_dim, v, kv_dim * sizeof(float));
    kv->len++;

    // ===== Step 6: Scaled dot-product attention (GQA) =====
    int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;  // 8
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float *attn_out = (float *)calloc(q_dim, sizeof(float));

    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * HEAD_DIM;
        float *oh = attn_out + h * HEAD_DIM;

        // Compute attention scores for all cached positions
        for (int p = 0; p < kv->len; p++) {
            float *kp = kv->k_cache + p * kv_dim + kv_h * HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += qh[d] * kp[d];
            }
            float score = dot * scale;

            // Softmax
            float max_score = 0.0f;
            if (p == 0) {
                max_score = score;
            } else {
                // Find max in scores so far
                for (int pp = 0; pp <= p; pp++) {
                    float s = 0.0f;
                    float *kp_tmp = kv->k_cache + pp * kv_dim + kv_h * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) s += qh[d] * kp_tmp[d];
                    s *= scale;
                    if (s > max_score) max_score = s;
                }
            }

            float exp_sum = 0.0f;
            for (int pp = 0; pp < kv->len; pp++) {
                float s = 0.0f;
                float *kp_tmp = kv->k_cache + pp * kv_dim + kv_h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) s += qh[d] * kp_tmp[d];
                s *= scale;
                exp_sum += expf(s - max_score);
            }

            float attn_weight = expf(score - max_score) / exp_sum;

            // Weighted sum of values
            float *vp = kv->v_cache + p * kv_dim + kv_h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                oh[d] += attn_weight * vp[d];
            }
        }
    }

    // ===== Step 7: Apply sigmoid gate to attention output =====
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-q_gate[i]));
        attn_out[i] *= g;
    }

    // ===== Step 8: O projection via GPU BF16 matvec =====
    float *attn_projected = (float *)calloc(HIDDEN_DIM, sizeof(float));
    {
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.self_attn.o_proj.weight", layer_idx);
        gpu_bf16_matvec_cpu_io(attn_out, attn_projected, HIDDEN_DIM, q_dim, wd, tensor_name, stream);
    }

    // ===== Step 9: Residual connection =====
    for (int i = 0; i < HIDDEN_DIM; i++) {
        output[i] = residual[i] + attn_projected[i];
    }

    // Cleanup
    free(normed);
    free(residual);
    free(q);
    free(q_gate);
    free(k);
    free(v);
    free(attn_out);
    free(attn_projected);
}

// ============================================================================
// Linear attention forward (GatedDeltaNet) - stub for now
// ============================================================================
static void forward_linear_attention_cpu(
    const float *input,
    float *output,
    WeightData *wd,
    int layer_idx,
    LinearAttnState *state,
    int position
) {
    // Linear attention uses GatedDeltaNet architecture
    // This is more complex - implementing delta accumulation
    // For now, use pass-through as stub

    (void)wd;
    (void)state;
    (void)position;

    fprintf(stderr, "  Layer %d: linear attention (stub - pass through)\n", layer_idx);
    memcpy(output, input, HIDDEN_DIM * sizeof(float));
}

// ============================================================================
// Linear attention forward (GPU implementation)
// ============================================================================
static void forward_linear_attention_gpu(
    const float *d_normed,    // [HIDDEN_DIM] on GPU
    float *d_attn_out,        // [HIDDEN_DIM] on GPU
    WeightData *wd,
    int layer_idx,
    LinearAttnState *state,
    cudaStream_t stream
) {
    char tname[256];

    // Allocate temp GPU buffers for projections
    float *d_qkv;   CHECK_CUDA(cudaMalloc(&d_qkv, LINEAR_CONV_DIM * sizeof(float)));
    float *d_z;     CHECK_CUDA(cudaMalloc(&d_z, LINEAR_TOTAL_VALUE * sizeof(float)));
    float *d_beta;  CHECK_CUDA(cudaMalloc(&d_beta, LINEAR_NUM_V_HEADS * sizeof(float)));
    float *d_alpha; CHECK_CUDA(cudaMalloc(&d_alpha, LINEAR_NUM_V_HEADS * sizeof(float)));

    // 4 parallel projections: QKV, Z, B, A (all BF16 matvec)
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_qkv, LINEAR_CONV_DIM, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_z, LINEAR_TOTAL_VALUE, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_beta, LINEAR_NUM_V_HEADS, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_alpha, LINEAR_NUM_V_HEADS, HIDDEN_DIM, wd, tname, stream);

    // Conv1d step
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.conv1d.weight", layer_idx);
    uint16_t *d_conv_w;
    CHECK_CUDA(cudaMalloc(&d_conv_w, LINEAR_CONV_DIM * CONV_KERNEL_SIZE * sizeof(uint16_t)));
    load_tensor_to_gpu(wd, tname, d_conv_w, stream);
    float *d_conv_out;
    CHECK_CUDA(cudaMalloc(&d_conv_out, LINEAR_CONV_DIM * sizeof(float)));
    cuda_conv1d_step(state->d_conv_state, d_qkv, d_conv_w, d_conv_out, LINEAR_CONV_DIM, stream);
    cudaFree(d_conv_w);

    // Split conv_out: q[0:2048], k[2048:4096], v[4096:12288]
    float *d_q = d_conv_out;
    float *d_k = d_conv_out + LINEAR_TOTAL_KEY;
    float *d_v = d_conv_out + 2 * LINEAR_TOTAL_KEY;

    // Q/K per-head RMS norm (in-place)
    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
    cuda_rms_norm_qk(d_q, d_k, LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM, inv_scale, stream);

    // Load A_log and dt_bias
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.A_log", layer_idx);
    float *d_A_log; CHECK_CUDA(cudaMalloc(&d_A_log, LINEAR_NUM_V_HEADS * sizeof(float)));
    load_tensor_to_gpu(wd, tname, d_A_log, stream);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *d_dt_bias; CHECK_CUDA(cudaMalloc(&d_dt_bias, LINEAR_NUM_V_HEADS * sizeof(uint16_t)));
    load_tensor_to_gpu(wd, tname, d_dt_bias, stream);

    // Compute decay and beta gates
    float *d_g_decay;  CHECK_CUDA(cudaMalloc(&d_g_decay, LINEAR_NUM_V_HEADS * sizeof(float)));
    float *d_beta_gate; CHECK_CUDA(cudaMalloc(&d_beta_gate, LINEAR_NUM_V_HEADS * sizeof(float)));
    cuda_compute_decay_beta(d_alpha, d_beta, d_A_log, d_dt_bias,
                             d_g_decay, d_beta_gate, LINEAR_NUM_V_HEADS, stream);
    cudaFree(d_A_log); cudaFree(d_dt_bias); cudaFree(d_alpha); cudaFree(d_beta);

    // Gated delta net recurrence
    float *d_out_values;
    CHECK_CUDA(cudaMalloc(&d_out_values, LINEAR_TOTAL_VALUE * sizeof(float)));
    cuda_gated_delta_net_step(state->d_ssm_state, d_q, d_k, d_v,
                               d_g_decay, d_beta_gate, d_out_values,
                               LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                               LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS, stream);
    cudaFree(d_g_decay); cudaFree(d_beta_gate);

    // Gated RMS norm: weight is float32 in file, kernel expects uint16_t (BF16)
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.norm.weight", layer_idx);
    float *cpu_norm_w = (float *)malloc(LINEAR_VALUE_DIM * sizeof(float));
    load_tensor(wd, tname, cpu_norm_w);
    uint16_t *cpu_norm_bf16 = (uint16_t *)malloc(LINEAR_VALUE_DIM * sizeof(uint16_t));
    for (int i = 0; i < LINEAR_VALUE_DIM; i++) {
        cpu_norm_bf16[i] = f32_to_bf16(cpu_norm_w[i]);
    }
    uint16_t *d_norm_w;
    CHECK_CUDA(cudaMalloc(&d_norm_w, LINEAR_VALUE_DIM * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemcpy(d_norm_w, cpu_norm_bf16, LINEAR_VALUE_DIM * sizeof(uint16_t), cudaMemcpyHostToDevice));
    free(cpu_norm_w);
    free(cpu_norm_bf16);
    float *d_gated;
    CHECK_CUDA(cudaMalloc(&d_gated, LINEAR_TOTAL_VALUE * sizeof(float)));
    cuda_gated_rms_norm(d_out_values, d_z, d_norm_w, d_gated,
                          LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, RMS_NORM_EPS, stream);
    cudaFree(d_norm_w); cudaFree(d_out_values); cudaFree(d_z);

    // Output projection: [2048, 8192] BF16 matvec
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.out_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(d_gated, d_attn_out, HIDDEN_DIM, LINEAR_TOTAL_VALUE, wd, tname, stream);
    cudaFree(d_gated);

    cudaFree(d_qkv);
    cudaFree(d_conv_out);
}

// ============================================================================
// Forward layer with full computation (CPU path for validation)
// ============================================================================

__attribute__((unused))
static void forward_layer_cpu(
    const float *input,
    float *output,
    WeightData *wd,
    int layer_idx,
    KVCache **kv_caches,
    LinearAttnState **linear_states
) {
    char tensor_name[256];

    // Allocate working buffers
    float *hidden = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float *hidden_normed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float *attn_output = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float *post_normed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float *routing_scores = (float *)malloc(NUM_EXPERTS * sizeof(float));
    int *topk_indices = (int *)malloc(NUM_EXPERTS_PER_TOK * sizeof(int));
    float *topk_weights = (float *)malloc(NUM_EXPERTS_PER_TOK * sizeof(float));
    float *moe_output = (float *)malloc(HIDDEN_DIM * sizeof(float));
    float *shared_output = (float *)malloc(HIDDEN_DIM * sizeof(float));

    if (!hidden || !hidden_normed || !attn_output || !post_normed ||
        !routing_scores || !topk_indices || !topk_weights ||
        !moe_output || !shared_output) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Copy input to hidden (residual buffer)
    memcpy(hidden, input, HIDDEN_DIM * sizeof(float));

    // =========================================================================
    // Step 1: Input RMS norm
    // =========================================================================
    snprintf(tensor_name, sizeof(tensor_name), "layers.%d.input_layernorm.weight", layer_idx);
    float *input_ln_weight = (float *)malloc(HIDDEN_DIM * sizeof(float));
    load_tensor_bf16_to_f32(wd, tensor_name, input_ln_weight);
    cpu_rms_norm(hidden, input_ln_weight, hidden_normed, HIDDEN_DIM);
    free(input_ln_weight);

    // =========================================================================
    // Step 2: Attention (linear or full)
    // =========================================================================
    if (is_full_attention(layer_idx)) {
        fprintf(stderr, "  Layer %d: full attention\n", layer_idx);
        int full_attn_idx = (layer_idx - 3) / 4;
        forward_full_attention_cpu(hidden_normed, attn_output, wd, layer_idx,
                                    kv_caches ? kv_caches[full_attn_idx] : NULL, layer_idx, 0);
    } else {
        fprintf(stderr, "  Layer %d: linear attention\n", layer_idx);
        forward_linear_attention_cpu(hidden_normed, attn_output, wd, layer_idx,
                                     linear_states ? linear_states[layer_idx] : NULL, layer_idx);
    }

    // =========================================================================
    // Step 3: Residual connection
    // =========================================================================
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = hidden[i] + attn_output[i];
    }

    // =========================================================================
    // Step 4: Post-attention RMS norm
    // =========================================================================
    snprintf(tensor_name, sizeof(tensor_name), "layers.%d.post_attention_layernorm.weight", layer_idx);
    float *post_ln_weight = (float *)malloc(HIDDEN_DIM * sizeof(float));
    load_tensor_bf16_to_f32(wd, tensor_name, post_ln_weight);
    cpu_rms_norm(hidden, post_ln_weight, post_normed, HIDDEN_DIM);
    free(post_ln_weight);

    // =========================================================================
    // Step 5: MoE routing (simplified - uniform routing for validation)
    // =========================================================================
    for (int i = 0; i < NUM_EXPERTS_PER_TOK; i++) {
        topk_indices[i] = i;
        topk_weights[i] = 1.0f / NUM_EXPERTS_PER_TOK;
    }

    // =========================================================================
    // Step 6: Expert execution (pass through for validation)
    // =========================================================================
    memset(moe_output, 0, HIDDEN_DIM * sizeof(float));
    memcpy(moe_output, post_normed, HIDDEN_DIM * sizeof(float));

    // =========================================================================
    // Step 7: Shared expert (pass through for validation)
    // =========================================================================
    memcpy(shared_output, post_normed, HIDDEN_DIM * sizeof(float));

    // =========================================================================
    // Step 8: Final residual
    // =========================================================================
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = hidden[i] + moe_output[i] + shared_output[i];
    }

    // =========================================================================
    // Step 9: Output
    // =========================================================================
    memcpy(output, hidden, HIDDEN_DIM * sizeof(float));

    // Cleanup
    free(hidden);
    free(hidden_normed);
    free(attn_output);
    free(post_normed);
    free(routing_scores);
    free(topk_indices);
    free(topk_weights);
    free(moe_output);
    free(shared_output);
}

// ============================================================================
// Forward layer with GPU kernels (MoE path)
// ============================================================================

static void forward_layer_gpu(
    float *d_hidden,         // [HIDDEN_DIM] on GPU, in/out
    WeightData *wd,
    LayerBuffers *b,
    int layer_idx,
    int position,
    KVCache **kv_caches,
    LinearAttnState **linear_states,
    cudaStream_t stream
) {
    char tname[256];
    float *d_norm_w;

    // Step 1: Input RMS norm (weight is BF16 in file, upload as float32)
    CHECK_CUDA(cudaMalloc(&d_norm_w, HIDDEN_DIM * sizeof(float)));
    snprintf(tname, sizeof(tname), "layers.%d.input_layernorm.weight", layer_idx);
    {
        float *cpu_norm = (float *)malloc(HIDDEN_DIM * sizeof(float));
        load_tensor_bf16_to_f32(wd, tname, cpu_norm);
        CHECK_CUDA(cudaMemcpy(d_norm_w, cpu_norm, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
        free(cpu_norm);
    }
    cuda_rms_norm(d_hidden, d_norm_w, b->d_rms_out, HIDDEN_DIM, RMS_NORM_EPS, stream);
    cudaFree(d_norm_w);

    // Step 2: Attention (GPU for full attn, CPU for linear attn)
    if (is_full_attention(layer_idx)) {
        int fa_idx = (layer_idx - 3) / 4;
        forward_full_attention_gpu(b->d_rms_out, b->d_output, wd, layer_idx,
                                    kv_caches[fa_idx], position, stream);
    } else {
        forward_linear_attention_gpu(b->d_rms_out, b->d_output, wd, layer_idx,
                                      linear_states[layer_idx], stream);
    }

    // Step 3: Residual add (hidden += attn_output)
    cuda_residual_add(d_hidden, b->d_output, d_hidden, HIDDEN_DIM, stream);

    // Step 4: Post-attn RMS norm (weight is BF16 in file, upload as float32)
    CHECK_CUDA(cudaMalloc(&d_norm_w, HIDDEN_DIM * sizeof(float)));
    snprintf(tname, sizeof(tname), "layers.%d.post_attention_layernorm.weight", layer_idx);
    {
        float *cpu_norm = (float *)malloc(HIDDEN_DIM * sizeof(float));
        load_tensor_bf16_to_f32(wd, tname, cpu_norm);
        CHECK_CUDA(cudaMemcpy(d_norm_w, cpu_norm, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
        free(cpu_norm);
    }
    cuda_rms_norm(d_hidden, d_norm_w, b->d_rms_out, HIDDEN_DIM, RMS_NORM_EPS, stream);
    cudaFree(d_norm_w);

    // ========== MoE routing (GPU gate matvec + CPU softmax + topK) ==========
    float *d_routing_scores;
    CHECK_CUDA(cudaMalloc(&d_routing_scores, NUM_EXPERTS * sizeof(float)));

    snprintf(tname, sizeof(tname), "layers.%d.mlp.gate.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_rms_out, d_routing_scores, NUM_EXPERTS, HIDDEN_DIM,
                           wd, tname, stream);

    float *cpu_scores = (float *)malloc(NUM_EXPERTS * sizeof(float));
    CHECK_CUDA(cudaMemcpy(cpu_scores, d_routing_scores, NUM_EXPERTS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_routing_scores);

    // CPU softmax
    float max_score = cpu_scores[0];
    for (int i = 1; i < NUM_EXPERTS; i++)
        if (cpu_scores[i] > max_score) max_score = cpu_scores[i];
    float sum_exp = 0.0f;
    for (int i = 0; i < NUM_EXPERTS; i++) {
        cpu_scores[i] = expf(cpu_scores[i] - max_score);
        sum_exp += cpu_scores[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < NUM_EXPERTS; i++) cpu_scores[i] *= inv_sum;

    int topk_idx[NUM_EXPERTS_PER_TOK];
    float topk_w[NUM_EXPERTS_PER_TOK];
    cpu_topk(cpu_scores, topk_idx, topk_w, NUM_EXPERTS, NUM_EXPERTS_PER_TOK);
    free(cpu_scores);

    // ========== Expert forward on GPU ==========
    CHECK_CUDA(cudaMemset(b->d_expert_out, 0,
                          NUM_EXPERTS_PER_TOK * HIDDEN_DIM * sizeof(float)));

    for (int k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
        int eid = topk_idx[k];

        // gate_proj: [MOE_INTERMEDIATE, HIDDEN_DIM] = [512, 2048] GPTQ dequant
        size_t gate_qw_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / 8) * sizeof(uint32_t);
        size_t gate_sc_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * sizeof(uint16_t);
        ensure_scratch(gate_qw_bytes + gate_sc_bytes * 2 + 4096);
        uint32_t *d_gate_qw = (uint32_t *)d_scratch_w;
        uint16_t *d_gate_sc = (uint16_t *)((char *)d_scratch_w + gate_qw_bytes);
        uint16_t *d_gate_qz = (uint16_t *)((char *)d_gate_sc + gate_sc_bytes);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.gate_proj.qweight", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_gate_qw, stream);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.gate_proj.scales", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_gate_sc, stream);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.gate_proj.qzeros", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_gate_qz, stream);
        cuda_dequant_matvec_gptq(d_gate_qw, d_gate_sc, d_gate_qz, b->d_rms_out,
                                  b->d_gate, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, stream);

        // up_proj: [MOE_INTERMEDIATE, HIDDEN_DIM] = [512, 2048] GPTQ dequant
        size_t up_qw_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / 8) * sizeof(uint32_t);
        size_t up_sc_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * sizeof(uint16_t);
        ensure_scratch(up_qw_bytes + up_sc_bytes * 2 + 4096);
        uint32_t *d_up_qw = (uint32_t *)d_scratch_w;
        uint16_t *d_up_sc = (uint16_t *)((char *)d_scratch_w + up_qw_bytes);
        uint16_t *d_up_qz = (uint16_t *)((char *)d_up_sc + up_sc_bytes);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.up_proj.qweight", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_up_qw, stream);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.up_proj.scales", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_up_sc, stream);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.up_proj.qzeros", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_up_qz, stream);
        cuda_dequant_matvec_gptq(d_up_qw, d_up_sc, d_up_qz, b->d_rms_out,
                                  b->d_up, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, stream);

        // SwiGLU(gate_out, up_out) -> intermediate [MOE_INTERMEDIATE]
        cuda_swiglu(b->d_gate, b->d_up, b->d_swiglu, MOE_INTERMEDIATE, stream);

        // down_proj: [HIDDEN_DIM, MOE_INTERMEDIATE] = [2048, 512] GPTQ dequant
        size_t dn_qw_bytes = HIDDEN_DIM * (MOE_INTERMEDIATE / 8) * sizeof(uint32_t);
        size_t dn_sc_bytes = HIDDEN_DIM * (MOE_INTERMEDIATE / GROUP_SIZE) * sizeof(uint16_t);
        ensure_scratch(dn_qw_bytes + dn_sc_bytes * 2 + 4096);
        uint32_t *d_dn_qw = (uint32_t *)d_scratch_w;
        uint16_t *d_dn_sc = (uint16_t *)((char *)d_scratch_w + dn_qw_bytes);
        uint16_t *d_dn_qz = (uint16_t *)((char *)d_dn_sc + dn_sc_bytes);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.down_proj.qweight", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_dn_qw, stream);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.down_proj.scales", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_dn_sc, stream);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.down_proj.qzeros", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_dn_qz, stream);
        cuda_dequant_matvec_gptq(d_dn_qw, d_dn_sc, d_dn_qz, b->d_swiglu,
                                  b->d_expert_out + k * HIDDEN_DIM,
                                  HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE, stream);
    }

    // ========== Weighted sum of expert outputs ==========
    float *d_routing_weights;
    CHECK_CUDA(cudaMalloc(&d_routing_weights, NUM_EXPERTS_PER_TOK * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_routing_weights, topk_w,
                          NUM_EXPERTS_PER_TOK * sizeof(float), cudaMemcpyHostToDevice));
    cuda_weighted_sum(b->d_expert_out, d_routing_weights, b->d_output,
                       NUM_EXPERTS_PER_TOK, HIDDEN_DIM, stream);
    cudaFree(d_routing_weights);

    // ========== Shared expert ==========
    // gate_proj: [MOE_INTERMEDIATE, HIDDEN_DIM] = [512, 2048] BF16
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_rms_out, b->d_gate, MOE_INTERMEDIATE, HIDDEN_DIM,
                           wd, tname, stream);
    // up_proj: [MOE_INTERMEDIATE, HIDDEN_DIM] = [512, 2048] BF16
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_rms_out, b->d_up, MOE_INTERMEDIATE, HIDDEN_DIM,
                           wd, tname, stream);
    cuda_swiglu(b->d_gate, b->d_up, b->d_swiglu, MOE_INTERMEDIATE, stream);

    // down_proj: [HIDDEN_DIM, MOE_INTERMEDIATE] = [2048, 512] BF16
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_swiglu, b->d_combined, HIDDEN_DIM, MOE_INTERMEDIATE,
                           wd, tname, stream);

    // Shared gate: sigmoid(shared_expert_gate @ post_normed)
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    uint16_t *d_sg_vec;
    CHECK_CUDA(cudaMalloc(&d_sg_vec, HIDDEN_DIM * sizeof(uint16_t)));
    load_tensor_to_gpu(wd, tname, d_sg_vec, stream);
    float *d_sg_val;
    CHECK_CUDA(cudaMalloc(&d_sg_val, sizeof(float)));
    cuda_bf16_matvec(d_sg_vec, b->d_rms_out, d_sg_val, 1, HIDDEN_DIM, stream);
    float sg_cpu;
    CHECK_CUDA(cudaMemcpy(&sg_cpu, d_sg_val, sizeof(float), cudaMemcpyDeviceToHost));
    float shared_gate_val = 1.0f / (1.0f + expf(-sg_cpu));
    cudaFree(d_sg_vec); cudaFree(d_sg_val);

    // Scale shared output by gate, then add MoE + shared to hidden
    float *shared_cpu = (float *)malloc(HIDDEN_DIM * sizeof(float));
    CHECK_CUDA(cudaMemcpy(shared_cpu, b->d_combined, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < HIDDEN_DIM; i++) shared_cpu[i] *= shared_gate_val;
    CHECK_CUDA(cudaMemcpy(b->d_combined, shared_cpu, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(shared_cpu);

    // Residual: hidden += moe + shared
    cuda_residual_add(d_hidden, b->d_output, d_hidden, HIDDEN_DIM, stream);
    cuda_residual_add(d_hidden, b->d_combined, d_hidden, HIDDEN_DIM, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

// ============================================================================
// Main
// ============================================================================

static void print_help(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --prompt TEXT    Input prompt (required)\n");
    fprintf(stderr, "  --tokens N       Max tokens to generate (default: 100)\n");
    fprintf(stderr, "  --weights PATH   Path to weights file (default: model_weights.bin)\n");
    fprintf(stderr, "  --help           Show this help\n");
}

int main(int argc, char **argv) {
    const char *prompt = NULL;
    const char *weights_path = "model_weights.bin";
    int max_tokens = 100;

    // Parse arguments
    static struct option long_options[] = {
        {"prompt", required_argument, 0, 'p'},
        {"tokens", required_argument, 0, 't'},
        {"weights", required_argument, 0, 'w'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:t:w:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'p':
                prompt = optarg;
                break;
            case 't':
                max_tokens = atoi(optarg);
                break;
            case 'w':
                weights_path = optarg;
                break;
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    if (prompt == NULL) {
        fprintf(stderr, "Error: --prompt is required\n");
        print_help(argv[0]);
        return 1;
    }

    fprintf(stderr, "=== CUDA Inference Engine ===\n");
    fprintf(stderr, "Prompt: %s\n", prompt);
    fprintf(stderr, "Max tokens: %d\n", max_tokens);
    fprintf(stderr, "Weights: %s\n", weights_path);

    // Initialize CUDA
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "Using device %d: %s\n", device, prop.name);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Initialize cuBLAS
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    // Load tokenizer
    fprintf(stderr, "Loading tokenizer...\n");
    bpe_tokenizer tok;
    if (bpe_load(&tok, "tokenizer.bin") != 0) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // Encode prompt
    uint32_t input_ids[4096];
    int num_tokens = bpe_encode(&tok, prompt, input_ids, 4096);
    fprintf(stderr, "Encoded %d tokens\n", num_tokens);

    // Load weights
    fprintf(stderr, "Loading weights from %s...\n", weights_path);
    MappedFile mf;
    WeightData wd;
    if (open_weights(weights_path, &mf, &wd) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        bpe_free(&tok);
        return 1;
    }

    // Initialize layer buffers
    fprintf(stderr, "Initializing CUDA buffers...\n");
    LayerBuffers buffers;
    init_layer_buffers(&buffers);

    // Initialize KV caches for full attention layers (10 layers: 3, 7, 11, ...)
    KVCache *kv_caches[10];
    int full_attn_idx = 0;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        if (is_full_attention(layer)) {
            kv_caches[full_attn_idx] = create_kv_cache();
            full_attn_idx++;
        }
    }

    // Initialize linear attention states for all 40 layers
    LinearAttnState *linear_states[40];
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        linear_states[layer] = create_linear_state();
    }

    // ========== Autoregressive generation loop ==========
    size_t data_start_aligned = (4 + wd.header_size + 63) & ~63ULL;
    uint16_t *embed_base = (uint16_t *)((uint8_t *)mf.data + data_start_aligned);
    uint16_t *lm_head_data = (uint16_t *)((uint8_t *)mf.data + data_start_aligned + 1017118720ULL);

    // Load final_layer_norm weight (BF16 -> float32)
    float *final_norm_cpu = (float *)malloc(HIDDEN_DIM * sizeof(float));
    uint16_t *final_norm_src = (uint16_t *)((uint8_t *)mf.data + data_start_aligned + 2034237440ULL);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        final_norm_cpu[i] = bf16_to_f32(final_norm_src[i]);
    }

    float *d_hidden;
    CHECK_CUDA(cudaMalloc(&d_hidden, HIDDEN_DIM * sizeof(float)));

    int generation_position = 0;
    uint32_t current_token = input_ids[num_tokens - 1];
    int generated = 0;

    // Allocate reusable per-step buffers
    float *d_final_norm_w, *d_logits;
    uint16_t *d_lm_chunk;
    CHECK_CUDA(cudaMalloc(&d_final_norm_w, HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lm_chunk, LM_CHUNK * HIDDEN_DIM * sizeof(uint16_t)));

    CHECK_CUDA(cudaMemcpy(d_final_norm_w, final_norm_cpu, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(final_norm_cpu);

    for (int t = 0; t < max_tokens; t++) {
        // ---- Token embedding (BF16 -> float32 on CPU) ----
        uint16_t *token_embed = embed_base + (size_t)current_token * HIDDEN_DIM;
        float cpu_embed[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            cpu_embed[i] = bf16_to_f32(token_embed[i]);
        }
        CHECK_CUDA(cudaMemcpy(d_hidden, cpu_embed, HIDDEN_DIM * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (t == 0) {
            float ss=0; for(int i=0;i<HIDDEN_DIM;i++)ss+=cpu_embed[i]*cpu_embed[i];
            fprintf(stderr,"  embed token=%d rms=%.4f first4=%.3f %.3f %.3f %.3f\n",
                current_token,sqrtf(ss/HIDDEN_DIM),cpu_embed[0],cpu_embed[1],cpu_embed[2],cpu_embed[3]);
        }

        // ---- 40 transformer layers ----
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            forward_layer_gpu(d_hidden, &wd, &buffers, layer, generation_position,
                               kv_caches, linear_states, stream);
            if (t == 0 && (layer < 3 || layer == 3 || layer >= 37)) {
                float *dbg = (float *)malloc(HIDDEN_DIM * sizeof(float));
                CHECK_CUDA(cudaMemcpy(dbg, d_hidden, HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost));
                float ss = 0; int n = 0;
                for (int i = 0; i < HIDDEN_DIM; i++) { if (isnan(dbg[i])) n++; ss += dbg[i]*dbg[i]; }
                fprintf(stderr, "  L%d rms=%.4f nan=%d\n", layer, sqrtf(ss/HIDDEN_DIM), n);
                free(dbg);
            }
        }

        // ---- Final RMS norm ----
        cuda_rms_norm(d_hidden, d_final_norm_w, buffers.d_rms_out,
                      HIDDEN_DIM, RMS_NORM_EPS, stream);

        // ---- lm_head: chunked BF16 matvec [248320, 2048] ----
        CHECK_CUDA(cudaMemset(d_logits, 0, VOCAB_SIZE * sizeof(float)));
        for (int chunk = 0; chunk < VOCAB_SIZE; chunk += LM_CHUNK) {
            int cs = (chunk + LM_CHUNK <= VOCAB_SIZE) ? LM_CHUNK : (VOCAB_SIZE - chunk);
            size_t off = (size_t)chunk * HIDDEN_DIM;
            CHECK_CUDA(cudaMemcpy(d_lm_chunk, lm_head_data + off,
                                  (size_t)cs * HIDDEN_DIM * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice));
            cuda_bf16_matvec(d_lm_chunk, buffers.d_rms_out, d_logits + chunk,
                              cs, HIDDEN_DIM, stream);
        }

        // ---- Argmax ----
        CHECK_CUDA(cudaStreamSynchronize(stream));
        float *cpu_logits = (float *)malloc(VOCAB_SIZE * sizeof(float));
        CHECK_CUDA(cudaMemcpy(cpu_logits, d_logits, VOCAB_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));

        int next_token = 0;
        float max_logit = cpu_logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (cpu_logits[i] > max_logit) {
                max_logit = cpu_logits[i];
                next_token = i;
            }
        }
        if (generated == 0) fprintf(stderr, "[dbg] t=%d max_l=%.2f l[0:3]=%.2f %.2f %.2f\n", next_token, max_logit, cpu_logits[0], cpu_logits[1], cpu_logits[2]);
        free(cpu_logits);

        // Check EOS before printing
        if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
            generated++;
            break;
        }

        // ---- Decode and print ----
        if (t == 0) fprintf(stderr, "\n");
        char token_str[256];
        int token_len = bpe_decode_token(&tok, next_token, token_str, sizeof(token_str));
        if (token_len > 0) {
            fwrite(token_str, 1, token_len, stdout);
            fflush(stdout);
        }

        generation_position++;
        current_token = next_token;
        generated++;
    }

    cudaFree(d_final_norm_w);
    cudaFree(d_logits);
    cudaFree(d_lm_chunk);

    fprintf(stderr, "\n\nGenerated %d tokens\n", generated);

    // Free KV caches
    full_attn_idx = 0;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        if (is_full_attention(layer)) {
            free_kv_cache(kv_caches[full_attn_idx]);
            full_attn_idx++;
        }
    }

    // Free linear states
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        free_linear_state(linear_states[layer]);
    }

    cudaFree(d_hidden);
    free_layer_buffers(&buffers);
    close_weights(&mf, &wd);
    bpe_free(&tok);
    cublasDestroy(cublas);
    cudaStreamDestroy(stream);

    fprintf(stderr, "Done.\n");
    return 0;
}