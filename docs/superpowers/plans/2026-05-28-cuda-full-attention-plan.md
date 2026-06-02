# Full Attention GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Full Attention 的 attention compute（scores/softmax/context/gate）从 CPU 替换为正确的 CUDA kernel，KV cache 迁到 GPU

**Architecture:** 4 个 CUDA kernel 从 metal_infer 移植（warp shuffle 替代 simd_sum），GPU 持久化 KV cache 消除 CPU↔GPU 拷贝

**Tech Stack:** CUDA 13.0, nvcc, PyTorch (验证), warp shuffle intrinsics

---

## 文件结构

```
cuda_infer/
├── kernels.cu          # [修改] 新增 4 个 attention kernel
├── kernels.h           # [修改] 新增 4 个 extern "C" 声明
├── infer.cu            # [修改] 重写 forward_full_attention，GPU KV cache
└── tests/
    ├── test_attn_scores.py    # [新建] attn_scores kernel 验证
    ├── test_attn_softmax.py   # [新建] attn_softmax kernel 验证
    ├── test_attn_values.py    # [新建] attn_values kernel 验证
    └── test_sigmoid_gate.py   # [新建] sigmoid_gate kernel 验证
```

---

### Task 1: attn_scores Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_attn_scores.py`

- [ ] **Step 1: 添加 kernel 声明到 kernels.h**

```cuda
void cuda_attn_scores(
    const float *d_q, const float *d_k_cache,
    float *d_scores, int head_dim, int kv_dim,
    int seq_len, int seq_stride, float scale,
    int heads_per_kv, int num_seq_tgs,
    cudaStream_t stream);
```

- [ ] **Step 2: 写 PyTorch 参考测试**

`cuda_infer/tests/test_attn_scores.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
KV_DIM = NUM_KV_HEADS * HEAD_DIM
MAX_SEQ = 32
HEADS_PER_KV = NUM_ATTN_HEADS // NUM_KV_HEADS

def cuda_attn_scores(q, k_cache, head_dim, kv_dim, seq_len, seq_stride):
    lib = get_lib()
    scores = torch.zeros(NUM_ATTN_HEADS * seq_stride, dtype=torch.float32, device='cuda')
    scale = 1.0 / (head_dim ** 0.5)
    lib.cuda_attn_scores(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k_cache.data_ptr()),
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_int(head_dim), ctypes.c_int(kv_dim),
        ctypes.c_int(seq_len), ctypes.c_int(seq_stride),
        ctypes.c_float(scale), ctypes.c_int(HEADS_PER_KV),
        ctypes.c_int(seq_len),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return scores

def attn_scores_ref(q, k_cache, seq_len):
    scale = 1.0 / (HEAD_DIM ** 0.5)
    q = q.cpu().reshape(NUM_ATTN_HEADS, HEAD_DIM)
    scores = torch.zeros(NUM_ATTN_HEADS, seq_len)
    for h in range(NUM_ATTN_HEADS):
        kv_h = h // HEADS_PER_KV
        for p in range(seq_len):
            kp = k_cache[p * KV_DIM + kv_h * HEAD_DIM : p * KV_DIM + (kv_h+1) * HEAD_DIM]
            scores[h, p] = torch.dot(q[h], kp) * scale
    return scores

def test_attn_scores():
    torch.manual_seed(42)
    seq_len = 5
    q = torch.randn(NUM_ATTN_HEADS * HEAD_DIM, device='cuda')
    k_cache = torch.randn(MAX_SEQ * KV_DIM, device='cuda')
    expected = attn_scores_ref(q, k_cache.cpu(), seq_len)
    actual = cuda_attn_scores(q, k_cache, HEAD_DIM, KV_DIM, seq_len, MAX_SEQ)
    actual = actual.cpu().reshape(NUM_ATTN_HEADS, MAX_SEQ)[:, :seq_len]
    assert_close(actual, expected, atol=1e-4, msg="attn_scores")
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_attn_scores.py -v
```
Expected: FAIL — `cuda_attn_scores` not found in libkernels.so.

- [ ] **Step 4: 实现 kernel**

添加到 `kernels.cu` 末尾：

```cuda
// ============================================================================
// Kernel: Attention scores (Q @ K^T / sqrt(d)) — batched over (pos, head)
// ============================================================================
// Grid: (num_heads, seq_len) — one threadgroup per (head, position)
// Each TG of 256 threads reduces dot product over head_dim=256
// GQA: kv_h = head / heads_per_kv

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

    extern __shared__ float sdata[];
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
    dim3 gridDim(num_seq_tgs, NUM_ATTN_HEADS);
    attn_scores_kernel<<<gridDim, blockDim, blockDim.x * sizeof(float), stream>>>(
        d_q, d_k_cache, d_scores, head_dim, kv_dim,
        seq_len, seq_stride, scale, heads_per_kv);
    CHECK_CUDA(cudaGetLastError());
}
}
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_attn_scores.py -v
```
Expected: 1 passed.

- [ ] **Step 6: 提交**

```bash
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_attn_scores.py
git commit -m "feat(cuda_infer): add attention scores kernel (Q @ K^T)"
```

---

### Task 2: attn_softmax Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_attn_softmax.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_attn_softmax(
    float *d_scores, int seq_len, int seq_stride,
    cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_attn_softmax.py`:

```python
import torch
import torch.nn.functional as F
from conftest import get_lib, assert_close
import ctypes

NUM_ATTN_HEADS = 16
MAX_SEQ = 32

def cuda_attn_softmax(scores, seq_len, seq_stride):
    lib = get_lib()
    lib.cuda_attn_softmax(
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_int(seq_len), ctypes.c_int(seq_stride),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return scores

def test_attn_softmax():
    torch.manual_seed(42)
    seq_len = 7
    raw = torch.randn(NUM_ATTN_HEADS, MAX_SEQ, device='cuda')
    raw[:, seq_len:] = -1e30
    expected = F.softmax(raw.cpu()[:, :seq_len], dim=-1)
    actual = cuda_attn_softmax(raw, seq_len, MAX_SEQ)
    actual = actual.cpu().reshape(NUM_ATTN_HEADS, MAX_SEQ)[:, :seq_len]
    assert_close(actual, expected, atol=1e-5, msg="attn_softmax")
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_attn_softmax.py -v
```
Expected: FAIL.

- [ ] **Step 4: 实现 kernel**

```cuda
// ============================================================================
// Kernel: Attention softmax (in-place, batched per head)
// ============================================================================

__global__ void attn_softmax_kernel(
    float *scores, int seq_len, int seq_stride
) {
    int h = blockIdx.x;
    float *s = scores + h * seq_stride;

    __shared__ float shared[256];

    float local_max = -1e30f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        if (s[i] > local_max) local_max = s[i];
    }

    int tid = threadIdx.x;
    shared[tid] = local_max;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2 && shared[tid + s2] > shared[tid])
            shared[tid] = shared[tid + s2];
        __syncthreads();
    }
    float max_val = shared[0];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
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

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        s[i] *= inv_sum;
    }
}

extern "C" {
void cuda_attn_softmax(
    float *d_scores, int seq_len, int seq_stride,
    cudaStream_t stream
) {
    attn_softmax_kernel<<<NUM_ATTN_HEADS, 256, 0, stream>>>(
        d_scores, seq_len, seq_stride);
    CHECK_CUDA(cudaGetLastError());
}
}
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_attn_softmax.py -v
```
Expected: 1 passed.

- [ ] **Step 6: 提交**

```bash
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_attn_softmax.py
git commit -m "feat(cuda_infer): add attention softmax kernel"
```

---

### Task 3: attn_values Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_attn_values.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_attn_values(
    const float *d_scores, const float *d_v_cache,
    float *d_out, int head_dim, int kv_dim,
    int seq_len, int seq_stride, int heads_per_kv,
    cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_attn_values.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
KV_DIM = NUM_KV_HEADS * HEAD_DIM
MAX_SEQ = 32
HEADS_PER_KV = NUM_ATTN_HEADS // NUM_KV_HEADS

def cuda_attn_values(scores, v_cache, head_dim, kv_dim, seq_len, seq_stride):
    lib = get_lib()
    out = torch.zeros(NUM_ATTN_HEADS * head_dim, dtype=torch.float32, device='cuda')
    lib.cuda_attn_values(
        ctypes.c_void_p(scores.data_ptr()),
        ctypes.c_void_p(v_cache.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(head_dim), ctypes.c_int(kv_dim),
        ctypes.c_int(seq_len), ctypes.c_int(seq_stride),
        ctypes.c_int(HEADS_PER_KV),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return out

def attn_values_ref(scores, v_cache, seq_len):
    scores = scores.cpu().reshape(NUM_ATTN_HEADS, MAX_SEQ)[:, :seq_len]
    out = torch.zeros(NUM_ATTN_HEADS, HEAD_DIM)
    for h in range(NUM_ATTN_HEADS):
        kv_h = h // HEADS_PER_KV
        for p in range(seq_len):
            vp = v_cache[p * KV_DIM + kv_h * HEAD_DIM : p * KV_DIM + (kv_h+1) * HEAD_DIM]
            out[h] += scores[h, p] * vp
    return out.flatten()

def test_attn_values():
    torch.manual_seed(42)
    seq_len = 5
    scores = torch.rand(NUM_ATTN_HEADS, MAX_SEQ, device='cuda')
    scores[:, seq_len:] = 0
    scores[:, :seq_len] = torch.softmax(scores[:, :seq_len], dim=-1)
    v_cache = torch.randn(MAX_SEQ * KV_DIM, device='cuda')

    expected = attn_values_ref(scores, v_cache.cpu(), seq_len)
    actual = cuda_attn_values(scores, v_cache, HEAD_DIM, KV_DIM, seq_len, MAX_SEQ)
    assert_close(actual, expected, atol=1e-4, msg="attn_values")
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_attn_values.py -v
```
Expected: FAIL.

- [ ] **Step 4: 实现 kernel**

```cuda
// ============================================================================
// Kernel: Attention value aggregation (softmax @ V)
// ============================================================================
// One thread per (head, dim): out[head*head_dim + d] = sum_p(scores[p] * V_cache[p, kv_h, d])

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
    int total = NUM_ATTN_HEADS * head_dim;
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
    int total = NUM_ATTN_HEADS * head_dim;
    dim3 blockDim(256);
    dim3 gridDim((total + 255) / 256);
    attn_values_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_scores, d_v_cache, d_out, head_dim, kv_dim,
        seq_len, seq_stride, heads_per_kv);
    CHECK_CUDA(cudaGetLastError());
}
}
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_attn_values.py -v
```
Expected: 1 passed.

- [ ] **Step 6: 提交**

```bash
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_attn_values.py
git commit -m "feat(cuda_infer): add attention values kernel (softmax @ V)"
```

---

### Task 4: sigmoid_gate Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_sigmoid_gate.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_sigmoid_gate(
    float *d_x_out, const float *d_gate, int dim,
    cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_sigmoid_gate.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

NUM_ATTN_HEADS = 16
HEAD_DIM = 256
Q_DIM = NUM_ATTN_HEADS * HEAD_DIM

def cuda_sigmoid_gate(x_out, gate, dim):
    lib = get_lib()
    lib.cuda_sigmoid_gate(
        ctypes.c_void_p(x_out.data_ptr()),
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_int(dim),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return x_out

def test_sigmoid_gate():
    torch.manual_seed(42)
    context = torch.randn(Q_DIM, device='cuda')
    gate = torch.randn(Q_DIM, device='cuda')
    expected = context.cpu() * torch.sigmoid(gate.cpu())
    actual = cuda_sigmoid_gate(context, gate, Q_DIM)
    assert_close(actual, expected, msg="sigmoid_gate")
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_sigmoid_gate.py -v
```
Expected: FAIL.

- [ ] **Step 4: 实现 kernel**

```cuda
// ============================================================================
// Kernel: Sigmoid gate (in-place)
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
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_sigmoid_gate.py -v
```
Expected: 1 passed.

- [ ] **Step 6: 提交**

```bash
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_sigmoid_gate.py
git commit -m "feat(cuda_infer): add sigmoid gate kernel"
```

---

### Task 5: GPU KV Cache + infer.cu 集成

**Files:**
- Modify: `cuda_infer/infer.cu`

- [ ] **Step 1: 添加 GPU KV cache 结构**

替换 `KVCache` 的 CPU 分配为 GPU 分配。在 `infer.cu` 的 KVCache 定义处：

```cuda
typedef struct {
    float *d_k_cache;  // [MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] on GPU
    float *d_v_cache;  // [MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM] on GPU
    float *k_cache;    // CPU copy for backward compat during migration
    float *v_cache;
    int len;
} KVCache;
```

更新 `create_kv_cache`：

```cuda
static KVCache *create_kv_cache(void) {
    KVCache *kv = (KVCache *)calloc(1, sizeof(KVCache));
    size_t size = MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    CHECK_CUDA(cudaMalloc(&kv->d_k_cache, size));
    CHECK_CUDA(cudaMalloc(&kv->d_v_cache, size));
    kv->k_cache = (float *)calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    kv->v_cache = (float *)calloc(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
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
```

- [ ] **Step 2: 重写 forward_full_attention 为 GPU 版**

替换 `forward_full_attention_cpu` 为 `forward_full_attention_gpu`，输入/输出在 GPU，中间只用必要的 CPU 拷贝：

```cuda
static void forward_full_attention_gpu(
    const float *d_normed,    // [HIDDEN_DIM] on GPU
    float *d_attn_out,        // [HIDDEN_DIM] on GPU
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

    // Q/K/V projection weights are BF16, use gpu_bf16_matvec_cpu_io helper
    float *q_proj = (float *)malloc(q_proj_dim * sizeof(float));
    float *k_out = (float *)malloc(kv_dim * sizeof(float));
    float *v_out = (float *)malloc(kv_dim * sizeof(float));

    float *cpu_normed = (float *)malloc(HIDDEN_DIM * sizeof(float));
    CHECK_CUDA(cudaMemcpy(cpu_normed, d_normed, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));

    snprintf(tname, sizeof(tname), "layers.%d.self_attn.q_proj.weight", layer_idx);
    gpu_bf16_matvec_cpu_io(cpu_normed, q_proj, q_proj_dim, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.k_proj.weight", layer_idx);
    gpu_bf16_matvec_cpu_io(cpu_normed, k_out, kv_dim, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.v_proj.weight", layer_idx);
    gpu_bf16_matvec_cpu_io(cpu_normed, v_out, kv_dim, HIDDEN_DIM, wd, tname, stream);

    free(cpu_normed);

    // Split Q and q_gate
    float *q = (float *)malloc(q_dim * sizeof(float));
    float *q_gate = (float *)malloc(q_dim * sizeof(float));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        memcpy(q + h * HEAD_DIM, q_proj + h * 2 * HEAD_DIM, HEAD_DIM * sizeof(float));
        memcpy(q_gate + h * HEAD_DIM, q_proj + h * 2 * HEAD_DIM + HEAD_DIM,
               HEAD_DIM * sizeof(float));
    }
    free(q_proj);

    // Load per-head norm weights
    float *q_norm_w = (float *)malloc(HEAD_DIM * sizeof(float));
    float *k_norm_w = (float *)malloc(HEAD_DIM * sizeof(float));
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.q_norm.weight", layer_idx);
    load_tensor(wd, tname, q_norm_w);
    snprintf(tname, sizeof(tname), "layers.%d.self_attn.k_norm.weight", layer_idx);
    load_tensor(wd, tname, k_norm_w);

    // Upload Q, K, gate to GPU
    float *d_q, *d_k, *d_v, *d_q_gate;
    CHECK_CUDA(cudaMalloc(&d_q, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q_gate, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, kv_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, kv_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_q, q, q_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_gate, q_gate, q_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, k_out, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, v_out, kv_dim * sizeof(float), cudaMemcpyHostToDevice));
    free(q); free(q_gate); free(k_out); free(v_out);

    // Per-head Q/K RMS norm (GPU)
    float *d_norm_w;
    CHECK_CUDA(cudaMalloc(&d_norm_w, HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_norm_w, q_norm_w, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
    float *d_q_normed, *d_k_normed;
    CHECK_CUDA(cudaMalloc(&d_q_normed, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_normed, kv_dim * sizeof(float)));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        cuda_rms_norm(d_q + h * HEAD_DIM, d_norm_w, d_q_normed + h * HEAD_DIM,
                       HEAD_DIM, RMS_NORM_EPS, stream);
    }
    CHECK_CUDA(cudaMemcpy(d_norm_w, k_norm_w, HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice));
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        cuda_rms_norm(d_k + h * HEAD_DIM, d_norm_w, d_k_normed + h * HEAD_DIM,
                       HEAD_DIM, RMS_NORM_EPS, stream);
    }
    cudaFree(d_norm_w);
    free(q_norm_w); free(k_norm_w);

    // RoPE (GPU)
    float *d_q_rope, *d_k_rope;
    CHECK_CUDA(cudaMalloc(&d_q_rope, q_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k_rope, kv_dim * sizeof(float)));
    cuda_rope(d_q_normed, d_q_rope, NUM_ATTN_HEADS, HEAD_DIM, position, 10000000.0f, stream);
    cuda_rope(d_k_normed, d_k_rope, NUM_KV_HEADS, HEAD_DIM, position, 10000000.0f, stream);
    cudaFree(d_q_normed); cudaFree(d_k_normed);

    // Update KV cache on GPU
    int cache_pos = kv->len;
    CHECK_CUDA(cudaMemcpy(kv->d_k_cache + cache_pos * kv_dim, d_k_rope,
                          kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(kv->d_v_cache + cache_pos * kv_dim, d_v,
                          kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    kv->len++;
    int seq_len = kv->len;

    // Attention scores (GPU)
    float *d_scores;
    CHECK_CUDA(cudaMalloc(&d_scores, NUM_ATTN_HEADS * MAX_SEQ_LEN * sizeof(float)));
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    cuda_attn_scores(d_q_rope, kv->d_k_cache, d_scores, HEAD_DIM, kv_dim,
                      seq_len, MAX_SEQ_LEN, scale, NUM_ATTN_HEADS / NUM_KV_HEADS,
                      seq_len, stream);

    // Softmax (GPU)
    cuda_attn_softmax(d_scores, seq_len, MAX_SEQ_LEN, stream);

    // Values (GPU)
    float *d_context;
    CHECK_CUDA(cudaMalloc(&d_context, q_dim * sizeof(float)));
    cuda_attn_values(d_scores, kv->d_v_cache, d_context, HEAD_DIM, kv_dim,
                      seq_len, MAX_SEQ_LEN, NUM_ATTN_HEADS / NUM_KV_HEADS, stream);

    // Sigmoid gate (GPU)
    cuda_sigmoid_gate(d_context, d_q_gate, q_dim, stream);

    // O projection (GPU via BF16 matvec)
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
```

- [ ] **Step 3: 更新调用点**

在 `forward_layer_gpu` 中，将 `forward_full_attention_cpu` 调用替换为 `forward_full_attention_gpu`。函数修改为纯 GPU 路径，不再需要 CPU↔GPU 拷贝 attention 输入输出。

```cuda
    if (is_full_attention(layer_idx)) {
        int fa_idx = (layer_idx - 3) / 4;
        forward_full_attention_gpu(b->d_rms_out, b->d_output, wd, layer_idx,
                                    kv_caches[fa_idx], position, stream);
    }
```

删除 `forward_layer_gpu` 中 attention 前后的 `cudaMemcpy` 和 CPU buffer 分配。

- [ ] **Step 4: 编译和运行**

```bash
cd cuda_infer && make clean && make
./infer --prompt "Hello" --tokens 5 2>&1 | head -20
```
Expected: 编译成功，40 层跑完无 CUDA error。Full attention 层 hidden state 不出现 NaN。

- [ ] **Step 5: 验证全部测试**

```bash
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/ -v
```
Expected: 12 passed（8 旧 + 4 新）。

- [ ] **Step 6: 提交**

```bash
git add cuda_infer/infer.cu
git commit -m "feat(cuda_infer): full attention GPU — scores, softmax, values, sigmoid gate"
```

---

## Self-Review

**Spec Coverage:**
- [x] attn_scores kernel (Task 1)
- [x] attn_softmax kernel (Task 2)
- [x] attn_values kernel (Task 3)
- [x] sigmoid_gate kernel (Task 4)
- [x] GPU KV cache migration (Task 5)
- [x] infer.cu integration (Task 5)
- [x] PyTorch validation tests (Tasks 1-4)
- [x] Q/K per-head RMS norm (Task 5 — reuses cuda_rms_norm)

**Placeholder scan:** No TBD/TODO. All code blocks are concrete.

**Type consistency:** All kernel signatures match between kernels.h declarations and kernels.cu implementations. `heads_per_kv` = 8 (NUM_ATTN_HEADS / NUM_KV_HEADS) used consistently. MAX_SEQ_LEN=8192 from existing code.

**Gap:** The `gpu_bf16_matvec_cpu_io` helper (added in previous Task 10) is referenced in Task 5. It should already exist in infer.cu. If not, Task 5 implementer will need to add it.
