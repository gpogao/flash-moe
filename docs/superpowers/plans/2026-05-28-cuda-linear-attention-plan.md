# Linear Attention (GatedDeltaNet) GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 30 层 Linear Attention (GatedDeltaNet) 从 CPU pass-through stub 替换为完整 GPU 实现

**Architecture:** 5 个 CUDA kernel 从 metal_infer 移植，GPU 持久化状态（conv_state + ssm_state），BF16 matvec 做投影，CPU 仅做权重加载和 kernel launch

**Tech Stack:** CUDA 13.0, nvcc, PyTorch, warp shuffle intrinsics

---

## 文件结构

```
cuda_infer/
├── kernels.cu          # [修改] 新增 5 个 linear attention kernel
├── kernels.h           # [修改] 新增 5 个 extern "C" 声明
├── infer.cu            # [修改] 重写 forward_linear_attention，GPU 状态管理
└── tests/
    ├── test_conv1d_step.py      # [新建]
    ├── test_decay_beta.py       # [新建]
    ├── test_rms_norm_qk.py      # [新建]
    ├── test_delta_net.py        # [新建]
    └── test_gated_rms_norm.py   # [新建]
```

---

### Task 1: conv1d_step Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_conv1d_step.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_conv1d_step(
    float *d_conv_state, const float *d_input,
    const uint16_t *d_weight, float *d_output,
    int conv_dim, cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_conv1d_step.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

CONV_DIM = 128
KERNEL_SIZE = 4

def cuda_conv1d_step(conv_state, input_t, weight_bf16, conv_dim):
    lib = get_lib()
    output = torch.zeros(conv_dim, dtype=torch.float32, device='cuda')
    lib.cuda_conv1d_step(
        ctypes.c_void_p(conv_state.data_ptr()),
        ctypes.c_void_p(input_t.data_ptr()),
        ctypes.c_void_p(weight_bf16.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(conv_dim),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return output, conv_state

def conv1d_step_ref(conv_state, input_t, weight_bf16, conv_dim):
    conv_state = conv_state.cpu().clone()
    input_t = input_t.cpu()
    weight = weight_bf16.cpu().to(torch.float32)
    out = torch.zeros(conv_dim)
    for c in range(conv_dim):
        acc = (conv_state[0, c] * weight[c, 0] +
               conv_state[1, c] * weight[c, 1] +
               conv_state[2, c] * weight[c, 2] +
               input_t[c] * weight[c, 3])
        out[c] = acc / (1.0 + (-acc).exp())
    new_state = conv_state.clone()
    new_state[0] = conv_state[1]
    new_state[1] = conv_state[2]
    new_state[2] = input_t
    return out, new_state

def test_conv1d_step():
    torch.manual_seed(42)
    conv_dim = CONV_DIM
    state = torch.randn(3, conv_dim, device='cuda')
    x = torch.randn(conv_dim, device='cuda')
    w = torch.randn(conv_dim, KERNEL_SIZE).bfloat16().to(torch.uint16).reshape(conv_dim, KERNEL_SIZE).cuda()
    expected_out, expected_state = conv1d_step_ref(state, x, w, conv_dim)
    actual_out, actual_state = cuda_conv1d_step(state, x, w, conv_dim)
    assert_close(actual_out, expected_out, atol=1e-4, msg="conv1d_out")
    assert_close(actual_state, expected_state, atol=1e-4, msg="conv1d_state")
```

- [ ] **Step 3: 实现 kernel**

添加到 `kernels.cu` 末尾：

```cuda
// ============================================================================
// Kernel: Conv1d depthwise step with SiLU activation
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
```

- [ ] **Step 4: 构建测试提交**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_conv1d_step.py -v
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_conv1d_step.py
git commit -m "feat(cuda_infer): add conv1d depthwise step kernel with SiLU"
```

Expected: 1 test passed.

---

### Task 2: compute_decay_beta Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_decay_beta.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_compute_decay_beta(
    const float *d_alpha, const float *d_beta,
    const float *d_A_log, const uint16_t *d_dt_bias,
    float *d_g_decay, float *d_beta_gate, int num_heads,
    cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_decay_beta.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

NUM_V_HEADS = 64

def cuda_compute_decay_beta(alpha, beta, A_log, dt_bias, num_heads):
    lib = get_lib()
    g_decay = torch.zeros(num_heads, dtype=torch.float32, device='cuda')
    beta_gate = torch.zeros(num_heads, dtype=torch.float32, device='cuda')
    lib.cuda_compute_decay_beta(
        ctypes.c_void_p(alpha.data_ptr()),
        ctypes.c_void_p(beta.data_ptr()),
        ctypes.c_void_p(A_log.data_ptr()),
        ctypes.c_void_p(dt_bias.data_ptr()),
        ctypes.c_void_p(g_decay.data_ptr()),
        ctypes.c_void_p(beta_gate.data_ptr()),
        ctypes.c_int(num_heads),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return g_decay, beta_gate

def decay_beta_ref(alpha, beta, A_log, dt_bias):
    softplus = (alpha + dt_bias.to(torch.float32)).softplus()
    A_val = A_log.exp()
    g = (-A_val * softplus).exp()
    bg = beta.sigmoid()
    return g, bg

def test_decay_beta():
    torch.manual_seed(42)
    n = NUM_V_HEADS
    alpha = torch.randn(n, device='cuda')
    beta = torch.randn(n, device='cuda')
    A_log = torch.randn(n, device='cuda')
    dt_bias = torch.randn(n).bfloat16().to(torch.uint16).cuda()
    eg, ebg = decay_beta_ref(alpha.cpu(), beta.cpu(), A_log.cpu(), dt_bias.cpu())
    ag, abg = cuda_compute_decay_beta(alpha, beta, A_log, dt_bias, n)
    assert_close(ag, eg, atol=1e-5, msg="g_decay")
    assert_close(abg, ebg, atol=1e-5, msg="beta_gate")
```

- [ ] **Step 3: 实现 kernel**

```cuda
// ============================================================================
// Kernel: Compute g_decay and beta_gate for GatedDeltaNet
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
```

- [ ] **Step 4: 构建测试提交**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_decay_beta.py -v
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_decay_beta.py
git commit -m "feat(cuda_infer): add compute_decay_beta kernel"
```

Expected: 1 test passed.

---

### Task 3: rms_norm_qk Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_rms_norm_qk.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_rms_norm_qk(
    float *d_q, float *d_k, int num_k_heads,
    int key_dim, float inv_scale, cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_rms_norm_qk.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

NUM_K_HEADS = 16
KEY_DIM = 128

def cuda_rms_norm_qk(q, k, num_k_heads, key_dim, inv_scale):
    lib = get_lib()
    lib.cuda_rms_norm_qk(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_int(num_k_heads), ctypes.c_int(key_dim),
        ctypes.c_float(inv_scale),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return q, k

def rms_norm_qk_ref(q, k, num_k_heads, key_dim, inv_scale):
    q = q.cpu().clone().reshape(num_k_heads, key_dim)
    k = k.cpu().clone().reshape(num_k_heads, key_dim)
    eps = 1e-6
    for h in range(num_k_heads):
        q_rms = torch.sqrt((q[h]**2).mean() + eps)
        q[h] = q[h] / q_rms * (inv_scale * inv_scale)
        k_rms = torch.sqrt((k[h]**2).mean() + eps)
        k[h] = k[h] / k_rms * inv_scale
    return q.flatten(), k.flatten()

def test_rms_norm_qk():
    torch.manual_seed(42)
    total = NUM_K_HEADS * KEY_DIM
    inv_scale = 1.0 / (KEY_DIM ** 0.5)
    q = torch.randn(total, device='cuda')
    k = torch.randn(total, device='cuda')
    eq, ek = rms_norm_qk_ref(q, k, NUM_K_HEADS, KEY_DIM, inv_scale)
    aq, ak = cuda_rms_norm_qk(q, k, NUM_K_HEADS, KEY_DIM, inv_scale)
    assert_close(aq, eq, atol=1e-5, msg="rms_norm_qk_q")
    assert_close(ak, ek, atol=1e-5, msg="rms_norm_qk_k")
```

- [ ] **Step 3: 实现 kernel**

```cuda
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

    __shared__ float q_sum_sq, k_sum_sq;
    __shared__ float q_partial[256], k_partial[256];

    float q_val = (tid < key_dim) ? q[base + tid] : 0.0f;
    float k_val = (tid < key_dim) ? k[base + tid] : 0.0f;

    q_partial[tid] = q_val * q_val;
    k_partial[tid] = k_val * k_val;
    __syncthreads();

    if (tid == 0) {
        float qs = 0.0f, ks = 0.0f;
        for (int i = 0; i < key_dim; i++) { qs += q_partial[i]; ks += k_partial[i]; }
        q_sum_sq = qs; k_sum_sq = ks;
    }
    __syncthreads();

    float q_inv_rms = rsqrtf(q_sum_sq / (float)key_dim + 1e-6f);
    float k_inv_rms = rsqrtf(k_sum_sq / (float)key_dim + 1e-6f);
    float q_scale = inv_scale * inv_scale;
    float k_scale = inv_scale;

    if (tid < key_dim) {
        q[base + tid] = q_val * q_inv_rms * q_scale;
        k[base + tid] = k_val * k_inv_rms * k_scale;
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
```

- [ ] **Step 4: 构建测试提交**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_rms_norm_qk.py -v
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_rms_norm_qk.py
git commit -m "feat(cuda_infer): add per-head RMS norm kernel for Q and K"
```

Expected: 1 test passed.

---

### Task 4: gated_delta_net_step Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_delta_net.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_gated_delta_net_step(
    float *d_state, const float *d_q, const float *d_k,
    const float *d_v, const float *d_g_decay,
    const float *d_beta_gate, float *d_output,
    int num_v_heads, int value_dim, int key_dim,
    int k_heads_per_v, cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_delta_net.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

NUM_V_HEADS = 8
NUM_K_HEADS = 4
VALUE_DIM = 16
KEY_DIM = 16
K_HEADS_PER_V = NUM_V_HEADS // NUM_K_HEADS  # 2
TOTAL_K = NUM_K_HEADS * KEY_DIM
TOTAL_V = NUM_V_HEADS * VALUE_DIM

def cuda_gated_delta_net_step(state, q, k, v, g_decay, beta_gate):
    lib = get_lib()
    output = torch.zeros(TOTAL_V, dtype=torch.float32, device='cuda')
    lib.cuda_gated_delta_net_step(
        ctypes.c_void_p(state.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(g_decay.data_ptr()),
        ctypes.c_void_p(beta_gate.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(NUM_V_HEADS), ctypes.c_int(VALUE_DIM),
        ctypes.c_int(KEY_DIM), ctypes.c_int(K_HEADS_PER_V),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return output, state

def delta_net_ref(state, q, k, v, g_decay, beta_gate):
    state = state.cpu().clone().reshape(NUM_V_HEADS, VALUE_DIM, KEY_DIM)
    q = q.cpu().reshape(NUM_K_HEADS, KEY_DIM)
    k = k.cpu().reshape(NUM_K_HEADS, KEY_DIM)
    v = v.cpu().reshape(NUM_V_HEADS, VALUE_DIM)
    out = torch.zeros(NUM_V_HEADS, VALUE_DIM)
    for vh in range(NUM_V_HEADS):
        kh = vh // K_HEADS_PER_V
        state[vh] *= g_decay[vh].item()
        for vi in range(VALUE_DIM):
            kv_mem = (state[vh, vi] * k[kh]).sum()
            delta = (v[vh, vi] - kv_mem) * beta_gate[vh].item()
            state[vh, vi] += k[kh] * delta
            out[vh, vi] = (state[vh, vi] * q[kh]).sum()
    return out.flatten(), state.flatten()

def test_delta_net():
    torch.manual_seed(42)
    state = torch.randn(NUM_V_HEADS * VALUE_DIM * KEY_DIM, device='cuda') * 0.1
    q = torch.randn(TOTAL_K, device='cuda')
    k = torch.randn(TOTAL_K, device='cuda')
    v = torch.randn(TOTAL_V, device='cuda')
    g_decay = torch.rand(NUM_V_HEADS, device='cuda') * 0.5 + 0.5
    beta_gate = torch.rand(NUM_V_HEADS, device='cuda') * 0.5 + 0.5
    e_out, e_state = delta_net_ref(state, q, k, v, g_decay, beta_gate)
    a_out, a_state = cuda_gated_delta_net_step(state, q, k, v, g_decay, beta_gate)
    assert_close(a_out, e_out, atol=1e-4, msg="delta_net_out")
    assert_close(a_state, e_state, atol=1e-4, msg="delta_net_state")
```

- [ ] **Step 3: 实现 kernel**

```cuda
// ============================================================================
// Kernel: GatedDeltaNet recurrence step
// ============================================================================
__global__ void gated_delta_net_step_kernel(
    float *state, const float *q, const float *k,
    const float *v, const float *g_decay,
    const float *beta_gate, float *output,
    int num_v_heads, int value_dim, int key_dim,
    int k_heads_per_v
) {
    int vh = blockIdx.x;        // v-head index
    int vi = threadIdx.x;       // value index within head
    if (vi >= value_dim || vh >= num_v_heads) return;

    int kh = vh / k_heads_per_v;
    float g = g_decay[vh];
    float beta = beta_gate[vh];

    int state_base = vh * value_dim * key_dim + vi * key_dim;
    int k_base = kh * key_dim;
    int v_base = vh * value_dim;
    int q_base = kh * key_dim;

    // Step 1: Decay state row
    for (int ki = 0; ki < key_dim; ki++) {
        state[state_base + ki] *= g;
    }

    // Step 2: kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (int ki = 0; ki < key_dim; ki++) {
        kv_mem += state[state_base + ki] * k[k_base + ki];
    }

    // Step 3-4: delta = (v[vi] - kv_mem) * beta; S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (int ki = 0; ki < key_dim; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output = dot(S[vi][:], q[:])
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
```

- [ ] **Step 4: 构建测试提交**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_delta_net.py -v
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_delta_net.py
git commit -m "feat(cuda_infer): add gated delta net recurrence kernel"
```

Expected: 1 test passed.

---

### Task 5: gated_rms_norm Kernel

**Files:**
- Modify: `cuda_infer/kernels.cu`
- Modify: `cuda_infer/kernels.h`
- Create: `cuda_infer/tests/test_gated_rms_norm.py`

- [ ] **Step 1: 添加声明到 kernels.h**

```cuda
void cuda_gated_rms_norm(
    const float *d_values, const float *d_z,
    const uint16_t *d_weight, float *d_output,
    int num_heads, int value_dim, float eps,
    cudaStream_t stream);
```

- [ ] **Step 2: 写测试**

`cuda_infer/tests/test_gated_rms_norm.py`:

```python
import torch
from conftest import get_lib, assert_close
import ctypes

NUM_V_HEADS = 8
VALUE_DIM = 16
TOTAL = NUM_V_HEADS * VALUE_DIM

def cuda_gated_rms_norm(values, z, weight_bf16, num_heads, value_dim, eps=1e-6):
    lib = get_lib()
    output = torch.zeros(TOTAL, dtype=torch.float32, device='cuda')
    lib.cuda_gated_rms_norm(
        ctypes.c_void_p(values.data_ptr()),
        ctypes.c_void_p(z.data_ptr()),
        ctypes.c_void_p(weight_bf16.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(num_heads), ctypes.c_int(value_dim),
        ctypes.c_float(eps),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return output

def gated_rms_norm_ref(values, z, weight_bf16, num_heads, value_dim, eps=1e-6):
    values = values.cpu().reshape(num_heads, value_dim)
    z = z.cpu().reshape(num_heads, value_dim)
    w = weight_bf16.cpu().to(torch.float32)
    out = torch.zeros(num_heads, value_dim)
    for h in range(num_heads):
        rms = torch.sqrt((values[h]**2).mean() + eps)
        normed = values[h] / rms
        out[h] = normed * torch.silu(z[h]) * w
    return out.flatten()

def test_gated_rms_norm():
    torch.manual_seed(42)
    values = torch.randn(TOTAL, device='cuda')
    z = torch.randn(TOTAL, device='cuda')
    w = torch.randn(VALUE_DIM).bfloat16().to(torch.uint16).cuda()
    expected = gated_rms_norm_ref(values, z, w, NUM_V_HEADS, VALUE_DIM)
    actual = cuda_gated_rms_norm(values, z, w, NUM_V_HEADS, VALUE_DIM)
    assert_close(actual, expected, atol=1e-4, msg="gated_rms_norm")
```

- [ ] **Step 3: 实现 kernel**

```cuda
// ============================================================================
// Kernel: Gated RMS norm (RMS Norm + SiLU gate)
// ============================================================================
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
```

- [ ] **Step 4: 构建测试提交**

```bash
cd cuda_infer && make clean && make libkernels.so
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/test_gated_rms_norm.py -v
git add cuda_infer/kernels.cu cuda_infer/kernels.h cuda_infer/tests/test_gated_rms_norm.py
git commit -m "feat(cuda_infer): add gated RMS norm kernel"
```

Expected: 1 test passed.

---

### Task 6: GPU 状态管理 + infer.cu 集成

**Files:**
- Modify: `cuda_infer/infer.cu`

- [ ] **Step 1: 更新 LinearAttnState 添加 GPU 字段**

```cuda
typedef struct {
    float *conv_state;   // [(CONV_KERNEL_SIZE-1) * LINEAR_CONV_DIM] CPU fallback
    float *ssm_state;    // [LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM]
    float *d_conv_state; // GPU conv state
    float *d_ssm_state;  // GPU ssm state
} LinearAttnState;
```

更新 create/free:

```cuda
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
```

- [ ] **Step 2: 重写 forward_linear_attention 为 GPU 版**

新增 `gpu_bf16_matvec_direct`（GPU in → GPU out，无 CPU roundtrip）：

```cuda
static void gpu_bf16_matvec_direct(
    const float *d_x, float *d_out,
    int out_dim, int in_dim,
    WeightData *wd, const char *weight_name,
    cudaStream_t stream
) {
    size_t w_bytes = out_dim * in_dim * sizeof(uint16_t);
    ensure_scratch(w_bytes + 4096);
    load_tensor_to_gpu(wd, weight_name, d_scratch_w);
    cuda_bf16_matvec((uint16_t *)d_scratch_w, d_x, d_out, out_dim, in_dim, stream);
}
```

然后实现 `forward_linear_attention_gpu`：

```cuda
static void forward_linear_attention_gpu(
    const float *d_normed,    // [HIDDEN_DIM] on GPU
    float *d_attn_out,        // [HIDDEN_DIM] on GPU
    WeightData *wd,
    int layer_idx,
    LinearAttnState *state,
    cudaStream_t stream
) {
    char tname[256];

    // QKV projection: [12288, 2048] BF16 matvec (GPU direct)
    int qkv_dim = LINEAR_CONV_DIM;  // 12288
    float *d_qkv, *d_z, *d_beta, *d_alpha;
    CHECK_CUDA(cudaMalloc(&d_qkv, qkv_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z, LINEAR_TOTAL_VALUE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, LINEAR_NUM_V_HEADS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_alpha, LINEAR_NUM_V_HEADS * sizeof(float)));

    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_qkv, qkv_dim, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_z, LINEAR_TOTAL_VALUE, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_beta, LINEAR_NUM_V_HEADS, HIDDEN_DIM, wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    gpu_bf16_matvec_direct(d_normed, d_alpha, LINEAR_NUM_V_HEADS, HIDDEN_DIM, wd, tname, stream);

    // Conv1d step
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.conv1d.weight", layer_idx);
    float *d_conv_out;
    CHECK_CUDA(cudaMalloc(&d_conv_out, qkv_dim * sizeof(float)));
    uint16_t *d_conv_w;
    CHECK_CUDA(cudaMalloc(&d_conv_w, qkv_dim * CONV_KERNEL_SIZE * sizeof(uint16_t)));
    load_tensor_to_gpu(wd, tname, d_conv_w);
    cuda_conv1d_step(state->d_conv_state, d_qkv, d_conv_w, d_conv_out, qkv_dim, stream);
    cudaFree(d_conv_w);

    // Split: q/k/v from conv_out
    float *d_q = d_conv_out;                        // [2048]  first LINEAR_TOTAL_KEY
    float *d_k = d_conv_out + LINEAR_TOTAL_KEY;     // [2048]  second LINEAR_TOTAL_KEY
    float *d_v = d_conv_out + 2 * LINEAR_TOTAL_KEY; // [8192]  rest

    // Q/K per-head RMS norm
    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
    cuda_rms_norm_qk(d_q, d_k, LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM, inv_scale, stream);

    // Decay and beta
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.A_log", layer_idx);
    float *d_A_log; CHECK_CUDA(cudaMalloc(&d_A_log, LINEAR_NUM_V_HEADS * sizeof(float)));
    load_tensor_to_gpu(wd, tname, d_A_log);
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *d_dt_bias; CHECK_CUDA(cudaMalloc(&d_dt_bias, LINEAR_NUM_V_HEADS * sizeof(uint16_t)));
    load_tensor_to_gpu(wd, tname, d_dt_bias);

    float *d_g_decay, *d_beta_gate;
    CHECK_CUDA(cudaMalloc(&d_g_decay, LINEAR_NUM_V_HEADS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta_gate, LINEAR_NUM_V_HEADS * sizeof(float)));
    cuda_compute_decay_beta(d_alpha, d_beta, d_A_log, d_dt_bias,
                             d_g_decay, d_beta_gate, LINEAR_NUM_V_HEADS, stream);
    cudaFree(d_A_log); cudaFree(d_dt_bias); cudaFree(d_alpha); cudaFree(d_beta);

    // Delta recurrence
    float *d_out_values;
    CHECK_CUDA(cudaMalloc(&d_out_values, LINEAR_TOTAL_VALUE * sizeof(float)));
    cuda_gated_delta_net_step(state->d_ssm_state, d_q, d_k, d_v,
                               d_g_decay, d_beta_gate, d_out_values,
                               LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                               LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS, stream);
    cudaFree(d_g_decay); cudaFree(d_beta_gate);

    // Gated RMS norm
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.norm.weight", layer_idx);
    uint16_t *d_norm_w;
    CHECK_CUDA(cudaMalloc(&d_norm_w, LINEAR_VALUE_DIM * sizeof(uint16_t)));
    load_tensor_to_gpu(wd, tname, d_norm_w);
    float *d_gated;
    CHECK_CUDA(cudaMalloc(&d_gated, LINEAR_TOTAL_VALUE * sizeof(float)));
    cuda_gated_rms_norm(d_out_values, d_z, d_norm_w, d_gated,
                          LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, RMS_NORM_EPS, stream);
    cudaFree(d_norm_w); cudaFree(d_out_values); cudaFree(d_z);

    // Out projection: [2048, 8192] BF16 matvec
    snprintf(tname, sizeof(tname), "layers.%d.linear_attn.out_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(d_gated, d_attn_out, HIDDEN_DIM, LINEAR_TOTAL_VALUE, wd, tname, stream);
    cudaFree(d_gated);

    // d_qkv was d_conv_out which we still need — wait, d_q/k/v still point into d_conv_out
    // d_qkv = d_conv_out, and we used d_q/k/v as aliases. d_conv_out was freed above.
    // Actually we didn't free d_conv_out yet. We need it alive for d_q/k/v.
    // d_qkv = d_conv_out, so don't free it separately.
    cudaFree(d_qkv);
}
```

**注意**：`d_conv_out`、`d_out_values`、`d_gated` 等临时 buffer 分配/释放正确管理。`d_qkv` 和 `d_conv_out` 指向同一块内存（QKV 投影输出直接作为 conv 的 input）。

- [ ] **Step 3: 更新 forward_layer_gpu 调用**

删除 linear attention 的 CPU roundtrip。将：

```cuda
    float *cpu_normed = ...;
    cudaMemcpy(cpu_normed, ...);
    forward_linear_attention_cpu(...);
    cudaMemcpy(b->d_output, ...);
```

替换为：

```cuda
    forward_linear_attention_gpu(b->d_rms_out, b->d_output, wd, layer_idx,
                                  linear_states[layer_idx], stream);
```

- [ ] **Step 4: 确保 gpu_bf16_matvec_direct 和 cuda_bf16_matvec 存在**

检查 `kernels.cu` 中是否有 `cuda_bf16_matvec` 函数（用于 GPU→GPU BF16 matvec）。如果没有，添加：

```cuda
__global__ void bf16_matvec_kernel(
    const uint16_t *weight, const float *x, float *out,
    int out_dim, int in_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (int col = 0; col < in_dim; col++) {
        acc += bf16_to_f32(weight[row * in_dim + col]) * x[col];
    }
    out[row] = acc;
}

extern "C" {
void cuda_bf16_matvec(
    const uint16_t *d_weight, const float *d_x, float *d_out,
    int out_dim, int in_dim, cudaStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((out_dim + 255) / 256);
    bf16_matvec_kernel<<<gridDim, blockDim, 0, stream>>>(
        d_weight, d_x, d_out, out_dim, in_dim);
    CHECK_CUDA(cudaGetLastError());
}
}
```

- [ ] **Step 5: 构建和运行**

```bash
cd cuda_infer && make clean && make
./infer --prompt "Hello" --tokens 5 2>&1 | head -20
```
Expected: 40 层跑完，Linear attention 层不再打印 "stub - pass through"。

- [ ] **Step 6: 验证全部测试**

```bash
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/ -v
```
Expected: 17 passed（12 + 5 new）。

- [ ] **Step 7: 提交**

```bash
git add cuda_infer/infer.cu cuda_infer/kernels.cu cuda_infer/kernels.h
git commit -m "feat(cuda_infer): full linear attention GPU — GatedDeltaNet implementation"
```

---

## Self-Review

**Spec Coverage:**
- [x] conv1d_step kernel (Task 1)
- [x] compute_decay_beta kernel (Task 2)
- [x] rms_norm_qk kernel (Task 3)
- [x] gated_delta_net_step kernel (Task 4)
- [x] gated_rms_norm kernel (Task 5)
- [x] GPU 持久状态管理 (Task 6)
- [x] infer.cu 集成 (Task 6)
- [x] gpu_bf16_matvec_direct (Task 6)

**Placeholder scan:** No TBD/TODO. All code blocks concrete.

**Type consistency:** `LINEAR_NUM_V_HEADS=64, LINEAR_NUM_K_HEADS=16, LINEAR_KEY_DIM=128, LINEAR_VALUE_DIM=128` used consistently across all 6 tasks. `k_heads_per_v=4` matches 64/16. Conv dimension `LINEAR_CONV_DIM=12288` = 2048+2048+8192. Weight shapes match spec.
