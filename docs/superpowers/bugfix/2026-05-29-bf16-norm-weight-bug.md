# BF16 Norm Weight Loading Bug

**Date:** 2026-05-29
**Branch:** cuda
**Status:** Root cause identified, fix pending

## Symptom

```
./infer --prompt "Hello" --tokens 20
```

输出全部为同一字符 `!`，debug 显示 hidden state 从 Layer 3（第一个 Full Attention 层）开始全部为 NaN。

## Root Cause (最终确认: 3 个 bugs)

### Bug 1: `tensor_index.bin` 不包含 `shape` 字段

`load_tensor_bf16_to_f32()` 使用 `t->shape[0]` 计算元素数量，但 `tensor_index.bin` 只存储 `offset` 和 `size`（不存储 shape）。`shape[0]` 始终为 0（calloc 初始值），BF16→float32 转换循环从未执行。目标 buffer（malloc 未初始化）保留垃圾值。

**修复**: `n = (int)(t->size / sizeof(uint16_t))` 从文件大小反推元素数量。

### Bug 2: `linear_attn.norm.weight` 是 float32 不是 BF16

该 tensor 存储为 float32（512 bytes = 128 floats），但被错误地用 `load_tensor_bf16_to_f32` 加载，导致 `n = 512/2 = 256` 写入 128 元素的 buffer，堆溢出。

**修复**: 用 `load_tensor()` 替代 `load_tensor_bf16_to_f32()`。

### Bug 3: Stream race condition

`load_tensor_to_gpu()` 使用同步 `cudaMemcpy(HostToDevice)`（default stream），与 kernel launch 的 `stream` 产生竞争。scratch buffer 在被 kernel 读取时被后续的 cudaMemcpy 覆盖。

**修复**: `cudaMemcpyAsync(HostToDevice, stream)` 替代 `cudaMemcpy(HostToDevice)`。

### 具体机制

以 `input_layernorm.weight` 为例：

| 项目 | 值 |
|------|-----|
| Tensor shape | [2048] |
| Storage format | BF16 (2 bytes/element) |
| File size | 4096 bytes |
| `float*` buffer (calloc) | 8192 bytes |
| `memcpy(dest, src, 4096)` | 只写入了前半 buffer |

写入了 4096 字节的 BF16 数据到 8192 字节的 float buffer：

```
buffer bytes: [B0, B1, B2, B3, ..., B4094, B4095, 0, 0, ..., 0]
              └── float[0] ──┘└── float[1] ──┘         └─ 后半全零 ─┘
```

每个 float32 由两个 BF16 值的 uint16 拼接而成，产生无意义的浮点数。后 1024 个 float 保持 calloc 的零值。

### 受影响的 Weight

所有以 `.weight` 结尾的 norm tensor 都是 BF16：

| Tensor | Shape | File Size | Affect |
|--------|-------|-----------|--------|
| `input_layernorm.weight` | [2048] | 4096 | 每层 |
| `post_attention_layernorm.weight` | [2048] | 4096 | 每层 |
| `final_layer_norm` | [2048] | 4096 | 全局 |
| `self_attn.q_norm.weight` | [256] | 512 | Full Attn 层 |
| `self_attn.k_norm.weight` | [256] | 512 | Full Attn 层 |
| `linear_attn.norm.weight` | [128] | 256 | Linear Attn 层 |

## Debugging Process

### Phase 1: Symptom identification

输出全是 `!`，加 debug 输出发现 logits 全部 NaN，hidden state rms=NaN。

### Phase 2: Layer-level NaN source

在 40 层循环中逐层检查 hidden state：

```
L0 rms=0.0128 nan=0
L1 rms=0.0128 nan=0
L2 rms=0.0128 nan=0
L3 rms=nan nan=2048  ← NaN starts here
```

Layer 3 是第一个 Full Attention 层（layers 0,1,2 是 Linear Attention）。

### Phase 3: Full Attention step-level tracing

在 `forward_full_attention_gpu` 中逐步检查中间值：

```
q_proj rms=0.1052 nan=0     ← Q projection clean
v_out  rms=0.0772 nan=0     ← V projection clean
k_out  rms=0.0866 nan=0     ← K projection clean
q_cpu  rms=0.0867 nan=0     ← Split Q clean
q_gate rms=0.1208 nan=0     ← Gate clean
q_normed rms=0.2290 nan=0   ← Per-head norm clean
q_rope rms=0.2290 nan=0     ← RoPE clean
V_cache[0] rms=0.0772 nan=0 ← KV cache clean
scores_h0 rms=inf nan=0     ← Scores are INF
softmax_h0 rms=0.2500 nan=0 ← Softmax clean (seq_len=1)
ctx_raw rms=nan nan=1024    ← Context NaN!
```

Q、K、V 投影和 RoPE 全部 clean。Scores INF 来自 GPU buffer 未初始化位置。但 `ctx_raw` (attn_values kernel 输出) 有 NaN。

### Phase 4: Bypass tests

**Full attention bypass**（return d_normed as output）→ 40 层全部 clean，证实 NaN 来自 attention compute。

**CPU manual context**（用 CPU 做 GQA expansion 替代 scores/softmax/values/gate）→ 发现 RMS norm 输入就有问题。

### Phase 5: Compare with model_weights.json

检查 `model_weights.json` 发现所有 norm 权重的 size 与 shape 不匹配 float32：

```json
"layers.0.input_layernorm.weight": {"offset": ..., "size": 4096, "shape": [2048]}
```

2048 floats × 4 bytes = 8192 bytes，但实际 size 只有 4096 bytes → 确认是 BF16。

### Phase 6: Root cause confirmation

`load_tensor()` 函数无类型感知，直接 `memcpy(dest, src, t->size)`：

```c
static void load_tensor(WeightData *wd, const char *name, void *dest) {
    int idx = find_tensor(wd, name);
    TensorInfo *t = &wd->tensors[idx];
    size_t data_start = 4 + wd->header_size;
    size_t data_start_aligned = (data_start + 63) & ~63ULL;
    void *src = (uint8_t *)wd->base + (t->offset - data_start_aligned);
    memcpy(dest, src, t->size);  // BUG: no BF16→float32 conversion
}
```

### Phase 7: Additional findings

**Stream synchronization bug**: `cudaMemcpy DeviceToDevice`（无 stream 参数）在 per-thread default stream 上与 kernel（`stream`）产生竞争。

- `gpu_bf16_matvec_cpu_io`: kernel launch → `cudaMemcpy DeviceToHost` 无 sync
- `forward_full_attention_gpu`: `cudaMemcpy DeviceToDevice` for KV cache 无 sync

修复：添加 `cudaStreamSynchronize(stream)` 或使用 `cudaMemcpyAsync`。

## Fix Plan (已实施)

### Fix 1: `n = t->size / sizeof(uint16_t)` (commit `8e64abd`)

替代 `n = t->shape[0]`。

### Fix 2: `linear_attn.norm.weight` 用 `load_tensor()` (commit `8e64abd`)

### Fix 3: `cudaMemcpyAsync(..., stream)` (commit `dff1cdd`)

## Results (after all fixes)

```
./infer --prompt "Hello" --tokens 10
```

输出真实文本 "gettiarraarraarre顾名思..."，40 层无 NaN/INF：
```
L0 rms=0.0137 → L39 rms=0.1795 (reasonable growth)
max_logit=8.04 (reasonable)
```

## Lessons Learned

1. **JSON manifest 中 size 和 shape 不匹配时，应检查 dtype**。2048×float32=8192≠4096，立即暴露 BF16 格式。
2. **逐层/逐步骤 NaN 追踪**是定位深度学习 pipeline 问题的最有效方法。
3. **compute-sanitizer 确认无内存错误**后，NaN 必为数值问题，应从数据格式入手。
4. **Bypass 测试**（跳过特定子模块）快速缩小问题范围。
5. **load_tensor 应该是类型感知的**，避免字节级别的格式错误。
