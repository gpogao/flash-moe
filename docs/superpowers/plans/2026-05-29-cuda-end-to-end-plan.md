# End-to-End Inference: Embedding, lm_head, Decode Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 token embedding、final RMS norm + lm_head、自回归 decode loop，打通端到端文本生成

**Architecture:** 全部复用已有 kernel（bf16 matvec、rms_norm）。CPU 做 embedding 查表 + BPE decode + 控制循环

**Tech Stack:** BF16 matvec, RMS norm, mmap weight loading, BPE tokenizer

---

### Task 1: Embedding + Final Norm + lm_head + Decode Loop

**Files:**
- Modify: `cuda_infer/infer.cu`

- [ ] **Step 1: 实现 token embedding**

替换 `main()` 中的占位 embedding 初始化。当前代码用 `(input_ids[i] % 256) / 128 - 1`。embed_tokens 在 mmap 中 offset=0，BF16 格式。

```c
// Load embed_tokens weight to GPU
uint16_t *d_embed;
CHECK_CUDA(cudaMalloc(&d_embed, HIDDEN_DIM * sizeof(uint16_t)));

// Read embedding for the LAST token only (single-token decode)
uint32_t token_id = input_ids[num_tokens - 1];
size_t embed_offset = (size_t)token_id * HIDDEN_DIM * sizeof(uint16_t);
// embed_tokens is at offset 0 in model_weights.bin
void *embed_src = (uint8_t *)mf.data + ((4 + wd.header_size + 63) & ~63ULL) + embed_offset;
CHECK_CUDA(cudaMemcpy(d_embed, embed_src, HIDDEN_DIM * sizeof(uint16_t),
                      cudaMemcpyHostToDevice));

// Convert BF16 → float32 on GPU via bf16_matvec with identity
// Actually, just load the BF16 embedding into d_hidden and use rms_norm
// Simpler: load BF16 to CPU, convert, upload
float *cpu_embed = (float *)malloc(HIDDEN_DIM * sizeof(float));
uint16_t *embed_bf16 = (uint16_t *)embed_src;
for (int i = 0; i < HIDDEN_DIM; i++) {
    cpu_embed[i] = bf16_to_f32(embed_bf16[i]);
}
CHECK_CUDA(cudaMemcpy(d_hidden, cpu_embed, HIDDEN_DIM * sizeof(float),
                      cudaMemcpyHostToDevice));
cudaFree(d_embed);
free(cpu_embed);
```

- [ ] **Step 2: 实现 final RMS norm + lm_head**

在 40 层循环之后，添加：

```c
// Final RMS norm: final_layer_norm.weight
float *d_final_norm_w;
CHECK_CUDA(cudaMalloc(&d_final_norm_w, HIDDEN_DIM * sizeof(float)));
load_tensor_to_gpu(&wd, "final_layer_norm", d_final_norm_w);
cuda_rms_norm(d_hidden, d_final_norm_w, b.d_rms_out, HIDDEN_DIM, RMS_NORM_EPS, stream);
cudaFree(d_final_norm_w);

// lm_head: [248320, 2048] BF16 matvec → logits [248320]
float *d_logits;
CHECK_CUDA(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));
// lm_head is at offset 1017118720 in the binary
// Use gpu_bf16_matvec_direct or load the weight ourselves
snprintf(tname, sizeof(tname), "lm_head");
gpu_bf16_matvec_direct(b.d_rms_out, d_logits, VOCAB_SIZE, HIDDEN_DIM, &wd, tname, stream);
```

Wait - `gpu_bf16_matvec_direct` load the full lm_head weight (970MB) to GPU scratch buffer, which is too big. Better approach: use the BF16 matvec kernel but we need to pass the right pointer.

Actually, `gpu_bf16_matvec_direct` uses `ensure_scratch(w_bytes + 4096)` which would try to allocate 970MB. That's bad. Need a different approach: mmap the weight and pass the pointer directly without copying to scratch.

```c
// lm_head: load BF16 weight pointer directly from mmap (no GPU scratch copy)
// lm_head offset = 1017118720 in the binary file
int lm_idx = find_tensor(&wd, "lm_head");
if (lm_idx >= 0) {
    TensorInfo *t = &wd.tensors[lm_idx];
    size_t data_start = 4 + wd.header_size;
    size_t data_start_aligned = (data_start + 63) & ~63ULL;
    uint16_t *lm_head_ptr = (uint16_t *)((uint8_t *)wd.base + (t->offset - data_start_aligned));

    // cudaMalloc the weight on GPU, copy from mmap
    float *d_logits;
    CHECK_CUDA(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));
    uint16_t *d_lm_head;
    CHECK_CUDA(cudaMalloc(&d_lm_head, VOCAB_SIZE * HIDDEN_DIM * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemcpy(d_lm_head, lm_head_ptr,
                          VOCAB_SIZE * HIDDEN_DIM * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    cuda_bf16_matvec(d_lm_head, b.d_rms_out, d_logits, VOCAB_SIZE, HIDDEN_DIM, stream);
    cudaFree(d_lm_head);
```

Actually the lm_head is 970MB - too big for GPU. Let's do it in chunks: load chunks of lm_head, do partial matvec, accumulate.

Or even simpler for first token: just keep it on CPU. Read 2048 BF16 weights per vocab entry, compute dot products. 248320 × 2048 = 500M operations. On CPU this would be slow (~seconds).

Best approach for correctness: GPU chunked matvec. Split lm_head into chunks of e.g. 8192 rows at a time:
- Load 8192 × 2048 BF16 weights to GPU → matvec → partial logits
- Repeat 248320/8192 ≈ 30 times

```c
#define LM_HEAD_CHUNK 8192
float *d_logits;
CHECK_CUDA(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));
uint16_t *d_lm_chunk;
CHECK_CUDA(cudaMalloc(&d_lm_chunk, LM_HEAD_CHUNK * HIDDEN_DIM * sizeof(uint16_t)));

int lm_idx = find_tensor(&wd, "lm_head");
TensorInfo *t = &wd.tensors[lm_idx];
size_t data_start = 4 + wd.header_size;
size_t data_start_aligned = (data_start + 63) & ~63ULL;
uint16_t *lm_head_ptr = (uint16_t *)((uint8_t *)wd.base + (t->offset - data_start_aligned));

for (int chunk = 0; chunk < VOCAB_SIZE; chunk += LM_HEAD_CHUNK) {
    int chunk_size = min(LM_HEAD_CHUNK, VOCAB_SIZE - chunk);
    size_t offset = (size_t)chunk * HIDDEN_DIM;
    CHECK_CUDA(cudaMemcpy(d_lm_chunk, lm_head_ptr + offset,
                          chunk_size * HIDDEN_DIM * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    cuda_bf16_matvec(d_lm_chunk, b.d_rms_out, d_logits + chunk,
                      chunk_size, HIDDEN_DIM, stream);
}
cudaFree(d_lm_chunk);
```

- [ ] **Step 3: Argmax**

```c
// Find argmax of logits on CPU
float *cpu_logits = (float *)malloc(VOCAB_SIZE * sizeof(float));
CHECK_CUDA(cudaMemcpy(cpu_logits, d_logits, VOCAB_SIZE * sizeof(float),
                      cudaMemcpyDeviceToHost));
cudaFree(d_logits);

int next_token = 0;
float max_logit = cpu_logits[0];
for (int i = 1; i < VOCAB_SIZE; i++) {
    if (cpu_logits[i] > max_logit) {
        max_logit = cpu_logits[i];
        next_token = i;
    }
}
free(cpu_logits);
```

- [ ] **Step 4: BPE decode**

```c
// Decode single token
char token_str[BPE_MAX_TOKEN_LEN];
int token_len = bpe_decode_token(&tokenizer, next_token, token_str, BPE_MAX_TOKEN_LEN);
printf("%.*s", token_len, token_str);
fflush(stdout);
```

- [ ] **Step 5: Autoregressive loop**

Wrap the forward pass in a loop over `max_tokens`:

```c
int position = 0;
uint32_t current_token = input_ids[num_tokens - 1];

for (int t = 0; t < max_tokens; t++) {
    // Embed current token
    size_t embed_offset = (size_t)current_token * HIDDEN_DIM * sizeof(uint16_t);
    void *embed_src = (uint8_t *)mf.data + ((4 + wd.header_size + 63) & ~63ULL) + embed_offset;
    float cpu_embed[HIDDEN_DIM];
    uint16_t *ebf16 = (uint16_t *)embed_src;
    for (int i = 0; i < HIDDEN_DIM; i++) cpu_embed[i] = bf16_to_f32(ebf16[i]);
    CHECK_CUDA(cudaMemcpy(d_hidden, cpu_embed, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Run 40 layers
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        forward_layer_gpu(d_hidden, &wd, &buffers, layer, position,
                          kv_caches, linear_states, stream);
    }

    // Final norm + lm_head (chunked)
    // ... (as above) ...

    // Argmax + decode
    // ... (as above) ...

    position++;
    current_token = next_token;

    // Check EOS
    if (current_token == EOS_TOKEN_1 || current_token == EOS_TOKEN_2) break;
}
```

### Step 2: Build and test

```bash
cd cuda_infer && make clean && make
./infer --prompt "Hello" --tokens 10 2>&1 | head -30
```
Expected: 输出有效的 UTF-8 文本。

### Step 3: Verify tests

```bash
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/ -v
```
Expected: 17 passed。

### Step 4: Commit

```bash
git add cuda_infer/infer.cu
git commit -m "feat(cuda_infer): end-to-end inference — embedding, lm_head, decode loop"
```
