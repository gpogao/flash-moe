# MoE Routing + Expert Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现完整的 MoE routing（gate→softmax→topK）+ expert forward（GPTQ dequant）+ shared expert（BF16 matvec），替换当前占位 stub

**Architecture:** Routing gate BF16 matvec on GPU → CPU softmax+topK → 从 mmap 加载 K=8 expert GPTQ 权重到 GPU → GPU dequant matvec per expert → GPU weighted sum + shared expert BF16 matvec → GPU residual add

**Tech Stack:** CUDA 13.0, BF16 matvec, GPTQ-Int4 dequant matvec, mmap weight loading

---

## 文件结构

```
cuda_infer/
├── infer.cu            # [修改] 重写 forward_layer_gpu 的 MoE 部分
```

---

### Task 1: MoE Routing + Expert Forward + Shared Expert

**Files:**
- Modify: `cuda_infer/infer.cu`

- [ ] **Step 1: 重写 forward_layer_gpu 的 MoE 部分**

替换 `forward_layer_gpu` 中 Steps 5-9（当前占位：均匀 topK + 复制 normed hidden）：

```c
    // === Step 5: MoE routing (GPU gate matvec + CPU softmax + topK) ===
    float *d_routing_scores;
    CHECK_CUDA(cudaMalloc(&d_routing_scores, NUM_EXPERTS * sizeof(float)));

    snprintf(tname, sizeof(tname), "layers.%d.mlp.gate.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_rms_out, d_routing_scores, NUM_EXPERTS, HIDDEN_DIM,
                           wd, tname, stream);

    float *cpu_scores = (float *)malloc(NUM_EXPERTS * sizeof(float));
    CHECK_CUDA(cudaMemcpy(cpu_scores, d_routing_scores, NUM_EXPERTS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_routing_scores);

    // Softmax
    float max_score = cpu_scores[0];
    for (int i = 1; i < NUM_EXPERTS; i++)
        if (cpu_scores[i] > max_score) max_score = cpu_scores[i];
    float sum_exp = 0.0f;
    for (int i = 0; i < NUM_EXPERTS; i++) {
        cpu_scores[i] = expf(cpu_scores[i] - max_score);
        sum_exp += cpu_scores[i];
    }
    for (int i = 0; i < NUM_EXPERTS; i++) cpu_scores[i] /= sum_exp;

    int topk_idx[NUM_EXPERTS_PER_TOK];
    float topk_w[NUM_EXPERTS_PER_TOK];
    cpu_topk(cpu_scores, topk_idx, topk_w, NUM_EXPERTS, NUM_EXPERTS_PER_TOK);
    free(cpu_scores);

    // === Step 6: Expert forward on GPU ===
    // Load each selected expert's GPTQ weights and run dequant matvec
    CHECK_CUDA(cudaMemset(b->d_expert_out, 0,
                          NUM_EXPERTS_PER_TOK * HIDDEN_DIM * sizeof(float)));

    for (int k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
        int eid = topk_idx[k];

        // Load gate_proj
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.gate_proj.qweight", layer_idx, eid);
        size_t gate_qw_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / 8) * sizeof(uint32_t);
        size_t gate_sc_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * sizeof(uint16_t);
        ensure_scratch(gate_qw_bytes + gate_sc_bytes * 2 + 4096);
        uint32_t *d_gate_qw = (uint32_t *)d_scratch_w;
        uint16_t *d_gate_sc = (uint16_t *)((char *)d_scratch_w + gate_qw_bytes);
        uint16_t *d_gate_qz = (uint16_t *)((char *)d_gate_sc + gate_sc_bytes);
        load_tensor_to_gpu(wd, tname, d_gate_qw);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.gate_proj.scales", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_gate_sc);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.gate_proj.qzeros", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_gate_qz);

        cuda_dequant_matvec_gptq(d_gate_qw, d_gate_sc, d_gate_qz, b->d_rms_out,
                                  b->d_gate, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, stream);

        // Load up_proj
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.up_proj.qweight", layer_idx, eid);
        size_t up_qw_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / 8) * sizeof(uint32_t);
        size_t up_sc_bytes = MOE_INTERMEDIATE * (HIDDEN_DIM / GROUP_SIZE) * sizeof(uint16_t);
        ensure_scratch(up_qw_bytes + up_sc_bytes * 2 + 4096);
        uint32_t *d_up_qw = (uint32_t *)d_scratch_w;
        uint16_t *d_up_sc = (uint16_t *)((char *)d_scratch_w + up_qw_bytes);
        uint16_t *d_up_qz = (uint16_t *)((char *)d_up_sc + up_sc_bytes);
        load_tensor_to_gpu(wd, tname, d_up_qw);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.up_proj.scales", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_up_sc);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.up_proj.qzeros", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_up_qz);

        cuda_dequant_matvec_gptq(d_up_qw, d_up_sc, d_up_qz, b->d_rms_out,
                                  b->d_up, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, stream);

        cuda_swiglu(b->d_gate, b->d_up, b->d_swiglu, MOE_INTERMEDIATE, stream);

        // Load down_proj
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.down_proj.qweight", layer_idx, eid);
        size_t dn_qw_bytes = HIDDEN_DIM * (MOE_INTERMEDIATE / 8) * sizeof(uint32_t);
        size_t dn_sc_bytes = HIDDEN_DIM * (MOE_INTERMEDIATE / GROUP_SIZE) * sizeof(uint16_t);
        ensure_scratch(dn_qw_bytes + dn_sc_bytes * 2 + 4096);
        uint32_t *d_dn_qw = (uint32_t *)d_scratch_w;
        uint16_t *d_dn_sc = (uint16_t *)((char *)d_scratch_w + dn_qw_bytes);
        uint16_t *d_dn_qz = (uint16_t *)((char *)d_dn_sc + dn_sc_bytes);
        load_tensor_to_gpu(wd, tname, d_dn_qw);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.down_proj.scales", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_dn_sc);
        snprintf(tname, sizeof(tname), "layers.%d.mlp.experts.%d.down_proj.qzeros", layer_idx, eid);
        load_tensor_to_gpu(wd, tname, d_dn_qz);

        cuda_dequant_matvec_gptq(d_dn_qw, d_dn_sc, d_dn_qz, b->d_swiglu,
                                  b->d_expert_out + k * HIDDEN_DIM,
                                  HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE, stream);
    }

    // === Step 7: Weighted sum of expert outputs ===
    float *d_routing_weights;
    CHECK_CUDA(cudaMalloc(&d_routing_weights, NUM_EXPERTS_PER_TOK * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_routing_weights, topk_w,
                          NUM_EXPERTS_PER_TOK * sizeof(float), cudaMemcpyHostToDevice));
    cuda_weighted_sum(b->d_expert_out, d_routing_weights, b->d_output,
                       NUM_EXPERTS_PER_TOK, HIDDEN_DIM, stream);
    cudaFree(d_routing_weights);

    // === Step 8: Shared expert forward ===
    // gate_proj and up_proj: [512, 2048] BF16 matvec
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_rms_out, b->d_gate, MOE_INTERMEDIATE, HIDDEN_DIM,
                           wd, tname, stream);
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_rms_out, b->d_up, MOE_INTERMEDIATE, HIDDEN_DIM,
                           wd, tname, stream);
    cuda_swiglu(b->d_gate, b->d_up, b->d_swiglu, MOE_INTERMEDIATE, stream);

    // down_proj: [2048, 512] BF16 matvec
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    gpu_bf16_matvec_direct(b->d_swiglu, b->d_combined, HIDDEN_DIM, MOE_INTERMEDIATE,
                           wd, tname, stream);

    // Shared gate: [2048] BF16 vector dot with normed → sigmoid
    snprintf(tname, sizeof(tname), "layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    float *d_shared_gate_vec;
    CHECK_CUDA(cudaMalloc(&d_shared_gate_vec, HIDDEN_DIM * sizeof(uint16_t)));
    load_tensor_to_gpu(wd, tname, d_shared_gate_vec);
    // gate_val = sigmoid(dot(shared_gate, post_normed))
    float *d_gate_val;
    CHECK_CUDA(cudaMalloc(&d_gate_val, sizeof(float)));
    cuda_bf16_matvec((uint16_t *)d_shared_gate_vec, b->d_rms_out,
                      d_gate_val, 1, HIDDEN_DIM, stream);
    float gate_val_cpu;
    CHECK_CUDA(cudaMemcpy(&gate_val_cpu, d_gate_val, sizeof(float), cudaMemcpyDeviceToHost));
    float shared_gate = 1.0f / (1.0f + expf(-gate_val_cpu));
    cudaFree(d_shared_gate_vec); cudaFree(d_gate_val);

    // shared_out = sigmoid(gate) * down_proj_output + moe_out
    for (int i = 0; i < HIDDEN_DIM; i++) {
        // CPU combine: shared_gate * d_combined → TODO GPU kernel
    }
    // For now, just do residual add with moe_out only
    cuda_residual_add(d_hidden, b->d_output, d_hidden, HIDDEN_DIM, stream);
    // Also add shared expert: upload gate_val, do scaled add
    // Simplified: read d_combined to CPU, scale, upload, add
    float *shared_cpu = (float *)malloc(HIDDEN_DIM * sizeof(float));
    CHECK_CUDA(cudaMemcpy(shared_cpu, b->d_combined, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < HIDDEN_DIM; i++) shared_cpu[i] *= shared_gate;
    CHECK_CUDA(cudaMemcpy(b->d_combined, shared_cpu, HIDDEN_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));
    cuda_residual_add(d_hidden, b->d_combined, d_hidden, HIDDEN_DIM, stream);
    free(shared_cpu);
```

### Step 2: Build and test

```bash
cd cuda_infer && make clean && make
./infer --prompt "Hello" --tokens 5 2>&1 | head -30
```
Expected: 40 layers complete, expert weights loaded and used. No CUDA errors.

### Step 3: Verify ALL python tests

```bash
cd /home/perryshan/wlk/source/flash-moe && PYTHONPATH=cuda_infer/tests uv run pytest cuda_infer/tests/ -v
```
Expected: 17 passed.

### Step 4: Commit

```bash
git add cuda_infer/infer.cu
git commit -m "feat(cuda_infer): implement MoE routing, expert forward, and shared expert"
```
