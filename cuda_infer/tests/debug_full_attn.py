"""Debug: compare PyTorch vs CUDA for full attention layer 3"""
import torch, ctypes, struct, json, os, sys, numpy as np

HIDDEN_DIM = 2048
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
MAX_SEQ_LEN = 8192

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'libkernels.so'))

import pathlib
CUDA_DIR = pathlib.Path(__file__).parent.parent

def read_tensor(bin_path, tensor_name):
    bin_path = str(CUDA_DIR / 'model_weights.bin')
    with open(str(CUDA_DIR / 'model_weights.json')) as f:
        manifest = json.load(f)
    layers = manifest['layers']
    for lname, tensors in layers.items():
        for tname, info in tensors.items():
            if tname == tensor_name:
                with open(bin_path, 'rb') as f:
                    header_size = struct.unpack('I', f.read(4))[0]
                    data_start = (4 + header_size + 63) & ~63
                    f.seek(data_start + info['offset'])
                    data = f.read(info['size'])
                    shape = info['shape']
                    print(f"  {tensor_name}: shape={shape}, size={info['size']}")
                    return data, shape
    raise KeyError(f"Tensor {tensor_name} not found")

def bf16_to_f32(data):
    arr = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
    arr = arr << 16
    return arr.view(np.float32).reshape(-1)

def f32_to_bf16(arr):
    u32 = arr.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)

# Step 1: Generate random input matching "H" token embedding
torch.manual_seed(42)
hidden_cpu = torch.randn(HIDDEN_DIM) * 0.01

print(f"Input: rms={hidden_cpu.pow(2).mean().sqrt():.6f}")

# Step 2: Load layer 3 input_layernorm weight
ln_data, ln_shape = read_tensor('model_weights.bin', 'layers.3.input_layernorm.weight')
ln_w = torch.from_numpy(np.frombuffer(ln_data, dtype=np.float32))

# Step 3: RMS norm (PyTorch)
rms = torch.sqrt(hidden_cpu.pow(2).mean() + 1e-6)
normed = hidden_cpu / rms * ln_w
print(f"normed: rms={normed.pow(2).mean().sqrt():.6f}")

# Step 4: Q projection (BF16 matvec)
q_w_data, q_w_shape = read_tensor('model_weights.bin', 'layers.3.self_attn.q_proj.weight')
q_w = torch.from_numpy(bf16_to_f32(q_w_data).reshape(q_w_shape))
q_proj = torch.matmul(q_w, normed)
print(f"q_proj: rms={q_proj.pow(2).mean().sqrt():.6f} nan={torch.isnan(q_proj).sum().item()}")

# Step 5: Split Q and q_gate
q_dim = NUM_ATTN_HEADS * HEAD_DIM
q = q_proj[:q_dim].reshape(NUM_ATTN_HEADS, HEAD_DIM)
q_gate = q_proj[q_dim:].reshape(NUM_ATTN_HEADS, HEAD_DIM)

# Step 6: K projection
k_w_data, k_w_shape = read_tensor('model_weights.bin', 'layers.3.self_attn.k_proj.weight')
k_w = torch.from_numpy(bf16_to_f32(k_w_data).reshape(k_w_shape))
k = torch.matmul(k_w, normed).reshape(NUM_KV_HEADS, HEAD_DIM)

# Step 7: V projection
v_w_data, v_w_shape = read_tensor('model_weights.bin', 'layers.3.self_attn.v_proj.weight')
v_w = torch.from_numpy(bf16_to_f32(v_w_data).reshape(v_w_shape))
v = torch.matmul(v_w, normed).reshape(NUM_KV_HEADS, HEAD_DIM)

print(f"K rms={k.pow(2).mean().sqrt():.6f}  V rms={v.pow(2).mean().sqrt():.6f}")

# Step 8: Per-head Q/K RMS norm
q_norm_data, _ = read_tensor('model_weights.bin', 'layers.3.self_attn.q_norm.weight')
q_norm_w = torch.from_numpy(np.frombuffer(q_norm_data, dtype=np.float32))
k_norm_data, _ = read_tensor('model_weights.bin', 'layers.3.self_attn.k_norm.weight')
k_norm_w = torch.from_numpy(np.frombuffer(k_norm_data, dtype=np.float32))

q_normed = torch.zeros_like(q)
for h in range(NUM_ATTN_HEADS):
    r = torch.sqrt(q[h].pow(2).mean() + 1e-6)
    q_normed[h] = q[h] / r * q_norm_w

k_normed = torch.zeros_like(k)
for h in range(NUM_KV_HEADS):
    r = torch.sqrt(k[h].pow(2).mean() + 1e-6)
    k_normed[h] = k[h] / r * k_norm_w

# Step 9: RoPE (PyTorch)
position = 0
rotary_dim = 64
base = 10000000.0
half = HEAD_DIM // 2

def apply_rope(x, num_heads, pos):
    x = x.clone()
    for h in range(num_heads):
        for d in range(rotary_dim // 2):
            angle = pos / (base ** (2.0 * d / rotary_dim))
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)
            x0 = x[h, d].item()
            x1 = x[h, d + half].item()
            x[h, d] = x0 * cos_val - x1 * sin_val
            x[h, d + half] = x0 * sin_val + x1 * cos_val
    return x

q_rope = apply_rope(q_normed, NUM_ATTN_HEADS, position)
k_rope = apply_rope(k_normed, NUM_KV_HEADS, position)

print(f"q_rope rms={q_rope.pow(2).mean().sqrt():.6f} nan={torch.isnan(q_rope).sum().item()}")
print(f"k_rope rms={k_rope.pow(2).mean().sqrt():.6f} nan={torch.isnan(k_rope).sum().item()}")

# Step 10: Attention scores (PyTorch) - self-attention with seq_len=1
scale = 1.0 / np.sqrt(HEAD_DIM)
hpk = NUM_ATTN_HEADS // NUM_KV_HEADS
scores = torch.zeros(NUM_ATTN_HEADS)
for h in range(NUM_ATTN_HEADS):
    kv_h = h // hpk
    scores[h] = torch.dot(q_rope[h], k_rope[kv_h]) * scale

print(f"scores: rms={scores.pow(2).mean().sqrt():.6f} nan={torch.isnan(scores).sum().item()}")
print(f"scores values={scores[:4]}")

# Step 11: Softmax (seq_len=1 → all 1.0)
attn_w = torch.ones(NUM_ATTN_HEADS)

# Step 12: Context = V expanded with GQA
context = torch.zeros(NUM_ATTN_HEADS, HEAD_DIM)
for h in range(NUM_ATTN_HEADS):
    kv_h = h // hpk
    context[h] = v[kv_h]

print(f"context rms={context.pow(2).mean().sqrt():.6f} nan={torch.isnan(context).sum().item()}")

# Step 13: Sigmoid gate
context_flat = context.flatten()
q_gate_flat = q_gate.flatten()
ctx_gated = context_flat * torch.sigmoid(q_gate_flat)
print(f"ctx_gated rms={ctx_gated.pow(2).mean().sqrt():.6f} nan={torch.isnan(ctx_gated).sum().item()}")

# Step 14: O projection
o_w_data, o_w_shape = read_tensor('model_weights.bin', 'layers.3.self_attn.o_proj.weight')
o_w = torch.from_numpy(bf16_to_f32(o_w_data).reshape(o_w_shape))
attn_out = torch.matmul(o_w, ctx_gated)
print(f"attn_out rms={attn_out.pow(2).mean().sqrt():.6f} nan={torch.isnan(attn_out).sum().item()}")

# Step 15: Now compare with CUDA kernel outputs
# Upload Q to GPU and call cuda_attn_scores
d_q = q_rope.flatten().cuda()
d_k = k_rope.flatten().cuda()
d_scores = torch.zeros(NUM_ATTN_HEADS * MAX_SEQ_LEN, device='cuda')

lib.cuda_attn_scores(
    ctypes.c_void_p(d_q.data_ptr()),
    ctypes.c_void_p(d_k.data_ptr()),
    ctypes.c_void_p(d_scores.data_ptr()),
    ctypes.c_int(HEAD_DIM), ctypes.c_int(NUM_KV_HEADS * HEAD_DIM),
    ctypes.c_int(1), ctypes.c_int(MAX_SEQ_LEN),
    ctypes.c_float(scale), ctypes.c_int(hpk),
    ctypes.c_int(1),
    ctypes.c_void_p(0),
)
torch.cuda.synchronize()

cuda_header_scores = torch.zeros(NUM_ATTN_HEADS)
for h in range(NUM_ATTN_HEADS):
    cuda_header_scores[h] = d_scores[h * MAX_SEQ_LEN].cpu()

print(f"\nCUDA scores: rms={cuda_header_scores.pow(2).mean().sqrt():.6f}")
print(f"CUDA scores values={cuda_header_scores[:4]}")
print(f"Diff: {(cuda_header_scores - scores).abs().max():.6f}")
