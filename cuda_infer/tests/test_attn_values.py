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
            start = p * KV_DIM + kv_h * HEAD_DIM
            vp = v_cache[start:start + HEAD_DIM]
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
    assert_close(actual.cpu(), expected, atol=1e-4, msg="attn_values")
