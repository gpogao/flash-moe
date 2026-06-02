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
            start = p * KV_DIM + kv_h * HEAD_DIM
            kp = k_cache[start:start + HEAD_DIM]
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
