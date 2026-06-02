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
