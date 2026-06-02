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
    assert_close(aq.cpu(), eq, atol=1e-5, msg="rms_norm_qk_q")
    assert_close(ak.cpu(), ek, atol=1e-5, msg="rms_norm_qk_k")
