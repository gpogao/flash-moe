import torch
import torch.nn.functional as F
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
    w = weight_bf16.to(torch.float32)
    out = torch.zeros(num_heads, value_dim)
    for h in range(num_heads):
        rms = torch.sqrt((values[h]**2).mean() + eps)
        normed = values[h] / rms
        out[h] = normed * F.silu(z[h]) * w
    return out.flatten()


def test_gated_rms_norm():
    torch.manual_seed(42)
    values = torch.randn(TOTAL, device='cuda')
    z = torch.randn(TOTAL, device='cuda')
    w = torch.randn(VALUE_DIM).bfloat16()
    w_u16 = w.view(torch.uint16).cuda()
    expected = gated_rms_norm_ref(values, z, w, NUM_V_HEADS, VALUE_DIM)
    actual = cuda_gated_rms_norm(values, z, w_u16, NUM_V_HEADS, VALUE_DIM)
    assert_close(actual.cpu(), expected, atol=1e-4, msg="gated_rms_norm")
