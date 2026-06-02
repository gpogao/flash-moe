import ctypes
import os
import torch
import numpy as np

_lib = None


def get_lib():
    global _lib
    if _lib is None:
        so_path = os.path.join(os.path.dirname(__file__), '..', 'libkernels.so')
        _lib = ctypes.CDLL(so_path)
    return _lib


def cuda_dequant_matvec_gptq(qweight, scales, qzeros, x, out_dim, in_dim, group_size):
    lib = get_lib()
    out = torch.zeros(out_dim, dtype=torch.float32, device='cuda')
    lib.cuda_dequant_matvec_gptq(
        ctypes.c_void_p(qweight.data_ptr()),
        ctypes.c_void_p(scales.data_ptr()),
        ctypes.c_void_p(qzeros.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(out_dim), ctypes.c_int(in_dim), ctypes.c_int(group_size),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return out


def cuda_swiglu(gate, up, dim):
    lib = get_lib()
    out = torch.zeros(dim, dtype=torch.float32, device='cuda')
    lib.cuda_swiglu(
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_void_p(up.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(dim),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return out


def cuda_rms_norm(x, weight, dim, eps=1e-6):
    lib = get_lib()
    out = torch.zeros(dim, dtype=torch.float32, device='cuda')
    lib.cuda_rms_norm(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(dim), ctypes.c_float(eps),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return out


def cuda_weighted_sum(expert_outputs, weights, num_experts, hidden):
    lib = get_lib()
    out = torch.zeros(hidden, dtype=torch.float32, device='cuda')
    lib.cuda_weighted_sum(
        ctypes.c_void_p(expert_outputs.data_ptr()),
        ctypes.c_void_p(weights.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(num_experts), ctypes.c_int(hidden),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return out


def cuda_rope(x, num_heads, head_dim, position, base=10000000.0):
    lib = get_lib()
    total = num_heads * head_dim
    out = torch.zeros(total, dtype=torch.float32, device='cuda')
    lib.cuda_rope(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(num_heads), ctypes.c_int(head_dim),
        ctypes.c_int(position), ctypes.c_float(base),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return out


def cuda_residual_add(a, b, dim):
    lib = get_lib()
    out = torch.zeros(dim, dtype=torch.float32, device='cuda')
    lib.cuda_residual_add(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(dim),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return out


def assert_close(actual, expected, atol=1e-4, rtol=1e-3, msg=""):
    diff = (actual - expected).abs().max().item()
    if diff >= atol:
        raise AssertionError(
            f"{msg} max diff {diff:.6f} exceeds atol={atol}"
        )
