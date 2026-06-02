import torch
import torch.nn.functional as F
from conftest import get_lib, assert_close
import ctypes

NUM_V_HEADS = 64


def cuda_compute_decay_beta(alpha, beta, A_log, dt_bias, num_heads):
    lib = get_lib()
    g_decay = torch.zeros(num_heads, dtype=torch.float32, device='cuda')
    beta_gate = torch.zeros(num_heads, dtype=torch.float32, device='cuda')
    lib.cuda_compute_decay_beta(
        ctypes.c_void_p(alpha.data_ptr()),
        ctypes.c_void_p(beta.data_ptr()),
        ctypes.c_void_p(A_log.data_ptr()),
        ctypes.c_void_p(dt_bias.data_ptr()),
        ctypes.c_void_p(g_decay.data_ptr()),
        ctypes.c_void_p(beta_gate.data_ptr()),
        ctypes.c_int(num_heads),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return g_decay, beta_gate


def decay_beta_ref(alpha, beta, A_log, dt_bias):
    softplus = F.softplus(alpha + dt_bias.to(torch.float32))
    A_val = A_log.exp()
    g = (-A_val * softplus).exp()
    bg = beta.sigmoid()
    return g, bg


def test_decay_beta():
    torch.manual_seed(42)
    n = NUM_V_HEADS
    alpha = torch.randn(n, device='cuda')
    beta = torch.randn(n, device='cuda')
    A_log = torch.randn(n, device='cuda')
    dt_bias = torch.randn(n).bfloat16()
    dt_bias_u16 = dt_bias.view(torch.uint16).cuda()
    eg, ebg = decay_beta_ref(alpha.cpu(), beta.cpu(), A_log.cpu(), dt_bias)
    ag, abg = cuda_compute_decay_beta(alpha, beta, A_log, dt_bias_u16, n)
    assert_close(ag.cpu(), eg, atol=1e-5, msg="g_decay")
    assert_close(abg.cpu(), ebg, atol=1e-5, msg="beta_gate")
