import torch
from conftest import get_lib, assert_close
import ctypes

NUM_V_HEADS = 8
NUM_K_HEADS = 4
VALUE_DIM = 16
KEY_DIM = 16
K_HEADS_PER_V = NUM_V_HEADS // NUM_K_HEADS
TOTAL_K = NUM_K_HEADS * KEY_DIM
TOTAL_V = NUM_V_HEADS * VALUE_DIM

def cuda_gated_delta_net_step(state, q, k, v, g_decay, beta_gate):
    lib = get_lib()
    output = torch.zeros(TOTAL_V, dtype=torch.float32, device='cuda')
    lib.cuda_gated_delta_net_step(
        ctypes.c_void_p(state.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(g_decay.data_ptr()),
        ctypes.c_void_p(beta_gate.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(NUM_V_HEADS), ctypes.c_int(VALUE_DIM),
        ctypes.c_int(KEY_DIM), ctypes.c_int(K_HEADS_PER_V),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return output, state

def delta_net_ref(state, q, k, v, g_decay, beta_gate):
    state = state.cpu().clone().reshape(NUM_V_HEADS, VALUE_DIM, KEY_DIM)
    q = q.cpu().reshape(NUM_K_HEADS, KEY_DIM)
    k = k.cpu().reshape(NUM_K_HEADS, KEY_DIM)
    v = v.cpu().reshape(NUM_V_HEADS, VALUE_DIM)
    out = torch.zeros(NUM_V_HEADS, VALUE_DIM)
    for vh in range(NUM_V_HEADS):
        kh = vh // K_HEADS_PER_V
        g = g_decay[vh].item()
        bg = beta_gate[vh].item()
        state[vh] *= g
        for vi in range(VALUE_DIM):
            kv_mem = (state[vh, vi] * k[kh]).sum()
            delta = (v[vh, vi] - kv_mem) * bg
            state[vh, vi] += k[kh] * delta
            out[vh, vi] = (state[vh, vi] * q[kh]).sum()
    return out.flatten(), state.flatten()

def test_delta_net():
    torch.manual_seed(42)
    state = torch.randn(NUM_V_HEADS * VALUE_DIM * KEY_DIM, device='cuda') * 0.1
    q = torch.randn(TOTAL_K, device='cuda')
    k = torch.randn(TOTAL_K, device='cuda')
    v = torch.randn(TOTAL_V, device='cuda')
    g_decay = torch.rand(NUM_V_HEADS, device='cuda') * 0.5 + 0.5
    beta_gate = torch.rand(NUM_V_HEADS, device='cuda') * 0.5 + 0.5
    e_out, e_state = delta_net_ref(state, q, k, v, g_decay, beta_gate)
    a_out, a_state = cuda_gated_delta_net_step(state, q, k, v, g_decay, beta_gate)
    assert_close(a_out.cpu(), e_out, atol=1e-4, msg="delta_net_out")
    assert_close(a_state.cpu(), e_state, atol=1e-4, msg="delta_net_state")
