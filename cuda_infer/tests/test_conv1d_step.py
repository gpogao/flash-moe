import torch
from conftest import get_lib, assert_close
import ctypes

CONV_DIM = 128
KERNEL_SIZE = 4


def cuda_conv1d_step(conv_state, input_t, weight_bf16, conv_dim):
    lib = get_lib()
    output = torch.zeros(conv_dim, dtype=torch.float32, device='cuda')
    lib.cuda_conv1d_step(
        ctypes.c_void_p(conv_state.data_ptr()),
        ctypes.c_void_p(input_t.data_ptr()),
        ctypes.c_void_p(weight_bf16.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int(conv_dim),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return output, conv_state


def conv1d_step_ref(conv_state, input_t, weight_bf16, conv_dim):
    conv_state = conv_state.cpu().clone()
    input_t = input_t.cpu()
    weight = weight_bf16.cpu().view(torch.bfloat16).to(torch.float32)
    out = torch.zeros(conv_dim)
    for c in range(conv_dim):
        acc = (conv_state[0, c] * weight[c, 0] +
               conv_state[1, c] * weight[c, 1] +
               conv_state[2, c] * weight[c, 2] +
               input_t[c] * weight[c, 3])
        out[c] = acc / (1.0 + (-acc).exp())
    new_state = conv_state.clone()
    new_state[0] = conv_state[1]
    new_state[1] = conv_state[2]
    new_state[2] = input_t
    return out, new_state


def test_conv1d_step():
    torch.manual_seed(42)
    conv_dim = CONV_DIM
    state = torch.randn(3, conv_dim, device='cuda')
    x = torch.randn(conv_dim, device='cuda')
    w = torch.randn(conv_dim, KERNEL_SIZE).bfloat16()
    w_uint16 = w.view(torch.uint16).cuda()
    expected_out, expected_state = conv1d_step_ref(state, x, w_uint16, conv_dim)
    actual_out, actual_state = cuda_conv1d_step(state, x, w_uint16, conv_dim)
    assert_close(actual_out.cpu(), expected_out, atol=1e-4, msg="conv1d_out")
    assert_close(actual_state.cpu(), expected_state, atol=1e-4, msg="conv1d_state")
