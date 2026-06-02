import torch
from conftest import get_lib, assert_close
import ctypes

NUM_ATTN_HEADS = 16
HEAD_DIM = 256
Q_DIM = NUM_ATTN_HEADS * HEAD_DIM


def cuda_sigmoid_gate(x_out, gate, dim):
    lib = get_lib()
    lib.cuda_sigmoid_gate(
        ctypes.c_void_p(x_out.data_ptr()),
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_int(dim),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return x_out


def test_sigmoid_gate():
    torch.manual_seed(42)
    context = torch.randn(Q_DIM, device='cuda')
    gate = torch.randn(Q_DIM, device='cuda')
    expected = (context.cpu() * torch.sigmoid(gate.cpu())).cuda()
    actual = cuda_sigmoid_gate(context.clone(), gate, Q_DIM)
    assert_close(actual, expected, atol=1e-5, msg="sigmoid_gate")
