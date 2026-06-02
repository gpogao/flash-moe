import torch
from conftest import cuda_residual_add, assert_close

HIDDEN_DIM = 2048

def test_residual_add():
    a = torch.randn(HIDDEN_DIM, device='cuda')
    b = torch.randn(HIDDEN_DIM, device='cuda')
    expected = a.cpu() + b.cpu()
    actual = cuda_residual_add(a, b, HIDDEN_DIM)
    assert_close(actual.cpu(), expected, msg="residual_add")
