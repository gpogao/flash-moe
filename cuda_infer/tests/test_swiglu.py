import torch
import torch.nn.functional as F
from conftest import cuda_swiglu, assert_close

MOE_INTERMEDIATE = 512

def test_swiglu():
    gate = torch.randn(MOE_INTERMEDIATE, device='cuda')
    up = torch.randn(MOE_INTERMEDIATE, device='cuda')
    expected = F.silu(gate.cpu()) * up.cpu()
    actual = cuda_swiglu(gate, up, MOE_INTERMEDIATE)
    assert_close(actual.cpu(), expected, msg="swiglu")
