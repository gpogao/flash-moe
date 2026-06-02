import torch
from conftest import cuda_rms_norm, assert_close

HIDDEN_DIM = 2048
RMS_NORM_EPS = 1e-6

def rms_norm_ref(x, weight, eps=1e-6):
    rms = torch.sqrt((x**2).mean(-1, keepdim=True) + eps)
    return x / rms * weight

def test_rms_norm():
    x = torch.randn(HIDDEN_DIM, device='cuda')
    w = torch.randn(HIDDEN_DIM, device='cuda')
    expected = rms_norm_ref(x, w, RMS_NORM_EPS)
    actual = cuda_rms_norm(x, w, HIDDEN_DIM, RMS_NORM_EPS)
    assert_close(actual, expected, atol=1e-3, msg="rms_norm")
