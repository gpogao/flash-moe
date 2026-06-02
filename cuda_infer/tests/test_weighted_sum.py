import torch
from conftest import cuda_weighted_sum, assert_close

HIDDEN_DIM = 2048
NUM_EXPERTS_PER_TOK = 8

def test_weighted_sum():
    expert_outs = torch.randn(NUM_EXPERTS_PER_TOK, HIDDEN_DIM, device='cuda')
    weights = torch.randn(NUM_EXPERTS_PER_TOK, device='cuda')
    expected = (weights.cpu().unsqueeze(-1) * expert_outs.cpu()).sum(dim=0).cuda()
    actual = cuda_weighted_sum(expert_outs, weights, NUM_EXPERTS_PER_TOK, HIDDEN_DIM)
    assert_close(actual, expected, msg="weighted_sum")
