import torch
from conftest import cuda_rope, assert_close

NUM_ATTN_HEADS = 16
HEAD_DIM = 256


def rope_ref(x, num_heads, head_dim, position, base=10000000.0):
    x = x.cpu().clone().reshape(num_heads, head_dim)
    out = x.clone()
    rotary_dim = 64
    for h in range(num_heads):
        for d in range(rotary_dim // 2):
            angle = position / (base ** (2.0 * d / rotary_dim))
            cos_val = float(torch.cos(torch.tensor(angle)))
            sin_val = float(torch.sin(torch.tensor(angle)))
            # Use .item() to get scalar copies, avoiding PyTorch view aliasing
            # where out[h, d] would share memory and get corrupted after first write
            x0 = out[h, d].item()
            x1 = out[h, d + head_dim // 2].item()
            out[h, d] = x0 * cos_val - x1 * sin_val
            out[h, d + head_dim // 2] = x0 * sin_val + x1 * cos_val
    return out.flatten()


def test_rope():
    total = NUM_ATTN_HEADS * HEAD_DIM
    x = torch.randn(total, device='cuda')
    pos = 7
    expected = rope_ref(x, NUM_ATTN_HEADS, HEAD_DIM, pos)
    actual = cuda_rope(x, NUM_ATTN_HEADS, HEAD_DIM, pos)
    assert_close(actual, expected.cuda(), atol=1e-4, msg="rope")
