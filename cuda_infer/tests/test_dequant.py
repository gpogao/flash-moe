import torch
from conftest import cuda_dequant_matvec_gptq, assert_close

HIDDEN_DIM = 2048
MOE_INTERMEDIATE = 512
GROUP_SIZE = 128


def unpack_nibbles(qweight, out_dim, in_dim):
    packed_cols = in_dim // 8
    nibbles = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    for row in range(out_dim):
        for col in range(packed_cols):
            val = int(qweight[row, col])
            for n in range(8):
                nibbles[row, col * 8 + n] = float((val >> (n * 4)) & 0xF)
    return nibbles


def dequant_gptq_ref(qweight, scales, qzeros, out_dim, in_dim, group_size):
    nibbles = unpack_nibbles(qweight, out_dim, in_dim)
    num_groups = in_dim // group_size
    result = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    for row in range(out_dim):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            scale = float(scales[row, g].to(torch.float32))
            zero = float(qzeros[row, g].to(torch.float32))
            result[row, start:end] = (nibbles[row, start:end] - zero) * scale
    return result


def make_gptq_test_data(out_dim, in_dim, group_size):
    num_groups = in_dim // group_size
    qweight = torch.randint(0, 2**32, (out_dim, in_dim // 8),
                            dtype=torch.uint32, device='cuda')
    scales = torch.randn(out_dim, num_groups, device='cuda').bfloat16()
    qzeros = (torch.rand(out_dim, num_groups, device='cuda') * 8).bfloat16()
    x = torch.randn(in_dim, device='cuda')
    return qweight, scales, qzeros, x


def test_dequant_gate_up():
    out_dim, in_dim = MOE_INTERMEDIATE, HIDDEN_DIM
    qweight, scales, qzeros, x = make_gptq_test_data(out_dim, in_dim, GROUP_SIZE)
    W_deq = dequant_gptq_ref(qweight.cpu(), scales.cpu(), qzeros.cpu(),
                              out_dim, in_dim, GROUP_SIZE)
    expected = torch.matmul(W_deq, x.cpu()).cuda()
    actual = cuda_dequant_matvec_gptq(qweight, scales, qzeros, x,
                                       out_dim, in_dim, GROUP_SIZE)
    assert_close(actual, expected, msg="dequant gate_up")


def test_dequant_down():
    out_dim, in_dim = HIDDEN_DIM, MOE_INTERMEDIATE
    qweight, scales, qzeros, x = make_gptq_test_data(out_dim, in_dim, GROUP_SIZE)
    W_deq = dequant_gptq_ref(qweight.cpu(), scales.cpu(), qzeros.cpu(),
                              out_dim, in_dim, GROUP_SIZE)
    expected = torch.matmul(W_deq, x.cpu()).cuda()
    actual = cuda_dequant_matvec_gptq(qweight, scales, qzeros, x,
                                       out_dim, in_dim, GROUP_SIZE)
    assert_close(actual, expected, msg="dequant down_proj")
