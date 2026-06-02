import torch
import torch.nn.functional as F
from conftest import (
    cuda_dequant_matvec_gptq, cuda_swiglu,
    cuda_rms_norm, cuda_weighted_sum, cuda_residual_add,
    assert_close,
)

HIDDEN_DIM = 2048
MOE_INTERMEDIATE = 512
GROUP_SIZE = 128
NUM_EXPERTS_PER_TOK = 8


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


def make_gptq_tensor(out_dim, in_dim):
    num_groups = in_dim // GROUP_SIZE
    qweight = torch.randint(0, 2**32, (out_dim, in_dim // 8),
                            dtype=torch.uint32, device='cuda')
    scales = torch.randn(out_dim, num_groups, device='cuda').bfloat16()
    qzeros = (torch.rand(out_dim, num_groups, device='cuda') * 8).bfloat16()
    return qweight, scales, qzeros


def dequant_ref(qweight, scales, qzeros, x, out_dim, in_dim):
    W = dequant_gptq_ref(qweight.cpu(), scales.cpu(), qzeros.cpu(),
                          out_dim, in_dim, GROUP_SIZE)
    return torch.matmul(W, x.cpu())


def test_moe_layer_pipeline():
    torch.manual_seed(42)
    hidden = torch.randn(HIDDEN_DIM, device='cuda')
    norm_w = torch.randn(HIDDEN_DIM, device='cuda')

    expert_gate_qw, expert_gate_sc, expert_gate_qz = make_gptq_tensor(MOE_INTERMEDIATE, HIDDEN_DIM)
    expert_up_qw, expert_up_sc, expert_up_qz = make_gptq_tensor(MOE_INTERMEDIATE, HIDDEN_DIM)
    expert_down_qw, expert_down_sc, expert_down_qz = make_gptq_tensor(HIDDEN_DIM, MOE_INTERMEDIATE)
    routing_weights = torch.randn(NUM_EXPERTS_PER_TOK, device='cuda').softmax(dim=0)

    # GPU pipeline
    gpu_normed = cuda_rms_norm(hidden, norm_w, HIDDEN_DIM)
    gpu_expert_outs = torch.zeros(NUM_EXPERTS_PER_TOK * HIDDEN_DIM, device='cuda')
    for k in range(NUM_EXPERTS_PER_TOK):
        gate_out = cuda_dequant_matvec_gptq(expert_gate_qw, expert_gate_sc, expert_gate_qz,
                                             gpu_normed, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE)
        up_out = cuda_dequant_matvec_gptq(expert_up_qw, expert_up_sc, expert_up_qz,
                                           gpu_normed, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE)
        swiglu_out = cuda_swiglu(gate_out, up_out, MOE_INTERMEDIATE)
        down_out = cuda_dequant_matvec_gptq(expert_down_qw, expert_down_sc, expert_down_qz,
                                             swiglu_out, HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE)
        gpu_expert_outs[k * HIDDEN_DIM:(k + 1) * HIDDEN_DIM] = down_out
    gpu_combined = cuda_weighted_sum(gpu_expert_outs, routing_weights,
                                      NUM_EXPERTS_PER_TOK, HIDDEN_DIM)
    gpu_output = cuda_residual_add(hidden, gpu_combined, HIDDEN_DIM)

    # PyTorch reference
    h = hidden.cpu()
    rms = torch.sqrt((h**2).mean(-1, keepdim=True) + 1e-6)
    ref_normed = h / rms * norm_w.cpu()
    ref_expert_outs = []
    for k in range(NUM_EXPERTS_PER_TOK):
        gate = dequant_ref(expert_gate_qw, expert_gate_sc, expert_gate_qz, ref_normed,
                           MOE_INTERMEDIATE, HIDDEN_DIM)
        up = dequant_ref(expert_up_qw, expert_up_sc, expert_up_qz, ref_normed,
                         MOE_INTERMEDIATE, HIDDEN_DIM)
        swiglu = F.silu(gate) * up
        down = dequant_ref(expert_down_qw, expert_down_sc, expert_down_qz, swiglu,
                           HIDDEN_DIM, MOE_INTERMEDIATE)
        ref_expert_outs.append(down)
    ref_combined = (routing_weights.cpu().unsqueeze(-1) * torch.stack(ref_expert_outs)).sum(dim=0)
    ref_output = h + ref_combined

    assert_close(gpu_output.cpu(), ref_output, atol=25, msg="moe_layer_pipeline")
