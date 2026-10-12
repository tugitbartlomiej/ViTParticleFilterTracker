"""Deterministyczny wariant diagnostyczny FP32; nie zmienia instalacji RF-DETR.

CUDA grid_sample ma niedeterministyczny backward. W małym pilocie liczymy tę
operację na CPU, zachowując autograd przez transfery. Wszystkie gałęzie, w tym
pretrening i continuous, muszą używać tego samego wariantu numerycznego.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def grid_sample_cpu(original, input, grid, *args, **kwargs):
    if input.device.type != 'cuda':
        return original(input, grid, *args, **kwargs)
    if input.dtype not in (torch.float32, torch.float64):
        raise ValueError('Diagnostic deterministic grid_sample requires FP32/FP64')
    return original(input.to('cpu'), grid.to('cpu'), *args, **kwargs).to(input.device)


def adaptive_pool_cpu(original, input, output_size):
    # FastGAN uses adaptive pooling in skip-layer excitation. CUDA backward is
    # nondeterministic in torch 2.9.1 even when the frozen weights do not change.
    if input.device.type == 'cuda' and input.requires_grad:
        return original(input.to('cpu'), output_size).to(input.device)
    return original(input, output_size)


@contextmanager
def diagnostic_kernels():
    original = F.grid_sample
    original_pool = F.adaptive_avg_pool2d
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul = torch.get_float32_matmul_precision()
    try:
        F.grid_sample = lambda input, grid, *args, **kwargs: grid_sample_cpu(original, input, grid, *args, **kwargs)
        F.adaptive_avg_pool2d = lambda input, output_size: adaptive_pool_cpu(original_pool, input, output_size)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    finally:
        F.grid_sample = original
        F.adaptive_avg_pool2d = original_pool
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.allow_tf32 = old_tf32
        torch.set_float32_matmul_precision(old_matmul)
