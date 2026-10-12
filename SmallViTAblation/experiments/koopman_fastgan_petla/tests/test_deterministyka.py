import pytest
import torch
import torch.nn.functional as F
from deterministyka import diagnostic_kernels


def test_context_restores_after_exception_and_preserves_gradients():
    original = F.grid_sample
    original_pool = F.adaptive_avg_pool2d
    old = torch.are_deterministic_algorithms_enabled()
    x = torch.arange(9., dtype=torch.float64).reshape(1, 1, 3, 3).requires_grad_()
    g = torch.tensor([[[[.2, -.1]]]], dtype=torch.float64, requires_grad=True)
    expected = original(x, g, align_corners=False)
    expected_grads = torch.autograd.grad(expected.sum(), (x, g))
    with pytest.raises(RuntimeError, match='controlled'):
        with diagnostic_kernels():
            got = F.grid_sample(x, g, align_corners=False)
            grads = torch.autograd.grad(got.sum(), (x, g))
            assert torch.equal(got, expected)
            assert all(torch.equal(a, b) for a, b in zip(grads, expected_grads))
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            pooled = F.adaptive_avg_pool2d(x, 1)
            assert torch.equal(torch.autograd.grad(pooled.sum(), x)[0], torch.ones_like(x)/9)
            raise RuntimeError('controlled')
    assert F.grid_sample is original
    assert F.adaptive_avg_pool2d is original_pool
    assert torch.are_deterministic_algorithms_enabled() == old
