"""Eden CPU contract tests for a diagnostic transaction (no detector required)."""
import random
import numpy as np
import pytest
import torch
from ..rfdetr_backend import diagnostic_transaction, virtual_parameters


def test_diagnostics_restore_rng_modes_and_preserve_existing_gradients():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[0].eval()  # mixed modes must survive
    p = next(model.parameters())
    p.grad = torch.ones_like(p)
    torch_state, np_state, py_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    with pytest.raises(RuntimeError):
        with diagnostic_transaction(model):
            torch.rand(3)
            np.random.random()
            random.random()
            model.eval()
            raise RuntimeError('simulated failed diagnosis')
    assert model.training and not model[0].training
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert random.getstate() == py_state
    assert torch.equal(p.grad, torch.ones_like(p))


def test_virtual_gradient_step_restores_parameters_after_error():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    with pytest.raises(RuntimeError):
        with virtual_parameters([p], np.array([2., 4.]), .1):
            assert torch.allclose(p, torch.tensor([.8, 1.6]))
            raise RuntimeError('simulated failure')
    assert torch.equal(p, torch.tensor([1., 2.]))
