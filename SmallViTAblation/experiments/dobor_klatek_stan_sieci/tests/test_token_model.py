import numpy as np
import torch
from SmallViTAblation.experiments.dobor_klatek_stan_sieci.model import LangevinConfig
from SmallViTAblation.experiments.dobor_klatek_stan_sieci.token_model import ScaledTokenLangevin


def fixture():
    torch.manual_seed(2)
    model = ScaledTokenLangevin(LangevinConfig(latent_dim=4, input_dim=6, context=8, hidden_dim=16),
                                np.ones(4), np.ones(4)*2)
    # Activate all trained output pathways, which otherwise initialize at zero.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                p.normal_(0, .03)
    z = torch.randn(2, 10, 4)
    inputs = torch.randn(2, 10, 6)
    observed = torch.rand(2, 10) > .5
    return model, z, inputs, observed


def test_target_latent_metadata_and_future_do_not_change_prediction():
    model, z, inputs, observed = fixture()
    before = model(z, inputs, observed)['prediction'][:, :6].detach()
    z[:, 6:] += 100
    inputs[:, 6:] -= 100
    observed[:, 6:] = ~observed[:, 6:]
    after = model(z, inputs, observed)['prediction'][:, :6].detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_particle_scale_is_per_trajectory_and_causal():
    model, _, _, _ = fixture()
    v = torch.ones(2, 8, 4)
    v[1] *= 4
    scale = model.particle_scale(v)
    torch.testing.assert_close(scale[0], torch.ones_like(scale[0]))
    torch.testing.assert_close(scale[1], torch.full_like(scale[1], 4))
    v[:, 5:] = 1000
    torch.testing.assert_close(model.particle_scale(v)[:, :5], scale[:, :5])


def test_scaled_equation_terms_and_gradients_are_consistent():
    model, z, inputs, observed = fixture()
    out = model(z, inputs, observed)
    torch.testing.assert_close(out['prediction'], z[:, :-1]+sum(out['terms'].values()))
    torch.testing.assert_close(out['base_residual'], z[:, 1:]-z[:, :-1]-sum(out['terms'][k] for k in ('D','JP','W','Z')))
    out['residual'].square().mean().backward()
    assert model.gamma.grad is not None and torch.isfinite(model.gamma.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
