"""Sprawdza przyczynowosc portu Grant; odczyt celu lub zly lag obala poprawny port.

Run from SmallViTAblation:
    python -m unittest discover -s experiments/dobor_klatek_stan_sieci/tests -p test_model.py -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODEL_PATH = Path(__file__).resolve().parents[1] / "model.py"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, *, active=True, device="cpu", **options):
        self.assertTrue(MODEL_PATH.is_file(), "Port ExtendedLangevin is not implemented")
        name = "test_extended_langevin_implementation"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, MODEL_PATH)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        module = sys.modules[name]
        settings = dict(latent_dim=4, input_dim=3, context=8, hidden_dim=16,
                        heads=2, residual_rank=2)
        settings.update(options)
        torch.manual_seed(21)
        model = module.ExtendedLangevin(module.LangevinConfig(**settings)).to(device)
        if active:
            # Grant starts output maps at zero: activate them so leakage tests
            # cannot pass merely because every learned term currently vanishes.
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.ndim >= 2:
                        parameter.normal_(std=0.17)
        return model

    def sequence(self, device="cpu"):
        generator = torch.Generator().manual_seed(39)
        z = torch.randn(2, 9, 4, generator=generator).to(device)
        inputs = torch.randn(2, 9, 3, generator=generator).to(device)
        observed = torch.tensor([[True, True, False, False, True, True, False, True, True],
                                 [False, True, True, False, True, False, False, False, True]], device=device)
        return z, inputs, observed

    def test_components_reconstruct_prediction_and_both_residuals(self):
        model = self.make_model()
        z, inputs, observed = self.sequence()
        result = model(z, inputs, observed)
        self.assertEqual(result["prediction"].shape, (2, 8, 4))
        terms = result["terms"]
        self.assertEqual(set(terms), {"D", "JP", "W", "Z", "N"})
        torch.testing.assert_close(result["prediction"], z[:, :-1] + sum(terms.values()))
        torch.testing.assert_close(result["base_residual"],
                                   z[:, 1:] - z[:, :-1] - sum(terms[k] for k in ("D", "JP", "W", "Z")))
        torch.testing.assert_close(result["residual"], z[:, 1:] - result["prediction"])
        for name in ("D", "JP", "W", "N"):
            self.assertGreater(float(terms[name].detach().abs().max()), 1e-7, name)

    def test_prefix_matches_full_sequence_in_both_information_modes(self):
        z, inputs, observed = self.sequence()
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                model = self.make_model(conditional=conditional)
                full = model(z, inputs, observed)
                prefix = model(z[:, :6], inputs[:, :6], observed[:, :6])
                torch.testing.assert_close(prefix["prediction"], full["prediction"][:, :5])
                for name in full["terms"]:
                    torch.testing.assert_close(prefix["terms"][name], full["terms"][name][:, :5])

    def test_future_latents_cannot_change_prediction_of_first_changed_target(self):
        z, inputs, observed = self.sequence()
        changed = z.clone()
        changed[:, 5:] = 30 + z[:, 5:].flip(1)
        for conditional in (False, True):
            model = self.make_model(conditional=conditional)
            original = model(z, inputs, observed)["prediction"]
            altered = model(changed, inputs, observed)["prediction"]
            torch.testing.assert_close(original[:, :5], altered[:, :5])
            self.assertGreater(float((original[:, 5:] - altered[:, 5:]).detach().abs().max()), 1e-5)

    def test_strict_model_does_not_read_target_metadata_or_visibility(self):
        model = self.make_model()
        z, inputs, observed = self.sequence()
        changed_inputs, changed_observed = inputs.clone(), observed.clone()
        changed_inputs[:, 5:] += 50
        changed_observed[:, 5:] = ~changed_observed[:, 5:]
        original = model(z, inputs, observed)
        altered = model(z, changed_inputs, changed_observed)
        torch.testing.assert_close(original["prediction"][:, :5], altered["prediction"][:, :5])
        torch.testing.assert_close(original["terms"]["Z"][:, :5], altered["terms"]["Z"][:, :5])
        self.assertGreater(float((original["prediction"][:, 5:] - altered["prediction"][:, 5:]).detach().abs().max()), 1e-6)

    def test_conditional_model_reads_only_target_metadata_not_later_metadata(self):
        model = self.make_model(conditional=True)
        z, inputs, observed = self.sequence()
        changed_inputs, changed_observed = inputs.clone(), observed.clone()
        changed_inputs[:, 5:] += 50
        changed_observed[:, 5:] = ~changed_observed[:, 5:]
        original = model(z, inputs, observed)
        altered = model(z, changed_inputs, changed_observed)
        torch.testing.assert_close(original["prediction"][:, :4], altered["prediction"][:, :4])
        self.assertGreater(float((original["prediction"][:, 4] - altered["prediction"][:, 4]).detach().abs().max()), 1e-6)

    def test_residual_filter_has_one_step_delay_and_damped_impulse(self):
        model = self.make_model(active=False, events=False, forcing=False)
        with torch.no_grad():
            model.N.V.weight.zero_()
            model.N.U.weight.zero_()
            model.N.V.weight[0, 0] = 1
            model.N.U.weight[0, 0] = 1
            model.N.theta.zero_()  # rho=0.5
            model.N.omega.zero_()
        z = torch.zeros(1, 5, 4)
        z[:, 1:, 0] = 1
        result = model(z, torch.zeros(1, 5, 3), torch.ones(1, 5, dtype=torch.bool))
        torch.testing.assert_close(result["terms"]["N"][0, :, 0], torch.tensor([0., 1., .5, .25]))
        # The target at index 3 changes base residual at origin 2, not N at 2.
        changed = z.clone()
        changed[:, 3, 0] = 10
        altered = model(changed, torch.zeros(1, 5, 3), torch.ones(1, 5, dtype=torch.bool))
        torch.testing.assert_close(result["terms"]["N"][:, :3], altered["terms"]["N"][:, :3])
        self.assertGreater(float((result["terms"]["N"][:, 3] - altered["terms"]["N"][:, 3]).detach().abs().max()), 1.)

    def test_event_anchor_return_and_missing_history(self):
        model = self.make_model(active=False, conditional=True, forcing=False, residuals=False)
        z = torch.zeros(1, 5, 4)
        z[0, :, 0] = torch.tensor([2., 5., 8., 11., 14.])
        observed = torch.tensor([[True, False, False, True, True]])
        events = model(z, torch.zeros(1, 5, 3), observed)["terms"]["Z"]
        # Before the observed return, displacement from anchor 2 is 8-2=6.
        torch.testing.assert_close(events[0, :, 0], torch.tensor([0., 0., -5.4, 0.]))
        no_anchor = torch.zeros_like(observed)
        torch.testing.assert_close(model(z, torch.zeros(1, 5, 3), no_anchor)["terms"]["Z"], torch.zeros(1, 4, 4))

    def test_strict_event_is_expected_return_not_target_flag(self):
        model = self.make_model(active=False, forcing=False, residuals=False)
        z = torch.zeros(1, 5, 4)
        z[0, :, 0] = torch.tensor([2., 5., 8., 11., 14.])
        observed = torch.tensor([[True, False, False, True, True]])
        events = model(z, torch.zeros(1, 5, 3), observed)["terms"]["Z"]
        torch.testing.assert_close(events[0, :, 0], torch.tensor([0., -1.35, -2.7, 0.]))

    def test_disabled_terms_are_zero_and_cannot_read_metadata(self):
        model = self.make_model(events=False, forcing=False, residuals=False)
        z, inputs, observed = self.sequence()
        result = model(z, inputs, observed)
        for name in ("W", "Z", "N"):
            self.assertEqual(int(torch.count_nonzero(result["terms"][name])), 0)
        altered = model(z, inputs + 100, ~observed)
        torch.testing.assert_close(result["prediction"], altered["prediction"])

    def test_final_prediction_backpropagates_without_target_latent_gradient(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                model = self.make_model(device=device)
                z, inputs, observed = self.sequence(device)
                z.requires_grad_(True)
                result = model(z, inputs, observed)
                loss = (result["prediction"][:, -1] - z[:, -1].detach()).square().mean()
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                torch.testing.assert_close(z.grad[:, -1], torch.zeros_like(z.grad[:, -1]))
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_invalid_shapes_and_non_boolean_observation_are_rejected(self):
        model = self.make_model()
        z, inputs, observed = self.sequence()
        with self.assertRaises(ValueError):
            model(z[:, :1], inputs[:, :1], observed[:, :1])
        with self.assertRaises(ValueError):
            model(z, inputs[:, :-1], observed)
        with self.assertRaises((TypeError, ValueError)):
            model(z, inputs, observed.float())


if __name__ == "__main__":
    unittest.main()
