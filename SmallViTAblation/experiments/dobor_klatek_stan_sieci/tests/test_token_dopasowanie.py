"""Sprawdza cząstki per patch i izolację grup; pooling lub fit z held GT obala pilot."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE = ROOT / "experiments/dobor_klatek_stan_sieci/token_dopasowanie.py"


def synthetic_tokens():
    rng = np.random.default_rng(817)
    rows, pool, values, inputs, l0 = [], [], [], [], []
    for group in range(6):
        for frame in range(12):
            foreground = (frame // 3 + group) % 4
            cx, cy = (.25 + .5 * (foreground % 2)), (.25 + .5 * (foreground // 2))
            row = dict(frame_id=f"g{group}_f{frame}", group_id=f"g{group}", video_id=f"v{group}",
                       frame_index=frame * 5, split="train")
            rows.append(row)
            pool.append(dict(row, native_width=100, native_height=100,
                             annotations=[dict(bbox=[100*cx-5, 100*cy-5, 10., 10.])]))
            patch = rng.normal(0, .05, (4, 3))
            patch[:, 0] += np.where(np.arange(4) == foreground, 1., -1.)
            patch[:, 1] += np.sin(frame / 5. + np.arange(4))
            patch[:, 2] += group / 6.
            values.append(patch)
            inputs.append([.8, cx, cy, .1, .1, .5])
            if frame % 2 == 0:
                l0.append(row["frame_id"])
    return rows, np.asarray(values, dtype=np.float16), np.asarray(inputs, dtype=np.float32), pool, l0


class TokenFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        self.assertTrue(MODULE.is_file(), "Per-patch fitting not implemented")
        return importlib.import_module("experiments.dobor_klatek_stan_sieci.token_dopasowanie")

    def fit(self, data, artifact_dir=None, **options):
        config = dict(grid_shape=(2, 2), context=3, step=5, folds=3, epochs=1,
                      batch_size=64, max_windows=60, seed=42, device="cpu", artifact_dir=artifact_dir)
        config.update(options)
        return self.module().fit_token_cross_fitted(*data, **config)

    def test_particle_trajectory_preserves_fixed_patch_and_never_pools(self):
        module = self.module()
        tokens = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
        windows = np.array([[0, 1, 2], [2, 3, 4]])
        result = module.particle_trajectories(tokens, windows, np.array([1, 3]), np.zeros(3), np.ones(3))
        np.testing.assert_array_equal(result[0], np.array([[3, 4, 5], [15, 16, 17], [27, 28, 29]], dtype=np.float32))
        np.testing.assert_array_equal(result[1], np.array([[33, 34, 35], [45, 46, 47], [57, 58, 59]], dtype=np.float32))
        altered = tokens.copy()
        altered[:, 2] += 10000
        np.testing.assert_array_equal(result, module.particle_trajectories(altered, windows, np.array([1, 3]), np.zeros(3), np.ones(3)))

    def test_weak_targets_use_centers_and_tiny_box_has_one_nearest_patch(self):
        module = self.module()
        row = dict(native_width=100, native_height=100, annotations=[dict(bbox=[0, 0, 50, 50])])
        np.testing.assert_array_equal(module.token_targets(row, (2, 2)), [True, False, False, False])
        row["annotations"][0]["bbox"] = [87., 86., 1., 1.]
        np.testing.assert_array_equal(module.token_targets(row, (2, 2)), [False, False, False, True])

    def test_cross_fitting_bce_and_models_are_independent_of_held_labels(self):
        data = synthetic_tokens()
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            before = self.fit(data, first_dir)
            missing = np.array([row["frame_index"] < 15 for row in data[0]])
            self.assertTrue(np.isfinite(before["e_obs"]).all())
            self.assertTrue(np.isnan(before["e_dyn"][missing]).all())
            self.assertTrue(np.isfinite(before["e_dyn"][~missing]).all())
            self.assertTrue(np.isfinite(before["e_persistence"][~missing]).all())
            self.assertFalse(before["audit"]["spatial_pooling"])
            for fold in before["audit"]["folds"]:
                self.assertFalse(set(fold["fit_groups"]) & set(fold["held_groups"]))
                self.assertTrue(set(fold["readout_l0_ids"]) <= set(data[-1]))
                self.assertEqual(fold["fit_particles"], 60)
                self.assertFalse(fold["information"]["target_latent"])
                self.assertFalse(fold["information"]["target_metadata"])
            changed = list(data)
            changed[3] = copy.deepcopy(data[3])
            held_groups = set(before["audit"]["folds"][0]["held_groups"])
            mask = np.array([row["group_id"] in held_groups for row in data[0]])
            for i in np.flatnonzero(mask):
                old = changed[3][i]["annotations"][0]["bbox"]
                old[0] = 70. if old[0] < 50. else 20.
            after = self.fit(tuple(changed), second_dir)
            first = torch.load(Path(first_dir)/"GLE_fold0.pt", weights_only=False, map_location="cpu")
            second = torch.load(Path(second_dir)/"GLE_fold0.pt", weights_only=False, map_location="cpu")
            np.testing.assert_array_equal(first["readout"]["coefficients"], second["readout"]["coefficients"])
            np.testing.assert_array_equal(first["token_scaling"]["mean"], second["token_scaling"]["mean"])
            np.testing.assert_array_equal(first["koopman"]["operator"], second["koopman"]["operator"])
            for name in first["state_dict"]:
                torch.testing.assert_close(first["state_dict"][name], second["state_dict"][name], rtol=0, atol=0)
            self.assertGreater(float(np.mean(after["e_obs"][mask] - before["e_obs"][mask])), 1.)
            self.assertEqual(len(before["diagnostics"]), len(data[3]))
            self.assertTrue(all(row["scored_foreground"] == 1 and row["scored_background"] == 3 for row in before["diagnostics"]))

    def test_no_temporal_windows_retains_observation_scores_and_rng(self):
        data = synthetic_tokens()
        torch.manual_seed(13)
        saved = torch.random.get_rng_state().clone()
        result = self.fit(data, step=7)
        torch.testing.assert_close(saved, torch.random.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(np.isfinite(result["e_obs"]).all())
        self.assertTrue(np.isnan(result["e_dyn"]).all())
        self.assertTrue(all(fold["fit_particles"] == 0 for fold in result["audit"]["folds"]))


if __name__ == "__main__":
    unittest.main()
