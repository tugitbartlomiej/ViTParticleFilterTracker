"""Sprawdza izolacje dopasowania; wplyw etykiety celu na jego predykcje obala cross-fitting."""
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
MODULE = ROOT / "experiments/dobor_klatek_stan_sieci/dopasowanie.py"


def synthetic_data():
    rng = np.random.default_rng(71)
    rows, pool, features, inputs, observed, initial = [], [], [], [], [], []
    for group in range(6):
        for frame in range(32):
            phase = frame / 6. + group / 4.
            signal = np.array([np.sin(phase), np.cos(phase), np.sin(phase / 2.),
                               np.cos(phase / 2.), group / 6., frame / 32.])
            feature = signal + rng.normal(0, .015, 6)
            cx, cy, width, height = .5 + .15 * signal[0], .5 + .1 * signal[1], .1, .2
            score = .2 if frame % 11 == 0 else .8
            row = dict(frame_id=f"g{group}_f{frame}", group_id=f"g{group}", video_id=f"v{group}",
                       frame_index=frame * 5, split="train")
            annotated = dict(row, native_width=100, native_height=100,
                             annotations=[dict(bbox=[100*(cx-width/2), 100*(cy-height/2),
                                                     100*width, 100*height])])
            rows.append(row)
            pool.append(annotated)
            features.append(feature)
            inputs.append([cx, cy, width, height, score, .1 * signal[2]])
            observed.append(score >= .5)
            if frame % 2 == 0:
                initial.append(row["frame_id"])
    return rows, np.asarray(features, dtype=np.float32), np.asarray(inputs, dtype=np.float32), np.asarray(observed), pool, initial


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fit_function(self):
        self.assertTrue(MODULE.is_file(), "Cross-fitted fitting is not implemented")
        return importlib.import_module("experiments.dobor_klatek_stan_sieci.dopasowanie").fit_cross_fitted

    def run_fit(self, data, artifact_dir=None, **options):
        config = dict(context=4, step=5, dim=4, folds=3, epochs=2, batch_size=64,
                      max_windows=20000, seed=42, device="cpu", artifact_dir=artifact_dir)
        config.update(options)
        return self.fit_function()(*data, **config)

    def test_group_isolation_label_independence_artifacts_and_missing_windows(self):
        data = synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            before = self.run_fit(data, artifact_dir=directory)
            self.assertEqual(before["e_obs"].shape, (192,))
            self.assertEqual(before["pred_obs_box"].shape, (192, 4))
            self.assertTrue(np.isfinite(before["e_obs"]).all())
            self.assertTrue(np.isfinite(before["pred_obs_box"]).all())
            missing = np.array([row["frame_index"] < 20 for row in data[0]])
            self.assertTrue(np.isnan(before["e_dyn"][missing]).all())
            self.assertTrue(np.isnan(before["pred_dyn_box"][missing]).all())
            self.assertTrue(np.isfinite(before["e_dyn"][~missing]).all())
            self.assertTrue(np.isfinite(before["pred_dyn_box"][~missing]).all())
            self.assertTrue(((before["pred_obs_box"] >= 0) & (before["pred_obs_box"] <= 1)).all())
            self.assertEqual(len(before["audit"]["folds"]), 3)
            for fold in before["audit"]["folds"]:
                self.assertFalse(set(fold["fit_groups"]) & set(fold["held_groups"]))
                selected = set(fold["readout_l0_ids"])
                self.assertTrue(selected <= set(data[-1]))
                self.assertTrue(selected)
                for row in data[0]:
                    if row["frame_id"] in selected:
                        self.assertIn(row["group_id"], fold["fit_groups"])
                artifact = Path(directory) / f"GLE_fold{fold['fold']}.pt"
                self.assertTrue(artifact.is_file())
                payload = torch.load(artifact, map_location="cpu", weights_only=False)
                for key in ("config", "state_dict", "pca", "input_scaling", "readout", "koopman", "audit"):
                    self.assertIn(key, payload)
                self.assertFalse(payload["config"]["conditional"])
            modified = list(data)
            modified[4] = copy.deepcopy(data[4])
            held_groups = set(before["audit"]["folds"][0]["held_groups"])
            held = np.array([row["group_id"] in held_groups for row in data[0]])
            for index in np.flatnonzero(held):
                modified[4][index]["annotations"][0]["bbox"] = [2., 4., 8., 10.]
            after = self.run_fit(tuple(modified))
            np.testing.assert_array_equal(before["pred_obs_box"][held], after["pred_obs_box"][held])
            np.testing.assert_array_equal(before["pred_dyn_box"][held], after["pred_dyn_box"][held])
            self.assertGreater(float(np.max(np.abs(before["e_obs"][held] - after["e_obs"][held]))), .01)
            # Stable prediction despite changed held labels also verifies seeded
            # training and fitting are deterministic within this CPU fixture.

    def test_default_rng_state_survives_seeded_training(self):
        data = synthetic_data()
        np.random.seed(93)
        torch.manual_seed(94)
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        self.run_fit(data, epochs=1)
        after_np = np.random.get_state()
        self.assertEqual(np_state[0], after_np[0])
        np.testing.assert_array_equal(np_state[1], after_np[1])
        self.assertEqual(np_state[2:], after_np[2:])
        torch.testing.assert_close(torch_state, torch.random.get_rng_state(), rtol=0, atol=0)

    def test_absent_complete_windows_falls_back_to_observation_readout(self):
        data = synthetic_data()
        result = self.run_fit(data, step=7, epochs=1)
        self.assertTrue(np.isfinite(result["e_obs"]).all())
        self.assertTrue(np.isnan(result["e_dyn"]).all())
        for fold in result["audit"]["folds"]:
            self.assertEqual(fold["fit_windows"], 0)

    def test_nonfinite_features_and_unavailable_l0_are_rejected(self):
        data = list(synthetic_data())
        data[1] = data[1].copy()
        data[1][0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.run_fit(tuple(data))
        data = list(synthetic_data())
        data[-1] = ["missing_frame"]
        with self.assertRaises(ValueError):
            self.run_fit(tuple(data))


if __name__ == "__main__":
    unittest.main()
