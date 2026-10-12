"""Testy numeryczne CPU; uruchomienie: python -m unittest test_dynamika -v."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


# Import działa także podczas discovery z katalogu głównego repozytorium.
_SPEC = importlib.util.spec_from_file_location(
    "dynamika_under_test", Path(__file__).with_name("dynamika.py")
)
dyn = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dyn)


def trajectory():
    """Niezależnie skonstruowany, sprzężony układ afiniczny."""
    a = np.array([[0.85, -0.28], [0.25, 0.90]])
    b = np.array([0.12, -0.06])
    c = np.empty((13, 12, 2))
    c[0] = np.random.default_rng(19).normal(size=(12, 2))
    for t in range(1, len(c)):
        c[t] = c[t - 1] @ a.T + b
    return c, a, b, np.arange(len(c)) * 100, np.arange(12) < 4


class DirectionsTests(unittest.TestCase):
    def setUp(self):
        z = np.column_stack((np.arange(8) - 3.5, np.arange(8) % 2))
        self.labels = np.column_stack((z, 2 * z[:, 0], np.ones(8)))
        x = np.column_stack((z, z[:, 0] + z[:, 1], np.ones(8)))
        self.features = np.repeat(x[None], 5, axis=0)
        self.mask = np.arange(8) < 6

    def test_collinear_labels_limit_rank_and_constant_features_are_safe(self):
        basis = dyn.fit_directions(self.features, self.labels, self.mask, 3)
        self.assertEqual(basis["label_rank"], 2)
        self.assertEqual(basis["rank"], 2)
        self.assertEqual(basis["status"], "candidate")
        assert_allclose(basis["projection"].T @ basis["projection"], np.eye(2), atol=1e-12)
        self.assertEqual(basis["scale"][-1], 1.0)
        c = dyn.project(self.features, basis)
        self.assertEqual(c.shape, (5, 8, 2))
        self.assertTrue(np.isfinite(c).all())

    def test_supervised_subspace_recovers_independent_signal(self):
        # Szum czasowy w trzeciej osi jest ortogonalny do obu etykiet.
        labels = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        x = np.empty((2, 4, 3))
        x[:, :, :2] = labels
        x[0, :, 2], x[1, :, 2] = 5., -5.
        basis = dyn.fit_directions(x, labels, np.ones(4, dtype=bool), 2)
        assert_allclose(basis["projection"] @ basis["projection"].T,
                        np.diag([1., 1., 0.]), atol=1e-12)

    def test_future_and_holdout_do_not_change_any_fitted_direction(self):
        before = dyn.fit_directions(self.features, self.labels, self.mask, 3)
        changed = self.features.copy()
        changed[3:] = np.nan
        changed[:, ~self.mask] = np.nan
        labels = self.labels.copy()
        labels[~self.mask] = np.nan
        after = dyn.fit_directions(changed, labels, self.mask, 3)
        for key in ("mean", "scale", "projection", "label_mean", "label_scale"):
            assert_array_equal(before[key], after[key])

    def test_native_width_and_rank_cap(self):
        rng = np.random.default_rng(3)
        labels = rng.normal(size=(12, 10))
        x = labels @ rng.normal(size=(10, 1536))
        basis = dyn.fit_directions(x[None], labels, np.ones(12, bool), 1, rank=20)
        self.assertEqual(basis["projection"].shape, (1536, 8))

    def test_no_varying_labels_is_an_explicit_failure(self):
        with self.assertRaises(ValueError):
            dyn.fit_directions(self.features, np.ones((8, 6)), self.mask, 3)


class LabelQualityTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile(np.array([[0., -1., 1.], [0., 1., 1.],
                                       [1., -1., 1.], [1., 1., 1.]]), (3, 1))
        self.features = np.repeat(self.labels[None, :, :2], 4, axis=0)
        self.mask = np.arange(12) < 4
        self.groups = np.arange(12) // 4
        self.basis = dyn.fit_directions(self.features, self.labels, self.mask,
                                        time_stop=2, ridge=0.)

    def test_label_predictions_restore_original_units_from_full_and_projected_codes(self):
        full = dyn.predict_labels(self.features, self.basis)
        projected = dyn.predict_labels_from_codes(dyn.project(self.features, self.basis), self.basis)
        assert_allclose(full, np.broadcast_to(self.labels, full.shape), atol=1e-12)
        assert_allclose(projected, full, atol=1e-12)

    def test_holdout_quality_reports_both_readouts_without_generation_gate(self):
        report = dyn.evaluate_directions(self.features, self.labels, self.basis,
                                         image_groups=self.groups)
        self.assertEqual(report["status"], "candidate")
        self.assertFalse(report["affects_generation_gate"])
        self.assertNotIn("accepted", report)
        assert_array_equal(report["holdout_indices"], np.arange(4, 12))
        for method in ("full", "projected"):
            self.assertAlmostEqual(report[method]["presence"]["auc"], 1.)
            self.assertAlmostEqual(report[method]["presence"]["accuracy"], 1.)
            self.assertAlmostEqual(report[method]["r2"]["label_1"], 1.)
            self.assertIsNone(report[method]["r2"]["label_2"])
            self.assertEqual(len(report[method]["per_checkpoint"]), 4)
            self.assertEqual(len(report[method]["per_group"]), 2)
        json.dumps(dyn.to_jsonable(report), allow_nan=False)

    def test_quality_uses_only_requested_holdout_and_time_indices_without_refitting(self):
        original = dyn.to_jsonable(self.basis)
        self.features[:, self.mask] = np.nan
        self.features[2:] = np.nan
        self.labels[self.mask] = np.nan
        report = dyn.evaluate_directions(self.features, self.labels, self.basis,
                                         image_groups=self.groups, time_indices=[0, 1])
        self.assertAlmostEqual(report["full"]["presence"]["auc"], 1.)
        self.assertEqual(dyn.to_jsonable(self.basis), original)

    def test_single_class_auc_is_undefined_and_tied_scores_have_half_auc(self):
        self.features[:, :, 0] = .5
        report = dyn.evaluate_directions(self.features, self.labels, self.basis,
                                         image_groups=self.groups)
        self.assertAlmostEqual(report["full"]["presence"]["auc"], .5)
        self.assertAlmostEqual(report["full"]["presence"]["accuracy"], .5)
        self.labels[:, 0] = 1.
        single = dyn.evaluate_directions(self.features, self.labels, self.basis,
                                         image_groups=self.groups)
        self.assertIsNone(single["full"]["presence"]["auc"])

    def test_quality_rejects_overlap_of_source_groups(self):
        groups = self.groups.copy()
        groups[4] = 0
        with self.assertRaises(ValueError):
            dyn.evaluate_directions(self.features, self.labels, self.basis, image_groups=groups)

    def test_wrong_holdout_labels_are_reported_as_failure_without_refit(self):
        labels = self.labels.copy()
        labels[~self.mask, 0] = 1 - labels[~self.mask, 0]
        labels[~self.mask, 1] *= -1
        report = dyn.evaluate_directions(self.features, labels, self.basis,
                                         image_groups=self.groups)
        self.assertAlmostEqual(report["full"]["presence"]["auc"], 0.)
        self.assertAlmostEqual(report["full"]["presence"]["accuracy"], 0.)
        self.assertAlmostEqual(report["full"]["r2"]["label_1"], -3.)

    def test_weighted_auc_balances_groups_and_does_not_count_replicas_as_new_groups(self):
        self.features[:, 4:8, 0] = [.1, .6, .4, .9]
        self.features[:, 8:12, 0] = [1., 1., 0., 0.]
        first = dyn.evaluate_directions(self.features, self.labels, self.basis,
                                        image_groups=self.groups)
        # Trzy poprawnie uporządkowane pary spośród 4*4, bez remisów.
        self.assertAlmostEqual(first["full"]["presence"]["auc"], 3 / 16)
        self.assertAlmostEqual(first["full"]["presence"]["accuracy"], .25)
        indices = np.r_[np.arange(12), np.arange(4, 8), np.arange(4, 8)]
        second = dyn.evaluate_directions(self.features[:, indices], self.labels[indices],
                                         self.basis, image_groups=self.groups[indices])
        self.assertAlmostEqual(second["full"]["presence"]["auc"], 3 / 16)
        self.assertAlmostEqual(second["full"]["presence"]["accuracy"], .25)

    def test_rank_truncation_has_a_separate_projected_readout(self):
        labels = self.labels.copy()
        labels[:, 1] *= 3
        labels[:, 2] = labels[:, 1] / 3
        basis = dyn.fit_directions(self.features, labels, self.mask, 2, rank=1, ridge=0.)
        full = dyn.predict_labels(self.features, basis)
        reduced = dyn.predict_labels_from_codes(dyn.project(self.features, basis), basis)
        assert_allclose(full, np.broadcast_to(labels, full.shape), atol=1e-12)
        self.assertGreater(np.linalg.norm(full - reduced), .5)


class KoopmanTests(unittest.TestCase):
    def setUp(self):
        self.c, self.a, self.b, self.steps, self.mask = trajectory()

    def test_joint_affine_operator_and_multistep_forecast(self):
        model = dyn.fit_koopman(self.c, self.steps, self.mask, 6, ridge=0.)
        assert_allclose(model["A"], self.a, atol=1e-12)
        assert_allclose(model["b"], self.b, atol=1e-12)
        assert_allclose(dyn.forecast(model, self.c[5], 5), self.c[10], atol=1e-11)
        original = self.c[5].copy()
        assert_array_equal(dyn.forecast(model, self.c[5], 0), original)
        assert_array_equal(self.c[5], original)

    def test_future_and_holdout_never_affect_scaling_or_operator(self):
        before = dyn.fit_koopman(self.c, self.steps, self.mask, 6, ridge=.02)
        changed = self.c.copy()
        changed[6:] = np.nan
        changed[:, ~self.mask] = np.nan
        after = dyn.fit_koopman(changed, self.steps, self.mask, 6, ridge=.02)
        for key in ("A", "b", "mean", "scale"):
            assert_array_equal(before[key], after[key])
        assert_allclose(before["mean"], self.c[:6, self.mask].mean(axis=(0, 1)))

    def test_irregular_steps_including_future_are_rejected(self):
        for index in (3, 10):
            steps = self.steps.copy()
            steps[index] += 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                dyn.fit_koopman(self.c, steps, self.mask, 6, ridge=.01)
        for steps in (self.steps[::-1], np.zeros(len(self.steps))):
            with self.assertRaises(ValueError):
                dyn.fit_koopman(self.c, steps, self.mask, 6, ridge=.01)

    def test_affine_intercept_is_not_penalized_and_constant_axis_is_safe(self):
        c = np.array([[[2., 7.], [2., 7.]], [[5., 7.], [5., 7.]]])
        model = dyn.fit_koopman(c, [0, 10], np.ones(2, bool), 2, ridge=1e6)
        assert_allclose(dyn.forecast(model, c[0], 1), c[1], atol=1e-12)
        self.assertTrue(np.isfinite(model["scale"]).all())

    def test_invalid_horizon_ridge_mask_and_cutoff_are_rejected(self):
        model = dyn.fit_koopman(self.c, self.steps, self.mask, 6, ridge=.01)
        for horizon in (-1, 1.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                dyn.forecast(model, self.c[0], horizon)
        for ridge in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                dyn.fit_koopman(self.c, self.steps, self.mask, 6, ridge)
        for mask in (np.zeros(12, bool), np.ones(12, int), np.ones(11, bool)):
            with self.assertRaises(ValueError):
                dyn.fit_koopman(self.c, self.steps, mask, 6, .01)
        for cutoff in (1, 14, 5.5):
            with self.assertRaises(ValueError):
                dyn.fit_koopman(self.c, self.steps, self.mask, cutoff, .01)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.c, _, _, self.steps, self.mask = trajectory()
        self.groups = np.arange(12) // 2

    def validate(self, c=None, **kwargs):
        options = dict(image_groups=self.groups, horizons=(1, 3),
                       n_bootstrap=256, seed=27)
        options.update(kwargs)
        return dyn.validate(self.c if c is None else c, self.steps,
                            self.mask, 6, ridge=0., **options)

    def test_known_dynamics_beat_both_baselines_on_holdout_groups(self):
        report = self.validate()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["scores"]["koopman"]["n_groups"], 4)
        self.assertLess(report["scores"]["koopman"]["mean_vector_error"], 1e-10)
        for baseline in ("constant", "linear_trend"):
            self.assertGreater(report["comparisons"][baseline]["ci_low"], 0.)
        for fold in report["folds"]:
            self.assertGreaterEqual(fold["target"], 6)
            self.assertLess(fold["origin"], fold["target"])
            assert_allclose(fold["predictions"]["koopman"],
                            self.c[fold["target"], ~self.mask], atol=1e-11)

    def test_future_cannot_change_earlier_rolling_predictions_or_frozen_fit(self):
        before = self.validate()
        changed = self.c.copy()
        changed[9:] = 10000.
        after = self.validate(c=changed)
        assert_array_equal(before["model"]["A"], after["model"]["A"])
        for first, second in zip(before["folds"], after["folds"]):
            if first["origin"] < 9:
                for method in ("koopman", "constant", "linear_trend"):
                    assert_array_equal(first["predictions"][method],
                                       second["predictions"][method])

    def test_linear_baseline_uses_only_last_two_observations(self):
        report = self.validate(horizons=(3,))
        first = report["folds"][0]
        expected = 4 * self.c[5, ~self.mask] - 3 * self.c[4, ~self.mask]
        assert_allclose(first["predictions"]["linear_trend"], expected)

    def test_constant_and_linear_trend_ties_are_rejected(self):
        for slope in (0., .3):
            c = np.ones_like(self.c) + slope * np.arange(13)[:, None, None]
            report = self.validate(c=c)
            self.assertFalse(report["accepted"])

    def test_beating_constant_but_losing_to_trend_is_rejected(self):
        c = np.ones_like(self.c) + .3 * np.arange(13)[:, None, None]
        report = dyn.validate(c, self.steps, self.mask, 6, ridge=.01,
                              image_groups=self.groups, horizons=(1,), n_bootstrap=256)
        errors = report["scores"]
        self.assertLess(errors["koopman"]["mean_vector_error"], errors["constant"]["mean_vector_error"])
        self.assertGreater(errors["koopman"]["mean_vector_error"], errors["linear_trend"]["mean_vector_error"])
        self.assertFalse(report["accepted"])

    def test_better_means_without_group_uncertainty_support_are_rejected(self):
        c = self.c.copy()
        c[6:, 4:6] = c[5, 4:6]
        report = self.validate(c=c, horizons=(1,))
        for comparison in report["comparisons"].values():
            self.assertTrue(comparison["better_mean"])
            self.assertLess(comparison["ci_low"], 0.)
        self.assertFalse(report["accepted"])

    def test_production_timing_21_snapshots_100_update_forecasts(self):
        initial, a, b, _, fit = trajectory()
        c = np.empty((21, 12, 2))
        c[0] = initial[0]
        for t in range(1, 21):
            c[t] = c[t - 1] @ a.T + b
        report = dyn.validate(c, np.arange(21) * 20, fit, fit_stop=15,
                              ridge=0., image_groups=self.groups,
                              horizons=(5,), n_bootstrap=256)
        self.assertEqual([(f["origin"], f["target"]) for f in report["folds"]],
                         [(14, 19), (15, 20)])
        self.assertTrue(all(f["horizon_updates"] == 100 for f in report["folds"]))
        self.assertTrue(report["accepted"])

    def test_fit_and_holdout_may_not_share_a_group(self):
        groups = self.groups.copy()
        groups[4] = groups[0]
        with self.assertRaises(ValueError):
            self.validate(image_groups=groups)

    def test_one_group_has_no_claim_of_statistical_superiority(self):
        groups = self.groups.copy()
        groups[~self.mask] = 20
        self.assertFalse(self.validate(image_groups=groups)["accepted"])

    def test_group_macro_scores_ignore_replication_inside_one_group(self):
        # Powielenie obu obrazów jednej grupy nie zwiększa jej wagi.
        first = self.validate()
        indices = np.r_[np.arange(12), 4, 5, 4, 5]
        second = dyn.validate(self.c[:, indices], self.steps, self.mask[indices],
                              6, ridge=0., image_groups=self.groups[indices],
                              horizons=(1, 3), n_bootstrap=256, seed=27)
        for method in first["scores"]:
            self.assertAlmostEqual(first["scores"][method]["mean_vector_error"],
                                   second["scores"][method]["mean_vector_error"], places=12)
        assert_allclose(first["comparisons"]["constant"]["ci_low"],
                        second["comparisons"]["constant"]["ci_low"])

    def test_permutations_are_supplemental_and_reproducible(self):
        first = self.validate(n_permutations=3)
        second = self.validate(n_permutations=3)
        self.assertEqual(first["accepted"], self.validate()["accepted"])
        self.assertTrue(first["permutation"]["supplemental"])
        assert_array_equal(first["permutation"]["errors"], second["permutation"]["errors"])
        json.loads(json.dumps(dyn.to_jsonable(first), allow_nan=False))


class ArmTests(unittest.TestCase):
    def test_vector_error_cosine_and_additional_projection(self):
        result = dyn.compare_arms([[0., 0.]], [[1., 0.]], [[3., 0.]], [[2., 0.]])
        assert_allclose(result["per_image"]["control_vector_error"], [1.])
        assert_allclose(result["per_image"]["prim_vector_error"], [1.])
        assert_allclose(result["per_image"]["additional_projection"], [2.])
        assert_allclose(result["per_image"]["prim_cosine"], [1.])
        # Zgodność kierunku i dodatni rzut nie oznaczają zbliżenia do celu.
        assert_allclose(result["per_image"]["distance_gain"], [0.])

    def test_zero_cosine_is_marked_undefined_instead_of_perfect_alignment(self):
        value, valid = dyn.safe_cosine([[0., 0.], [1., 0.], [1., 0.]],
                                      [[0., 0.], [0., 0.], [-1., 0.]])
        assert_allclose(value, [0., 0., -1.])
        assert_array_equal(valid, [False, False, True])
        result = dyn.compare_arms([[0., 0.]], [[0., 0.]], [[1., 0.]], [[0., 0.]])
        self.assertFalse(result["per_image"]["projection_valid"][0])
        self.assertIsNone(result["summary"]["prim_cosine"])
        json.dumps(dyn.to_jsonable(result), allow_nan=False)

    def test_frozen_scaling_changes_metric_and_negative_projection_is_preserved(self):
        result = dyn.compare_arms([[0., 0.]], [[2., 0.]], [[0., 0.]],
                                  [[2., 0.]], scale=[2., 1.])
        assert_allclose(result["per_image"]["additional_projection"], [-1.])
        assert_allclose(result["per_image"]["distance_gain"], [-1.])
        for scale in ([0., 1.], [-1., 1.]):
            with self.assertRaises(ValueError):
                dyn.compare_arms([[0., 0.]], [[2., 0.]], [[0., 0.]],
                                 [[2., 0.]], scale=scale)

    def test_grouped_arm_uncertainty_is_paired(self):
        result = dyn.compare_arms(np.zeros((4, 2)), np.zeros((4, 2)),
                                  np.tile([1., 0.], (4, 1)), np.tile([2., 0.], (4, 1)),
                                  image_groups=["a", "a", "b", "b"], n_bootstrap=64)
        self.assertEqual(result["distance_gain_ci"]["n_groups"], 2)
        self.assertAlmostEqual(result["distance_gain_ci"]["ci_low"], 1.)


class ImportAndReportTests(unittest.TestCase):
    def test_import_uses_no_detector_or_training_dependencies(self):
        code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn', 'rfdetr', 'cv2'}:
            raise RuntimeError('Niedozwolona zależność: ' + fullname)
sys.meta_path.insert(0, Guard())
import dynamika
"""
        run = subprocess.run([sys.executable, "-B", "-c", code],
                             cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_json_helper_handles_numpy_and_nonfinite_values(self):
        report = {"a": np.array([1., np.nan, np.inf]), "b": np.int64(2),
                  "c": np.bool_(True), "d": (np.float32(-np.inf),)}
        converted = dyn.to_jsonable(report)
        self.assertEqual(json.loads(json.dumps(converted, allow_nan=False)),
                         {"a": [1., None, None], "b": 2, "c": True, "d": [None]})


if __name__ == "__main__":
    unittest.main()
