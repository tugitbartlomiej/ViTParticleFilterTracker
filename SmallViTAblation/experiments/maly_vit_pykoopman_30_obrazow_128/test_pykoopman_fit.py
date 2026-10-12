"""Numerical smoke and leakage checks on synthetic data, before real holdout."""
import unittest
import numpy as np
import pykoopman_fit as experiment


class WeightedAffineSmoke(unittest.TestCase):
    def get_function(self, name):
        self.assertTrue(hasattr(experiment, name), f"Missing required behavior: {name}")
        return getattr(experiment, name)

    def test_known_affine_operator_in_original_coordinates(self):
        fit = self.get_function("fit_model")
        rng = np.random.default_rng(712)
        x = rng.normal(size=(400, 3))
        a = np.array([[0.91, 0.12, 0], [-0.03, 0.89, 0.05], [0.02, 0, 0.93]])
        b = np.array([0.3, -0.2, 0.1])
        y = x @ a.T + b
        w = rng.uniform(0.01, 2, size=400); w /= w.sum()
        model = fit(x, y, w, 0)
        np.testing.assert_allclose(model["A"], a, atol=1e-11)
        np.testing.assert_allclose(model["b"], b, atol=1e-11)
        self.assertEqual(model["operator_raw"].shape, (4, 4))
        np.testing.assert_allclose(model["operator_raw"][-1], [0, 0, 0, 1], atol=1e-11)

    def test_weights_match_independent_lstsq_with_nonzero_bias(self):
        fit = self.get_function("fit_model")
        rng = np.random.default_rng(718)
        x = rng.normal(size=(200, 3))
        y = x @ rng.normal(size=(3, 3)) + 0.7 + rng.normal(size=(200, 3)) * 0.3
        w = rng.uniform(0.001, 1, size=200); w /= w.sum()
        model = fit(x, y, w, 0)
        design = np.c_[x, np.ones(len(x))]
        reference = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w[:, None]), rcond=None)[0].T
        np.testing.assert_allclose(np.c_[model["A"], model["b"]], reference, atol=1e-11)

    def test_regularization_is_to_identity_and_preserves_constant(self):
        fit = self.get_function("fit_model")
        rng = np.random.default_rng(721)
        x = rng.normal(size=(200, 3)); y = rng.normal(size=(200, 3)) + 0.2
        w = rng.uniform(0.001, 1, size=200); w /= w.sum()
        lam = 1e-2
        model = fit(x, y, w, lam)
        alpha = lam * np.sum(w[:, None] * x*x) / 3
        xx = np.vstack([np.c_[x, np.ones(len(x))] * np.sqrt(w[:, None]), np.c_[np.eye(3)*np.sqrt(alpha), np.zeros(3)]])
        yy = np.vstack([y * np.sqrt(w[:, None]), np.eye(3)*np.sqrt(alpha)])
        reference = np.linalg.lstsq(xx, yy, rcond=None)[0].T
        np.testing.assert_allclose(np.c_[model["A"], model["b"]], reference, atol=1e-11)
        np.testing.assert_allclose(model["operator_raw"][-1], [0, 0, 0, 1], atol=1e-11)

    def test_pair_builder_preserves_image_and_particle_identity(self):
        pairs = self.get_function("build_pairs")
        latent = np.zeros((4, 3, 2, 3))
        for t in range(4):
            for i in range(3):
                for p in range(2): latent[t,i,p] = [i, p, t]
        coverage = np.array([[1., .25], [0., 1.], [.5, .5]])
        tool = coverage > 0
        x, y, w, meta = pairs(latent, coverage, tool, [0,2], 2)
        np.testing.assert_array_equal(x[:, :2], y[:, :2])
        np.testing.assert_array_equal(y[:, 2]-x[:, 2], 1)
        self.assertEqual(meta["image_time_transitions"], 4)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertAlmostEqual(w[x[:,0]==0].sum(), .5)
        self.assertAlmostEqual(w[x[:,0]==2].sum(), .5)
        self.assertLessEqual(y[:,2].max(), 2)

    def test_forecast_uses_own_prediction_only(self):
        forecast = self.get_function("forecast")
        a = np.array([[.9, .1], [0., .8]]); b = np.array([.1,-.1])
        model = {"A":a, "b":b}
        start = np.array([[1.,2.],[-1.,3.]])
        pred = forecast(model, start, 5)
        expected = start.copy()
        for j in range(5):
            expected = expected @ a.T + b
            np.testing.assert_allclose(pred[j], expected)

    def test_metrics_persistence_and_opposite_sign(self):
        metrics = self.get_function("metrics_one")
        origin = np.zeros((2,2)); truth = np.array([[1., -1.],[2.,-2.]])
        w = np.array([.75,.25]); majority = np.array([1.,-1.])
        unchanged = metrics(origin, truth, origin, w, majority)
        self.assertAlmostEqual(unchanged["mse_skill_vs_persistence"], 0)
        self.assertAlmostEqual(unchanged["sign_accuracy"], 0)
        self.assertAlmostEqual(unchanged["majority_sign_accuracy"], 1)
        self.assertAlmostEqual(unchanged["norm_relative_error"], 1)
        self.assertAlmostEqual(unchanged["cosine_defined_mass"], 0)
        perfect = metrics(origin, truth, truth, w, majority)
        self.assertAlmostEqual(perfect["mse_skill_vs_persistence"], 1)
        self.assertAlmostEqual(perfect["sign_accuracy"], 1)
        self.assertAlmostEqual(perfect["cosine_motion"], 1)
        self.assertAlmostEqual(perfect["norm_relative_error"], 0)
        opposite = metrics(origin, truth, -truth, w, majority)
        self.assertAlmostEqual(opposite["sign_accuracy"], 0)
        self.assertAlmostEqual(opposite["cosine_motion"], -1)

    def test_all_sign_counts_exact_zero_components(self):
        import report_pykoopman as report
        self.assertTrue(hasattr(report,"sign_all"),"Missing exact-zero-inclusive comparability metric")
        true=np.array([[0.,.2,-.2],[0.,.3,-.3]])
        pred=np.array([[0.,-.2,-.2],[.1,.3,-.3]])
        actual=report.sign_all(true,pred,np.array([.75,.25]),np.array([0.,1.,-1.]))
        self.assertAlmostEqual(actual["sign_accuracy_all128"],2/3)
        self.assertAlmostEqual(actual["true_exact_zero_component_mass"],1/3)
        self.assertAlmostEqual(actual["majority_sign_accuracy_all128"],1)


if __name__ == "__main__": unittest.main(verbosity=2)
