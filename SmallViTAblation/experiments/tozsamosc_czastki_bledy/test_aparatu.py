"""Czy aparat zachowuje granice filmow, causalnosc i geometrie masek?

Zlamanie dowolnej wlasnosci blokuje interpretacje eksperymentu. Testy napisane,
ale NIE uruchomione na polecenie uzytkownika z 2026-09-23.
"""
import unittest

import numpy as np

from .dane import polygon_grid
from .modele import history_indices, iou_xywh, kalman_features
from .uruchom import metrics


class Aparat(unittest.TestCase):
    def test_history_does_not_bridge_gap_or_operation(self):
        d = dict(operation=np.array(['a', 'a', 'a', 'b']),
                 video=np.array(['v', 'v', 'v', 'v']), number=np.array([0, 3, 9, 12]))
        h = history_indices(d, 2, 3)
        self.assertEqual(h[1, 0], 0)
        self.assertEqual(h[2, 0], -1)
        self.assertEqual(h[3, 0], -1)

    def test_mask_is_not_box_fill(self):
        a = dict(width=20, height=20, polygons=[[0, 0, 19, 0, 0, 19]])
        weights, _ = polygon_grid(a, (2, 2))
        self.assertGreater(weights[0], .9)
        self.assertLess(weights[-1], .1)

    def test_iou(self):
        boxes = np.array([[0., 0., 2., 2.], [5., 5., 2., 2.]])
        np.testing.assert_allclose(iou_xywh(boxes, np.tile(boxes[0], (2, 1))), [1, 0])

    def test_kalman_has_no_future_access(self):
        past = np.arange(-1, 19)[:, None]
        s = np.linspace(0, 1, 20)[:, None]
        args = (past, np.array([[.8]]), np.zeros(1), np.array([[.1]]), .05)
        a = kalman_features(s, *args)
        changed = s.copy()
        changed[12:] += 100
        b = kalman_features(changed, *args)
        np.testing.assert_allclose(a[1:12], b[1:12])

    def test_nominal_covariance_does_not_depend_on_observed_values(self):
        past = np.arange(-1, 19)[:, None]
        args = (past, np.array([[.8]]), np.zeros(1), np.array([[.1]]), .05)
        a = kalman_features(np.zeros((20, 1)), *args)
        b = kalman_features(np.arange(20.)[:, None], *args)
        np.testing.assert_allclose(a[1:, 1], b[1:, 1])

    def test_auc_handles_ties(self):
        result = metrics(np.array([0, 1, 0, 1]), np.ones(4) * .5, .5)
        self.assertEqual(result['auc'], .5)


if __name__ == '__main__':
    unittest.main()
