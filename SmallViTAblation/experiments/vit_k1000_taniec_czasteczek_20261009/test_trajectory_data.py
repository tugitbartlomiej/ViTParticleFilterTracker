import unittest
import numpy as np

from trajectory_data import parse_images, pack_selected, selected_payload


class TinyDense:
    def __init__(self):
        self.steps = [0, 5, 10]

    def load(self):
        return {"steps": self.steps, "point_count": 6}

    def binary(self, step):
        values = np.zeros((6, 4), dtype="<f4")
        values[:, 0] = np.arange(6) + step * 10
        values[:, 1] = np.arange(6) + 100
        values[:, 2] = step
        values[:, 3] = np.arange(6) / 10
        return values.tobytes()


class TrajectoryDataTest(unittest.TestCase):
    def test_ranges_are_expanded_deduplicated_and_keep_user_order(self):
        self.assertEqual(parse_images("12, 40-42, 12, 3"), [12, 40, 41, 42, 3])

    def test_rejects_invalid_or_excessive_selection(self):
        for value in ("", "3-1", "-2", "1500", "1,a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_images(value)
        with self.assertRaises(ValueError):
            parse_images("0-24", maximum=24)

    def test_selected_particles_keep_image_patch_and_step_identity(self):
        packed, meta = pack_selected(TinyDense(), [2, 0], particles_per_image=2)
        self.assertEqual(packed.shape, (4, 3, 4))
        self.assertEqual(meta, {"images": [2, 0], "steps": [0, 5, 10], "particles": 4})
        np.testing.assert_array_equal(packed[0, :, 0], [4, 54, 104])
        np.testing.assert_array_equal(packed[1, :, 0], [5, 55, 105])
        np.testing.assert_array_equal(packed[2, :, 0], [0, 50, 100])
        np.testing.assert_array_equal(packed[3, :, 0], [1, 51, 101])

    def test_binary_payload_reports_exact_shape_and_identity(self):
        body, meta = selected_payload(TinyDense(), "2,0", particles_per_image=2)
        self.assertEqual(len(body), 4 * 3 * 4 * 4)
        self.assertEqual(meta["shape"], [4, 3, 4])
        self.assertEqual(meta["images"], [2, 0])
        self.assertFalse(meta["sampling"])
        np.testing.assert_allclose(meta["bounds"][0], [-4.2, 109.2])


if __name__ == "__main__":
    unittest.main()
