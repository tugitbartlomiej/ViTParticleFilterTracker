import unittest
import numpy as np
from global_terrain import histogram,smooth,payload

class Density(unittest.TestCase):
    def test_full_count_and_axes(self):
        xyz=np.array([[.25,.75,0],[.25,.75,0],[.75,.25,0]])
        h=histogram(xyz,(0,1),[[0,1],[0,1],[-1,1]])
        self.assertEqual(h.sum(),3);self.assertEqual(h[48,16],2);self.assertEqual(h[16,48],1)
    def test_reject_lost_particles(self):
        with self.assertRaises(ValueError):histogram(np.array([[2,0,0]]),(0,1),[[0,1],[0,1],[-1,1]])
    def test_smoothing_preserves_interior_mass(self):
        h=np.zeros((2,64,64));h[0,32,32]=10;h[1,32,32]=20
        s=smooth(h);np.testing.assert_allclose(s.sum(axis=(-1,-2)),[10,20])
    def test_real_population(self):
        d=payload(0)
        self.assertEqual(d['point_count'],900000);self.assertEqual(d['whole_observation_count'],90900000)
        self.assertEqual(d['steps'],list(range(0,501,5)))
        self.assertTrue(np.isfinite(d['fields']).all());self.assertFalse(d['sampling'])

if __name__=='__main__':unittest.main()
