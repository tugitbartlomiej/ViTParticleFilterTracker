import unittest
import numpy as np
from tool_masks import patch_membership

class MaskTiles(unittest.TestCase):
    def test_row_major_crossing(self):
        m=np.zeros((320,480),bool);m[15:18,15:18]=True
        self.assertEqual(patch_membership(m)['tool_patches'],[0,1,30,31])
    def test_contour_excludes_interior_tiles(self):
        m=np.zeros((320,480),bool);m[0:48,0:48]=True
        p=patch_membership(m)
        self.assertIn(31,p['tool_patches']);self.assertNotIn(31,p['contour_patches'])
        self.assertTrue(set(p['contour_patches'])<=set(p['tool_patches']))
    def test_empty_no_bbox_fallback(self):
        self.assertEqual(patch_membership(np.zeros((320,480),bool))['tool_patches'],[])
    def test_dimensions(self):
        with self.assertRaises(ValueError):patch_membership(np.zeros((480,320),bool))

if __name__=='__main__':unittest.main()
