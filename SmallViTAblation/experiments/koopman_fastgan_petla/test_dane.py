import unittest
import tempfile
from pathlib import Path
import numpy as np
from dane import sampling_indices, load_pool

class DataTests(unittest.TestCase):
    def test_paired_replacement(self):
        a,_ = sampling_indices(30,40,42,3)
        b,slots = sampling_indices(30,40,42,3,30,2)
        self.assertEqual(np.flatnonzero(np.array(a)!=b).tolist(), slots)
        self.assertEqual(b.count(30),2)
        self.assertEqual(len(a),len(b))
    def test_group_leak(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.jsonl'
            p.write_text('{"frame_id":"a","group_id":"g","split":"train"}\n{"frame_id":"b","group_id":"g","split":"test"}')
            with self.assertRaises(ValueError): load_pool(p)
    def test_repeatable(self):
        self.assertEqual(sampling_indices(5,8,1,4),sampling_indices(5,8,1,4))

if __name__ == '__main__': unittest.main()
