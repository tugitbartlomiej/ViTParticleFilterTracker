"""Reject unsafe partial exports, missing-factor resumes and status overwrites."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import dane,analiza,eksport

class Guards(unittest.TestCase):
    def test_missing_array_with_done_marker(self):
        with tempfile.TemporaryDirectory(prefix='koopman_guard_') as root:
            p=Path(root);(p/'IMAGE_000_DONE.json').write_text('{}')
            with patch.object(dane,'OUT',p):
                with self.assertRaisesRegex(RuntimeError,'unsafe resume'):dane.array('MISSING',(2,3))
            self.assertFalse((p/'MISSING.npy').exists())
    def test_completed_refusal_preserves_status(self):
        with tempfile.TemporaryDirectory(prefix='koopman_guard_') as root:
            p=Path(root);(p/'REPORT.json').write_text('{"state":"completed"}')
            (p/'STATUS.json').write_text('{"stage":"completed"}')
            original=(p/'STATUS.json').read_bytes()
            with patch.object(analiza,'OUT',p),patch.object(analiza,'RUN_OWNED',False):
                with self.assertRaises(analiza.RefusedRun):analiza.run(SimpleNamespace())
                self.assertFalse(analiza.RUN_OWNED)
            self.assertEqual(original,(p/'STATUS.json').read_bytes())
    def test_incomplete_export_refused(self):
        with tempfile.TemporaryDirectory(prefix='koopman_guard_') as root:
            with patch.object(eksport,'OUT',Path(root)):
                with self.assertRaisesRegex(RuntimeError,'not complete'):eksport.run(0,0,'final')

if __name__=='__main__':unittest.main(verbosity=2)
