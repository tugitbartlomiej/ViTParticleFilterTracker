"""Private synthetic fixtures test image/token/time order and the publish gate."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import export_dense_timeline as exporter

class ExportContract(unittest.TestCase):
    def fixture(self, root, corrupt=False):
        steps = np.arange(0, 501, 5, dtype=np.int32)
        rows=[]; expected={}
        for image in range(2):
            positions=np.empty((101,600,3), np.float32)
            positions[:,:,0]=image
            positions[:,:,1]=np.arange(600)
            positions[:,:,2]=steps[:,None]
            norms=np.broadcast_to(steps[:,None], (101,600)).astype(np.float32).copy()
            path=root/f'{image}.npz'
            np.savez(path, projected_positions=positions, movement_norm=norms, steps=steps,
                     telemetry_step=np.arange(501),telemetry_confidence=np.full(501,.9,np.float32),
                     telemetry_iou=np.full(501,.91,np.float32))
            rows.append({'path':path.name,'image_index':image,'sha256':exporter.sha(path)})
            expected[image]=positions
        if corrupt:rows[-1]['sha256']='0'*64
        source=root/'SOURCE.json'
        source.write_text(json.dumps({'status':'complete','point_count':1200,'pca_basis_sha256':exporter.BASIS_SHA,
                                     'interpolation':False,'steps':steps.tolist(),'files':list(reversed(rows))}))
        return source,expected

    def run_export(self, source, target):
        with patch.object(exporter,'COUNT',1200),patch.object(exporter,'IMAGE_COUNT',2),patch('sys.argv',[
                'export_dense_timeline.py','--source-manifest',str(source),'--output',str(target),'--steps-key','steps'
                ]),contextlib.redirect_stdout(io.StringIO()):
            exporter.main()

    def test_shuffled_shards_keep_identity_and_genuine_time(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source,expected=self.fixture(root);target=root/'export'
            self.run_export(source,target)
            manifest=json.loads((target/'manifest.json').read_text())
            self.assertEqual(manifest['steps'],list(range(0,501,5)))
            self.assertIs(manifest['interpolation'],False)
            for step in [0,5,95,100,255,500]:
                record=manifest['files'][str(step)];path=target/record['name']
                self.assertEqual(exporter.sha(path),record['sha256'])
                actual=np.fromfile(path,dtype='<f4').reshape(2,600,4)
                for image in range(2):
                    np.testing.assert_array_equal(actual[image,:,:3],expected[image][step//5])
                np.testing.assert_array_equal(actual[:,:,3],np.full((2,600),step,np.float32))

    def test_corrupt_input_never_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source,_=self.fixture(root,corrupt=True);target=root/'export'
            with self.assertRaises(ValueError):self.run_export(source,target)
            self.assertFalse((target/'manifest.json').exists())

if __name__=='__main__':unittest.main()
