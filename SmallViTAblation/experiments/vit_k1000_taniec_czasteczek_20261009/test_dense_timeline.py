"""Small synthetic fixtures verify the gate without fabricating research states."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import dense_timeline as mod

class DenseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.count=patch.object(mod,'COUNT',2);self.count.start()
        self.manifest={'status':'complete','point_count':2,'pca_basis_sha256':mod.BASIS_SHA,'stride':5,'steps':list(range(0,501,5)),'interpolation':False,'files':{},'stats':{},'bounds':{'movement':[[0,1]]*3,'natural':[[0,1]]*3},'norm_scale':500.,'heat_scale':500.}
        for step in self.manifest['steps']:
            raw=np.array([[step,1,2,step],[step+1,3,4,step]],dtype='<f4').tobytes();name=f'step_{step:03d}.bin';(self.root/name).write_bytes(raw)
            self.manifest['files'][str(step)]={'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};self.manifest['stats'][str(step)]={'both90':0,'confidence':.1,'iou':.2,'norm_median':float(step),'norm_max':float(step),'heatmap':[[float(step)]]}
        self.save();self.reader=mod.DenseTimeline(self.root)
    def save(self):(self.root/'manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')
    def tearDown(self):self.reader.binary.cache_clear();self.count.stop();self.tmp.cleanup()
    def test_complete_real_timeline(self):
        self.assertEqual(len(self.reader.load()['steps']),101)
        self.assertEqual(np.frombuffer(self.reader.binary(5),dtype='<f4')[0],5)
        self.assertEqual(self.reader.frame(5,True)['count'],2)
    def test_missing_state_refused(self):
        (self.root/'step_005.bin').unlink()
        with self.assertRaises(ValueError):self.reader.load()
    def test_tampered_state_refused(self):
        (self.root/'step_005.bin').write_bytes(b'x'*32)
        with self.assertRaises(ValueError):self.reader.binary(5)
    def test_interpolation_refused(self):
        self.manifest['interpolation']=True;self.save()
        with self.assertRaises(ValueError):self.reader.load()
    def test_missing_steps_refused(self):
        self.manifest['steps'].remove(5);self.save()
        with self.assertRaises(ValueError):self.reader.load()
    def test_changed_basis_refused(self):
        self.manifest['pca_basis_sha256']='another basis';self.save()
        with self.assertRaises(ValueError):self.reader.load()
    def test_path_escape_refused(self):
        with self.assertRaises(ValueError):self.reader.path('../other.bin')
    def test_bounded_server_cache(self):
        for step in range(0,101,5):self.reader.binary(step)
        self.assertLessEqual(self.reader.binary.cache_info().currsize,4)

if __name__=='__main__':unittest.main()
