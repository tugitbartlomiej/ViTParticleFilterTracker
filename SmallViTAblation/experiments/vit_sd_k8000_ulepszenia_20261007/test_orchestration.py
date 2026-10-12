import json,signal,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import orchestrate

class OrchestrationTests(unittest.TestCase):
    def run_case(self,callback):
        with tempfile.TemporaryDirectory() as tmp:
            here=Path(tmp)/'kod';out=Path(tmp)/'wyniki'
            here.mkdir();out.mkdir();(here/'CODE_SHA256.json').write_text('{}')
            (out/'diagnostyka').mkdir()
            (out/'diagnostyka/SAMPLER_SELECTION.json').write_text('{"steps":50,"spacing":"leading"}')
            previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                with patch.object(orchestrate,'__file__',str(here/'orchestrate.py')),patch.object(orchestrate,'ARMS',('A_control','B_mixed')),patch.object(sys,'argv',['orchestrate.py','--manifest','m','--vit','v','--checkpoint','c','--output',str(out)]),patch.object(orchestrate.subprocess,'run',side_effect=lambda *a,**k:callback(out)):
                    rc=orchestrate.main()
                return rc,{p.name:json.loads(p.read_text()) for p in out.glob('*.json')},(out/'STOP').exists()
            finally:
                for s,fn in previous.items():signal.signal(s,fn)
    def test_time_stop_between_arms_records_safe_resume_boundary(self):
        def finish_a(out):
            (out/'A_control').mkdir();(out/'A_control/FINAL.json').write_text('{}')
            (out/'STOP_TIME').touch()
            return type('Result',(),{'returncode':0})()
        rc,files,_=self.run_case(finish_a)
        self.assertEqual(rc,75)
        self.assertEqual(files['RESUME_BOUNDARY.json']['completed_arms'],['A_control'])
        self.assertEqual(files['RESUME_BOUNDARY.json']['next_arm'],'B_mixed')
    def test_sigterm_requests_stop_instead_of_killing_parent(self):
        def terminate(out):
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue((out/'STOP').exists())
            return type('Result',(),{'returncode':0})()
        rc,files,stopped=self.run_case(terminate)
        self.assertEqual(rc,0);self.assertTrue(stopped)
        self.assertNotIn('COMPARISON.json',files)
if __name__=='__main__':unittest.main()

