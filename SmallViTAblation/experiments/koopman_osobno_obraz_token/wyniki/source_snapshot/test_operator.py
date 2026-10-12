"""Reject the experiment if independent units, chronological pairing or algebra fail."""
import os
os.environ['OMP_NUM_THREADS']='4'
import time,unittest,json,torch
from operator_lokalny import Operators,select_ridge

torch.set_num_threads(4)
class Tests(unittest.TestCase):
    def test_primal_dual_rollout(self):
        torch.manual_seed(4);x=torch.randn(3,18,9,dtype=torch.float64);y=x+.02*torch.randn_like(x)
        m=Operators().fit(x,y,.001);K,b=m.explicit()
        primal=torch.linalg.solve(m.xc.transpose(1,2)@m.xc+m.ridge[:,None,None]*torch.eye(9),m.xc.transpose(1,2)@m.dc)
        torch.testing.assert_close(K-torch.eye(9),primal,atol=1e-10,rtol=1e-8)
        z=torch.randn(3,5,9,dtype=torch.float64);current=z.clone()
        for h in range(1,21):
            current=current@K+b[:,None,:]
            torch.testing.assert_close(m.predict(z,h),current,atol=1e-8,rtol=1e-8)
    def test_independent_units(self):
        torch.manual_seed(5);x=torch.randn(4,15,24,dtype=torch.float64);y=x+.03*torch.randn_like(x)
        m=Operators().fit(x,y,.001);alone=Operators().fit(x[1:2],y[1:2],.001)
        torch.testing.assert_close(m.coeff[1],alone.coeff[0],atol=1e-10,rtol=1e-9)
        y[2]+=10;changed=Operators().fit(x,y,.001)
        torch.testing.assert_close(m.coeff[1],changed.coeff[1],atol=0,rtol=0)
    def test_constant_drift(self):
        x=torch.ones(3,15,12,dtype=torch.float64);delta=torch.arange(12,dtype=torch.float64)/100
        m=Operators().fit(x,x+delta,.001)
        torch.testing.assert_close(m.predict(x,20),x+20*delta)
    def test_selection_and_splits(self):
        e=torch.tensor([[1.,2.],[2.,1.],[1.,1.]],dtype=torch.float64)
        torch.testing.assert_close(select_ridge(e),torch.tensor([.001,.1,.1],dtype=torch.float64))
        self.assertEqual(list(range(40,48))[-1]+1,48)
        self.assertEqual(list(range(48,72))[-1]+1,72)
        self.assertEqual([a+20 for a in [48,50,52]],[68,70,72])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gpu_cpu_precision(self):
        torch.manual_seed(6);x=torch.randn(4,48,384,dtype=torch.float64);y=x+.01*torch.randn_like(x)
        cpu=Operators().fit(x,y,.001);gpu=Operators().fit(x.cuda(),y.cuda(),.001)
        torch.testing.assert_close(gpu.coeff.cpu(),cpu.coeff,atol=1e-9,rtol=1e-8)
        torch.testing.assert_close(gpu.predict(y.cuda(),20).cpu(),cpu.predict(y,20),atol=1e-8,rtol=1e-8)

def benchmark():
    torch.manual_seed(7);x=torch.randn(32,72,384,dtype=torch.float64);y=x+.01*torch.randn_like(x)
    results={}
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        a,b=x.to(device),y.to(device);Operators().fit(a,b,.001)
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(5):Operators().fit(a,b,.001)
        if device=='cuda':torch.cuda.synchronize()
        results[device]=(time.perf_counter()-start)/5
    print('BENCHMARK',json.dumps(results))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    if not r.wasSuccessful():raise SystemExit(1)
    benchmark()
