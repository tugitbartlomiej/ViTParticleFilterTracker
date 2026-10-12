"""One-vs2/4 rank CPU Gloo proofs, actual gradient/Adam/telemetry path."""
from pathlib import Path
import json,socket,tempfile,unittest
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import model as core
import train
from state import Normalizer
from test_expanded import aux_fixture

def worker(rank,world,port,directory):
    dist.init_process_group('gloo',init_method='tcp://127.0.0.1:'+str(port),rank=rank,world_size=world,timeout=timedelta(seconds=300))
    try:
        aux=aux_fixture();norm=Normalizer.fit(2.3,aux,80)
        reference=core.ExpandedNNDMD(3,native_dim=4,learned_dim=5,hidden=7,seed=321,device='cpu')
        parallel=core.ExpandedNNDMD(3,native_dim=4,learned_dim=5,hidden=7,seed=321,device='cpu')
        reports=[]
        for step,b in enumerate([32,28]):
            raw=np.random.default_rng(70+step).normal(size=(13,b,2,4)).astype(np.float32)
            ids=np.arange(b);local=np.array_split(ids,world)[rank]
            options=dict(micro_tokens=17,rec_weight=.7,aux_weight=.2,identity_weight=.03,clip_norm=1.)
            one=core.optimizer_step(reference,raw,20,aux,norm,80,**options)
            many=core.distributed_optimizer_step(parallel,raw[:,local],20,aux,norm,80,global_images=b,**options)
            assert abs(one['loss_normalized']-many['loss_normalized'])<2e-6
            np.testing.assert_allclose(one['prediction_mse_raw'],many['prediction_mse_raw'],rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(one['gradient_norm_before_clip'],many['gradient_norm_before_clip'],rtol=2e-6,atol=2e-6)
            grad_error=0.;weight_error=0.
            for (name,a),(other,c) in zip(reference.module.named_parameters(),parallel.module.named_parameters()):
                assert name==other
                torch.testing.assert_close(a.grad,c.grad,rtol=2e-5,atol=2e-6)
                torch.testing.assert_close(a,c,rtol=2e-5,atol=5e-6)
                grad_error=max(grad_error,float((a.grad-c.grad).abs().max()))
                weight_error=max(weight_error,float((a-c).abs().max()))
            # Same origin-only future-poison proof with a genuinely updated model.
            old=aux.values[21:].copy();aux.values[21:]=np.nan
            p1=core.rollout(reference,raw[:3],20,aux,norm,n=10,micro_tokens=17)[0]
            p2=core.rollout(parallel,raw[:3],20,aux,norm,n=10,micro_tokens=17)[0]
            aux.values[21:]=old
            np.testing.assert_allclose(p1,p2,rtol=2e-5,atol=1e-5)
            assert many['images']==b and many['local_images']==len(local)
            assert many['optimizer_step']==step+1 and many['world_size']==world
            if rank==0:
                row=train.training_record(many,'distributed_proof',1,20,step,ids,step+1,.1,.2)
                train.append(Path(directory)/'TRAINING.jsonl',row)
                train.save_checkpoint(Path(directory)/'latest.pt',parallel,1,norm,{'world_size':world},step+1,'inner_train',20)
            reports.append(dict(batch_images=b,local_images=len(local),gradient_max_error=grad_error,weights_max_error=weight_error))
        # Synchronous Python failure on just one rank must raise everywhere, not hang.
        try:core.collective_error_gate('intentional rank1 failure' if rank==1 else None,'test')
        except RuntimeError as error:assert 'intentional rank1 failure' in str(error)
        else:raise AssertionError('Collective failure was not propagated')
        Path(directory,f'rank{rank}.json').write_text(json.dumps(reports),encoding='utf-8')
    finally:dist.destroy_process_group()

def driver_worker(rank,world,port,directory):
    """Real optimizer/driver with CPU Gloo and tiny network/data scheduling adapters."""
    import sys,types
    from unittest.mock import patch
    dist.init_process_group('gloo',init_method='tcp://127.0.0.1:'+str(port),rank=rank,world_size=world,timeout=timedelta(seconds=300))
    root=Path(directory);cfg=json.loads((root/'CONFIG.json').read_text())
    events=[]
    class Reader:
        def begin_epoch(self,phase,epoch):events.append([phase,epoch])
        def ensure_window(self,origin,width):
            if origin==90:assert (root/'FREEZE.json').exists(),'Future source requested before freeze'
            dist.barrier();events.append(['ensure',origin]);self.origin=origin
        def get_state(self,ids,cp):
            return (np.arange(768,dtype=np.float32)[None,None,:]*.01+np.asarray(ids)[:,None,None]*.02+cp*.03).astype(np.float32)
        def get_history_batch(self,ids,origin):
            return np.stack([self.get_state(ids,t) for t in [max(0,origin-2),max(0,origin-1),*range(origin,origin+11)]])
    module=types.ModuleType('dane');module.FrozenWindowReader=Reader
    original=core.ExpandedNNDMD
    def factory(q,**ignored):return original(q,native_dim=768,learned_dim=5,hidden=7,device='cpu',seed=cfg['seed'],
                                           lr_network=cfg['lr_network'],lr_generator=cfg['lr_generator'])
    eval_count=0
    def fake_evaluate(model,reader,origin,aux,norm,config,progress,prediction_path=None,window_prepared=False):
        nonlocal eval_count
        assert window_prepared
        if origin==90:assert (root/'FREEZE.json').exists()
        if prediction_path:np.save(prediction_path,np.zeros(1,np.float32))
        eval_count+=1
        return {n:{'aggregate_rmse_state':.1+.01*eval_count,'aggregate_rmse_persistence':.3,'horizons':[]} for n in ['full768','J128']}
    def limited_batches(count,size,seed=None):
        yield np.arange(32);yield np.arange(32,60)
    try:
        with patch.dict(sys.modules,{'dane':module}),patch.object(train,'HERE',root),patch.object(train,'initialize_runtime',lambda:(rank,world,0)),\
             patch.object(train,'ExpandedNNDMD',factory),patch.object(train,'image_batches',limited_batches),\
             patch.object(train,'training_origins',lambda last:range(1)),patch.object(train,'evaluate',fake_evaluate),\
             patch.object(train,'dataset_preflight',lambda *args:None),\
             patch.object(torch.cuda,'is_available',lambda:True),patch.object(torch.cuda,'get_device_name',lambda n:'CPU driver fixture'),\
             patch.object(torch.cuda,'is_current_stream_capturing',lambda:False),\
             patch.object(torch.cuda,'max_memory_allocated',lambda:0):
            train.main(root/'CONFIG.json')
        Path(directory,f'driver_rank{rank}.json').write_text(json.dumps(events))
    finally:
        if dist.is_initialized():dist.destroy_process_group()

class DistributedTests(unittest.TestCase):
    def test_explicit_host_transport_handles_gradient_count_and_statistics(self):
        from unittest.mock import patch
        class DeviceTensor:
            is_cuda=True
            device=torch.device('cpu')
            def __init__(self,value):self.value=value.clone()
            def detach(self):return self.value.detach()
            def copy_(self,value):self.value.copy_(value)
        for dtype in [torch.int64,torch.float32,torch.float64]:
            actual=DeviceTensor(torch.tensor([1,2,3],dtype=dtype))
            with patch.object(dist,'get_backend',lambda:'gloo'),patch.object(dist,'all_reduce',lambda value,op:value.mul_(11)):
                core.collective_sum(actual)
            torch.testing.assert_close(actual.value,torch.tensor([11,22,33],dtype=dtype))

    def test_new_collective_api_exists(self):
        self.assertTrue(hasattr(core,'distributed_optimizer_step'),'Missing image-weighted distributed step')
    def test_gloo_2_and4_equal_single_for32_and28(self):
        for world in (4,11):
            with self.subTest(world=world),tempfile.TemporaryDirectory() as directory:
                with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
                mp.spawn(worker,args=(world,port,directory),nprocs=world,join=True)
                rows=[json.loads(x) for x in Path(directory,'TRAINING.jsonl').read_text().splitlines()]
                self.assertEqual([r['images'] for r in rows],[32,28]);self.assertEqual([r['optimizer_step'] for r in rows],[1,2])
                saved=torch.load(Path(directory,'latest.pt'),map_location='cpu',weights_only=False)
                self.assertEqual(saved['optimizer_steps'],2)
                reports=[json.loads(Path(directory,f'rank{r}.json').read_text()) for r in range(world)]
                proof={'world_size':world,'cases':reports,'telemetry_rows':rows,'status':'PASS'}
                Path(__file__).with_name('PROOF_CPU_GLOO_%d.json'%world).write_text(json.dumps(proof,indent=2),encoding='utf-8')
    def test_full_driver_singlewriter_freeze_and_global_steps_gloo(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);here=Path(__file__).resolve().parent
            for name in ['model.py','state.py','train.py','dane.py','CONFIG.json','PLAN_MODELU.md','MANIFEST_1500.json','AUX_HISTORY.npz']:
                shutil.copyfile(here/name,root/name)
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            mp.spawn(driver_worker,args=(2,port,directory),nprocs=2,join=True)
            rows=[json.loads(x) for x in (root/'TRAINING.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),14);self.assertEqual([r['total_optimizer_steps'] for r in rows],list(range(1,15)))
            self.assertEqual([r['images'] for r in rows],[32,28]*7)
            self.assertEqual(json.loads((root/'SELECTION.json').read_text())['selected_epoch'],1)
            freeze=json.loads((root/'FREEZE.json').read_text());self.assertEqual(freeze['total_optimizer_steps'],14)
            self.assertEqual(freeze['final_optimizer_steps'],2)
            self.assertEqual(json.loads((root/'status.json').read_text())['phase'],'complete')
            self.assertEqual(json.loads((root/'status.json').read_text())['actual_training_steps'],14)
            self.assertFalse(list(root.glob('*.tmp')))
            for rank in (0,1):self.assertIn(['ensure',90],json.loads((root/f'driver_rank{rank}.json').read_text()))

if __name__=='__main__':unittest.main()
