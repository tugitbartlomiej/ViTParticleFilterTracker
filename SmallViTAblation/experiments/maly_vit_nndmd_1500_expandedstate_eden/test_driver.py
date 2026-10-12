import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import train
from state import AuxHistory,Normalizer

class DriverTests(unittest.TestCase):
    def test_actual_adam_steps_log_and_atomic_checkpoint_loop(self):
        import torch
        from model import ExpandedNNDMD,optimizer_step
        from test_expanded import aux_fixture
        aux=aux_fixture();norm=Normalizer.fit(2.3,aux,80)
        model=ExpandedNNDMD(3,native_dim=4,learned_dim=5,hidden=7,device='cpu',seed=321)
        before={k:v.detach().clone() for k,v in model.module.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);log=path/'TRAINING.jsonl'
            total_steps=0
            for batch,b in enumerate([32,28]):
                ids=np.arange(batch*32,batch*32+b)
                raw=np.random.default_rng(30+batch).normal(size=(13,b,2,4)).astype(np.float32)
                stats=optimizer_step(model,raw,20,aux,norm,80,micro_tokens=17)
                total_steps+=1
                row=train.training_record(stats,'inner_train',1,20,batch,ids,total_steps,.1,.2)
                train.append(log,row)
                train.atomic_json(path/'status.json',dict(actual_training_steps=total_steps,loss=row['loss_normalized']))
                train.save_checkpoint(path/'latest.pt',model,1,norm,{'test':True},total_steps,'inner_train',20)
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r['optimizer_step'] for r in rows],[1,2]);self.assertEqual([r['images'] for r in rows],[32,28])
            self.assertEqual([len(r['image_indices']) for r in rows],[32,28]);self.assertEqual(rows[0]['origin'],20)
            self.assertEqual(json.loads((path/'status.json').read_text())['actual_training_steps'],2)
            saved=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['optimizer_steps'],2);self.assertEqual(saved['total_optimizer_steps'],2)
            self.assertEqual(len(saved['optimizer']['state']),len(list(model.module.parameters())))
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.module.state_dict().items()))
            self.assertFalse(list(path.glob('*.tmp')))
    def test_complete_image_coverage_and_tail(self):
        batches=list(train.image_batches(1500,32,123))
        self.assertEqual(len(batches),47);self.assertEqual(len(batches[-1]),28)
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(1500))
        self.assertEqual(len(train.training_origins(80)),71);self.assertEqual(len(train.training_origins(90)),81)
        self.assertEqual(47*71,3337);self.assertEqual(47*81,3807)
        self.assertTrue(all(o+10<=80 for o in train.training_origins(80)))
        self.assertTrue(all(o+10<=90 for o in train.training_origins(90)))
    def test_early_stop_resume_rule_and_best_not_last(self):
        rows=[{'epoch':i+1,'full768':{'aggregate_rmse_state':v}} for i,v in enumerate([.5,.4,.400001,.400002])]
        self.assertTrue(train.early_stop(rows,2,1e-5))
        self.assertEqual(train.select_epoch(rows),2)
        self.assertFalse(train.early_stop(rows[:3],2,1e-5))
        rows.append({'epoch':5,'full768':{'aggregate_rmse_state':.39}})
        self.assertFalse(train.early_stop(rows,2,1e-5))
    def test_j128_axes_saved_without_permutation(self):
        p=np.arange(10*2*600*768,dtype=np.float32).reshape(10,2,600,768)
        channels=np.random.default_rng(3).choice(768,128,replace=False).tolist()
        actual=p[:,1][...,channels]
        self.assertEqual(actual.shape,(10,600,128))
        for h in [0,9]:np.testing.assert_array_equal(actual[h],p[h,1][:,channels])
    def test_weighted_global_rmse_is_not_mean_rmse(self):
        channels=list(range(128));meter=train.Metrics(10,channels)
        for b,error in [(32,1.),(28,3.)]:
            origin=np.zeros((b,2,768),np.float32);truth=np.ones((10,b,2,768),np.float32)
            pred=truth+error;proxy=np.zeros((10,b*2,59),np.float32)
            meter.add(truth,pred,origin,origin,origin,proxy,origin)
        x=np.zeros((101,59),np.float32);valid=np.zeros_like(x,bool)
        aux=AuxHistory(x,valid,x,valid,np.zeros(101))
        result=meter.finish(80,aux);expected=np.sqrt((32+28*9)/60)
        self.assertAlmostEqual(result['full768']['aggregate_rmse_state'],expected)
        self.assertAlmostEqual(result['J128']['aggregate_rmse_state'],expected)
        self.assertEqual(result['full768']['horizons'][0]['persistence_sign_all_fraction'],0.)
    def test_test_window_requires_frozen_model_before_reader(self):
        class Reader:
            def ensure_window(self,*args):raise AssertionError('Future reader called before freeze')
        with tempfile.TemporaryDirectory() as directory,patch.object(train,'HERE',Path(directory)):
            with self.assertRaises(FileNotFoundError):train.evaluate(None,Reader(),90,None,None,{},lambda **kw:None)
    def test_fake_reader_all1500_validation_origin_only(self):
        x=np.zeros((101,59),np.float32);valid=np.zeros_like(x,bool)
        aux=AuxHistory(x,valid,x,valid,np.zeros(101));norm=Normalizer.fit(1.,aux,80)
        seen=[]
        class Reader:
            def ensure_window(self,origin,width):self.origin=origin;self.width=width
            def get_history_batch(self,ids,origin):
                seen.extend(ids.tolist());raw=np.empty((13,len(ids),2,768),np.float32)
                for k,t in enumerate([origin-2,origin-1,*range(origin,origin+11)]):raw[k]=float(t)
                return raw
        def mocked_rollout(model,past,origin,aux,norm,n,micro):
            self.assertEqual(past.shape[0],3);self.assertEqual(float(past.max()),80.)
            pred=np.stack([past[-1]+k+1 for k in range(n)])
            proxy=np.zeros((10,past.shape[1]*2,59),np.float32)
            return pred,None,None,proxy
        cfg={'channels_j128':list(range(128)),'image_batch':32,'micro_tokens':1024,'prediction_example_indices':[]}
        with patch.object(train,'rollout',mocked_rollout),patch.object(train,'reconstruction',lambda m,p,*a:p[-1]):
            result=train.evaluate(None,Reader(),80,aux,norm,cfg,lambda **kw:None)
        self.assertEqual(seen,list(range(1500)));self.assertEqual(result['full768']['aggregate_rmse_state'],0.)
        self.assertEqual(result['J128']['horizons'][-1]['sign_all_fraction'],1.)
        self.assertEqual(result['full768']['horizons'][-1]['rmse_continuation'],0.)
    def test_atomic_checkpoint_carries_resume_cursor(self):
        import torch
        from model import ExpandedNNDMD
        x=np.zeros((101,1),np.float32);valid=np.ones_like(x,bool)
        aux=AuxHistory(x,valid,x,valid,np.zeros(101));norm=Normalizer.fit(1.,aux,80)
        m=ExpandedNNDMD(1,native_dim=2,learned_dim=2,hidden=3,device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';train.save_checkpoint(path,m,3,norm,{'x':1},23,'inner_train',14)
            saved=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(saved['last_completed_origin'],14);self.assertEqual(saved['phase'],'inner_train')
            self.assertEqual(saved['total_optimizer_steps'],23);self.assertFalse(Path(str(path)+'.tmp').exists())

if __name__=='__main__':unittest.main()
