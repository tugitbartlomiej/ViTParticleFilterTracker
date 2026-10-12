import unittest
import torch
from training_utils import EMA, learning_rate, roi_relative_error, roi_ddp_weight, final_from_checkpoint, stop_reason

class RulesTests(unittest.TestCase):
    def test_roi_ddp_invariant_to_rank_label_distribution(self):
        for counts in ([4,0,0,0],[1,1,1,1],[2,2,0,0]):
            parameter=torch.tensor(2.,requires_grad=True)
            rank_losses=[]
            for count in counts:
                local_mean=parameter.square() if count else parameter*0
                rank_losses.append(local_mean*roi_ddp_weight(4,torch.tensor(count),torch.tensor(sum(counts))))
            loss=sum(rank_losses)/4
            loss.backward()
            self.assertEqual(loss.item(),4.)
            self.assertEqual(parameter.grad.item(),4.)
    def test_completed_checkpoint_rebuilds_final(self):
        s=dict(finished=True,reason='budget_complete',new_steps=4000,best=.4,evaluation={'ok':True})
        final=final_from_checkpoint(s,'B_mixed',False)
        self.assertEqual(final['state'],'completed')
        self.assertEqual(final['new_steps'],4000)
        s['reason']='time_limit'
        with self.assertRaises(ValueError): final_from_checkpoint(s,'B_mixed',False)
    def test_roi_localization_and_missing_labels(self):
        t=torch.ones(2,600,768)
        a=t.clone(); a[:,0,:]=3
        b=torch.tensor([[0.,0.,1/30,1/20],[.5,.5,.6,.6]])
        self.assertAlmostEqual(roi_relative_error(a,t,b,torch.tensor([True,False])).item(),4.)
        self.assertEqual(roi_relative_error(a,t,b,torch.tensor([False,False])).item(),0)
    def test_roi_gradient_mask(self):
        t=torch.ones(1,600,768); a=torch.zeros_like(t,requires_grad=True)
        v=roi_relative_error(a,t,torch.tensor([[0.,0.,1/30,1/20]]),torch.tensor([True]))
        v.backward()
        self.assertTrue(a.grad[0,0].abs().sum()>0)
        self.assertEqual(a.grad[0,1:].abs().sum().item(),0)
    def test_lr_budget(self):
        self.assertEqual(learning_rate('B_mixed',2000,4000),5e-5)
        self.assertAlmostEqual(learning_rate('C_optim',0,4000),1e-5)
        self.assertAlmostEqual(learning_rate('C_optim',3999,4000),2e-6)
    def test_ema_resume_and_restore(self):
        m=torch.nn.Linear(1,1,bias=False); m.weight.data.fill_(1)
        e=EMA(m,.5); m.weight.data.fill_(3); e.update(m)
        r=EMA(m,.5); r.load_state_dict(e.state_dict())
        with r.applied(m): self.assertEqual(m.weight.item(),2)
        self.assertEqual(m.weight.item(),3)
        self.assertEqual(r.updates,1)
    def test_earlystop_requires_coverage_and_warmup(self):
        kw=dict(budget=4000,stale=4,degradation=0,
                metrics=dict(full_relative_mse=.5,cosine=.7,image_std=.2,wrong_condition_mse=.6))
        self.assertIsNone(stop_reason(1000,coverage=1,**kw))
        self.assertIsNone(stop_reason(2000,coverage=.8,**kw))
        self.assertEqual(stop_reason(2000,coverage=1,**kw),'plateau')
        self.assertEqual(stop_reason(4000,coverage=.8,**kw),'budget_complete')
if __name__=='__main__': unittest.main()
