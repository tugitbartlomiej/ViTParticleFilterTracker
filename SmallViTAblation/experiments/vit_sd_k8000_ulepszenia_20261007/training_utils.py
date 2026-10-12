"""Small, tested optimization rules; no data or filesystem side effects."""
import math
from contextlib import contextmanager
import torch

ARMS = ('A_control', 'B_mixed', 'C_optim', 'D_roi')
BASE_SHA = '6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7'

def learning_rate(arm, step, budget):
    if arm not in ARMS or budget <= 0 or step < 0:
        raise ValueError('Invalid arm, step or budget')
    if arm in ARMS[:2]:
        return 5e-5
    progress = min(step / max(1, budget-1), 1.0)
    return 2e-6 + .5 * (1e-5-2e-6) * (1+math.cos(math.pi*progress))

def roi_relative_error(actual, target, boxes, valid, return_count=False):
    """Intersection mask in 20x30 token grid, all 768 channels, per-image normalization.
    Invalid/missing labels contribute no ROI gradient, never a negative example.
    """
    if actual.shape != target.shape or actual.ndim != 3 or actual.shape[1:] != (600,768):
        raise ValueError('Expected matching [B,600,768] full latent')
    boxes=boxes.to(actual.device).float()
    valid=valid.to(actual.device).bool()
    if boxes.shape != (len(actual),4) or valid.shape != (len(actual),):
        raise ValueError('Wrong ROI shape')
    finite=torch.isfinite(boxes).all(1)
    valid=valid & finite & (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
    valid=valid & (boxes[:,0]>=0) & (boxes[:,1]>=0) & (boxes[:,2]<=1) & (boxes[:,3]<=1)
    safe=torch.nan_to_num(boxes).clamp(0,1)
    y=torch.arange(20,device=actual.device).view(1,20,1)/20
    x=torch.arange(30,device=actual.device).view(1,1,30)/30
    mask=((x+1/30>safe[:,0,None,None]) & (x<safe[:,2,None,None]) &
          (y+1/20>safe[:,1,None,None]) & (y<safe[:,3,None,None]))
    mask=mask.flatten(1) & valid[:,None]
    count=mask.sum(1)
    use=count>0
    err=(actual.float()-target.float()).square().mean(-1)
    energy=target.float().square().mean(-1)
    per=((err*mask).sum(1)/count.clamp_min(1)) / (
         (energy*mask).sum(1)/count.clamp_min(1)).clamp_min(1e-6)
    value=per[use].mean() if use.any() else actual.sum()*0
    return (value,use.sum()) if return_count else value

def roi_ddp_weight(world, local_count, global_count):
    """DDP averages rank gradients: restore a global mean over annotated images."""
    if world<1: raise ValueError('Invalid world size')
    return world*local_count/global_count.clamp_min(1)

def final_from_checkpoint(state,arm,smoke):
    if not state.get('finished') or state.get('reason') not in (
        'smoke_complete','quality','validation_degradation','plateau','budget_complete'):
        raise ValueError('Not a normally completed arm')
    return dict(arm=arm,state='completed',reason=state['reason'],new_steps=state['new_steps'],
                best=state['best'],evaluation=state['evaluation'],smoke=smoke)

class EMA:
    def __init__(self, model, decay=.999):
        if not 0 <= decay < 1: raise ValueError('Invalid decay')
        self.decay=decay
        self.shadow={n:p.detach().clone() for n,p in model.state_dict().items()}
        self.updates=0
    @torch.no_grad()
    def update(self, model):
        for n,p in model.state_dict().items():
            if p.is_floating_point(): self.shadow[n].lerp_(p.detach(),1-self.decay)
            else: self.shadow[n].copy_(p)
        self.updates+=1
    def state_dict(self):
        return dict(decay=self.decay,updates=self.updates,shadow=self.shadow)
    def load_state_dict(self, state):
        if state['decay']!=self.decay or self.shadow.keys()!=state['shadow'].keys():
            raise ValueError('Incompatible EMA state')
        for n,p in self.shadow.items():
            if p.shape!=state['shadow'][n].shape: raise ValueError('EMA shape mismatch')
            p.copy_(state['shadow'][n])
        self.updates=state['updates']
    @contextmanager
    def applied(self,model):
        saved={n:p.detach().clone() for n,p in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow,strict=True)
            yield
        finally:
            model.load_state_dict(saved,strict=True)

def stop_reason(step, budget, coverage, stale, degradation, metrics, smoke=False):
    if smoke and step>=budget: return 'smoke_complete'
    if step>=2000 and coverage>=.90:
        wrong=metrics.get('wrong_condition_mse')
        quality=(metrics['full_relative_mse']<=.10 and metrics['cosine']>=.90
                 and metrics['image_std']>.06 and wrong is not None
                 and wrong>1.10*metrics['full_relative_mse'])
        if quality: return 'quality'
        if degradation>=3: return 'validation_degradation'
        if stale>=3: return 'plateau'
    return 'budget_complete' if step>=budget else None
