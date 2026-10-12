"""Porównanie rzeczywistych checkpointów; wykonuje tylko obliczenia CPU."""
import argparse
from pathlib import Path
import torch
from dane import write_json

def weight_error(left,right):
    if left.keys()!=right.keys():raise AssertionError('Different state_dict keys')
    result=0.
    for key in left:
        a,b=left[key],right[key]
        if a.shape!=b.shape:raise AssertionError(f'Different shape: {key}')
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise AssertionError(f'Nonfinite state: {key}')
        if a.numel():result=max(result,float((a.double()-b.double()).abs().max()))
    return result

def equivalent(x,y):
    if isinstance(x,torch.Tensor):return x.shape==y.shape and bool(torch.allclose(x,y,atol=1e-7,rtol=1e-6))
    if isinstance(x,dict):return x.keys()==y.keys() and all(equivalent(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equivalent(a,b) for a,b in zip(x,y))
    return x==y

def verify(root):
    root=Path(root);torch.set_num_threads(2)
    control=torch.load(root/'control'/'FINAL.ckpt',map_location='cpu',weights_only=False)
    report={}
    for name in ['repeat','continuous']:
        other=torch.load(root/name/'FINAL.ckpt',map_location='cpu',weights_only=False)
        error=weight_error(control['state_dict'],other['state_dict'])
        if error>1e-6:raise AssertionError(f'Resume model differs: {name}/{error}')
        for key in ['optimizer_states','lr_schedulers','global_step','epoch','experiment_rng']:
            if not equivalent(control[key],other[key]):raise AssertionError(f'Resume state differs: {name}/{key}')
        report[name]=dict(weight_max_abs=error,optimizer_scheduler_rng_equal=True,step=other['global_step'])
        del other
    write_json(root/'VERIFY_STATES.json',report)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();print(verify(a.root))
