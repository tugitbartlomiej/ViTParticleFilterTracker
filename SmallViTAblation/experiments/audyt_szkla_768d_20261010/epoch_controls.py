"""Gauge/scale sensitivity of long-run aging using FP32 epoch snapshots.
Same 8 images, fixed lag in epochs, no crossing of the step-22974 resume boundary.
"""
from pathlib import Path
import json,numpy as np
from analyze import OUT,save,film_ci
def run():
    files={ep:OUT/'precision_controls'/f'k{ep*1094:05d}.npz' for ep in range(1,61)}
    assert all(p.exists() for p in files.values())
    # Each epoch is loaded once; ~885MB per condition, no derived 768D reduction.
    x=np.empty((60,8,600,768),np.float32);frozen=np.empty_like(x)
    for ep,p in files.items():
        with np.load(p) as f:x[ep-1]=f['fp32'];frozen[ep-1]=f['frozen_bn']
        if ep%10==0:print('epoch controls loaded',ep,flush=True)
    out={}
    for name,source in [('fp32_raw',x),('fp32_unit',x),('frozen_bn_raw',frozen),('frozen_bn_unit',frozen)]:
        use_unit=name.endswith('_unit');curves=[]
        for age in [5,10,15,25,35,45,50]:
            starts=np.arange(age,age+3);ls=[];vs=[]
            for lag in [1,2,3,5,8]:
                if any(k<=21<k+lag for k in starts):continue
                a=source[starts-1].astype(np.float64);b=source[starts+lag-1].astype(np.float64)
                if use_unit:
                    a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
                    b/=np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
                v=((a-b)**2).sum(-1).mean((0,2));ls.append(lag);vs.append(v)
            v=np.stack(vs,1);curves.append(dict(age_epoch=age,origin_width_epochs=2,lag_epochs=ls,mean=v.mean(0),per_image=v,ci=film_ci(v)))
        out[name]=curves;print('epoch condition',name,flush=True)
    save('EPOCH_SCALE_BN_CONTROLS.json',out)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,name in zip(axes,['fp32_raw','fp32_unit']):
        for curve in out[name]:ax.plot(curve['lag_epochs'],curve['mean'],'o-',label=f"age {curve['age_epoch']}")
        ax.set(xlabel='Lag [epochs]',ylabel='MSD in full 768D',title=name);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Long run: scale sensitivity, FP32, equal age windows, resume excluded')
    fig.tight_layout();fig.savefig(OUT/'epoch_scale_controls.png',dpi=170);plt.close(fig)
if __name__=='__main__':run()
