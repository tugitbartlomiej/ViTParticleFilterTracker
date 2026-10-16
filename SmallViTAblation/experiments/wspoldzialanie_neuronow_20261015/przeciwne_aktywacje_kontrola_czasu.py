"""Diagnostic: permute complete checkpoint states before differencing; not a p-value test."""
import numpy as np
import pandas as pd
import torch
from przeciwne_aktywacje import ROOT,OUT

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ix=np.load(OUT/'indeksy.npz');keep=ix['steps']>=33400
    z=np.load(OUT/'Z_wybrane_float32_pozne.npy',mmap_mode='r')
    v=np.load(OUT/'V.npy',mmap_mode='r')[keep]
    old=np.load(ROOT/'outputs/optyka_hipoteza_20261015/pomiar.npz')
    xp=old['x_post'][ix['original_index'][keep]].reshape(len(z),72,192)
    family=np.zeros(768,bool);family[np.unique(np.load(OUT/'pary.npz')['early'])]=True
    groups={(b+1,g):ix[g].ravel()&np.repeat(np.arange(3)==b,24) for b in range(3) for g in ['bbox','far']}
    rng=np.random.default_rng(412015);rows=[];dev='cuda' if torch.cuda.is_available() else 'cpu'
    for repeat in range(3):
        order=rng.permutation(len(z))
        for t in range(2,len(z)):
            a,b,c=order[t],order[t-1],order[t-2]
            va=torch.tensor(v[a],device=dev);vb=torch.tensor(v[b],device=dev)
            za=torch.tensor(np.asarray(z[a]),device=dev);zb=torch.tensor(np.asarray(z[b]),device=dev)
            dx=torch.tensor(xp[b]-xp[c],device=dev)
            nu=(za.square()*va.square().sum(1)+zb.square()*vb.square().sum(1)-2*za*zb*(va*vb).sum(1)).clamp_min(0).sqrt()
            dot=(dx@va.T)*za-(dx@vb.T)*zb;den=dx.norm(dim=1)[:,None]*nu
            for (im,g),mask in groups.items():
                for name,n in [('opposed_member',family),('other',~family)]:
                    ds=dot[mask][:,n];ns=den[mask][:,n]
                    rows.append(dict(repeat=repeat,image=im,group=g,neuron_group=name,weighted_cos=float(ds.sum()/ns.sum().clamp_min(1e-20)),
                        opposing_weight=float((ns*(ds<0)).sum()/ns.sum().clamp_min(1e-20))))
    df=pd.DataFrame(rows);s=df.groupby(['repeat','image','group','neuron_group'])[['weighted_cos','opposing_weight']].mean().reset_index()
    s.to_csv(OUT/'kontrola_przemieszania_stanow.csv',index=False)
    print(s.groupby(['image','group','neuron_group'])[['weighted_cos','opposing_weight']].mean().round(4).to_string())

if __name__=='__main__':main()
