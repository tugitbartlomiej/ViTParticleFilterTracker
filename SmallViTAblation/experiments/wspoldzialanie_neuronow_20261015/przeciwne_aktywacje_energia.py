"""Additional descriptive decomposition of negative/positive pair cross terms."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from przeciwne_aktywacje import ROOT,OUT,NAMES,masks

def main():
    torch.set_num_threads(4)
    info=np.load(OUT/'indeksy.npz');keep=np.flatnonzero(info['steps']>=55000)
    vv=np.load(OUT/'V.npy',mmap_mode='r');pp=np.load(OUT/'pary.npz')
    hs=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'dane.npz') for n in NAMES]
    zz=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'Z.npy',mmap_mode='r') for n in NAMES]
    mm=[masks(h['gt']) for h in hs];rng=np.random.default_rng(262015)
    # Same ordered group enumeration as primary analysis for identical shuffles.
    perm=[{g:rng.permutation(np.flatnonzero(m)) for g,m in md.items()} for md in mm]
    rows=[];detailed=[]
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    for ti in keep:
        v=torch.tensor(vv[ti],device=dev);n=v.norm(dim=1)
        for b in range(3):
            z=torch.tensor(np.array(zz[b][info['original_index'][ti]],dtype=np.float32),device=dev)
            for name in ['final','early','control']:
                c,d=torch.tensor(pp[name],device=dev).T;dot=(v[c]*v[d]).sum(-1)
                for g in ['bbox','far']:
                    ma=np.flatnonzero(mm[b][g]);a=z[ma][:,c];bb=z[ma][:,d]
                    E=a.square()*n[c].square()+bb.square()*n[d].square()
                    cross=2*a*bb*dot
                    bs=z[perm[b][g]][:,d];ES=a.square()*n[c].square()+bs.square()*n[d].square()
                    cs=2*a*bs*dot
                    rows.append(dict(step=int(info['steps'][ti]),image=b+1,group=g,pair_set=name,
                        diag=float(E.sum()),negative_cross=float(cross.clamp_max(0).sum()),positive_cross=float(cross.clamp_min(0).sum()),
                        shuffle_diag=float(ES.sum()),shuffle_cross=float(cs.sum())))
                    if name=='final' and ti==keep[-1]:
                        ce=cross.sum(0).cpu().numpy();ee=E.sum(0).cpu().numpy()
                        for j,(c0,d0) in enumerate(pp[name]):detailed.append(dict(image=b+1,group=g,c=int(c0),d=int(d0),diag=float(ee[j]),cross=float(ce[j]),ratio=float(ce[j]/max(ee[j],1e-20))))
    df=pd.DataFrame(rows);df.to_csv(OUT/'energia_historia.csv',index=False)
    pd.DataFrame(detailed).to_csv(OUT/'energia_par_k65600.csv',index=False)
    s=df.groupby(['pair_set','image','group'])[['diag','negative_cross','positive_cross','shuffle_diag','shuffle_cross']].sum().reset_index()
    s['negative_ratio']=s.negative_cross/s.diag;s['positive_ratio']=s.positive_cross/s.diag
    s['net_ratio']=(s.negative_cross+s.positive_cross)/s.diag;s['shuffle_ratio']=s.shuffle_cross/s.shuffle_diag
    s.to_csv(OUT/'energia_podsumowanie.csv',index=False);print(s.to_string(index=False))

if __name__=='__main__':main()
