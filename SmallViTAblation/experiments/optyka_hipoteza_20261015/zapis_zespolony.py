"""Probe a literal scalar complex-amplitude encoding of neuron contributions.

This is a representation test, not a Maxwell model. Basis is fitted at step
33300 using images 0 and 1 only. No original model or cache is modified.
"""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/optyka_hipoteza_20261015'
sys.path.insert(0,str(ROOT/'app'))
import lasery_dane as LD
from ekstrakcja import decode

def main():
    torch.set_num_threads(4)
    d=np.load(OUT/'pomiar.npz');steps=d['steps'];xpre=d['x_pre'];Uall=d['u'];ids=torch.tensor(d['ids'].reshape(-1))
    meta=json.loads((OUT/'ekstrakcja.json').read_text(encoding='utf-8'))
    weights=Path(meta['source_weight_dir']);net=LD.modul_modelu().MalyViT().double().eval().requires_grad_(False)
    def load(k):net.load_state_dict(torch.load(weights/f'k{k:05d}.pt',map_location='cpu',weights_only=True)['net'],strict=True)
    load(33300);i=int(np.flatnonzero(steps==33300)[0]);block=net.bloki[-1]
    with torch.no_grad():
        z=block.mlp.act(block.mlp.fc1(block.norm2(torch.from_numpy(xpre[i,:2].reshape(-1,192)).double())))
        V=block.mlp.fc2.weight
        covariance=(V*z.square().mean(0))@V.T
        _,Q=torch.linalg.eigh(covariance);Q=Q.flip(1)
    ranks=[2,8,32,96,192];rows=[];checks=[]
    chosen=[0,1000,10940,32820,33300,*range(43000,65001,2000),65640]
    examples=[]
    for k in chosen:
        i=int(np.flatnonzero(steps==k)[0]);load(k);block=net.bloki[-1]
        X=torch.from_numpy(xpre[i].reshape(-1,192)).double()
        with torch.no_grad():
            z=block.mlp.act(block.mlp.fc1(block.norm2(X)));V=block.mlp.fc2.weight;bias=block.mlp.fc2.bias
            U=z@V.T;Y=X+U+bias
            original=decode(net.glowa(net.norm(Y)),ids)
            amplitude=z*(V.T@Q[:,:2])[:,0]
            quadrature=z*(V.T@Q[:,:2])[:,1]
            E=torch.complex(amplitude,quadrature)
            E_sum=E.sum(-1)
            recon2=torch.stack([E_sum.real,E_sum.imag],-1)@Q[:,:2].T
            linear2=U@Q[:,:2]@Q[:,:2].T
            checks.append(dict(step=k,complex_identity_rel=float((recon2-linear2).norm()/U.norm()),
                cached_U_rel=float((U-torch.from_numpy(Uall[i].reshape(-1,192)).double()).norm()/U.norm())))
            for r in ranks:
                Ur=U@Q[:,:r]@Q[:,:r].T
                component_fraction=(z.square()@((V.T@Q[:,:r]).square().sum(-1)))/(z.square()@V.square().sum(0))
                resultant_fraction=Ur.square().sum(-1)/U.square().sum(-1)
                alt=decode(net.glowa(net.norm(X+Ur+bias)),ids)
                err=(Ur-U).norm(dim=-1)/U.norm(dim=-1).clamp_min(1e-20)
                dy=(alt-original).reshape(3,24,5)
                for im in range(3):
                    rows.append(dict(step=k,image=im,rank=r,relative_U_error_median=float(err.reshape(3,24)[im].median()),
                        relative_U_error_center=float(err.reshape(3,24)[im,0]),
                        component_power_fraction_center=float(component_fraction.reshape(3,24)[im,0]),
                        resultant_power_fraction_center=float(resultant_fraction.reshape(3,24)[im,0]),
                        center_logit_error=float(dy[im,0,0]),
                        center_box_rmse_px=float((dy[im,0,1:]*torch.tensor([480,320,480,320])).square().mean().sqrt())))
            if k==65640:
                examples= [dict(neuron=c,amplitude=float(E[48,c].abs()),phase_radians=float(E[48,c].angle())) for c in range(768)]
        # Independent central-difference check of the head susceptibility in float64.
        if k in [0,33300,65640]:
            q=Y[48:49].detach().requires_grad_(True)
            yy=decode(net.glowa(net.norm(q)),ids[48:49])
            B=torch.stack([torch.autograd.grad(yy[:,c].sum(),q,retain_graph=c<4)[0][0] for c in range(5)])
            direction=U[48]/U[48].norm();eps=1e-4*Y[48].norm()
            with torch.no_grad():
                plus=decode(net.glowa(net.norm(q+eps*direction)),ids[48:49])
                minus=decode(net.glowa(net.norm(q-eps*direction)),ids[48:49])
                fd=(plus-minus)[0]/(2*eps);jvp=B@direction
            checks[-1]['head_jvp_fd_relative']=float((fd-jvp).norm()/jvp.norm())
    out=dict(basis_fit_step=33300,basis_fit_images=[0,1],encoding='E_c=<u_c,e1>+i<u_c,e2>; phase=atan2; this is a coordinate encoding, not inferred optical phase',
         scope='Compression of final MLP increment, preserving full input residual and head; no parameter changes',rows=rows,checks=checks,
         phases_example_image3_final=examples,
         note='A scalar complex value has two real coordinates. Full 192D representation can be stored in 96 complex coordinates without gaining information.')
    (OUT/'zapis_zespolony.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    np.save(OUT/'baza_zespolona_33300.npy',Q.numpy())
    for r in ranks:
        subset=[v for v in rows if v['step']>=43000 and v['image']==2 and v['rank']==r]
        print(r,'median vector error',np.median([v['relative_U_error_center'] for v in subset]),
              'median bbox distortion px',np.median([v['center_box_rmse_px'] for v in subset]),flush=True)

if __name__=='__main__':main()
