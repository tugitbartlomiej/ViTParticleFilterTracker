"""Diagnostic: where is the zero of the app field f(u) relative to the token cell? Unconstrained damped Newton in the P3 slice,
at epochs 0, 60 and 200; distances in cell units (half distance to the nearest other token in P3) and in cloud radius."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dyn',HERE/'dynamika_koloidow.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

def main():
    sys.stdout.reconfigure(encoding='utf-8');frames,hist=Path(sys.argv[1]),Path(sys.argv[2]);dev=torch.device('cuda')
    spec=importlib.util.spec_from_file_location('model',D.CONT/'Kod_eksperymentu/model.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    rows=json.loads((frames/'rows.json').read_text());imgs=[]
    for r in rows:
        with np.load(frames/Path(r['image']).name) as d:imgs.append(d['rgb'].copy())
    x_in=torch.as_tensor(np.stack(imgs),device=dev).permute(0,3,1,2).float()/255
    net=M.MalyViT().to(dev).eval().requires_grad_(False);cache={};blk=net.bloki[D.K]
    blk.norm2.register_forward_hook(lambda m,a,o:cache.__setitem__('x',a[0].detach()))
    blk.mlp.act.register_forward_hook(lambda m,a,o:cache.__setitem__('z',o.detach()))
    cks={s:(p,h) for s,p,h in D.checkpoints(hist)}
    P3=D.p3_app(torch.load(cks[218750][0],map_location='cpu',weights_only=False)['net'][f'bloki.{D.K}.mlp.fc2.weight'].double().to(dev))
    for step in (0,65640,218750):
        st=torch.load(cks[step][0],map_location='cpu',weights_only=False);net.load_state_dict(st.get('net',st),strict=True)
        with torch.inference_mode():net(x_in)
        X=cache['x'].double();Z=cache['z'].double();V=blk.mlp.fc2.weight.double();W1=blk.mlp.fc1.weight.double();b1=blk.mlp.fc1.bias.double()
        gam=blk.norm2.weight.double();bet=blk.norm2.bias.double();eps=blk.norm2.eps
        x0=X.reshape(-1,192);zb=Z.mean(1).repeat_interleave(600,0);n=x0.shape[0]
        def f_of(U):
            x=x0+U@P3;mu=x.mean(1,keepdim=True);sig=(x.var(1,unbiased=False,keepdim=True)+eps).sqrt()
            z=torch.nn.functional.gelu(((x-mu)/sig*gam+bet)@W1.T+b1);return (z-zb)@V.T@P3.T
        def jac(U,h=1e-5):
            return torch.stack([(f_of(U+h*torch.eye(3,device=dev,dtype=torch.float64)[k])-f_of(U-h*torch.eye(3,device=dev,dtype=torch.float64)[k]))/(2*h) for k in range(3)],2)
        cell=[];R=[]
        for f in range(len(rows)):
            p=(X[f]-X[f].mean(0))@P3.T;d=torch.cdist(p,p);d.fill_diagonal_(float('inf'));cell.append(d.min(1).values/2);R.append(p.pow(2).sum(1).mean().sqrt().repeat(600))
        cell=torch.cat(cell);R=torch.cat(R)
        U=torch.zeros(n,3,device=dev,dtype=torch.float64);f0=f_of(U).norm(dim=1)
        for _ in range(60):
            F=f_of(U);J=jac(U);du=torch.linalg.lstsq(J,F[:,:,None]).solution[:,:,0]
            sc=torch.clamp(cell/du.norm(dim=1).clamp_min(1e-30),max=1.)[:,None]    # damped: at most one cell per step
            U=U-sc*du
        fe=f_of(U).norm(dim=1);conv=fe<1e-6*f0
        dist=U.norm(dim=1)
        print(f'epoka {step*32/35000:5.1f}: |f(0)| / komórka mediana {float((f0/cell).median()):.1f} | zbieżne {float(conv.double().mean())*100:5.1f}% | '
              f'odległość zera: mediana {float((dist[conv]/cell[conv]).median()):.1f} komórek = {float((dist[conv]/R[conv]).median()):.2f} R | '
              f'zero w komórce {float(((dist<=cell)&conv).double().mean())*100:.1f}%')

if __name__=='__main__':main()
