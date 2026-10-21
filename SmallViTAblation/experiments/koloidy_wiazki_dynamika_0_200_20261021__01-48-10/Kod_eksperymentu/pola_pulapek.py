"""Topology of the laser field around colloid traps over training (block 6, 26 checkpoints, 12 frames).

Same field as the app (app/ogniskowa.py): f(u) = P3 sum_c (z_c(x_j + P3^T u) - zbar_c) v_c, P3 = 3 principal directions of the
normalized fc2 columns of the epoch-200 checkpoint, zbar = mean over the frame's 600 tokens. Trap u* = zero of f found by Newton
inside the token cell (half side = half the distance to the nearest other token in the P3 projection, zasieg = 1). Character from
the eigenvalues of the 3x3 Jacobian: all Re<0 attracts, all Re>0 repels, mixed signs = saddle, complex pair = vortex.
Controls: the same 3x3 classification in random 3D slices, and the full 192D Jacobian (slice-independent): share of complex
eigenvalues, share of Re>0, rotation/strain ratio ||(J-J^T)/2|| / ||(J+J^T)/2||.
Usage: python pola_pulapek.py <frames folder> <history weights folder>
"""
from pathlib import Path
import csv,importlib.util,json,sys,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dyn',HERE/'dynamika_koloidow.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
NSUB=30;NRAND=10;NEWTON=25;CHUNK=96;TOL=1e-6

def classify(lam):
    # lam: (n,3) complex eigenvalues of a 3x3 Jacobian -> (vortex flag, sign class 0 attract / 1 repel / 2 saddle)
    re,im=lam.real,lam.imag;scale=lam.abs().amax(1,keepdim=True).clamp_min(1e-30)
    vortex=(im.abs()>1e-6*scale).any(1)
    neg=(re<0).all(1);pos=(re>0).all(1)
    return vortex,torch.where(neg,0,torch.where(pos,1,2))

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.backends.cuda.matmul.allow_tf32=False
    frames,hist=Path(sys.argv[1]),Path(sys.argv[2]);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec=importlib.util.spec_from_file_location('model',D.CONT/'Kod_eksperymentu/model.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    rows=json.loads((frames/'rows.json').read_text());imgs=[]
    for r in rows:
        p=frames/Path(r['image']).name;assert D.sha(p)==r['image_sha256']
        with np.load(p) as d:imgs.append(d['rgb'].copy())
    x_in=torch.as_tensor(np.stack(imgs),device=dev).permute(0,3,1,2).float()/255
    ids=torch.arange(600,device=dev);cx=(ids%30+.5)*16;cy=(ids//30+.5)*16
    tool=torch.zeros(len(rows),600,dtype=torch.bool,device=dev)
    for f,r in enumerate(rows):
        for x1,y1,x2,y2 in r['gt']:tool[f]|=(cx>=x1)&(cx<=x2)&(cy>=y1)&(cy<=y2)
    # Fixed particle sample: NSUB tool + NSUB background tokens per frame, evenly spaced.
    pick=[]
    for f in range(len(rows)):
        for m,kind in [(tool[f],1),(~tool[f],0)]:
            idx=torch.nonzero(m)[:,0];idx=idx[torch.linspace(0,len(idx)-1,min(NSUB,len(idx)),device=dev).long()]
            pick+= [(f,int(t),kind) for t in idx]
    PF=torch.tensor([p[0] for p in pick],device=dev);PT=torch.tensor([p[1] for p in pick],device=dev);PK=torch.tensor([p[2] for p in pick],device=dev)
    net=M.MalyViT().to(dev).eval().requires_grad_(False);cache={};blk=net.bloki[D.K]
    blk.norm2.register_forward_hook(lambda m,a,o:cache.__setitem__('x',a[0].detach()))
    blk.mlp.act.register_forward_hook(lambda m,a,o:cache.__setitem__('z',o.detach()))
    cks=D.checkpoints(hist)
    st200=torch.load(cks[-1][1],map_location='cpu',weights_only=False)['net']
    P3=D.p3_app(st200[f'bloki.{D.K}.mlp.fc2.weight'].double().to(dev))                       # (3,192)
    g=torch.Generator(device='cpu').manual_seed(3047)
    QR=[torch.linalg.qr(torch.randn(192,3,generator=g,dtype=torch.float64))[0].T.to(dev) for _ in range(NRAND)]
    out=[];per=[];start=time.monotonic()
    for step,path,h in cks:
        assert D.sha(path)==h
        st=torch.load(path,map_location='cpu',weights_only=False);net.load_state_dict(st.get('net',st),strict=True)
        with torch.inference_mode():net(x_in)
        X=cache['x'].double();Z=cache['z'].double()
        V=blk.mlp.fc2.weight.double();W1=blk.mlp.fc1.weight.double();b1=blk.mlp.fc1.bias.double()
        gam=blk.norm2.weight.double();bet=blk.norm2.bias.double();eps=blk.norm2.eps;A=W1*gam;Dm=192
        zbar=Z.mean(1)                                                                            # (frames,768)
        def field(x,zb,P):
            # f = P sum_c (z_c(x) - zbar_c) v_c and its 3x3 Jacobian P J P^T at points x (n,192).
            mu=x.mean(1,keepdim=True);sig=(x.var(1,unbiased=False,keepdim=True)+eps).sqrt();xh=(x-mu)/sig
            u=(xh*gam+bet)@W1.T+b1;z=torch.nn.functional.gelu(u);gp=D.gelu_prime(u)
            f=(z-zb)@V.T@P.T
            C=(A@P.T)[None]-(A.sum(1)[:,None]*(P.sum(1)[None])/Dm)[None]-(xh@A.T)[:,:,None]*(xh@P.T)[:,None,:]/Dm   # (n,768,3)
            Jp=torch.einsum('ac,nc,ncb->nab',P@V,gp,C)/sig[:,:,None]
            return f,Jp,xh,sig,gp
        x0=X[PF,PT];zb=zbar[PF]
        # Token cell in the P3 projection: half the distance to the nearest other token of the same frame.
        cell=torch.empty(len(pick),device=dev,dtype=torch.float64)
        for f in range(len(rows)):
            p=(X[f]-X[f].mean(0))@P3.T;d=torch.cdist(p,p);d.fill_diagonal_(float('inf'));nn=d.min(1).values/2
            sel=PF==f;cell[sel]=nn[PT[sel]]
        f0,J0,xh,sig,gp=field(x0,zb,P3)
        v0,c0=classify(torch.linalg.eigvals(J0))
        # Newton for the zero of f inside the cell (as in the app).
        U=torch.zeros(len(pick),3,device=dev,dtype=torch.float64);ok=torch.ones(len(pick),dtype=torch.bool,device=dev)
        for _ in range(NEWTON):
            f,Jp,_,_,_=field(x0+U@P3,zb,P3)
            step_u=torch.linalg.solve(Jp,f[:,:,None])[:,:,0];U=torch.where(ok[:,None],U-step_u,U)
            ok&=(U.abs().amax(1)<=cell)
        f,Jt,_,_,_=field(x0+U@P3,zb,P3);found=ok&(f.norm(dim=1)<=TOL*f0.norm(dim=1).clamp_min(1e-30)+1e-9)
        vt,ct=classify(torch.linalg.eigvals(Jt))
        # Random slices (control), evaluated at the colloid.
        rv,rc=[],[]
        for Q in QR:
            _,Jq,_,_,_=field(x0,zb,Q);a,b=classify(torch.linalg.eigvals(Jq));rv.append(a.double());rc.append((b==2).double())
        rv=torch.stack(rv).mean(0);rs=torch.stack(rc).mean(0)
        # Full 192D Jacobian (slice-independent), in chunks.
        fc,fp,rr,dv=[],[],[],[]
        for i in range(0,len(pick),CHUNK):
            s=slice(i,i+CHUNK);xs=xh[s];n=xs.shape[0]
            AP=A[None]-(A.sum(1)[None,:,None]/Dm)-(xs@A.T)[:,:,None]*xs[:,None,:]/Dm                 # (n,768,192)
            J=torch.einsum('dc,nc,nce->nde',V,gp[s],AP)/sig[s][:,:,None]
            lam=torch.linalg.eigvals(J.cpu()).to(dev);sc=lam.abs().amax(1,keepdim=True)
            nz=lam.abs()>1e-9*sc
            fc.append(((lam.imag.abs()>1e-6*sc)&nz).double().sum(1)/nz.double().sum(1))
            fp.append(((lam.real>1e-9*sc)&nz).double().sum(1)/nz.double().sum(1))
            Sm=(J+J.transpose(1,2))/2;Am=(J-J.transpose(1,2))/2
            rr.append(Am.flatten(1).norm(dim=1)/Sm.flatten(1).norm(dim=1));dv.append(torch.diagonal(J,dim1=1,dim2=2).sum(1)/Dm)
        fc,fp,rr,dv=[torch.cat(t) for t in (fc,fp,rr,dv)]
        e=step*32/35000;r=dict(step=step,epoch=e)
        for name,m in [('tool',PK==1),('bg',PK==0)]:
            r.update({f'{name}_P3_vortex':float(v0[m].double().mean()),f'{name}_P3_saddle':float((c0[m]==2).double().mean()),
                f'{name}_P3_attract':float((c0[m]==0).double().mean()),f'{name}_P3_repel':float((c0[m]==1).double().mean()),
                f'{name}_trap_found':float(found[m].double().mean()),
                f'{name}_trap_vortex':float(vt[m&found].double().mean()) if (m&found).any() else None,
                f'{name}_trap_saddle':float((ct[m&found]==2).double().mean()) if (m&found).any() else None,
                f'{name}_rand_vortex':float(rv[m].mean()),f'{name}_rand_saddle':float(rs[m].mean()),
                f'{name}_full_complex_share':float(fc[m].median()),f'{name}_full_repel_share':float(fp[m].median()),
                f'{name}_rotation_to_strain':float(rr[m].median()),f'{name}_divergence':float(dv[m].median())})
        out.append(r)
        per.append(np.stack([v0.cpu().numpy(),c0.cpu().numpy(),found.cpu().numpy(),vt.cpu().numpy(),ct.cpu().numpy()]))
        print(f"{e:6.1f} | P3 wir T/B {r['tool_P3_vortex']:.2f}/{r['bg_P3_vortex']:.2f} siodło {r['tool_P3_saddle']:.2f}/{r['bg_P3_saddle']:.2f} | "
              f"losowe wir {r['tool_rand_vortex']:.2f}/{r['bg_rand_vortex']:.2f} | pułapka w komórce {r['tool_trap_found']:.2f}/{r['bg_trap_found']:.2f} | "
              f"192D zespolone {r['tool_full_complex_share']:.2f}/{r['bg_full_complex_share']:.2f} rot/strain {r['tool_rotation_to_strain']:.2f}/{r['bg_rotation_to_strain']:.2f} "
              f"| {time.monotonic()-start:.0f}s",flush=True)
    with open(D.OUT/'POLA_PULAPEK.csv','w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    np.savez_compressed(D.OUT/'POLA_PULAPEK_CZASTKI.npz',klasy=np.stack(per),frame=PF.cpu().numpy(),token=PT.cpu().numpy(),tool=PK.cpu().numpy(),
        steps=np.array([s for s,_,_ in cks]),opis=np.array('per checkpoint: P3 vortex at colloid, P3 class at colloid (0 attract,1 repel,2 saddle), trap found, vortex at trap, class at trap'))

if __name__=='__main__':main()
