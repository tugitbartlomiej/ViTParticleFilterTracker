"""Read-only checkpoint probe; writes only its own experiment outputs."""
from pathlib import Path
import sys, json, time, hashlib, argparse
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/optyka_hipoteza_20261015'
sys.path.insert(0, str(ROOT / 'app'))
import lasery_dane as LD

NAMES = ['eafb5a973be3d490', 'c584ef4647149c7c', '2cca2743e1c74b14']
FEATURES = ['x_norm','z_norm','net_norm','diag_rms','v_norm_mean',
            'ratio_power','ratio_length','effective_channels',
            'ratio_logit','ratio_box','diag_logit_rms','diag_box_rms',
            'suscept_logit','suscept_box','bias_norm']

def tokens(gt, seed):
    p = np.stack([(np.arange(600)%30+.5)*16, (np.arange(600)//30+.5)*16],1)
    lo, hi = gt[:2], gt[2:]; center = (lo+hi)/2
    dist = np.linalg.norm((p-center)/np.maximum(hi-lo,16),axis=1)
    a = np.argsort(dist, kind='stable')[:8]
    # Euclidean distance to rectangle boundary, including exterior points.
    outside = np.maximum(np.maximum(lo-p,p-hi),0)
    edge = np.linalg.norm(outside,axis=1)
    inside = ((p>=lo)&(p<=hi)).all(1)
    edge[inside] = np.minimum(p[inside]-lo,hi-p[inside]).min(1)
    edge[a] = np.inf
    b = np.argsort(edge,kind='stable')[:8]
    eligible = np.flatnonzero((np.linalg.norm(outside,axis=1)>32)&~np.isin(np.arange(600),np.r_[a,b]))
    c = np.sort(np.random.default_rng(seed).choice(eligible,8,replace=False))
    return np.r_[a,b,c], np.repeat(['center','edge','background'],8)

def decode(raw, ids):
    d = raw[...,1:].exp()*16
    cx = (ids%30+.5)*16; cy=(ids//30+.5)*16
    return torch.stack([raw[...,0],(cx-d[...,0])/480,(cy-d[...,1])/320,
                       (cx+d[...,2])/480,(cy+d[...,3])/320],-1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(1510)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    histories=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'dane.npz') for n in NAMES]
    steps=histories[0]['kroki']; assert all(np.array_equal(h['kroki'],steps) for h in histories)
    if args.limit: steps=steps[:args.limit]
    rgb=np.stack([h['rgb'] for h in histories]); gt=np.stack([h['gt'] for h in histories])
    selections=[tokens(g,1510+i) for i,g in enumerate(gt)]
    ids=np.stack([s[0] for s in selections]); roles=np.stack([s[1] for s in selections])
    K=len(steps); n=ids.size
    M=LD.modul_modelu(); net=M.MalyViT().to(dev).eval().requires_grad_(False)
    assert isinstance(net.bloki[-1].ls2,torch.nn.Identity), 'Account for LayerScale explicitly'
    pam={}
    net.bloki[-1].norm2.register_forward_pre_hook(lambda m,a:pam.__setitem__('pre',a[0].detach()))
    net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:pam.__setitem__('z',o.detach()))
    net.norm.register_forward_pre_hook(lambda m,a:pam.__setitem__('post',a[0].detach()))
    image=torch.from_numpy(rgb.copy()).permute(0,3,1,2).float().div(255).to(dev)
    it=torch.tensor(ids,device=dev); ib=torch.arange(3,device=dev)[:,None]
    xf=np.zeros((K,3,24,192),np.float32); xp=np.zeros_like(xf); unet=np.zeros_like(xf)
    feat=np.zeros((K,3,24,len(FEATURES)),np.float32)
    yy=np.zeros((K,3,24,5),np.float32); rawout=np.zeros_like(yy)
    cache=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'Z.npy',mmap_mode='r') for n in NAMES]
    checks=[]; specs=[]
    started=time.time(); weightdir=Path(histories[0]['wagi'].item())
    spectral_steps={0,1000,10940,32820,65640}
    for i,k in enumerate(steps):
        path=weightdir/f'k{int(k):05d}.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        net.load_state_dict(state.get('net',state),strict=True)
        with torch.no_grad():
            o,_=net(image)
            pre=pam['pre'][ib,it].reshape(n,192)
            post=pam['post'][ib,it].reshape(n,192)
            z=pam['z'][ib,it].reshape(n,768)
            V=net.bloki[-1].mlp.fc2.weight
            bias=net.bloki[-1].mlp.fc2.bias
            U=z@V.T
            vn=V.norm(dim=0)
            pdiag=z.square()@vn.square()
            pnet=U.square().sum(-1)
            lengths=z.abs()*vn
            lengthsum=lengths.sum(-1)
            if i%100==0 or i==K-1:
                Udirect=(z[0,:,None]*V.T).sum(0)
                direct=(z[0,:,None]*V.T).double()
                G=direct@direct.T
                offdiag=G.sum()-G.diagonal().sum()
                reference=np.stack([np.asarray(cache[b][i,ids[b]],np.float32) for b in range(3)])
                zn=z.reshape(3,24,768).cpu().numpy()
                checks.append(dict(step=int(k), residual_relative=float((post-pre-U-bias).norm()/(post-pre).norm()),
                    sum_relative=float((Udirect-U[0]).norm()/U[0].norm()),
                    cross_relative=float((offdiag-(direct.sum(0).square().sum()-direct.square().sum())).abs()/direct.square().sum()),
                    cache_rms_relative=float(np.linalg.norm(reference-zn)/np.linalg.norm(zn))))
            if int(k) in spectral_steps:
                # Best rank-2 approximation of the neuron Gram matrix, no imposed phases.
                for b in range(3):
                    C=z[b*24,:,None]*V.T
                    eigen=torch.linalg.eigvalsh(C.T@C).clamp_min(0)
                    specs.append(dict(step=int(k),image=b,rank2_vector_energy=float(eigen[-2:].sum()/eigen.sum()),
                                      rank2_gram_energy=float(eigen[-2:].square().sum()/eigen.square().sum())))
        # Exact per-token head derivatives at actual post-MLP state.
        q=post.detach().clone().requires_grad_(True)
        raw=net.glowa(net.norm(q)); y=decode(raw,it.reshape(-1))
        B=torch.stack([torch.autograd.grad(y[:,j].sum(),q,retain_graph=j<4)[0] for j in range(5)],1).detach()
        with torch.no_grad():
            contributions=(B@V)*z[:,None,:]
            diag=contributions.square().sum(-1)
            total=contributions.sum(-1).square()
            optical=[pre.norm(dim=-1),z.norm(dim=-1),pnet.sqrt(),pdiag.sqrt(),vn.mean().expand(n),
                pnet/pdiag.clamp_min(1e-20),pnet.sqrt()/lengthsum.clamp_min(1e-20),
                lengthsum.square()/pdiag.clamp_min(1e-20),
                total[:,0]/diag[:,0].clamp_min(1e-20),total[:,1:].sum(-1)/diag[:,1:].sum(-1).clamp_min(1e-20),
                diag[:,0].sqrt(),diag[:,1:].sum(-1).sqrt(),B[:,0].norm(dim=-1),B[:,1:].flatten(1).norm(dim=-1),
                bias.norm().expand(n)]
            feat[i]=torch.stack(optical,-1).reshape(3,24,-1).cpu().numpy()
            xf[i]=post.reshape(3,24,192).cpu().numpy();xp[i]=pre.reshape(3,24,192).cpu().numpy()
            unet[i]=U.reshape(3,24,192).cpu().numpy()
            yy[i]=y.detach().reshape(3,24,5).cpu().numpy(); rawout[i]=raw.detach().reshape(3,24,5).cpu().numpy()
            rawcheck=o.reshape(3,600,5)[ib,it].reshape(n,5)
            assert torch.allclose(raw.detach(),rawcheck,atol=2e-5,rtol=2e-5)
        if i%50==0 or i==K-1:
            print(f'{i+1}/{K} step={k} elapsed={time.time()-started:.1f}s',flush=True)
    suffix='_smoke' if args.limit else ''
    np.savez_compressed(OUT/f'pomiar{suffix}.npz',steps=steps,ids=ids,roles=roles,gt=gt,features=feat,
                        feature_names=np.array(FEATURES),x_pre=xp,x_post=xf,u=unet,y=yy,raw=rawout)
    meta=dict(names=NAMES,image_hashes=[hashlib.sha256(r.tobytes()).hexdigest() for r in rgb],
              source_weight_dir=str(weightdir), steps=len(steps), torch=torch.__version__,
              norm_eps=net.norm.eps,mlp_norm_eps=net.bloki[-1].norm2.eps,device=dev,
              checks=checks,spectral_checks=specs,seconds=time.time()-started,
              note='Existing model, no training; optical metrics are latent-space quantities, not electromagnetic fields.')
    (OUT/f'ekstrakcja{suffix}.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(saved=str(OUT),seconds=meta['seconds'],checks=checks),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
