"""Block 6 as an optical system over training (step 0 to epoch 200): 600 colloids per frame (tokens, state before the MLP),
768 laser beams (MLP neurons: direction/power = fc2 column, intensity = GELU output) and traps (local stiffness of the force field).

Force on colloid j: F_j = sum_c z_jc v_c (fc2 bias excluded). Positions: x_j - frame mean, the same as the app (lasery_dane.py),
but in full 192D. Traps: J = dF/dx at x_j (exact, through LayerNorm); trace(J)/192 < 0 = converging field, negative
eigenvalues of (J+J^T)/2 = restoring directions.
Usage: python dynamika_koloidow.py <frames folder with rows.json> <history weights folder k00000.pt..k55000.pt>
"""
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
import torch

EXP=Path(__file__).resolve().parents[1]
BASE=EXP.parent
CONT=BASE/'vit_kontynuacja_do_200epok_20261020__19-06-42'
OUT=EXP/'Wyniki/Dane';OUT.mkdir(parents=True,exist_ok=True)
HIST_SHA={0:'c8d4147c0e55036e7348f05da2e24f58833cd09481e8a590aa3511a8736b4e1d',300:'a7dd5d2089949c41f2d2a87733aac6ebd972f42d2405692ac86cbb916a2ef69e',
    1000:'5acd1a6fc869d21d7eaac26cac5ee8664fbbf085e1faabbdee030f8f710e3595',3000:'276ec42c5a1bcee8189a9e18be48edb953d2d7389f2b6f74880794a6aa85ea31',
    6000:'586c59dec22f7fa4eb4a51507b165a6bc4eb1d65479775cf43c0cf43c06f621d',10000:'8c51b678088dc83ef0c5f17491e8bac103caca3756ae53d217120ffcf6f56a8d',
    16000:'c73270ed117a72b0cac3270b0962f91cff7638e355c0923da89f4bcfbbab342a',24000:'337e137ba089769e7ddf9179515463f13e1b30acde6beac3f3df4b750c97f3e1',
    33000:'b42eb496b4826c53d8b917ac934f73c0ac1d4d5439cdf461355b6927ebebb2b6',44000:'0e606b4a7b1a3d645f6d474bbfc507df5432bb886a6caddae5019d9b64d50c8f',
    55000:'9578761b4bb400c8bf0e374b9d4ad655edd2a684c4b41eae91a14dd1ecc0e967'}
CONT_STEPS=[65640,76563,87500,98438,109375,120313,131250,142188,153125,164063,175000,185938,196875,207813,218750]
K=5;NSUB=40;KNN=10
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def checkpoints(hist):
    out=[(s,Path(hist)/f'k{s:05d}.pt',HIST_SHA[s]) for s in sorted(HIST_SHA)]
    log={json.loads(l.split('CHECKPOINT ',1)[1])['step']:json.loads(l.split('CHECKPOINT ',1)[1])['sha256']
         for l in (CONT/'Wyniki/Dane/logi/train_1907130.log').read_text().splitlines() if l.startswith('CHECKPOINT ')}
    out+=[(s,CONT/f'Wyniki/Dane/checkpoints/k{s:06d}.pt',log[s]) for s in CONT_STEPS]
    return out

def gelu_prime(u):return 0.5*(1+torch.erf(u/2**.5))+u*torch.exp(-u*u/2)/(2*torch.pi)**.5

def p3_app(V):
    Vn=V/V.norm(dim=0,keepdim=True).clamp_min(1e-12)
    U,_,_=torch.linalg.svd(Vn-Vn.mean(1,keepdim=True),full_matrices=False)
    return U[:,:3].T

def knn(X,k):
    d=torch.cdist(X,X);d.fill_diagonal_(float('inf'));return d.topk(k,largest=False).indices

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    frames,hist=Path(sys.argv[1]),Path(sys.argv[2]);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec=importlib.util.spec_from_file_location('model',CONT/'Kod_eksperymentu/model.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    assert sha(CONT/'Kod_eksperymentu/model.py')=='0570c957a23a75283102806160fa197c5ba66c27776d05b27a3ed461929da57a'
    rows=json.loads((frames/'rows.json').read_text());imgs=[]
    for r in rows:
        p=frames/Path(r['image']).name;assert sha(p)==r['image_sha256']
        with np.load(p) as d:imgs.append(d['rgb'].copy())
    x_in=torch.as_tensor(np.stack(imgs),device=dev).permute(0,3,1,2).float()/255
    ids=torch.arange(600,device=dev);cx=(ids%30+.5)*16;cy=(ids//30+.5)*16
    tool=torch.zeros(len(rows),600,dtype=torch.bool,device=dev)
    for f,r in enumerate(rows):
        for x1,y1,x2,y2 in r['gt']:tool[f]|=(cx>=x1)&(cx<=x2)&(cy>=y1)&(cy<=y2)
    net=M.MalyViT().to(dev).eval().requires_grad_(False);cache={}
    blk=net.bloki[K];blk.norm2.register_forward_hook(lambda m,a,o:cache.__setitem__('x',a[0].detach()))
    blk.mlp.act.register_forward_hook(lambda m,a,o:cache.__setitem__('z',o.detach()))
    blk.mlp.fc1.register_forward_hook(lambda m,a,o:cache.__setitem__('u',o.detach()))
    cks=checkpoints(hist);inputs={}
    st200=torch.load(cks[-1][1],map_location='cpu',weights_only=False)['net']
    P3=p3_app(st200[f'bloki.{K}.mlp.fc2.weight'].double().to(dev))
    rec=[];pos={};prevV=None;prev_step=None;start=time.monotonic()
    for step,path,h in cks:
        assert sha(path)==h,'checkpoint changed: '+str(path);inputs[str(path)]=h
        st=torch.load(path,map_location='cpu',weights_only=False);st=st.get('net',st);net.load_state_dict(st,strict=True)
        with torch.inference_mode():net(x_in)
        X=cache['x'].double();Z=cache['z'].double();U=cache['u'].double()
        V=blk.mlp.fc2.weight.double();W1=blk.mlp.fc1.weight.double();g=blk.norm2.weight.double();eps=blk.norm2.eps
        A=W1*g;D=192;vnorm=V.norm(dim=0)
        c0=(A*V.T).sum(1)-A.sum(1)*V.sum(0)/D
        e=step*32/35000;r=dict(step=step,epoch=e)
        # Beams (lasers): power, bundle dimensionality, rotation since the previous checkpoint.
        Vh=V/vnorm.clamp_min(1e-12);ev=torch.linalg.eigvalsh(Vh@Vh.T)
        r.update(beam_power_mean=float(vnorm.mean()),beam_power_top10_share=float(vnorm.sort(descending=True).values[:77].sum()/vnorm.sum()),
            beam_bundle_dims=float(ev.sum()**2/(ev**2).sum()))
        if prevV is not None:
            ang=torch.rad2deg(torch.arccos((Vh*prevV).sum(0).clamp(-1,1)))
            r.update(beam_rotation_deg_per_epoch=float(ang.mean()/(e-prev_step*32/35000)),beam_rotation_median_deg=float(ang.median()))
        else:r.update(beam_rotation_deg_per_epoch=None,beam_rotation_median_deg=None)
        prevV=Vh;prev_step=step
        acc={k:[] for k in ['R','token_norm','sep','sep_visible_P3','tool_outside','cloud_dims','kick_rel','coherence','sort_tool','sort_bg',
            'restore_tool','restore_bg','active_beams','pull_power_share','div_tool','div_bg','neg_dirs_tool','neg_dirs_bg','stiff_neg_tool','stiff_neg_bg']}
        pos[step]=[]
        for f in range(len(rows)):
            x=X[f];z=Z[f];u=U[f];m=tool[f];xc=x-x.mean(0);R=xc.pow(2).sum(1).mean().sqrt()
            pos[step].append((xc/R).float().cpu())
            mt,mb=xc[m].mean(0),xc[~m].mean(0);dsep=mt-mb
            dist_bg=(xc-mb).norm(dim=1);q95=torch.quantile(dist_bg[~m],.95)
            cev=torch.linalg.eigvalsh(torch.cov(xc.T))
            F=z@V.T;s=z.abs()*vnorm;Fn=F.norm(dim=1)
            cos=lambda a,b:(a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-30)
            own=torch.where(m[:,None],mt,mb)-xc
            act=s>.05*s.max(1,keepdim=True).values
            # Exact trace of dF/dx through LayerNorm (divergence of the force field per dimension).
            mu=x.mean(1,keepdim=True);sig=(x.var(1,unbiased=False,keepdim=True)+eps).sqrt();xh=(x-mu)/sig
            gp=gelu_prime(u);trJ=(gp*(c0-(xh@A.T)*(xh@V)/D)).sum(1)/sig[:,0]
            acc['R'].append(R);acc['token_norm'].append(x.norm(dim=1).mean())
            acc['sep'].append(dsep.norm()/R);acc['sep_visible_P3'].append((P3@dsep).norm()/dsep.norm())
            acc['tool_outside'].append((dist_bg[m]>q95).float().mean());acc['cloud_dims'].append(cev.sum()**2/(cev**2).sum())
            acc['kick_rel'].append(Fn.mean()/R);acc['coherence'].append((Fn/s.sum(1)).median())
            acc['sort_tool'].append(cos(F[m],dsep[None]).mean());acc['sort_bg'].append(cos(F[~m],-dsep[None]).mean())
            acc['restore_tool'].append(cos(F[m],own[m]).mean());acc['restore_bg'].append(cos(F[~m],own[~m]).mean())
            acc['active_beams'].append(act.sum(1).float().mean());acc['pull_power_share'].append((s*(z<0)).sum(1).div(s.sum(1)).mean())
            acc['div_tool'].append(trJ[m].mean()/D);acc['div_bg'].append(trJ[~m].mean()/D)
            # Full Jacobian for a subsample: share of restoring directions and their stiffness.
            for name,mask in [('tool',m),('bg',~m)]:
                idx=torch.nonzero(mask)[:,0];idx=idx[torch.linspace(0,len(idx)-1,min(NSUB,len(idx)),device=dev).long()]
                P=torch.eye(D,device=dev,dtype=torch.float64)[None]-1/D-xh[idx,:,None]*xh[idx,None,:]/D
                J=torch.einsum('dc,tc,ce,tef->tdf',V,gp[idx],A,P)/sig[idx,:,None]
                lam=torch.linalg.eigvalsh((J+J.transpose(1,2))/2)
                acc[f'neg_dirs_{name}'].append((lam<0).double().mean());acc[f'stiff_neg_{name}'].append(lam.clamp(max=0).sum(1).div((lam<0).sum(1).clamp_min(1)).mean())
        r.update({k:float(torch.stack([torch.as_tensor(v) for v in vs]).mean()) for k,vs in acc.items()})
        rec.append(r);print('STEP',step,round(e,2),round(time.monotonic()-start,1),flush=True)
    csvwrite('METRYKI_CHECKPOINTY.csv',rec)
    # Motion between checkpoints: normalized positions, with and without removing a rigid rotation (orthogonal Procrustes).
    steps=[s for s,_,_ in cks];moves=[]
    def disp(a,b,aligned):
        if aligned:
            Uo,_,Vo=torch.linalg.svd(a.T@b);a=a@(Uo@Vo)
        return (b-a).norm(dim=1)
    for i in range(1,len(steps)):
        s0,s1=steps[i-1],steps[i];de=(s1-s0)*32/35000;row=dict(from_epoch=s0*32/35000,to_epoch=s1*32/35000)
        for aligned in (False,True):
            dt,db=[],[]
            for f in range(len(rows)):
                d=disp(pos[s0][f].double(),pos[s1][f].double(),aligned);m=tool[f].cpu();dt.append(d[m].mean());db.append(d[~m].mean())
            tag='aligned' if aligned else 'raw'
            row[f'speed_tool_{tag}_per_epoch']=float(torch.stack(dt).mean()/de);row[f'speed_bg_{tag}_per_epoch']=float(torch.stack(db).mean()/de)
        kn=[]
        for f in range(len(rows)):
            a,b=knn(pos[s0][f],KNN),knn(pos[s1][f],KNN)
            kn.append(float(np.mean([len(set(a[j].tolist())&set(b[j].tolist()))/KNN for j in range(600)])))
        row['knn_kept']=float(np.mean(kn));moves.append(row)
    csvwrite('RUCH_MIEDZY_CHECKPOINTAMI.csv',moves)
    msd=[]
    cs=[s for s in steps if s>=65640]
    for lag in range(1,len(cs)):
        vt,vb=[],[]
        for i in range(len(cs)-lag):
            for f in range(len(rows)):
                d=disp(pos[cs[i]][f].double(),pos[cs[i+lag]][f].double(),True);m=tool[f].cpu();vt.append(float((d[m]**2).mean()));vb.append(float((d[~m]**2).mean()))
        msd.append(dict(lag_epochs=lag*10,msd_tool=float(np.mean(vt)),msd_bg=float(np.mean(vb))))
    csvwrite('MSD_60_200.csv',msd)
    lg=np.log([r['lag_epochs'] for r in msd]);fit={k:float(np.polyfit(lg,np.log([r[k] for r in msd]),1)[0]) for k in ('msd_tool','msd_bg')}
    (OUT/'PROVENANCE.json').write_text(json.dumps(dict(frames=[r['id'] for r in rows],frame_sha256=[r['image_sha256'] for r in rows],
        tool_tokens_per_frame=[int(t) for t in tool.sum(1).tolist()],checkpoints=inputs,block=K+1,msd_exponent_60_200=fit,
        P3='app recipe (lasery_dane.py) from epoch-200 fc2 of block 6',script_sha256=sha(__file__),device=str(dev),
        torch=torch.__version__,seconds=time.monotonic()-start),indent=2),encoding='utf-8')
    print('MSD exponent 60-200:',fit)

if __name__=='__main__':main()
