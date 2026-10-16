"""Frozen-checkpoint neuron interventions. No writes to app or source data."""
from pathlib import Path
import sys, json, time, hashlib, argparse, copy
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/wspoldzialanie_neuronow_20261015'
PREV = ROOT/'outputs/optyka_hipoteza_20261015'
sys.path.insert(0,str(ROOT/'app'))
import lasery_dane as LD

def decode(raw, ids):
    d=raw[...,1:].exp()*16
    cx=(ids%30+.5)*16; cy=(ids//30+.5)*16
    return torch.stack([raw[...,0],cx-d[...,0],cy-d[...,1],cx+d[...,2],cy+d[...,3]],-1)

def iou(a,b):
    wh=np.maximum(np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]),0)
    inter=wh.prod(-1)
    area=np.maximum(a[...,2:]-a[...,:2],0).prod(-1)
    other=np.maximum(b[...,2:]-b[...,:2],0).prod(-1)
    return inter/np.maximum(area+other-inter,1e-20)

def pair_sets(A,U,rng):
    """A and U are [768,feature]; frozen A concatenates tokens before this call."""
    norm=np.linalg.norm(A,axis=1); unorm=np.linalg.norm(U,axis=1)
    cand=np.argsort(-norm,kind='stable')[:64]
    ii,jj=np.triu_indices(64,1); pairs=np.stack([cand[ii],cand[jj]],1)
    ns=norm[pairs]; valid=ns.max(1)/np.maximum(ns.min(1),1e-20)<=1.25
    pairs=pairs[valid]; ns=ns[valid]
    assert len(pairs)>=40
    cos=(A[pairs[:,0]]*A[pairs[:,1]]).sum(1)/np.maximum(ns.prod(1),1e-20)
    ucos=(U[pairs[:,0]]*U[pairs[:,1]]).sum(1)/np.maximum(unorm[pairs].prod(1),1e-20)
    def choose(score):
        chosen=[]; used=set()
        for j in np.argsort(score,kind='stable'):
            p=pairs[j]
            if not any(int(c) in used for c in p):
                chosen.append(p); used.update(map(int,p))
                if len(chosen)==4: break
        assert len(chosen)==4
        return np.array(chosen)
    result={'functional_cancel':choose(cos),'functional_reinforce':choose(-cos),
            'latent_cancel':choose(ucos),'latent_reinforce':choose(-ucos)}
    excluded={tuple(sorted(map(int,p))) for key in ['functional_cancel','functional_reinforce'] for p in result[key]}
    pool=np.array([p for p in pairs if tuple(sorted(map(int,p))) not in excluded])
    match=[]
    for key in ['cancel','reinforce']:
        selected=[]
        for p in result['functional_'+key]:
            target=np.sort(norm[p]); pooln=np.sort(norm[pool],axis=1)
            error=np.max(np.abs(pooln/target-1),axis=1)
            possible=np.flatnonzero(error<=.2)
            ix=int(rng.choice(possible)) if len(possible) else int(np.argmin(error))
            selected.append(pool[ix]); match.append(float(error[ix]))
        result['matched_'+key]=np.array(selected)
    result['top_sensitivity']=np.argsort(-norm,kind='stable')[:8].reshape(4,2)
    result['top_latent']=np.argsort(-unorm,kind='stable')[:8].reshape(4,2)
    result['random']=rng.choice(768,8,replace=False).reshape(4,2)
    return result,match

def jacobian(head,post,ids):
    q=post.detach().clone().requires_grad_(True)
    y=decode(head(q),ids)
    J=torch.stack([torch.autograd.grad(y[:,j].sum(),q,retain_graph=j<4)[0] for j in range(5)],1)
    return y.detach(),J.detach()

def npcpu(x): return x.detach().cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=OUT/('smoke' if args.smoke else '');out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(151015)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    data=np.load(PREV/'pomiar.npz'); meta=json.loads((PREV/'ekstrakcja.json').read_text(encoding='utf-8'))
    steps=data['steps']; ids=data['ids'][:,:8]; gt=data['gt']
    pre=data['x_pre'][:,:,:8]; post=data['x_post'][:,:,:8]
    chosen=np.unique([np.argmin(abs(steps-s)) for s in list(range(1000,65001,2000))+[65640]])
    if args.smoke: chosen=chosen[-1:]
    weightdir=Path(meta['source_weight_dir'])
    histories=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'dane.npz') for n in meta['names']]
    rgb=np.stack([h['rgb'] for h in histories])
    image=torch.tensor(rgb,device=dev).permute(0,3,1,2).float()/255
    M=LD.modul_modelu();net=M.MalyViT().to(dev).eval().requires_grad_(False)
    assert isinstance(net.bloki[-1].ls2,torch.nn.Identity)
    def load(k):
        state=torch.load(weightdir/f'k{int(k):05d}.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(state['net'],strict=True)
    def double_head():
        return torch.nn.Sequential(copy.deepcopy(net.norm),copy.deepcopy(net.glowa)).double()
    rng=np.random.default_rng(151015)
    # Freeze identities before evaluation; no image 2 nor later checkpoints used.
    fi=int(np.argmin(abs(steps-33000))); load(steps[fi])
    xf=torch.tensor(pre[fi,:2].reshape(-1,192),device=dev)
    pf=torch.tensor(post[fi,:2].reshape(-1,192),device=dev)
    zf=net.bloki[-1].mlp.act(net.bloki[-1].mlp.fc1(net.bloki[-1].norm2(xf)))
    V=net.bloki[-1].mlp.fc2.weight
    _,Jf=jacobian(double_head(),pf.double(),torch.tensor(ids[:2].reshape(-1),device=dev))
    Af=npcpu((Jf[:,1:]@V.double())*zf.double()[:,None,:]).transpose(2,0,1).reshape(768,-1)
    Uf=npcpu(zf[:,:,None]*V.T[None]).transpose(1,0,2).reshape(768,-1)
    freeze,freeze_match=pair_sets(Af,Uf,rng)
    frozen={'frozen_'+k.replace('functional_',''):v for k,v in freeze.items() if k in
            ['functional_cancel','functional_reinforce','matched_cancel','matched_reinforce']}
    (out/'zamrozone_pary.json').write_text(json.dumps(dict(step=int(steps[fi]),images=[0,1],
        pairs={k:v.tolist() for k,v in frozen.items()},match_errors=freeze_match),indent=2),encoding='utf-8')
    pam={}
    net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:pam.__setitem__('z',o.detach()))
    net.norm.register_forward_pre_hook(lambda m,a:pam.__setitem__('post',a[0].detach()))
    rows=[]; globalrows=[]; checks=[]; baselines=[]; matches=[]; start=time.time()
    allids=torch.arange(600,device=dev)
    alphas=[.1,.5,1.]
    for ci,index in enumerate(chosen):
        step=int(steps[index]);load(step)
        with torch.no_grad():
            original,_=net(image)
            fullpost=pam['post'].clone(); fullz=pam['z'].clone()
            V=net.bloki[-1].mlp.fc2.weight
            ps=torch.tensor(post[index].reshape(-1,192),device=dev)
            xp=torch.tensor(pre[index].reshape(-1,192),device=dev)
            z=net.bloki[-1].mlp.act(net.bloki[-1].mlp.fc1(net.bloki[-1].norm2(xp)))
            U=z[:,:,None]*V.T[None]
            rec=xp+U.sum(1)+net.bloki[-1].mlp.fc2.bias
            recerr=float((rec-ps).abs().max())
            checkps=fullpost[torch.arange(3,device=dev)[:,None],torch.tensor(ids,device=dev)].reshape(-1,192)
            checkerr=float((checkps-ps).abs().max())
        head=double_head();psd=ps.double();Ud=U.double()
        y,J=jacobian(head,psd,torch.tensor(ids.reshape(-1),device=dev))
        A=npcpu((J[:,1:]@V.double())*z.double()[:,None,:]).transpose(0,2,1)
        UC=npcpu(U); JC=npcpu(J); YC=npcpu(y)
        checks.append(dict(step=step,reconstruction_max_abs=recerr,cache_max_abs=checkerr))
        assert recerr<2e-5 and checkerr<2e-5
        # Verify global argmax agrees with model's local-max decoder k=1.
        with torch.no_grad():
            base_full=decode(original.reshape(3,600,5),allids)
            bestidx=original.reshape(3,600,5)[:,:,0].argmax(1)
            best=npcpu(base_full[torch.arange(3,device=dev),bestidx])
            official,_=M.dekoduj(original,k=1)
            assert np.allclose(best[:,1:],npcpu(official[:,0]),atol=1e-4)
        for t in range(24):
            b,j=divmod(t,8); tok=int(ids[b,j]); baseline=YC[t]
            sets,mat=pair_sets(A[t],UC[t],rng);sets.update(frozen);matches.extend(mat)
            methods=list(sets); pairs=np.concatenate(list(sets.values()))
            method=np.repeat(methods,4); slots=np.tile(np.arange(4),len(methods)); P=len(pairs)
            pairt=torch.tensor(pairs,device=dev)
            # Only evaluate required single neurons, preserving exact shared baselines.
            unique,inverse=np.unique(pairs,return_inverse=True); inv=inverse.reshape(P,2)
            unique_t=torch.tensor(unique,device=dev)
            norms=np.linalg.norm(A[t],axis=1); unorm=np.linalg.norm(UC[t],axis=1)
            cos=(A[t,pairs[:,0]]*A[t,pairs[:,1]]).sum(1)/np.maximum(norms[pairs].prod(1),1e-20)
            ucos=(UC[t,pairs[:,0]]*UC[t,pairs[:,1]]).sum(1)/np.maximum(unorm[pairs].prod(1),1e-20)
            qpred=np.linalg.norm(A[t,pairs].sum(1),axis=1)/np.maximum(norms[pairs].sum(1),1e-20)
            baseiou=float(iou(baseline[1:],gt[b]));baseerr=float(np.sqrt(np.mean((baseline[1:]-gt[b])**2)))
            baselines.append(dict(step=step,image=b,token_slot=j,token=tok,base_iou=baseiou,base_err_px=baseerr,
                                  base_logit=float(baseline[0]),**{f'b{k}':float(v) for k,v in enumerate(baseline[1:])}))
            for alpha in alphas:
                with torch.no_grad():
                    single=decode(head(psd[t]-alpha*Ud[t,unique_t]),torch.tensor(tok,device=dev))
                    joint=decode(head(psd[t]-alpha*Ud[t,pairt].sum(1)),torch.tensor(tok,device=dev))
                ds=npcpu(single)-baseline; dj=npcpu(joint)-baseline
                da=ds[inv[:,0]];db=ds[inv[:,1]];interaction=dj-da-db
                denom=np.linalg.norm(da[:,1:],axis=1)+np.linalg.norm(db[:,1:],axis=1)
                # Inactive/saturated GELU neurons can have zero measurable effect.
                # Keep conservative threshold from pilot even after double-head refinement.
                valid=denom>1e-3
                safe_denom=np.where(valid,denom,np.nan)
                pred=-alpha*A[t,pairs].sum(1)
                erradd=np.linalg.norm(interaction[:,1:],axis=1)/safe_denom
                errlin=np.linalg.norm(dj[:,1:]-pred,axis=1)/safe_denom
                for p in range(P):
                    r=dict(step=step,image=b,token_slot=j,token=tok,method=method[p],pair_slot=int(slots[p]),
                        c=int(pairs[p,0]),d=int(pairs[p,1]),alpha=alpha,cos=float(cos[p]),latent_cos=float(ucos[p]),
                        norm_a=float(norms[pairs[p,0]]),norm_b=float(norms[pairs[p,1]]),q_pred=float(qpred[p]),
                        valid_ratio=bool(valid[p]),q_actual=float(np.linalg.norm(dj[p,1:])/safe_denom[p]),nonlinear_ratio=float(erradd[p]),
                        linear_error_ratio=float(errlin[p]),single_a_px=float(np.linalg.norm(da[p,1:])/2),
                        single_b_px=float(np.linalg.norm(db[p,1:])/2),joint_px=float(np.linalg.norm(dj[p,1:])/2),
                        interaction_px=float(np.linalg.norm(interaction[p,1:])/2),logit_change=float(dj[p,0]),
                        base_iou=baseiou,delta_iou=float(iou(baseline[1:]+dj[p,1:],gt[b])-baseiou),
                        delta_err_px=float(np.sqrt(np.mean((baseline[1:]+dj[p,1:]-gt[b])**2))-baseerr))
                    for prefix,arr in [('da',da),('db',db),('dj',dj)]:
                        r.update({f'{prefix}{k}':float(v) for k,v in enumerate(arr[p,1:])})
                    rows.append(r)
                if j==0:
                    # Same neuron IDs disabled in every token, without repeating backbone.
                    # Last MLP is followed only by tokenwise norm/head, so this is exact.
                    chunks=[]
                    for p0 in range(0,P,12):
                        pp=pairt[p0:p0+12]
                        delta=(fullz[b,:,pp].permute(1,0,2)[...,None]*V.T[pp][:,None]).sum(2)
                        with torch.no_grad():
                            rr=net.glowa(net.norm(fullpost[b][None]-alpha*delta))
                            winner=rr[:,:,0].argmax(1)
                            yy=decode(rr[torch.arange(len(pp),device=dev),winner],winner)
                        chunks.append((npcpu(yy),npcpu(winner)))
                    glob=np.concatenate([c[0] for c in chunks]);win=np.concatenate([c[1] for c in chunks])
                    for p in range(P):
                        globalrows.append(dict(step=step,image=b,method=method[p],pair_slot=int(slots[p]),
                            c=int(pairs[p,0]),d=int(pairs[p,1]),alpha=alpha,base_token=int(bestidx[b]),new_token=int(win[p]),
                            switched=int(win[p]!=int(bestidx[b])),base_iou=float(iou(best[b,1:],gt[b])),
                            delta_iou=float(iou(glob[p,1:],gt[b])-iou(best[b,1:],gt[b])),
                            joint_px=float(np.linalg.norm(glob[p,1:]-best[b,1:])/2),logit_change=float(glob[p,0]-best[b,0]),
                            **{f'base{k}':float(v) for k,v in enumerate(best[b,1:])},
                            **{f'new{k}':float(v) for k,v in enumerate(glob[p,1:])}))
            if ci==0 and t==0:
                # Independent full-forward intervention via activation hook.
                pp=pairs[0]; expected=ps[t]-U[t,torch.tensor(pp,device=dev)].sum(0)
                def ablate(m,a,o):
                    q=o.clone();q[:,:,pp]=0;return q
                handle=net.bloki[-1].mlp.act.register_forward_hook(ablate)
                with torch.no_grad(): direct,_=net(image)
                handle.remove()
                with torch.no_grad():
                    fast=net.glowa(net.norm(fullpost-(fullz[:,:,pp,None]*V.T[pp][None,None]).sum(2)))
                hookerr=float((direct.reshape(3,600,5)-fast).abs().max())
                assert hookerr<3e-5
                checks[-1]['hook_max_abs']=hookerr
                # Independent float64 central difference, using copied head only.
                nh=copy.deepcopy(net.norm).double();hh=copy.deepcopy(net.glowa).double()
                q=ps[t].double().detach().requires_grad_(True);v=U[t,int(pp[0])].double()
                yy=decode(hh(nh(q)),torch.tensor(tok,device=dev))
                jj=torch.stack([torch.autograd.grad(yy[k],q,retain_graph=k<4)[0] for k in range(5)])
                eps=1e-3
                fd=(decode(hh(nh(q+eps*v)),torch.tensor(tok,device=dev))-decode(hh(nh(q-eps*v)),torch.tensor(tok,device=dev)))/(2*eps)
                jv=jj@v
                ferr=float(((fd-jv).norm()/jv.norm()).detach());assert ferr<1e-5
                checks[-1]['jvp_relative_error']=ferr
        print(f'{ci+1}/{len(chosen)} step={step}, rows={len(rows)}, seconds={time.time()-start:.1f}',flush=True)
    local=pd.DataFrame(rows);glob=pd.DataFrame(globalrows);base=pd.DataFrame(baselines)
    nullable=['q_actual','nonlinear_ratio','linear_error_ratio']
    assert np.isfinite(local.drop(columns=nullable).select_dtypes('number').values).all()
    assert np.isfinite(local.loc[local.valid_ratio,nullable].values).all()
    assert local.loc[~local.valid_ratio,nullable].isna().all().all()
    assert np.isfinite(glob.select_dtypes('number').values).all()
    assert not local.duplicated(['step','image','token','method','pair_slot','alpha']).any()
    local.to_csv(out/'interwencje_lokalne.csv.gz',index=False)
    glob.to_csv(out/'interwencje_globalne.csv.gz',index=False)
    base.to_csv(out/'ramki_bazowe.csv',index=False)
    manifest=dict(steps=steps[chosen].tolist(),images=meta['names'],image_hashes=meta['image_hashes'],
        freeze_step=int(steps[fi]),alphas=alphas,n_local=len(local),n_global=len(glob),checks=checks,
        matching_max_relative_error=max(matches),matching_fraction_within_20_percent=float(np.mean(np.array(matches)<=.2)),
        seconds=time.time()-start,torch=torch.__version__,device=dev,local_head_dtype='float64',global_head_dtype='float32',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(Path(__file__).with_name('PROTOKOL.md').read_bytes()).hexdigest(),
        checkpoint_sha256={str(int(steps[i])):hashlib.sha256((weightdir/f'k{int(steps[i]):05d}.pt').read_bytes()).hexdigest() for i in np.unique(np.r_[chosen,fi])})
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
