"""Exploratory larger-group follow-up, declared after pair results."""
from pomiar import ROOT,OUT,PREV,LD,decode,iou,jacobian,npcpu
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import json,copy,time,hashlib

def choose_groups(A,U,rng):
    norm=np.linalg.norm(A,axis=1);unorm=np.linalg.norm(U,axis=1)
    cand=np.argsort(-norm,kind='stable')[:128]
    ii,jj=np.triu_indices(128,1);pairs=np.stack([cand[ii],cand[jj]],1)
    ns=norm[pairs];pairs=pairs[ns.max(1)/np.maximum(ns.min(1),1e-20)<=1.25]
    cos=(A[pairs[:,0]]*A[pairs[:,1]]).sum(1)/np.maximum(norm[pairs].prod(1),1e-20)
    def choose(sign):
        used=set();chosen=[]
        for k in np.argsort(sign*cos,kind='stable'):
            p=pairs[k]
            if not any(int(v) in used for v in p):
                chosen.append(p);used.update(map(int,p))
                if len(chosen)==16:return np.array(chosen)
        raise ValueError('Insufficient disjoint pairs')
    selectors={'cancel':choose(1),'reinforce':choose(-1)}
    result=[]
    for n in [8,16,32]:
        for key,pp in selectors.items():
            chosen=pp[:n//2].copy();result.append((key,n,chosen,0.))
            available=set(range(768))-set(chosen.flatten());control=[];errs=[]
            for c in chosen.flatten():
                pool=np.array(sorted(available));err=np.abs(norm[pool]/norm[c]-1)
                possible=np.flatnonzero(err<=.2)
                idx=int(rng.choice(possible)) if len(possible) else int(np.argmin(err))
                selected=int(pool[idx]);available.remove(selected);control.append(selected);errs.append(float(err[idx]))
            result.append(('matched_'+key,n,np.array(control).reshape(-1,2),max(errs)))
        for key,order in [('top_sensitivity',np.argsort(-norm,kind='stable')),
                          ('top_latent',np.argsort(-unorm,kind='stable')),('random',rng.permutation(768))]:
            result.append((key,n,order[:n].reshape(-1,2),0.))
    return result

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    data=np.load(PREV/'pomiar.npz');meta=json.loads((PREV/'ekstrakcja.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'MANIFEST.json').read_text(encoding='utf-8'))
    gt=data['gt'];ids=data['ids'][:,0];wd=ROOT/'assets/MalyViT_35000_od_zera_s47/wagi'
    rgb=np.stack([np.load(ROOT/'app/cache_uczenia/lasery'/n/'dane.npz')['rgb'] for n in meta['names']])
    image=torch.tensor(rgb,device=dev).permute(0,3,1,2).float()/255
    M=LD.modul_modelu();net=M.MalyViT().to(dev).eval().requires_grad_(False);pam={}
    net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:pam.__setitem__('z',o.detach()))
    net.norm.register_forward_pre_hook(lambda m,a:pam.__setitem__('post',a[0].detach()))
    rows=[];groups=[];rng=np.random.default_rng(151016);start=time.time();global_checks=[]
    for si,step in enumerate(manifest['steps']):
        net.load_state_dict(torch.load(wd/f'k{step:05d}.pt',map_location='cpu',weights_only=True)['net'],strict=True)
        with torch.no_grad():raw,_=net(image)
        post=pam['post'].clone();z=pam['z'].clone();V=net.bloki[-1].mlp.fc2.weight
        head=torch.nn.Sequential(copy.deepcopy(net.norm),copy.deepcopy(net.glowa)).double()
        ps=post[torch.arange(3,device=dev),torch.tensor(ids,device=dev)]
        zs=z[torch.arange(3,device=dev),torch.tensor(ids,device=dev)]
        yl,J=jacobian(head,ps.double(),torch.tensor(ids,device=dev))
        A=npcpu((J[:,1:]@V.double())*zs.double()[:,None,:]).transpose(0,2,1)
        U=npcpu(zs[:,:,None]*V.T[None]);base_local=npcpu(yl)
        baseidx=raw.reshape(3,600,5)[:,:,0].argmax(1)
        base_global=npcpu(decode(raw.reshape(3,600,5)[torch.arange(3,device=dev),baseidx],baseidx))
        for b in range(3):
            selections=choose_groups(A[b],U[b],rng)
            for method,n,pairs,match in selections:
                groups.append(dict(step=step,image=b,method=method,size=n,match_max_error=match,neurons=pairs.tolist()))
                left=torch.tensor(pairs[:,0],device=dev);right=torch.tensor(pairs[:,1],device=dev)
                with torch.no_grad():
                    dl=(z[b,:,left,None]*V.T[left][None]).sum(1)
                    dr=(z[b,:,right,None]*V.T[right][None]).sum(1)
                    shifts=torch.stack([dl,dr,dl+dr])
                    # Local sum in float64 to isolate tiny nonlinear interactions reliably.
                    l=(zs[b,left,None].double()*V.T[left].double()).sum(0)
                    r=(zs[b,right,None].double()*V.T[right].double()).sum(0)
                    sl=torch.stack([l,r,l+r])
                for alpha in [.1,.5,1.]:
                    with torch.no_grad():
                        loc=npcpu(decode(head(ps[b].double()[None]-alpha*sl),torch.tensor(ids[b],device=dev)))
                        globraw=net.glowa(net.norm(post[b][None]-alpha*shifts))
                        win=globraw[:,:,0].argmax(1)
                        glob=npcpu(decode(globraw[torch.arange(3,device=dev),win],win))
                    for scope,vals,base in [('local',loc,base_local[b]),('global',glob,base_global[b])]:
                        delta=vals-base;den=np.linalg.norm(delta[0,1:])+np.linalg.norm(delta[1,1:])
                        inter=delta[2]-delta[0]-delta[1]
                        entry=dict(step=step,image=b,method=method,size=n,scope=scope,alpha=alpha,
                            half_a_px=float(np.linalg.norm(delta[0,1:])/2),half_b_px=float(np.linalg.norm(delta[1,1:])/2),
                            joint_px=float(np.linalg.norm(delta[2,1:])/2),q=float(np.linalg.norm(delta[2,1:])/den) if den>1e-3 else np.nan,
                            interaction_px=float(np.linalg.norm(inter[1:])/2),interaction_ratio=float(np.linalg.norm(inter[1:])/den) if den>1e-3 else np.nan,
                            base_iou=float(iou(base[1:],gt[b])),half_a_iou=float(iou(vals[0,1:],gt[b])),half_b_iou=float(iou(vals[1,1:],gt[b])),
                            joint_iou=float(iou(vals[2,1:],gt[b])),delta_iou=float(iou(vals[2,1:],gt[b])-iou(base[1:],gt[b])),
                            logit_change=float(delta[2,0]),match_max_error=match,
                            switched=int(win[2]!=baseidx[b]) if scope=='global' else 0,
                            **{f'base{k}':float(v) for k,v in enumerate(base[1:])},
                            **{f'new{k}':float(v) for k,v in enumerate(vals[2,1:])})
                        rows.append(entry)
                if si==len(manifest['steps'])-1 and b==2 and method=='cancel' and n==32:
                    pp=pairs.flatten()
                    def abl(m,a,o):
                        q=o.clone();q[:,:,pp]=0;return q
                    h=net.bloki[-1].mlp.act.register_forward_hook(abl)
                    with torch.no_grad():oo,_=net(image)
                    h.remove()
                    error=float((oo[b].reshape(600,5)-globraw[2]).abs().max())
                    assert error<3e-5;global_checks.append(dict(step=step,image=b,size=n,hook_max_abs=error))
        print(f'groups {si+1}/{len(manifest["steps"])} step={step} seconds={time.time()-start:.1f}',flush=True)
    frame=pd.DataFrame(rows)
    assert np.isfinite(frame.drop(columns=['q','interaction_ratio']).select_dtypes('number').values).all()
    frame.to_csv(OUT/'interwencje_grupowe.csv.gz',index=False)
    (OUT/'grupy_neuronow.json').write_text(json.dumps(groups),encoding='utf-8')
    info=dict(rows=len(frame),checks=global_checks,seconds=time.time()-start,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'grupy_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    late=frame[(frame.step>=55000)&(frame.alpha==1)]
    table=late.groupby(['scope','size','method'])[['half_a_px','half_b_px','joint_px','q','interaction_ratio','delta_iou','switched']].median()
    table.to_csv(OUT/'grupy_pozne.csv');print(table.to_string())
    print('Match maximum:',max(x['match_max_error'] for x in groups))

if __name__=='__main__':main()
