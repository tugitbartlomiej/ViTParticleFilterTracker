"""Pair 370-391 of block 6: token cancellation, weight history and ablation; detection IoU/score over epochs 0-200."""
from pathlib import Path
import argparse,csv,json,sys,time
import numpy as np

EXP=Path(__file__).resolve().parents[1]
CFG=json.loads((EXP/'Kod_eksperymentu/konfiguracja.json').read_text(encoding='utf-8'))
ZN=json.loads((EXP/'Kod_eksperymentu/konfiguracja_znoszenie.json').read_text(encoding='utf-8'))
BASE=Path(CFG['base_eden'])
sys.path.insert(0,str(BASE/CFG['measurement_code_relative']))
import common as C
import torch

K=ZN['block']-1;A,B=ZN['pair'];CONTROL=ZN['control_step']
def save(folder,name,value):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvwrite(folder,name,rows):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def finite(v):return None if v is None or not np.isfinite(v) else float(v)

def checkpoints(steps_filter):
    best={m['step']:m for m in json.loads((BASE/ZN['parent_best_checkpoint_relative']).read_text())['all_checkpoints']}
    out=[]
    for step in ZN['history_steps']:
        p=C.weight_path(step);h=C.sha(p)
        if step in best:assert h==best[step]['sha256'],'History weights changed: '+str(step)
        out.append(dict(step=step,path=p,sha256=h,kind='history'))
    for ck in CFG['checkpoints']:
        p=BASE/CFG['checkpoint_dir_relative']/ck['file'];h=C.sha(p);assert h==ck['sha256'],'Checkpoint changed: '+ck['file']
        out.append(dict(step=ck['step'],path=p,sha256=h,kind='continuation'))
    return [c for c in out if steps_filter is None or c['step'] in steps_filter],best

def load(net,ck):
    if ck['kind']=='history':C.load(net,ck['step']);return
    # Own continuation checkpoint with pinned SHA256; it stores RNG objects, so weights_only=False.
    st=torch.load(ck['path'],map_location='cpu',weights_only=False)
    assert st['krok']==ck['step'] and st['seed']==3047
    net.load_state_dict(st['net'],strict=True)

class Probe:
    """Block-6 GELU hook (optional zeroing of neurons) and block-6 output hook."""
    def __init__(self,net):
        self.ablate=None;self.keep=False;self.z=[];self.t=[]
        net.bloki[K].mlp.act.register_forward_hook(self.act);net.bloki[K].register_forward_hook(self.block)
    def act(self,m,a,o):
        if self.keep:self.z.append(o.detach().cpu().numpy())
        if self.ablate is not None:
            # Multiplicative 0/1 mask built on CPU: exact zeros, no CUDA index_put under deterministic mode.
            mask=torch.ones(o.shape[-1],dtype=o.dtype);mask[self.ablate]=0
            return o*mask.to(o.device)
    def block(self,m,a,o):
        if self.keep:self.t.append(o.detach().cpu().numpy())

def run(net,probe,images,batch,ablate=None,keep=False):
    probe.ablate=None if ablate is None else list(ablate);probe.keep=keep;probe.z=[];probe.t=[];raws=[]
    with torch.inference_mode():
        for i in range(0,len(images),batch):raws.append(C.forward(net,C.tensor(images[i:i+batch])).cpu().numpy())
    probe.ablate=None;probe.keep=False;raw=np.concatenate(raws).astype(np.float64)
    return (raw,np.concatenate(probe.z),np.concatenate(probe.t)) if keep else raw

def detection(raw,rows):
    # Same rule as the parent checkpoint selection: argmax-logit token, IoU to the best GT box.
    boxes,scores=C.decode(raw);n=len(rows);iou=np.full(n,np.nan);score=np.empty(n);tok=np.empty(n,dtype=np.int64)
    for j,r in enumerate(rows):
        t=int(np.argmax(raw[j,:,0]));tok[j]=t;score[j]=scores[j,t]
        if r['gt']:iou[j]=float(np.max(C.iou(boxes[j,t],r['gt'])))
    return iou,score,tok
def summary(iou,score):
    pos=np.isfinite(iou)
    return dict(mean_top1_iou=float(iou[pos].mean()),mean_score=float(score.mean()),mean_score_positive=float(score[pos].mean()),
        fraction_iou_ge_05=float((iou[pos]>=.5).mean()),positive_images=int(pos.sum()),empty_images=int((~pos).sum()),
        empty_mean_top1_score=finite(score[~pos].mean()) if (~pos).any() else None)

def gt_mask(rows):
    ids=np.arange(600);cx=(ids%30+.5)*16;cy=(ids//30+.5)*16;m=np.zeros((len(rows),600),dtype=bool)
    for j,r in enumerate(rows):
        for x1,y1,x2,y2 in r['gt']:m[j]|=(cx>=x1)&(cx<=x2)&(cy>=y1)&(cy<=y2)
    return m.reshape(-1)

def q(x,p):return finite(np.quantile(x,p)) if len(x) else None
def pair_tokens(z,V,ingt,step):
    z=z.astype(np.float64).reshape(-1,768);V=V.astype(np.float64);norms=np.linalg.norm(V,axis=1)
    s=np.abs(z)*norms;smax=s.max(1);za,zb=z[:,A],z[:,B];sa,sb=s[:,A],s[:,B]
    ua=za[:,None]*V[A];ub=zb[:,None]*V[B];y=z@V;ynorm=np.linalg.norm(y,axis=1);pn=np.linalg.norm(ua+ub,axis=1)
    thr=ZN['nonzero_strength_threshold'];rel=ZN['active_relative_threshold']
    both=(sa>thr)&(sb>thr);opp=both&(np.sign(za)!=np.sign(zb));act=both&(sa>rel*smax)&(sb>rel*smax)
    with np.errstate(divide='ignore',invalid='ignore'):
        theta=np.degrees(np.arccos(np.clip((ua*ub).sum(1)/(sa*sb),-1,1)))
        ratio=np.minimum(sa,sb)/np.maximum(sa,sb);kappa=pn**2/(sa**2+sb**2);canc=1-pn/(sa+sb);share=pn/ynorm;share_sum=(sa+sb)/ynorm
    rows=[]
    for name,m in [('both_nonzero',both),('same_sign',both&~opp),('opposite_sign',opp),('same_sign_both_active05',act&~opp),('opposite_sign_both_active05',act&opp)]:
        neg_a=m&(za<0);neg_b=m&(zb<0)
        rows.append(dict(step=step,epoch_equivalent=step*32/35000,category=name,tokens=int(m.sum()),
            frames_with_tokens=int(m.reshape(-1,600).any(1).sum()),fraction_in_gt=finite(ingt[m].mean()) if m.any() else None,
            aggregate_kappa=finite((pn[m]**2).sum()/(sa[m]**2+sb[m]**2).sum()) if m.any() else None,
            **{f'{k}_q{int(p*100)}':q(v[m],p) for k,v in [('angle',theta),('strength_ratio',ratio),('kappa',kappa),('cancellation',canc),('pair_share_of_mlp',share),('sum_share_of_mlp',share_sum)] for p in (.1,.5,.9)},
            negative_is_a=int(neg_a.sum()),negative_is_b=int(neg_b.sum()),
            mean_abs_negative_z=finite(np.concatenate([-za[neg_a],-zb[neg_b]]).mean()) if (neg_a|neg_b).any() else None,
            mean_strength_a=finite(sa[m].mean()) if m.any() else None,mean_strength_b=finite(sb[m].mean()) if m.any() else None))
    rows[0]['corr_za_zb_all_tokens']=finite(np.corrcoef(za,zb)[0,1])
    arrays=dict(za=za.astype(np.float32),zb=zb.astype(np.float32),smax=smax.astype(np.float32),ynorm=ynorm.astype(np.float32),ingt=ingt)
    return rows,arrays,s.mean(0)

def pair_weights(net,step):
    blk=net.bloki[K];d=lambda t:t.detach().cpu().double().numpy()
    W1,b1,V=d(blk.mlp.fc1.weight),d(blk.mlp.fc1.bias),d(blk.mlp.fc2.weight).T;g,be=d(blk.norm2.weight),d(blk.norm2.bias)
    cos=lambda x,y:float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
    eff=W1*g;eff=eff-eff.mean(1,keepdims=True);bh=b1+W1@be
    return dict(step=step,epoch_equivalent=step*32/35000,w2_angle_deg=float(np.degrees(np.arccos(np.clip(cos(V[A],V[B]),-1,1)))),
        w2_norm_a=float(np.linalg.norm(V[A])),w2_norm_b=float(np.linalg.norm(V[B])),w1_cos_raw=cos(W1[A],W1[B]),
        w1_cos_ln_folded=cos(eff[A],eff[B]),w1_norm_a=float(np.linalg.norm(W1[A])),w1_norm_b=float(np.linalg.norm(W1[B])),
        b_hat_a=float(bh[A]),b_hat_b=float(bh[B])),V

def logit_effect(raw0,raw):
    d=raw-raw0;return float(np.abs(d[...,0]).mean()),float(np.linalg.norm(d[...,1:],axis=-1).mean())

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    tag='smoke' if args.smoke else 'pelny'
    OUT=EXP/'Wyniki/Dane'/('smoke_znoszenie' if args.smoke else 'znoszenie_370_391')
    DET=EXP/'Wyniki/Dane'/('smoke_detekcja' if args.smoke else 'detekcja_0_200')
    if (OUT/'DONE.json').exists():raise SystemExit('Existing completed results preserved')
    start=time.monotonic();net=C.setup();probe=Probe(net)
    assert C.sha(Path(C.__file__))==CFG['measurement_common_sha256'] and C.sha(Path(C.M.__file__))==CFG['model_sha256']
    ckpts,best=checkpoints([0,CONTROL] if args.smoke else None)
    diag=C.CFG['rows'];assert len(diag)==50;ingt=gt_mask(diag);dimg=[C.image(r) for r in diag]
    data=json.loads((Path(C.CFG['source_root'])/'input/dataset.json').read_text())['rows']
    splits={s:[r for r in data if r['source_split']==s and r['role']==role] for s,role in ZN['detection_splits'].items()}
    assert all(len(v)==500 for v in splits.values())
    simg={s:[C.image(r) for r in rows] for s,rows in splits.items()};batch=ZN['detection_batch']
    OUT.mkdir(parents=True,exist_ok=True)
    det_rows=[];det_images=[];tok_rows=[];w_rows=[];abl_rows=[];abl_images={};qc=[];inputs={}
    conds=[('clean',None),('a',[A]),('b',[B]),('ab',[A,B])]
    for ck in ckpts:
        step=ck['step'];inputs[str(ck['path'])]=ck['sha256'];load(net,ck)
        wr,V=pair_weights(net,step);w_rows.append(wr)
        raw0,z,t0=run(net,probe,dimg,10,keep=True)
        tr,arr,mean_strength=pair_tokens(z,V,ingt,step);tok_rows.extend(tr)
        np.savez_compressed(OUT/f'tokens_k{step:06d}.npz',**arr)
        diag_raw={'clean':raw0}
        for name,idx in conds[1:]:
            if name=='ab':diag_raw[name],_,tab=run(net,probe,dimg,10,ablate=idx,keep=True)
            else:diag_raw[name]=run(net,probe,dimg,10,ablate=idx)
        # Last block: removing the two GELU outputs must shift the residual by exactly -(u_a+u_b).
        zf=z.astype(np.float64);exp=-(zf[...,A,None]*V[A]+zf[...,B,None]*V[B]);err=float(np.abs((tab-t0)-exp).max())
        qc.append(dict(step=step,residual_shift_max_abs_error=err,max_abs_expected_shift=float(np.abs(exp).max()),
            pass_check=err<=ZN['residual_check_tolerance']*max(1.,float(np.abs(exp).max()))))
        effects={n:logit_effect(raw0,diag_raw[n]) for n in ['a','b','ab']}
        lin=float(np.abs((diag_raw['ab']-raw0)[...,0]-((diag_raw['a']-raw0)[...,0]+(diag_raw['b']-raw0)[...,0])).mean())
        rec=dict(step=step,epoch_equivalent=step*32/35000,
            **{f'diag50_mean_abs_dlogit_{n}':effects[n][0] for n in effects},**{f'diag50_mean_dbox_{n}':effects[n][1] for n in effects},
            interaction_ratio_logit=effects['ab'][0]/max(effects['a'][0],effects['b'][0],1e-30),
            diag50_nonadditivity_logit=lin,neuron_strength_rank_a=int((mean_strength>mean_strength[A]).sum())+1,
            neuron_strength_rank_b=int((mean_strength>mean_strength[B]).sum())+1)
        d0=detection(raw0,diag);det_rows.append(dict(step=step,epoch_equivalent=step*32/35000,split='DIAG50',**summary(d0[0],d0[1])))
        for s,rows in splits.items():
            base=None
            for name,idx in conds:
                iou,score,tk=detection(run(net,probe,simg[s],batch,ablate=idx),rows);abl_images[f'k{step:06d}_{s}_{name}']=np.stack([iou,score,tk])
                sm=summary(iou,score)
                if name=='clean':
                    base=(iou,score,tk);det_rows.append(dict(step=step,epoch_equivalent=step*32/35000,split=s,**sm))
                    det_images.extend(dict(step=step,split=s,id=r['id'],film=r['film'],token=int(tk[j]),iou=finite(iou[j]),score=float(score[j])) for j,r in enumerate(rows))
                else:
                    pos=np.isfinite(iou)
                    rec[f'{s}_dIoU_{name}']=float((iou[pos]-base[0][pos]).mean());rec[f'{s}_dscore_{name}']=float((score-base[1]).mean())
                    rec[f'{s}_winner_changed_{name}']=int((tk!=base[2]).sum());rec[f'{s}_IoU_{name}']=sm['mean_top1_iou'];rec[f'{s}_score_{name}']=sm['mean_score']
        abl_rows.append(rec)
        if step==CONTROL:control(net,probe,dimg,simg,splits,batch,raw0,V,mean_strength,OUT,args.smoke)
        print('STEP_DONE',step,round(step*32/35000,2),round(time.monotonic()-start,1),flush=True)
    vq=[]
    for r in det_rows:
        if r['split']=='VAL' and r['step'] in best:
            vq.append(dict(step=r['step'],parent=best[r['step']]['mean_top1_iou'],now=r['mean_top1_iou'],abs_diff=abs(best[r['step']]['mean_top1_iou']-r['mean_top1_iou']),
                parent_score=best[r['step']]['mean_score'],now_score=r['mean_score']))
    save(DET,'QC_VAL_VS_PARENT_SELECTION.json',dict(max_abs_iou_diff=max((v['abs_diff'] for v in vq),default=None),rows=vq))
    csvwrite(DET,'DETEKCJA_0_200.csv',det_rows);csvwrite(DET,'DETEKCJA_OBRAZY_0_200.csv',det_images)
    csvwrite(OUT,'PARA_WAGI_0_200.csv',w_rows);csvwrite(OUT,'PARA_TOKENY_0_200.csv',tok_rows);csvwrite(OUT,'ABLACJA_PARY_0_200.csv',abl_rows)
    np.savez_compressed(OUT/'ABLACJA_OBRAZY_0_200.npz',**abl_images)
    save(OUT,'QC.json',dict(status='PASS' if all(x['pass_check'] for x in qc) else 'CHECK_FAILED',residual=qc))
    prov=dict(status='PASS',mode=tag,steps=[c['step'] for c in ckpts],pair=[A,B],block=K+1,inputs=inputs,
        script_sha256=C.sha(Path(__file__)),config_sha256=C.sha(EXP/'Kod_eksperymentu/konfiguracja.json'),
        config_znoszenie_sha256=C.sha(EXP/'Kod_eksperymentu/konfiguracja_znoszenie.json'),
        dataset_sha256=C.sha(Path(C.CFG['source_root'])/'input/dataset.json'),environment=C.environment(),seconds=time.monotonic()-start)
    save(DET,'PROVENANCE.json',prov);save(OUT,'PROVENANCE.json',prov)
    for folder in (DET,OUT):save(folder,'DONE.json',dict(status='PASS',files={str(p.relative_to(folder)):C.sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='DONE.json'}))
    print('COMPLETE',round(time.monotonic()-start,1),flush=True)

def control(net,probe,dimg,simg,splits,batch,raw0,V,mean_strength,OUT,smoke):
    rng=np.random.default_rng(ZN['control_seed']);n=3 if smoke else None
    singles=[]
    for c in (range(10) if smoke else range(768)):
        e=logit_effect(raw0,run(net,probe,dimg,10,ablate=[c]));singles.append(dict(neuron=c,mean_abs_dlogit=e[0],mean_dbox=e[1],mean_strength=float(mean_strength[c])))
    single={r['neuron']:r['mean_abs_dlogit'] for r in singles}
    base={s:detection(run(net,probe,simg[s],batch),rws) for s,rws in splits.items()}
    def unique(pairs,limit):
        seen=[]
        for p in pairs:
            if p[0]!=p[1] and p!=(min(A,B),max(A,B)) and p not in seen:seen.append(p)
        return seen[:limit]
    ii,jj=np.triu_indices(768,1)
    uniform=unique([tuple(sorted(map(int,rng.choice(768,2,replace=False)))) for _ in range(1000)],n or ZN['uniform_random_pairs'])
    rank=np.argsort(np.argsort(-mean_strength))
    pool=lambda x:[int(c) for c in np.argsort(np.abs(rank-rank[x]),kind='stable') if c not in (A,B)][:ZN['matched_pool_size']]
    pa,pb=pool(A),pool(B)
    matched=unique([tuple(sorted((int(rng.choice(pa)),int(rng.choice(pb))))) for _ in range(5000)],n or ZN['matched_pairs'])
    u=V/np.linalg.norm(V,axis=1,keepdims=True);ang=np.degrees(np.arccos(np.clip((u@u.T)[ii,jj],-1,1)))
    parallel=unique([(int(ii[k]),int(jj[k])) for k in np.argsort(ang,kind='stable')[:400]],n or ZN['near_parallel_pairs'])
    rows=[]
    for group,pairs in [('target',[(min(A,B),max(A,B))]),('uniform',uniform),('strength_matched',matched),('near_parallel_w2',parallel)]:
        for x,y in pairs:
            for c in (x,y):
                if c not in single:single[c]=logit_effect(raw0,run(net,probe,dimg,10,ablate=[c]))[0]
            e=logit_effect(raw0,run(net,probe,dimg,10,ablate=[x,y]))
            rec=dict(group=group,a=x,b=y,w2_angle_deg=float(np.degrees(np.arccos(np.clip(u[x]@u[y],-1,1)))),
                strength_a=float(mean_strength[x]),strength_b=float(mean_strength[y]),diag50_mean_abs_dlogit_ab=e[0],
                diag50_mean_abs_dlogit_a=single[x],diag50_mean_abs_dlogit_b=single[y],interaction_ratio_logit=e[0]/max(single[x],single[y],1e-30))
            for s,rws in splits.items():
                iou,score,_=detection(run(net,probe,simg[s],batch,ablate=[x,y]),rws)
                pos=np.isfinite(iou);rec[f'{s}_dIoU_ab']=float((iou[pos]-base[s][0][pos]).mean());rec[f'{s}_dscore_ab']=float((score-base[s][1]).mean())
            rows.append(rec)
    csvwrite(OUT,f'KONTROLA_k{CONTROL:06d}.csv',rows);csvwrite(OUT,f'POJEDYNCZE_k{CONTROL:06d}.csv',singles)

if __name__=='__main__':main()
