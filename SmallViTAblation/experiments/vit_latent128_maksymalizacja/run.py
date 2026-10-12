"""Czy interwencja J128 poprawia detekcję? Oddziela oracle od transferu bez GT."""
import argparse,json,hashlib,time,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from core import intervene,project,tail
from model_snapshot import MalyViT,cele,dekoduj,_giou
from metryki_source import ap50,iou

def js(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_hash(net):
    h=hashlib.sha256()
    for name,x in net.state_dict().items():h.update(name.encode());h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()
def losses(o,targets):
    heat,ltrb,poz=targets;p=o[...,0].sigmoid().clamp(1e-4,1-1e-4);pos=(heat==1).float()
    focal=(-(p.log()*(1-p)**2*pos+(1-p).log()*p**2*(1-heat)**4*(1-pos))).sum((1,2))/pos.sum((1,2)).clamp(min=1)
    pr=o[...,1:].exp();g=_giou(pr.reshape(-1,4),ltrb.reshape(-1,4)).reshape(p.shape)
    box=((1-g)*poz).sum((1,2))/poz.sum((1,2)).clamp(min=1e-12)
    return focal+2*box
def film_weights(rows,ids):
    films=[rows[i]['film'] for i in ids];return np.array([1/len(set(films))/films.count(f) for f in films])
def summarize(records,rows,ids):
    w=film_weights(rows,ids);return {k:float(w@np.array([records[i][k] for i in ids])) for k in ('loss','q','confidence','fp_proxy')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--radius',type=float,default=.5);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);start=time.time()
    torch.set_num_threads(4);torch.manual_seed(20261004);np.random.seed(20261004)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();device='cuda'
    for name,digest in json.loads((a.input/'HASHES.json').read_text()).items():assert sha(a.input/name)==digest,name
    rows=json.loads((a.input/'manifest.json').read_text());cm=np.load(a.input/'channels.npz');J=torch.tensor(cm['channels0'],device=device,dtype=torch.long);sd=torch.tensor(cm['scaler_sd_train'][cm['channels0']],device=device,dtype=torch.float32)
    assert len(J)==128 and len(J.unique())==128 and (sd>0).all()
    if a.smoke:rows=sum(([r for r in rows if r['split']==s][:2] for s in ('train','val','test')),[])
    sets={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','val','test')}
    films={s:{rows[i]['film'] for i in ids} for s,ids in sets.items()}
    assert not (films['train']&films['val'] or films['train']&films['test'] or films['val']&films['test'])
    net=MalyViT().to(device).eval();ck=torch.load(a.input/'checkpoint.pt',map_location='cpu',weights_only=True);assert ck['krok']==500;net.load_state_dict(ck['net']);net.requires_grad_(False);original_hash=state_hash(net)
    zlist=[];rlist=[];targets=[];gt=[];base_outputs=[];cache={};checks=[]
    h1=net.bloki[-1].norm2.register_forward_pre_hook(lambda m,args:cache.update(r=args[0].detach()))
    h2=net.bloki[-1].mlp.act.register_forward_hook(lambda m,args,o:cache.update(z=o.detach()))
    import cv2
    with torch.no_grad():
        for row in rows:
            im=cv2.cvtColor(cv2.imread(str(a.input/row['image'])),cv2.COLOR_BGR2RGB);im=cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA)
            x=torch.from_numpy(im).permute(2,0,1)[None].float().to(device)/255
            o,_=net(x);re=tail(net,cache['z'],cache['r']);torch.testing.assert_close(re,o,rtol=1e-5,atol=2e-5)
            checks.append(float((re-o).abs().max()));zlist.append(cache['z'][0].cpu());rlist.append(cache['r'][0].cpu());base_outputs.append(o[0].cpu())
            bx,by,bw,bh=row['bbox_xywh'];g=np.array([[bx,by,bx+bw,by+bh]],np.float32);gt.append(g);targets.append(cele([g],20,30))
    h1.remove();h2.remove();Z=torch.stack(zlist);R=torch.stack(rlist);T=tuple(torch.cat([t[k] for t in targets]) for k in range(3));baseline=torch.stack(base_outputs)
    def batch(ids):return Z[ids].to(device),R[ids].to(device),tuple(t[ids].to(device) for t in T)
    def forward(ids,d):
        z,r,t=batch(ids);return tail(net,intervene(z,d,J,sd),r),t
    def evaluate(pred):
        rec=[];boxes,scores=dekoduj(pred.to(device));boxes=boxes.cpu().numpy();scores=scores.cpu().numpy();ls=losses(pred.to(device),tuple(t.to(device) for t in T)).cpu().numpy()
        for i in range(len(rows)):
            overlap=iou(boxes[i],gt[i][0]);rec.append(dict(loss=float(ls[i]),q=float((scores[i]*overlap).max()),confidence=float(scores[i].max()),fp_proxy=int(((scores[i]>=.5)&(overlap<.5)).sum())))
        summary={s:dict(**summarize(rec,rows,ids),AP50=ap50([(boxes[i],scores[i]) for i in ids],[gt[i] for i in ids])) for s,ids in sets.items()}
        return rec,summary
    # Actual-data proof: hook replacement agrees with cached-tail and only J changes.
    d=torch.full((1,600,128),.01,device=device,requires_grad=True);zz=Z[:1].to(device);rr=R[:1].to(device)
    edited=intervene(zz,d,J,sd);mask=torch.ones(768,device=device,dtype=torch.bool);mask[J]=False;assert torch.equal(edited[:,:,mask],zz[:,:,mask])
    row=rows[0];im=cv2.cvtColor(cv2.imread(str(a.input/row['image'])),cv2.COLOR_BGR2RGB);im=cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA);x=torch.from_numpy(im).permute(2,0,1)[None].float().to(device)/255
    h=net.bloki[-1].mlp.act.register_forward_hook(lambda m,args,o:intervene(o,d,J,sd))
    actual,_=net(x);h.remove();expected=tail(net,edited,rr);torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-5)
    loss=losses(expected,tuple(t[:1].to(device) for t in T)).sum();loss.backward();assert torch.isfinite(d.grad).all() and d.grad.abs().sum()>0
    js(out/'PREFLIGHT.json',dict(tail_max_error=max(checks),hook_error=float((actual-expected).abs().max()),gradient_norm=float(d.grad.norm()),untouched640=True,checkpoint_sha=sha(a.input/'checkpoint.pt'),channels_sha=sha(a.input/'channels.npz'),device=torch.cuda.get_device_name(),torch=torch.__version__,source_job=os.getenv('SLURM_JOB_ID')))
    print('PREFLIGHT PASS',flush=True)
    methods={};records={};records['baseline'],methods['baseline']=evaluate(baseline)
    # Shared 128-vector: trained only on train, epoch selected only on val.
    delta=torch.zeros(1,1,128,device=device,requires_grad=True);opt=torch.optim.Adam([delta],lr=.05);best=delta.detach().clone();bestval=methods['baseline']['val']['loss'];bestepoch=0;history=[]
    rng=np.random.default_rng(20261004);epochs=2 if a.smoke else 20
    def outputs(d):
        result=[]
        with torch.no_grad():
            for lo in range(0,len(rows),16):result.append(forward(list(range(lo,min(lo+16,len(rows)))),d)[0].cpu())
        return torch.cat(result)
    for epoch in range(1,epochs+1):
        order=rng.permutation(sets['train']);fw=film_weights(rows,sets['train']);wmap=dict(zip(sets['train'],fw))
        for lo in range(0,len(order),16):
            ids=order[lo:lo+16].tolist();o,t=forward(ids,delta);w=torch.tensor([wmap[i]*len(order) for i in ids],device=device)
            loss=(losses(o,t)*w).mean()+.01*delta.square().mean();opt.zero_grad();loss.backward();opt.step();project(delta,a.radius)
        # Validation only: do not inspect test during selection.
        with torch.no_grad():
            vals=[]
            for lo in range(0,len(sets['val']),16):
                ids=sets['val'][lo:lo+16];o,t=forward(ids,delta);vals.extend(losses(o,t).cpu().tolist())
            val=float(film_weights(rows,sets['val'])@np.array(vals))
        if val<bestval:bestval=val;best=delta.detach().clone();bestepoch=epoch
        history.append(dict(epoch=epoch,validation_loss=val));print('shared epoch',epoch,'val',val,flush=True)
    js(out/'SELECTION.json',dict(radius=a.radius,epoch=bestepoch,val_loss=bestval,history=history,criterion='film_weighted_validation_loss',test_used_for_selection=False))
    pred=outputs(best);records['shared'],methods['shared']=evaluate(pred)
    np.savez_compressed(out/'SHARED.npz',channels0=J.cpu().numpy(),delta_standardized=best.cpu().numpy(),delta_raw=(best*sd).cpu().numpy())
    for seed in (11,22,33):
        sign=torch.tensor(np.random.default_rng(seed).choice([-1.,1.],best.shape),device=device,dtype=torch.float32);name=f'shared_random{seed}'
        records[name],methods[name]=evaluate(outputs(best.abs()*sign))
    # Oracle per image, label-dependent, explicitly NOT a held-out prediction.
    oracle=torch.zeros(len(rows),600,128);oracle_outputs=[];steps=3 if a.smoke else 80
    for lo in range(0,len(rows),16):
        ids=list(range(lo,min(lo+16,len(rows))));z,r,t=batch(ids);dd=torch.zeros(len(ids),600,128,device=device,requires_grad=True);op=torch.optim.Adam([dd],lr=.05)
        with torch.no_grad():bestloss=losses(tail(net,z,r),t);bd=dd.detach().clone()
        for step in range(steps):
            oo=tail(net,intervene(z,dd,J,sd),r);per=losses(oo,t)+.01*dd.square().mean((1,2));op.zero_grad();per.mean().backward();assert torch.isfinite(dd.grad).all();op.step();project(dd,a.radius)
            with torch.no_grad():
                per=losses(tail(net,intervene(z,dd,J,sd),r),t)+.01*dd.square().mean((1,2));better=per<bestloss;bd[better]=dd[better];bestloss=torch.minimum(bestloss,per)
        oracle[ids]=bd.cpu()
        with torch.no_grad():oracle_outputs.append(tail(net,intervene(z,bd,J,sd),r).cpu())
        print('oracle',lo+len(ids),'/',len(rows),flush=True)
    records['oracle'],methods['oracle']=evaluate(torch.cat(oracle_outputs))
    for seed in (11,22,33):
        gen=torch.Generator().manual_seed(seed);random=oracle.abs()*(torch.randint(0,2,oracle.shape,generator=gen)*2-1);pred=[]
        with torch.no_grad():
            for lo in range(0,len(rows),16):
                ids=list(range(lo,min(lo+16,len(rows))));pred.append(forward(ids,random[ids].to(device))[0].cpu())
        name=f'oracle_random{seed}';records[name],methods[name]=evaluate(torch.cat(pred))
    assert oracle.abs().max()<=a.radius+1e-6 and state_hash(net)==original_hash
    basej=Z[:,:,J.cpu()].numpy();draw=oracle.numpy()*sd.cpu().numpy()
    np.savez_compressed(out/'TARGETS_ORACLE.npz',source_indices=np.array([r['index'] for r in rows]),channels0=J.cpu().numpy(),base=basej,delta_raw=draw,target=basej+draw,delta_standardized=oracle.numpy())
    # CIs for transfer, whole films, paired against baseline.
    ids=sets['test'];films=sorted({rows[i]['film'] for i in ids});bootrng=np.random.default_rng(1004);cis={}
    for name in ('shared','oracle'):
        arr=np.array([[np.mean([records[name][i][k]-records['baseline'][i][k] for i in ids if rows[i]['film']==f]) for k in ('loss','q')] for f in films])
        draws=arr[bootrng.integers(0,len(films),(1000,len(films)))].mean(1);cis[name]=dict(zip(('loss','q'),np.quantile(draws,[.025,.975],axis=0).T.tolist()))
    js(out/'RECORDS.json',dict(rows=rows,methods=records))
    js(out/'RESULT.json',dict(status='complete',radius=a.radius,seed=20261004,smoke=a.smoke,methods=methods,paired_film_CI95=cis,selection_epoch=bestepoch,selection_val_loss=bestval,weights_buffers_unchanged=True,seconds=time.time()-start,oracle_is_label_optimized=True,shared_test_uses_no_labels_for_intervention=True,not_image_inversion=True))
    print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
