"""Czy kierunki wpływają selektywnie na detekcję? Random/sham i noop są kontrolą."""
import json,time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,torch
from dane import HERE,DATA,load,image,tensor,init_net,weights,js
from model_snapshot import cele,strata,dekoduj
from metryki_source import iou,ap50
from identyfikacja import score,metrics

METHODS=['nohook','shamnoop','identity','full_patch','random11','random22','random33']

def quality(out,gt):
    heat,ltrb,poz=cele([gt],20,30)
    loss,parts=strata(out,heat.to(out.device),ltrb.to(out.device),poz.to(out.device))
    bx,sc=dekoduj(out);bx=bx[0].cpu().numpy();sc=sc[0].cpu().numpy()
    return dict(loss=float(loss),heat=parts['heat'],box=parts['box'],q=float((sc*iou(bx,gt[0])).max()),max_confidence=float(sc.max())),(bx,sc)

def make_hook(indices,negative,U,mode,diag,net,single=None):
    ids=torch.as_tensor(indices,dtype=torch.long,device='cuda');neg=torch.as_tensor(negative,device='cuda')
    proj=torch.as_tensor(U,dtype=torch.float32,device='cuda')
    def hook(module,args,out):
        if mode=='shamnoop':return out.clone()
        cur=out[0,ids];D=cur-neg;identity=(D@proj)@proj.T
        if mode=='identity':change=identity
        elif mode=='full_patch':change=D
        elif mode.startswith('random') or mode.startswith('fc2random'):
            R=torch.as_tensor(diag['R'],dtype=torch.float32,device='cuda');change=(D@R)@R.T
            if mode.startswith('fc2random'):
                W=net.bloki[-1].mlp.fc2.weight.T
                target=torch.linalg.vector_norm(identity@W,dim=1,keepdim=True);den=torch.linalg.vector_norm(change@W,dim=1,keepdim=True)
            else:
                target=torch.linalg.vector_norm(identity,dim=1,keepdim=True);den=torch.linalg.vector_norm(change,dim=1,keepdim=True)
            assert not ((den<1e-12)&(target>1e-8)).any(),'degenerate random projection'
            change=change*(target/den.clamp(min=1e-12))
            if mode.startswith('fc2random'):
                diag['max_fc2_token_norm_mismatch']=float((torch.linalg.vector_norm(change@W,dim=1)-target[:,0]).abs().max())
            else:diag['max_token_norm_mismatch']=float((torch.linalg.vector_norm(change,dim=1)-target[:,0]).abs().max())
            diag['zero_random_denominators']=int((den<1e-12).sum())
        else:raise ValueError(mode)
        if single is not None:
            keep=torch.zeros(len(ids),1,device='cuda');keep[single]=1;change=change*keep
        diag.update(hidden_norm=float(torch.linalg.vector_norm(change)),functional_fc2_norm=float(torch.linalg.vector_norm(change@net.bloki[-1].mlp.fc2.weight.T)),identity_hidden_norm=float(torch.linalg.vector_norm(identity)))
        replacement=out.clone();replacement[0,ids]=cur-change
        mask=torch.ones(600,dtype=torch.bool,device='cuda');mask[ids]=False
        assert torch.equal(replacement[0,mask],out[0,mask])
        return replacement
    return hook

def bootstrap_detection(rows,test,records,pred,gt,n=200):
    films=sorted({rows[i]['video_id'] for i in test});rng=np.random.default_rng(910)
    keys=['loss','q','max_confidence'];boot={m:[] for m in METHODS};apboot={m:[] for m in METHODS}
    rec={m:[r for r in records if r['method']==m] for m in METHODS}
    for _ in range(n):
        picked=rng.choice(films,len(films),replace=True);sel=[];w=[]
        for f in picked:
            ids=[j for j,i in enumerate(test) if rows[i]['video_id']==f];sel.extend(ids);w.extend([1/len(ids)/len(films)]*len(ids))
        for m in METHODS:
            vals=np.array([[r[k] for k in keys] for r in rec[m]])
            boot[m].append(np.array(w)@vals[sel]);apboot[m].append(ap50([pred[m][j] for j in sel],[gt[j] for j in sel]))
    result={};base=np.array(boot['nohook']);baseAP=np.array(apboot['nohook'])
    for m in METHODS:
        values=np.array(boot[m]);aps=np.array(apboot[m]);result[m]=dict(
            film_bootstrap95={k:list(np.quantile(values[:,j],[.025,.975])) for j,k in enumerate(keys)},
            paired_delta_film_bootstrap95={k:list(np.quantile((values-base)[:,j],[.025,.975])) for j,k in enumerate(keys)},
            AP50_bootstrap95=list(np.quantile(aps,[.025,.975])),AP50_paired_delta_bootstrap95=list(np.quantile(aps-baseAP,[.025,.975])))
    return result

def main():
    start=time.time();rows,t=load();cache=dict(np.load(HERE/'TOKENY_PARY.npz'));fit=dict(np.load(HERE/'PODPRZESTRZEN.npz'));U=fit['U'];rank=U.shape[1]
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test']);net=init_net();bn={n:b.clone() for n,b in net.named_buffers()}
    randoms={seed:np.linalg.qr(np.random.default_rng(seed).normal(size=(768,rank)))[0].astype(np.float32) for seed in (11,22,33)}
    records=[];pred={m:[] for m in METHODS};outputs={m:[] for m in METHODS};gtall=[];center=[]
    with torch.no_grad():
        for j,i in enumerate(test):
            row=rows[i];x=tensor([image(DATA/row['image_path'])]);xywh=np.array(row['bbox_source_480'],np.float32)
            assert xywh[2]>0 and xywh[3]>0
            gt=np.array([[xywh[0],xywh[1],xywh[0]+xywh[2],xywh[1]+xywh[3]]],np.float32);gtall.append(gt)
            a,b=cache['offsets'][i:i+2];indices=cache['token_ids'][a:b];negative=cache['states'][a:b,1]
            baseline=None
            for mode in METHODS:
                diag={}
                if mode.startswith('random'):diag['R']=randoms[int(mode[6:])]
                handle=None
                if mode!='nohook':handle=net.bloki[-1].mlp.act.register_forward_hook(make_hook(indices,negative,U,mode,diag,net))
                o,_=net(x)
                if handle is not None:handle.remove()
                assert np.isfinite(o.cpu().numpy()).all()
                if mode=='nohook':baseline=o.clone()
                if mode=='shamnoop':assert torch.equal(o,baseline),'noop output differs'
                m,d=quality(o,gt);diag.pop('R',None)
                records.append(dict(source_index=int(i),frame_id=row['frame_id'],film=row['video_id'],method=mode,**m,**diag))
                pred[mode].append(d);outputs[mode].append(o[0].cpu().numpy())
            g=gt[0];cx=int(np.clip(round(((g[0]+g[2])/2)/16-.5),0,29));cy=int(np.clip(round(((g[1]+g[3])/2)/16-.5),0,19))
            peak=int(baseline[0,:,:,0].reshape(-1).argmax());center.append(dict(source_index=int(i),GT_center_token0=cy*30+cx,GT_center_in_tool_roi=bool(t['tool'][i,cy*30+cx]),baseline_peak_token0=peak,baseline_peak_in_tool_roi=bool(t['tool'][i,peak])))
            if (j+1)%20==0:print('detector intervention',j+1,'/',len(test),round(time.time()-start,1),'s',flush=True)
    assert all(torch.equal(b,bn[n]) for n,b in net.named_buffers())
    js(HERE/'DETEKCJA_OBRAZY.json',records);js(HERE/'CENTRA_ROI.json',center)
    w=weights(rows,test);boot=bootstrap_detection(rows,test,records,pred,gtall)
    summary={}
    for mode in METHODS:
        rec=[r for r in records if r['method']==mode];summary[mode]=dict(AP50=ap50(pred[mode],gtall),n=len(rec),**{k:float(w@np.array([r[k] for r in rec])) for k in ('loss','heat','box','q','max_confidence')},**boot[mode])
    np.savez_compressed(HERE/'MAPY_INTERWENCJI.npz',source_indices=test,**{m:np.array(outputs[m],np.float32) for m in METHODS})
    # Six deterministically chosen angle-spread test cases, before per-token scores are inspected.
    examples=[]
    for binid in (0,2,4,6,8,10):
        candidates=[i for i in test if int(rows[i]['angle_axis_deg']//15)==binid]
        if candidates:examples.append(int(candidates[0]))
    js(HERE/'PRZYKLADY.json',examples)
    individual=[];check=[]
    for i in examples:
        j=list(test).index(i);base=outputs['nohook'][j];changed=outputs['identity'][j];gt=gtall[j];base_metrics,_=quality(torch.tensor(base[None]),gt)
        a,b=cache['offsets'][i:i+2];ids=cache['token_ids'][a:b];values=[]
        for local,k in enumerate(ids):
            patched=base.copy();patched[k//30,k%30]=changed[k//30,k%30]
            mm,_=quality(torch.tensor(patched[None]),gt)
            values.append(dict(token0=int(k),token1=int(k+1),row=int(k//30),column=int(k%30),coverage=float(cache['coverage'][a+local]),delta_q=mm['q']-base_metrics['q'],delta_loss=mm['loss']-base_metrics['loss']))
        values.sort(key=lambda r:abs(r['delta_q']),reverse=True)
        individual.append(dict(source_index=i,frame_id=rows[i]['frame_id'],baseline=base_metrics,token_influence=values))
        if len(check)<2:
            chosen=values[0]['token0'];local=int(np.flatnonzero(ids==chosen)[0]);diag={}
            x=tensor([image(DATA/rows[i]['image_path'])]);handle=net.bloki[-1].mlp.act.register_forward_hook(make_hook(ids,cache['states'][a:b,1],U,'identity',diag,net,single=local))
            with torch.no_grad():actual,_=net(x)
            handle.remove();expected=base.copy();expected[chosen//30,chosen%30]=changed[chosen//30,chosen%30]
            error=float(np.max(np.abs(actual[0].cpu().numpy()-expected)));assert error<2e-5
            check.append(dict(source_index=i,token0=chosen,max_output_error=error,passed=True))
    js(HERE/'CZASTECZKI_WPLYW.json',individual)
    # All root-generated sham controls: separate original tool and edit ROIs.
    sham=[];sham_counts={}
    for tag,sham_path in [('mixed',HERE/'controls_sham/MANIFEST.json'),('heldout',HERE/'controls_sham/heldout/MANIFEST.json')]:
        if not sham_path.exists():continue
        current=[];shamstates=[]
        entries=json.loads(sham_path.read_text(encoding='utf-8'))['records']
        probe={k:fit['probe_'+k] for k in ('mu','sd','coef','intercept')}
        with torch.no_grad():
            for entry in entries:
                if entry['status']!='generated':continue
                i=entry['source_index'];_,z=net(tensor([image(DATA/rows[i]['image_path']),image(entry['synthetic_path'])]));z=z.cpu().numpy();D=z[0]-z[1]
                coverage=np.array(entry['coverage']);si=np.flatnonzero(coverage>0);ti=np.flatnonzero(t['tool'][i]);tc=t['area_coverage'][i,ti]
                toolpool=(z[:,ti]*tc[None,:,None]).sum(1)/tc.sum();orig=cache['pooled'][i,0]
                pv=(toolpool@U-probe['mu'])/probe['sd']
                logits=pv@probe['coef']+probe['intercept'];prob=1/(1+np.exp(-np.clip(logits,-60,60)))
                current.append(dict(control_set=tag,source_index=i,frame_id=rows[i]['frame_id'],split=rows[i]['split'],angle_native_deg=rows[i]['angle_axis_deg'],
                    original_tool_roi_delta_l2=float(np.average(np.linalg.norm(D[ti],axis=1),weights=tc)),original_tool_roi_projection_l2=float(np.average(np.linalg.norm(D[ti]@U,axis=1),weights=tc)),
                    sham_roi_delta_l2=float(np.average(np.linalg.norm(D[si],axis=1),weights=coverage[si])),sham_roi_projection_l2=float(np.average(np.linalg.norm(D[si]@U,axis=1),weights=coverage[si])),
                    nominal_token_roi_overlap=len(set(si)&set(ti)),presence_probability_original=float(prob[0]),presence_probability_sham=float(prob[1]),probability_delta=float(prob[0]-prob[1]),
                    original_presence_call=bool(prob[0]>=fit['threshold']),sham_presence_call=bool(prob[1]>=fit['threshold'])))
                joint=np.union1d(si,ti);shamstates.append(dict(source_index=i,ids=joint,states=z[:,joint].transpose(1,0,2)))
        js(HERE/f'SHAM_DIAGNOSTYKA_{tag}.json',current)
        np.savez_compressed(HERE/f'TOKENY_SHAM_{tag}.npz',**{f'ids_{r["source_index"]}':r['ids'] for r in shamstates},**{f'states_{r["source_index"]}':r['states'] for r in shamstates})
        sham.extend(current);sham_counts[tag]=len(current)
    report=dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),summary=summary,test_images=len(test),test_films=len({rows[i]['video_id'] for i in test}),
        GT_center_in_roi=sum(r['GT_center_in_tool_roi'] for r in center),baseline_peak_in_roi=sum(r['baseline_peak_in_tool_roi'] for r in center),
        noop_exact=True,buffers_unchanged=True,all_finite=True,individual_patch_real_hook_checks=check,sham_count=len(sham),sham_sets_separate_counts=sham_counts,
        random_max_token_norm_mismatch=max(r.get('max_token_norm_mismatch',0) for r in records),random_degenerate_tokens=sum(r.get('zero_random_denominators',0) for r in records),seconds=time.time()-start)
    js(HERE/'DETEKCJA.json',report)
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<250e6 and shutil.disk_usage(HERE).free>1e9
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
