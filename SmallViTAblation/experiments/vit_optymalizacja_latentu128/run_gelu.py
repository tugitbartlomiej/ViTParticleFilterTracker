"""Czy poprawa oracle utrzymuje się po wymuszeniu dolnej granicy GELU?

Diagnostyka po pierwszym wyniku, nie nowy test potwierdzający ani inversion.
"""
import argparse,json,time,os
from pathlib import Path
import numpy as np,torch,cv2
from core import intervene,project,tail
from run import js,sha,state_hash,losses,film_weights,summarize
from model_snapshot import MalyViT,cele,dekoduj
from metryki_source import iou,ap50

FLOOR=-0.16997120  # konserwatywnie powyżej minimum dokładnego GELU

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);start=time.time();torch.manual_seed(20261004);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name,digest in json.loads((a.input/'HASHES.json').read_text()).items():assert sha(a.input/name)==digest
    rows=json.loads((a.input/'manifest.json').read_text());cm=np.load(a.input/'channels.npz');J=torch.tensor(cm['channels0'],device='cuda',dtype=torch.long);sd=torch.tensor(cm['scaler_sd_train'][cm['channels0']],device='cuda',dtype=torch.float32)
    if a.smoke:rows=rows[:2]
    net=MalyViT().cuda().eval();ck=torch.load(a.input/'checkpoint.pt',map_location='cpu',weights_only=True);assert ck['krok']==500;net.load_state_dict(ck['net']);net.requires_grad_(False);digest=state_hash(net)
    assert net.bloki[-1].mlp.act.approximate=='none','Floor requires exact GELU'
    cache={};hh=[net.bloki[-1].norm2.register_forward_pre_hook(lambda m,args:cache.update(r=args[0].detach())),net.bloki[-1].mlp.act.register_forward_hook(lambda m,args,o:cache.update(z=o.detach()))]
    records={m:[] for m in ('baseline','oracle_gelu','random11','random22','random33')};preds={m:[] for m in records};gts=[];base_all=[];delta_all=[];norm_all=[];floor_count=0;total=0;max_noop=0.
    for lo in range(0,len(rows),16):
        group=rows[lo:lo+16];images=[];gt=[]
        for row in group:
            im=cv2.cvtColor(cv2.imread(str(a.input/row['image'])),cv2.COLOR_BGR2RGB);images.append(cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA))
            x,y,w,h=row['bbox_xywh'];gt.append(np.array([[x,y,x+w,y+h]],np.float32))
        x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().cuda()/255
        with torch.no_grad():baseline,_=net(x);z=cache['z'];res=cache['r'];check=tail(net,z,res);torch.testing.assert_close(check,baseline,rtol=1e-5,atol=2e-5);max_noop=max(max_noop,float((check-baseline).abs().max()))
        t=tuple(v.cuda() for v in cele(gt,20,30));delta=torch.zeros(len(group),600,128,device='cuda',requires_grad=True)
        # Keep exact zero feasible even for fp32 values a few ulps below FLOOR.
        lower=torch.maximum(torch.full_like(delta,-1.),torch.minimum((FLOOR-z[:,:,J])/sd,torch.zeros_like(delta)))
        opt=torch.optim.Adam([delta],lr=.05)
        with torch.no_grad():bestloss=losses(baseline,t);best=delta.detach().clone()
        for step in range(3 if a.smoke else 80):
            o=tail(net,intervene(z,delta,J,sd),res);per=losses(o,t)+.01*delta.square().mean((1,2));opt.zero_grad();per.mean().backward();assert torch.isfinite(delta.grad).all();opt.step()
            with torch.no_grad():
                delta.copy_(torch.maximum(delta.clamp(max=1.),lower));o=tail(net,intervene(z,delta,J,sd),res);per=losses(o,t)+.01*delta.square().mean((1,2));better=per<bestloss;best[better]=delta[better];bestloss=torch.minimum(bestloss,per)
        edits={'baseline':torch.zeros_like(best),'oracle_gelu':best}
        for seed in (11,22,33):
            gen=torch.Generator(device='cuda').manual_seed(seed+lo);sign=torch.randint(0,2,best.shape,generator=gen,device='cuda')*2-1;rd=best.abs()*sign
            # Positive sign replaces infeasible negative signs, preserving exact magnitudes.
            rd=torch.where(rd<lower,rd.abs(),rd);assert (rd>=lower-1e-7).all();edits[f'random{seed}']=rd
        with torch.no_grad():
            for mode,dd in edits.items():
                changed=intervene(z,dd,J,sd);mask=torch.ones(768,device='cuda',dtype=torch.bool);mask[J]=False;assert torch.equal(changed[:,:,mask],z[:,:,mask]);assert float(changed[:,:,J].min())>=FLOOR-1e-6
                o=tail(net,changed,res);ll=losses(o,t).cpu().numpy();boxes,scores=dekoduj(o);boxes=boxes.cpu().numpy();scores=scores.cpu().numpy()
                for k,g in enumerate(gt):
                    overlap=iou(boxes[k],g[0]);records[mode].append(dict(loss=float(ll[k]),q=float((scores[k]*overlap).max()),confidence=float(scores[k].max()),fp_proxy=int(((scores[k]>=.5)&(overlap<.5)).sum())))
                    preds[mode].append((boxes[k],scores[k]))
            target=z[:,:,J]+best*sd;assert best.abs().max()<=1.000001
            floor_count+=int((target<FLOOR-1e-6).sum());total+=target.numel();base_all.append(z[:,:,J].cpu().numpy());delta_all.append((best*sd).cpu().numpy());norm_all.append(best.cpu().numpy());gts.extend(gt)
        print('gelu oracle',lo+len(group),'/',len(rows),flush=True)
    for h in hh:h.remove()
    assert state_hash(net)==digest and floor_count==0
    sets={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','val','test')};methods={}
    for mode in records:
        methods[mode]={s:dict(**summarize(records[mode],rows,ids),AP50=ap50([preds[mode][i] for i in ids],[gts[i] for i in ids])) for s,ids in sets.items() if ids}
    base=np.concatenate(base_all);delta=np.concatenate(delta_all)
    np.savez_compressed(a.output/'TARGETS_GELU.npz',source_indices=np.array([r['index'] for r in rows]),channels0=J.cpu().numpy(),base=base,delta_raw=delta,target=base+delta,delta_standardized=np.concatenate(norm_all))
    js(a.output/'RECORDS.json',dict(rows=rows,methods=records))
    js(a.output/'RESULT.json',dict(status='complete',methods=methods,gelu_floor=FLOOR,floor_violations=floor_count,coordinates=total,max_noop_error=max_noop,weights_buffers_unchanged=True,seconds=time.time()-start,job=os.getenv('SLURM_JOB_ID'),device=torch.cuda.get_device_name(),checkpoint_sha=sha(a.input/'checkpoint.pt'),channels_sha=sha(a.input/'channels.npz'),posthoc_diagnostic=True,oracle_not_generalization=True,range_feasible_not_image_realizable=True,random_control='random signs, infeasible negative signs replaced by positive; exact absolute changes preserved'))
    print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
