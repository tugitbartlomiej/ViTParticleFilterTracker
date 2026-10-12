"""Czy silna interwencja J128 wystarczy do niemal doskonałej detekcji?

Obala ją brak poprawy AP/loss względem baseline i kontroli. Oracle zna GT.
"""
import argparse,json,time,sys
from pathlib import Path
import numpy as np,torch,cv2
from core import intervene,tail
from run import js,sha,state_hash,losses
from model_snapshot import MalyViT,cele,dekoduj
from metryki_source import iou
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik
INPUT=ROOT/'outputs/vit_optymalizacja_latentu128/input'
OUT=ROOT/'outputs/vit_latent128_maksymalizacja'
CKPT=Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k1000.pt')
FLOOR=-0.16997120

def ap(pred,gt,threshold):
    candidates=sorted([(float(s),i,b) for i,(boxes,scores) in enumerate(pred) for b,s in zip(boxes,scores)],key=lambda x:-x[0]);used=set();tp=[]
    for _,i,b in candidates:
        hit=i not in used and float(iou(b,gt[i][0]))>=threshold
        if hit:used.add(i)
        tp.append(hit)
    tp=np.cumsum(tp);rec=tp/len(gt);prec=tp/np.arange(1,len(tp)+1);prec=np.maximum.accumulate(prec[::-1])[::-1]
    idx=np.searchsorted(rec,np.linspace(0,1,101));return float(np.mean([prec[i] if i<len(prec) else 0 for i in idx]))

def stats(outputs,gt,targets):
    o=torch.cat(outputs);boxes,scores=dekoduj(o);boxes=boxes.numpy();scores=scores.numpy();ls=losses(o,targets).numpy();pred=list(zip(boxes,scores));qual=[];topiou=[]
    for i in range(len(gt)):
        overlap=iou(boxes[i],gt[i][0]);qual.append(float((scores[i]*overlap).max()));topiou.append(float(overlap[scores[i].argmax()]))
    aps={f'{t:.2f}':ap(pred,gt,t) for t in np.arange(.5,.951,.05)}
    return dict(AP50=aps['0.50'],AP75=aps['0.75'],AP50_95=float(np.mean(list(aps.values()))),mean_loss=float(ls.mean()),mean_q=float(np.mean(qual)),mean_top_iou=float(np.mean(topiou)),mean_top_confidence=float(scores.max(1).mean()),fraction_top_iou50_conf90=float(np.mean((np.array(topiou)>=.5)&(scores.max(1)>=.9))),per_image_loss=ls,per_image_q=qual,per_image_top_iou=topiou)

def load_data(rows,net):
    cache={};h1=net.bloki[-1].norm2.register_forward_pre_hook(lambda m,a:cache.update(r=a[0].detach()));h2=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o.detach()))
    zs=[];rs=[];os=[];gt=[];err=0.
    with torch.no_grad():
        for lo in range(0,len(rows),8):
            ims=[]
            for r in rows[lo:lo+8]:
                im=cv2.cvtColor(cv2.imread(str(INPUT/r['image'])),cv2.COLOR_BGR2RGB);ims.append(cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA));x,y,w,h=r['bbox_xywh'];gt.append(np.array([[x,y,x+w,y+h]],np.float32))
            x=torch.from_numpy(np.stack(ims)).permute(0,3,1,2).float().cuda()/255;o,_=net(x);check=tail(net,cache['z'],cache['r']);torch.testing.assert_close(o,check,rtol=1e-5,atol=2e-5);err=max(err,float((o-check).abs().max()));zs.append(cache['z'].cpu());rs.append(cache['r'].cpu());os.append(o.cpu())
    h1.remove();h2.remove();return torch.cat(zs),torch.cat(rs),os,gt,cele(gt,20,30),err

def optimize(net,Z,R,T,J,lr,steps,out):
    results=[];deltas=[];hist=[]
    for lo in range(0,len(Z),8):
        z=Z[lo:lo+8].cuda();r=R[lo:lo+8].cuda();t=tuple(v[lo:lo+8].cuda() for v in T);d=torch.zeros(z.shape[0],600,128,device='cuda',requires_grad=True)
        lower=torch.minimum(FLOOR-z[:,:,J],torch.zeros_like(d));optimizer=torch.optim.Adam([d],lr=lr)
        with torch.no_grad():bestloss=losses(tail(net,z,r),t);best=d.detach().clone()
        for step in range(steps):
            o=tail(net,intervene(z,d,J,1.),r);l=losses(o,t);optimizer.zero_grad();l.mean().backward();assert torch.isfinite(d.grad).all();optimizer.step()
            with torch.no_grad():
                d.copy_(torch.maximum(d,lower));current=losses(tail(net,intervene(z,d,J,1.),r),t);better=current<bestloss;best[better]=d[better];bestloss=torch.minimum(bestloss,current)
            if (step+1)%100==0:
                h=dict(offset=lo,step=step+1,mean_best_loss=float(bestloss.mean()),max_abs_delta=float(best.abs().max()));hist.append(h);print(json.dumps(h),flush=True)
        with torch.no_grad():
            changed=intervene(z,best,J,1.);mask=torch.ones(768,dtype=torch.bool,device='cuda');mask[J]=False;assert torch.equal(changed[:,:,mask],z[:,:,mask]);assert float(changed[:,:,J].min())>=FLOOR-1e-6;results.append(tail(net,changed,r).cpu());deltas.append(best.cpu())
        js(out/'PROGRESS.json',dict(done=lo+len(z),total=len(Z),lr=lr,steps=steps))
    return results,torch.cat(deltas),hist

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','full'],required=True);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261004);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=time.time()
    for name,digest in json.loads((INPUT/'HASHES.json').read_text()).items():assert sha(INPUT/name)==digest
    rows=json.loads((INPUT/'manifest.json').read_text());cm=np.load(INPUT/'channels.npz');J=torch.tensor(cm['channels0'],device='cuda',dtype=torch.long)
    if a.stage=='pilot':
        chosen=[];seen=set()
        for r in rows:
            if r['split']=='train' and r['film'] not in seen:chosen.append(r);seen.add(r['film'])
            if len(chosen)==12:break
        rows=chosen
    net=MalyViT().cuda().eval();ck=torch.load(CKPT,map_location='cpu',weights_only=True);assert ck['krok']==1000;net.load_state_dict(ck['net']);net.requires_grad_(False);assert net.bloki[-1].mlp.act.approximate=='none';before=state_hash(net)
    Z,R,base,gt,T,err=load_data(rows,net);baseline=stats(base,gt,T);js(OUT/f'BASELINE_{a.stage}.json',baseline)
    if a.stage=='pilot':
        candidates=[]
        for lr in (.03,.1,.3):
            folder=OUT/f'pilot_lr{lr}';folder.mkdir(exist_ok=True);pred,delta,history=optimize(net,Z,R,T,J,lr,500,folder);s=stats(pred,gt,T);js(folder/'RESULT.json',dict(lr=lr,metrics=s,history=history));candidates.append(dict(lr=lr,loss=s['mean_loss'],AP50=s['AP50']))
        chosen=min(candidates,key=lambda c:(c['loss'],c['lr']));js(OUT/'SELECTION.json',dict(selected=chosen,candidates=candidates,pilot_source_indices=[r['index'] for r in rows],criterion='minimum pilot detection loss',checkpoint_sha=sha(CKPT),channels_sha=sha(INPUT/'channels.npz')));print('SELECTION',chosen,flush=True)
    else:
        selected=json.loads((OUT/'SELECTION.json').read_text());assert selected['checkpoint_sha']==sha(CKPT);assert selected['channels_sha']==sha(INPUT/'channels.npz');lr=selected['selected']['lr'];pred,delta,history=optimize(net,Z,R,T,J,lr,800,OUT);metrics=stats(pred,gt,T)
        generator=torch.Generator().manual_seed(11);random=delta.abs()*(torch.randint(0,2,delta.shape,generator=generator)*2-1);lower=torch.minimum(FLOOR-Z[:,:,J.cpu()],torch.zeros_like(delta));random=torch.where(random<lower,random.abs(),random);rp=[]
        with torch.no_grad():
            for lo in range(0,len(Z),8):rp.append(tail(net,intervene(Z[lo:lo+8].cuda(),random[lo:lo+8].cuda(),J,1.),R[lo:lo+8].cuda()).cpu())
        control=stats(rp,gt,T);basej=Z[:,:,J.cpu()].numpy();dd=delta.numpy();np.savez_compressed(OUT/'TARGETS_FULL400.npz',source_indices=np.array([r['index'] for r in rows]),channels0=J.cpu().numpy(),base=basej,delta_raw=dd,target=basej+dd)
        np.savez_compressed(OUT/'DETECTIONS_FULL400.npz',baseline=torch.cat(base).numpy(),optimized=torch.cat(pred).numpy(),random=torch.cat(rp).numpy())
        result=dict(ziarno=20261004,status='complete',checkpoint_step=1000,checkpoint_sha=sha(CKPT),channels_sha=sha(INPUT/'channels.npz'),baseline=baseline,optimized=metrics,random_control=control,history=history,lr=lr,steps=800,images=len(rows),oracle_uses_all_labels=True,gelu_floor=FLOOR,min_target=float((basej+dd).min()),max_abs_delta=float(abs(dd).max()),mean_token_delta_norm=float(np.linalg.norm(dd,axis=2).mean()),no_other_channels_changed=True,weights_buffers_unchanged=state_hash(net)==before,tail_max_error=err,device=torch.cuda.get_device_name(),seconds=time.time()-start)
        assert result['weights_buffers_unchanged'];js(OUT/'RESULT.json',result);zapisz_wynik(__file__,'full400_k1000',result)
        print('FINAL', {m:{k:result[m][k] for k in ('AP50','AP50_95','mean_loss','mean_top_confidence','mean_top_iou')} for m in ('baseline','optimized','random_control')},flush=True)
    assert state_hash(net)==before;print('COMPLETE',a.stage,time.time()-start,flush=True)
if __name__=='__main__':main()
