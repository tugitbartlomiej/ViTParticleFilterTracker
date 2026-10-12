"""Post-hoc oracle: poprawia64 najtrudniejsze obrazy przez cel bezpośrednio na głowie."""
import json,time
import numpy as np,torch
import torch.nn.functional as F
from maximize import OUT,INPUT,CKPT,FLOOR,load_data,stats,js,zapisz_wynik
from model_snapshot import MalyViT
from core import intervene,tail
from run import state_hash

def main():
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261004);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=json.loads((OUT/'RESULT.json').read_text());rows=json.loads((INPUT/'manifest.json').read_text());scores=old['optimized'];ids=sorted(range(400),key=lambda i:(scores['per_image_top_iou'][i],scores['per_image_q'][i]))[:64];js(OUT/'REFINE_SELECTION.json',dict(ids=ids,criterion='lowest top1 IoU then quality, posthoc oracle',lr=.1,steps=600))
    n=np.load(OUT/'TARGETS_FULL400.npz');J=torch.tensor(n['channels0'],device='cuda');delta_all=n['delta_raw'].copy();base=n['base'];maps=np.load(OUT/'DETECTIONS_FULL400.npz')['optimized'].copy()
    net=MalyViT().cuda().eval();net.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True)['net']);net.requires_grad_(False);before=state_hash(net);Z,R,_,gt,T,err=load_data([rows[i] for i in ids],net)
    for lo in range(0,len(ids),8):
        group=ids[lo:lo+8];z=Z[lo:lo+8].cuda();res=R[lo:lo+8].cuda();d=torch.tensor(delta_all[group],device='cuda',requires_grad=True);lower=torch.minimum(FLOOR-z[:,:,J],torch.zeros_like(d));centers=[];dist=[]
        for g in gt[lo:lo+8]:
            x0,y0,x1,y1=g[0];cx=int(np.clip(round((x0+x1)/32-.5),0,29));cy=int(np.clip(round((y0+y1)/32-.5),0,19));centers.append(cy*30+cx);px=(cx+.5)*16;py=(cy+.5)*16;dist.append(np.maximum(np.array([px-x0,py-y0,x1-px,y1-py])/16,.001))
        center=torch.tensor(centers,device='cuda');targetlog=torch.tensor(np.stack(dist),device='cuda',dtype=torch.float32).log();batch=torch.arange(len(group),device='cuda');negmask=1-F.one_hot(center,600).float()
        def objective(o):
            flat=o.reshape(len(group),600,5);logits=flat[:,:,0];lc=F.cross_entropy(logits,center,reduction='none')+F.softplus(-logits[batch,center]);ln=(F.softplus(logits)*negmask).sum(1)/599
            box=F.smooth_l1_loss(flat[batch,center,1:],targetlog,reduction='none').mean(1)
            return lc+ln+5*box
        with torch.no_grad():bestloss=objective(tail(net,intervene(z,d,J,1.),res));best=d.detach().clone()
        opt=torch.optim.Adam([d],lr=.1)
        for step in range(600):
            loss=objective(tail(net,intervene(z,d,J,1.),res));opt.zero_grad();loss.mean().backward();assert torch.isfinite(d.grad).all();opt.step()
            with torch.no_grad():
                d.copy_(torch.maximum(d,lower));loss=objective(tail(net,intervene(z,d,J,1.),res));better=loss<bestloss;best[better]=d[better];bestloss=torch.minimum(bestloss,loss)
        with torch.no_grad():
            target=z[:,:,J]+best;assert target.min()>=FLOOR-1e-6;maps[group]=tail(net,intervene(z,best,J,1.),res).cpu().numpy();delta_all[group]=best.cpu().numpy()
        print('REFINED',lo+len(group),'/',len(ids),flush=True)
    assert state_hash(net)==before
    fullgt=[]
    for r in rows:x,y,w,h=r['bbox_xywh'];fullgt.append(np.array([[x,y,x+w,y+h]],np.float32))
    from model_snapshot import cele
    metrics=stats([torch.tensor(maps)],fullgt,cele(fullgt,20,30))
    np.savez_compressed(OUT/'TARGETS_REFINED400.npz',source_indices=n['source_indices'],channels0=n['channels0'],base=base,delta_raw=delta_all,target=base+delta_all)
    np.savez_compressed(OUT/'DETECTIONS_REFINED400.npz',optimized=maps)
    report=dict(ziarno=20261004,status='complete',metrics=metrics,selected_ids=ids,baseline_ap50=old['baseline']['AP50'],previous_ap50=old['optimized']['AP50'],posthoc_oracle=True,weights_buffers_unchanged=True,range_GELU=True,seconds=time.time()-start)
    js(OUT/'REFINE_RESULT.json',report);zapisz_wynik(__file__,'refine64_k1000',json.loads((OUT/'REFINE_RESULT.json').read_text()))
    print({k:metrics[k] for k in ('AP50','AP75','AP50_95','mean_top_iou','mean_top_confidence','fraction_top_iou50_conf90')},flush=True)
if __name__=='__main__':main()
