"""Czy uczenie wag daje naturalny cel GELU? Agreement<90 lub błędna ramka obala sukces w budżecie."""
import os,sys,time,json,csv,math,platform,subprocess,traceback,copy,shutil
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import numpy as np,torch,timm
from model_snapshot import MalyViT,dekoduj,cele,strata
from dane import HERE,ROOT,SOURCE,OUT,load,sha
sys.path.insert(0,str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik

CONFIG=dict(seed=20261005,steps=1000,start_step=1000,lr=0.00003,min_lr=0.00003,lr_schedule='constant_continuation',retention=10.0,detection_weight=0.02,
            pause_seconds=0.03,agreement_threshold=90.0,iou_threshold=0.90,confidence_threshold=0.90,
            plateau_window=250,plateau_gain_pp=0.2,precision='float32',autocast=False,tf32=False)
ARCH=dict(name='MalyViT',dim=192,glebokosc=6,glowy=3,mlp=4.0,model_source_sha256=sha(HERE/'model_snapshot.py'))
_REFERENCE=None
PREVIOUS=ROOT/'outputs/vit_wyuczenie_latentu_jedno_zdjecie_2000'
RESUME=PREVIOUS/'MODEL_final.pt'
def js(p,v):
    tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(12):
        try:os.replace(tmp,p);return True
        except OSError as e:
            if attempt<11:time.sleep(min(.02*2**attempt,.25));continue
            if p.name=='PROGRESS.json':
                print(f'PROGRESS_WRITE_SKIPPED {type(e).__name__}: {e}',file=sys.stderr,flush=True)
                try:tmp.unlink()
                except OSError:pass
                return False
            raise
def state(net):return {n:t.detach().cpu().clone() for n,t in net.state_dict().items()}
def iou(b,g):
    d=np.maximum(0,np.minimum(b[2:],g[2:])-np.maximum(b[:2],g[:2]));inter=float(np.prod(d))
    return inter/(float(np.prod(b[2:]-b[:2]))+float(np.prod(g[2:]-g[:2]))-inter+1e-9)
def det(o,g):
    boxes,scores=dekoduj(o.detach().cpu());b=boxes[0,0].numpy()
    return dict(confidence=float(scores[0,0]),iou=iou(b,g),bbox_xyxy=b.tolist(),all_boxes_xyxy=boxes[0].tolist(),all_scores=scores[0].tolist())
def measure(z,o,target,J,K,g,targets):
    z=z[0];err=(z-target).square();m128=float(err[:,J].mean());m640=float(err[:,K].mean());energy=float(target[:,J].square().mean())
    a=z[:,J].flatten();b=target[:,J].flatten()
    gt_loss=float(strata(o,*targets)[0].detach())
    return dict(MSE128=m128,MSE640=m640,MSE768=float(err.mean()),agreement128=100*max(0,1-math.sqrt(m128/energy)),
                cosine128=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),GTloss=gt_loss,**det(o,g))
def successful(m):return m['agreement128']>=85
def score(m):return m['agreement128']
def checkpoint(net,opt,names,stage,step,metrics,path):
    weights=state(net)
    changed=[n for n in weights if _REFERENCE is not None and not torch.equal(weights[n],_REFERENCE[n])]
    torch.save(dict(net=weights,optimizer=opt.state_dict() if opt else None,trainable_parameters=names,changed_parameters=changed,
                    architecture=ARCH,torch_version=str(torch.__version__),timm_version=str(timm.__version__),
                    config=CONFIG,stage=stage,step=step,metrics=metrics,source_checkpoint_sha256=sha(SOURCE/'MODEL_VIT_K1000.pt')),path)
def audit(net,original,names):
    now=state(net);changed=[n for n in now if not torch.equal(now[n],original[n])]
    assert set(changed)<=set(names),[n for n in changed if n not in names]
    bn=[n for n in now if 'running_' in n or 'num_batches_tracked' in n]
    return dict(changed_parameters=changed,allowed_parameters=names,frozen_unchanged=True,BN_buffers_unchanged=all(torch.equal(now[n],original[n]) for n in bn),
                every_tensor=[dict(name=n,changed=n in changed,allowed=n in names,max_abs_delta=float((now[n]-original[n]).abs().max())) for n in now])
def export(net,image,target,before,J,K,g,targets,tag,original,names,rgb):
    net.eval();cache={}
    def read(m,i,o):cache['z']=o
    h=net.bloki[-1].mlp.act.register_forward_hook(read)
    with torch.no_grad():o,_=net(image)
    h.remove();z=cache['z'];m=measure(z,o,target,J,K,g,targets)
    arr=z[0].cpu().numpy();j=J.cpu().numpy()
    np.savez_compressed(OUT/f'LATENT_NATURAL_{tag}.npz',natural=arr,before=before.cpu().numpy(),target=target.cpu().numpy(),channels0=j,grid_shape=[20,30],detector_map=o.cpu().numpy())
    with (OUT/f'VEKTORY_J128_{tag}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['token','row','col','channel0','before','target','natural','error'])
        aa=before.cpu().numpy();bb=target.cpu().numpy()
        for t in range(600):
            for c in j:w.writerow([t,t//30,t%30,int(c),float(aa[t,c]),float(bb[t,c]),float(arr[t,c]),float(arr[t,c]-bb[t,c])])
    js(OUT/f'AUDIT_{tag}.json',audit(net,original,names))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(8,5));ax.imshow(rgb)
    for box,color,label in [(g,'cyan','GT'),(m['bbox_xyxy'],'lime','Natural prediction')]:
        ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=2,label=label))
    ax.set_title(f"{tag}: agreement {m['agreement128']:.3f}% | confidence {m['confidence']:.6f} | IoU {m['iou']:.6f}");ax.legend();ax.axis('off');fig.tight_layout();fig.savefig(OUT/f'OVERLAY_{tag}.png',dpi=150);plt.close(fig)
    return m

CONFIG.update(steps=50000,start_step=2000,agreement_threshold=85.0,chunk_steps=1000,plateau_window=2000,plateau_gain_pp=0.10,stagnation_window=4000,stagnation_gain_pp=0.02,minimum_plateau_lr=1e-5)
