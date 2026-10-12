"""Jedno źródło danych i metryk; nie miesza splitów ani współrzędnych obrazów."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
from model_snapshot import MalyViT,cele,dekoduj,_giou
from metryki_source import iou,ap50

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
INPUT=ROOT/'outputs/vit_optymalizacja_latentu128/input'
OUT=ROOT/'outputs/vit_latent128_wspolna_regulacja'
CKPT=Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k1000.pt')

def js(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def conv(v):
        if isinstance(v,(np.ndarray,torch.Tensor)):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v))
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=conv)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
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

def image_tensor(row,device='cuda'):
    im=cv2.imread(str(INPUT/row['image']));assert im is not None
    im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).permute(2,0,1).float().to(device)/255

def ap_at(pred,gt,threshold):
    scores=[]
    for i,(boxes,conf) in enumerate(pred):
        for b,s in zip(boxes,conf):scores.append((float(s),i,b))
    scores.sort(key=lambda x:-x[0]);used=[np.zeros(len(g),bool) for g in gt];tp=[]
    for _,i,b in scores:
        if not len(gt[i]):tp.append(False);continue
        v=np.asarray([iou(b,g) for g in gt[i]]);v[used[i]]=-1;j=int(v.argmax());hit=v[j]>=threshold
        tp.append(hit)
        if hit:used[i][j]=True
    tp=np.cumsum(tp);rec=tp/max(sum(map(len,gt)),1);prec=tp/np.arange(1,len(tp)+1)
    prec=np.maximum.accumulate(prec[::-1])[::-1];idx=np.searchsorted(rec,np.linspace(0,1,101))
    return float(np.mean([prec[i] if i<len(prec) else 0 for i in idx]))

def film_weights(rows,ids):
    f=[rows[i]['film'] for i in ids]
    return np.array([1/len(set(f))/f.count(x) for x in f],np.float32)

def metrics(o,T,gt,rows,ids):
    ids=list(ids);oo=o.detach().cpu();tt=tuple(t[ids].cpu() for t in T)
    ll=losses(oo,tt).numpy();boxes,scores=dekoduj(oo);boxes=boxes.numpy();scores=scores.numpy()
    pred=list(zip(boxes,scores));g=[gt[i] for i in ids]
    aps=[ap_at(pred,g,float(v)) for v in np.arange(.5,.951,.05)]
    assert abs(aps[0]-ap50(pred,g))<1e-7
    return dict(AP50=aps[0],AP75=aps[5],AP50_95=float(np.mean(aps)),loss=float(film_weights(rows,ids)@ll),
                image_mean_loss=float(ll.mean()),n=len(ids))

def load():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,digest in json.loads((INPUT/'HASHES.json').read_text()).items():assert sha(INPUT/name)==digest,name
    rows=json.loads((INPUT/'manifest.json').read_text())
    sets={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','val','test')}
    films={s:{rows[i]['film'] for i in ids} for s,ids in sets.items()}
    assert len(rows)==400 and len({r['frame_id'] for r in rows})==400
    assert not (films['train']&films['val'] or films['train']&films['test'] or films['val']&films['test'])
    J=torch.tensor(np.load(INPUT/'channels.npz')['channels0'],dtype=torch.long,device='cuda')
    assert len(J)==len(J.unique())==128
    net=MalyViT().cuda().eval();ck=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert ck['krok']==1000;net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False)
    provenance=dict(checkpoint_step=1000,checkpoint_sha=sha(CKPT),channels_sha=sha(INPUT/'channels.npz'),
                    manifest_sha=sha(INPUT/'manifest.json'),splits={s:dict(images=len(ids),films=len(films[s])) for s,ids in sets.items()},
                    torch=torch.__version__,device=torch.cuda.get_device_name(),weights_hash=state_hash(net))
    gt=[]
    for r in rows:
        x,y,w,h=r['bbox_xywh'];gt.append(np.array([[x,y,x+w,y+h]],np.float32))
    T=tuple(t.cuda() for t in cele(gt,20,30))
    cache={};b=net.bloki[-1]
    hooks=[b.mlp.act.register_forward_pre_hook(lambda m,a:cache.update(pre=a[0].detach())),
           b.register_forward_hook(lambda m,a,o:cache.update(state=o.detach()))]
    H=[];Q=[];O=[]
    with torch.no_grad():
        for lo in range(0,len(rows),8):
            x=torch.stack([image_tensor(r) for r in rows[lo:lo+8]])
            o,_=net(x);H.append(cache['pre'][:,:,J].clone());Q.append(cache['state'].clone());O.append(o.cpu())
    for h in hooks:h.remove()
    H=torch.cat(H);Q=torch.cat(Q);base=torch.cat(O)
    mu=H[sets['train']].mean((0,1));sd=H[sets['train']].std((0,1)).clamp(min=.05)
    return net,rows,sets,J,T,gt,H,Q,base,mu,sd,provenance
