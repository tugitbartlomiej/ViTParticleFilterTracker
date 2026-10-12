"""Bounded per-image J128 activation intervention, frozen latest single-image ViT."""
import sys, json, time, hashlib, csv
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/vit_latent128_jedno_zdjecie_90'))
import torch, numpy as np, cv2
from torch.nn import functional as F
from model_snapshot import MalyViT,dekoduj,_giou
from metryki_source import iou
from PIL import Image,ImageDraw
OUT=ROOT/'outputs/vit_latent128_nowy_checkpoint_granice_20261006'
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
CK=ROOT/'outputs/vit_wyuczenie_latentu_jedno_zdjecie_do85/MODEL_best.pt'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(n,x):
    (OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def detect(o,g):
    b,s=dekoduj(o.detach().cpu()); b=b[0].numpy(); s=s[0].numpy()
    return dict(confidence=float(s[0]),iou=float(iou(b,g)[0]),bbox_xyxy=b[0].tolist())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Existing completed result will not be overwritten')
    torch.set_num_threads(4);torch.manual_seed(20261006)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=torch.load(CK,map_location='cpu',weights_only=True)
    net=MalyViT().cuda().eval();net.load_state_dict(ck['net'],strict=True);net.requires_grad_(False)
    saved={n:t.clone() for n,t in net.state_dict().items()}
    step=ck['step'];del ck
    row=json.loads((SOURCE/'RESULT.json').read_text())['record']
    assert sha(SOURCE/'ZDJECIE_ZRODLOWE.jpg')==row['sha256']
    x,y,w,h=row['bbox_xywh'];g=np.array([x,y,x+w,y+h],np.float32)
    rgb=cv2.resize(cv2.cvtColor(cv2.imread(str(SOURCE/'ZDJECIE_ZRODLOWE.jpg')),cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    image=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255
    block=net.bloki[-1];cache={}
    hz=block.mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o.detach().clone()))
    hq=block.register_forward_hook(lambda m,a,o:cache.update(q=o.detach().clone()))
    with torch.no_grad():base,_=net(image)
    hz.remove();hq.remove();z,q=cache['z'],cache['q']
    J=torch.from_numpy(np.load(SOURCE/'LATENT_PELNY.npz')['channels0']).long().cuda()
    W=block.mlp.fc1.weight.detach().double(); c=block.mlp.fc1.bias.detach().double()
    gamma=block.norm2.weight.detach().double();beta=block.norm2.bias.detach().double()
    wg=W*gamma; projected=wg-wg.mean(1,keepdim=True)
    radius=gamma.numel()**.5*projected.norm(dim=1);center=W@beta+c
    a_min,a_max=center-radius,center+radius
    gelu=lambda t:.5*t*(1+torch.erf(t/(2**.5)))
    critical=torch.tensor(-.7517915247,device='cuda',dtype=torch.float64)
    lower=torch.where((a_min<=critical)&(a_max>=critical),gelu(critical),torch.minimum(gelu(a_min),gelu(a_max)))
    upper=torch.maximum(gelu(a_min),gelu(a_max))
    lo,hi=lower.float()[J][None,None,:],upper.float()[J][None,None,:]
    assert bool((z[:,:,J]>=lo-2e-5).all() and (z[:,:,J]<=hi+2e-5).all())
    def fast(changed):
        delta=block.ls2(F.linear(changed-z[:,:,J],block.mlp.fc2.weight[:,J],None))
        return net.glowa(net.norm(q+delta)).reshape(1,20,30,5)
    torch.testing.assert_close(fast(z[:,:,J]),base,rtol=1e-5,atol=2e-5)
    cx=int(np.clip(round((g[0]+g[2])/32-.5),0,29));cy=int(np.clip(round((g[1]+g[3])/32-.5),0,19));idx=cy*30+cx
    distances=torch.tensor([[16*(cx+.5)-g[0],16*(cy+.5)-g[1],g[2]-16*(cx+.5),g[3]-16*(cy+.5)]],device='cuda')/16
    target_log=distances.log();neg=torch.ones(1,600,device='cuda');neg[0,idx]=0
    def loss(o):
        flat=o.reshape(1,600,5);logits=flat[:,:,0]
        return F.softplus(7-logits[0,idx])+(F.softplus(logits+2)*neg).sum()/599+.5*F.cross_entropy(logits,torch.tensor([idx],device='cuda'))+30*F.smooth_l1_loss(flat[:,idx,1:],target_log)+20*(1-_giou(flat[:,idx,1:].clamp(-10,10).exp(),distances).mean())
    baseline=detect(base,g);best=z[:,:,J].clone();best_metric=baseline
    def quality(d):return (d['iou']>=.99, d['confidence'] if d['iou']>=.99 else min(d['confidence'],d['iou']),d['iou'])
    history=[];updates=0
    for lr in (.01,.03,.1):
        changed=best.clone().requires_grad_();opt=torch.optim.Adam([changed],lr=lr)
        for s in range(1,1501):
            objective=loss(fast(changed))+1e-4*(changed-z[:,:,J]).square().mean()
            assert torch.isfinite(objective)
            opt.zero_grad();objective.backward();opt.step()
            with torch.no_grad():changed.copy_(torch.maximum(lo,torch.minimum(hi,changed)))
            updates+=1
            if s%10==0:
                d=detect(fast(changed),g)
                if quality(d)>quality(best_metric):best=changed.detach().clone();best_metric=d
                if s%100==0:
                    entry=dict(updates=updates,lr=lr,loss=float(objective.detach()),**d);history.append(entry)
                    print(json.dumps(entry),flush=True);write('PROGRESS.json',entry)
                if best_metric['confidence']>=.99 and best_metric['iou']>=.995:break
        if best_metric['confidence']>=.99 and best_metric['iou']>=.995:break
    after=z.clone();after[:,:,J]=best
    hook=block.mlp.act.register_forward_hook(lambda m,a,o:after)
    with torch.no_grad():actual,_=net(image)
    hook.remove();torch.testing.assert_close(actual,fast(best),rtol=2e-5,atol=3e-5)
    K=torch.ones(768,dtype=torch.bool,device='cuda');K[J]=False
    assert torch.equal(after[:,:,K],z[:,:,K])
    assert all(torch.equal(t,saved[n]) for n,t in net.state_dict().items())
    a=z[0].cpu().numpy();b=after[0].cpu().numpy();jj=J.cpu().numpy()
    np.savez_compressed(OUT/'LATENT_PELNY.npz',before=a,after=b,delta=b-a,channels0=jj,lower=lower.cpu().numpy(),upper=upper.cpu().numpy())
    assert np.array_equal(np.load(OUT/'LATENT_PELNY.npz')['after'],b)
    np.savez_compressed(OUT/'LATENT128.npz',before=a[:,jj],after=b[:,jj],delta=(b-a)[:,jj],channels0=jj)
    with (OUT/'KANALY128.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['channel0','lower','upper','before_mean','after_mean','delta_rms'])
        for j in jj:writer.writerow([j,float(lower[j]),float(upper[j]),float(a[:,j].mean()),float(b[:,j].mean()),float(np.sqrt(((b[:,j]-a[:,j])**2).mean()))])
    panel=Image.new('RGB',(960,360),'white');draw=ImageDraw.Draw(panel)
    for offset,d,title in [(0,baseline,'PRZED'),(480,best_metric,'PO: ograniczony latent J128')]:
        panel.paste(Image.fromarray(rgb),(offset,40));draw.text((offset+8,5),f'{title}: confidence {d["confidence"]:.4f}, IoU {d["iou"]:.4f}',fill='black')
        draw.rectangle((offset+g[0],40+g[1],offset+g[2],40+g[3]),outline='cyan',width=2)
        bb=d['bbox_xyxy'];draw.rectangle((offset+bb[0],40+bb[1],offset+bb[2],40+bb[3]),outline='lime',width=2)
    panel.save(OUT/'DETEKCJA_PRZED_PO.png')
    result=dict(status='complete',checkpoint=str(CK),checkpoint_step=step,checkpoint_sha256=sha(CK),source=row,baseline=baseline,intervention=detect(actual,g),updates=updates,channels0=jj.tolist(),unchanged640=True,weights_unchanged=True,per_channel_bounds_satisfied=True,full_forward_tail_max_error=float((actual-fast(best)).abs().max().detach()),J128_upper_max=float(upper[J].max()),before_J128_max=float(z[:,:,J].max()),after_J128_max=float(best.max()),history=history,limitations='Single-image label-assisted activation intervention. Per-channel bounds are necessary, not sufficient for simultaneous image realizability. This checkpoint was fitted on this same image; not held-out evaluation.')
    write('RESULT.json',result)
    print(json.dumps({'complete':True,'baseline':baseline,'intervention':result['intervention']}),flush=True)

if __name__=='__main__':main()
