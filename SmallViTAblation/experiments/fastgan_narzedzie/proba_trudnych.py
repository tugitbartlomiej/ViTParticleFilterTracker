"""Small frozen-model diagnostic. Real annotated foreground, GAN-driven surroundings.

Not a detector training run. Not full Langevin dynamics. All computations float32;
one fixed photometric operator, split by operation. No generator masks as labels.
"""
from __future__ import annotations
import os
for key, value in {'USE_TF':'0', 'USE_FLAX':'0', 'NO_ALBUMENTATIONS_UPDATE':'1', 'HF_HUB_OFFLINE':'1'}.items():
    os.environ.setdefault(key, value)
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('original', 'random', 'random_search', 'detector', 'detector_koopman')


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_crop(row, device):
    a = np.array(Image.open(HERE/'dane/pilot_1000'/row['file']).convert('RGBA'))
    x = torch.from_numpy(a[:,:,:3].copy()).permute(2,0,1)[None].float().to(device)/255
    mask = torch.from_numpy((a[:,:,3]>127).copy())[None,None].float().to(device)
    if mask.sum() < 20: raise ValueError('Too few annotated pixels')
    # One on the entire tool plus 3 pixels; gradual transition OUTSIDE this support.
    core = F.max_pool2d(mask, 7, 1, 3)
    alpha = torch.maximum(core, F.avg_pool2d(core, 15, 1, 7))
    yy, xx = torch.where(mask[0,0] > .5)
    box = torch.stack([xx.min(), yy.min(), xx.max()+1, yy.max()+1]).float()/256
    return x, mask, alpha, box


def composite(real, alpha, background):
    return real + .35*(1-alpha)*(background-real)


def transform(x, alpha):
    return x + (1-alpha)*(torch.clamp(x*1.15,0,1)-x)


def box_iou(boxes, gt):
    wh = (torch.minimum(boxes[:,2:],gt[2:])-torch.maximum(boxes[:,:2],gt[:2])).clamp_min(0)
    inter = wh.prod(-1)
    a = (boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(-1)
    b = (gt[2:]-gt[:2]).prod()
    return inter/(a+b-inter).clamp_min(1e-8)


class Detector:
    def __init__(self, checkpoint, base, device):
        from rfdetr import RFDETRNano
        self.wrapper = RFDETRNano(pretrain_weights=str(checkpoint), resolution=384,
                                  num_classes=1, trust_checkpoint=True)
        self.net = self.wrapper.model.model.to(device).eval().requires_grad_(False)
        bb = next(m for m in self.net.modules() if type(m).__name__=='WindowedDinov2WithRegistersBackbone')
        self.gamma = bb.layernorm.weight.detach().abs()
        self.beta = bb.layernorm.bias.detach()
        b = np.load(base)
        assert np.allclose(self.gamma.cpu(), b['rusztowanie_dlugosci'], atol=1e-6)
        assert np.allclose(self.beta.cpu(), b['rusztowanie_jadro'], atol=1e-6)
        self.centre = torch.tensor(b['srodek'],dtype=torch.float32,device=device)
        self.basis = torch.tensor(b['kier'],dtype=torch.float32,device=device)
        w = np.load(ROOT/'experiments/krystalizacja_latentu/wyniki/dowody/RDZEN_bez_etykiet_32.npy')
        self.identity = torch.tensor(w,dtype=torch.float32,device=device)
        self.mean = torch.tensor(self.wrapper.means,device=device)[None,:,None,None]
        self.std = torch.tensor(self.wrapper.stds,device=device)[None,:,None,None]
        self.feature = None
        def capture(m, i, o):
            self.feature = (o.feature_maps if hasattr(o,'feature_maps') else o[0])[-1]
        self.hook = bb.register_forward_hook(capture)

    def __call__(self, image, mask, gt):
        from rfdetr.utilities.tensors import NestedTensor
        image = F.interpolate(image,(384,384),mode='bilinear',align_corners=False)
        # Bypass RF-DETR's reusable inference padding buffer: retaining two input
        # graphs through that mutable buffer is invalid for gradient-based probes.
        normalized = (image-self.mean)/self.std
        padding = torch.zeros((image.shape[0],384,384),dtype=torch.bool,device=image.device)
        out = self.net(NestedTensor(normalized,padding))
        feat = self.feature
        self.feature = None
        prof = (feat.permute(0,2,3,1).reshape(-1,feat.shape[1])-self.beta)/self.gamma
        p = (prof-self.centre) @ self.basis
        weight = F.interpolate(mask,feat.shape[-2:],mode='area').reshape(-1)
        logits = out['pred_logits'][0]
        # RF-DETR remaps COCO category_id=1 to contiguous foreground label 0.
        # The evaluator likewise writes category_id=class_id+1; index 1 is NOT the tool.
        assert logits.shape[-1] == 2
        confidence = logits[:,0].sigmoid()
        c = out['pred_boxes'][0]
        boxes = torch.cat([c[:,:2]-c[:,2:]/2,c[:,:2]+c[:,2:]/2],-1)
        overlap = box_iou(boxes,gt)
        quality = (confidence*overlap).max()
        identity = p @ self.identity
        near = weight/weight.sum().clamp_min(1e-8)
        far = (weight==0).float(); far = far/far.sum().clamp_min(1)
        return dict(p=p, weight=weight, quality=quality,
                    hit=((confidence>=.5)&(overlap>=.5)).any(),
                    contrast=((near-far)*identity).sum(), identity_map=identity,
                    boxes=boxes, confidence=confidence, overlap=overlap)


class Generator:
    def __init__(self, checkpoint, device):
        spec = importlib.util.spec_from_file_location('fixed_fastgan',HERE/'vendor/deepfake_gi_fastGAN/models.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.net = mod.Generator(ngf=64,nz=256,nc=4,im_size=256).to(device).eval().requires_grad_(False)
        ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.net.load_state_dict(ck['ema']); del ck
        self.device = device

    def __call__(self,z,seed):
        # All layer-injected noises fixed, including in backward's saved graph.
        with torch.random.fork_rng(devices=[0] if self.device=='cuda' else []):
            torch.manual_seed(seed)
            rgba = (self.net(z)[0].clamp(-1,1)+1)/2
        # Suppress sharp synthetic instruments; NOT a certificate of valid background.
        bg = F.interpolate(F.adaptive_avg_pool2d(rgba[:,:3],(16,16)),(256,256),mode='bilinear',align_corners=False)
        return bg, rgba


def fit_affine(x,y):
    mu=x.mean(0); scale=x.std(0).clip(.01)
    a=np.c_[(x-mu)/scale,np.ones(len(x))]; b=np.c_[(y-mu)/scale,np.ones(len(x))]
    penalty=np.eye(a.shape[1])*0.001; penalty[-1,-1]=0
    k=np.linalg.solve(a.T@a/len(a)+penalty,a.T@b/len(a))
    return mu,scale,k


class Koopman:
    def __init__(self,x,y,hx,hy,device):
        mu,scale,k=fit_affine(x,y)
        a=np.c_[(x-mu)/scale,np.ones(len(x))]; b=np.c_[(y-mu)/scale,np.ones(len(x))]
        ha=np.c_[(hx-mu)/scale,np.ones(len(hx))]; hb=np.c_[(hy-mu)/scale,np.ones(len(hx))]
        lam,v=np.linalg.eig(k)
        chosen=[]
        for j in np.argsort(np.abs(lam-1)):
            if np.abs(lam[j]-1)>.20 or np.std(a@v[:,j])<1e-5 or lam[j].imag < -1e-7: continue
            chosen.append(j)
            if len(chosen)==6: break
        if not chosen: raise ValueError('No nonconstant near-one modes; do not invent invariant directions')
        v=v[:,chosen]; lam=lam[chosen]
        vr,vi=v.real,v.imag
        # Real-valued representation avoids unsupported complex gradients in some CUDA paths.
        p0=a@v; p1=b@v; res=p1-p0*lam
        variance=(np.abs(res)**2).mean(0)
        floor=max(float(np.median(variance))*.1,1e-8)
        variance=variance.clip(floor)
        def mse(pred): return float(np.mean((hb[:,:-1]-pred[:,:-1])**2))
        rng=np.random.default_rng(73)
        _,_,ks=fit_affine(x,y[rng.permutation(len(y))])
        held_mse,identity_mse,shuffle_mse=mse(ha@k),mse(ha),mse(ha@ks)
        holdres=hb@v-(ha@v)*lam
        idw=np.load(ROOT/'experiments/krystalizacja_latentu/wyniki/dowody/RDZEN_bez_etykiet_32.npy')
        xread=np.c_[(a@v).real,(a@v).imag,np.ones(len(a))]
        hread=np.c_[(ha@v).real,(ha@v).imag,np.ones(len(ha))]
        coef=np.linalg.lstsq(xread,x@idw,rcond=None)[0]; target=hx@idw
        identity_r2=float(1-np.sum((target-hread@coef)**2)/max(np.sum((target-target.mean())**2),1e-10))
        self.report=dict(selected_eigenvalues=[[float(l.real),float(l.imag)] for l in lam],
            heldout_operator_mse=held_mse,heldout_identity_mse=identity_mse,heldout_shuffled_mse=shuffle_mse,
            beats_identity=held_mse<identity_mse,identity_readout_r2_heldout=identity_r2,
            heldout_scaled_residual=float(np.mean(np.abs(holdres)**2/variance)),
            note='Affine EDMD of background illumination, no video time or Langevin memory; identity is an operational readout only')
        self.arrays=dict(mu=mu,scale=scale,k=k,vr=vr,vi=vi,lr=lam.real,li=lam.imag,variance=variance)
        for key,val in self.arrays.items(): setattr(self,key,torch.tensor(val,dtype=torch.float32,device=device))

    def residual(self,p,q,weights):
        a=torch.cat([(p-self.mu)/self.scale,torch.ones_like(p[:,:1])],-1)
        b=torch.cat([(q-self.mu)/self.scale,torch.ones_like(q[:,:1])],-1)
        ar,ai=a@self.vr,a@self.vi
        rr=b@self.vr-(ar*self.lr-ai*self.li)
        ri=b@self.vi-(ar*self.li+ai*self.lr)
        per_token=((rr.square()+ri.square())/self.variance).mean(-1)
        return (per_token*weights).sum()/weights.sum().clamp_min(1e-8), per_token


def measure(det,kop,x,mask,alpha,box):
    a=det(x,mask,box); b=det(transform(x,alpha),mask,box)
    residual,pm=kop.residual(a['p'],b['p'],a['weight'])
    return a,residual,pm


def scalar(a,r):
    j=int(torch.argmax(a['confidence']*a['overlap']).detach())
    return dict(quality=float(a['quality'].detach()),hit=bool(a['hit']),contrast=float(a['contrast'].detach()),
                residual=float(r.detach()),confidence=float(a['confidence'][j].detach()),
                iou=float(a['overlap'][j].detach()),box=a['boxes'][j].detach().cpu().tolist())


def save_image(path,x):
    a=(x[0].detach().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8')
    Image.fromarray(a).save(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fit',type=int,default=32); ap.add_argument('--holdout',type=int,default=16)
    ap.add_argument('--anchors',type=int,default=12); ap.add_argument('--steps',type=int,default=20)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    start=time.time(); device='cuda'; torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.48)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(8201); np.random.seed(8201)
    records=json.loads((HERE/'dane/pilot_1000/MANIFEST.json').read_text())['records']
    if any(r['split']!='train' for r in records): raise ValueError('Nontraining data')
    by={}
    for row in records: by.setdefault(row['group_id'],[]).append(row)
    rng=np.random.default_rng(8201); groups=rng.permutation(sorted(by)).tolist()
    count=args.fit+args.holdout+args.anchors
    if count>len(groups): raise ValueError('Need disjoint operations')
    rows=[sorted(by[g],key=lambda r:r['frame_id'])[0] for g in groups[:count]]
    checkpoint=ROOT/'results/eden_static36_2026-09-15/tracking_uniform_time_b500_s42/best.pth'
    base=ROOT/'results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500/baza.npz'
    genck=HERE/'wyniki/eden_20260923_50k/results_50k/checkpoint_050000.pth'
    assert file_sha(checkpoint)=='15e33f6d513ec2d34f7a37dd9db167478bc3a7191d1b5bd21da0b8ae0ae08995'
    write(out/'CONFIG.json',dict(**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          seed=8201,torch=str(torch.__version__),detector_sha256=file_sha(checkpoint),generator_sha256=file_sha(genck),
          code_sha256=file_sha(__file__),base_sha256=file_sha(base),partitions=dict(fit=groups[:args.fit],
          holdout=groups[args.fit:args.fit+args.holdout],anchors=groups[args.fit+args.holdout:count]),
          background_mix=.35,latent_radius_rms=.35,koopman_weight=.2,learning_rate=.08))
    det=Detector(checkpoint,base,device); gen=Generator(genck,device)
    # Independent public-API check of class-index mapping and differentiable preprocessing.
    check_image,check_mask,_,check_box=load_crop(rows[0],device)
    with torch.no_grad(): direct=det(check_image,check_mask,check_box)
    public=det.wrapper.predict(check_image[0],threshold=0.0)
    public_scores=public.confidence[public.class_id==0]
    if not len(public_scores): raise RuntimeError('Public detector has no class-0 outputs')
    api_delta=abs(float(public_scores.max())-float(direct['confidence'].max()))
    if api_delta>1e-4: raise RuntimeError(f'Public prediction disagreement: {api_delta}')
    det.feature=None
    hashes=(state_sha(det.net),state_sha(gen.net))
    print('MODELS READY',flush=True)
    xa,ya,hxa,hya=[],[],[],[]
    for i,row in enumerate(rows[:args.fit+args.holdout]):
        x,m,alpha,box=load_crop(row,device)
        with torch.no_grad():
            a=det(x,m,box); b=det(transform(x,alpha),m,box); ix=a['weight']>.025
            v=a['p'][ix].cpu().numpy(); w=b['p'][ix].cpu().numpy()
        (xa if i<args.fit else hxa).append(v); (ya if i<args.fit else hya).append(w)
        if (i+1)%8==0: print('CALIBRATION',i+1,flush=True)
    kop=Koopman(*[np.concatenate(z).astype(float) for z in (xa,ya,hxa,hya)],device)
    write(out/'OPERATOR.json',kop.report); np.savez(out/'OPERATOR.npz',**kop.arrays)
    print('OPERATOR',json.dumps(kop.report),flush=True)
    all_results=[]; checks=[]
    for i,row in enumerate(rows[args.fit+args.holdout:]):
        directory=out/f'case_{i:02d}'; directory.mkdir()
        x,m,alpha,box=load_crop(row,device); fixed_seed=9300+i
        z0=torch.tensor(rng.normal(size=(1,256)),dtype=torch.float32,device=device)
        baseline=composite(x,alpha,x)
        assert torch.equal(x,baseline)
        with torch.no_grad():
            aa=det(x,m,box); bb=det(x,m,box)
            repeat=float((aa['p']-bb['p']).abs().max())
            if repeat>1e-5: raise RuntimeError(f'Nondeterministic features: {repeat}')
        result=dict(frame_id=row['frame_id'],group_id=row['group_id'],arms={})
        latent_by={}; generated_by={}; trajectories={}
        # Both gradient arms start from exactly z0. Radius applies equally to random search.
        latent_by['original']=z0; latent_by['random']=z0
        best=float('inf'); bestz=z0
        with torch.no_grad():
            for step in range(args.steps):
                z=z0 if step==0 else z0+torch.tensor(rng.normal(size=(1,256)),dtype=torch.float32,device=device)*.35
                delta=z-z0; z=z0+delta*(.35/delta.square().mean().sqrt().clamp_min(.35))
                bg,_=gen(z,fixed_seed); candidate=composite(x,alpha,bg)
                val=det(candidate,m,box)['quality'].item()
                if val<best: best,bestz=val,z.detach().clone()
        latent_by['random_search']=bestz
        for arm in ('detector','detector_koopman'):
            z=z0.clone().requires_grad_(True); optim=torch.optim.Adam([z],lr=.08); history=[]
            for step in range(args.steps):
                optim.zero_grad(set_to_none=True)
                bg,_=gen(z,fixed_seed); candidate=composite(x,alpha,bg)
                a,r,_=measure(det,kop,candidate,m,alpha,box)
                objective=1-a['quality'] + (.2*torch.log1p(r) if arm=='detector_koopman' else 0)
                (-objective).backward()
                if z.grad is None or not torch.isfinite(z.grad).all(): raise RuntimeError('Invalid input gradient')
                history.append(dict(step=step,objective=float(objective.detach()),grad_norm=float(z.grad.norm()),**scalar(a,r)))
                optim.step()
                with torch.no_grad():
                    delta=z-z0; z.copy_(z0+delta*(.35/delta.square().mean().sqrt().clamp_min(.35)))
            latent_by[arm]=z.detach(); trajectories[arm]=history
        for arm in ARMS:
            with torch.no_grad():
                bg,rgba=gen(latent_by[arm],fixed_seed)
                candidate=x if arm=='original' else composite(x,alpha,bg)
                a,r,pm=measure(det,kop,candidate,m,alpha,box)
                item=scalar(a,r)
                foreground_error=float(((candidate-x)*m).abs().max())
                assert foreground_error==0
                item.update(foreground_max_error=foreground_error,background_rms=float(((candidate-x)*(1-m)).square().mean().sqrt()),
                            latent_rms=float((latent_by[arm]-z0).square().mean().sqrt()))
                result['arms'][arm]=item
                save_image(directory/f'{arm}.png',candidate)
                np.savez_compressed(directory/f'{arm}_particles.npz',profile=a['p'].cpu().numpy(),
                                    identity=a['identity_map'].cpu().numpy(),residual=pm.cpu().numpy(),mask_weight=a['weight'].cpu().numpy())
                if arm!='original': save_image(directory/f'{arm}_generator.png',rgba[:,:3])
                generated_by[arm]=candidate.detach().cpu()
        save_image(directory/'MASKA.png',m.repeat(1,3,1,1))
        write(directory/'RESULT.json',result); write(directory/'TRAJECTORIES.json',trajectories)
        np.savez(directory/'INPUTS.npz',**{k:v.detach().cpu().numpy() for k,v in latent_by.items()},noise_seed=fixed_seed)
        all_results.append(result); checks.append(dict(frame_id=row['frame_id'],repeat_max_error=repeat,self_paste_exact=True))
        write(out/'RESULTS.json',all_results)
        print('CASE',i+1,json.dumps({k:v['hit'] for k,v in result['arms'].items()}),flush=True)
    assert hashes==(state_sha(det.net),state_sha(gen.net)), 'Frozen model state changed'
    good=[r for r in all_results if r['arms']['original']['hit']]
    summary={arm:dict(hits=sum(r['arms'][arm]['hit'] for r in all_results),
                     new_misses=sum(not r['arms'][arm]['hit'] for r in good),
                     mean_quality=float(np.mean([r['arms'][arm]['quality'] for r in all_results])),
                     mean_residual=float(np.mean([r['arms'][arm]['residual'] for r in all_results]))) for arm in ARMS}
    report=dict(status='completed',seconds=time.time()-start,cases=len(all_results),original_hits=len(good),
                summary=summary,checks=checks,model_states_unchanged=True,public_api_max_score_delta=api_delta,operator=kop.report,
                scope='Diagnostic crop pilot; no detector training, no full Langevin memory; backgrounds require visual review')
    write(out/'REPORT.json',report)
    # All examples, annotated equally; no manual cherry-picking of successes.
    gallery=Image.new('RGB',(256*len(ARMS),288*len(all_results)), 'white'); d=ImageDraw.Draw(gallery)
    for i,r in enumerate(all_results):
        for j,arm in enumerate(ARMS):
            im=Image.open(out/f'case_{i:02d}'/f'{arm}.png').convert('RGB')
            info=r['arms'][arm]; draw=ImageDraw.Draw(im)
            if info['confidence']>=.5: draw.rectangle([v*256 for v in info['box']],outline='yellow',width=2)
            gallery.paste(im,(j*256,i*288+32))
            d.text((j*256+4,i*288+4),f'{i:02d} {arm} hit={int(info["hit"])} q={info["quality"]:.2f}',fill='black')
    gallery.save(out/'GALERIA.jpg',quality=93)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
