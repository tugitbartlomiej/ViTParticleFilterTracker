"""Lokalna falsyfikacja rankingu paczek przy zamrożonym punkcie K1000.
Nie zapisuje obrazów ani latentów. Rzeczywiste kroki tylko na kopii stanu.
"""
import argparse, copy, hashlib, json, os, sys, time, zipfile
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/vit_langevin_planowanie_paczek_sd'))
import model as M
import rownanie_original as RW
OUT = ROOT / 'outputs/vit_k1000_langevin_dobor35000_pilot_20261008'
SOURCE = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90'
FULL = '--full' in sys.argv
if FULL: OUT = OUT / 'skan_puli35000'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def save(name, obj):
    p = OUT / name
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    t.replace(p)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'ACTIVE.lock'
    # Exclusive file creation; an interrupted run requires explicit diagnosis.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    start = time.time()
    try:
        torch.set_num_threads(4); cv2.setNumThreads(1)
        torch.manual_seed(20261008)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cp = ROOT / 'outputs/vit_langevin_cztery_paczki_sd/MODEL_AUTHENTIC_K1000.pt'
        assert sha(cp) == 'e7a7727ccf19dd3801bd8ec4de65887480252ccdf7e21d385346c13a139259b6'
        state = torch.load(cp, map_location='cpu', weights_only=True)
        ref = torch.load(SOURCE / 'MODEL_VIT_K1000.pt', map_location='cpu', weights_only=True)
        assert state['krok'] == 1000
        assert all(torch.equal(v, ref['net'][n]) for n,v in state['net'].items())
        del ref
        with np.load(SOURCE / 'LATENT_PELNY.npz') as f:
            target=f['after'].astype(np.float64); before=f['before'].copy(); J=f['channels0'].copy()
        assert target.shape == before.shape == (600,768)
        other = np.setdiff1d(np.arange(768), J)
        selection=json.loads((SOURCE/'SELECTION.json').read_text(encoding='utf-8'))
        xx,yy,ww,hh=selection['record']['bbox_xywh']; gt=np.array([xx,yy,xx+ww,yy+hh])
        rgb=cv2.cvtColor(cv2.imread(str(SOURCE/'ZDJECIE_ZRODLOWE.jpg')),cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA)
        probe=torch.from_numpy(rgb).permute(2,0,1)[None].float().cuda()/255
        net=M.MalyViT().cuda(); net.zapisuj_latent(True)
        opt=torch.optim.AdamW([dict(params=[p for p in net.parameters() if p.ndim>=2],weight_decay=.05),
                              dict(params=[p for p in net.parameters() if p.ndim<2],weight_decay=0)],lr=5e-4)
        u=SimpleNamespace(net=net,opt=opt,_lr=lambda:5e-4)
        def restore():
            net.load_state_dict(state['net']);opt.load_state_dict(copy.deepcopy(state['opt']))
            opt.zero_grad(set_to_none=True); net._latent=None
            for group in opt.param_groups: group['lr']=5e-4
        def read(delta=None):
            net.eval(); par=dict(net.named_parameters())
            if delta is not None:par={n:p+delta[n] for n,p in par.items()}
            with torch.no_grad():
                o,z=torch.func.functional_call(net,(par,{n:b.clone() for n,b in net.named_buffers()}),(probe,))
                z=z[0].float().cpu().numpy().astype(np.float64)
                boxes,scores=M.dekoduj(o); b=boxes[0,0].cpu().numpy()
                inter=np.prod(np.maximum(np.minimum(b[2:],gt[2:])-np.maximum(b[:2],gt[:2]),0))
                union=np.prod(np.maximum(b[2:]-b[:2],0))+ww*hh-inter
                d={'confidence':float(scores[0,0]),'iou':float(inter/max(union,1e-30))}
            net._latent=None
            assert np.isfinite(z).all()
            return z,d
        def mse(z,ix):return float(np.mean((z[:,ix]-target[:,ix])**2))
        restore(); z0,d0=read()
        np.testing.assert_allclose(z0,before,rtol=1e-4,atol=3e-5)
        baseline={k:mse(z0,ix) for k,ix in [('mse128',J),('mse640',other),('mse768',np.arange(768))]}
        records=[json.loads(s) for s in (ROOT/'outputs/vit_dino_faiss_35000/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records)==35000 and all(r['split']=='TRAIN' for r in records)
        # Balanced films, deterministic random sample; exclude the target film.
        rng=np.random.default_rng(20261008); by={}
        for r in records:
            if r['film'] != selection['record']['film']:by.setdefault(r['film'],[]).append(r)
        films=sorted(by); rng.shuffle(films)
        for a in by.values():rng.shuffle(a)
        count=sum(len(a) for a in by.values()) if FULL else 256
        candidates=[]; cycle=0
        while len(candidates)<count:
            for film in films:
                if cycle<len(by[film]):candidates.append(by[film][cycle])
                if len(candidates)==count:break
            cycle+=1
        save('PROVENANCE.json',{'checkpoint_sha256':sha(cp),'target_sha256':sha(SOURCE/'LATENT_PELNY.npz'),
            'manifest_sha256':sha(ROOT/'outputs/vit_dino_faiss_35000/manifest.jsonl'),
            'model_tensors_equal':True,'optimizer_step':1000,'precision':'FP32 gradients and forwards',
            'candidate_images':count,'candidate_batches':(count+7)//8,'batch_size':8,'excluded_probe_film':selection['record']['film'],
            'scope':'full eligible manifest scan in fixed batches; not combinatorial global optimum' if FULL else 'bounded pilot',
            'future_forces_used':False})
        rows=[]
        total=(count+7)//8
        for batch_id in range(total):
            rs=candidates[batch_id*8:(batch_id+1)*8]; images=[]; boxes=[]
            for r in rs:
                with zipfile.ZipFile(r['archive']) as z:raw=z.read(r['member'])
                assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                assert im is not None and im.shape[:2]==(r['height'],r['width'])
                images.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA))
                bb=[]
                for a in r['gt']:
                    x,y,w,h=a['bbox_xywh'];bb.append([x*480/r['width'],y*320/r['height'],(x+w)*480/r['width'],(y+h)*320/r['height']])
                boxes.append(np.asarray(bb,np.float32).reshape(-1,4))
            x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float().cuda()/255
            c=tuple(t.cuda() for t in M.cele(boxes,20,30))
            restore(); net.train()
            params=dict(net.named_parameters());buffers={n:b.clone() for n,b in net.named_buffers()}
            output,_=torch.func.functional_call(net,(params,buffers),(x,))
            loss,_=M.strata(output,*c)
            grads=torch.autograd.grad(loss,list(params.values()))
            norm=torch.sqrt(sum(v.square().sum() for v in grads))
            scale=min(1.0,1.0/(float(norm)+1e-6))
            g={n:v.detach()*scale for n,v in zip(params,grads)}
            net._latent=None;del output,loss,grads
            new,momentum=RW.czesci_kroku(u,g)
            # Sum in parameter space includes AdamW decay; BN remains frozen.
            delta={n:new[n]+momentum[n] for n in params}
            names={id(p):n for n,p in params.items()}
            for group in opt.param_groups:
                for p in group['params']:delta[names[id(p)]]=delta[names[id(p)]]-group['lr']*group['weight_decay']*p.detach()
            predicted,_=read(delta)
            # Classical decomposed equation retained for comparison (no decay/BN).
            zn,_=read(new);zm,_=read(momentum);classic=zn+zm-z0
            sealed={'batch_id':batch_id,'ids':[r['id'] for r in rs],
                    'predicted_mse128':mse(predicted,J),'classic_mse128':mse(classic,J)}
            save(f'decision_{batch_id:02d}.json',sealed)  # before actual step
            assert all(torch.equal(v.cpu(),state['net'][n]) for n,v in net.state_dict().items())
            # Actual step on a disposable restored state. No activation substitution.
            restore();net.train();o,_=net(x);L,_=M.strata(o,*c);L.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
            actual,det=read();movement=actual-z0
            def skill(p):return 100*(1-float(np.sum((actual[:,J]-p[:,J])**2))/max(float(np.sum(movement[:,J]**2)),1e-30))
            row={**sealed,'actual_mse128':mse(actual,J),'actual_mse640':mse(actual,other),
                 'actual_mse768':mse(actual,np.arange(768)),'combined_skill128_percent':skill(predicted),
                 'classic_skill128_percent':skill(classic),'actual_gain128':1-mse(actual,J)/baseline['mse128'],**det}
            rows.append(row)
            save('STATUS.json',{'state':'running','finished_batches':len(rows),'total_batches':total,'seconds':time.time()-start,
                               'positive_batches':sum(r['actual_gain128']>0 for r in rows),
                               'best_gain128':max(r['actual_gain128'] for r in rows)})
            if batch_id%10==0 or batch_id==total-1:save('MEASUREMENTS.json',rows)
            if not FULL or batch_id%25==0:print(json.dumps({'batch':batch_id,'total':total,'gain':row['actual_gain128'],'seconds':time.time()-start}),flush=True)
            del x,c,o,L,g,new,momentum,delta
        selected=min(rows,key=lambda r:r['predicted_mse128']);classic_selected=min(rows,key=lambda r:r['classic_mse128'])
        rankp=np.argsort(np.argsort([r['predicted_mse128'] for r in rows]));ranka=np.argsort(np.argsort([r['actual_mse128'] for r in rows]))
        result={'state':'complete','baseline':{**baseline,**d0},'selected':selected,'classic_selected':classic_selected,
                'random_expected_mse128':float(np.mean([r['actual_mse128'] for r in rows])),
                'actual_best':min(rows,key=lambda r:r['actual_mse128']),
                'rank_correlation':float(np.corrcoef(rankp,ranka)[0,1]),'seconds':time.time()-start,
                'limits':['one old target image','fixed batching, not all combinations','batch8 instead of historical32',
                          'ranking measured atK1000 only','combined forecast omits BN evolution; actual step includes it'],
                'median_combined_skill128_percent':float(np.median([r['combined_skill128_percent'] for r in rows])),
                'median_classic_skill128_percent':float(np.median([r['classic_skill128_percent'] for r in rows]))}
        result['positive_actual_batches']=sum(r['actual_gain128']>0 for r in rows)
        result['candidate_images']=count; result['candidate_batches']=total
        if FULL:
            ranked=sorted(rows,key=lambda r:r['predicted_mse128'])
            ids=[i for r in ranked for i in r['ids']][:1500]
            assert len(ids)==1500 and len(set(ids))==1500
            save('KANDYDACI_1500.json',{'ids':ids,'rule':'prefix ranked fixed batches atK1000',
                  'not_verified_sequence':True,'does_not_guarantee_toward_target':True,
                  'positive_batches_in_scan':result['positive_actual_batches']})
        save('RESULT.json',result);save('STATUS.json',{'state':'complete','seconds':time.time()-start})
        restore()
        assert all(torch.equal(v.cpu(),state['net'][n]) for n,v in net.state_dict().items())
    except Exception:
        import traceback
        save('ERROR.json',{'error':traceback.format_exc(),'seconds':time.time()-start});raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':main()
