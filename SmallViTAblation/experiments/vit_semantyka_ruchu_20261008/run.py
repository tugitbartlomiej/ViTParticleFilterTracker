"""Controlled factor probes and temporal-motion rollback in last MLP branch."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, time
import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
FACTORS = ['pozycja', 'kolor', 'proporcje_ksztaltu']

def load_model_module():
    path = ROOT / 'experiments/vit_sd_pelny_latent_filmy_eden_20261006/model_snapshot.py'
    spec = importlib.util.spec_from_file_location('semantic_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def mask_box(record):
    mask = np.zeros((320,480), np.uint8)
    scale = np.array(record['input_size_wh']) / record['original_size_wh']
    for poly in record['cystotome_seg']:
        points = np.array(poly, np.float32).reshape(-1,2) * scale
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    ys,xs = np.where(mask)
    assert len(xs), 'Empty annotated tool mask'
    return mask, np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],np.float32)

def variants(image, mask, box):
    result=[]
    for dx,dy in [(16,0),(-16,0),(0,16),(0,-16)]:
        result.append(cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(480,320),borderMode=cv2.BORDER_REFLECT_101))
    lab=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    for channel,sign in [(1,1),(1,-1),(2,1),(2,-1)]:
        changed=lab.copy(); changed[...,channel]+=mask.astype(np.float32)*sign*12
        rgb=cv2.cvtColor(changed,cv2.COLOR_LAB2RGB)
        changed_rgb=np.rint(rgb.clip(0,1)*255).astype(np.uint8)
        changed_rgb[mask==0]=image[mask==0]
        result.append(changed_rgb)
    cx,cy=(box[:2]+box[2:])/2
    for sx in [1.15,1/1.15]:
        sy=1/sx
        matrix=np.float32([[sx,0,cx*(1-sx)],[0,sy,cy*(1-sy)]])
        result.append(cv2.warpAffine(image,matrix,(480,320),borderMode=cv2.BORDER_REFLECT_101))
    return result

def iou(boxes, truth):
    lo=torch.maximum(boxes[:,:2],truth[:,:2]); hi=torch.minimum(boxes[:,2:],truth[:,2:])
    intersection=(hi-lo).clamp(min=0).prod(1)
    area=(boxes[:,2:]-boxes[:,:2]).clamp(min=0).prod(1)
    gt=(truth[:,2:]-truth[:,:2]).prod(1)
    return intersection/(area+gt-intersection).clamp(min=1e-8)

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='outputs/vit_semantyka_ruchu_20261008/pilot_v1');parser.add_argument('--max-step',type=int,default=1000)
    args=parser.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    start=time.time();torch.manual_seed(47);np.random.seed(47);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda';module=load_model_module();model=module.MalyViT().to(device).eval()
    images=np.load(SOURCE/'dane/obrazy.npy');records=json.loads((SOURCE/'dane/images.json').read_text(encoding='utf-8'))
    masks,boxes=zip(*(mask_box(r) for r in records));truth=torch.tensor(np.stack(boxes),device=device)
    selected=np.stack([cv2.resize(cv2.dilate(m,np.ones((33,33),np.uint8)),(30,20),interpolation=cv2.INTER_AREA).reshape(-1)>0 for m in masks])
    # Boolean region selections are fixed across checkpoints, with unchanged input/token identity.
    selected_t=torch.tensor(selected,device=device)
    fit=torch.tensor([r['role']=='fit' for r in records],device=device);test=~fit
    all_variants=np.stack([variants(im,m,b) for im,m,b in zip(images,masks,boxes)])
    inputs=torch.tensor(images.transpose(0,3,1,2).copy(),device=device,dtype=torch.float32)/255
    varinputs=torch.tensor(all_variants.transpose(0,1,4,2,3).copy(),device=device,dtype=torch.float32)/255
    slices=[slice(0,4),slice(4,8),slice(8,10)]
    capture={};replacement={'value':None}
    def hook(_,__,value):
        capture['z']=value.detach().clone()
        if replacement['value'] is not None:
            return replacement['value']
    handle=model.bloki[-1].mlp.act.register_forward_hook(hook)
    @torch.no_grad()
    def forward(x):
        predictions=[];latents=[]
        for batch in x.split(3):
            pred,_=model(batch); predictions.append(pred); latents.append(capture['z'])
        return torch.cat(predictions),torch.cat(latents)
    @torch.no_grad()
    def intervene(z):
        results=[]
        for first in range(0,12,3):
            replacement['value']=z[first:first+3]; pred,_=model(inputs[first:first+3]);results.append(pred)
        replacement['value']=None
        b,s=module.dekoduj(torch.cat(results));return iou(b[:,0],truth),s[:,0]
    # Five seeded random controls; each independent of checkpoint/factor/test outcomes.
    random_bases=[torch.linalg.qr(torch.randn(768,8,device=device),mode='reduced').Q for _ in range(5)]
    rows=[];probe_rows=[];overlap_rows=[];base_rows=[];saved_bases=[];saved_z=[];previous=None;previous_bases=None;previous_iou=None
    steps=list(range(0,args.max_step+1,50))
    with torch.inference_mode():
        for step in steps:
            checkpoint=SOURCE/f'wagi/k{step:04d}.pt'
            weights=torch.load(checkpoint,map_location='cpu',weights_only=False);assert weights['krok']==step
            model.load_state_dict(weights['net']);pred,z=forward(inputs)
            predicted_boxes,scores=module.dekoduj(pred);quality=iou(predicted_boxes[:,0],truth)
            for i in range(12):base_rows.append(dict(step=step,image=i,role=records[i]['role'],iou=float(quality[i]),score=float(scores[i,0])))
            bases=[];responses=[]
            for f,sl in zip(FACTORS,slices):
                v=varinputs[:,sl]; _,vz=forward(v.flatten(0,1)); response=vz.reshape(12,v.shape[1],600,768)-z[:,None]
                responses.append(response)
                fitted=response[fit].transpose(1,2)[selected_t[fit]].reshape(-1,768)
                # Uncentred SVD retains common systematic displacement, not only covariance.
                gram=fitted.T@fitted;vals,vecs=torch.linalg.eigh(gram);q=vecs[:,-8:];bases.append(q)
            for f,q in zip(FACTORS,bases):
                for g,res in zip(FACTORS,responses):
                    for i in range(12):
                        rr=res[i,:,selected_t[i],:].reshape(-1,768)
                        energy=rr.square().sum();captured=(rr@q).square().sum()
                        probe_rows.append(dict(step=step,image=i,role=records[i]['role'],basis=f,perturbation=g,capture=float(captured/energy.clamp(min=1e-12)),response_rms=float(rr.square().mean().sqrt())))
            for f,q in zip(FACTORS,bases):
                for g,r in zip(FACTORS,bases):overlap_rows.append(dict(step=step,basis=f,other=g,overlap=float((q.T@r).square().sum()/8)))
            if previous is not None:
                delta=z-previous
                random_edits=[(delta@q)@q.T for q in random_bases]
                random_quality=[intervene(z-edit)[0] for edit in random_edits]
                for f,q in zip(FACTORS,previous_bases):
                    edit=(delta@q)@q.T;rollback,_=intervene(z-edit)
                    # Norm-matched random direction control per image; includes all tokens just like semantic rollback.
                    norms=edit.flatten(1).norm(dim=1)
                    matched_quality=[]
                    for rnd in random_edits:
                        scale=norms/rnd.flatten(1).norm(dim=1).clamp(min=1e-12)
                        matched_quality.append(intervene(z-rnd*scale[:,None,None])[0])
                    for i in range(12):
                        d=delta[i,selected_t[i]];fraction=(d@q).square().sum()/d.square().sum().clamp(min=1e-12)
                        rows.append(dict(step=step,previous_step=step-50,image=i,role=records[i]['role'],factor=f,iou=float(quality[i]),delta_iou=float(quality[i]-previous_iou[i]),motion_rms=float(d.square().mean().sqrt()),motion_fraction=float(fraction),rollback_iou=float(rollback[i]),rollback_effect=float(quality[i]-rollback[i]),random_rank_effect=float(quality[i]-torch.stack(random_quality)[:,i].mean()),random_norm_effect=float(quality[i]-torch.stack(matched_quality)[:,i].mean()),edit_norm=float(norms[i])))
            saved_bases.append(torch.stack(bases).cpu().numpy());saved_z.append(z.cpu().numpy().astype(np.float16))
            previous=z;previous_bases=bases;previous_iou=quality
            write_csv(out/'MOTION.csv',rows) if rows else None
            write_csv(out/'PROBES.csv',probe_rows);write_csv(out/'DETECTION.csv',base_rows);write_csv(out/'OVERLAP.csv',overlap_rows)
            print(f'checkpoint {step}: mean IoU={float(quality[test].mean()):.4f}; elapsed={time.time()-start:.1f}s',flush=True)
    handle.remove()
    np.savez_compressed(out/'STATES_BASES.npz',steps=steps,bases=np.stack(saved_bases),latents=np.stack(saved_z),selected=selected,boxes=np.stack(boxes))
    # Verify measured branch against original saved trajectory at endpoints for first probe.
    history=np.load(SOURCE/'dane/LATENT_GESTY.npy',mmap_mode='r')
    checks=[]
    for idx in [0,len(steps)-1]:
        actual=history[steps[idx],0].astype(np.float32);observed=saved_z[idx][0].astype(np.float32)
        checks.append(dict(step=steps[idx],rms_float16_export_error=float(np.sqrt(np.mean((actual-observed)**2))),source_rms=float(np.sqrt(np.mean(actual**2)))))
    summaries={}
    for f in FACTORS:
        rs=[r for r in rows if r['role']=='holdout' and r['factor']==f]
        rho,p=spearmanr([r['motion_fraction'] for r in rs],[r['delta_iou'] for r in rs])
        centered_x=[];centered_y=[]
        for step in steps[1:]:
            part=[r for r in rs if r['step']==step]
            x=np.array([r['motion_fraction'] for r in part]);y=np.array([r['delta_iou'] for r in part])
            centered_x.extend(x-x.mean());centered_y.extend(y-y.mean())
        within=spearmanr(centered_x,centered_y).statistic
        summaries[f]=dict(spearman=float(rho),within_checkpoint_spearman=float(within),mean_motion_fraction=float(np.mean([r['motion_fraction'] for r in rs])),mean_rollback_effect=float(np.mean([r['rollback_effect'] for r in rs])),mean_norm_control_effect=float(np.mean([r['random_norm_effect'] for r in rs])))
    info=dict(steps=steps,fit_images=[i for i,r in enumerate(records) if r['role']=='fit'],test_images=[i for i,r in enumerate(records) if r['role']=='holdout'],rank=8,elapsed_seconds=time.time()-start,gpu=torch.cuda.get_device_name(),gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20,source_checks=checks,summary=summaries,source=str(SOURCE),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations=['One training run; all images are training images for detector','Shape probe also warps background','Overlapping subspaces; fractions are not additive','Correlations are descriptive with temporal dependence','Interventions affect last MLP branch only; may leave activation distribution'])
    (out/'RESULT.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'run_snapshot.py').write_bytes(Path(__file__).read_bytes())
    # Diagnostic visualization, no generator required.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    hold=[r for r in base_rows if r['role']=='holdout']
    axes[0,0].plot(steps,[np.mean([r['iou'] for r in hold if r['step']==s]) for s in steps]);axes[0,0].set(title='IoU: cztery obrazy kontrolne',xlabel='Krok uczenia',ylabel='IoU',ylim=(0,1))
    for f in FACTORS:
        rr=[r for r in rows if r['role']=='holdout' and r['factor']==f]
        axes[0,1].plot(steps[1:],[np.mean([r['motion_fraction'] for r in rr if r['step']==s]) for s in steps[1:]],label=f)
        axes[1,0].plot(steps[1:],[np.mean([r['rollback_effect']-r['random_norm_effect'] for r in rr if r['step']==s]) for s in steps[1:]],label=f)
    axes[0,1].set(title='Udział ruchu w sondowanych podprzestrzeniach',xlabel='Krok',ylabel='Udział energii (nieaddytywny)');axes[0,1].legend()
    axes[1,0].axhline(0,color='k',lw=.5);axes[1,0].set(title='Efekt cofnięcia minus losowa kontrola normy',xlabel='Krok',ylabel='Różnica IoU');axes[1,0].legend()
    matrix=np.array([[np.mean([r['capture'] for r in probe_rows if r['role']=='holdout' and r['basis']==f and r['perturbation']==g]) for g in FACTORS] for f in FACTORS])
    im=axes[1,1].imshow(matrix,vmin=0,vmax=1);axes[1,1].set_xticks(range(3),FACTORS,rotation=15);axes[1,1].set_yticks(range(3),FACTORS);axes[1,1].set(title='Selektywność sond: średnia ze wszystkich kroków',xlabel='Perturbacja',ylabel='Podprzestrzeń');fig.colorbar(im,ax=axes[1,1])
    for i in range(3):
        for j in range(3):axes[1,1].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]<.5 else 'black')
    fig.tight_layout();fig.savefig(out/'DIAGNOSTICS.png',dpi=160);plt.close(fig)
    collage=np.concatenate([images[8],all_variants[8,0],all_variants[8,4],all_variants[8,8]],axis=1);cv2.imwrite(str(out/'PERTURBATIONS.png'),cv2.cvtColor(collage,cv2.COLOR_RGB2BGR))
    report=['# Semantyka ruchu — pilotaż','',f'21 checkpointów (lub zakres uruchomienia: {len(steps)}), osiem obrazów dopasowania, cztery obrazy sprawdzenia sond. Wszystkie pochodzą z treningu detektora. Czas: {info["elapsed_seconds"]:.1f} s.','', 'IoU dotyczy ramki o najwyższej pewności dla pojedynczego narzędzia; nie jest mAP. Podprzestrzenie mają po osiem kierunków i mogą się nakładać.','']
    for f,s in summaries.items():report.append(f'- {f}: średni udział ruchu {s["mean_motion_fraction"]:.3f}; korelacja z ΔIoU {s["spearman"]:.3f}; po usunięciu średniej dla każdego kroku {s["within_checkpoint_spearman"]:.3f}; efekt cofnięcia IoU {s["mean_rollback_effect"]:.4f}; losowa kontrola o tej samej normie {s["mean_norm_control_effect"]:.4f}.')
    report+=['','Dodatni efekt cofnięcia oznacza, że usunięcie tej części ruchu pogarsza detekcję. Ujemny oznacza poprawę po ingerencji. Interpretację semantyczną dopuszcza dopiero selektywność sond, a nie ich nazwa. Korelacje są opisowe, nie wyliczamy naiwnych wartości istotności dla zależnych kroków.','', 'Kształt oznacza tu zmianę proporcji całego kadru; tło też się zmienia. Kolor dotyczy maski narzędzia. Pozycja oznacza cyfrową translację, nie pełny ruch kamery 3D. Ruch jest mierzony na niezmienionym obrazie w pełnych 768 wymiarach, a nie w rzucie Fibonacciego.','', 'Interwencja cofa część zmiany ostatniej gałęzi MLP, z bazą wyznaczoną na wcześniejszym checkpointcie. Nie jest odwróceniem uczenia całej sieci. Wartości mogą wyjść poza rozkład aktywacji. Jedna trajektoria treningu i cztery obrazy nie uzasadniają jeszcze twierdzenia publikacyjnego.','', 'Pliki: MOTION.csv, PROBES.csv, DETECTION.csv, OVERLAP.csv, STATES_BASES.npz, RESULT.json, DIAGNOSTICS.png, PERTURBATIONS.png.']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(info['summary'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
