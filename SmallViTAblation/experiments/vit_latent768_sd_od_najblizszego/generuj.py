"""Czy SD img2img z najbliższego zdjęcia zbliża latent do sztucznego celu?
Brak spadkuMSE względem źródła i kontroli obala skuteczność w tej próbie.
Cel nieosiągalny dokładnie. Bez swobodnej optymalizacji końcowego obrazu.
"""
print('IMPORTING',flush=True)
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from generator_sd import GeneratorSD, SKALA_VAE
from model_snapshot import MalyViT, dekoduj

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_latent768_sd_od_najblizszego'
SOURCE=ROOT/'outputs/vit_latent768_pelny_zbior/NAJBLIZSZE_ZDJECIE.jpg'
TARGET=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
SD=Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele')
sys.path.insert(0,str(ROOT))
from src.narzedzia import deklaracja_wynikow as registry


def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def js(p,d):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(p)


def main():
    print('IMPORTS_READY',flush=True)
    start=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Zakonczony wynik istnieje')
    seed,steps=20261006,60
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    raw=cv2.cvtColor(cv2.imread(str(SOURCE)),cv2.COLOR_BGR2RGB)
    source_rgb=cv2.resize(raw,(576,384),interpolation=cv2.INTER_AREA)
    source_image=torch.from_numpy(source_rgb).permute(2,0,1)[None].float().cuda()/255
    source_low=F.avg_pool2d(source_image,16)
    with np.load(TARGET/'LATENT_PELNY.npz') as f:
        target_np=f['after'].copy();J_np=f['channels0'].copy()
    target=torch.from_numpy(target_np).cuda()[None]
    J=torch.as_tensor(J_np,dtype=torch.long,device='cuda')
    other=torch.ones(768,dtype=torch.bool,device='cuda');other[J]=False
    net=MalyViT().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(TARGET/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net'])
    net.zapisuj_latent(True)
    def weights_hash(module):
        h=hashlib.sha256()
        for name,v in module.state_dict().items():h.update(name.encode());h.update(v.cpu().numpy().tobytes())
        return h.hexdigest()
    net_before=weights_hash(net)
    with torch.no_grad():
        _,z=net(F.interpolate(source_image,size=(320,480),mode='bilinear',align_corners=False,antialias=True))
        source_mse=(z-target).square().mean().detach()
    G=GeneratorSD(unet=SD/'ema_020000',vae=SD/'sd-vae-ft-mse',tekst=SD/'pusty_tekst.pt')
    G.unet.requires_grad_(False);G.vae.requires_grad_(False)
    G.vae.enable_gradient_checkpointing();G.vae.enable_slicing()
    with torch.no_grad():
        initial=G.vae.encode((source_image*2-1).half()).latent_dist.mean*SKALA_VAE
    rng=torch.Generator(device='cuda').manual_seed(seed)
    noise=torch.randn(initial.shape,device='cuda',dtype=torch.float16,generator=rng)
    masks=torch.zeros((1,3,48,72),device='cuda',dtype=torch.float16)
    np.savez_compressed(OUT/'START_VAE_I_SZUM.npz',initial=initial.cpu().numpy(),noise=noise.cpu().numpy(),seed=seed)
    shutil.copy2(SOURCE,OUT/'ZDJECIE_STARTOWE.jpg')
    cv2.imwrite(str(OUT/'WEJSCIE_SD.png'),cv2.cvtColor(source_rgb,cv2.COLOR_RGB2BGR))
    def decode(x):return (G.vae.decode(x.half()/SKALA_VAE).sample.float().clamp(-1,1)+1)/2
    def energy(clean):
        image=decode(clean)
        _,z=net(F.interpolate(image,size=(320,480),mode='bilinear',align_corners=False,antialias=True))
        latent=(z-target).square().mean()/source_mse
        lowpass=(F.avg_pool2d(image,16)-source_low).square().mean()
        vae=(clean-initial.float()).square().mean()
        return latent+.05*lowpass/.01+.005*vae/.1,latent,lowpass
    def save(x,name):
        with torch.no_grad():rgb=(decode(x)[0].permute(1,2,0)*255).round().byte().cpu().numpy()
        p=OUT/name;assert cv2.imwrite(str(p),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));return p
    def evaluate(path,dump=False):
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        inp=torch.from_numpy(np.ascontiguousarray(cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA))).permute(2,0,1)[None].float().cuda()/255
        with torch.no_grad():
            output,z=net(inp);d=(z-target).square()
            boxes,scores=dekoduj(output)
            atsd=torch.from_numpy(np.ascontiguousarray(cv2.resize(rgb,(576,384),interpolation=cv2.INTER_AREA))).permute(2,0,1)[None].float().cuda()/255
            low=(F.avg_pool2d(atsd,16)-source_low).square().mean().sqrt()
            cosine=F.cosine_similarity(z.flatten(1),target.flatten(1),dim=1)[0]
            result=dict(mse768=float(d.mean()),mse128=float(d[:,:,J].mean()),mse640=float(d[:,:,other].mean()),cosine768=float(cosine),confidence=float(scores[0,0]),bbox_xyxy=boxes[0,0].cpu().tolist(),lowpass_rms=float(low))
        if dump:np.savez_compressed(OUT/'LATENT_VIT_OBRAZU_SD.npz',natural=z[0].cpu().numpy(),target=target_np,delta=z[0].cpu().numpy()-target_np,channels0=J_np,detector_output=output.cpu().numpy())
        return result
    source_metrics=evaluate(OUT/'ZDJECIE_STARTOWE.jpg')
    G.unet.to('cpu');torch.cuda.empty_cache()
    save(initial,'REKONSTRUKCJA_VAE.png')
    print('MODELS_READY',flush=True)
    history=[];variants=[]
    for strength in (.20,.35):
        G.sched.set_timesteps(steps)
        ts=G.sched.timesteps
        first=steps-round(steps*strength)
        ac=G.sched.alphas_cumprod.cuda()
        a0=ac[ts[first]]
        start_x=(a0.sqrt()*initial.float()+(1-a0).sqrt()*noise.float()).half()
        # A preview of decoding the noisy VAE; this is not pixel-space Gaussian noise.
        save(start_x,f'PODGLAD_ZASZUMIENIA_s{int(strength*100)}.png')
        np.savez_compressed(OUT/f'ZASZUMIONY_VAE_s{int(strength*100)}.npz',latent=start_x.cpu().numpy(),timestep=int(ts[first]))
        for eta in (0.,.006,.018):
            tag=f's{int(strength*100)}_eta{eta:.3f}'
            x=start_x.clone()
            for i in range(first,steps):
                t=ts[i];G.unet.to('cuda')
                with torch.no_grad():
                    eps=G.unet(torch.cat([x,masks],1),t,encoder_hidden_states=G.emb).sample.float()
                    a=ac[t].float();clean=(x.float()-(1-a).sqrt()*eps)/a.sqrt()
                if eta>0 and i<steps-3:
                    G.unet.to('cpu');torch.cuda.empty_cache()
                    for iteration in range(2):
                        variable=clean.detach().requires_grad_(True)
                        loss,latent,lowpass=energy(variable)
                        grad,=torch.autograd.grad(loss,variable)
                        assert torch.isfinite(grad).all()
                        grad_rms=grad.square().mean().sqrt().clamp_min(1e-9)
                        clean=variable.detach()
                        accepted=False
                        for rate in (eta,eta/3,eta/9):
                            candidate=(variable-rate*grad/grad_rms).detach()
                            with torch.no_grad():new_loss,_,_=energy(candidate)
                            if new_loss<loss.detach():clean=candidate;accepted=True;break
                        history.append(dict(tag=tag,step=i+1,iteration=iteration,energy=float(loss.detach()),relative_latent_mse=float(latent.detach()),lowpass_mse=float(lowpass.detach()),accepted=accepted))
                    eps=(x.float()-a.sqrt()*clean)/(1-a).sqrt()
                x=G.sched.step(eps,t,x.float()).prev_sample.half()
                print('DDIM',tag,i-first+1,steps-first,flush=True)
                js(OUT/'PROGRESS.json',dict(status='running',variant=tag,step=i-first+1,total=steps-first,variants_complete=len(variants)))
            G.unet.to('cpu');torch.cuda.empty_cache()
            p=save(x,f'OBRAZ_{tag}.png')
            metrics=evaluate(p)
            variant=dict(tag=tag,strength=strength,eta=eta,image=str(p),guided=eta>0,executed_steps=steps-first,metrics=metrics)
            variants.append(variant)
            np.savez_compressed(OUT/f'VAE_{tag}.npz',latent=x.cpu().numpy())
            js(OUT/'VARIANTS.json',variants)
            print('VARIANT_DONE',json.dumps(variant),flush=True)
    guided=[v for v in variants if v['guided']]
    eligible=[v for v in guided if v['metrics']['lowpass_rms']<=.10]
    selected=min(eligible,key=lambda v:v['metrics']['mse768']) if eligible else min(guided,key=lambda v:v['metrics']['lowpass_rms'])
    matched_control=next(v for v in variants if v['strength']==selected['strength'] and not v['guided'])
    image_path=OUT/'OBRAZ_SD.png';shutil.copy2(selected['image'],image_path)
    final=evaluate(image_path,True)
    assert weights_hash(net)==net_before
    base=dict(status='complete',ziarno=seed,seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),torch=torch.__version__,source=str(SOURCE),source_sha256=sha(SOURCE),target_sha256=sha(TARGET/'LATENT_PELNY.npz'),checkpoint_sha256=sha(TARGET/'MODEL_VIT_K1000.pt'),sd_sources=G.zrodlo,sd_unet_sha256=sha(SD/'ema_020000/diffusion_pytorch_model.safetensors'),sd_vae_sha256=sha(SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'),text_sha256=sha(SD/'pusty_tekst.pt'),ddim_steps=steps,last_unguided_steps=3,masks='zero',text='empty',weights_vit_unchanged=True,source_metrics=source_metrics,selected=selected,matched_control=matched_control,generated=final,structure_constraint_met=bool(eligible),mse_reduction_vs_source=1-final['mse768']/source_metrics['mse768'],mse_reduction_vs_control=1-final['mse768']/matched_control['metrics']['mse768'],variants=variants,history=history,image=str(image_path),limitation='Original artificial target is analytically unattainable; this measures approach only. Lowpass constraint is not a guarantee of realism. Detector confidence is not AP.')
    js(OUT/'RESULT.json',base)
    js(OUT/'PROGRESS.json',dict(status='complete',selected=selected['tag']))
    # Avoid full untracked-tree scan (huge image dataset); current source tree is dirty.
    registry._brudne_drzewo=lambda:True
    registry.zapisz_wynik(__file__,'img2img',dict(base,git_dirty_note='Known dirty worktree from pre-run tracked status; untracked datasets not recursively scanned.'))
    report=f"# SD z najbliższego zdjęcia\n\nŹródłoTR_0064_S1_P03/1181, ten sam sztuczny latent600×768. Wariant wybrany: {selected['tag']}. MSE768 źródła={source_metrics['mse768']:.6f}, obrazuSD={final['mse768']:.6f}, kontroliSD={matched_control['metrics']['mse768']:.6f}. Poprawa względem zdjęcia={100*base['mse_reduction_vs_source']:.3f}%, względem kontroli={100*base['mse_reduction_vs_control']:.3f}%.\n\nPewność detektora: źródło{100*source_metrics['confidence']:.2f}%, SD{100*final['confidence']:.2f}%. MSE128={final['mse128']:.6f}, MSE640={final['mse640']:.6f}. LowpassRMS={final['lowpass_rms']:.6f}.\n\nPozostałe warianty i kontrole zachowano wVARIANTS.json. Zaszumione latentyVAE są zapisane oddzielnie od wizualnego podglądu ich dekodowania. Wagi pozostają zamrożone, latentViT nie był podmieniany. Dokładna realizacja sztucznego celu wykluczona granicami aktywacji. Pewność nie jest jakością detekcji ani mAP.\n"
    (OUT/'RAPORT.md').write_text(report,encoding='utf-8')
    (Path(__file__).parent/'WYNIKI_SD.md').write_text(report,encoding='utf-8')
    shutil.copy2(__file__,OUT/'generuj_source.py')
    js(OUT/'HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='HASHES.json' and p.suffix!='.log'})
    print('COMPLETE',json.dumps(dict(selected=selected['tag'],generated=final,source=source_metrics,seconds=base['seconds'])),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        js(OUT/'FAILURE.json',dict(status='failed',type=type(e).__name__,error=str(e)))
        raise
