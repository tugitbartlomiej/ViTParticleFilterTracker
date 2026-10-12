"""Czy własny SD od samego szumu odtwarza cel600x768? Brak poprawy względem
tego samego szumu bez prowadzenia obala korzyść tej pojedynczej próby.
Nie wczytuje fotografii, etykiet ani masek zależnych od przykładu.
"""
print('IMPORTING', flush=True)
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from generator_sd import GeneratorSD, SKALA_VAE
from model_snapshot import MalyViT, dekoduj

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/vit_latent768_sd_od_szumu'
INPUT = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90'
SD = Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele')
sys.path.insert(0, str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    print('IMPORTS_READY', flush=True)
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    seed, steps = 20261005, 40
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(INPUT / 'LATENT_PELNY.npz') as f:
        target_np = f['after'].copy()
        channels_np = f['channels0'].copy()
    assert target_np.shape == (600,768) and np.isfinite(target_np).all()
    target = torch.from_numpy(target_np).cuda()[None]
    print('CUDA_READY', flush=True)
    channels = torch.as_tensor(channels_np, device='cuda', dtype=torch.long)
    other = torch.ones(768, device='cuda', dtype=torch.bool)
    other[channels] = False
    scale = target.square().mean(dim=1, keepdim=True).sqrt().clamp_min(.25)
    net = MalyViT().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(INPUT/'MODEL_VIT_K1000.pt', map_location='cpu', weights_only=True)['net'])
    net.zapisuj_latent(True)
    print('VIT_READY', flush=True)
    G = GeneratorSD(unet=SD/'ema_020000', vae=SD/'sd-vae-ft-mse', tekst=SD/'pusty_tekst.pt')
    G.unet.requires_grad_(False)
    G.vae.requires_grad_(False)
    G.vae.enable_gradient_checkpointing()
    G.vae.enable_slicing()
    masks = torch.zeros((1,3,48,72), device='cuda', dtype=torch.float16)
    rng = torch.Generator(device='cuda').manual_seed(seed)
    noise = torch.randn((1,4,48,72), generator=rng, device='cuda', dtype=torch.float16)
    np.savez_compressed(OUT/'SZUM_STARTOWY.npz', noise=noise.cpu().numpy())
    # Illustration of noise only; not an encoded source photograph.
    preview = noise[0,:3].float().cpu().permute(1,2,0).numpy()
    preview = np.clip((preview+3)*255/6,0,255).astype('uint8')
    cv2.imwrite(str(OUT/'SZUM_STARTOWY.png'),cv2.cvtColor(cv2.resize(preview,(576,384),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR))

    def decode(x):
        return (G.vae.decode(x.half()/SKALA_VAE).sample.float().clamp(-1,1)+1)/2

    def energy(x):
        rgb = decode(x)
        _, z = net(F.interpolate(rgb,size=(320,480),mode='bilinear',align_corners=False,antialias=True))
        d = ((z-target)/scale).square()
        ej, eo = d[:,:,channels].mean(), d[:,:,other].mean()
        return .5*(ej+eo), ej, eo

    def save(x, name):
        with torch.no_grad():
            rgb = (decode(x)[0].permute(1,2,0)*255).round().byte().cpu().numpy()
        p = OUT/name
        assert cv2.imwrite(str(p),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        return p

    history = []
    def sample(guided):
        G.sched.set_timesteps(steps)
        ac = G.sched.alphas_cumprod.cuda()
        x = noise.clone()
        for i,t in enumerate(G.sched.timesteps):
            G.unet.to('cuda')
            with torch.no_grad():
                eps = G.unet(torch.cat([x,masks],1),t,encoder_hidden_states=G.emb).sample.float()
                a = ac[t].float()
                clean = (x.float()-(1-a).sqrt()*eps)/a.sqrt()
            if guided and 8 <= i < steps-1:
                G.unet.to('cpu')
                torch.cuda.empty_cache()
                for j in range(2):
                    v = clean.detach().requires_grad_(True)
                    loss, ej, eo = energy(v)
                    grad, = torch.autograd.grad(loss,v)
                    assert torch.isfinite(grad).all()
                    rms = grad.square().mean().sqrt().clamp_min(1e-8)
                    # Backtracking only accepts improvements in decoded-image energy.
                    chosen = clean
                    for rate in (.04,.01,.002):
                        candidate = (v-rate*grad/rms).detach()
                        with torch.no_grad():
                            value,_,_ = energy(candidate)
                        if value < loss.detach():
                            chosen = candidate
                            break
                    clean = chosen.detach()
                    history.append(dict(stage='ddim',step=i+1,iteration=j,energy=float(loss.detach()),j128=float(ej.detach()),other640=float(eo.detach())))
                eps = (x.float()-a.sqrt()*clean)/(1-a).sqrt()
            x = G.sched.step(eps,t,x.float()).prev_sample.half()
            print('DDIM', 'guided' if guided else 'control',i+1,steps,flush=True)
            write(OUT/'PROGRESS.json',dict(status='running',stage='guided' if guided else 'control',step=i+1,total=steps,seconds=time.perf_counter()-start))
        G.unet.to('cpu')
        torch.cuda.empty_cache()
        return x

    print('MODELS_READY',flush=True)
    control = sample(False)
    save(control,'OBRAZ_SD_BEZ_PROWADZENIA.png')
    guided = sample(True)
    save(guided,'OBRAZ_SD_PO_DDIM.png')
    anchor = guided.detach().float()
    variable = anchor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([variable],lr=.025)
    with torch.no_grad():
        best_energy = float(energy(variable)[0])
    best = variable.detach().clone()
    for i in range(120):
        optimizer.zero_grad(set_to_none=True)
        value,ej,eo = energy(variable)
        objective = value+.02*(variable-anchor).square().mean()
        objective.backward()
        assert torch.isfinite(variable.grad).all()
        if float(value.detach()) < best_energy:
            best_energy = float(value.detach())
            best = variable.detach().clone()
        optimizer.step()
        history.append(dict(stage='refine',step=i+1,energy=float(value.detach()),j128=float(ej.detach()),other640=float(eo.detach())))
        if (i+1)%10 == 0:
            save(best,'OBRAZ_SD_W_TRAKCIE.png')
            print('REFINE',i+1,120,best_energy,flush=True)
            write(OUT/'PROGRESS.json',dict(status='running',stage='refine',step=i+1,total=120,energy=best_energy,seconds=time.perf_counter()-start))
    with torch.no_grad():
        last = float(energy(variable)[0])
        if last < best_energy:
            best = variable.detach().clone()
    final_path = save(best,'OBRAZ_SD.png')
    np.savez_compressed(OUT/'LATENT_SD_VAE.npz',latent=best.cpu().numpy(),scale=SKALA_VAE)

    def evaluate(p, dump=False):
        rgb = cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA))).permute(2,0,1)[None].float().cuda()/255
        with torch.no_grad():
            o,z = net(image)
            diff = (z-target).square()
            d = diff/scale.square()
            boxes,scores = dekoduj(o)
            k = int(scores[0].argmax())
            result = dict(energy=float(.5*(d[:,:,channels].mean()+d[:,:,other].mean())),mse768=float(diff.mean()),mse128=float(diff[:,:,channels].mean()),mse640=float(diff[:,:,other].mean()),confidence=float(scores[0,k]),bbox_xyxy=boxes[0,k].cpu().tolist())
        if dump:
            np.savez_compressed(OUT/'LATENT_VIT_OBRAZU_SD.npz',latent=z.cpu().numpy(),target=target_np,channels0=channels_np,detector_output=o.cpu().numpy())
        return result
    baseline = evaluate(OUT/'OBRAZ_SD_BEZ_PROWADZENIA.png')
    final = evaluate(final_path,True)
    result = dict(status='complete',ziarno=seed,seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),torch=torch.__version__,image=str(final_path),target_path=str(INPUT/'LATENT_PELNY.npz'),target_sha256=sha(INPUT/'LATENT_PELNY.npz'),checkpoint_sha256=sha(INPUT/'MODEL_VIT_K1000.pt'),sd_sources=G.zrodlo,sd_unet_sha256=sha(SD/'ema_020000/diffusion_pytorch_model.safetensors'),sd_vae_sha256=sha(SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'),text_sha256=sha(SD/'pusty_tekst.pt'),source_image_used=False,annotations_used=False,masks='all zeros',text='fixed empty',target_shape=list(target_np.shape),ddim_steps=steps,refinement_steps=120,control=baseline,generated=final,energy_reduction_vs_control=1-final['energy']/baseline['energy'],history=history,limitation='One noise seed. Artificial ViT activation may have no exact natural-image preimage. Confidence is not accuracy or AP.')
    write(OUT/'RESULT.json',result)
    write(OUT/'PROGRESS.json',dict(status='complete',seconds=result['seconds']))
    zapisz_wynik(__file__,'sd_od_szumu',result)
    report = f"# SD z samego szumu\n\nŹródło obrazu i maski nie były użyte. Pełny latent600×768 służył do prowadzenia gradientem.\n\nEnergia kontrolna: {baseline['energy']:.6f}; końcowa: {final['energy']:.6f}. Redukcja: {100*result['energy_reduction_vs_control']:.2f}%.\n\nPewność naturalnego detektora: {100*final['confidence']:.2f}%. Nie jest to mAP ani potwierdzenie poprawnej ramki. Surowy MSE768: {final['mse768']:.6f}; MSE128: {final['mse128']:.6f}.\n\nTen obraz nie odtwarza dokładnie celu; możliwy brak osiągalnego obrazu dla sztucznej interwencji.\n"
    (OUT/'RAPORT.md').write_text(report,encoding='utf-8')
    (Path(__file__).parent/'WYNIKI_SD.md').write_text(report,encoding='utf-8')
    print('COMPLETE',json.dumps(dict(generated=final,control=baseline,seconds=result['seconds'])),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'FAILURE.json',dict(status='failed',type=type(e).__name__,error=str(e)))
        raise
