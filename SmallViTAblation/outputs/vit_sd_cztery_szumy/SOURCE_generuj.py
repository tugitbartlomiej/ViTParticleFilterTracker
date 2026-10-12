"""Czy ograniczone prowadzenie poprawia dopasowanie dla czterech szumów?
Brak poprawy względem kontroli z tym samym ziarnem obala korzyść prowadzenia.
Dokładny cel jest nieosiągalny. Zmiany dotyczą tylko denoisingu, wagi zamrożone.
"""
print('IMPORTING', flush=True)
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dane import ROOT, OUT, TARGET, DIAGNOSTIC, SD, SEEDS, load_target
from generator_sd import GeneratorSD, SKALA_VAE
from model_snapshot import MalyViT, dekoduj
sys.path.insert(0, str(ROOT))
from src.narzedzia import deklaracja_wynikow as registry

STEPS, FIRST_GUIDED, LAST_UNGUIDED = 60, 30, 3
ETA, TRUST_RMS, PRIOR_WEIGHT = .006, .012, .005

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def js(p, data):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(p)

def weights_hash(net):
    h = hashlib.sha256()
    for name, value in net.state_dict().items():
        h.update(name.encode()); h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'RESULT.json').exists() or list(OUT.glob('SZUM_*.npz')):
        raise RuntimeError('Existing run: refusing to overwrite saved noises/results')
    def progress(**data):
        js(OUT/'PROGRESS.json', dict(pid=os.getpid(), seconds=time.perf_counter()-start, **data))
    progress(status='starting', phase='inputs')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target_np, channels_np = load_target()
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding='utf-8'))
    assert diagnostic['exact_target_excluded'] is True
    provenance = dict(seeds=list(SEEDS), target=str(TARGET/'LATENT_PELNY.npz'),
        target_sha256=sha(TARGET/'LATENT_PELNY.npz'), checkpoint=str(TARGET/'MODEL_VIT_K1000.pt'),
        checkpoint_sha256=sha(TARGET/'MODEL_VIT_K1000.pt'), diagnostic=str(DIAGNOSTIC),
        diagnostic_sha256=sha(DIAGNOSTIC), exact_target_excluded=True,
        sd_sources=dict(unet=str(SD/'ema_020000'), vae=str(SD/'sd-vae-ft-mse'), text=str(SD/'pusty_tekst.pt')),
        sd_unet_sha256=sha(SD/'ema_020000/diffusion_pytorch_model.safetensors'),
        sd_vae_sha256=sha(SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'),
        text_sha256=sha(SD/'pusty_tekst.pt'), source_image_used=False, training=False,
        target_shape=list(target_np.shape), sampler='DDIM eta=0', ddim_steps=STEPS,
        first_guided_index=FIRST_GUIDED, last_unguided_steps=LAST_UNGUIDED,
        guidance_eta=ETA, clean_step_trust_rms=TRUST_RMS, prior_weight=PRIOR_WEIGHT,
        gradient_updates_per_guided_step=2, post_vae_refinement_steps=0,
        masks='three zero masks', text='fixed empty embedding', torch=torch.__version__,
        git_dirty_note='Known dirty worktree; _brudne_drzewo=lambda:True avoids large untracked dataset scan.')
    for p in Path(__file__).parent.iterdir():
        if p.is_file(): shutil.copy2(p, OUT/('SOURCE_'+p.name))
    shutil.copy2(DIAGNOSTIC, OUT/'DIAGNOSTYKA_OSIAGALNOSCI.json')
    js(OUT/'PROVENANCE.json', provenance)
    # Wait cooperatively for capacity; never terminate any other GPU process.
    while True:
        free, total = torch.cuda.mem_get_info()
        if free >= 3500*1024*1024: break
        progress(status='waiting_gpu', phase='capacity', free_bytes=free)
        print('WAIT_GPU', free, flush=True); time.sleep(30)
    progress(status='loading_models', phase='vit_sd')
    target = torch.from_numpy(target_np).cuda()[None]
    channels = torch.as_tensor(channels_np, dtype=torch.long, device='cuda')
    other = torch.ones(768, dtype=torch.bool, device='cuda'); other[channels] = False
    norm = target.square().mean().clamp_min(.25).detach()
    net = MalyViT().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(TARGET/'MODEL_VIT_K1000.pt', map_location='cpu', weights_only=True)['net'])
    net.zapisuj_latent(True)
    before = weights_hash(net)
    G = GeneratorSD(dev='cpu', unet=SD/'ema_020000', vae=SD/'sd-vae-ft-mse', tekst=SD/'pusty_tekst.pt')
    G.unet.requires_grad_(False); G.vae.requires_grad_(False)
    G.unet.set_attention_slice('auto')
    G.vae.enable_gradient_checkpointing(); G.vae.enable_slicing()
    G.emb = G.emb.cuda()
    masks = torch.zeros((1,3,48,72), dtype=torch.float16, device='cuda')
    provenance['device'] = torch.cuda.get_device_name()
    js(OUT/'PROVENANCE.json', provenance)
    noises = {}
    for seed in SEEDS:
        rng = torch.Generator(device='cuda').manual_seed(seed)
        noise = torch.randn((1,4,48,72), device='cuda', dtype=torch.float16, generator=rng)
        noises[seed] = noise
        np.savez_compressed(OUT/f'SZUM_{seed}.npz', noise=noise.cpu().numpy(), seed=seed,
                            distribution='independent_standard_gaussian', initial_sigma=1.0)
    print('MODELS_READY FOUR_NOISES_SAVED', flush=True)
    history, variants, arrays = [], [], {}
    def decode(x):
        return (G.vae.decode(x.half()/SKALA_VAE).sample.float().clamp(-1,1)+1)/2
    def energy(clean, anchor):
        image = decode(clean)
        _, z = net(F.interpolate(image, size=(320,480), mode='bilinear', align_corners=False, antialias=True))
        mse = (z-target).square().mean()/norm
        prior = (clean-anchor).square().mean()/TRUST_RMS**2
        return mse+PRIOR_WEIGHT*prior, mse
    def evaluate(x, tag):
        G.unet.to('cpu'); torch.cuda.empty_cache(); G.vae.to('cuda')
        with torch.no_grad():
            rgb = (decode(x)[0].permute(1,2,0)*255).round().byte().cpu().numpy()
        image_path = OUT/f'OBRAZ_{tag}.png'
        assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # Measurement of the serialized natural image, never a replaced hook.
        rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(np.ascontiguousarray(cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA))).permute(2,0,1)[None].float().cuda()/255
        with torch.no_grad():
            output,z = net(inp); d = (z-target).square()
            boxes,scores = dekoduj(output); k = int(scores[0].argmax())
            metrics = dict(mse768=float(d.mean()), mse128=float(d[:,:,channels].mean()),
                mse640=float(d[:,:,other].mean()), cosine768=float(F.cosine_similarity(z.flatten(1),target.flatten(1))[0]),
                confidence=float(scores[0,k]), bbox_xyxy=boxes[0,k].cpu().tolist())
            natural = z[0].cpu().numpy().copy()
        np.savez_compressed(OUT/f'LATENT_{tag}.npz', natural=natural, target=target_np,
            delta=natural-target_np, channels0=channels_np, detector_output=output.cpu().numpy())
        np.savez_compressed(OUT/f'VAE_{tag}.npz', latent=x.cpu().numpy(), scale=SKALA_VAE)
        arrays[tag] = (rgb.astype(np.float32)/255, natural)
        net._latent = None
        return image_path, metrics
    for seed in SEEDS:
        for guided in (False, True):
            tag = f'{seed}_prowadzony' if guided else f'{seed}_kontrola'
            x = noises[seed].clone()
            G.sched.set_timesteps(STEPS); ac = G.sched.alphas_cumprod.cuda()
            for i,t in enumerate(G.sched.timesteps):
                G.vae.to('cpu'); torch.cuda.empty_cache(); G.unet.to('cuda')
                with torch.no_grad():
                    eps = G.unet(torch.cat([x,masks],1),t,encoder_hidden_states=G.emb).sample.float()
                    a = ac[t].float(); clean = (x.float()-(1-a).sqrt()*eps)/a.sqrt()
                if guided and FIRST_GUIDED <= i < STEPS-LAST_UNGUIDED:
                    G.unet.to('cpu'); torch.cuda.empty_cache(); G.vae.to('cuda')
                    anchor = clean.detach().clone()
                    for iteration in range(2):
                        variable = clean.detach().requires_grad_(True)
                        loss,mse = energy(variable,anchor)
                        grad, = torch.autograd.grad(loss,variable)
                        assert torch.isfinite(grad).all() and torch.isfinite(loss)
                        direction = grad/grad.square().mean().sqrt().clamp_min(1e-9)
                        accepted = False; used_rate = 0.
                        for rate in (ETA, ETA/3, ETA/9):
                            candidate = (variable.detach()-rate*direction).detach()
                            delta = candidate-anchor
                            candidate = anchor+delta*(TRUST_RMS/delta.square().mean().sqrt().clamp_min(TRUST_RMS)).clamp_max(1.)
                            with torch.no_grad(): new_loss,_ = energy(candidate,anchor)
                            if new_loss < loss.detach():
                                clean = candidate.detach(); accepted = True; used_rate = rate; break
                        history.append(dict(tag=tag, step=i+1, iteration=iteration, accepted=accepted,
                            rate=used_rate, relative_mse=float(mse.detach()), objective=float(loss.detach()),
                            delta_rms=float((clean-anchor).square().mean().sqrt())))
                        net._latent = None
                        del variable, loss, mse, grad, direction, candidate, new_loss
                    eps = (x.float()-a.sqrt()*clean)/(1-a).sqrt()
                x = G.sched.step(eps,t,x.float()).prev_sample.half()
                assert torch.isfinite(x).all()
                progress(status='running', phase='denoising', tag=tag, seed=seed,
                    guided=guided, step=i+1, total=STEPS, variants_complete=len(variants))
                print('DDIM',tag,i+1,STEPS,flush=True)
            p,metrics = evaluate(x,tag)
            variant = dict(tag=tag, ziarno=seed, guided=guided, image=str(p), image_sha256=sha(p),
                noise_sha256=sha(OUT/f'SZUM_{seed}.npz'), metrics=metrics)
            variants.append(variant); js(OUT/'VARIANTS.json',variants); js(OUT/'HISTORY.json',history)
            registry._brudne_drzewo = lambda: True
            registry.zapisz_wynik(__file__,tag,variant)
            print('VARIANT_DONE',json.dumps(variant),flush=True)
    diversity = []
    for guided in (False,True):
        group = [v for v in variants if v['guided']==guided]
        for i,v in enumerate(group):
            for w in group[i+1:]:
                im,z = arrays[v['tag']]; im2,z2 = arrays[w['tag']]
                diversity.append(dict(guided=guided, seed_a=v['ziarno'], seed_b=w['ziarno'],
                    pixel_rms=float(np.sqrt(np.mean((im-im2)**2))), latent_mse=float(np.mean((z-z2)**2)),
                    latent_cosine=float(np.dot(z.ravel(),z2.ravel())/(np.linalg.norm(z)*np.linalg.norm(z2)))))
    comparisons = []
    for seed in SEEDS:
        c,g = [v for v in variants if v['ziarno']==seed]
        comparisons.append(dict(seed=seed, control=c['metrics'], guided=g['metrics'],
            mse_reduction_vs_control=1-g['metrics']['mse768']/c['metrics']['mse768']))
    assert weights_hash(net)==before
    result = dict(status='complete', ziarno=list(SEEDS), seconds=time.perf_counter()-start,
        provenance=provenance, weights_vit_unchanged=True, variants=variants,
        comparisons=comparisons, diversity=diversity,
        limitation='Approximations only: original artificial target proven unattainable for frozen checkpoint. Confidence is not AP. Four seeds do not establish generalization; no realism metric was validated.')
    js(OUT/'RESULT.json',result); js(OUT/'DIVERSITY.json',diversity)
    registry.zapisz_wynik(__file__,'cztery_szumy',result)
    report = '# Cztery niezależne szumy SD\n\nWyłącznie przybliżenia oryginalnego sztucznego celu. Dokładny cel wyklucza diagnostyka osiągalności. Wagi ViT zamrożone. Bez zdjęcia źródłowego, treningu i optymalizacji po DDIM.\n\n'
    for v in comparisons:
        g,c = v['guided'],v['control']
        report += f"Ziarno {v['seed']}: MSE768={g['mse768']:.6f}, kontrola={c['mse768']:.6f}, zmiana={100*v['mse_reduction_vs_control']:.3f} procent. MSE128={g['mse128']:.6f}, MSE640={g['mse640']:.6f}, cosinus={g['cosine768']:.6f}, pewność={g['confidence']:.6f}.\n\n"
    report += 'Różnorodność: DIVERSITY.json, pary obrazów i ich naturalnych latentów dla obu ramion. Pewność detektora nie jest mAP ani dowodem poprawnej detekcji. Prowadzenie może obniżyć MSE bez zachowania realizmu; obrazy wymagają oceny człowieka. Ta próba nie dotyczy treningu na zdjęciu z wyszukiwania.\n'
    (OUT/'RAPORT.md').write_text(report,encoding='utf-8')
    (Path(__file__).parent/'WYNIKI_SD.md').write_text(report,encoding='utf-8')
    js(OUT/'HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('HASHES.json','PROGRESS.json') and p.suffix not in ('.log','.tmp')})
    progress(status='complete', phase='complete', variants_complete=8)
    print('COMPLETE',flush=True)

if __name__ == '__main__':
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        js(OUT/'FAILURE.json',dict(status='failed',type=type(e).__name__,error=str(e),pid=os.getpid()))
        js(OUT/'PROGRESS.json',dict(status='failed',error=str(e),pid=os.getpid()))
        raise
