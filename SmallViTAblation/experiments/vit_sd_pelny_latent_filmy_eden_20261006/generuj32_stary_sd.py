"""Old mask-trained SD, guided by full latest artificial ViT latent."""
import test_lokalny as h
import importlib.util, json, os, traceback, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from model import MalyViT

def main():
    h.OUT=h.ROOT/'outputs/vit_sd_32_stary_sd_nowy_latent_20261006'
    h.OUT.mkdir(parents=True,exist_ok=True)
    if (h.OUT/'RESULT.json').exists(): raise RuntimeError('Completed result exists')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    target_path=h.ROOT/'outputs/vit_latent128_nowy_checkpoint_granice_20261006/LATENT_PELNY.npz'
    vit_path=h.ROOT/'outputs/vit_wyuczenie_latentu_jedno_zdjecie_do85/MODEL_best.pt'
    with np.load(target_path) as arr:
        target=arr['after'].astype(np.float32);j=arr['channels0'].astype(int)
    assert target.shape==(600,768) and np.isfinite(target).all()
    k=np.setdiff1d(np.arange(768),j)
    seeds=json.loads((h.ROOT/'outputs/vit_sd_32_nowy_latent_20261006/MANIFEST.json').read_text(encoding='utf-8'))['seeds']
    config=dict(steps=100,first_guided=50,last_unguided=3,guide_every=2,step_rms=.01,lambda640=.1)
    manifest=dict(target=str(target_path),target_sha256=h.sha(target_path),vit=str(vit_path),vit_sha256=h.sha(vit_path),
        unet=str(h.SD/'ema_020000'),unet_sha256=h.sha(h.SD/'ema_020000/diffusion_pytorch_model.safetensors'),
        vae_sha256=h.sha(h.SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'),
        seeds=seeds,config=config,start='Independent Gaussian noise, no image input; zero spatial masks',
        method='Old mask-trained SD with full-latent gradient guidance; no newly trained Eden adapter loaded')
    h.write('MANIFEST.json',manifest)
    h.status('loading_old_sd')
    spec=importlib.util.spec_from_file_location('old_generator',h.ROOT/'experiments/vit_sd_cztery_szumy/generator_sd.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    g=old.GeneratorSD(dev='cpu',unet=h.SD/'ema_020000',vae=h.SD/'sd-vae-ft-mse',tekst=h.SD/'pusty_tekst.pt')
    g.unet.requires_grad_(False);g.vae.requires_grad_(False)
    g.unet.set_attention_slice('auto');g.vae.enable_gradient_checkpointing();g.vae.enable_slicing()
    g.emb=g.emb.cuda()
    vit=MalyViT();vit.load_state_dict(torch.load(vit_path,map_location='cpu',weights_only=True)['net'],strict=True)
    vit.cuda().eval().requires_grad_(False);vit.zapisuj_latent(True)
    tt=torch.from_numpy(target)[None].cuda()
    ej=max(float((target[:,j]**2).mean()),1e-12);ek=max(float((target[:,k]**2).mean()),1e-12)
    def decode(x): return (g.vae.decode(x.half()/old.SKALA_VAE).sample.float().clamp(-1,1)+1)/2
    def energy(x):
        image=decode(x)
        _,z=vit(F.interpolate(image,size=(320,480),mode='bilinear',align_corners=False,antialias=True))
        error=(z-tt).square()
        return error[:,:,j].mean()/ej+config['lambda640']*error[:,:,k].mean()/ek
    rows=[];hist=[]
    for index,seed in enumerate(seeds,1):
        path=h.OUT/f'{index:02d}_seed_{seed}.png'
        if not path.exists():
            rng=torch.Generator(device='cuda').manual_seed(seed)
            x=torch.randn((1,4,48,72),generator=rng,device='cuda',dtype=torch.float16)
            masks=torch.zeros((1,3,48,72),device='cuda',dtype=torch.float16)
            g.sched.set_timesteps(config['steps']);ac=g.sched.alphas_cumprod.cuda()
            for step,t in enumerate(g.sched.timesteps):
                g.vae.cpu();torch.cuda.empty_cache();g.unet.cuda()
                with torch.no_grad():
                    eps=g.unet(torch.cat([x,masks],1),t,encoder_hidden_states=g.emb).sample.float()
                    a=ac[t].float();clean=(x.float()-(1-a).sqrt()*eps)/a.sqrt()
                guided=config['first_guided']<=step<config['steps']-config['last_unguided'] and step%config['guide_every']==0
                if guided:
                    g.unet.cpu();torch.cuda.empty_cache();g.vae.cuda()
                    variable=clean.detach().requires_grad_(True)
                    loss=energy(variable);grad,=torch.autograd.grad(loss,variable)
                    assert torch.isfinite(loss) and torch.isfinite(grad).all()
                    direction=grad/grad.square().mean().sqrt().clamp_min(1e-9)
                    candidate=clean.detach()-config['step_rms']*direction
                    with torch.no_grad(): after=energy(candidate)
                    accepted=bool(after<loss.detach())
                    if accepted: clean=candidate
                    hist.append(dict(sample=index,step=step+1,loss=float(loss.detach()),after=float(after),accepted=accepted))
                    eps=(x.float()-a.sqrt()*clean)/(1-a).sqrt()
                    vit._latent=None
                    del variable,loss,grad,direction,candidate,after
                x=g.sched.step(eps,t,x.float()).prev_sample.half()
                assert torch.isfinite(x).all()
                if step%10==0: h.status('generating_old_sd',sample=index,total=32,step=step+1,steps=100,seed=seed)
            g.unet.cpu();torch.cuda.empty_cache();g.vae.cuda()
            with torch.no_grad(): pixels=(decode(x)[0].permute(1,2,0)*255).round().byte().cpu().numpy()
            Image.fromarray(pixels).save(path)
            g.vae.cpu();torch.cuda.empty_cache()
            del x,pixels
            h.write('GUIDANCE_HISTORY.json',hist)
        with Image.open(path) as im: rgb=np.asarray(im.convert('RGB')).astype(np.float32)/255
        rows.append(dict(index=index,seed=seed,file=path.name,metrics=h.inspect(vit,rgb,target,j,k)))
        h.write('METRICS.json',rows)
        h.status('generated_old_sd',completed=index,total=32)
    sheet=Image.new('RGB',(1152,1696),'white');draw=ImageDraw.Draw(sheet);cards=[]
    for row in rows:
        i=row['index']-1;x=i%4*288;y=i//4*212
        with Image.open(h.OUT/row['file']) as im: sheet.paste(im.resize((288,192)),(x,y))
        draw.text((x+4,y+193),f"{i+1:02d} seed {row['seed']}",fill='black')
        cards.append(f'<figure><img src="{row["file"]}"><figcaption>{i+1:02d}; J128={row["metrics"]["value_agreement128"]:.2f}%; cosine={row["metrics"]["cosine128"]:.4f}</figcaption></figure>')
    sheet.save(h.OUT/'PRZEGLAD_32.png')
    (h.OUT/'GALERIA.html').write_text('<meta charset="utf-8"><title>Stary SD — 32 próby</title><style>main{display:grid;grid-template-columns:repeat(4,1fr)}img{width:100%}figure{margin:8px}body{font:16px sans-serif}</style><h1>Stary SD: sterowanie pełnym sztucznym latentem</h1><main>'+''.join(cards)+'</main>',encoding='utf-8')
    h.write('RESULT.json',dict(manifest=manifest,samples=rows,note='Natural forward evaluation; no generated-image ground truth or IoU claimed.'))
    h.status('complete',samples=32)

if __name__=='__main__':
    try: main()
    except Exception:
        h.status('failed',traceback=traceback.format_exc())
        raise
