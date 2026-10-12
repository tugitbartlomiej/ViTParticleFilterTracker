"""32 independent pure-noise samples from the trained Eden SD."""
import test_lokalny as h
import json, secrets, traceback
import numpy as np
import torch
from PIL import Image, ImageDraw
from generate import load_trained
from model import MalyViT

def main():
    checkpoint = h.OUT/'best_checkpoint.pt'
    h.OUT = h.ROOT/'outputs/vit_sd_32_nowy_latent_20261006'
    h.OUT.mkdir(parents=True, exist_ok=True)
    if (h.OUT/'RESULT.json').exists():
        raise RuntimeError('Completed results already exist')
    torch.set_num_threads(4)
    target_path=h.ROOT/'outputs/vit_latent128_nowy_checkpoint_granice_20261006/LATENT_PELNY.npz'
    vit_path=h.ROOT/'outputs/vit_wyuczenie_latentu_jedno_zdjecie_do85/MODEL_best.pt'
    with np.load(target_path) as arrays:
        target=arrays['after'].astype(np.float32)
        j=arrays['channels0'].astype(int)
    assert target.shape==(600,768) and np.isfinite(target).all()
    k=np.setdiff1d(np.arange(768),j)
    manifest_path=h.OUT/'MANIFEST.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['target_sha256']==h.sha(target_path)
        assert manifest['checkpoint_sha256']==h.sha(checkpoint)
    else:
        seeds=[]
        while len(seeds)<32:
            seed=secrets.randbits(32)
            if seed not in seeds: seeds.append(seed)
        manifest=dict(target=str(target_path),target_sha256=h.sha(target_path),
            checkpoint=str(checkpoint),checkpoint_sha256=h.sha(checkpoint),
            evaluation_vit=str(vit_path),evaluation_vit_sha256=h.sha(vit_path),
            seeds=seeds,steps=100,start='Independent Gaussian noise; no input image',
            limitation='SD trained with original ViT K1000; supplied target belongs to ViT fitted at step 6402. Compatibility not established.')
        h.write('MANIFEST.json',manifest)
    h.status('loading',samples=32)
    trained=load_trained(h.SD/'ema_020000',h.SD/'sd-vae-ft-mse',checkpoint)
    trained.vae.enable_tiling()
    manifest['checkpoint_step']=trained.step
    h.write('MANIFEST.json',manifest)
    for index,seed in enumerate(manifest['seeds'],1):
        path=h.OUT/f'{index:02d}_seed_{seed}.png'
        if path.exists():
            with Image.open(path) as im: im.verify()
            continue
        h.status('generating',sample=index,total=32,seed=seed,steps=100)
        rgb=trained.generate(torch.from_numpy(target)[None],steps=100,seed=seed)
        assert torch.isfinite(rgb).all()
        pixels=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
        Image.fromarray(pixels).save(path)
        del rgb,pixels
        h.status('generated',completed=index,total=32)
    del trained
    torch.cuda.empty_cache()
    h.status('evaluating')
    vit=MalyViT()
    vit.load_state_dict(torch.load(vit_path,map_location='cpu',weights_only=True)['net'],strict=True)
    vit.cuda().eval().requires_grad_(False)
    vit.zapisuj_latent(True)
    records=[]
    sheet=Image.new('RGB',(4*288,8*212),'white')
    draw=ImageDraw.Draw(sheet)
    cards=[]
    for index,seed in enumerate(manifest['seeds'],1):
        path=h.OUT/f'{index:02d}_seed_{seed}.png'
        with Image.open(path) as im:
            rgb=np.asarray(im.convert('RGB')).astype(np.float32)/255
            thumb=im.convert('RGB').resize((288,192))
        metrics=h.inspect(vit,rgb,target,j,k)
        records.append(dict(index=index,seed=seed,file=path.name,metrics=metrics))
        x=((index-1)%4)*288; y=((index-1)//4)*212
        sheet.paste(thumb,(x,y));draw.text((x+4,y+193),f'{index:02d} seed {seed}',fill='black')
        cards.append(f'<figure><img src="{path.name}"><figcaption>{index:02d}: seed {seed}; J128={metrics["value_agreement128"]:.2f}%; cosine={metrics["cosine128"]:.4f}</figcaption></figure>')
    sheet.save(h.OUT/'PRZEGLAD_32.png')
    h.write('RESULT.json',dict(manifest=manifest,samples=records,
        note='Generated images have no ground-truth boxes; no IoU or mAP claimed. Evaluation uses natural forward without activation replacement.'))
    (h.OUT/'GALERIA.html').write_text('<!doctype html><meta charset="utf-8"><title>32 generacje SD</title><style>body{font:16px sans-serif;background:#eee}main{display:grid;grid-template-columns:repeat(4,1fr)}figure{margin:8px;background:white;padding:8px}img{width:100%}</style><h1>32 obrazy SD z nowego sztucznego latentu</h1><p>100 kroków DDIM, niezależne szumy. SD: checkpoint 8500 z Edena. Cel: pełny latent ViT po 6402 aktualizacjach; różnica checkpointów pozostaje ograniczeniem eksperymentu.</p><main>'+''.join(cards)+'</main>',encoding='utf-8')
    h.status('complete',samples=32)

if __name__=='__main__':
    try: main()
    except Exception:
        h.status('failed',traceback=traceback.format_exc())
        raise
