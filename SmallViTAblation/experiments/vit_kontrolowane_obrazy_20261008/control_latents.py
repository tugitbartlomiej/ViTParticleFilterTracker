"""Full-token responses to solid colors/gradients, relative to mid-gray."""
from pathlib import Path
import importlib.util,json,csv
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/vit_kontrolowane_obrazy_20261008/pilot_v1'
spec=importlib.util.spec_from_file_location('prior',ROOT/'experiments/vit_semantyka_ruchu_20261008/run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
torch.set_num_threads(4);module=prior.load_model_module();model=module.MalyViT().eval();model.zapisuj_latent(True)
manifest=json.loads((OUT/'IMAGES.json').read_text(encoding='utf8'));names=manifest['controls'];archive=np.load(OUT/'BASES.npz');steps=archive['steps'];means=[];stds=[];rows=[];channels=[]
@torch.inference_mode()
def latent(name):
    bgr=cv2.imread(str(OUT/'images'/f'control_{name}.png'));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    _,z=model(torch.tensor(rgb.transpose(2,0,1).copy(),dtype=torch.float32)[None]/255);return z[0].clone()
for k,step in enumerate(steps):
    model.load_state_dict(torch.load(prior.SOURCE/f'wagi/k{step:04d}.pt',map_location='cpu',weights_only=False)['net'])
    reference=latent('gray127');current_means=[];current_stds=[]
    for name in names:
        z=latent(name);d=z-reference;current_means.append(z.mean(0).numpy());current_stds.append(z.std(0).numpy())
        rms=float(d.square().mean().sqrt());energy=d.square().sum()
        for label,q in zip(archive['labels'],archive['bases'][k]):
            fraction=float((d@torch.tensor(q)).square().sum()/energy.clamp(min=1e-12))
            rows.append(dict(step=int(step),control=name,basis=str(label),response_rms=rms,capture=fraction))
        ranked=torch.argsort(d.square().mean(0),descending=True)[:20]
        for rank,j in enumerate(ranked):channels.append(dict(step=int(step),control=name,rank=rank+1,channel=int(j),change_rms=float(d[:,j].square().mean().sqrt()),signed_mean_change=float(d[:,j].mean())))
    means.append(current_means);stds.append(current_stds)
for filename,data in [('CONTROL_LATENTS.csv',rows),('CONTROL_CHANNELS.csv',channels)]:
    with (OUT/filename).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
np.savez_compressed(OUT/'CONTROL_FEATURES.npz',steps=steps,names=names,channel_mean=np.array(means),channel_std_over_tokens=np.array(stds))
assert all(np.isfinite(float(r['response_rms'])) for r in rows)
with (OUT/'REPORT.md').open('a',encoding='utf8') as f:f.write('\n\n## Latenty jednolitych obrazów\n\nCONTROL_LATENTS.csv zapisuje RMS zmiany wszystkich 600×768 współrzędnych względem szarości 127 oraz rzut na sześć sond. CONTROL_CHANNELS.csv zawiera 20 kanałów o największej zmianie dla każdego obrazu/checkpointu; są to kanały reakcji na bodziec, nie udowodnione neurony koloru. CONTROL_FEATURES.npz przechowuje średnią i odchylenie po tokenach dla wszystkich 768 kanałów. Czerń, biel i podstawowe RGB różnią się także jasnością, więc te odpowiedzi nie izolują chromatyczności tak jak maskowane perturbacje Lab.\n')
print('Solid-color latent measurements complete; 55 stimuli, 330 probe projections, all finite.')
