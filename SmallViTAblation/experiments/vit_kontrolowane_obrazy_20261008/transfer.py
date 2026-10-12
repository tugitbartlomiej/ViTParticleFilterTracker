"""Check controlled-image channel probes on held-out natural frames at final checkpoint."""
from pathlib import Path
import importlib.util,csv,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/vit_kontrolowane_obrazy_20261008/pilot_v1'
spec=importlib.util.spec_from_file_location('old',ROOT/'experiments/vit_semantyka_ruchu_20261008/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
module=old.load_model_module();model=module.MalyViT().eval();model.load_state_dict(torch.load(old.SOURCE/'wagi/k1000.pt',map_location='cpu',weights_only=False)['net'])
capture={};h=model.bloki[-1].mlp.act.register_forward_hook(lambda m,i,o:capture.update(z=o.detach().clone()))
@torch.inference_mode()
def latent(im):
    model(torch.tensor(im.transpose(2,0,1).copy(),dtype=torch.float32)[None]/255);return capture['z'][0]
images=np.load(old.SOURCE/'dane/obrazy.npy');records=json.loads((old.SOURCE/'dane/images.json').read_text(encoding='utf8'))
archive=np.load(OUT/'BASES.npz');qs=archive['bases'][-1];labels=archive['labels'];rows=[]
for i in range(8,12):
    mask,box=old.mask_box(records[i]);base=latent(images[i]);v=old.variants(images[i],mask,box)
    # Position/geometry probes here include whole-frame effects, explicitly recorded; color remains tool-local.
    for factor,sl in zip(['pozycja','kolor','proporcje'],[slice(0,4),slice(4,8),slice(8,10)]):
        changes=torch.cat([latent(im)-base for im in v[sl]])
        for label,q in zip(labels,qs):
            method,basis=str(label).split(':');qt=torch.tensor(q)
            frac=float((changes@qt).square().sum()/changes.square().sum().clamp(min=1e-12))
            rows.append(dict(image=i,method=method,basis=basis,perturbation=factor,capture=frac,scope='tool_mask' if factor=='kolor' else 'whole_frame'))
    print(f'natural transfer frame {i}',flush=True)
h.remove()
with (OUT/'NATURAL_TRANSFER.csv').open('w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
