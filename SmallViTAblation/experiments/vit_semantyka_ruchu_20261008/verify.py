"""Independent checks of exported probe geometry, input controls and readout identity."""
from pathlib import Path
import csv,json,importlib.util
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('experiment',Path(__file__).with_name('run.py'))
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
out=ROOT/'outputs/vit_semantyka_ruchu_20261008/pilot_v1'
state=np.load(out/'STATES_BASES.npz')
bases=state['bases'];steps=state['steps'];z=state['latents'].astype(np.float32)
orth=max(float(np.max(np.abs(q.T@q-np.eye(8)))) for group in bases for q in group)
assert orth<1e-4 and np.isfinite(z).all()
with (out/'MOTION.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==20*12*3
assert len({(r['step'],r['image'],r['factor']) for r in rows})==len(rows)
errors=[]
for r in rows:
    k=list(steps).index(int(r['step']));i=int(r['image']);f=exp.FACTORS.index(r['factor'])
    assert int(r['previous_step'])==int(steps[k-1])
    d=(z[k,i]-z[k-1,i])[state['selected'][i]]
    calculated=np.square(d@bases[k-1,f]).sum()/np.square(d).sum()
    errors.append(abs(float(r['motion_fraction'])-float(calculated)))
assert max(errors)<.001 # archive is float16; computation was full float32
records=json.loads((exp.SOURCE/'dane/images.json').read_text(encoding='utf-8'))
images=np.load(exp.SOURCE/'dane/obrazy.npy')
mask,box=exp.mask_box(records[8]);v=exp.variants(images[8],mask,box)
assert all(np.array_equal(im[mask==0],images[8][mask==0]) for im in v[4:8])
assert np.array_equal(v[0][:,16:],images[8][:,:-16])
assert np.array_equal(v[1][:,:-16],images[8][:,16:])
# Check saved-image/checkpoint forward and zero intervention with actual model, avoiding source trajectory quantization.
torch.set_num_threads(4);module=exp.load_model_module();model=module.MalyViT().cuda().eval()
model.load_state_dict(torch.load(exp.SOURCE/'wagi/k1000.pt',map_location='cpu',weights_only=False)['net'])
x=torch.tensor(images[8:9].transpose(0,3,1,2).copy(),device='cuda',dtype=torch.float32)/255
captured={}
handle=model.bloki[-1].mlp.act.register_forward_hook(lambda m,i,o:captured.update(z=o.clone()))
with torch.inference_mode():
    original,_=model(x);activation=captured['z'];handle.remove()
    replacement=model.bloki[-1].mlp.act.register_forward_hook(lambda m,i,o:activation)
    identity,_=model(x);replacement.remove()
assert torch.equal(original,identity)
export_error=float(np.sqrt(np.mean((activation.cpu().numpy()[0]-z[-1,8])**2)))
result=dict(status='PASS',motion_rows=len(rows),orthonormal_max_error=orth,motion_fraction_archive_max_error=max(errors),color_outside_mask_unchanged=True,translation_pixel_identity=True,zero_intervention_output_equal=True,export_rms_error=export_error)
(out/'QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
