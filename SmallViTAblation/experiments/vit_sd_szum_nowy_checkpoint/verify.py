"""Czy świeży checkpoint odtwarza naturalny latent zapisanego obrazu?
Niezgodny latent, zmieniony cel lub wpływ hooka obalają wynik generacji.
Nie uruchamia kolejnego SD.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,importlib.util,json
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_sd_szum_nowy_checkpoint'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    r=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'));p=r['provenance']
    ready=json.loads((OUT/'READY_CONSUMED.json').read_text(encoding='utf-8'))
    assert ready['fresh_reload_verified'] and ready['plain_forward_without_hooks_verified']
    assert sha(p['checkpoint'])==ready['checkpoint_sha256']==p['checkpoint_sha256']
    model_path=Path(ready['model_source_path']);assert sha(model_path)==p['model_source_sha256']
    spec=importlib.util.spec_from_file_location('verification_vit',model_path);mdl=importlib.util.module_from_spec(spec);spec.loader.exec_module(mdl)
    net=mdl.MalyViT().eval().requires_grad_(False)
    net.load_state_dict(torch.load(p['checkpoint'],map_location='cpu',weights_only=False)['net'],strict=True)
    image=OUT/'OBRAZ_SD.png';assert sha(image)==r['image_sha256']
    rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
    inp=torch.from_numpy(cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA).copy()).permute(2,0,1)[None].float()/255
    assert all(len(m._forward_hooks)==0 for m in net.modules())
    with torch.no_grad():plain,_=net(inp)
    cache={}
    def read(m,i,o):cache['z']=o.detach().cpu().numpy().copy()
    handle=net.bloki[-1].mlp.act.register_forward_hook(read)
    with torch.no_grad():output,_=net(inp)
    handle.remove();assert torch.equal(plain,output)
    z=cache['z'][0]
    with np.load(OUT/'LATENT_VIT_OBRAZU_SD.npz') as f:
        np.testing.assert_allclose(z,f['natural'],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(output.numpy(),f['detector_map'],rtol=2e-5,atol=2e-5)
        error=float(np.max(abs(z-f['natural'])));target=f['target'];J=f['channels0']
    source=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
    assert sha(source)==p['target_sha256']
    with np.load(source) as f:assert np.array_equal(f['after'],target) and np.array_equal(f['channels0'],J)
    assert z.shape==target.shape==(600,768)
    mse=float(((z.astype(float)[:,J]-target[:,J])**2).mean())
    score=100*max(0,1-np.sqrt(mse/np.mean(target[:,J]**2)))
    assert abs(mse-r['metrics']['mse128'])<2e-4
    with np.load(OUT/'SZUM_STARTOWY.npz') as f:
        noise=f['noise'];assert noise.shape==(1,4,48,72) and int(f['seed'])==p['config']['seed']
    assert not p['config']['source_image_conditioning'] and p['config']['start']=='independent_standard_gaussian'
    v=dict(verified=True,fresh_process_pid=os.getpid(),checkpoint_sha256=p['checkpoint_sha256'],
        plain_forward_no_hooks=True,reading_hook_detector_bitwise_unchanged=True,natural_shape=[600,768],
        max_abs_reload_latent_error=error,mse128_cpu=mse,score128_cpu=score,target_unchanged=True,
        source_conditioning=False,noise_mean=float(noise.astype(float).mean()),noise_std=float(noise.astype(float).std()))
    (OUT/'VALIDATION.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v))
if __name__=='__main__':main()
