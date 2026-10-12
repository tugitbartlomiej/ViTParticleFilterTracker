"""Jak bliski jest naturalny latent nowego SD do oryginalnego J128?
S128<90 obala sukces w jednej próbie. Start wyłącznie Gaussian noise,
nowy checkpoint niezależnie weryfikowany i zamrożony, bez podmiany aktywacji.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,importlib.util,json,shutil,time,sys
from pathlib import Path
import cv2,numpy as np,torch,timm
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
OWN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/vit_sd_szum_nowy_checkpoint'
TRAIN=ROOT/'outputs/vit_wyuczenie_latentu_jedno_zdjecie_do85'
TARGET=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
SD=Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele')
EXPECTED='2e00434405969a0742b30f15e16cb6156f99e0a6aae2ee9e0f93ce90249a66df'
CONFIG=dict(seed=202610057719,steps=80,first_guided=24,last_unguided=3,
            updates_per_step=2,trust_rms=.05,lambda640=.1,prior_weight=.01,
            rates=[.018,.006,.002],start='independent_standard_gaussian',source_image_conditioning=False)

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(name,data):
    p=OUT/name;tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def progress(status,**kw):
    free=total=None
    if torch.cuda.is_initialized():free,total=torch.cuda.mem_get_info()
    data=dict(status=status,pid=os.getpid(),time=time.time(),gpu_free_bytes=free,gpu_total_bytes=total,**kw)
    js('PROGRESS.json',data);print(json.dumps(data),flush=True)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def weights_sha(net):
    h=hashlib.sha256()
    for n,v in net.state_dict().items():h.update(n.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RESULT.json').exists():raise RuntimeError('Completed result exists; no overwrite')
    start=time.perf_counter();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ready_path=TRAIN/'READY_FOR_SD.json'
    while not ready_path.exists():progress('waiting_checkpoint',dependency=str(ready_path));time.sleep(10)
    ready=json.loads(ready_path.read_text(encoding='utf-8-sig'))
    js('READY_CONSUMED.json',ready)
    checkpoint=TRAIN/'MODEL_best.pt'
    # Finalization metadata must attest this exact file; never select a live checkpoint.
    hashes=[]
    def find_hashes(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if isinstance(v,str) and ('sha256' in k.lower() or 'sha'==k.lower()) and len(v)==64:hashes.append(v)
                find_hashes(v)
        elif isinstance(obj,list):
            for v in obj:find_hashes(v)
    find_hashes(ready)
    assert checkpoint.exists() and sha(checkpoint) in hashes,'READY does not attest MODEL_best.pt hash'
    checkpoint_sha=sha(checkpoint)
    # READY is a finalization contract; reproduce fresh reload ourselves below.
    progress('fresh_checkpoint_reload',checkpoint_sha256=checkpoint_sha)
    model_path=ROOT/'experiments/vit_wyuczenie_latentu_jedno_zdjecie_do85/model_snapshot.py'
    mdl=module('own_new_vit',model_path)
    net=mdl.MalyViT().cpu().eval().requires_grad_(False)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net.load_state_dict(saved['net'],strict=True)
    assert sha(checkpoint)==checkpoint_sha
    assert sha(TARGET/'LATENT_PELNY.npz')==EXPECTED
    with np.load(TARGET/'LATENT_PELNY.npz') as f:target=f['after'].astype(np.float64);J=f['channels0'].copy()
    assert target.shape==(600,768) and J.shape==(128,) and np.isfinite(target).all()
    K=np.setdiff1d(np.arange(768),J);assert len(K)==640
    energies=[float(np.mean(target[:,idx]**2)) for idx in (J,K)]
    assert min(energies)>0
    def metrics(z,output=None):
        z=np.asarray(z,dtype=np.float64);assert z.shape==target.shape and np.isfinite(z).all()
        d=(z-target)**2;m128=float(d[:,J].mean());m640=float(d[:,K].mean());m768=float(d.mean())
        a=z[:,J].ravel();b=target[:,J].ravel()
        r=dict(mse128=m128,mse640=m640,mse768=m768,
            score128=100*max(0,1-np.sqrt(m128/energies[0])),
            score768=100*max(0,1-np.sqrt(m768/np.mean(target**2))),
            cosine128=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
            norm_ratio128=float(np.linalg.norm(a)/np.linalg.norm(b)),success90=bool(m128<=.01*energies[0]))
        if output is not None:
            with torch.no_grad():boxes,scores=mdl.dekoduj(output.cpu())
            r.update(detector_confidence=float(scores[0,0]),detector_bbox=boxes[0,0].tolist())
        return r
    # Independent plain source forward. Hook only reads, returns None; compare detector bitwise.
    raw=cv2.imread(str(TARGET/'WEJSCIE_VIT.png'));assert raw is not None and raw.shape==(320,480,3)
    inp=torch.from_numpy(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)[None].float()/255
    with torch.no_grad():plain,_=net(inp)
    net.zapisuj_latent(True)
    with torch.no_grad():output,z=net(inp)
    assert torch.equal(plain,output)
    natural=z[0].numpy().copy();reference=TRAIN/'LATENT_NATURAL_best.npz'
    assert reference.exists(),'Missing finalized best natural latent'
    with np.load(reference) as f:
        np.testing.assert_allclose(natural,f['natural'],rtol=2e-5,atol=2e-5)
        assert np.array_equal(f['target'],target)
        np.testing.assert_allclose(output.numpy(),f['detector_map'],rtol=2e-5,atol=2e-5)
    source_metrics=metrics(natural,output)
    js('FRESH_RELOAD.json',dict(checkpoint_sha256=checkpoint_sha,source_metrics=source_metrics,
        source_png_sha256=sha(TARGET/'WEJSCIE_VIT.png'),reference_latent_sha256=sha(reference),
        plain_detector_bitwise_unchanged=True,natural_latent_matches_export=True,verified=True))
    del inp,raw,plain,output,z,natural,saved
    net._latent=None;weight_hash=weights_sha(net)
    net.to('cuda')
    gpath=ROOT/'experiments/vit_sd_cztery_szumy/generator_sd.py'
    gmod=module('own_local_sd',gpath)
    progress('loading_local_sd')
    G=gmod.GeneratorSD(dev='cpu',unet=SD/'ema_020000',vae=SD/'sd-vae-ft-mse',tekst=SD/'pusty_tekst.pt')
    G.unet.requires_grad_(False);G.vae.requires_grad_(False)
    G.unet.set_attention_slice('auto');G.vae.enable_gradient_checkpointing();G.vae.enable_slicing();G.emb=G.emb.cuda()
    provenance=dict(config=CONFIG,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        checkpoint_step=int(torch.load(checkpoint,map_location='cpu',weights_only=False)['step']),
        ready_sha256=sha(ready_path),target_sha256=EXPECTED,target_shape=[600,768],channels0=J.tolist(),
        target_energy128=energies[0],target_energy640=energies[1],source_metrics=source_metrics,
        python=sys.version,torch=str(torch.__version__),timm=timm.__version__,device=torch.cuda.get_device_name(),
        sd_sources=G.zrodlo,sd_unet_sha256=sha(SD/'ema_020000/diffusion_pytorch_model.safetensors'),
        sd_vae_sha256=sha(SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'),text_sha256=sha(SD/'pusty_tekst.pt'),
        code_sha256={p.name:sha(p) for p in OWN.iterdir() if p.is_file()},
        model_source_sha256=sha(model_path),generator_source_sha256=sha(gpath),
        git_dirty=True,git_dirty_note='Scoped status only; conservative dirty stamp avoids broad dataset scan.',pid=os.getpid())
    js('PROVENANCE.json',provenance)
    rng=torch.Generator(device='cuda').manual_seed(CONFIG['seed'])
    x=torch.randn((1,4,48,72),generator=rng,device='cuda',dtype=torch.float16)
    np.savez_compressed(OUT/'SZUM_STARTOWY.npz',noise=x.cpu().numpy(),seed=CONFIG['seed'],distribution='standard_gaussian',source_image_used=False)
    masks=torch.zeros((1,3,48,72),device='cuda',dtype=torch.float16)
    tt=torch.from_numpy(target.astype(np.float32)).cuda()[None];jt=torch.tensor(J,device='cuda');kt=torch.tensor(K,device='cuda')
    G.sched.set_timesteps(CONFIG['steps']);ac=G.sched.alphas_cumprod.cuda()
    def decode(clean):return (G.vae.decode(clean.half()/gmod.SKALA_VAE).sample.float().clamp(-1,1)+1)/2
    def energy(clean,anchor):
        image=decode(clean);_,z=net(F.interpolate(image,size=(320,480),mode='bilinear',align_corners=False,antialias=True))
        e=(z-tt).square();m128=e[:,:,jt].mean();m640=e[:,:,kt].mean()
        value=m128/energies[0]+CONFIG['lambda640']*m640/energies[1]
        value+=CONFIG['prior_weight']*(clean-anchor).square().mean()/CONFIG['trust_rms']**2
        return value,m128,m640
    def save_image(latent,name):
        G.unet.to('cpu');torch.cuda.empty_cache();G.vae.to('cuda')
        with torch.no_grad():rgb=(decode(latent)[0].permute(1,2,0)*255).round().byte().cpu().numpy()
        assert cv2.imwrite(str(OUT/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    history=[]
    for i,t in enumerate(G.sched.timesteps):
        G.vae.to('cpu');torch.cuda.empty_cache();G.unet.to('cuda')
        with torch.no_grad():
            eps=G.unet(torch.cat([x,masks],1),t,encoder_hidden_states=G.emb).sample.float()
            a=ac[t].float();clean=(x.float()-(1-a).sqrt()*eps)/a.sqrt()
        if CONFIG['first_guided']<=i<CONFIG['steps']-CONFIG['last_unguided']:
            G.unet.to('cpu');torch.cuda.empty_cache();G.vae.to('cuda');anchor=clean.detach().clone()
            for iteration in range(CONFIG['updates_per_step']):
                variable=clean.detach().requires_grad_(True);loss,m128,m640=energy(variable,anchor)
                grad,=torch.autograd.grad(loss,variable);assert torch.isfinite(grad).all() and torch.isfinite(loss)
                direction=grad/grad.square().mean().sqrt().clamp_min(1e-9);accepted=False;used_rate=0.
                for rate in CONFIG['rates']:
                    candidate=variable.detach()-rate*direction;delta=candidate-anchor
                    candidate=(anchor+delta*(CONFIG['trust_rms']/delta.square().mean().sqrt().clamp_min(CONFIG['trust_rms']))).detach()
                    with torch.no_grad():new_loss,new128,new640=energy(candidate,anchor)
                    if new_loss<loss.detach() and new128<=m128.detach():clean=candidate;accepted=True;used_rate=rate;break
                history.append(dict(step=i+1,iteration=iteration,objective=float(loss.detach()),mse128=float(m128.detach()),
                    mse640=float(m640.detach()),accepted=accepted,rate=used_rate,trust_delta_rms=float((clean-anchor).square().mean().sqrt())))
                net._latent=None
                del variable,loss,m128,m640,grad,direction,candidate,new_loss,new128,new640
            eps=(x.float()-a.sqrt()*clean)/(1-a).sqrt()
            js('GUIDANCE_HISTORY.json',history)
        x=G.sched.step(eps,t,x.float()).prev_sample.half();assert torch.isfinite(x).all()
        progress('generating',step=i+1,total=CONFIG['steps'],guided=CONFIG['first_guided']<=i<77,gradient_updates=len(history))
        if i+1 in (24,40,60):save_image(x,f'PODGLAD_step{i+1:03d}.png')
    save_image(x,'OBRAZ_SD.png')
    np.savez_compressed(OUT/'LATENT_SD_VAE.npz',latent=x.cpu().numpy(),scale=gmod.SKALA_VAE)
    image_path=OUT/'OBRAZ_SD.png'
    raw=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    inp=torch.from_numpy(cv2.resize(raw,(480,320),interpolation=cv2.INTER_AREA).copy()).permute(2,0,1)[None].cuda().float()/255
    net.zapisuj_latent(False)
    with torch.no_grad():plain,_=net(inp)
    net.zapisuj_latent(True)
    with torch.no_grad():output,z=net(inp)
    assert torch.equal(plain,output)
    natural=z[0].cpu().numpy().copy();m=metrics(natural,output)
    np.savez_compressed(OUT/'LATENT_VIT_OBRAZU_SD.npz',natural=natural,target=target,channels0=J,detector_map=output.cpu().numpy(),grid_shape=[20,30])
    with np.load(OUT/'LATENT_VIT_OBRAZU_SD.npz') as f:assert np.array_equal(f['natural'],natural) and np.array_equal(f['target'],target)
    assert weights_sha(net)==weight_hash and sha(checkpoint)==checkpoint_sha and sha(TARGET/'LATENT_PELNY.npz')==EXPECTED
    result=dict(status='complete',metrics=m,source_baseline=source_metrics,provenance=provenance,
        seconds=time.perf_counter()-start,weights_vit_unchanged=True,target_unchanged=True,
        plain_detector_bitwise_unchanged=True,one_new_image=True,image=str(image_path),image_sha256=sha(image_path),
        noise_sha256=sha(OUT/'SZUM_STARTOWY.npz'),peak_cuda_allocated=torch.cuda.max_memory_allocated(),
        limitation='One generated sample. No GT accuracy or generalization claim. Cosine and norm ratio are distinct from coordinate agreement. Old checkpoint feasibility bound does not apply.')
    js('RESULT.json',result);js('BIEG_sd.json',result)
    text=f"# Jeden nowy obraz SD z czystego szumu\n\nS128={m['score128']:.6f} procent; MSE128={m['mse128']:.9f}; MSE640={m['mse640']:.9f}; MSE768={m['mse768']:.9f}. Cosine128={m['cosine128']:.9f}; norm ratio128={m['norm_ratio128']:.9f}. S768={m['score768']:.6f}.\n\nCheckpoint: {checkpoint_sha}. Oryginalny cel: {EXPECTED}. Seed={CONFIG['seed']}; 80 DDIM, 106 ograniczonych aktualizacji. Start wyłącznie szum, zerowe maski i pusty tekst. Brak img2img, treningu, podmiany aktywacji lub optymalizacji po DDIM.\n\nPewność detektora={m['detector_confidence']:.6f} jest drugorzędna, bez GT nie jest accuracy. Zdjęcie bazowe oceniono wyłącznie przy weryfikacji reloadu: S128={source_metrics['score128']:.6f}. Nie było wejściem SD. Podglądy to etapy tego samego biegu. Brak dodatkowego ramienia SD oznacza brak oszacowania przyczynowej korzyści prowadzenia. Jedna próba, bez dowodu generalizacji i bez gwarancji 90.\n"
    (OUT/'RAPORT.md').write_text(text,encoding='utf-8');(OWN/'WYNIKI_SD.md').write_text(text,encoding='utf-8')
    for p in OWN.iterdir():
        if p.is_file():shutil.copy2(p,OUT/('SOURCE_'+p.name))
    js('HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('HASHES.json','PROGRESS.json') and p.suffix!='.log'})
    progress('complete',metrics=m,seconds=result['seconds'])

if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);progress('failed',error=str(e));raise
