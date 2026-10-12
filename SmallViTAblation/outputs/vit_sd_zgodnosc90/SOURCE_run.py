"""Czy nowe prowadzenie SD poprawia pełny MSE? Brak poprawy względem
kontroli obala korzyść próby. Certyfikat wyklucza próg 90 dla oryginalnego celu.
"""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OWN = Path(__file__).resolve().parent
OUT = ROOT/'outputs/vit_sd_zgodnosc90'
TARGET = ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
BOUNDS = ROOT/'outputs/vit_latent768_pelny_zbior'
SD = Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele')
SEED = 2026100591
EXPECTED = '2e00434405969a0742b30f15e16cb6156f99e0a6aae2ee9e0f93ce90249a66df'

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def js(name, value):
    p = OUT/name
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(p)

def progress(status, **kw):
    js('PROGRESS.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))
    print(status,json.dumps(kw),flush=True)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def weight_hash(net):
    h=hashlib.sha256()
    for name,v in net.state_dict().items():h.update(name.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def main(audit_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    progress('certifying')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert sha(TARGET/'LATENT_PELNY.npz')==EXPECTED
    with np.load(TARGET/'LATENT_PELNY.npz') as f:
        target=f['after'].astype(np.float64);before=f['before'].astype(np.float64);J=f['channels0'].copy()
    assert target.shape==before.shape==(600,768) and J.shape==(128,)
    assert len(np.unique(J))==128 and np.isfinite(target).all()
    other=np.ones(768,bool);other[J]=False
    state=torch.load(TARGET/'MODEL_VIT_K1000.pt',map_location='cpu',weights_only=True)['net']
    W=state['bloki.5.mlp.fc1.weight'].double().numpy();b=state['bloki.5.mlp.fc1.bias'].double().numpy()
    gamma=state['bloki.5.norm2.weight'].double().numpy();beta=state['bloki.5.norm2.bias'].double().numpy()
    assert W.shape==(768,192)
    coef=W*gamma[None]
    upper=np.maximum(0,W@beta+b+np.sqrt(192)*np.linalg.norm(coef-coef.mean(1,keepdims=True),axis=1))+.01
    with np.load(BOUNDS/'GRANICE_LATENTU.npz') as f:
        assert np.array_equal(f['target'],target)
        np.testing.assert_allclose(f['upper768'],upper,rtol=0,atol=1e-10)
    diagnostic=json.loads((BOUNDS/'DIAGNOSTYKA_OSIAGALNOSCI.json').read_text())
    assert diagnostic['target_sha256']==EXPECTED
    excess=np.maximum(target-upper[None],0);lb=float(np.mean(excess**2));ms=float(np.mean(target**2))
    np.testing.assert_allclose(lb,diagnostic['necessary_mse768_lower_bound'],rtol=1e-12)
    certificate=dict(target_sha256=EXPECTED,checkpoint_sha256=sha(TARGET/'MODEL_VIT_K1000.pt'),
        target_mean_square=ms,mse90_max=.01*ms,mse_lower_bound=lb,
        agreement_upper=100*max(0,1-np.sqrt(lb/ms)),threshold90_excluded=bool(lb>.01*ms),
        formula='S=100*max(0,1-sqrt(MSE768/mean(target^2)))',
        proof='LayerNorm q has sum(q)=0 and norm(q)<=sqrt(192). Center W*gamma; apply Cauchy-Schwarz. GELU(h)<=max(0,h). Upper bound includes 0.01 numeric safety. Sum squared positive excess over all 600x768 coordinates.',
        diagnostic_sha256=sha(BOUNDS/'DIAGNOSTYKA_OSIAGALNOSCI.json'),bounds_sha256=sha(BOUNDS/'GRANICE_LATENTU.npz'))
    js('CERTYFIKAT.json',certificate)
    assert certificate['threshold90_excluded'], 'Unexpected feasibility: review plan before generating'
    def metrics(z):
        z=np.asarray(z,dtype=np.float64);assert z.shape==target.shape and np.isfinite(z).all()
        d=(z-target)**2;mse=float(d.mean())
        return dict(agreement_score=100*max(0,1-np.sqrt(mse/ms)),mse768=mse,
            mse128=float(d[:,J].mean()),mse640=float(d[:,other].mean()),
            cosine768=float(np.sum(z*target)/(np.linalg.norm(z)*np.linalg.norm(target))),success90=bool(mse<=.01*ms))
    device='cpu' if audit_only else 'cuda'
    model_path=ROOT/'experiments/vit_sd_cztery_szumy/model_snapshot.py'
    net=module('own_vit',model_path).MalyViT().to(device).eval().requires_grad_(False)
    net.load_state_dict(state);net.zapisuj_latent(True);weights=weight_hash(net)
    def evaluate(p,dump=None):
        raw=cv2.imread(str(p));assert raw is not None,str(p)
        rgb=cv2.cvtColor(cv2.resize(raw,(480,320),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        inp=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].to(device).float()/255
        with torch.no_grad():_,z=net(inp)
        natural=z[0].cpu().numpy().copy();assert natural.shape==(600,768)
        assert (natural<=upper[None]+1e-5).all()
        r=dict(image=str(p),image_sha256=sha(p),**metrics(natural))
        if dump:np.savez_compressed(OUT/dump,natural=natural,target=target,channels0=J)
        net._latent=None
        return r
    families=['vit_sd_cztery_szumy','vit_latent768_sd_od_najblizszego','vit_latent768_sd_od_szumu',
        'vit_sd_obraz_najblizszy','vit_latent128_obraz_sd','vit_latent768_najblizsze_zdjecie']
    candidates=[TARGET/'ZDJECIE_ZRODLOWE.jpg',TARGET/'WEJSCIE_VIT.png',BOUNDS/'NAJBLIZSZE_ZDJECIE.jpg',BOUNDS/'NAJBLIZSZE_TRAIN.jpg']
    for fam in families:
        for p in sorted((ROOT/'outputs'/fam).glob('*')):
            if p.suffix.lower() in ('.png','.jpg') and p.name.startswith(('OBRAZ','NAJBLIZSZE','REKONSTRUKCJA')):candidates.append(p)
    records=[]
    for i,p in enumerate(candidates):
        if p.exists():records.append(evaluate(p))
        progress('comparing_saved',index=i+1,total=len(candidates))
    source=BOUNDS/'NAJBLIZSZE_ZDJECIE.jpg'
    best=min(records,key=lambda r:r['mse768'])
    best_sd=min([r for r in records if Path(r['image']).name.startswith('OBRAZ')],key=lambda r:r['mse768'])
    js('BASELINES.json',dict(original_before=metrics(before),actual_images=records,best=best,best_saved_sd=best_sd))
    assert weight_hash(net)==weights
    for label,r in [('BEST_AVAILABLE',best),('BEST_SAVED_SD',best_sd)]:
        assert cv2.imwrite(str(OUT/(label+'.png')),cv2.imread(r['image']))
        evaluate(OUT/(label+'.png'),label+'_LATENT.npz')
    provenance=dict(**certificate,pid=os.getpid(),seed=SEED,model_source_sha256=sha(model_path),
        own_source_sha256=sha(__file__),torch=torch.__version__,device=device,git_dirty=True,
        git_dirty_note='Conservative dirty stamp, no broad untracked scan; scoped status checked before edits.',
        source=str(source),source_sha256=sha(source),new_generation_requested=not audit_only)
    js('PROVENANCE.json',provenance)
    variants=[];history=[]
    if not audit_only:
        progress('loading_local_sd')
        generator_path=ROOT/'experiments/vit_sd_cztery_szumy/generator_sd.py'
        gmod=module('own_sd',generator_path)
        G=gmod.GeneratorSD(dev='cpu',unet=SD/'ema_020000',vae=SD/'sd-vae-ft-mse',tekst=SD/'pusty_tekst.pt')
        G.unet.requires_grad_(False);G.vae.requires_grad_(False)
        G.unet.set_attention_slice('auto');G.vae.enable_gradient_checkpointing();G.vae.enable_slicing()
        G.emb=G.emb.cuda();G.vae.to('cuda')
        target_t=torch.from_numpy(target.astype(np.float32)).cuda()[None]
        raw=cv2.cvtColor(cv2.imread(str(source)),cv2.COLOR_BGR2RGB)
        inp=torch.from_numpy(cv2.resize(raw,(576,384),interpolation=cv2.INTER_AREA)).permute(2,0,1)[None].cuda().float()/255
        with torch.no_grad():initial=G.vae.encode((inp*2-1).half()).latent_dist.mean*gmod.SKALA_VAE
        rng=torch.Generator(device='cuda').manual_seed(SEED)
        noise=torch.randn(initial.shape,device='cuda',dtype=torch.float16,generator=rng)
        np.savez_compressed(OUT/'START_VAE_I_SZUM.npz',initial=initial.cpu().numpy(),noise=noise.cpu().numpy(),seed=SEED)
        masks=torch.zeros((1,3,48,72),device='cuda',dtype=torch.float16)
        def decode(x):return (G.vae.decode(x.half()/gmod.SKALA_VAE).sample.float().clamp(-1,1)+1)/2
        def energy(x,anchor):
            image=decode(x)
            _,z=net(F.interpolate(image,size=(320,480),mode='bilinear',align_corners=False,antialias=True))
            raw_mse=(z-target_t).square().mean()
            return raw_mse+.01*(x-anchor).square().mean()/.05**2,raw_mse
        provenance.update(sd_sources=G.zrodlo,generator_source_sha256=sha(generator_path),
            sd_unet_sha256=sha(SD/'ema_020000/diffusion_pytorch_model.safetensors'),
            sd_vae_sha256=sha(SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'),
            text_sha256=sha(SD/'pusty_tekst.pt'),ddim_steps=40,strength=.30,executed_steps=12,trust_rms=.05)
        js('PROVENANCE.json',provenance)
        for guided in (False,True):
            tag='nowy_prowadzony' if guided else 'nowy_kontrola'
            G.sched.set_timesteps(40);ts=G.sched.timesteps;first=28;ac=G.sched.alphas_cumprod.cuda()
            a0=ac[ts[first]];x=(a0.sqrt()*initial.float()+(1-a0).sqrt()*noise.float()).half()
            for i in range(first,40):
                G.vae.to('cpu');torch.cuda.empty_cache();G.unet.to('cuda')
                t=ts[i];a=ac[t].float()
                with torch.no_grad():
                    eps=G.unet(torch.cat([x,masks],1),t,encoder_hidden_states=G.emb).sample.float()
                    clean=(x.float()-(1-a).sqrt()*eps)/a.sqrt()
                if guided and i<37:
                    G.unet.to('cpu');torch.cuda.empty_cache();G.vae.to('cuda')
                    anchor=clean.detach().clone();variable=clean.detach().requires_grad_(True)
                    loss,mse=energy(variable,anchor);grad,=torch.autograd.grad(loss,variable)
                    assert torch.isfinite(grad).all()
                    direction=grad/grad.square().mean().sqrt().clamp_min(1e-9)
                    accepted=False
                    for rate in (.018,.006,.002):
                        candidate=(variable.detach()-rate*direction).detach()
                        delta=candidate-anchor
                        candidate=anchor+delta*(.05/delta.square().mean().sqrt().clamp_min(.05))
                        with torch.no_grad():new_loss,new_mse=energy(candidate,anchor)
                        if new_loss<loss.detach() and new_mse<=mse.detach():clean=candidate;accepted=True;break
                    history.append(dict(step=i+1,raw_mse=float(mse.detach()),accepted=accepted))
                    eps=(x.float()-a.sqrt()*clean)/(1-a).sqrt()
                    del variable,loss,grad
                    net._latent=None
                x=G.sched.step(eps,t,x.float()).prev_sample.half()
                progress('generating',variant=tag,step=i-first+1,total=12,variants_complete=len(variants))
            G.unet.to('cpu');torch.cuda.empty_cache();G.vae.to('cuda')
            with torch.no_grad():rgb=(decode(x)[0].permute(1,2,0)*255).round().byte().cpu().numpy()
            p=OUT/('OBRAZ_'+tag+'.png');assert cv2.imwrite(str(p),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            r=evaluate(p,'LATENT_'+tag+'.npz');r.update(guided=guided,seed=SEED,new_local_sd_generation=True)
            variants.append(r);js('VARIANTS.json',variants)
            progress('variant_complete',variant=tag,metrics=r)
        newbest=min(variants,key=lambda r:r['mse768'])
        shutil.copy2(newbest['image'],OUT/'OBRAZ_SD_NOWY.png')
        evaluate(OUT/'OBRAZ_SD_NOWY.png','LATENT_VIT_OBRAZU_SD_NOWY.npz')
        best=min(records+variants,key=lambda r:r['mse768'])
        assert cv2.imwrite(str(OUT/'BEST_AVAILABLE.png'),cv2.imread(best['image']))
        evaluate(OUT/'BEST_AVAILABLE.png','BEST_AVAILABLE_LATENT.npz')
    assert weight_hash(net)==weights and sha(TARGET/'LATENT_PELNY.npz')==EXPECTED
    result=dict(status='complete',certificate=certificate,baselines=records,best_available=best,
        variants=variants,history=history,new_images_generated=len(variants),success90=False,
        weights_vit_unchanged=True,target_unchanged=True,threshold90_search_stopped=True,
        best_effort_generation_executed=bool(variants),pid=os.getpid(),
        directional_cosine_is_distinct=True,limitation='Best among evaluated images, not a global optimum. Cosine is directional similarity, not coordinate agreement. No guarantee of 90.')
    js('RESULT.json',result)
    text=f"# Wynik\n\nPróg 90 wykluczony. Górna granica S: {certificate['agreement_upper']:.9f} procent. MSE celu: {ms:.9f}; próg MSE: {.01*ms:.9f}; konieczna dolna granica: {lb:.9f}.\n\nNajlepszy dostępny: S={best['agreement_score']:.6f}, MSE768={best['mse768']:.9f}, MSE128={best['mse128']:.9f}, MSE640={best['mse640']:.9f}, cosine={best['cosine768']:.9f}. Źródło: {best['image']}.\n\nWygenerowano {len(variants)} nowych obrazów SD. Pełne wyniki kontroli i prowadzenia w RESULT.json. Cel i wagi zachowane. Certyfikat nie dowodzi osiągalności maksimum. Dopasowanie do jednego celu nie jest oceną generalizacji. Cosine można badać oddzielnie; nie stanowi procentowej zgodności współrzędnych.\n"
    (OUT/'RAPORT.md').write_text(text,encoding='utf-8');(OWN/'WYNIKI_SD.md').write_text(text,encoding='utf-8')
    for p in OWN.iterdir():
        if p.is_file():shutil.copy2(p,OUT/('SOURCE_'+p.name))
    js('HASHES.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix not in ('.log',) and p.name not in ('HASHES.json','PROGRESS.json')})
    progress('complete',new_images_generated=len(variants),best=best)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    try:main(args.audit_only)
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);progress('failed',error=str(e));raise
