"""One local SD sample from Gaussian noise, guided by full artificial ViT target."""
import os,sys,time,json,hashlib,traceback
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,torch,cv2
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'outputs'/HERE.name
sys.path.insert(0,str(ROOT/'experiments/vit_latent768_sd_od_szumu'))
from generator_sd import GeneratorSD,SKALA_VAE
from model_snapshot import MalyViT,dekoduj
TARGET=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz'
CKPT=ROOT/'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
SD=Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele');START=time.time()
CFG=dict(seed=202610061923,steps=64,first_guided=20,last_unguided=3,trust_rms=.035,rates=[.004,.002,.001],lambda640=.2,prior_weight=.02,source_image_used=False,start='pure standard Gaussian noise',scorer='original frozen ViT k1000 matching artificial target and translator',guidance='full600x768 target, separately normalized128/640 errors; bounded late predicted-clean correction',save_latents=False)
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for i in range(30):
        try:os.replace(t,p);break
        except PermissionError:
            if i==29:raise
            time.sleep(.1)
def progress(stage,**kw):
    o=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.time()-START,**kw);write('PROGRESS.json',o);print(json.dumps(o),flush=True)
def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'RESULT.json').exists()
    fd=os.open(OUT/'RUN.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:run()
    except Exception:write('FAILED.json',dict(traceback=traceback.format_exc()));progress('failed');raise
    finally:(OUT/'RUN.lock').unlink(missing_ok=True)
def run():
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    a=np.load(TARGET);target=a['after'].astype(np.float32);J=a['channels0'].astype(int);K=np.setdiff1d(np.arange(768),J)
    assert target.shape==(600,768) and len(J)==128
    energy128=float(np.mean(target[:,J]**2));energy640=float(np.mean(target[:,K]**2))
    net=MalyViT().cuda().eval().requires_grad_(False);net.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True)['net']);net.zapisuj_latent(True)
    progress('loading_local_SD')
    g=GeneratorSD(dev='cpu',unet=SD/'ema_020000',vae=SD/'sd-vae-ft-mse',tekst=SD/'pusty_tekst.pt')
    g.unet.requires_grad_(False);g.vae.requires_grad_(False);g.unet.set_attention_slice('auto');g.vae.enable_gradient_checkpointing();g.vae.enable_slicing();g.emb=g.emb.cuda()
    provenance=dict(config=CFG,target_sha256=sha(TARGET),vit_checkpoint_sha256=sha(CKPT),SD_sources=g.zrodlo,UNet_sha256=sha(SD/'ema_020000/diffusion_pytorch_model.safetensors'),VAE_sha256=sha(SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'),code_sha256=sha(HERE/'run.py'))
    write('CONFIG.json',provenance)
    rng=torch.Generator(device='cuda').manual_seed(CFG['seed']);x=torch.randn((1,4,48,72),generator=rng,device='cuda',dtype=torch.float16)
    noise_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
    masks=torch.zeros((1,3,48,72),device='cuda',dtype=torch.float16)
    tt=torch.from_numpy(target).cuda()[None];jt=torch.tensor(J,device='cuda');kt=torch.tensor(K,device='cuda')
    g.sched.set_timesteps(CFG['steps']);ac=g.sched.alphas_cumprod.cuda();history=[]
    def decode(z):return (g.vae.decode(z.half()/SKALA_VAE).sample.float().clamp(-1,1)+1)/2
    def objective(z,anchor):
        im=decode(z);_,latent=net(F.interpolate(im,size=(320,480),mode='bilinear',align_corners=False,antialias=True))
        err=(latent-tt).square();e128=err[:,:,jt].mean();e640=err[:,:,kt].mean()
        loss=e128/energy128+CFG['lambda640']*e640/energy640+CFG['prior_weight']*(z-anchor).square().mean()/CFG['trust_rms']**2
        return loss,e128,e640
    def output_image(name):
        g.unet.to('cpu');torch.cuda.empty_cache();g.vae.to('cuda')
        with torch.no_grad():rgb=(decode(x)[0].permute(1,2,0)*255).round().byte().cpu().numpy()
        assert cv2.imwrite(str(OUT/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    for i,t in enumerate(g.sched.timesteps):
        g.vae.to('cpu');torch.cuda.empty_cache();g.unet.to('cuda')
        with torch.no_grad():
            eps=g.unet(torch.cat([x,masks],1),t,encoder_hidden_states=g.emb).sample.float()
            alpha=ac[t].float();clean=(x.float()-(1-alpha).sqrt()*eps)/alpha.sqrt()
        guided=CFG['first_guided']<=i<CFG['steps']-CFG['last_unguided']
        if guided:
            g.unet.to('cpu');torch.cuda.empty_cache();g.vae.to('cuda');anchor=clean.detach().clone()
            var=clean.detach().requires_grad_(True);loss,e128,e640=objective(var,anchor)
            grad,=torch.autograd.grad(loss,var);assert torch.isfinite(grad).all()
            direction=grad/grad.square().mean().sqrt().clamp_min(1e-9);accepted=False;used_rate=0
            for rate in CFG['rates']:
                proposal=var.detach()-rate*direction;delta=proposal-anchor
                proposal=(anchor+delta*(CFG['trust_rms']/delta.square().mean().sqrt().clamp_min(CFG['trust_rms']))).detach()
                with torch.no_grad():new_loss,new128,new640=objective(proposal,anchor)
                if new_loss<loss.detach() and new128<=e128.detach():clean=proposal;accepted=True;used_rate=rate;break
            history.append(dict(step=i+1,MSE128=float(e128.detach()),MSE640=float(e640.detach()),accepted=accepted,rate=used_rate));write('GUIDANCE_HISTORY.json',history)
            eps=(x.float()-alpha.sqrt()*clean)/(1-alpha).sqrt()
            net._latent=None;del var,loss,e128,e640,grad,direction,proposal,new_loss,new128,new640,anchor
        x=g.sched.step(eps,t,x.float()).prev_sample.half();assert torch.isfinite(x).all()
        progress('generating',step=i+1,total=CFG['steps'],guided=guided)
        if i+1 in [32,48]:output_image(f'PODGLAD_{i+1}.png')
    output_image('OBRAZ_SD.png')
    im=cv2.cvtColor(cv2.imread(str(OUT/'OBRAZ_SD.png')),cv2.COLOR_BGR2RGB)
    inp=torch.from_numpy(cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA).copy()).permute(2,0,1)[None].cuda().float()/255
    net.zapisuj_latent(False)
    with torch.no_grad():plain,_=net(inp)
    net.zapisuj_latent(True)
    with torch.no_grad():out,z=net(inp)
    assert torch.equal(out,plain);real=z[0].cpu().numpy();err=(real-target)**2
    m128=float(err[:,J].mean());v=real[:,J].ravel().astype(np.float64);q=target[:,J].ravel().astype(np.float64)
    boxes,scores=dekoduj(out.cpu())
    metrics=dict(MSE128=m128,MSE640=float(err[:,K].mean()),MSE768=float(err.mean()),cosine128=float(v@q/max(np.linalg.norm(v)*np.linalg.norm(q),1e-12)),value_agreement128=100*max(0,1-np.sqrt(m128/energy128)),confidence=float(scores[0,0]),bbox=boxes[0,0].tolist())
    assert sha(TARGET)==provenance['target_sha256'] and sha(CKPT)==provenance['vit_checkpoint_sha256']
    result=dict(status='complete',image=str(OUT/'OBRAZ_SD.png'),image_sha256=sha(OUT/'OBRAZ_SD.png'),metrics=metrics,provenance=provenance,noise_sha256=noise_sha,source_image_used=False,natural_forward_without_activation_replacement=True,no_saved_latent_arrays=True,seconds=time.time()-START)
    write('RESULT.json',result);progress('complete',metrics=metrics)
if __name__=='__main__':main()
