"""Noisy original image -> DDIM conditioned on exact full artificial ViT latent."""
import test_lokalny as h
import traceback, json, shutil
import torch, numpy as np, cv2
from PIL import Image
from diffusers import DDIMScheduler
from generate import load_trained
from model import load_vit

@torch.inference_mode()
def run():
    old = h.OUT
    h.OUT = h.ROOT/'outputs/vit_sd_img2img_stary_latent99_20261006'
    h.OUT.mkdir(parents=True, exist_ok=True)
    h.status('loading')
    torch.set_num_threads(4)
    directory = h.ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
    target_path = directory/'LATENT_PELNY.npz'
    assert h.sha(target_path)=='2e00434405969a0742b30f15e16cb6156f99e0a6aae2ee9e0f93ce90249a66df'
    a = np.load(target_path)
    after, before = a['after'].astype(np.float32), a['before'].astype(np.float32)
    j = a['channels0'].astype(int); k = np.setdiff1d(np.arange(768),j)
    source = np.asarray(Image.open(directory/'ZDJECIE_ZRODLOWE.jpg').convert('RGB')).astype(np.float32)/255
    sd = load_trained(h.SD/'ema_020000', h.SD/'sd-vae-ft-mse', old/'best_checkpoint.pt')
    sd.vae.enable_tiling()
    source_sd = cv2.resize(source,(576,384),interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(source_sd).permute(2,0,1)[None].float().cuda()
    clean = sd.vae.encode(x*2-1).latent_dist.mode()*float(sd.vae.config.scaling_factor)
    seed, steps = 2026100603, 100
    noise = torch.randn(clean.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
    report = dict(checkpoint_step=sd.step,target_sha256=h.sha(target_path),seed=seed,ddim_steps=steps,
                  condition='full 600x768 after, unchanged exact values',
                  start='VAE posterior mode of original image, plus Gaussian noise',
                  guidance='learned full-latent adapter used at every denoising step; no test-time latent-gradient correction',
                  save_latents=False, variants={})
    outputs = {}
    for label, strength, target in [('cel_020',.20,after),('cel_040',.40,after),('kontrola_naturalny_040',.40,before)]:
        h.status('generating',variant=label,strength=strength)
        scheduler=DDIMScheduler.from_config(sd.scheduler_config,clip_sample=False,set_alpha_to_one=False,steps_offset=1)
        scheduler.set_timesteps(steps,device='cuda')
        timesteps=scheduler.timesteps[steps-int(steps*strength):]
        sample=scheduler.add_noise(clean,noise,timesteps[:1])
        condition=torch.from_numpy(target)[None].cuda()
        for timestep in timesteps:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=sd.model(scheduler.scale_model_input(sample,timestep),timestep,condition)
            sample=scheduler.step(pred.float(),timestep,sample,eta=0.).prev_sample
        rgb=(sd.vae.decode(sample.float()/float(sd.vae.config.scaling_factor)).sample+1)/2
        assert torch.isfinite(rgb).all()
        image=rgb.clamp(0,1)[0].permute(1,2,0).cpu().numpy()
        path=h.OUT/(label+'.png')
        Image.fromarray(np.uint8(image*255)).save(path)
        outputs[label]=image
        report['variants'][label]=dict(strength=strength,denoising_steps=len(timesteps),image=path.name)
    del sd,x,clean,noise,sample,pred,rgb,condition
    torch.cuda.empty_cache()
    vit=load_vit(directory/'MODEL_VIT_K1000.pt',torch.device('cuda'))
    report['source_against_artificial_target']=h.inspect(vit,source,after,j,k)
    for label,image in outputs.items():
        report['variants'][label]['metrics_against_artificial_target']=h.inspect(vit,image,after,j,k)
        report['variants'][label]['image_MSE_to_original']=float(((image-source_sd)**2).mean())
    best=min(('cel_020','cel_040'),key=lambda label: report['variants'][label]['metrics_against_artificial_target']['MSE128'])
    shutil.copyfile(h.OUT/(best+'.png'),h.OUT/'OBRAZ_SD.png')
    report['best_by_MSE128']=best
    report['note']='Generated images lack their own ground-truth annotations; historical 99.93% IoU is not assigned to them.'
    h.write('RESULT.json',report)
    h.status('complete',best=best)

if __name__=='__main__':
    try: run()
    except Exception:
        h.status('failed',traceback=traceback.format_exc())
        raise
