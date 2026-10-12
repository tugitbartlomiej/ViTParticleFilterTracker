"""Fresh best-model inference with sequential CPU offload; media only in RAM."""
import gc, hashlib, io, json, os, sys, time, traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/vit_sd_k8000_ulepszenia_20261007'
COPY = OUT / 'eden_kopia'
sys.path.insert(0, str(COPY / 'kod'))
print('Loading numpy/torch', flush=True)
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
print('Loading model/diffusers/timm', flush=True)
from model import new_vae, Denoiser, teacher, VIT_SHA
print('Loading evaluation/video libraries', flush=True)
from evaluation import (fixed_samples, roi_mask, latent_metrics, aggregate_records,
                        _teacher, _detection, _image_metrics, _make_scheduler)

EXPECTED = '1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41'
SEEDS = (20261007, 20261008)
FILMS = ('TR_0127_S1_P03', 'TR_0129_S1_P03')
MEDIA = {}
START = time.time()

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def write(name, obj):
    p = OUT / name; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    for i in range(30):
        try: os.replace(tmp, p); return
        except PermissionError:
            if i == 29: raise
            time.sleep(.1)

def status(phase, **kw):
    obj = dict(phase=phase, pid=os.getpid(), elapsed=time.time()-START, **kw)
    write('TEST_LOKALNY_STATUS.json', obj); print(json.dumps(obj), flush=True)

def finite(x):
    if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite local inference')
    return x

def media(name, image):
    array = image[0].permute(1,2,0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(array,0,1)*255)).save(buf, format='JPEG', quality=92)
    MEDIA[name] = buf.getvalue()

def offload(module):
    module.cpu(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    status('verifying_checkpoint')
    path = COPY / 'wybrany_model/best_sd.pt'
    assert sha(path) == EXPECTED
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['signature']['teacher_sha256'] == VIT_SHA
    assert ck['signature']['arm'] == 'A_control' and not ck['signature']['smoke']
    assert ck['step'] == 10500 and ck['new_steps'] == 3000
    manifest = COPY / 'wejscia/FILMY.json'
    assert sha(manifest) == ck['signature']['manifest_sha256']
    net, vae = Denoiser().eval(), new_vae().eval()
    net.load_state_dict(ck['net'], strict=True); vae.load_state_dict(ck['vae'], strict=True)
    net.requires_grad_(False); vae.requires_grad_(False)
    scale = float(ck['scale'])
    assert np.isfinite(scale) and scale > 0
    result = dict(checkpoint_sha256=EXPECTED, checkpoint_step=ck['step'],
                  signature=ck['signature'], best_validation=ck['evaluation'],
                  fresh_strict_load=True, seeds=SEEDS, start='Gaussian noise only',
                  image_or_latent_saved=False, vae_scale=scale,
                  full_relative_mse_definition='mean((actual-target)^2)/mean(target^2)',
                  artificial_test=dict(state='not_run', reason='No verified artificial K8000 target. Old K1000/K6402 targets were not substituted.'),
                  iou_note='Generation IoU against source boxes only measures layout retention; it is not independently labelled accuracy.',
                  wrong_control_note='Same seed and sampler, condition swapped between the two films. Generated tensors are reused exactly for paired controls.')
    del ck; gc.collect()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result['device'] = str(device)
    result['code_sha256'] = {p.name:sha(p) for p in (COPY/'kod').glob('*.py')}
    result['test_code_sha256'] = sha(__file__)
    vit_path = ROOT / 'assets/MalyViT_K8000_s47_losowe/MALY_VIT_K8000_s47_losowe.pth'
    assert sha(vit_path) == VIT_SHA
    vit = teacher(vit_path, device)
    rows = json.loads(manifest.read_text(encoding='utf-8-sig'))
    local_rows = []
    data = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking'
    for film in FILMS:
        row = dict(next(r for r in rows if r['video_id']==film and r['split']=='val'))
        row['zip_path'] = str(data/'videos'/Path(row['zip_path']).name)
        row['annotation_zip'] = str(data/'annotations'/Path(row['annotation_zip']).name)
        local_rows.append(row)
    status('decoding_held_out_frames')
    samples, rejected = fixed_samples(local_rows, 'val', fractions=(.33,))
    assert not rejected and len(samples)==2, rejected
    targets, masks, refs = [], [], []
    for i, sample in enumerate(samples):
        output, z = _teacher(vit, sample['vit_image'][None].to(device))
        mask = roi_mask(sample['bbox'], sample['has_tool'], device='cpu')
        targets.append(z.cpu()); masks.append(mask)
        det = _detection(output, sample['bbox'], mask)
        refs.append(dict(film=sample['film'], frame=sample['frame'], split='val',
                         bbox_normalized=sample['bbox'].tolist(), has_tool=sample['has_tool'],
                         confidence=det['natural_detector_confidence'],
                         source_label_iou=det['reference_layout_iou_not_generated_ground_truth']))
        media(f'ref_{i}.jpg', sample['image'][None])
    offload(vit); del output,z
    status('vae_reconstruction')
    vae.to(device)
    recon, shapes = [], []
    for i, sample in enumerate(samples):
        h = finite(vae.encode(sample['image'][None].to(device)*2-1).latent_dist.mode())
        shapes.append(tuple(h.shape))
        im = ((finite(vae.decode(h).sample.float())+1)/2).clamp(0,1).cpu()
        recon.append(im); media(f'vae_{i}.jpg', im)
    del h,im; offload(vae)
    latents = {}
    net.to(device)
    amp_dtype = torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    result['amp_dtype'] = str(amp_dtype) if device.type=='cuda' else 'float32'
    for i, target in enumerate(targets):
        for seed in SEEDS:
            status('generating', film=samples[i]['film'], seed=seed, step=0)
            scheduler = _make_scheduler(50, 'leading', device)
            result['sampler_timesteps'] = scheduler.timesteps.cpu().tolist()
            condition = target.to(device)
            h = torch.randn(shapes[i], generator=torch.Generator(device=device).manual_seed(seed), device=device)*scheduler.init_noise_sigma
            for n,t in enumerate(scheduler.timesteps):
                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type=='cuda'):
                    pred = net(scheduler.scale_model_input(h,t),t,condition)
                h = finite(scheduler.step(pred.float(),t,h,eta=0.0).prev_sample)
                if (n+1)%10==0: status('generating',film=samples[i]['film'],seed=seed,step=n+1)
            latents[i,seed] = h.cpu()
            del h,pred,condition
    offload(net)
    status('decoding_generated_images')
    vae.to(device)
    images = {key:((finite(vae.decode(h.to(device)/scale).sample.float())+1)/2).clamp(0,1).cpu()
              for key,h in latents.items()}
    del latents; offload(vae)
    vit.to(device)
    records, actuals, outputs = [], {}, {}
    for key, im in images.items():
        outputs[key], z = _teacher(vit,F.interpolate(im.to(device),(320,480),mode='bilinear',align_corners=False))
        outputs[key] = outputs[key].cpu(); actuals[key] = z.cpu()
        media(f'gen_{key[0]}_{key[1]}.jpg', im)
    reconstruction_metrics = []
    for i, sample in enumerate(samples):
        _, z = _teacher(vit,F.interpolate(recon[i].to(device),(320,480),mode='bilinear',align_corners=False))
        rm = latent_metrics(z.cpu(),targets[i],masks[i]); rm.update(_image_metrics(recon[i],sample['image'][None]))
        reconstruction_metrics.append(rm)
        for seed in SEEDS:
            im, wrong = images[i,seed], images[1-i,seed]
            r = dict(film=sample['film'],frame=int(sample['frame']),seed=seed,
                     has_tool=bool(sample['has_tool']),roi_tokens=int(masks[i].sum()) if masks[i] is not None else 0,
                     wrong_condition_available=True,wrong_condition_film=samples[1-i]['film'],wrong_condition_frame=int(samples[1-i]['frame']))
            r.update(latent_metrics(actuals[i,seed],targets[i],masks[i]))
            r.update(_image_metrics(im,sample['image'][None]))
            r.update(_detection(outputs[i,seed],sample['bbox'],masks[i]))
            wrong_error = latent_metrics(actuals[1-i,seed],targets[i])['full_relative_mse']
            r.update(wrong_condition_mse=wrong_error,relative_gain_vs_wrong_condition=1-r['full_relative_mse']/max(wrong_error,1e-12),
                     condition_image_mse=(im-wrong).square().mean().item(),
                     clipped_fraction=((im<=0)|(im>=1)).float().mean().item())
            records.append(r)
    result.update(aggregate_records(records))
    result.update(references=refs,vae_reconstructions=reconstruction_metrics,all_values_finite=True,
                  natural_forward_without_activation_override=True,ddim_steps=50,spacing='leading')
    write('TEST_LOKALNY.json',result)
    del net,vae,vit,images,actuals,outputs,targets,recon,samples,z,im,wrong
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    sections = []
    for i, ref in enumerate(refs):
        names = [f'ref_{i}.jpg',f'vae_{i}.jpg']+[f'gen_{i}_{s}.jpg' for s in SEEDS]
        labels = ['Referencja VAL','Rekonstrukcja VAE']+[f'SD — seed {s}' for s in SEEDS]
        sections.append(f'<h2>{ref["film"]} · klatka {ref["frame"]}</h2><div class="grid">'+''.join(f'<figure><figcaption>{label}</figcaption><img src="/{name}"></figure>' for name,label in zip(names,labels))+'</div>')
    html = ('<meta charset="utf-8"><title>SD K8000 — test lokalny</title><style>body{font-family:Arial;background:#f5f4ee;color:#193239;margin:24px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%}figcaption{font-size:16px;margin:8px 0}p{max-width:1200px}</style><h1>A_control · świeże wczytanie najlepszego checkpointa</h1><p>Krok 10500 · 50 kroków DDIM · dwa odłożone filmy i dwa seedy. Start od losowego szumu, pełny latent K8000. Obrazy pozostają w RAM.</p>'+''.join(sections)+'<p>Kontrola: przy tym samym seedzie zamieniono latenty obu filmów. Ten mały test sprawdza odtwarzalność i wygląd; nie zastępuje walidacji na 25 filmach.</p>').encode('utf8')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            key = self.path.lstrip('/')
            if key in MEDIA: body,mime=MEDIA[key],'image/jpeg'
            elif key=='': body,mime=html,'text/html; charset=utf-8'
            else: self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
        def log_message(self,*args): pass
    server = HTTPServer(('127.0.0.1',0),Handler)
    status('natural_test_complete',preview_url=f'http://127.0.0.1:{server.server_port}/',checkpoint_step=10500,artificial_test='no compatible K8000 target')
    server.serve_forever()

if __name__=='__main__':
    try: main()
    except Exception:
        status('failed',error=traceback.format_exc()); raise
