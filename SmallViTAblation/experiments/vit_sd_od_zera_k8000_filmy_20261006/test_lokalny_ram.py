"""Fresh K8000-conditioned inference. Only metrics/code persist; media stay in RAM."""
import os, sys, json, io, time, hashlib, traceback
from pathlib import Path
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_sd_od_zera_k8000_filmy_20261006'
COPY=OUT/'eden_kopia'
sys.path.insert(0,str(COPY/'kod'))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler
from model import new_vae,Denoiser,teacher,features,schedule,VIT_SHA
from model_snapshot import dekoduj
from dane import iter_video
from http.server import HTTPServer,BaseHTTPRequestHandler

EXPECTED='6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7'
MEDIA={}
START=time.time()
def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    p=OUT/name; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    for attempt in range(30):
        try: os.replace(tmp,p); return
        except PermissionError:
            if attempt==29: raise
            time.sleep(.1)
def status(phase,**kw):
    v=dict(phase=phase,pid=os.getpid(),elapsed=time.time()-START,**kw)
    write('TEST_LOKALNY_STATUS.json',v); print(json.dumps(v),flush=True)
def image_bytes(name,x):
    array=x[0].permute(1,2,0).cpu().numpy()
    b=io.BytesIO(); Image.fromarray(np.uint8(np.clip(array,0,1)*255)).save(b,format='JPEG',quality=88)
    MEDIA[name]=b.getvalue()
def iou(a,b):
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    total=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/max(total,1e-12))
def get_sample(film,rows):
    row=dict(next(r for r in rows if r['video_id']==film and r['split']=='val'))
    data=ROOT.parent/'Dataset/Cataract-LMM/3_Object_Tracking'
    row['zip_path']=str(data/'videos'/Path(row['zip_path']).name)
    row['annotation_zip']=str(data/'annotations'/Path(row['annotation_zip']).name)
    frame=max(1,int(row['frame_count']*.33))
    samples=list(iter_video(row,selected_frames={frame}))
    assert len(samples)==1
    return samples[0]
def detection(vit,x,box=None):
    o,_=vit(x); vit._latent=None; boxes,scores=dekoduj(o)
    result=dict(confidence=float(scores[0,0]),bbox_xyxy=boxes[0,0].float().cpu().tolist())
    if box is not None: result['iou']=iou(np.array(result['bbox_xyxy']),np.array(box))
    return result

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    status('verifying_checkpoint')
    ckpath=COPY/'trening/best_sd.pt'
    assert sha(ckpath)==EXPECTED,'Checkpoint hash mismatch'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    assert ck['signature']['teacher_sha256']==VIT_SHA
    assert not ck['signature']['smoke']
    manifest=COPY/'wejscia/FILMY.json'
    assert sha(manifest)==ck['signature']['manifest_sha256']
    net=Denoiser().eval(); vae=new_vae().eval()
    net.load_state_dict(ck['net'],strict=True); vae.load_state_dict(ck['vae'],strict=True)
    scale=float(ck['scale'])
    result=dict(checkpoint_sha256=EXPECTED,checkpoint_step=ck['step'],
        signature=ck['signature'],best_validation=ck['evaluation']['film_mean'],
        fresh_strict_load=True,seed=20261006,ddim_steps=50,start='Gaussian noise only',
        image_or_latent_saved=False,vae_scale=scale,
        full_relative_mse_definition='mean((actual-target)^2)/mean(target^2)',
        artificial_test=dict(state='blocked',reason='No verified artificial K8000 target found. Existing targets use K1000 or K6402; neither was substituted.'))
    del ck
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result['device']=str(device)
    result['code_sha256']={p.name:sha(p) for p in (COPY/'kod').glob('*.py')}
    vit=teacher(ROOT/'assets/MalyViT_K8000_s47_losowe/MALY_VIT_K8000_s47_losowe.pth',device)
    status('decoding_held_out_frames')
    rows=json.loads(manifest.read_text(encoding='utf-8-sig'))
    sample=get_sample('TR_0127_S1_P03',rows); other=get_sample('TR_0129_S1_P03',rows)
    ref=sample['vit_image'][None].to(device)
    z=features(vit,ref).float().cpu()
    wrong_z=features(vit,other['vit_image'][None].to(device)).float().cpu()
    box=(sample['bbox']*torch.tensor([480,320,480,320])).tolist() if sample['has_tool'] else None
    result['reference']=dict(film=sample['film'],frame=sample['frame'],split='val',has_label=sample['has_tool'],
        detection=detection(vit,ref,box))
    result['wrong_condition_film']=other['film']
    image_bytes('reference.jpg',sample['image'][None])
    vit.cpu(); del ref; torch.cuda.empty_cache()
    latents={}
    for label,condition in [('natural',z),('wrong_condition',wrong_z)]:
        status('generating',condition=label)
        net.to(device)
        sched=DDIMScheduler.from_config(schedule().config); sched.set_timesteps(50,device=device)
        result['sampler_timesteps']=sched.timesteps.cpu().tolist()
        h=torch.randn((1,4,48,72),generator=torch.Generator(device=device).manual_seed(20261006),device=device)
        condition=condition.to(device)
        for idx,t in enumerate(sched.timesteps):
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                pred=net(h,t,condition)
            h=sched.step(pred.float(),t,h).prev_sample
            assert torch.isfinite(h).all(),'Nonfinite denoising'
            if (idx+1)%10==0: status('generating',condition=label,step=idx+1)
        latents[label]=h.cpu()
        del pred,h,condition
    net.cpu(); torch.cuda.empty_cache()
    vae.to(device)
    images={label:((vae.decode(h.to(device)/scale).sample+1)/2).clamp(0,1).cpu() for label,h in latents.items()}
    vae.cpu(); del latents; torch.cuda.empty_cache()
    vit.to(device)
    for label,im in images.items():
        assert torch.isfinite(im).all()
        x=F.interpolate(im.to(device),(320,480),mode='bilinear',align_corners=False)
        actual=features(vit,x).float().cpu()
        target=z
        err=(actual-target).square().mean().item()
        result[label]=dict(full_mse=err,full_relative_mse=err/target.square().mean().clamp_min(1e-6).item(),
            full_cosine=F.cosine_similarity(actual.flatten(1),target.flatten(1)).item(),
            spatial_std=im.flatten(2).std(dim=-1).mean().item(),
            clipped_fraction=float(((im<=0)|(im>=1)).float().mean()),
            detection=detection(vit,x))
        if label=='natural' and box is not None:
            result[label]['reference_layout_iou_not_generated_ground_truth']=iou(np.array(result[label]['detection']['bbox_xyxy']),np.array(box))
        image_bytes(label+'.jpg',im)
    result['condition_image_mse']=(images['natural']-images['wrong_condition']).square().mean().item()
    result['relative_gain_vs_wrong_condition']=1-result['natural']['full_relative_mse']/max(result['wrong_condition']['full_relative_mse'],1e-12)
    result['all_values_finite']=True
    result['iou_note']='Generated reference-layout IoU is a diagnostic, not independently labelled detection accuracy.'
    write('TEST_LOKALNY.json',result)
    del net,vae,vit,images,sample,other,z,wrong_z,x,actual
    torch.cuda.empty_cache()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            key=self.path.lstrip('/')
            if key in MEDIA: body,mime=MEDIA[key],'image/jpeg'
            else:
                captions=['Referencja VAL','SD: właściwy latent','SD: latent innego filmu']
                names=['reference.jpg','natural.jpg','wrong_condition.jpg']
                body=('<meta charset="utf-8"><title>SD K8000 — lokalny test</title><style>body{font-family:Arial;background:#f5f4ee;color:#193239;margin:24px}.grid{display:flex;gap:14px}figure{margin:0;width:33%}img{width:100%}figcaption{font-size:18px;margin:8px 0}p{max-width:1100px}</style><h1>SD z pełnego latentu K8000</h1><p>Checkpoint 7500 · 50 kroków DDIM · identyczny losowy szum · obrazy tylko w pamięci</p><div class="grid">'+''.join(f'<figure><figcaption>{c}</figcaption><img src="/{n}"></figure>' for c,n in zip(captions,names))+'</div><p>Wynik na jednej odłożonej klatce. Stary sztuczny cel K6402 nie jest zgodny z K8000.</p>').encode('utf8')
                mime='text/html; charset=utf-8'
            self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
        def log_message(self,*a): pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    status('natural_test_complete_artificial_blocked',preview_url=f'http://127.0.0.1:{server.server_port}',checkpoint_step=result['checkpoint_step'])
    server.serve_forever()

if __name__=='__main__':
    try: main()
    except Exception:
        status('failed',traceback=traceback.format_exc()); raise
