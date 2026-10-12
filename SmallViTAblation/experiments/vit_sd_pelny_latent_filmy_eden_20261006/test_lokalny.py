"""Fresh checkpoint inference. Images/latents live only in RAM; JSON metrics on disk."""
import os, sys, io, json, time, hashlib, zipfile, csv, traceback
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import cv2
import torch
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from generate import load_trained
from model import load_vit, vit_forward
from model_snapshot import dekoduj
from dane import read_annotations

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/vit_sd_pelny_latent_filmy_eden_20261006'
SD = Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele')
IMAGES = {}
START = time.time()

def write(name, value):
    p = OUT/name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    for i in range(30):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if i == 29: raise
            time.sleep(.1)

def status(phase, **extra):
    value = dict(phase=phase, pid=os.getpid(), elapsed_seconds=time.time()-START, **extra)
    write('TEST_LOKALNY_STATUS.json', value)
    print(json.dumps(value), flush=True)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def retain(name, rgb):
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    IMAGES[name] = buf.getvalue()

def iou(a, b):
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    areas = np.maximum(0, a[2:]-a[:2]).prod()+np.maximum(0, b[2:]-b[:2]).prod()
    return float(inter/max(areas-inter, 1e-12))

def inspect(vit, rgb, target, j, k, reference_box=None):
    resized = cv2.resize(rgb, (480,320), interpolation=cv2.INTER_AREA)
    with torch.inference_mode():
        output, actual = vit_forward(vit, torch.from_numpy(resized).permute(2,0,1)[None].float().cuda())
        boxes, scores = dekoduj(output)
    z = actual[0].float().cpu().numpy()
    error = (z-target)**2
    mse = float(error[:,j].mean())
    u, v = z[:,j].ravel(), target[:,j].ravel()
    result = dict(MSE128=mse, MSE640=float(error[:,k].mean()),
                  value_agreement128=100*max(0.,1-np.sqrt(mse/max(float((v*v).mean()),1e-12))),
                  cosine128=float(u@v/max(np.linalg.norm(u)*np.linalg.norm(v),1e-12)),
                  confidence=float(scores[0,0]), bbox=boxes[0,0].float().cpu().tolist())
    if reference_box is not None:
        result['reference_layout_iou_not_generated_ground_truth'] = iou(np.array(result['bbox']), reference_box)
    return result

def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    status('loading')
    ckpath = OUT/'best_checkpoint.pt'
    ck = torch.load(ckpath, map_location='cpu', weights_only=True)
    provenance = ck['signature']['provenance']
    step = int(ck['step'])
    vitpath = ROOT/'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
    assert sha(vitpath) == provenance['vit_sha256']
    del ck
    target_file = np.load(ROOT/'outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz')
    artificial = target_file['after'].astype(np.float32)
    j = target_file['channels0'].astype(int)
    k = np.setdiff1d(np.arange(768), j)
    film = 'TR_0127_S1_P03'
    manifest = json.loads((OUT/'FILMY.json').read_text(encoding='utf-8-sig'))
    assert any(r['video_id']==film and r['split']=='val' for r in manifest)
    archive = ROOT.parent/'Dataset/Cataract-LMM/3_Object_Tracking/annotations'/f'{film}.zip'
    annotations = read_annotations(archive, film)
    frame = next(n for n,b in annotations.items() if b is not None)
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if Path(n).stem==f'{film}_{frame:06d}' or Path(n).stem==f'{film}_{frame}')
        rgb = np.asarray(Image.open(io.BytesIO(z.read(name))).convert('RGB')).astype(np.float32)/255
    retain('reference.png', rgb)
    box = np.array(annotations[frame])*np.array([480/rgb.shape[1],320/rgb.shape[0]]*2)
    vit = load_vit(vitpath, torch.device('cuda'))
    natural_image = torch.from_numpy(cv2.resize(rgb,(480,320),interpolation=cv2.INTER_AREA)).permute(2,0,1)[None].cuda()
    with torch.inference_mode():
        output, natural = vit_forward(vit, natural_image)
        boxes, scores = dekoduj(output)
    natural = natural[0].cpu().numpy()
    source_result = dict(confidence=float(scores[0,0]), IoU=iou(boxes[0,0].cpu().numpy(),box))
    trained = load_trained(SD/'ema_020000', SD/'sd-vae-ft-mse', ckpath)
    trained.vae.enable_tiling()
    results = dict(checkpoint_step=step, checkpoint_sha256=sha(ckpath),
                   provenance_verified=True, natural_probe=dict(film=film,frame=frame,split='val',source_detection=source_result),
                   seed=20261006, ddim_steps=50, start='Gaussian noise only', save_images=False, save_latents=False,
                   value_agreement_definition='100*max(0,1-sqrt(MSE128/mean(target128^2)))',
                   IoU_note='Generated-image reference layout IoU is diagnostic, not labelled detection accuracy.')
    for label, target in [('natural',natural),('artificial',artificial)]:
        status('generating', target=label, checkpoint_step=step)
        image = trained.generate(torch.from_numpy(target)[None],steps=50,seed=20261006)
        assert torch.isfinite(image).all()
        generated = image[0].permute(1,2,0).cpu().numpy()
        retain(label+'.png', generated)
        results[label] = inspect(vit, generated, target, j, k, box if label=='natural' else None)
        write('TEST_LOKALNY.json', results)
    del trained,vit,image,natural_image
    torch.cuda.empty_cache()
    status('complete', checkpoint_step=step, preview_url='http://127.0.0.1:8769')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            key = self.path.lstrip('/')
            if key in IMAGES:
                body, mime = IMAGES[key], 'image/png'
            else:
                body = ('<meta charset="utf-8"><style>body{background:#eee;font-family:Arial}img{width:31%;vertical-align:top}h2{font-size:24px}</style><h2>Referencja z filmu VAL | SD: naturalny latent | SD: sztuczny latent</h2>'+''.join(f'<img src="/{s}">' for s in ['reference.png','natural.png','artificial.png'])).encode()
                mime = 'text/html; charset=utf-8'
            self.send_response(200); self.send_header('Content-Type',mime); self.end_headers(); self.wfile.write(body)
        def log_message(self,*args): pass
    HTTPServer(('127.0.0.1',8769),Handler).serve_forever()

if __name__=='__main__':
    try: main()
    except Exception:
        status('failed', traceback=traceback.format_exc())
        raise
