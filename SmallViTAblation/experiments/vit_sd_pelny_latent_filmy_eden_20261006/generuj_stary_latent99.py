"""One new SD sample conditioned on the exact old full oracle latent; RAM images."""
import test_lokalny as helper
import json, traceback, os, io
from pathlib import Path
import torch, numpy as np
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from generate import load_trained
from model import load_vit

def main():
    original_out = helper.OUT
    helper.OUT = helper.ROOT/'outputs/vit_sd_stary_latent_iou99_20261006'
    helper.OUT.mkdir(parents=True, exist_ok=True)
    helper.status('loading')
    torch.set_num_threads(4)
    target_dir = helper.ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
    target_path = target_dir/'LATENT_PELNY.npz'
    assert helper.sha(target_path)=='2e00434405969a0742b30f15e16cb6156f99e0a6aae2ee9e0f93ce90249a66df'
    arrays = np.load(target_path)
    target = arrays['after'].astype(np.float32)
    j = arrays['channels0'].astype(int)
    k = np.setdiff1d(np.arange(768), j)
    assert target.shape==(600,768) and len(j)==128
    ckpath = original_out/'best_checkpoint.pt'
    trained = load_trained(helper.SD/'ema_020000', helper.SD/'sd-vae-ft-mse', ckpath)
    trained.vae.enable_tiling()
    helper.status('generating', steps=100, seed=2026100602, checkpoint_step=trained.step)
    rgb = trained.generate(torch.from_numpy(target)[None], steps=100, seed=2026100602)
    assert torch.isfinite(rgb).all()
    image = rgb[0].permute(1,2,0).cpu().numpy()
    step = trained.step
    del trained,rgb
    torch.cuda.empty_cache()
    vit = load_vit(target_dir/'MODEL_VIT_K1000.pt', torch.device('cuda'))
    metrics = helper.inspect(vit,image,target,j,k)
    helper.retain('generated.png',image)
    source = np.asarray(Image.open(target_dir/'ZDJECIE_ZRODLOWE.jpg').convert('RGB')).astype(np.float32)/255
    helper.retain('source.png',source)
    result = dict(target_sha256=helper.sha(target_path), checkpoint_sha256=helper.sha(ckpath),
                  checkpoint_step=step, target='exact original full after latent; all 600x768 values',
                  seed=2026100602, ddim_steps=100, start='Gaussian noise, no source image input',
                  save_images=False,save_latents=False, metrics=metrics,
                  note='Historical IoU belongs to the original image with injected activations, not this generation.')
    helper.write('RESULT.json',result)
    del vit
    torch.cuda.empty_cache()
    helper.status('complete', preview_url='http://127.0.0.1:8770')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            key=self.path.lstrip('/')
            if key in helper.IMAGES:
                body,mime=helper.IMAGES[key],'image/png'
            else:
                body=('<meta charset="utf-8"><h2>Oryginalne zdjęcie | SD ze starego sztucznego latentu (100 kroków)</h2><img width="48%" src="/source.png"><img width="48%" src="/generated.png">').encode()
                mime='text/html; charset=utf-8'
            self.send_response(200);self.send_header('Content-Type',mime);self.end_headers();self.wfile.write(body)
        def log_message(self,*args): pass
    HTTPServer(('127.0.0.1',8770),Handler).serve_forever()

if __name__=='__main__':
    try: main()
    except Exception:
        helper.status('failed',traceback=traceback.format_exc())
        raise
