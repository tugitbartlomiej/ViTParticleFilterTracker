"""Jeden oryginalny JPEG, nie wygenerowany obraz; GT już w układzie 480x320."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
OUT=ROOT/'outputs'/HERE.name
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(device):
    meta=json.loads((SOURCE/'RESULT.json').read_text(encoding='utf-8'))
    assert meta['source_index']==29 and '00001926' in meta['record']['frame_id']
    assert sha(SOURCE/'MODEL_VIT_K1000.pt')==meta['checkpoint_sha']
    assert sha(SOURCE/'ZDJECIE_ZRODLOWE.jpg')==meta['record']['sha256']
    bgr=cv2.imread(str(SOURCE/'ZDJECIE_ZRODLOWE.jpg'))
    rgb=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
    assert np.array_equal(rgb,cv2.cvtColor(cv2.imread(str(SOURCE/'WEJSCIE_VIT.png')),cv2.COLOR_BGR2RGB))
    a=np.load(SOURCE/'LATENT_PELNY.npz')
    before=a['before'].copy();target=a['after'].copy();J=a['channels0'].copy()
    assert before.shape==target.shape==(600,768) and len(set(J.tolist()))==128
    K=np.setdiff1d(np.arange(768),J)
    assert np.array_equal(before[:,K],target[:,K])
    assert np.allclose(target-before,a['delta'],rtol=0,atol=1e-6)
    x,y,w,h=meta['record']['bbox_xywh'];gt=np.array([x,y,x+w,y+h],np.float32)
    image=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().to(device)/255
    return meta,rgb,image,torch.from_numpy(before).to(device),torch.from_numpy(target).to(device),torch.tensor(J,device=device),torch.tensor(K,device=device),gt
