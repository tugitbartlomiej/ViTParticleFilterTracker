"""Pełne tokeny są stałe; pomyłka sondy lub przyszła baza obala przyczynowość."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import json,hashlib
import numpy as np,cv2,torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
PAIRS=HERE.parent/'maly_vit_test_orientacji_narzedzia_400'
PREVIOUS=HERE.parent/'maly_vit_tozsamosc_czasteczki_pary_400'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def js(path,x):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v))
    Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
def image(path):return cv2.resize(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB),(480,320),interpolation=cv2.INTER_AREA)
def tensor(ims):return torch.from_numpy(np.stack(ims)).permute(0,3,1,2).contiguous().float().cuda()/255
def init_net(step):
    from model_snapshot import MalyViT
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=MalyViT().cuda().eval();ck=torch.load(SOURCE/'wagi'/f'k{step:04d}.pt',weights_only=True,map_location='cpu')
    assert ck['krok']==step;net.load_state_dict(ck['net'],strict=True)
    for p in net.parameters():p.requires_grad_(False)
    net.zapisuj_latent(True);return net
