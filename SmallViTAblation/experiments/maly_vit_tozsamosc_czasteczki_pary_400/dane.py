"""Dane par są stałe; inne indeksy maski lub zmieniony hash obalają pomiar."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import hashlib,json
import cv2,numpy as np,torch

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'maly_vit_test_orientacji_narzedzia_400'
SOURCE=HERE.parent/'maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
CHECKPOINT=SOURCE/'wagi/k1000.pt'
SEED=20261001

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def js(path,obj):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')

def load():
    rows=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    pairs=json.loads((DATA/'PARY.json').read_text(encoding='utf-8'))
    t=dict(np.load(DATA/'TOKEN_MASKI.npz'))
    assert len(rows)==len(pairs)==400 and t['tool'].shape==(400,600)
    assert np.array_equal(t['frame_ids'],[r['frame_id'] for r in rows])
    films=sorted({r['video_id'] for r in rows});rng=np.random.default_rng(SEED);rng.shuffle(films)
    assert len(films)==116
    split={f:('train' if i<69 else 'val' if i<92 else 'test') for i,f in enumerate(films)}
    for i,r in enumerate(rows):
        assert r['frame_id']==pairs[i]['frame_id']
        r['split']=split[r['video_id']]
        r['synthetic_absolute']=str(DATA/pairs[i]['synthetic'])
        assert sha(DATA/r['image_path'])==r['image_sha256']
        assert sha(r['synthetic_absolute'])==pairs[i]['synthetic_sha256']
    counts={s:sum(r['split']==s for r in rows) for s in ('train','val','test')}
    js(HERE/'PODZIAL.json',dict(seed=SEED,films=split,image_counts=counts,records=[dict(source_index=i,frame_id=r['frame_id'],film=r['video_id'],split=r['split']) for i,r in enumerate(rows)]))
    return rows,t

def image(path):
    im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    return cv2.resize(im,(480,320),interpolation=cv2.INTER_AREA)

def tensor(images):return torch.from_numpy(np.stack(images)).permute(0,3,1,2).contiguous().float().cuda()/255

def init_net():
    from model_snapshot import MalyViT
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=MalyViT().cuda().eval();ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert ck['krok']==1000;net.load_state_dict(ck['net'],strict=True)
    for p in net.parameters():p.requires_grad_(False)
    net.zapisuj_latent(True)
    return net

def weights(rows,ids):
    films={rows[i]['video_id'] for i in ids};counts={f:sum(rows[i]['video_id']==f for i in ids) for f in films}
    return np.array([1/len(films)/counts[rows[i]['video_id']] for i in ids])
