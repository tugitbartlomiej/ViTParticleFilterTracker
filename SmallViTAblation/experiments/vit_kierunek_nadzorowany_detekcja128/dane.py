"""Oryginały i ramki źródłowe, zgodność indeksów i podział całych filmów."""
import json, hashlib, os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs'/HERE.name
PAIRS=ROOT/'experiments/maly_vit_test_orientacji_narzedzia_400'
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
CKPT=SOURCE/'MODEL_VIT_K1000.pt'
SPLIT=ROOT/'outputs/vit_cav_pelny_latent_pary400/SPLITS.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p,x):
    p=Path(p); assert p.resolve().is_relative_to(HERE) or p.resolve().is_relative_to(OUT)
    def conv(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(str(type(v)))
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=conv,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def load():
    rows=read(PAIRS/'manifest.json');pairs=read(PAIRS/'PARY.json');split=read(SPLIT)
    originals=[r for r in split['records'] if r['kind']=='original']
    assert len(rows)==len(pairs)==len(originals)==400
    records=[]
    for i,(r,p,s) in enumerate(zip(rows,pairs,originals)):
        assert r['frame_id']==p['frame_id']==s['frame_id']
        assert p['comparison_token_row']==s['pair_index']==i and s['index']==2*i
        assert r['video_id']==s['film'] and s['film'] in split['films'][s['split']]
        path=PAIRS/p['original'];digest=sha(path)
        assert digest==r['image_sha256']==s['sha256']
        x,y,w,h=r['bbox_source_native'];sx,sy=480/r['width'],320/r['height']
        scaled=np.array([x*sx,y*sy,w*sx,h*sy])
        assert np.allclose(scaled,r['bbox_source_480'],atol=1e-4)
        assert w>0 and h>0 and x>=0 and y>=0 and x+w<=r['width'] and y+h<=r['height']
        # Polygon and source bbox consistency; never read or use an ROI mask.
        points=np.concatenate([np.asarray(poly).reshape(-1,2) for poly in r['source_polygons']])
        assert np.allclose(points.min(0),[x,y]) and np.allclose(points.max(0)-points.min(0)+1,[w,h])
        records.append(dict(index=i,source_index=i,source_cav_index=s['index'],frame_id=r['frame_id'],film=s['film'],split=s['split'],path=str(path),sha256=digest,bbox_xywh=scaled,annotation_source=r['source']))
    ids={s:np.array([i for i,r in enumerate(records) if r['split']==s]) for s in ['train','val','test']}
    films={s:sorted({records[i]['film'] for i in v}) for s,v in ids.items()}
    assert list(map(len,ids.values()))==[240,75,85]
    assert not(set(films['train'])&set(films['val']) or set(films['train'])&set(films['test']) or set(films['val'])&set(films['test']))
    with np.load(SOURCE/'LATENT_PELNY.npz') as latent:J=latent['channels0'].astype(np.int64)
    assert J.shape==(128,) and len(set(J))==128 and J.min()>=0 and J.max()<768
    js(OUT/'MANIFEST.json',dict(records=records,indices=ids,films=films,source_split=str(SPLIT),source_split_sha256=sha(SPLIT),pairs_sha256=sha(PAIRS/'PARY.json'),manifest_sha256=sha(PAIRS/'manifest.json'),channels0=J,channels_source_sha256=sha(SOURCE/'LATENT_PELNY.npz'),checkpoint_sha256=sha(CKPT),historical_development_set=True,independent_generalization=False))
    return records,ids,J
