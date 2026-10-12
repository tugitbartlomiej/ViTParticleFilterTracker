"""Dane sparowane; historyczne J128 i filmy nie są nowym niezależnym testem."""
import json, hashlib, os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs'/HERE.name
PAIRS=ROOT/'experiments/maly_vit_test_orientacji_narzedzia_400'
SPLIT=ROOT/'experiments/maly_vit_tozsamosc_czasteczki_pary_400/PODZIAL.json'
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
CKPT=SOURCE/'MODEL_VIT_K1000.pt'
PAPER=Path('F:/Studia/Articles/Zbiory/JEPA/PDF_all/kim18d.pdf')
SEED=20261005
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p,x):
    def conv(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(str(type(v)))
    p=Path(p); assert p.resolve().is_relative_to(HERE) or p.resolve().is_relative_to(OUT)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=conv,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def load():
    rows=read(PAIRS/'manifest.json'); pairs=read(PAIRS/'PARY.json'); split=read(SPLIT)
    masks=np.load(PAIRS/'TOKEN_MASKI.npz');coverage=masks['tool_coverage'].astype(np.float32)
    # Historical pooled cache uses area_coverage on tool-mask tokens, not tool_coverage.
    coverage=masks['area_coverage'].astype(np.float32)*masks['tool']
    weights=coverage/coverage.sum(1,keepdims=True)
    latent=np.load(SOURCE/'LATENT_PELNY.npz'); J=latent['channels0'].astype(int)
    assert latent['before'].shape==(600,768) and len(J)==128 and len(set(J))==128
    assert np.isfinite(latent['before']).all() and coverage.shape==(400,600)
    for i,(r,p,s) in enumerate(zip(rows,pairs,split['records'])):
        assert r['frame_id']==p['frame_id']==s['frame_id']==str(masks['frame_ids'][i])
        assert s['source_index']==i and p['comparison_token_row']==i
        r['tcav_split']=s['split']; assert split['films'][r['video_id']]==s['split']
        assert (PAIRS/p['original']).exists() and (PAIRS/p['synthetic']).exists()
    ids={n:np.array([i for i,r in enumerate(rows) if r['tcav_split']==n]) for n in ['train','val','test']}
    films={n:set(rows[i]['video_id'] for i in v) for n,v in ids.items()}
    assert len(rows)==len(pairs)==400 and tuple(map(len,ids.values()))==(240,75,85)
    assert not (films['train']&films['val'] or films['train']&films['test'] or films['val']&films['test'])
    return rows,pairs,weights,J,ids
