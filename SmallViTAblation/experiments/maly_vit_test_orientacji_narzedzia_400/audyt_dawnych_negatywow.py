"""Czy dawne empty_black są czystą interwencją usunięcia narzędzia?
Zmiana poza maską albo crop zamiast pełnej klatki ogranicza tę interpretację.
Tylko odczyt12 istniejących par; żadnej edycji i żadnych forwardów.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
P=ROOT/'experiments/koopman_fastgan_petla/wyniki/local_1810448/run/probe'
rows=json.loads((P/'manifest.json').read_text());pairs=[]
for r in rows:
    if r['variant']!='original':continue
    n=next(a for a in rows if a['frame_id']==r['frame_id'] and a['variant']=='empty_black')
    a=np.array(Image.open(P/r['image']).convert('RGB'));b=np.array(Image.open(P/n['image']).convert('RGB'))
    mask=np.array(Image.open(P/r['mask']).convert('L'))>0
    diff=np.abs(a.astype(float)-b.astype(float));changed=np.any(a!=b,axis=-1)
    pairs.append(dict(frame_id=r['frame_id'],split=r['split'],size_wh=list(a.shape[1::-1]),
        original=str(P/r['image']),negative=str(P/n['image']),mask_original=str(P/r['mask']),
        outside_mask_changed_fraction=float(changed[~mask].mean()),outside_mask_MAE=float(diff[~mask].mean()),
        inside_mask_MAE=float(diff[mask].mean()),
        source_manifest_sha256=hashlib.sha256((P/'manifest.json').read_bytes()).hexdigest(),
        limitation='existing crop256 empty_black, not400 full-image LaMa negatives'))
(HERE/'AUDYT_DAWNYCH_NEGATYWOW.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2),encoding='utf-8')
print('pairs',len(pairs),'outside changed',np.mean([r['outside_mask_changed_fraction'] for r in pairs]))
