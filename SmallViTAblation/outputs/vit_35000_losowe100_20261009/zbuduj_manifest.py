"""100 losowych klatek z puli 35 000 (jedna ramka narzędzia) -> obrazy/ (oryginalne bajty z zipów) + MANIFEST_100.jsonl
w formacie pomiaru naturalnego (jak MANIFEST_1500: rozciągnięcie do 480 x 320, ramka skalowana osobno w X i Y)."""
import hashlib, json, zipfile
from pathlib import Path
import numpy as np

TU = Path(__file__).resolve().parent
ROOT = TU.parents[1]
POOL = ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl'
ZIPY = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
rows = [json.loads(l) for l in open(POOL, encoding='utf-8')]
jedna = [i for i, r in enumerate(rows) if len(r['gt']) == 1]
wyb = sorted(np.random.default_rng(2026).choice(jedna, 100, replace=False).tolist())
out = []
for k, i in enumerate(wyb):
    r = rows[i]
    with zipfile.ZipFile(ZIPY / Path(r['archive'].replace(chr(92), '/')).name) as z:
        raw = z.read(r['member'])
    assert hashlib.sha256(raw).hexdigest() == r['image_sha256']
    nazwa = Path(r['member']).name
    (TU / 'obrazy' / nazwa).write_bytes(raw)
    x, y, w, h = r['gt'][0]['bbox_xywh']
    W, H = r['width'], r['height']
    out.append(dict(index=k, image_id=k, annotation_id=k, image_name=nazwa, image_sha256=r['image_sha256'],
                    frame_id=f'cataract_lmm/tracking/{r["film"]}/{int(r["frame"]):08d}', pool_row=i,
                    category=r['gt'][0]['category'], source_width=W, source_height=H,
                    bbox_xywh_source=[x, y, w, h],
                    bbox_xyxy_480=[x * 480 / W, y * 320 / H, (x + w) * 480 / W, (y + h) * 320 / H],
                    resize_scale_xy=[480 / W, 320 / H], source_dimensions_verified_from_bytes=False))
(TU / 'MANIFEST_100.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out), encoding='utf-8')
print(len(out), 'klatek z', len({r['frame_id'].split('/')[2] for r in out}), 'filmów; pula z jedną ramką:', len(jedna))
