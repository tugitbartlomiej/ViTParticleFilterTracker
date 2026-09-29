#!/usr/bin/env python3
"""Dane do modelu dyfuzyjnego narzędzia (PLAN.md): okna 240×240 z klatek treningowych, RGB + maska narzędzia.

Źródło: klatki split=train z `pool_manifest.jsonl` (10 637 klatek, 118 filmów; walidacji i testu nie ruszamy),
segmentacja cystotomu z `annotation.csv` w archiwach Cataract-LMM (jak `fastgan_narzedzie/przygotuj_wycinki.py`),
sprawdzenie sumy SHA256 pikseli z manifestem.
Skala: klatka 720×480 zmniejszona do 480×320 (2/3), czyli dokładnie skala, w której uczy się mały ViT i w której
sondy mają 176 px. Okno 240 px (360 px natywnie) ze środkiem w końcówce narzędzia (`tip_point`, a gdy go brak — środek
maski), dosunięte do brzegu klatki, bez dopełniania. Uczenie tnie z okna losowy wycinek 176×176, więc końcówka trafia
w różne miejsca wycinka.

    python przygotuj_dane.py [--limit N] [--out ../dane]
    ->  okna240.npy (N,240,240,4) uint8, okna240.json (opis okien), MANIFEST.json, SHA256SUMS, PODGLAD.jpg
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SVA = HERE.parents[2]                                   # SmallViTAblation
VIT = HERE.parents[3]                                   # ViTParticleFilterTracker
MANIFEST = SVA / 'feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl'
LOKALNE = SVA / 'feature_cache/lokalny_static36/images'
ARCHIWA = VIT / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
SKALA, W_, H_, OKNO = 2 / 3, 480, 320, 240


def sha_pliku(p, blok=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(blok):
            h.update(b)
    return h.hexdigest()


def adnotacje(video):
    with zipfile.ZipFile(ARCHIWA / f'{video}.zip') as z:
        raw = z.read(f'{video}/annotation.csv')
    return {r['filename']: r for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))}, hashlib.sha256(raw).hexdigest()


def piksele(r):
    v, n = r['video_id'], int(r['frame_index'])
    lok = LOKALNE / f'{v}_{n:08d}.jpg'
    if lok.exists():
        dane = lok.read_bytes()
    else:
        with zipfile.ZipFile(ARCHIWA / f'{v}.zip') as z:
            dane = z.read(f'{v}/{v}_{n:06d}.jpg')
    rgb = np.asarray(Image.open(io.BytesIO(dane)).convert('RGB'))
    if hashlib.sha256(rgb.tobytes()).hexdigest() != r['pixel_sha256']:
        raise ValueError(f'niezgodne piksele: {r["frame_id"]}')
    return rgb


def okno(r, seg):
    """Okno 240×240×4 w skali 2/3 i jego opis."""
    rgb = piksele(r)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for poly in seg:
        cv2.fillPoly(mask, [np.rint(np.asarray(poly, np.float32).reshape(-1, 2)).astype(np.int32)], 255)
    rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (W_, H_), interpolation=cv2.INTER_AREA)
    if r.get('tip_point'):
        cx, cy = (np.asarray(r['tip_point'], np.float64) * SKALA).tolist(); zrodlo = 'tip_point'
    else:
        ys, xs = np.nonzero(mask > 127)
        cx, cy = float(xs.mean()), float(ys.mean()); zrodlo = 'srodek_maski'
    x0 = int(np.clip(round(cx - OKNO / 2), 0, W_ - OKNO)); y0 = int(np.clip(round(cy - OKNO / 2), 0, H_ - OKNO))
    o = np.dstack([rgb[y0:y0 + OKNO, x0:x0 + OKNO], mask[y0:y0 + OKNO, x0:x0 + OKNO]])
    return o, dict(frame_id=r['frame_id'], video=r['video_id'], klatka=int(r['frame_index']), x0=x0, y0=y0,
                   koncowka=[round(cx - x0, 2), round(cy - y0, 2)], zrodlo_srodka=zrodlo,
                   maska_udzial=round(float((o[..., 3] > 127).mean()), 5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE.parent / 'dane'))
    ap.add_argument('--limit', type=int, default=0, help='tylko pierwsze N okien (próba)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'okna240.npy').exists():
        raise FileExistsError(out / 'okna240.npy')
    wiersze = [json.loads(l) for l in MANIFEST.read_text(encoding='utf-8').splitlines() if l.strip()]
    wiersze = [r for r in wiersze if r['split'] == 'train']
    filmy = sorted({r['video_id'] for r in wiersze})
    # przebieg 1: tylko adnotacje (bez obrazów) — które klatki mają segmentację narzędzia
    zadania, pominiete, sha_adn = [], [], {}
    for v in filmy:
        adn, sha_adn[v] = adnotacje(v)
        for r in sorted((r for r in wiersze if r['video_id'] == v), key=lambda r: r['frame_index']):
            a_ = adn.get(f'{v}_{int(r["frame_index"]):06d}.jpg')
            seg = json.loads(a_['cystotome_seg']) if a_ and a_.get('cystotome_seg') not in (None, '', '[]') else None
            (zadania if seg else pominiete).append((r, seg) if seg else r['frame_id'])
    print(f'klatek train {len(wiersze)}, z segmentacją {len(zadania)}, bez segmentacji {len(pominiete)}', flush=True)
    if a.limit:
        zadania = zadania[:: max(1, len(zadania) // a.limit)][:a.limit]
    N = len(zadania)
    X = np.lib.format.open_memmap(out / 'okna240.npy', mode='w+', dtype=np.uint8, shape=(N, OKNO, OKNO, 4))
    opisy: list[dict] = [{} for _ in range(N)]

    def zrob(i):
        o, d = okno(*zadania[i])
        X[i] = o; opisy[i] = d

    with ThreadPoolExecutor(8) as ex:
        for k, _ in enumerate(ex.map(zrob, range(N))):
            if k % 1000 == 999:
                print(f'  {k + 1}/{N}', flush=True)
    X.flush()
    puste = [i for i, d in enumerate(opisy) if d['maska_udzial'] == 0]
    (out / 'okna240.json').write_text(json.dumps(opisy, ensure_ascii=False), encoding='utf-8')
    sumy = {n: sha_pliku(out / n) for n in ('okna240.npy', 'okna240.json')}
    (out / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in sumy.items()), encoding='utf-8', newline='\n')
    man = dict(okien=N, ksztalt=[N, OKNO, OKNO, 4], kanaly='RGB + maska cystotomu (0-255, INTER_AREA)', skala=SKALA,
               okno_px_skala_2_3=OKNO, okno_px_natywnie=round(OKNO / SKALA), wycinek_uczenia=176,
               srodek={k: sum(d['zrodlo_srodka'] == k for d in opisy) for k in ('tip_point', 'srodek_maski')},
               okna_bez_maski=len(puste), bez_segmentacji=len(pominiete), filmow=len(filmy),
               zrodlo_manifest=str(MANIFEST), manifest_sha256=sha_pliku(MANIFEST), adnotacje_sha256=sha_adn,
               limit=a.limit, sha256=sumy)
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding='utf-8')
    rng = np.random.default_rng(0)
    kafle = []
    for i in rng.choice(N, min(32, N), replace=False):
        o = X[i].copy(); rgb = o[..., :3].copy()
        m = o[..., 3] > 127
        rgb[m] = (0.6 * rgb[m] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
        kx, ky = opisy[i]['koncowka']
        cv2.circle(rgb, (int(kx), int(ky)), 4, (0, 255, 0), 1)
        kafle.append(rgb)
    while len(kafle) % 8:
        kafle.append(np.zeros_like(kafle[0]))
    siatka = np.concatenate([np.concatenate(kafle[j:j + 8], 1) for j in range(0, len(kafle), 8)], 0)
    Image.fromarray(siatka).save(out / 'PODGLAD.jpg', quality=90)
    print(json.dumps({k: man[k] for k in ('okien', 'srodek', 'okna_bez_maski', 'bez_segmentacji')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
