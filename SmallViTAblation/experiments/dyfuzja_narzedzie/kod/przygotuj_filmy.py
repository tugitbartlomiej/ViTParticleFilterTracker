#!/usr/bin/env python3
"""Dane do modelu dyfuzyjnego z CAŁYCH filmów treningowych (prośba użytkownika 2026-09-29: „ucz na klatkach filmów”).

Źródło: archiwa Cataract-LMM `3_Object_Tracking/annotations/<film>.zip` — każdy film ma wszystkie klatki JPEG i maskę
cystotomu (`cystotome_seg`) na każdej klatce. Tylko 118 filmów ze splitu train `pool_manifest.jsonl` (25 filmów
walidacji i filmy spoza manifestu pomijamy; zbiory filmów są rozłączne). Bierzemy co --co-ktora klatkę z maską
(domyślnie co 3.; sąsiednie klatki są prawie identyczne).
Okno: klatka zmniejszona do 480×320 (skala małego ViT), okno 240 px ze środkiem w środku prostokąta opisanego na
obrysie narzędzia (sama etykieta i obrys, bez końcówki — uwaga użytkownika), dosunięte do brzegu klatki.
Zapis w JEDNYM pliku (limit plików na Edenie): okna.bin = kolejno JPEG (RGB, jakość 95) i PNG (maska) każdego okna,
okna_indeks.npy (N,4) int64 = [początek JPEG, długość JPEG, początek PNG, długość PNG], okna.json = opis okien.

    python przygotuj_filmy.py [--co-ktora 3] [--limit-filmow K] [--out ../dane/filmy3]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SVA = HERE.parents[2]
VIT = HERE.parents[3]
MANIFEST = SVA / 'feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl'
ARCHIWA = VIT / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
SKALA, W_, H_, OKNO = 2 / 3, 480, 320, 240


def sha_pliku(p, blok=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(blok):
            h.update(b)
    return h.hexdigest()


def film(arg):
    """Wszystkie okna jednego filmu: lista (jpg bytes, png bytes, opis)."""
    v, co = arg
    out, brak = [], 0
    with zipfile.ZipFile(ARCHIWA / f'{v}.zip') as z:
        nazwy = set(z.namelist())
        raw = z.read(f'{v}/annotation.csv')
        wiersze = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        for r in wiersze:
            n = int(r['frame_number'])
            if n % co or r['cystotome_seg'] in ('', '[]'):
                continue
            if f'{v}/{r["filename"]}' not in nazwy:                          # adnotacja bez obrazu w archiwum
                brak += 1
                continue
            bgr = cv2.imdecode(np.frombuffer(z.read(f'{v}/{r["filename"]}'), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (480, 720):
                continue
            mask = np.zeros(bgr.shape[:2], np.uint8)
            for poly in json.loads(r['cystotome_seg']):
                cv2.fillPoly(mask, [np.rint(np.asarray(poly, np.float32).reshape(-1, 2)).astype(np.int32)], 255)
            bgr = cv2.resize(bgr, (W_, H_), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W_, H_), interpolation=cv2.INTER_AREA)
            if not (mask > 127).any():
                continue
            ys, xs = np.nonzero(mask > 127)                                  # środek prostokąta opisanego na obrysie
            cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
            x0 = int(np.clip(round(cx - OKNO / 2), 0, W_ - OKNO)); y0 = int(np.clip(round(cy - OKNO / 2), 0, H_ - OKNO))
            ok1, jpg = cv2.imencode('.jpg', bgr[y0:y0 + OKNO, x0:x0 + OKNO], [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok2, png = cv2.imencode('.png', mask[y0:y0 + OKNO, x0:x0 + OKNO])
            if not (ok1 and ok2):
                continue
            udz = float((mask[y0:y0 + OKNO, x0:x0 + OKNO] > 127).mean())
            if udz == 0:
                continue
            out.append((jpg.tobytes(), png.tobytes(), dict(video=v, klatka=n, x0=x0, y0=y0,
                                                           srodek=[round(float(cx) - x0, 2), round(float(cy) - y0, 2)],
                                                           maska_udzial=round(udz, 5))))
    return v, out, hashlib.sha256(raw).hexdigest(), len(wiersze), brak


def dekoduj(buf, ind, i):
    """Okno i jako (240,240,4) uint8 RGB+maska (używane też przez trenuj.py)."""
    a, la, b, lb = ind[i]
    bgr = cv2.imdecode(np.frombuffer(buf[a:a + la], np.uint8), cv2.IMREAD_COLOR)
    m = cv2.imdecode(np.frombuffer(buf[b:b + lb], np.uint8), cv2.IMREAD_GRAYSCALE)
    return np.dstack([bgr[..., ::-1], m])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE.parent / 'dane/filmy3'))
    ap.add_argument('--co-ktora', type=int, default=3)
    ap.add_argument('--limit-filmow', type=int, default=0, help='tylko pierwsze K filmów (próba)')
    ap.add_argument('--procesy', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'okna.bin').exists():
        raise FileExistsError(out / 'okna.bin')
    wiersze = [json.loads(l) for l in MANIFEST.read_text(encoding='utf-8').splitlines() if l.strip()]
    train = sorted({r['video_id'] for r in wiersze if r['split'] == 'train'})
    inne = sorted({r['video_id'] for r in wiersze if r['split'] != 'train'})
    assert not set(train) & set(inne), 'filmy train i val nie są rozłączne'
    if a.limit_filmow:
        train = train[:a.limit_filmow]
    opisy, indeks, poz, sha_adn, klatek, braki = [], [], 0, {}, 0, {}
    czesc = out / 'okna.bin.czesc'                                            # nazwa docelowa dopiero po sukcesie
    with open(czesc, 'wb') as f, ProcessPoolExecutor(a.procesy) as ex:
        for k, (v, okna, sha, nk, brak) in enumerate(ex.map(film, [(v, a.co_ktora) for v in train])):
            sha_adn[v] = sha; klatek += nk
            if brak:
                braki[v] = brak
            for jpg, png, d in okna:
                f.write(jpg); f.write(png)
                indeks.append((poz, len(jpg), poz + len(jpg), len(png))); poz += len(jpg) + len(png)
                opisy.append(d)
            print(f'  {k + 1}/{len(train)} {v}: {len(okna)} okien (razem {len(opisy)}, {poz / 2 ** 30:.2f} GB)', flush=True)
    czesc.rename(out / 'okna.bin')
    np.save(out / 'okna_indeks.npy', np.asarray(indeks, np.int64))
    (out / 'okna.json').write_text(json.dumps(opisy, ensure_ascii=False), encoding='utf-8')
    sumy = {n: sha_pliku(out / n) for n in ('okna.bin', 'okna_indeks.npy', 'okna.json')}
    (out / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in sumy.items()), encoding='utf-8', newline='\n')
    man = dict(okien=len(opisy), filmow=len(train), klatek_w_filmach=klatek, co_ktora=a.co_ktora, okno_px_skala_2_3=OKNO,
               skala=SKALA, wycinek_uczenia=176, format='okna.bin: JPEG q95 RGB + PNG maska; okna_indeks.npy (N,4)',
               srodek_okna='środek prostokąta opisanego na obrysie narzędzia',
               okien_na_film={v: sum(d['video'] == v for d in opisy) for v in train},
               klatki_bez_obrazu_w_archiwum=braki, filmy_pominiete_walidacja=inne, zrodlo_manifest=str(MANIFEST), manifest_sha256=sha_pliku(MANIFEST),
               adnotacje_sha256=sha_adn, sha256=sumy, bajtow=poz)
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding='utf-8')
    buf = np.memmap(out / 'okna.bin', dtype=np.uint8, mode='r'); ind = np.load(out / 'okna_indeks.npy')
    rng = np.random.default_rng(0); kafle = []
    for i in rng.choice(len(opisy), min(32, len(opisy)), replace=False):
        o = dekoduj(buf, ind, i); rgb = np.ascontiguousarray(o[..., :3]); m = o[..., 3] > 127
        rgb[m] = (0.6 * rgb[m] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
        kx, ky = opisy[i]['srodek']; cv2.circle(rgb, (int(kx), int(ky)), 4, (0, 255, 0), 1)
        kafle.append(rgb)
    while len(kafle) % 8:
        kafle.append(np.zeros_like(kafle[0]))
    siatka = np.concatenate([np.concatenate(kafle[j:j + 8], 1) for j in range(0, len(kafle), 8)], 0)
    cv2.imwrite(str(out / 'PODGLAD.jpg'), siatka[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps({k: man[k] for k in ('okien', 'filmow', 'klatek_w_filmach', 'bajtow')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
