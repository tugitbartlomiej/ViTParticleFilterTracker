"""Wybrane zdjęcia do pomiaru trajektorii cząsteczek: podzbiór manifestu 1500 albo puli 35 000 klatek uczących.

`zbuduj(zrodlo, indeksy)` zapisuje manifest JSONL w formacie pomiaru naturalnego (index 0..n-1, jedna ramka,
rozciągnięcie do 480 × 320 ze skalą X i Y osobno, SHA-256 bajtów) w outputs/zbiory_zdjec/<nazwa>/ i zwraca
(manifest, katalog zdjęć). Zdjęcia 1500 leżą już na dysku; klatki puli 35 000 są wypakowywane z lokalnych zipów
Cataract-LMM raz, do wspólnej pamięci podręcznej, z kontrolą SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import ruch_zbioru as R

ROOT = R.ROOT
WYJSCIE = ROOT / 'outputs/zbiory_zdjec'
PULA_35000 = ROOT / 'experiments/maly_vit_od_zera_35000_21epok_20261009/wejscia/MANIFEST_35000.jsonl'
ZIPY = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
CACHE_35000 = WYJSCIE / '_pula35000_obrazy'
ZRODLA = {'1500': 'Manifest 1500 obrazów pomiaru', '35000': 'Pula 35 000 klatek uczących (zbiór treningu 35 000)'}

_wiersze = {}


def wiersze(zrodlo):
    """Wszystkie wiersze źródła (w pamięci po pierwszym odczycie)."""
    if zrodlo not in _wiersze:
        plik = R.DEFAULT_MANIFEST if zrodlo == '1500' else PULA_35000
        _wiersze[zrodlo] = [json.loads(l) for l in open(plik, encoding='utf-8') if l.strip()]
    return _wiersze[zrodlo]


def opisy(zrodlo):
    """(indeks, nazwa zdjęcia, film) dla listy wyboru."""
    if zrodlo == '1500':
        return [(i, r['image_name'], r['frame_id'].split('/')[2]) for i, r in enumerate(wiersze(zrodlo))]
    return [(i, Path(r['member']).name, r['film']) for i, r in enumerate(wiersze(zrodlo))]


def parsuj(tekst, n):
    """'0-9, 25, 100-102' -> posortowane, unikalne indeksy < n. Pusty tekst: pusta lista."""
    out = set()
    for kawalek in re.split(r'[,\s;]+', tekst.strip()):
        if not kawalek:
            continue
        m = re.fullmatch(r'(\d+)\s*-\s*(\d+)', kawalek)
        a, b = (int(m[1]), int(m[2])) if m else (int(kawalek), int(kawalek))
        if a > b:
            a, b = b, a
        if b >= n:
            raise ValueError(f'indeks {b} poza zakresem 0..{n - 1}')
        out.update(range(a, b + 1))
    return sorted(out)


def zwiez(indeksy):
    """Posortowane indeksy -> zwięzły tekst zakresów ('0-9, 25')."""
    s, out = sorted(set(indeksy)), []
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        out.append(str(s[i]) if i == j else f'{s[i]}-{s[j]}')
        i = j + 1
    return ', '.join(out)


def losowe(zrodlo, n, ziarno=2026):
    import numpy as np
    m = len(wiersze(zrodlo))
    return sorted(np.random.default_rng(ziarno).choice(m, min(n, m), replace=False).tolist())


def _wypakuj(r):
    """Klatka puli z zipu do pamięci podręcznej (raz), SHA-256 zgodne z manifestem."""
    cel = CACHE_35000 / Path(r['member']).name
    if cel.is_file() and hashlib.sha256(cel.read_bytes()).hexdigest() == r['image_sha256']:
        return cel
    with zipfile.ZipFile(ZIPY / Path(r['archive'].replace(chr(92), '/')).name) as z:
        raw = z.read(r['member'])
    if hashlib.sha256(raw).hexdigest() != r['image_sha256']:
        raise ValueError(f'SHA-256 klatki niezgodne z manifestem puli: {r["member"]}')
    CACHE_35000.mkdir(parents=True, exist_ok=True)
    tmp = cel.with_suffix('.part')
    tmp.write_bytes(raw)
    tmp.replace(cel)
    return cel


def zbuduj(zrodlo, indeksy, postep=lambda i, n: None):
    """Manifest wybranych zdjęć. Zwraca (ścieżka manifestu, katalog zdjęć dla image_root)."""
    indeksy = sorted(set(int(i) for i in indeksy))
    if not indeksy:
        raise ValueError('nie wybrano żadnego zdjęcia')
    w = wiersze(zrodlo)
    klucz = hashlib.sha256(f'{zrodlo}:{indeksy}'.encode()).hexdigest()[:10]
    katalog = WYJSCIE / f'{zrodlo}_{len(indeksy)}_{klucz}'
    katalog.mkdir(parents=True, exist_ok=True)
    out = []
    for k, i in enumerate(indeksy):
        r = w[i]
        if zrodlo == '1500':
            row = dict(r, index=k, zrodlo_indeks=i)
        else:
            _wypakuj(r)
            x, y, bw, bh = r['gt'][0]['bbox_xywh']
            W, H = r['width'], r['height']
            row = dict(index=k, image_id=i, annotation_id=i, image_name=Path(r['member']).name,
                       image_sha256=r['image_sha256'], frame_id=f'cataract_lmm/tracking/{r["film"]}/{int(r["frame"]):08d}',
                       zrodlo_indeks=i, category=r['gt'][0]['category'], source_width=W, source_height=H,
                       bbox_xywh_source=[x, y, bw, bh],
                       bbox_xyxy_480=[x * 480 / W, y * 320 / H, (x + bw) * 480 / W, (y + bh) * 320 / H],
                       resize_scale_xy=[480 / W, 320 / H], source_dimensions_verified_from_bytes=False)
        out.append(row)
        postep(k + 1, len(indeksy))
    manifest = katalog / 'MANIFEST.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out), encoding='utf-8')
    (katalog / 'WYBOR.json').write_text(json.dumps(dict(zrodlo=zrodlo, indeksy=indeksy, opis=ZRODLA[zrodlo]),
                                                   ensure_ascii=False, indent=1), encoding='utf-8')
    return manifest, (R._IMAGE_ROOT if zrodlo == '1500' else CACHE_35000)
