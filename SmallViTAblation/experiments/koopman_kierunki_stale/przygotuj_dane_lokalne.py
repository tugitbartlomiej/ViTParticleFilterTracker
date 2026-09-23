#!/usr/bin/env python3
"""Lokalny katalog danych tracking_static36 do treningu na RTX (bez Edenu), 2026-09-23.

Wyjmuje z archiwow Cataract-LMM tylko potrzebne obrazy: klatki podanych wariantow i caly zbior walidacyjny
(annotations_val.json, 12 043 klatek). Nazwa w archiwum: {film}/{film}_{nr:06d}.jpg, w katalogu danych:
images/{film}_{nr:08d}.jpg (jak w pool_manifest). Zgodnosc sprawdzana skrotem pikseli z pool_manifest
dla klatek uczacych.

    py -3.11 experiments/koopman_kierunki_stale/przygotuj_dane_lokalne.py --warianty wyniki/*.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

KORZEN = Path(__file__).resolve().parents[2]
ZRODLO = KORZEN / 'feature_cache/eden_static36/data/tracking_static36'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--warianty', type=Path, nargs='+', required=True)
    ap.add_argument('--zipy', type=Path, default=KORZEN.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations')
    ap.add_argument('--val', type=Path, default=KORZEN / 'experiments/krystalizacja_latentu/dane/annotations_val.json')
    ap.add_argument('--out', type=Path, default=KORZEN / 'feature_cache/lokalny_static36')
    args = ap.parse_args(argv)

    (args.out / 'images').mkdir(parents=True, exist_ok=True)
    shutil.copy2(ZRODLO / 'pool_manifest.jsonl', args.out / 'pool_manifest.jsonl')
    shutil.copy2(args.val, args.out / 'annotations_val.json')
    man = {}
    for line in (ZRODLO / 'pool_manifest.jsonl').open(encoding='utf-8'):
        r = json.loads(line); man[r['frame_id']] = r
    potrzebne = set()
    for w in args.warianty:
        potrzebne |= set(json.loads(w.read_text(encoding='utf-8'))['frame_ids'])
    uczace = set(potrzebne)
    val = json.loads(args.val.read_text(encoding='utf-8'))
    potrzebne |= {im['frame_id'] for im in val['images']}
    po_filmie: dict[str, list[str]] = {}
    for f in potrzebne:
        po_filmie.setdefault(f.split('/')[-2], []).append(f)
    zle, nowe = [], 0
    for film, lista in sorted(po_filmie.items()):
        with zipfile.ZipFile(args.zipy / f'{film}.zip') as z:
            for f in lista:
                nr = int(f.split('/')[-1])
                cel = args.out / 'images' / f'{film}_{nr:08d}.jpg'
                if cel.exists():
                    continue
                dane = z.read(f'{film}/{film}_{nr:06d}.jpg')
                if f in uczace and man[f].get('pixel_sha256'):
                    px = np.asarray(Image.open(io.BytesIO(dane)).convert('RGB'))
                    if hashlib.sha256(px.tobytes()).hexdigest() != man[f]['pixel_sha256']:
                        zle.append(f)
                cel.write_bytes(dane); nowe += 1
    print(f'obrazow potrzebnych {len(potrzebne)}, nowo zapisanych {nowe}, niezgodnych skrotow {len(zle)} {zle[:3]}')
    return 1 if zle else 0


if __name__ == '__main__':
    raise SystemExit(main())
