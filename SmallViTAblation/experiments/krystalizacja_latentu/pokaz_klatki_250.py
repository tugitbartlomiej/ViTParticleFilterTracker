#!/usr/bin/env python3
"""Wypakowuje klatki wybrane do biegow na 250 klatkach i sklada z nich mozaiki (2026-09-23).

Klatka `.../TR_xxxx/000000NN` to plik `TR_xxxx_0000NN.jpg` z archiwum adnotacji Cataract-LMM
(sprawdzone skrotem pikseli z `pool_manifest.jsonl`). Na kazdej klatce rysujemy ramke z etykiety,
zeby bylo widac narzedzie; do samego wyboru etykiety nie byly uzywane.

    py -3.11 experiments/krystalizacja_latentu/pokaz_klatki_250.py \\
        --dataset ../Dataset/Cataract-LMM/3_Object_Tracking/annotations \\
        --manifest feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl \\
        --out experiments/krystalizacja_latentu/wyniki/klatki_250
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

RAMIONA = {
    'K6_po_kwadratach': 'curated/selection_maly_budzet/maly_K6_identity_cells_b250_sel42.json',
    'B1m_losowo': 'curated/selection_maly_budzet/maly_B1m_random_b250_sel42.json',
    'B2m_rownomiernie': 'curated/selection_maly_budzet/maly_B2m_uniform_time_b250_sel42.json',
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kolumn', type=int, default=20)
    ap.add_argument('--miniatura', type=int, default=144)
    args = ap.parse_args(argv)

    ramki = {}
    for line in args.manifest.open(encoding='utf-8'):
        r = json.loads(line)
        ramki[r['frame_id']] = [a['bbox'] for a in r.get('annotations') or []]

    for nazwa, plik in RAMIONA.items():
        ids = json.loads(Path(plik).read_text(encoding='utf-8'))['frame_ids']
        kat = args.out / nazwa
        kat.mkdir(parents=True, exist_ok=True)
        w, h = args.miniatura, args.miniatura * 2 // 3
        wiersze = (len(ids) + args.kolumn - 1) // args.kolumn
        mozaika = Image.new('RGB', (args.kolumn * w, wiersze * h), 'white')
        zipy: dict[str, zipfile.ZipFile] = {}
        for i, fid in enumerate(ids):
            film, nr = fid.split('/')[-2], int(fid.split('/')[-1])
            z = zipy.setdefault(film, zipfile.ZipFile(args.dataset / f'{film}.zip'))
            img = Image.open(io.BytesIO(z.read(f'{film}/{film}_{nr:06d}.jpg'))).convert('RGB')
            rys = ImageDraw.Draw(img)
            for x, y, bw, bh in ramki.get(fid, []):
                rys.rectangle([x, y, x + bw, y + bh], outline=(0, 255, 0), width=3)
            img.save(kat / f'{i:03d}_{film}_{nr:06d}.jpg', quality=90)
            mozaika.paste(img.resize((w, h)), ((i % args.kolumn) * w, (i // args.kolumn) * h))
        for z in zipy.values():
            z.close()
        mozaika.save(args.out / f'MOZAIKA_{nazwa}.jpg', quality=88)
        print(f'{nazwa}: {len(ids)} klatek z {len(zipy)} filmow -> {kat}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
