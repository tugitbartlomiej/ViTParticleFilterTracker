#!/usr/bin/env python3
"""Zdjęcia bez narzędzia do wyznaczenia tożsamości narzędzia (prośba użytkownika 2026-09-29, PLAN.md).

Dla każdej ze 180 sond z narzędziem (12 wycinków × 15 wersji) dwa zdjęcia „to samo miejsce, bez narzędzia”:
  oko     — ta sama sonda, narzędzie zamalowane z otoczenia (OpenCV inpainting Telea w masce poszerzonej o ~8 px);
            dla wersji na czarnym i żółtym tle wychodzi jednolite tło tego koloru;
  czarne  — całe czarne zdjęcie.
Latent „bez narzędzia” czytamy potem na TYCH SAMYCH tokenach, na których sonda ma narzędzie, więc różnica sonda −
zdjęcie bez narzędzia to samo narzędzie, bez wpływu miejsca na obrazie i reszty sceny.

    <venv>/python tla.py  ->  tla/images/*.png, tla/manifest.json, wyniki/TLA_PODGLAD.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / 'maly_vit_od_zera' / 'eden' / 'wejscia' / 'probe'
OUT = HERE / 'tla'


def main() -> int:
    man = json.loads((PROBE / 'manifest.json').read_text(encoding='utf-8'))
    (OUT / 'images').mkdir(parents=True, exist_ok=True); (HERE / 'wyniki').mkdir(exist_ok=True)
    wyn, podglad = [], []
    for i, m in enumerate(man):
        if not m['semantic_presence']:
            continue
        im = cv2.imread(str(PROBE / m['image']))
        mk = (cv2.imread(str(PROBE / m['mask']), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        dyl = cv2.dilate(mk, np.ones((17, 17), np.uint8))
        oko = cv2.inpaint(im, dyl * 255, 7, cv2.INPAINT_TELEA)
        czarne = np.zeros_like(im)
        for typ, x in (('oko', oko), ('czarne', czarne)):
            p = OUT / 'images' / f'{i:03d}_{typ}.png'
            cv2.imwrite(str(p), x)
            wyn.append(dict(sonda=i, typ=typ, image=f'images/{p.name}', mask_sondy=m['mask'], variant=m['variant'],
                            source_file=m['source_file']))
        if m['variant'] in ('original', 'rotation90', 'black', 'yellow') and len(podglad) < 16:
            podglad.append(np.concatenate([im, oko], 0))
    (OUT / 'manifest.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    cv2.imwrite(str(HERE / 'wyniki' / 'TLA_PODGLAD.png'), np.concatenate(podglad[:8], 1))
    print(f'zapisano {len(wyn)} zdjęć bez narzędzia dla {len(wyn) // 2} sond')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
