#!/usr/bin/env python3
"""Sprawdzenie generatora SD na lokalnej karcie: 8 klatek z układów prawdziwych klatek puli (4 × R2 bez zmian,
4 × R1 z przestawionym narzędziem), mozaika z konturem maski narzędzia, czas rysowania, zgodność kształtów.

    uruchom_lokalnie.cmd test_sd_lokalnie.py      ->  wyniki/test_sd/MOZAIKA.jpg, TEST_SD.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'kod'))
import generator_sd as GS  # noqa: E402
import uklady as UK  # noqa: E402


def main() -> int:
    out = HERE / 'wyniki' / 'test_sd'; out.mkdir(parents=True, exist_ok=True)
    dane = Path(os.environ['MALY_VIT_DANE'])
    wiersze = [json.loads(l) for l in open(dane / 'pool_manifest.jsonl', encoding='utf-8') if l.strip()]
    wiersze = [r for r in wiersze if r.get('split') == 'train']
    rng = np.random.default_rng(7)
    uk = []
    for j in rng.permutation(len(wiersze)):
        u = UK.uklad(wiersze[int(j)])
        if u is not None:
            uk.append(u)
        if len(uk) == 8:
            break
    uk = uk[:4] + [UK.przeksztalc(u, rng)[0] for u in uk[4:]]
    M = np.stack([UK.maski(u) for u in uk])
    t0 = time.time(); G = GS.GeneratorSD(); t_lad = time.time() - t0
    t0 = time.time(); X = G.rysuj(M, ziarno=11, kroki=30, cfg=2.0); t_rys = time.time() - t0
    kafle = []
    for x, u in zip(X, uk):
        im = np.ascontiguousarray(x)
        mk = cv2.resize((M[len(kafle), 0] > 127).astype(np.uint8), (UK.SD_SZER, UK.SD_WYS), interpolation=cv2.INTER_NEAREST)
        kont, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, kont, -1, (255, 40, 40), 1)
        kafle.append(im)
    mozaika = np.concatenate([np.concatenate(kafle[:4], 1), np.concatenate(kafle[4:], 1)], 0)
    cv2.imwrite(str(out / 'MOZAIKA.jpg'), cv2.cvtColor(mozaika, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    rap = dict(zrodlo=G.zrodlo, ladowanie_s=t_lad, rysowanie_s=t_rys, s_na_klatke=t_rys / len(X), ksztalt=list(X.shape),
               klatki=[u['frame_id'] for u in uk], jasnosc=float(X.mean()), ramki_vit=[UK.ramka_vit(u).tolist() for u in uk])
    (out / 'TEST_SD.json').write_text(json.dumps(rap, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in rap.items() if k not in ('ramki_vit',)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
