#!/usr/bin/env python3
"""Przykłady Stable Diffusion dla użytkownika (2026-09-30): oko z narzędziem z masek prawdziwych klatek, to samo z
narzędziem przestawionym, rysunek z samą maską narzędzia (bez rogówki i źrenicy) oraz narzędzie z klatki SD wycięte po
masce i wklejone na czarne i żółte tło (jak warianty sond `black` i `yellow`). SD nie widziało czarnych ani żółtych teł
w uczeniu, więc tła dostajemy przez wklejenie, nie przez rysowanie.

    uruchom_lokalnie.cmd przyklady_sd.py   ->  wyniki/przyklady_sd/PRZYKLADY.jpg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'kod'))
import generator_sd as GS  # noqa: E402
import uklady as UK  # noqa: E402

ZOLTY = (235, 205, 40)


def podpis(im, t):
    im = np.ascontiguousarray(im)
    cv2.putText(im, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(im, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return im


def main() -> int:
    out = HERE / 'wyniki' / 'przyklady_sd'; out.mkdir(parents=True, exist_ok=True)
    dane = Path(os.environ['MALY_VIT_DANE'])
    w = [json.loads(l) for l in open(dane / 'pool_manifest.jsonl', encoding='utf-8') if l.strip()]
    w = [r for r in w if r.get('split') == 'train']
    rng = np.random.default_rng(21)
    uk, obr = [], []
    for j in rng.permutation(len(w)):
        u = UK.uklad(w[int(j)])
        if u is not None:
            uk.append(u)
            im = cv2.cvtColor(cv2.imread(str(dane / w[int(j)]['path'])), cv2.COLOR_BGR2RGB)
            obr.append(cv2.resize(im, (UK.SD_SZER, UK.SD_WYS), interpolation=cv2.INTER_AREA))
        if len(uk) == 4:
            break
    przest = [UK.przeksztalc(u, rng)[0] for u in uk]
    M = np.stack([UK.maski(u) for u in uk] + [UK.maski(u) for u in przest])
    M_nar = M[:4].copy(); M_nar[:, 1:] = 0                      # sama maska narzędzia
    G = GS.GeneratorSD()
    X = G.rysuj(np.concatenate([M, M_nar]), ziarno=5)
    pelne = X[:4]; X_prz = X[4:8]; X_nar = X[8:12]

    def kontur(im, u):
        im = np.ascontiguousarray(im.copy())
        mk = UK.maska_narzedzia_sd(u).astype(np.uint8)
        k, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, k, -1, (255, 40, 40), 1)
        return im

    W, H = 288, 192
    r1 = [podpis(cv2.resize(o, (W, H)), 'prawdziwa klatka') for o in obr]
    r2 = [podpis(cv2.resize(kontur(x, u), (W, H)), 'SD: ten sam uklad') for x, u in zip(pelne, uk)]
    r3 = [podpis(cv2.resize(kontur(x, u), (W, H)), 'SD: narzedzie przestawione') for x, u in zip(X_prz, przest)]
    r4 = [podpis(cv2.resize(kontur(x, u), (W, H)), 'SD: tylko maska narzedzia') for x, u in zip(X_nar, uk)]
    wyc = []
    for x, u in zip(pelne, uk):
        img, m = UK.wycinek_176(x, u)
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cz = np.zeros_like(img); cz[m] = img[m]
        zo = np.empty_like(img); zo[:] = ZOLTY; zo[m] = img[m]
        wyc.append(np.concatenate([podpis(cv2.resize(img, (96, 96)), ''), cv2.resize(cz, (96, 96)), cv2.resize(zo, (96, 96))], 1))
    r5 = [cv2.copyMakeBorder(x, 0, H - 96, 0, W - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in wyc]
    r5 = [podpis(x, 'wycinek | czarne | zolte') for x in r5]
    rzedy = [np.concatenate(r, 1) for r in (r1, r2, r3, r4, r5)]
    siatka = np.concatenate(rzedy, 0)
    cv2.imwrite(str(out / 'PRZYKLADY.jpg'), cv2.cvtColor(siatka, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # większe wycinki na czarnym i żółtym tle
    duze = []
    for x, u in zip(pelne, uk):
        img, m = UK.wycinek_176(x, u)
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cz = np.zeros_like(img); cz[m] = img[m]
        zo = np.empty_like(img); zo[:] = ZOLTY; zo[m] = img[m]
        duze.append(np.concatenate([img, cz, zo], 0))
    cv2.imwrite(str(out / 'TLA.jpg'), cv2.cvtColor(np.concatenate(duze, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / 'PRZYKLADY.json').write_text(json.dumps(dict(sd=G.zrodlo, klatki=[u['frame_id'] for u in uk]), ensure_ascii=False, indent=1),
                                        encoding='utf-8')
    print('gotowe', G.zrodlo['unet'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
