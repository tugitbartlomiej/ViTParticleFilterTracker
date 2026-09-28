"""Wybór prawdziwych klatek do dodania (PLAN.md).

koopman: obraz z FastGAN (target.png) każdej kotwicy → wyszukiwanie w indeksie DINOv2 puli (wycinki narzędzia,
         `koopman_fastgan_petla`, tylko odczyt) top-90 → klatki na przemian z list kotwic, po kolei wg podobieństwa;
losowe:  losowe klatki z tej samej puli.
Oba ramiona z tą samą regułą różnorodności: bez klatek bliżej niż ±30 klatek tego samego filmu od klatek bazy
i już dodanych (diagnoza poprzedniej pętli: wybór skupiał się na sąsiednich, prawie identycznych klatkach).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

import dane as D

OKNO, TOP_K = 30, 90


def _vf(fid):
    v, f = fid.split('/')[-2:]
    return v, int(f)


class Zajete:
    def __init__(self, fids):
        self.m = {}
        for f in fids:
            self.dodaj(f)

    def dodaj(self, fid):
        v, f = _vf(fid); self.m.setdefault(v, []).append(f)

    def blisko(self, fid):
        v, f = _vf(fid)
        return any(abs(f - g) <= OKNO for g in self.m.get(v, ()))


def zapytaj(png, wykluczone, out_json):
    sys.path.insert(0, str(D.PETLA))
    from wyszukiwanie import query_index
    import json
    if Path(out_json).exists():
        return json.loads(Path(out_json).read_text(encoding='utf-8'))['candidates']
    return query_index(D.INDEKS, png, sorted(wykluczone), out=out_json, top_k=TOP_K, device='cuda')['candidates']


def koopman(listy, pula_byid, zajete, n):
    wyb, poz = [], [0] * len(listy)
    while len(wyb) < n and any(p < len(l) for p, l in zip(poz, listy)):
        for i, l in enumerate(listy):
            while poz[i] < len(l):
                c = l[poz[i]]; poz[i] += 1
                if c['frame_id'] in pula_byid and not zajete.blisko(c['frame_id']):
                    wyb.append(dict(frame_id=c['frame_id'], kotwica=i, podobienstwo=float(c['score']), pozycja=poz[i] - 1))
                    zajete.dodaj(c['frame_id'])
                    break
            if len(wyb) >= n:
                break
    return wyb


def losowe(pula_wiersze, zajete, n, rng):
    wyb = []
    for j in rng.permutation(len(pula_wiersze)):
        f = pula_wiersze[j]['frame_id']
        if not zajete.blisko(f):
            wyb.append(dict(frame_id=f)); zajete.dodaj(f)
            if len(wyb) >= n:
                break
    return wyb
