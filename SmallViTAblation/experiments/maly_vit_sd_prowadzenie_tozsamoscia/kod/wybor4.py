"""Wybór klatek w wersji 4 (PLAN.md): grupy kandydatów z list wyszukiwania i z klatek SD, ocena równaniem
(wirtualny krok), bezpiecznik i wybór całego zestawu z różnorodnością.

Grupa = do ROZMIAR klatek: prawdziwe klatki z jednej listy wyszukiwania (klatki podobne do jednego zapytania) albo
kolejne klatki SD. Bezpiecznik (T6 w dokumencie 44, wersja praktyczna): grupa wchodzi tylko wtedy, gdy jej przewidziany
zysk po odjęciu szumu prognozy przekracza medianę zysku grup losowych powiększoną o ich szum; resztę zestawu wypełnia
losowanie. Różnorodność: reguła ±30 klatek filmu (jak w wersjach 2-3) i limit klatek z jednego filmu.
"""
from __future__ import annotations

import numpy as np

from wybor import Zajete, _vf  # noqa: F401  (Zajete używane przez petla4)

ROZMIAR, NA_FILM = 8, 20


def grupy_z_list(listy, pula_id, zajete, uzyte, rozmiar=ROZMIAR, na_liste=1):
    """Z każdej listy (kandydaci wyszukiwania po kolei) do `na_liste` grup po `rozmiar` klatek z puli, nie bliżej niż
    ±30 od zajętych i nieużytych w innych grupach. Zwraca listę dict(fids, lista)."""
    out = []
    for i, l in enumerate(listy):
        gr, n_gr = [], 0
        for c in l:
            f = c['frame_id']
            if f not in pula_id or f in uzyte or zajete.blisko(f):
                continue
            v, k = _vf(f)
            if any(_vf(g)[0] == v and abs(_vf(g)[1] - k) <= 30 for g in gr):
                continue
            gr.append(f); uzyte.add(f)
            if len(gr) == rozmiar:
                out.append(dict(fids=gr, lista=i, syntetyczne=False)); gr = []; n_gr += 1
                if n_gr >= na_liste:
                    break
        if gr and n_gr < na_liste and len(gr) >= rozmiar // 2:
            out.append(dict(fids=gr, lista=i, syntetyczne=False))
    return out


def grupy_losowe(pula_w, zajete, uzyte, n_grup, rng, rozmiar=ROZMIAR):
    out = []
    for j in rng.permutation(len(pula_w)):
        f = pula_w[int(j)]['frame_id']
        if f in uzyte or zajete.blisko(f):
            continue
        if not out or len(out[-1]['fids']) == rozmiar:
            if len(out) == n_grup:
                break
            out.append(dict(fids=[], lista=-1, syntetyczne=False, losowa=True))
        out[-1]['fids'].append(f); uzyte.add(f)
    return [g for g in out if len(g['fids']) == rozmiar]


def prog_bezpiecznika(losowe):
    z = np.array([g['zysk'] for g in losowe]); s = np.array([g['szum'] for g in losowe])
    return float(np.median(z) + np.median(s)) if len(z) else -np.inf


def wybierz(grupy, zajete, n, max_synt, rng, wolne_pula, prog=-np.inf, kolejnosc='zysk'):
    """Zestaw n klatek. kolejnosc='zysk': grupy od największego przewidzianego zysku, tylko ponad progiem
    bezpiecznika; 'losowa': grupy w losowej kolejności, bez progu (ramię `sd_losowo`). Klatki SD do max_synt.
    Resztę wypełnia losowanie z wolne_pula. Zwraca (wybrane, raport)."""
    if kolejnosc == 'zysk':
        kolej = sorted([g for g in grupy if g['zysk'] - g['szum'] > prog], key=lambda g: -g['zysk'])
    else:
        kolej = [grupy[int(i)] for i in rng.permutation(len(grupy))]
    wyb, na_film, synt = [], {}, 0
    for g in kolej:
        for f in g['fids']:
            if len(wyb) >= n:
                break
            if g['syntetyczne']:
                if synt >= max_synt:
                    break
                wyb.append(dict(frame_id=f, zrodlo='sd', grupa=g.get('id'), zysk=g.get('zysk'))); synt += 1
                continue
            v, _ = _vf(f)
            if zajete.blisko(f) or na_film.get(v, 0) >= NA_FILM:
                continue
            wyb.append(dict(frame_id=f, zrodlo='pula', grupa=g.get('id'), zysk=g.get('zysk')))
            zajete.dodaj(f); na_film[v] = na_film.get(v, 0) + 1
        if len(wyb) >= n:
            break
    z_grup = len(wyb)
    for j in rng.permutation(len(wolne_pula)):
        if len(wyb) >= n:
            break
        f = wolne_pula[int(j)]['frame_id']; v, _ = _vf(f)
        if zajete.blisko(f) or na_film.get(v, 0) >= NA_FILM:
            continue
        wyb.append(dict(frame_id=f, zrodlo='pula_losowo', grupa=None, zysk=None))
        zajete.dodaj(f); na_film[v] = na_film.get(v, 0) + 1
    rap = dict(grup=len(grupy), grup_ponad_progiem=len(kolej) if kolejnosc == 'zysk' else None, prog=prog,
               z_grup=z_grup, syntetycznych=synt, losowo_dopelnione=len(wyb) - z_grup)
    return wyb, rap
