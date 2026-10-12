"""Kręgosłup trajektorii wag: udział pierwszego kierunku głównego, trwałość dryfu, netto do drogi (PLAN.md).

Dane: wagi co 10 kroków i latent sond z biegu gęstej historii (`../../maly_vit_ruch_czastek_gesta_historia`).
Bez treningu, bez GPU. Wszystko z macierzy iloczynów skalarnych przesunięć (101 x 101), osobno dla grup wag.

    py -3.11 kregoslup.py --wyniki ../wyniki        (kilka minut, około 2,5 GB RAM)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

BIEG = Path(__file__).resolve().parents[2] / 'maly_vit_ruch_czastek_gesta_historia' / 'wyniki' / 'pelny'
KROKI = list(range(0, 1001, 10))
OPOZNIENIA = (1, 2, 3, 5, 10)
OKNA = (5, 10)                      # w zapisach co 10 kroków: 50 i 100 kroków
ODNIESIENIA = (0, 30)               # indeks zapisu: krok 0 i krok 300


def grupa(nazwa):
    c = nazwa.split('.')
    return '.'.join(c[:2]) if c[0] == 'bloki' else c[0]


def wczytaj_wagi():
    W, zakresy = None, {}
    for i, k in enumerate(KROKI):
        net = torch.load(BIEG / 'wagi' / f'k{k:04d}.pt', map_location='cpu', weights_only=False)['net']
        czesci, poz = [], 0
        for nazwa, v in net.items():
            if not torch.is_floating_point(v) or 'running_' in nazwa:
                continue
            a = v.reshape(-1).float().numpy()
            if i == 0:
                g = grupa(nazwa); z = zakresy.setdefault(g, [poz, poz]); z[1] = poz + a.size
            czesci.append(a); poz += a.size
        if W is None:
            W = np.empty((len(KROKI), poz), dtype=np.float32)
        W[i] = np.concatenate(czesci)
    return W, zakresy


def gram(Y, kolumny=None, kawalek=200_000):
    """K[a,b] = <Y_a, Y_b> w podwójnej precyzji, po kawałkach kolumn."""
    a, b = (0, Y.shape[1]) if kolumny is None else kolumny
    K = np.zeros((Y.shape[0], Y.shape[0]))
    for s in range(a, b, kawalek):
        c = Y[:, s:min(s + kawalek, b)].astype(np.float64)
        K += c @ c.T
    return K


def miary(K, i0):
    """Miary trajektorii z macierzy iloczynów przesunięć (dowolny stan odniesienia), od zapisu i0."""
    T = K.shape[0]
    Kr = K - K[:, [i0]] - K[[i0], :] + K[i0, i0]                    # przesunięcia względem stanu i0
    S = Kr[i0 + 1:, i0 + 1:]
    w, U = np.linalg.eigh(S); w, U = np.clip(w[::-1], 0, None), U[:, ::-1]
    udz = w / w.sum()
    a = np.sqrt(w[0]) * U[:, 0]; a = a if a[-1] > 0 else -a           # współrzędna postępu w zapisach i0+1..T-1
    da = np.diff(np.concatenate([[0.0], a]))

    def il(p, q, r, s):                                               # <θ_p − θ_q, θ_r − θ_s>
        return K[p, r] - K[p, s] - K[q, r] + K[q, s]

    kroki = [(i + 1, i) for i in range(i0, T - 1)]
    n2 = np.array([il(p, q, p, q) for p, q in kroki])
    droga = float(np.sqrt(np.clip(n2, 0, None)).sum()); netto = float(np.sqrt(max(il(T - 1, i0, T - 1, i0), 0)))
    acf = {}
    for k in OPOZNIENIA:
        licz = sum(il(*kroki[i], *kroki[i + k]) for i in range(len(kroki) - k))
        acf[str(10 * k)] = float(licz / n2[:len(kroki) - k].sum())
    okna = {}
    for wn in OKNA:
        cs = []
        for i in range(i0 + wn, T - wn):
            l = il(i + wn, i, i, i - wn); m = np.sqrt(il(i + wn, i, i + wn, i) * il(i, i - wn, i, i - wn))
            cs.append(l / m)
        cs = np.array(cs)
        okna[str(10 * wn)] = dict(mediana=float(np.median(cs)), q25=float(np.percentile(cs, 25)), q75=float(np.percentile(cs, 75)),
                                  min=float(cs.min()), max=float(cs.max()), n=int(len(cs)))
    return dict(udzial_pc1=float(udz[0]), udzial_pc2=float(udz[1]), udzial_pc3=float(udz[2]), udzial_5=float(udz[:5].sum()),
                kierunkow_do_90=int(np.searchsorted(np.cumsum(udz), 0.9) + 1), netto_do_drogi=netto / droga,
                norma_przesuniecia=netto, autokorelacja_przyrostow=acf, cosinus_sasiednich_okien=okna,
                postep_udzial_dodatnich=float((da > 0).mean()), postep=a.tolist())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    W, zakresy = wczytaj_wagi()
    print('wagi', W.shape, {g: z[1] - z[0] for g, z in zakresy.items()}, f'{time.time() - t0:.0f} s', flush=True)
    norma0 = float(np.sqrt(gram(W[:1]).item())); norma_k = float(np.sqrt(gram(W[-1:]).item()))
    W -= W[0].copy()                                                   # przesunięcia względem kroku 0 (w miejscu)
    Kg = {g: gram(W, tuple(z)) for g, z in zakresy.items()}
    Kg['cala_siec'] = sum(Kg.values())
    del W
    wyn = dict(meta=dict(zrodlo=str(BIEG), kroki=[KROKI[0], KROKI[-1], 10], odniesienia=[KROKI[i] for i in ODNIESIENIA],
                         parametry={g: z[1] - z[0] for g, z in zakresy.items()}, norma_wag_krok0=norma0, norma_wag_krok1000=norma_k,
                         opis='udzialy: wartosci wlasne przesuniec wzgledem stanu odniesienia, bez centrowania'))
    wyn['wagi'] = {g: {str(KROKI[i0]): miary(K, i0) for i0 in ODNIESIENIA} for g, K in Kg.items()}
    np.savez_compressed(out / 'GRAMY_WAG.npz', **{g.replace('.', '_'): K for g, K in Kg.items()})
    z = np.load(BIEG / 'dane' / 'LATENT_SOND.npz')
    assert list(z['kroki']) == KROKI
    ob = np.array([i % 16 != 8 for i in range(192)])                   # 180 sond z narzędziem (bez wersji empty_black)
    for nazwa, kan in (('latent_narzedzie', 0), ('latent_tlo', 1)):
        X = z['latent'][:, ob, kan, :].astype(np.float32).reshape(len(KROKI), -1)
        X -= X[0].copy()
        K = gram(X)
        wyn[nazwa] = {str(KROKI[i0]): miary(K, i0) for i0 in ODNIESIENIA}
    wyn['meta']['czas_s'] = time.time() - t0
    (out / 'KREGOSLUP.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    for g in ('cala_siec', 'lodyga', 'bloki.0', 'bloki.5', 'glowa'):
        m = wyn['wagi'][g]['300']
        print(g, 'PC1', round(m['udzial_pc1'], 3), 'do90', m['kierunkow_do_90'], 'okno100', round(m['cosinus_sasiednich_okien']['100']['mediana'], 3),
              'netto/droga', round(m['netto_do_drogi'], 3), 'postep+', round(m['postep_udzial_dodatnich'], 2))
    for g in ('latent_narzedzie', 'latent_tlo'):
        m = wyn[g]['300']
        print(g, 'PC1', round(m['udzial_pc1'], 3), 'do90', m['kierunkow_do_90'], 'okno100', round(m['cosinus_sasiednich_okien']['100']['mediana'], 3),
              'netto/droga', round(m['netto_do_drogi'], 3), 'postep+', round(m['postep_udzial_dodatnich'], 2))
    print('zapisano', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
