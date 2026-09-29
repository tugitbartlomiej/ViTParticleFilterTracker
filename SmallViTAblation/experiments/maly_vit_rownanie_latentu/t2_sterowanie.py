#!/usr/bin/env python3
"""T2: równanie Langevina latentu sond ze sterowaniem danymi (PLAN.md) — krok = jedna aktualizacja wag.

Co rozstrzyga: czy dołożenie wpływu paczki danych do równania z T1 podnosi umiejętność jednego kroku powyżej 80.
Obali: umiejętność ≤ 80 albo taki sam zysk z placebo (siła z innego kroku).

Równanie (każda z 768 współrzędnych osobno, współczynniki wspólne dla sond, dopasowanie kroki 1000–2000, test
2000–3000, nagranie `nagranie_s47_sily`):
  Δz_{t+1} = D + JP + N            (T1: dryf do średnich ruchomych, prędkość, przyspieszenie, tłumione obracające się
                                     jądra na prędkości, szum barwny z reszt)
           + b_d · S^nowa_t        (sterowanie: siła nowej paczki — zmiana latentu sondy, gdy wagi przesunąć tylko
                                     o część kroku AdamW pochodzącą z gradientu tej paczki)
           + c_d · S^ped_t         (stan optymalizatora: pęd, czyli pamięć poprzednich paczek)
           + e_d · S^zanik_t       (zanik wag)
Warianty w wynikach:
  t1                 bez sterowania (podłoga)
  t1+paczka          + S^nowa (sterowanie danymi)
  t1+paczka_placebo  + S^nowa z kroku przesuniętego o 250 (podłoga placebo)
  t1+jadro           + sterowanie bez przeliczania sieci: Σ_b cos(h_p, h_b) g_b i jego średnia ruchoma 0,9
                     (h = wejście fc1 ostatniego bloku, g = gradient straty względem latentu obrazu paczki)
  t1+paczka+ped      + S^nowa + S^ped
  t1+wszystko        + S^nowa + S^ped + S^zanik
  same_sily          tylko b·S^nowa + c·S^ped + e·S^zanik (bez historii latentu)

Dodatkowo `rownanie_proste`: Δz_{t+1} = a·Δz_t + b·S^nowa_t (dwa współczynniki wspólne dla wszystkiego), `historia_av`
(samo a·Δz_t) oraz statystyka ruchu nagrania (norma i korelacja przyrostów co 1 i co 10 kroków, jak test T0).

    python t2_sterowanie.py [--katalog nagranie_s47_sily --ziarno 47 --fit 1000 2000 --test 2000 2999 --wynik wyniki/T2.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dane as DN
import t1_bez_sterowania as T1

FIT, TEST, OD = (1000, 2000), (2000, 2999), 900


def proste(S, Fn, ob, fit, test):
    """Równanie proste i statystyka ruchu (wszystkie współrzędne naraz, float32)."""
    Z = lambda i: S[i][:, ob, 0].astype(np.float32)
    def xy(t):
        return Z(t + 1) - Z(t), Z(t) - Z(t - 1), Fn[t - OD][:, ob].astype(np.float32)
    yf, vf, ff = xy(fit); yt, vt, ft = xy(test)
    X = np.stack([vf.reshape(-1), ff.reshape(-1)], 1).astype(np.float64)
    w = np.linalg.lstsq(X, yf.reshape(-1).astype(np.float64), rcond=None)[0]
    a = float((vf * yf).sum() / (vf * vf).sum())
    e = ((yt - (w[0] * vt + w[1] * ft)) ** 2).sum(-1); s0 = (yt ** 2).sum(-1)
    wyn = dict(rownanie_proste=round(DN.umiejetnosc(yt, w[0] * vt + w[1] * ft), 2), a=round(float(w[0]), 4), b=round(float(w[1]), 4),
               historia_av=round(DN.umiejetnosc(yt, a * vt), 2), a_samo=round(a, 4),
               mediana_po_krokach=round(float(np.median(100 * (1 - e.sum(1) / s0.sum(1)))), 2),
               p10_po_krokach=round(float(np.percentile(100 * (1 - e.sum(1) / s0.sum(1)), 10)), 2),
               mediana_po_sondach=round(float(np.median(100 * (1 - e.sum(0) / s0.sum(0)))), 2),
               najgorsza_sonda=round(float((100 * (1 - e.sum(0) / s0.sum(0))).min()), 2))
    for co in (1, 10):
        Zs = S[1000::co][:, ob, 0].astype(np.float32); D = np.diff(Zs, axis=0)
        wyn[f'ruch_co_{co}'] = dict(norma=round(float(np.linalg.norm(D, axis=-1).mean()), 4),
                                   korelacja=round(float((D[1:] * D[:-1]).sum() / np.sqrt((D[1:] ** 2).sum() * (D[:-1] ** 2).sum())), 4))
    return wyn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--katalog', default=str(DN.HERE / 'nagranie_s47_sily'))
    ap.add_argument('--ziarno', type=int, default=47)
    ap.add_argument('--fit', type=int, nargs=2, default=FIT)
    ap.add_argument('--test', type=int, nargs=2, default=TEST)
    ap.add_argument('--wynik', default=str(DN.HERE / 'wyniki' / 'T2.json'))
    ar = ap.parse_args(argv)
    t0 = time.time()
    kat = Path(ar.katalog)
    N = {p.stem: np.load(p, mmap_mode='r') for p in kat.glob('*.npy')}
    meta = json.loads((kat / 'META.json').read_text(encoding='utf-8'))
    ob = np.array(meta['obecn'])
    S = N['sondy_lat']
    T = S.shape[0]
    fit = np.arange(*ar.fit); test = np.arange(*ar.test)
    pr = proste(S, N['sila_nowa'], ob, fit, test)
    print('proste', pr, flush=True)
    # sterowanie jądrem (bez przeliczania sieci), liczone raz dla wszystkich współrzędnych
    Hs = N['sondy_h'][:T - 1, :, 0].astype(np.float32)[:, ob]                      # (T-1, P, 192), stan t
    Hb = N['paczka_h'][:, :, 0].astype(np.float32)                                # (T-1, 32, 192), paczka t
    Gb = N['paczka_g'][:, :, 0].astype(np.float32)                                # (T-1, 32, 768)
    ma = N['paczka_ma_narz'].astype(np.float32)
    nrm = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
    kap = np.einsum('tpk,tbk->tpb', nrm(Hs), nrm(Hb)) * ma[:, None, :]            # (T-1, P, 32)
    U = np.einsum('tpb,tbd->tpd', kap, Gb)                                        # (T-1, P, 768)
    Ue = np.zeros_like(U); e = np.zeros(U.shape[1:], np.float32)
    for t in range(len(U)):
        e = 0.9 * e + U[t]; Ue[t] = e
    print(f'jądro gotowe {time.time() - t0:.0f} s', flush=True)
    Dm = S.shape[-1]; kaw = 8
    sse = {}; ss = 0.0
    def dodaj(k, v):
        sse[k] = sse.get(k, 0.0) + v
    for c0 in range(0, Dm, kaw):
        Z = S[:, :, 0, c0:c0 + kaw].astype(np.float32)[:, ob]                    # (T, P, c)
        Y = np.zeros_like(Z); Y[:-1] = Z[1:] - Z[:-1]
        F1 = T1.cechy(Z)
        R = np.zeros_like(Z)
        W1 = T1.dopasuj_diag(F1, Y, fit)
        P1 = np.einsum('tpcf,cf->tpc', F1, W1); R[1:] = Y[:-1] - P1[:-1]
        F1 = np.concatenate([F1, np.stack(T1.filtry(R), -1)], -1)             # T1 z członem N
        def sila(k, przes=0):
            A = np.zeros_like(Z)
            src = N[f'sila_{k}'][:, ob, c0:c0 + kaw].astype(np.float32)          # indeks i = t − OD
            idx = np.arange(OD, T - 1)
            j = idx - OD + przes
            ok = (j >= 0) & (j < len(src))
            A[idx[ok]] = src[j[ok]]
            return A[..., None]
        Sn, Sp, Sz = sila('nowa'), sila('ped'), sila('zanik')
        Snp = sila('nowa', 250)
        Uk = np.zeros_like(Z); Uk[:-1] = U[:, :, c0:c0 + kaw]; Uke = np.zeros_like(Z); Uke[:-1] = Ue[:, :, c0:c0 + kaw]
        cat = lambda *a: np.concatenate(a, -1)
        warianty = {
            't1': lambda: F1,
            't1+paczka': lambda: cat(F1, Sn),
            't1+paczka_placebo': lambda: cat(F1, Snp),
            't1+jadro': lambda: cat(F1, Uk[..., None], Uke[..., None]),
            't1+paczka+ped': lambda: cat(F1, Sn, Sp),
            't1+wszystko': lambda: cat(F1, Sn, Sp, Sz),
            'same_sily': lambda: cat(Sn, Sp, Sz),
        }
        yt = Y[test]; ss += float((yt.astype(np.float64) ** 2).sum())
        for k, bud in warianty.items():
            F = bud()
            W = T1.dopasuj_diag(F, Y, fit)
            dodaj(k, float(((yt - np.einsum('tpcf,cf->tpc', F[test], W)) ** 2).sum()))
            if k == 'same_sily' and c0 == 0:
                print('  współczynniki same_sily (pierwsze 8 współrzędnych):', np.round(W, 2).tolist(), flush=True)
        if c0 % 128 == 0:
            print(f'{c0 + kaw}/{Dm} współrzędnych, {time.time() - t0:.0f} s: ' +
                  ', '.join(f'{k} {100 * (1 - v / ss):.1f}' for k, v in sse.items()), flush=True)
    wyn = {k: round(100 * (1 - v / ss), 2) for k, v in sse.items()}
    print('T2', wyn, flush=True)
    Path(ar.wynik).parent.mkdir(parents=True, exist_ok=True)
    Path(ar.wynik).write_text(json.dumps(dict(ziarno=ar.ziarno, umiejetnosc_test=wyn, **pr, dopasowanie=list(ar.fit),
                                              test=list(ar.test), nagranie=str(kat), sekundy=round(time.time() - t0)),
                                         indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
