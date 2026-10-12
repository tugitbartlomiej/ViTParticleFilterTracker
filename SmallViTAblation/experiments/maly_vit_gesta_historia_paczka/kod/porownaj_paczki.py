"""Porównanie przewidywalności latentu dla paczek 32, 128, 320 na TYCH SAMYCH 60 tokenach na obraz (PLAN.md).

Rozstrzyga: czy prognoza z samej historii latentu (równanie z jądrem pamięci J=5 i ściąganiem do średniej,
dopasowane wprost na każdy horyzont) rośnie z rozmiarem paczki i czy przy 320 przekracza 0,85 na 10 kroków.
Obala: wynik przy 320 na h=10 poniżej 0,5.
Dla każdej paczki: prognoza bez zmian (odniesienie), równanie J=5 (wprost na h; współczynniki wspólne),
OU (ściąganie do średniej z 51 stanów), sufit wyroczni (średnia ruchoma 21) i podłoga (wyrocznia na samych
drganiach). Dopasowanie: cele do kroku 700, obrazy 0-5 / 6-11 na krzyż. Test: starty 700..890 (h<=10),
700..900 co 20 (h=100). Dodatkowo stosunek dryfu do drgań D²/σ² i autokorelacja przyrostów (lag 1 i 5).
Paczka 32: te same tokeny wycięte z gęstej historii biegu 1818732 (TOKENY_WYBRANE.json biegu 320).
Uruchomienie: py -3.11 porownaj_paczki.py --wyniki ../wyniki/porownanie
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
GESTA = EXP.parent / 'maly_vit_ruch_czastek_gesta_historia' / 'wyniki' / 'pelny' / 'dane' / 'LATENT_GESTY.npy'
H10 = list(range(1, 11)); H = H10 + [100]
J, K_MU, W_SUF = 5, 51, 21
T_FIT_MAX = 700
STARTY_FIT = range(200, 700, 1)
STARTY_TEST10 = range(700, 891, 1)
STARTY_TEST100 = range(700, 901, 20)
LAM = 1e-4
KAPPA = np.concatenate([[0.0], np.logspace(-4, 0.5, 300)])


N_TLO = 20              # losowych tokenów tła na obraz do analizy (z 60 zapisanych); oszczędność RAM


def wybor_jednostek(sel):
    """Z 60 zapisanych tokenów na obraz: wszystkie narzędzia plus N_TLO pierwszych tła (zapis jest już losowy)."""
    idx, narz, obraz = [], [], []
    for i, o in enumerate(sel['obrazy']):
        n = np.array(o['narzedzie']); k = np.flatnonzero(n).tolist() + np.flatnonzero(~n)[:N_TLO].tolist()
        idx.append(np.array(k)); narz += [bool(n[j]) for j in k]; obraz += [i] * len(k)
    return idx, np.array(narz), np.array(obraz)


def wczytaj_bieg(katalog):
    d = Path(katalog) / 'dane'
    Z = np.load(d / 'LATENT_GESTY.npy', mmap_mode='r')
    sel = json.loads((d / 'TOKENY_WYBRANE.json').read_text(encoding='utf-8'))
    meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8'))
    idx, narz, obraz = wybor_jednostek(sel)
    Y = np.concatenate([np.asarray(Z[:, i, idx[i], :]) for i in range(12)], 1)      # (T, ~458, 768)
    return Y, narz, obraz, meta, sel


def wczytaj_p32(sel):
    Z = np.load(GESTA, mmap_mode='r')
    idx, narz, obraz = wybor_jednostek(sel)
    Y = np.concatenate([np.asarray(Z[:, i, np.array(o['tokeny'])[idx[i]], :]) for i, o in enumerate(sel['obrazy'])], 1)
    return Y, narz, obraz


def srednia_ruchoma(Y, w, centralna):
    """Średnia z w stanów: centralna (wokół t, NaN przy brzegach) albo z ostatnich w stanów (do t włącznie).
    Liczona kawałkami po jednostkach, bez pełnej tablicy float64."""
    T = Y.shape[0]; r = w // 2
    M = np.full(Y.shape, np.nan, np.float32)
    for u0 in range(0, Y.shape[1], 32):
        cs = np.zeros((T + 1,) + Y[:, u0:u0 + 32].shape[1:]); np.cumsum(Y[:, u0:u0 + 32], 0, out=cs[1:])
        if centralna:
            M[r:T - r, u0:u0 + 32] = ((cs[w:] - cs[:-w]) / w)[: T - 2 * r]
        else:
            M[w - 1:, u0:u0 + 32] = (cs[w:] - cs[:-w]) / w
    return M


def cechy(y, MU, t):
    return np.stack([y[t] - MU[t]] + [y[t - j] - y[t - j - 1] for j in range(J)], 1)


def dopasuj_wprost(y, MU, maska):
    F = 1 + J; G = np.zeros((F, F)); b = np.zeros((F, len(H)))
    for t in STARTY_FIT:
        X = cechy(y, MU, t)[maska]; G += np.einsum('ufd,ugd->fg', X, X)
        for i, h in enumerate(H):
            if t + h > T_FIT_MAX: continue
            b[:, i] += np.einsum('ufd,ud->f', X, (y[t + h] - y[t])[maska])
    return np.linalg.solve(G + LAM * np.trace(G) * np.eye(F), b)


def dopasuj_ou(y, MU, maska):
    A = np.zeros(len(H)); B = np.zeros(len(H)); C = np.zeros(len(H))
    for t in STARTY_FIT:
        d = (MU[t] - y[t])[maska]
        bb = np.einsum('ud,ud->', d, d)
        for i, h in enumerate(H):
            if t + h > T_FIT_MAX: continue
            dl = (y[t + h] - y[t])[maska]
            A[i] += np.einsum('ud,ud->', dl, d); B[i] += bb; C[i] += np.einsum('ud,ud->', dl, dl)
    al = 1 - np.exp(-np.outer(KAPPA, np.array(H)))
    loss = (C - 2 * al * A + al ** 2 * B).sum(1)
    return float(KAPPA[int(np.argmin(loss))])


def analizuj(Y, maska_grup):
    M = srednia_ruchoma(Y, W_SUF, True); MU = srednia_ruchoma(Y, K_MU, False)
    E = Y - M
    sse = {m: np.zeros(len(H)) for m in ('bez_zmian', 'rownanie_J5', 'ou', 'sufit', 'podloga', 'podloga_bez_zmian')}
    D2 = np.zeros(len(H)); S2 = np.zeros(len(H)); n = np.zeros(len(H))
    for fit, test in maska_grup:
        beta = dopasuj_wprost(Y, MU, fit); kappa = dopasuj_ou(Y, MU, fit)
        for i, h in enumerate(H):
            starty = STARTY_TEST10 if h <= 10 else STARTY_TEST100
            for t in starty:
                X = cechy(Y, MU, t)[test]; xt = Y[t][test]; prawda = Y[t + h][test]
                e = lambda p: np.einsum('ud,ud->', p - prawda, p - prawda)
                sse['bez_zmian'][i] += e(xt)
                sse['rownanie_J5'][i] += e(xt + np.einsum('ufd,f->ud', X, beta[:, i]))
                mu = MU[t][test]
                sse['ou'][i] += e(mu + (xt - mu) * np.exp(-kappa * h))
                if np.isfinite(M[t + h]).all() and np.isfinite(M[t]).all():
                    sse['sufit'][i] += e(M[t + h][test])
                    sse['podloga'][i] += np.einsum('ud,ud->', E[t + h][test], E[t + h][test])
                    sse['podloga_bez_zmian'][i] += np.einsum('ud,ud->', E[t + h][test] - E[t][test], E[t + h][test] - E[t][test])
                    dd = (M[t + h] - M[t])[test]; D2[i] += np.einsum('ud,ud->', dd, dd)
                    S2[i] += np.einsum('ud,ud->', E[t][test], E[t][test]); n[i] += 1
    wyn = {m: (1 - sse[m] / sse['bez_zmian']).tolist() for m in ('rownanie_J5', 'ou', 'sufit')}
    wyn['podloga'] = (1 - sse['podloga'] / sse['podloga_bez_zmian']).tolist()
    wyn['D2_do_sigma2'] = (D2 / S2).tolist()
    dY = Y[1:] - Y[:-1]
    r0 = np.einsum('tud,tud->', dY[300:], dY[300:])
    wyn['autokorelacja_przyrostow'] = {str(l): float(np.einsum('tud,tud->', dY[300:-l], dY[300 + l:]) / r0) for l in (1, 2, 3, 5, 10)}
    wyn['sredni_krok_wzgledny'] = float(np.sqrt((dY[300:] ** 2).sum((1, 2)) / (Y[300:-1] ** 2).sum((1, 2))).mean())
    return wyn


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True)
    p.add_argument('--biegi', nargs='*', default=['pelny_p128', 'pelny_p320']); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); wyn = dict(meta=dict(horyzonty=H, J=J, k_mu=K_MU, w_sufit=W_SUF, fit_cele_do=T_FIT_MAX,
                                           starty_test10=[700, 890], starty_test100=list(STARTY_TEST100), lam=LAM),
                                 paczki={})
    sel0 = None
    for bieg in arg.biegi:
        kat = EXP / 'wyniki' / bieg
        if not (kat / 'dane' / 'LATENT_GESTY.npy').exists():
            print('brak', kat, flush=True); continue
        Y, narz, obraz, meta, sel = wczytaj_bieg(kat); sel0 = sel0 or sel
        grupy = [(obraz < 6, obraz >= 6), (obraz >= 6, obraz < 6)]
        wyn['paczki'][str(meta['hyper']['paczka'])] = dict(bieg=bieg, slurm=meta.get('slurm_job'),
                                                          mAP50_koniec=meta.get('mAP50_koniec'),
                                                          wszystkie=analizuj(Y, grupy),
                                                          narzedzie=analizuj(Y, [(f & narz, t & narz) for f, t in grupy]))
        print(bieg, 'gotowe', f'{time.time() - t0:.0f} s', {k: round(v[9], 3) for k, v in wyn['paczki'][str(meta['hyper']['paczka'])]['wszystkie'].items() if isinstance(v, list)}, flush=True)
        del Y
    if sel0 is not None and GESTA.exists():
        Y, narz, obraz = wczytaj_p32(sel0)
        grupy = [(obraz < 6, obraz >= 6), (obraz >= 6, obraz < 6)]
        wyn['paczki']['32'] = dict(bieg='maly_vit_ruch_czastek_gesta_historia/wyniki/pelny (1818732), te same tokeny',
                                   wszystkie=analizuj(Y, grupy), narzedzie=analizuj(Y, [(f & narz, t & narz) for f, t in grupy]))
        print('p32 gotowe', f'{time.time() - t0:.0f} s', flush=True)
    wyn['meta']['czas_s'] = time.time() - t0
    (out / 'POROWNANIE.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    for pk in sorted(wyn['paczki'], key=int):
        w = wyn['paczki'][pk]['wszystkie']
        print(f"paczka {pk:>3}: h1 {w['rownanie_J5'][0]:.3f}  h10 {w['rownanie_J5'][9]:.3f}  h100 {w['rownanie_J5'][10]:.3f} "
              f"| OU h10 {w['ou'][9]:.3f} | sufit h10 {w['sufit'][9]:.3f} h100 {w['sufit'][10]:.3f} "
              f"| D2/s2 h10 {w['D2_do_sigma2'][9]:.2f} | acf1 {w['autokorelacja_przyrostow']['1']:.2f}")
    print('zapisano', out / 'POROWNANIE.json')


if __name__ == '__main__':
    main()
