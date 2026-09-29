#!/usr/bin/env python3
"""Na ile kroków naprzód Koopman i Langevin przewidują siłę tożsamości narzędzia (pytanie użytkownika 2026-09-29).

Dane: zapisane historie siły tożsamości `SILA.npy` (zapis co 10 kroków, 180 sond z narzędziem) z pętli
`maly_vit_petla_sila_tozsamosci` — wersja 3 (10 ziaren, 7 obrotów, krok 4500) i iteracja 2 (20 ziaren, 14 obrotów,
krok 8000), wszystkie ramiona. Bez uczenia sieci, tylko numpy.

Bloki po 50 kroków (jak w pętli). Punkt startu prognozy = początek każdego obrotu (co 500 kroków od kroku 1000).
Dopasowanie tylko na historii do punktu startu. Horyzont H = 1..20 bloków (50..1000 kroków).

Modele (osobno):
  koopman     rekurencyjnie, dokładnie jak w pętli (`prognoza.py`): ΔY_{t+1} = Σ_{j<4} A_j ΔY_{t−j}
  langevin    rekurencyjnie, jak w pętli: ΔY_{t+1} = K Y_t + Σ_{j<4} B_j ΔY_{t−j} + a_p
  *_wprost    te same cechy, ale osobne dopasowanie dla każdego H: Y_{t+H} − Y_t = f(cechy w t)
              (prognoza wprost na horyzont, bez składania kroków; Lin i in. 2021 pokazują, że składanie wielu
              małych kroków Koopmana mnoży błąd)
Odniesienia: brak zmiany (Y_{t+H} = Y_t) i średnia zmiana (ostatnie 10 bloków przedłużone liniowo).

Miara: umiejętność = 100 · (1 − SSE_model / SSE_brak_zmiany), sumy po sondach i punktach startu danego ziarna.
> 0 znaczy lepiej niż brak zmiany. Wynik: mediana po ziarnach i odsetek ziaren z umiejętnością > 0.

    python horyzont.py  ->  wyniki/HORYZONT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PETLA = HERE.parent / 'maly_vit_petla_sila_tozsamosci'
sys.path.insert(0, str(PETLA / 'kod'))
import prognoza as PR                                                   # noqa: E402  (ten sam kod co na Edenie)

HMAX, L, LAM = 20, PR.L_PAM, PR.LAM
ZBIORY = {'wersja3': PETLA / 'wyniki_eden', 'iteracja2': PETLA / 'wyniki_eden_it2'}
RAMIONA = ('bez', 'losowe', 'kolejne', 'tozsamosc', 'current')
MODELE = ('koopman', 'langevin', 'koopman_wprost', 'langevin_wprost', 'srednia_zmiana')


def cechy(Y, t, model):
    """Cechy w chwili t dla wszystkich sond, jak `prognoza._cechy`: Y (T, P, 1)."""
    D = np.diff(Y, axis=0)
    c = [D[t - 1 - j] for j in range(L)]
    if model.startswith('langevin'):
        c = [Y[t]] + c + [np.eye(Y.shape[1])]
    return np.concatenate(c, 1)


def grzbiet(X, y):
    A = X.T @ X
    return np.linalg.solve(A + LAM * np.diag(np.diag(A)) + 1e-9 * np.eye(len(A)), X.T @ y)


def wprost(Yh, model, hs):
    """Osobne dopasowanie Y_{t+H} − Y_t dla każdego H z hs na historii Yh; prognoza z ostatniego punktu."""
    T = len(Yh)
    F = {t: cechy(Yh, t, model) for t in range(L, T)}
    out = {}
    for H in hs:
        ts = [t for t in range(L, T - H)]
        if len(ts) < 3:
            continue
        X = np.concatenate([F[t] for t in ts]); y = np.concatenate([Yh[t + H] - Yh[t] for t in ts])
        W = grzbiet(X, y)
        out[H] = Yh[-1] + F[T - 1] @ W
    return out


def ziarno(S):
    """S: (T_zapisów, P). Zwraca dla modeli i H: SSE modelu, SSE braku zmiany (sumy po punktach startu)."""
    Y = PR.bloki(S[:, :, None].astype(np.float64))                       # (Tb, P, 1)
    Tb = len(Y)
    starty = [t0 for t0 in range(19, Tb - 1, 10) if t0 >= L + 8]
    sse = {m: np.zeros(HMAX + 1) for m in MODELE}; sse0 = np.zeros(HMAX + 1); n = np.zeros(HMAX + 1)
    for t0 in starty:
        hs = [H for H in range(1, HMAX + 1) if t0 + H < Tb]
        if not hs:
            continue
        Yh = Y[:t0 + 1]
        prog = {}
        for m in ('koopman', 'langevin'):
            R = PR.rollout(Yh, PR.dopasuj(Yh, m), max(hs), m)                # (Hmax, P, 1)
            prog[m] = {H: R[H - 1] for H in hs}
            prog[m + '_wprost'] = wprost(Yh, m, hs)
        dz = np.diff(Yh[-11:], axis=0).mean(0)
        prog['srednia_zmiana'] = {H: Yh[-1] + H * dz for H in hs}
        for H in hs:
            prawda = Y[t0 + H]
            sse0[H] += ((prawda - Yh[-1]) ** 2).sum(); n[H] += 1
            for m in MODELE:
                if H in prog[m]:
                    sse[m][H] += ((prawda - prog[m][H]) ** 2).sum()
    return sse, sse0, n


def main() -> int:
    (HERE / 'wyniki').mkdir(exist_ok=True)
    wynik = {}
    for nazwa, kat in ZBIORY.items():
        wynik[nazwa] = {}
        for ram in RAMIONA:
            pliki = sorted(kat.glob(f'petla_s*/{ram}/SILA.npy'))
            if not pliki:
                continue
            um = {m: [] for m in MODELE}; punkty = None
            for p in pliki:
                sse, sse0, n = ziarno(np.load(p))
                punkty = n
                for m in MODELE:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        um[m].append(100 * (1 - sse[m] / sse0))
            r = {}
            for m in MODELE:
                U = np.array(um[m])[:, 1:]                                   # (ziarna, HMAX)
                r[m] = dict(mediana=[None if not np.isfinite(x) else round(float(x), 1) for x in np.nanmedian(U, 0)],
                            odsetek_ziaren_lepszych=[round(float(np.mean(U[:, h] > 0)), 2) for h in range(U.shape[1])])
                dodatnie = [h + 1 for h in range(U.shape[1]) if np.isfinite(np.nanmedian(U[:, h])) and np.nanmedian(U[:, h]) > 0]
                r[m]['ostatni_H_z_dodatnia_mediana'] = max(dodatnie) if dodatnie else 0
            r['ziaren'] = len(pliki); r['punktow_startu_na_H'] = [int(x) for x in punkty[1:]]
            wynik[nazwa][ram] = r
            print(nazwa, ram, 'ziaren', len(pliki))
            for m in MODELE:
                med = r[m]['mediana']
                print(f"  {m:16s} H=1:{med[0]} 2:{med[1]} 5:{med[4]} 10:{med[9]} 15:{med[14]} 20:{med[19]} | "
                      f"ostatni H>0: {r[m]['ostatni_H_z_dodatnia_mediana']}", flush=True)
    wynik['opis'] = dict(blok_krokow=50, horyzont_blokow=list(range(1, HMAX + 1)),
                         miara='100*(1 - SSE/SSE_brak_zmiany), mediana po ziarnach')
    (HERE / 'wyniki' / 'HORYZONT.json').write_text(json.dumps(wynik, indent=1, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
