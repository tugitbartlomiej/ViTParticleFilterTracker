#!/usr/bin/env python3
"""Diagnoza domknięcia latentu tożsamości w pętli ../petla_od_poczatku_large (PLAN.md tutaj): trzy warunki na każdym
końcu obrotu, osobno dla pętli koopman, losowe, bez i ziaren 42, 43, 44.

Latent tożsamości sondy i: a_i = w·(narzędzie_i − tło_i) (skalar) i 5 liczb (w, S_4) z kierunki.npz ziarna.
Sondy: 12 obrazów × 6 wariantów (original, rotation90, black, yellow, length08, foreground_bright); 8 obrazów fit,
4 holdout.

  W1 zatrzymanie:     v = średni |Δa| na 5 aktualizacji / rozrzut a między sondami (ułamek na krok)
  W2 niezmienniczość: R = średnie odchylenie a między wariantami tego samego obrazu / odchylenie średnich obrazów;
                      przesunięcie wariantu = średnia (a_wariant − a_original) po obrazach;
                      rozpoznanie = odsetek sond, których najbliższy środek obrazu (5 liczb, bez tej sondy) jest ich
                      własnym obrazem (przypadek 1/12)
  W3 rozdzielenie:    min_i a_i (narzędzie odróżnione od tła, gdy > 0)

    py -3.11 diagnoza_domkniecia.py      ->  wyniki/DIAGNOZA.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PETLA = HERE.parent / 'petla_od_poczatku_large'
sys.path.insert(0, str(PETLA))
import obrot_large as O  # noqa: E402

WARIANTY = ('original', 'rotation90', 'black', 'yellow', 'length08', 'foreground_bright')


def main() -> int:
    rows, mapa, fit = O.sondy()
    src = np.array([r['source_file'] for r in rows]); var = np.array([r['variant'] for r in rows])
    obrazy = sorted(set(src)); gi = np.array([obrazy.index(x) for x in src]); hold = ~fit

    def R(a, maska):
        gs = sorted(set(gi[maska]))
        grupy = [a[maska & (gi == g)] for g in gs]
        return float(np.mean([x.std() for x in grupy]) / np.std([x.mean() for x in grupy]))

    def rozpoznanie(Z):
        ok = []
        for i in range(len(Z)):
            c = np.stack([Z[(gi == g) & (np.arange(len(Z)) != i)].mean(0) for g in range(len(obrazy))])
            ok.append(int(np.argmin(((c - Z[i]) ** 2).sum(1)) == gi[i]))
        return np.array(ok)

    wyn = {}
    for s in (42, 43, 44):
        P = np.load(O.W / f's{s}' / 'kierunki.npz')['projection']
        D0 = O.latent(s, 'wspolny', 0, mapa)
        for petla in ('koopman', 'losowe', 'bez'):
            ks = [k for k in range(1, 7) if len(list((O.W / f's{s}' / petla / f'k{k:02d}' / 'latent').glob('*.npz'))) >= 32]
            Y = np.concatenate([D0] + [O.latent(s, petla, k, mapa) for k in ks]) @ P       # (T, 72, 5)
            a = Y[..., 0]
            skala = Y[32].std(0)
            wpisy = []
            for j in range(len(ks) + 2):
                t = 0 if j == 0 else 32 * j
                okno = a[max(t - 31, 0):t + 1] if t else a[:2]
                v = float(np.abs(np.diff(okno, axis=0)).mean() / a[t].std())
                ok = rozpoznanie(Y[t] / skala)
                wpisy.append(dict(
                    obrot=('start' if j == 0 else f'k{j - 1:02d}'), krok=int(t), skalar_sredni=float(a[t].mean()),
                    W1_predkosc_na_krok=v, W2_R=R(a[t], np.ones(72, bool)), W2_R_holdout=R(a[t], hold),
                    W2_przesuniecia={w: float(np.mean([a[t][(gi == g) & (var == w)][0] - a[t][(gi == g) & (var == 'original')][0]
                                                        for g in range(len(obrazy))])) for w in WARIANTY[1:]},
                    W2_rozpoznanie=float(ok.mean()), W2_rozpoznanie_holdout=float(ok[hold].mean()),
                    W3_min_skalar=float(a[t].min())))
            wyn.setdefault(str(s), {})[petla] = wpisy
    out = HERE / 'wyniki' / 'DIAGNOZA.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(opis=__doc__, sond=len(rows), obrazow=len(obrazy), wynik=wyn), indent=1,
                              ensure_ascii=False), encoding='utf-8')
    for s, d in wyn.items():
        for petla, w in d.items():
            print(f's{s} {petla:8s} ' + ' | '.join(f'{x["obrot"]} v {x["W1_predkosc_na_krok"]:.4f} R {x["W2_R"]:.2f} '
                                               f'rozp {x["W2_rozpoznanie"]:.2f} min {x["W3_min_skalar"]:.1f}' for x in w))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
