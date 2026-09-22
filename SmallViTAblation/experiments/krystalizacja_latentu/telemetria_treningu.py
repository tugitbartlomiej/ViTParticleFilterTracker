#!/usr/bin/env python3
"""Kiedy siec jest juz wytrenowana: telemetria wszystkich biegow kampanii (2026-09-22).

Wejscie: jeden JSON ze zrzutem wszystkich skalarow TensorBoard z biegow na Edenie
(`TELEMETRIA_stage1.json`, klucz = nazwa biegu, wartosc = tag -> [[krok, wartosc], ...]).

Dla kazdego biegu liczymy:
  szczyt            epoka o najwyzszej jakosci na walidacji
  wystarczy         pierwsza epoka, po ktorej jakosc jest juz o wlos od szczytu (domyslnie 0.5%)
  spadek            o ile jakosc spada od szczytu do konca
  strata uczaca     ile razy spada strata na zbiorze uczacym od pierwszej do ostatniej epoki
  strata walidacji  jesli bieg ja zapisal: epoka najnizszej straty na walidacji i jej wartosc
  rozjazd           roznica straty walidacyjnej i uczacej na koncu (im wieksza, tym mocniej
                    siec dopasowala sie do samych klatek uczacych)

Na koniec zbiorczo po ramionach: mediana epoki szczytu i epoki "wystarczy".

    py -3.11 experiments/krystalizacja_latentu/telemetria_treningu.py \
        --telemetria experiments/krystalizacja_latentu/dane/TELEMETRIA_stage1.json \
        --out experiments/krystalizacja_latentu/wyniki/REPORT_telemetria.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def ser(run: dict, tag: str) -> np.ndarray:
    v = run.get(tag) or []
    return np.array([x[1] for x in v], dtype=float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--telemetria', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tolerancja', type=float, default=0.005, help='ile wzglednie wolno stracic wzgledem szczytu')
    args = ap.parse_args(argv)

    dane = json.loads(args.telemetria.read_text(encoding='utf-8'))
    print(f'biegow {len(dane)}', flush=True)
    tagi = sorted({t for r in dane.values() for t in r})
    ma_val_loss = [n for n, r in dane.items() if any(t.startswith('val/loss') for t in r)]
    print(f'tagow razem {len(tagi)}, biegow ze strata walidacji {len(ma_val_loss)}', flush=True)

    raport, wiersze = {}, []
    for nazwa in sorted(dane):
        r = dane[nazwa]
        mp = ser(r, 'val/mAP_50_95')
        tl = ser(r, 'train/loss')
        vl = ser(r, 'val/loss')
        if not len(mp):
            continue
        szczyt = int(np.argmax(mp))
        prog = mp.max() * (1 - args.tolerancja)
        wystarczy = int(np.argmax(mp >= prog))
        w = dict(epok=len(mp), szczyt=szczyt, mAP_szczyt=float(mp.max()), wystarczy=wystarczy,
                 spadek=float(mp.max() - mp[-1]),
                 strata_start=float(tl[0]) if len(tl) else None,
                 strata_koniec=float(tl[-1]) if len(tl) else None,
                 strata_stosunek=float(tl[0] / tl[-1]) if len(tl) else None)
        if len(vl):
            w['val_loss_min_epoka'] = int(np.argmin(vl))
            w['val_loss_min'] = float(vl.min())
            w['val_loss_koniec'] = float(vl[-1])
            w['rozjazd_koniec'] = float(vl[-1] - tl[-1]) if len(tl) else None
        raport[nazwa] = w
        wiersze.append((nazwa, w))

    print(f'\n{"bieg":42s} {"szczyt":>7s} {"wystarczy":>10s} {"mAP":>7s} {"spadek":>7s} '
          f'{"strata x":>9s} {"val loss min":>13s}')
    for nazwa, w in wiersze:
        vlm = f'ep{w["val_loss_min_epoka"]} {w["val_loss_min"]:.3f}' if 'val_loss_min_epoka' in w else '-'
        print(f'{nazwa:42s} {w["szczyt"]:7d} {w["wystarczy"]:10d} {w["mAP_szczyt"]:7.4f} '
              f'{w["spadek"]:7.4f} {w["strata_stosunek"]:9.2f} {vlm:>13s}', flush=True)

    grupy = defaultdict(list)
    for nazwa, w in wiersze:
        grupy[nazwa.split('_b1500')[0].replace('matrix_', '')].append(w)
    print(f'\n{"ramie":26s} {"biegow":>7s} {"szczyt (mediana)":>17s} {"wystarczy (mediana)":>20s} {"spadek":>8s}')
    zbiorczo = {}
    for g in sorted(grupy):
        v = grupy[g]
        zbiorczo[g] = dict(biegow=len(v), szczyt=float(np.median([x['szczyt'] for x in v])),
                           wystarczy=float(np.median([x['wystarczy'] for x in v])),
                           spadek=float(np.mean([x['spadek'] for x in v])))
        print(f'{g:26s} {len(v):7d} {zbiorczo[g]["szczyt"]:17.0f} {zbiorczo[g]["wystarczy"]:20.0f} '
              f'{zbiorczo[g]["spadek"]:8.4f}', flush=True)

    wszystkie_szczyt = [w['szczyt'] for _, w in wiersze]
    wszystkie_wystarczy = [w['wystarczy'] for _, w in wiersze]
    print(f'\nrazem: szczyt w epoce {np.median(wszystkie_szczyt):.0f} '
          f'(od {min(wszystkie_szczyt)} do {max(wszystkie_szczyt)}), '
          f'granica tolerancji w epoce {np.median(wszystkie_wystarczy):.0f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'tolerancja': args.tolerancja, 'biegi': raport, 'ramiona': zbiorczo,
                                    'tagi': tagi}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
