#!/usr/bin/env python3
"""Kiedy przestac uczyc: trzy sygnaly obok siebie, na kilku ziarnach (2026-09-22).

Porownujemy trzy sposoby wskazania epoki, w ktorej siec jest juz wytrenowana:

  jakosc na walidacji   szczyt mAP; to jest wzorzec, ale wymaga etykiet walidacyjnych
  strata na walidacji   najnizsza strata na tych samych klatkach; standardowy sposob,
                        tez wymaga etykiet
  trwalosc modow        srednia czesc rzeczywista wartosci wlasnych operatora kroku klatki,
                        liczona na samym latencie; BEZ zadnych etykiet

Dla kazdego ziarna wypisujemy, w ktorej epoce kazdy z tych sygnalow ma swoj szczyt albo dolek,
i o ile jakosci traci sie, ida za sygnalem bez etykiet zamiast za wzorcem.

    py -3.11 experiments/krystalizacja_latentu/kiedy_przestac.py \
        --biegi <katalog z results_telemetria/*> --latenty <katalog z latents_telemetria/*> \
        --out experiments/krystalizacja_latentu/wyniki/REPORT_kiedy_przestac.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within, whiten


def trwalosc_krzywa(katalog: Path, blok: str, dim: int, modes: int, max_gap: int):
    def klucz(p: Path) -> int:
        m = re.search(r'epoch_(\d+)', p.name)
        return int(m.group(1)) if m else -1

    pliki = sorted(katalog.glob('LATENT_*.npz'), key=klucz)
    pliki = [p for p in pliki if klucz(p) >= 0]
    if not pliki:
        return [], []
    epoki = [klucz(p) for p in pliki]
    d0 = np.load(pliki[0], allow_pickle=True)
    fids = [str(f) for f in d0['frame_ids']]
    video = np.array([f.split('/')[-2] for f in fids])
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)
    par = pairs_within(video, frame, max_gap)
    out = []
    for p in pliki:
        z = np.load(p, allow_pickle=True)[blok].astype(np.float64)
        zw = whiten(z, dim)
        lam, _, _ = edmd(zw[par[:, 0]], zw[par[:, 1]])
        out.append(float(lam.real[:modes].mean()))
    return epoki, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--biegi', type=Path, required=True, help='katalog z podkatalogami biegow (TELEMETRIA.json)')
    ap.add_argument('--latenty', type=Path, help='katalog z podkatalogami latentow biegow')
    ap.add_argument('--trwalosc', type=Path, help='gotowy JSON z krzywymi trwalosci (policzony na klastrze)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--blok', default='box_pool')
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--max-gap', type=int, default=6)
    args = ap.parse_args(argv)

    gotowa = json.loads(args.trwalosc.read_text(encoding='utf-8')) if args.trwalosc else None
    raport, wiersze = {}, []
    for bieg in sorted(p for p in args.biegi.iterdir() if p.is_dir()):
        tele = bieg / 'TELEMETRIA.json'
        if not tele.is_file():
            print(f'pomijam {bieg.name}: brak TELEMETRIA.json', flush=True)
            continue
        t = json.loads(tele.read_text(encoding='utf-8'))
        mp = np.array([x[1] for x in t.get('val/mAP_50_95', [])])
        vl = np.array([x[1] for x in t.get('val/loss', [])])
        tl = np.array([x[1] for x in t.get('train/loss', [])])
        if gotowa is not None and bieg.name in gotowa:
            epoki, tr = gotowa[bieg.name]['epoki'], gotowa[bieg.name][args.blok]
        else:
            epoki, tr = trwalosc_krzywa(args.latenty / bieg.name, args.blok, args.dim, args.modes, args.max_gap)
        w = {'epok': int(len(mp)), 'szczyt_mAP': int(np.argmax(mp)) if len(mp) else None,
             'mAP_max': float(mp.max()) if len(mp) else None,
             'mAP_koniec': float(mp[-1]) if len(mp) else None,
             'dolek_val_loss': int(np.argmin(vl)) if len(vl) else None,
             'val_loss_min': float(vl.min()) if len(vl) else None,
             'val_loss_koniec': float(vl[-1]) if len(vl) else None,
             'rozjazd_koniec': float(vl[-1] - tl[-1]) if len(vl) and len(tl) else None,
             'epoki_checkpointow': epoki, 'trwalosc': tr}
        if tr:
            k = int(np.argmax(tr))
            w['szczyt_trwalosci'] = int(epoki[k])
            w['mAP_w_szczycie_trwalosci'] = float(mp[epoki[k]]) if epoki[k] < len(mp) else None
            if w['mAP_max'] is not None and w['mAP_w_szczycie_trwalosci'] is not None:
                w['strata_za_sygnalem'] = float(w['mAP_max'] - w['mAP_w_szczycie_trwalosci'])
        raport[bieg.name] = w
        wiersze.append((bieg.name, w))

    print(f'\n{"bieg":38s} {"szczyt mAP":>11s} {"dolek straty":>13s} {"szczyt trwalosci":>17s} '
          f'{"strata za sygnalem":>19s}')
    for nazwa, w in wiersze:
        s = w.get('strata_za_sygnalem')
        strata = f'{s:.4f}' if s is not None else '-'
        print(f'{nazwa:38s} {str(w["szczyt_mAP"]):>11s} {str(w["dolek_val_loss"]):>13s} '
              f'{str(w.get("szczyt_trwalosci")):>17s} {strata:>19s}', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
