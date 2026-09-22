#!/usr/bin/env python3
"""Czy po samym zestawie wybranych klatek widac, ktory trening wyjdzie lepiej (2026-09-22).

Mamy dziewiec skonczonych ramion wyboru klatek, kazde po trzy ziarna, ten sam budzet 1500 i ten
sam przepis treningu. Roznia sie WYLACZNIE tym, ktore klatki trafily do uczenia. Dla kazdego
ramienia liczymy kilka wielkosci opisujacych sam zestaw, bez zadnej etykiety i bez treningu,
a potem sprawdzamy, ktora z nich ustawia ramiona w tej samej kolejnosci co koncowa jakosc.

Wielkosci liczone w kilku przestrzeniach o tej samej liczbie liczb:
  wolne mody       wspolrzedne w najtrwalszych modach operatora czasu (Koopman)
  skladowe glowne  kierunki najwiekszej zmiennosci (kontrola)
  kierunki losowe  podloga

W kazdej przestrzeni:
  luka p90 / max   jak daleko typowa i najgorzej obsluzona klatka puli ma do najblizszej wybranej
  rozpietosc       logarytm objetosci chmury wybranych klatek (im wiekszy, tym szerszy zestaw)

Poza tym, niezaleznie od przestrzeni:
  filmow           ile roznych nagran trafilo do zestawu
  szybkie klatki   udzial klatek, w ktorych narzedzie przesuwa sie szybko miedzy probkami

    py -3.11 experiments/krystalizacja_latentu/co_przewiduje_wynik.py \
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \
        --wybory curated/selection_matrix_stage1 \
        --wyniki docs/Analiza_ponownej_publikacji/evidence_selection_stage1_2026-09-21/results_stage1 \
        --out experiments/krystalizacja_latentu/wyniki/REPORT_co_przewiduje_wynik.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within
from pokrycie_a_bledy import crop_descriptor, najblizsza, wczytaj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--wybory', type=Path, required=True)
    ap.add_argument('--wyniki', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--max-gap', type=int, default=6)
    args = ap.parse_args(argv)

    pula = wczytaj(args.pula)
    grid = tuple(pula['meta']['grid'])
    prof = pula['profil_red'].astype(np.float64)
    pudlo = pula['pred_box'].astype(np.float64)
    x = crop_descriptor(prof, pudlo, grid, args.cells)
    ok = np.flatnonzero(np.isfinite(x).all(1) & (pula['score'] > .5))
    a = x[ok] - x[ok].mean(0)
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    z = a @ (vt[:args.dim].T / np.maximum(sv[:args.dim] / np.sqrt(len(a)), 1e-9))
    fid = [pula['frame_ids'][i] for i in ok]
    vid = np.array([f.split('/')[-2] for f in fid])
    nr = np.array([int(f.split('/')[-1]) for f in fid], float)
    par = pairs_within(vid, nr, args.max_gap)
    lam, u, _ = edmd(z[par[:, 0]], z[par[:, 1]])
    print(f'klatek puli z opisem {len(ok)}, par {len(par)}, '
          f'Re najwolniejszych {np.round(lam.real[:args.modes], 3).tolist()}', flush=True)

    # predkosc narzedzia miedzy kolejnymi probkami tego samego filmu (bez etykiet)
    srodek = pudlo[ok][:, :2] + pudlo[ok][:, 2:] / 2
    predkosc = np.zeros(len(ok))
    for v in np.unique(vid):
        i = np.flatnonzero(vid == v)
        i = i[np.argsort(nr[i])]
        d = np.linalg.norm(np.diff(srodek[i], axis=0), axis=1) / np.maximum(np.diff(nr[i]), 1)
        predkosc[i[1:]] = d
        if len(i) > 1:
            predkosc[i[0]] = d[0]
    prog_szybki = float(np.percentile(predkosc, 80))

    rng = np.random.default_rng(0)
    b_wolne = u[:, :args.modes].real
    b_wolne /= np.maximum(np.linalg.norm(b_wolne, axis=0, keepdims=True), 1e-9)
    b_los = rng.normal(size=(args.dim, args.modes))
    b_los /= np.linalg.norm(b_los, axis=0, keepdims=True)
    przestrzenie = {}
    for nazwa, b in (('wolne mody', b_wolne), ('skladowe glowne', np.eye(args.dim)[:, :args.modes]),
                     ('kierunki losowe', b_los)):
        k = z @ b
        przestrzenie[nazwa] = k / np.maximum(k.std(0), 1e-9)

    jakosc = defaultdict(list)
    for d in sorted(args.wyniki.glob('matrix_*')):
        p = d / 'validation_report.json'
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding='utf-8'))['checkpoints'].get('best_val', {}).get('metrics')
        if m:
            jakosc[d.name.split('_b1500')[0].replace('matrix_', '')].append(m['mAP_0.5:0.95'])
    ramiona = sorted(jakosc)
    print(f'ramion {len(ramiona)}: {", ".join(ramiona)}\n', flush=True)

    pos = {f: i for i, f in enumerate(fid)}
    cechy: dict[str, dict[str, float]] = {}
    for ramie in ramiona:
        plik = next(args.wybory.glob(f'matrix_{ramie}_b1500_*.json'))
        sel = json.loads(plik.read_text(encoding='utf-8'))
        ids = sel.get('selected_frame_ids') or sel['frame_ids']
        idx = np.array([pos[f] for f in ids if f in pos])
        c = {'filmow': float(len(set(vid[idx].tolist()))),
             'szybkie klatki': float((predkosc[idx] > prog_szybki).mean())}
        for nazwa, k in przestrzenie.items():
            d = najblizsza(k, k[idx])
            c[f'luka p90 {nazwa}'] = float(np.percentile(d, 90))
            c[f'luka max {nazwa}'] = float(d.max())
            g = k[idx] - k[idx].mean(0)
            c[f'rozpietosc {nazwa}'] = float(np.linalg.slogdet(g.T @ g / len(g) + 1e-6 * np.eye(k.shape[1]))[1])
        cechy[ramie] = c
        print(f'{ramie:24s} mAP {np.mean(jakosc[ramie]):.4f}  filmow {c["filmow"]:.0f}  '
              f'szybkich {100 * c["szybkie klatki"]:.0f}%  luka p90 (wolne) {c["luka p90 wolne mody"]:.2f}',
              flush=True)

    y = np.array([np.mean(jakosc[r]) for r in ramiona])
    bez_b0 = [i for i, r in enumerate(ramiona) if not r.startswith('B0')]
    nazwy = list(next(iter(cechy.values())))
    print(f'\n{"cecha zestawu":34s} {"r (9 ramion)":>13s} {"r (bez B0)":>11s}')
    raport = {'ramiona': ramiona, 'mAP': {r: float(np.mean(jakosc[r])) for r in ramiona},
              'cechy': cechy, 'korelacje': {}}
    for n in nazwy:
        v = np.array([cechy[r][n] for r in ramiona])
        r1 = float(np.corrcoef(v, y)[0, 1]) if v.std() > 1e-12 else 0.0
        v2, y2 = v[bez_b0], y[bez_b0]
        r2 = float(np.corrcoef(v2, y2)[0, 1]) if v2.std() > 1e-12 else 0.0
        raport['korelacje'][n] = dict(wszystkie=r1, bez_B0=r2)
        print(f'{n:34s} {r1:+13.2f} {r2:+11.2f}', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
