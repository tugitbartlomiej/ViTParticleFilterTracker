#!/usr/bin/env python3
"""Trzy zestawy po 250 klatek do porownania z ramieniem 1500 (2026-09-23).

  K6 po kwadratach      przestrzen tozsamosci dzielimy na kratke i bierzemy po jednej
                        klatce z kwadratu, zaczynajac od najliczniejszych
  B2m rownomiernie      co n-ta klatka w kazdym filmie, kwota proporcjonalna do dlugosci
  B1m losowo            podloga

Tozsamosc to mody operatora kroku, ktore trwaja w czasie, ale ZMIENIAJA sie w filmie;
mod staly przez caly film opisuje oko, swiatlo i kadr, a nie narzedzie. Caly rachunek idzie
bez etykiet: wycinek bierzemy z ramki PRZEWIDZIANEJ przez detektor.

    py -3.11 experiments/krystalizacja_latentu/buduj_ramiona_250.py \\
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \\
        --out curated/selection_maly_budzet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within
from nasycenie_tozsamosci import wybor_po_kwadratach, wybor_rownomierny
from pokrycie_a_bledy import crop_descriptor, wczytaj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--budzet', type=int, default=250)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--prog-re', type=float, default=0.85)
    ap.add_argument('--prog-w-filmie', type=float, default=0.35)
    ap.add_argument('--osie', type=int, default=3)
    ap.add_argument('--kratka', type=int, default=16)
    ap.add_argument('--max-gap', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42, help='ziarno SELEKCJI, nie treningu')
    args = ap.parse_args(argv)

    pula = wczytaj(args.pula)
    grid = tuple(pula['meta']['grid'])
    x = crop_descriptor(pula['profil_red'].astype(np.float64),
                        pula['pred_box'].astype(np.float64), grid, args.cells)
    ok = np.flatnonzero(np.isfinite(x).all(1) & (pula['score'] > .5))
    fid = [pula['frame_ids'][i] for i in ok]
    vid = np.array([f.split('/')[-2] for f in fid])
    nr = np.array([int(f.split('/')[-1]) for f in fid], float)
    a = x[ok] - x[ok].mean(0)
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    z = a @ (vt[:args.dim].T / np.maximum(sv[:args.dim] / np.sqrt(len(a)), 1e-9))
    par = pairs_within(vid, nr, args.max_gap)
    lam, u, mu = edmd(z[par[:, 0]], z[par[:, 1]])
    wolne = np.flatnonzero(lam.real > args.prog_re)
    kod = ((z - mu) @ u[:, wolne]).real
    mid = {v: kod[vid == v].mean(0) for v in set(vid.tolist())}
    dyn = kod - np.array([mid[v] for v in vid])
    w_filmie = (dyn ** 2).sum(0) / np.maximum(((kod - kod.mean(0)) ** 2).sum(0), 1e-12)
    kand = [k for k in range(len(wolne)) if w_filmie[k] >= args.prog_w_filmie]
    if len(kand) < args.osie:
        kand = list(np.argsort(-w_filmie)[:args.osie])
    osie = kand[:args.osie]
    c = kod[:, osie]
    print(f'klatek {len(fid)}, filmow {len(set(vid.tolist()))}, osie {osie}, '
          f'Re {np.round(lam.real[wolne[osie]], 3).tolist()}', flush=True)

    krawedzie = [np.quantile(c[:, j], np.linspace(0, 1, args.kratka + 1)) for j in range(c.shape[1])]
    klucz = np.zeros(len(c), dtype=np.int64)
    for j in range(c.shape[1]):
        klucz = klucz * args.kratka + np.clip(np.searchsorted(krawedzie[j][1:-1], c[:, j]), 0, args.kratka - 1)
    zajete = set(klucz.tolist())
    rng = np.random.default_rng(args.seed)

    ramiona = {
        'K6_identity_cells': wybor_po_kwadratach(klucz, args.budzet, args.seed),
        'B2m_uniform_time': wybor_rownomierny(vid, nr, args.budzet),
        'B1m_random': rng.choice(len(c), args.budzet, replace=False),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'budzet': args.budzet, 'klatek_w_puli': len(fid), 'osie': [int(v) for v in osie],
                'Re_osi': [float(lam.real[wolne[k]]) for k in osie], 'kratka': args.kratka,
                'kwadratow_zajetych': len(zajete), 'seed_selekcji': args.seed,
                'zrodlo_latentu': str(args.pula), 'ramiona': {}}
    print(f'\n{"ramie":24s} {"klatek":>7s} {"filmow":>7s} {"pokrytych kwadratow":>20s}')
    for nazwa, sel in ramiona.items():
        sel = np.asarray(sel)[:args.budzet]
        ids = [fid[i] for i in sel]
        pokrycie = len(set(klucz[sel].tolist())) / len(zajete)
        plik = args.out / f'maly_{nazwa}_b{args.budzet}_sel{args.seed}.json'
        plik.write_text(json.dumps({
            'schema_version': 1, 'variant_id': plik.stem, 'run_id': plik.stem,
            'strategy': nazwa, 'seed': args.seed, 'budget': args.budzet,
            'frame_ids': ids, 'selected_frame_ids': ids,
            'labels_used_for_ranking': False}, indent=2), encoding='utf-8')
        manifest['ramiona'][nazwa] = dict(plik=plik.name, klatek=len(ids),
                                          filmow=len(set(vid[sel].tolist())), pokrycie=pokrycie)
        print(f'{nazwa:24s} {len(ids):7d} {len(set(vid[sel].tolist())):7d} {100 * pokrycie:19.1f}%', flush=True)
    (args.out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane w', args.out.resolve(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
