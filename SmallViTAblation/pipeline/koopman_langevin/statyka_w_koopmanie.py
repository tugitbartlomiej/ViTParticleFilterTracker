#!/usr/bin/env python3
"""Czy statyka filmu siedzi w modach Koopmana, a skladowe glowne ja mieszaja (teza autora, 2026-09-22).

Teza: kierunki glowne to tylko "gdzie dane najbardziej sie roznia" i nie wiadomo, co nios;
mody Koopmana sa z definicji uszeregowane po TRWALOSCI w czasie, wiec to, co w filmie stoi
(narzedzie jako przedmiot, oko, swiatlo, kadr), powinno siedziec w modach najtrwalszych.

Rozklad kazdej wspolrzednej na dwie czesci, bez zadnych etykiet:
  statyka    srednia wspolrzednej w danym filmie (to, co przez caly film stoi)
  dynamika   odchylenie od tej sredniej wewnatrz filmu (to, co sie w filmie rusza)

Dla kazdego kierunku liczymy:
  udzial statyki      ile jego zmiennosci to roznice miedzy filmami
  trwalosc            korelacja wartosci w klatce i w nastepnej, WEWNATRZ filmu (po odjeciu statyki)
Porownujemy trzy zestawy po tyle samo kierunkow: mody Koopmana (po czesci rzeczywistej wartosci
wlasnej), skladowe glowne oraz kierunki losowe. Na koniec: ile statyki i ile dynamiki kazdy zestaw
w ogole zatrzymuje.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.statyka_w_koopmanie \\
        --recording .../CZASTKI_pool_b0500 --pool .../pool_manifest.jsonl --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .niezmiennik_czasowy import crop_descriptor, edmd
from .patch_analysis import load_recording


def rozklad(kod: np.ndarray, video: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = {v: kod[video == v].mean(0) for v in set(video.tolist())}
    stat = np.array([mid[v] for v in video])
    return stat, kod - stat


def trwalosc(kod: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    out = []
    for k in range(kod.shape[1]):
        a, b = kod[pairs[:, 0], k], kod[pairs[:, 1], k]
        out.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0)
    return np.array(out)


def udzial(czesc: np.ndarray, calosc: np.ndarray) -> np.ndarray:
    return (czesc ** 2).sum(0) / np.maximum((calosc ** 2).sum(0), 1e-12)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path)
    ap.add_argument('--pool', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--max-gap', type=int, default=6)
    ap.add_argument('--modes', type=int, default=8)
    args = ap.parse_args(argv)
    rec = load_recording(args.recording)
    grid = tuple(rec['meta']['grid'])
    fids = rec['frame_ids']
    at = {f: i for i, f in enumerate(fids)}
    box = np.full((len(fids), 4), np.nan)
    if args.package:
        tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
        for clip, t in tracks.items():
            for i, n in enumerate(t['frame']):
                j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
                if j is not None and t['box'][i]:
                    box[j] = t['box'][i]
    else:
        for line in args.pool.open(encoding='utf-8'):
            r = json.loads(line)
            j = at.get(r['frame_id'])
            ann = r.get('annotations') or []
            if j is not None and ann and r.get('split', 'train') == 'train':
                box[j] = ann[0]['bbox']
    prof = rec['profil_red'].astype(np.float64)
    x = crop_descriptor(prof, box, grid, args.cells)
    ok = np.isfinite(x).all(1) & np.isfinite(box).all(1)
    rows = np.flatnonzero(ok)
    video = np.array([f.split('/')[-2] for f in fids])[rows]
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)[rows]
    xm = x[rows].astype(np.float64)
    xm = xm - xm.mean(0)
    _, s, vt = np.linalg.svd(xm, full_matrices=False)
    base = vt[:args.dim].T / np.maximum(s[:args.dim] / np.sqrt(len(xm)), 1e-9)
    z = xm @ base
    pairs = []
    for v in np.unique(video):
        idx = np.flatnonzero(video == v)
        idx = idx[np.argsort(frame[idx])]
        d = np.diff(frame[idx])
        for i in np.flatnonzero(d <= args.max_gap):
            pairs.append((idx[i], idx[i + 1]))
    pairs = np.array(pairs)
    lam, modes, mu = edmd(z[pairs[:, 0]], z[pairs[:, 1]])
    print(f'klatek {len(rows)}, filmow {len(set(video.tolist()))}, par {len(pairs)}', flush=True)

    stat_all, dyn_all = rozklad(z, video)
    print(f'w calym opisie: statyka (roznice miedzy filmami) {100 * (stat_all ** 2).sum() / (z ** 2).sum():.0f}%\n')

    rng = np.random.default_rng(0)
    zestawy = {'mody Koopmana (najtrwalsze)': modes[:, np.argsort(-lam.real)[:args.modes]].real,
               'skladowe glowne (najsilniejsze)': np.eye(args.dim)[:, :args.modes],
               'kierunki losowe': rng.normal(size=(args.dim, args.modes))}
    raport = {'statyka_w_opisie': float((stat_all ** 2).sum() / (z ** 2).sum())}
    print(f'{"zestaw":32s} {"udzial statyki":>15s} {"trwalosc w filmie":>18s} {"zatrzymana statyka":>19s} {"zatrzymana dynamika":>20s}')
    for nazwa, b in zestawy.items():
        b = b / np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-9)
        kod = z @ b
        stat, dyn = rozklad(kod, video)
        u = udzial(stat, kod)
        tr = trwalosc(dyn, pairs)
        q, _ = np.linalg.qr(b)                                     # ile calej statyki i dynamiki lapie ten podprzestrzen
        zs = float(((stat_all @ q) ** 2).sum() / (stat_all ** 2).sum())
        zd = float(((dyn_all @ q) ** 2).sum() / (dyn_all ** 2).sum())
        raport[nazwa] = dict(udzial_statyki=[float(v) for v in u], trwalosc=[float(v) for v in tr],
                             zatrzymana_statyka=zs, zatrzymana_dynamika=zd)
        print(f'{nazwa:32s} {100 * u.mean():14.0f}% {tr.mean():18.2f} {100 * zs:18.0f}% {100 * zd:19.0f}%', flush=True)
    kol = np.argsort(-lam.real)[:args.modes]
    print('\nmod po modzie (Koopman): Re(lambda), udzial statyki, trwalosc dynamiki')
    b = modes[:, kol].real
    b = b / np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-9)
    kod = z @ b
    stat, dyn = rozklad(kod, video)
    u, tr = udzial(stat, kod), trwalosc(dyn, pairs)
    for k in range(args.modes):
        print(f'   Re {lam[kol[k]].real:6.3f}   statyka {100 * u[k]:3.0f}%   trwalosc {tr[k]:5.2f}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
