#!/usr/bin/env python3
"""Trzy drogi do wektora tozsamosci narzedzia, zmierzone tym samym metrem (pomysl autora, 2026-09-22).

  A. PRZESUNIECIE   operator na parach tokenow narzedzia z dwoch klatek, w ktorych narzedzie jest
                    gdzie indziej; kod klatki to srednia po tokenach w ramce detektora
  B. WIDMO CZASU    operator na opisie calego wycinka z ramki, krok to nastepna klatka filmu
  C. DEEP KOOPMAN   enkoder podnosi opis wycinka tam, gdzie krok klatki jest mnozeniem przez macierz
  D. RAZEM          wszystkie trzy zestawy wspolrzednych obok siebie

Wszedzie bierzemy TYLKO CZESC RZECZYWISTA wspolrzednych i tylko mody o dodatniej czesci rzeczywistej
wartosci wlasnej, czyli takie, ktore po kroku wracaja tym samym znakiem (na zyczenie autora).
Kazda metoda dostaje tyle samo liczb (`--modes`), zeby porownanie bylo uczciwe.

Miara (etykiety WYLACZNIE do oceny, uczenie i ocena na roznych filmach):
  wydluzenie, wielkosc   stan narzedzia; im wiecej, tym lepiej
  polozenie w kadrze     zanieczyszczenie: tozsamosc nie powinna zalezec od miejsca
  udzial miedzy filmami  zanieczyszczenie: odcisk nagrania (oko, swiatlo, przyblizenie)
  wynik lazcny           wydluzenie plus wielkosc minus polozenie minus polowa udzialu miedzy filmami

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.porownanie_tozsamosci \\
        --recording .../CZASTKI_b0500 --package .../detector_probe_clips --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .deep_koopman_ramki import train_deep
from .niezmiennik_czasowy import crop_descriptor, edmd, skill
from .patch_analysis import load_recording, translational_pairs
from .widmo_tozsamosci import operator


def ocena(kod: np.ndarray, box: np.ndarray, video: np.ndarray) -> dict:
    cele = {'wydluzenie': np.log(box[:, 2] / np.maximum(box[:, 3], 1)),
            'wielkosc': np.log(box[:, 2] * box[:, 3]),
            'polozenie': box[:, 0] + box[:, 2] / 2}
    out = {k: skill(kod, y, video) for k, y in cele.items()}
    mid = {v: kod[video == v].mean(0) for v in set(video.tolist())}
    within = ((kod - np.array([mid[v] for v in video])) ** 2).sum()
    out['miedzy_filmami'] = float(100 * (1 - within / (kod ** 2).sum()))
    out['lacznie'] = out['wydluzenie'] + out['wielkosc'] - abs(out['polozenie']) - .5 * out['miedzy_filmami']
    out['wymiarow'] = int(kod.shape[1])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path)
    ap.add_argument('--pool', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--max-gap', type=int, default=4)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--lift', type=int, default=16)
    ap.add_argument('--steps', type=int, default=3000)
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
    score, mask = rec['score'].astype(float), rec['maska_pred']
    video = np.array([f.split('/')[-2] for f in fids])
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)

    # ---------------------------------------------------------------- A. przesuniecie narzedzia
    x0, x1 = translational_pairs(rec, grid, per_clip=400)
    lam_a, u_a, mu_a = operator(x0, x1)
    keep_a = np.flatnonzero(lam_a.real > 0)[:args.modes]
    print(f'A: par {len(x0)}, wybrane Re(lambda) {np.round(lam_a[keep_a].real, 3).tolist()}', flush=True)
    kod_a = np.full((len(fids), len(keep_a)), np.nan)
    for t in range(len(fids)):
        if score[t] > .5 and mask[t].sum() >= 4:
            kod_a[t] = ((prof[t, mask[t]] - mu_a) @ u_a[:, keep_a]).real.mean(0)

    # ---------------------------------------------------------------- B. widmo czasu na wycinku
    x = crop_descriptor(prof, box, grid, args.cells)
    ok = np.isfinite(x).all(1) & np.isfinite(box).all(1) & np.isfinite(kod_a).all(1)
    rows = np.flatnonzero(ok)
    xm = x[rows].astype(np.float64)
    xm = xm - xm.mean(0)
    _, s, vt = np.linalg.svd(xm, full_matrices=False)
    base = vt[:args.dim].T / np.maximum(s[:args.dim] / np.sqrt(len(xm)), 1e-9)
    z = np.full((len(fids), args.dim), np.nan)
    z[rows] = xm @ base
    pairs = []
    for v in np.unique(video[rows]):
        idx = rows[video[rows] == v]
        idx = idx[np.argsort(frame[idx])]
        d = np.diff(frame[idx])
        for i in np.flatnonzero(d <= args.max_gap):
            pairs.append((idx[i], idx[i + 1]))
    pairs = np.array(pairs)
    lam_b, u_b, mu_b = edmd(z[pairs[:, 0]], z[pairs[:, 1]])
    keep_b = np.flatnonzero(lam_b.real > 0)[:args.modes]
    print(f'B: par {len(pairs)}, wybrane Re(lambda) {np.round(lam_b[keep_b].real, 3).tolist()}', flush=True)
    kod_b = ((z - mu_b) @ u_b[:, keep_b]).real

    # ---------------------------------------------------------------- C. deep Koopman
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pos = {r: k for k, r in enumerate(rows)}
    local_pairs = np.array([[pos[a], pos[b]] for a, b in pairs])
    deep = train_deep(z[rows], local_pairs, video[rows], args.lift, 0, args.steps, 256, device)
    kod_c = np.full((len(fids), deep['code'].shape[1]), np.nan)
    kod_c[rows[deep['finite']]] = deep['code'][:, :deep['code'].shape[1]].real if np.iscomplexobj(deep['code']) else deep['code']
    kod_c = kod_c[:, :args.modes]
    print(f"C: deep, |lambda| {np.round(np.abs(deep['lam'][:args.modes]), 3).tolist()}, umiejetnosc {deep['skill']:.1f}%", flush=True)

    dobre = rows[np.isfinite(kod_a[rows]).all(1) & np.isfinite(kod_b[rows]).all(1) & np.isfinite(kod_c[rows]).all(1)]
    warianty = {'A przesuniecie narzedzia': kod_a[dobre], 'B widmo czasu na wycinku': kod_b[dobre],
                'C deep Koopman': kod_c[dobre],
                'D wszystkie trzy razem': np.hstack([kod_a[dobre], kod_b[dobre], kod_c[dobre]]),
                'kontrola: PCA opisu': z[dobre][:, :args.modes]}
    print(f'\nklatek w porownaniu: {len(dobre)}, filmow {len(set(video[dobre].tolist()))}')
    print(f'{"metoda":28s} {"wym":>4s} {"wydluzenie":>11s} {"wielkosc":>9s} {"polozenie":>10s} {"miedzy filmami":>15s} {"lacznie":>8s}')
    raport = {}
    for nazwa, kod in warianty.items():
        r = ocena(kod, box[dobre], video[dobre])
        raport[nazwa] = r
        print(f'{nazwa:28s} {r["wymiarow"]:4d} {r["wydluzenie"]:10.1f}% {r["wielkosc"]:8.1f}% {r["polozenie"]:9.1f}% '
              f'{r["miedzy_filmami"]:14.0f}% {r["lacznie"]:8.1f}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(frames=int(len(dobre)), modes=args.modes, lift=args.lift,
                                        lam_A=[float(v) for v in lam_a[keep_a].real],
                                        lam_B=[float(v) for v in lam_b[keep_b].real],
                                        deep_skill=deep['skill'], warianty=raport), indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
