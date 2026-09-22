#!/usr/bin/env python3
"""Czy tozsamosc narzedzia rosnie wraz z czescia rzeczywista wartosci wlasnej (hipoteza autora).

Hipoteza: mod, ktorego czesc rzeczywista jest najblizej jedynki, niesie najwiecej tozsamosci narzedzia;
osobno sprawdzic mody o dodatniej i ujemnej czesci rzeczywistej.

Test: KAZDY mod operatora translacyjnego osobno, jako JEDNA kolumna cech, bez laczenia.
Dla kazdego liczone sa:

    odroznienie  AUC tokenu narzedzia wobec tokenu tla (uczenie i ocena na ROZNYCH filmach;
                 prawdziwe ramki uzyte WYLACZNIE do oceny); 0,5 to brak sygnalu
    przezycie    zgodnosc wartosci przed i po przesunieciu narzedzia
    wydluzenie   ile procent proporcji prawdziwej ramki odczytuje kod klatki z tego jednego modu

Na koncu: korelacja rangowa miedzy czescia rzeczywista wartosci wlasnej a odroznieniem, osobno
dla modow o dodatniej i ujemnej czesci rzeczywistej, oraz to samo dla modulu wartosci wlasnej.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.mod_po_modzie \\
        --recording .../CZASTKI_b0500 --package .../detector_probe_clips --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .czesc_rzeczywista_a_tozsamosc import auc_tokenow
from .patch_analysis import grid_masks, load_recording, translational_pairs
from .widmo_tozsamosci import operator, skill


def ranga(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float('nan')
    return float(np.corrcoef(ranga(a), ranga(b))[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    rec = load_recording(args.recording)
    h, w = rec['meta']['grid']
    fids = rec['frame_ids']
    clips = np.array([f.split('/')[-2] for f in fids])
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    truth = grid_masks(tracks, fids, (h, w))
    prof = rec['profil_red'].astype(np.float64)
    score, maska = rec['score'].astype(float), rec['maska_pred']

    x0, x1 = translational_pairs(rec, (h, w), per_clip=400)
    lam, u, mu = operator(x0, x1)
    print(f'par {len(x0)}, modow {len(lam)}', flush=True)

    box = np.full((len(fids), 4), np.nan)
    at = {f: i for i, f in enumerate(fids)}
    for clip, t in tracks.items():
        for i, n in enumerate(t['frame']):
            j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
            if j is not None and t['box'][i]:
                box[j] = t['box'][i]
    ma_ramke = np.isfinite(box).all(1)

    rng = np.random.default_rng(args.seed)
    rows = np.flatnonzero(truth.min(1) > -2)
    wybor = rows[rng.choice(len(rows), min(args.frames, len(rows)), replace=False)]
    keep = [truth[t] != -1 for t in wybor]
    y = np.concatenate([(truth[t][k] == 1).astype(float) for t, k in zip(wybor, keep)])
    g = np.concatenate([np.full(int(k.sum()), clips[t]) for t, k in zip(wybor, keep)])
    tok = np.vstack([prof[t][k] - mu for t, k in zip(wybor, keep)])          # [N, 32] raz, potem rzuty
    kod_baza = np.full((len(fids), prof.shape[-1]), np.nan)
    for t in range(len(fids)):
        if score[t] > .5 and maska[t].sum() >= 4:
            kod_baza[t] = (prof[t, maska[t]] - mu).mean(0)
    ok = ma_ramke & np.isfinite(kod_baza).all(1)
    cel = np.log(box[ok, 2] / np.maximum(box[ok, 3], 1))
    a0, a1 = x0 - mu, x1 - mu

    wiersze = []
    print(f'\n{"nr":>3s} {"Re lambda":>10s} {"Im":>7s} {"|lambda|":>9s} {"odroznienie":>12s} {"przezycie":>10s} {"wydluzenie":>11s}')
    for k in range(len(lam)):
        v = u[:, k]
        f_tok = (tok @ v).real.reshape(-1, 1)
        auc = auc_tokenow(f_tok, y, g)
        p0, p1 = (a0 @ v).real, (a1 @ v).real
        przezycie = float(np.corrcoef(p0, p1)[0, 1])
        wyd = skill((kod_baza[ok] @ v).real.reshape(-1, 1), cel, clips[ok])
        wiersze.append(dict(nr=k, re=float(lam[k].real), im=float(lam[k].imag), abs=float(abs(lam[k])),
                            odroznienie_auc=auc, przezycie=przezycie, wydluzenie=wyd))
        print(f'{k:3d} {lam[k].real:10.3f} {lam[k].imag:7.3f} {abs(lam[k]):9.3f} {auc:12.3f} {przezycie:10.2f} {wyd:10.1f}%', flush=True)

    re = np.array([r['re'] for r in wiersze])
    ab = np.array([r['abs'] for r in wiersze])
    au = np.array([r['odroznienie_auc'] for r in wiersze])
    wy = np.array([r['wydluzenie'] for r in wiersze])
    dod, uj = re > 0, re < 0
    korel = {
        'wszystkie: Re a odroznienie': spearman(re, au),
        'wszystkie: |lambda| a odroznienie': spearman(ab, au),
        'dodatnie Re: Re a odroznienie': spearman(re[dod], au[dod]),
        'ujemne Re: |Re| a odroznienie': spearman(-re[uj], au[uj]),
        'wszystkie: Re a wydluzenie': spearman(re, wy),
        'dodatnie Re: Re a wydluzenie': spearman(re[dod], wy[dod]),
    }
    print()
    for k, v in korel.items():
        print(f'{k:42s} {v:+.2f}')
    najlepszy = int(np.argmax(au))
    print(f'\nnajwyzsze odroznienie ma mod nr {najlepszy}: Re {re[najlepszy]:.3f}, |lambda| {ab[najlepszy]:.3f}, AUC {au[najlepszy]:.3f}')
    print(f'mod o najwiekszej Re (nr {int(np.argmax(re))}): AUC {au[int(np.argmax(re))]:.3f}')
    print(f'sredni AUC: dodatnie Re {au[dod].mean():.3f} ({dod.sum()} modow), ujemne Re {au[uj].mean():.3f} ({uj.sum()} modow)')

    raport = dict(pary=int(len(x0)), mody=wiersze, korelacje_rangowe=korel,
                  najlepszy_mod=najlepszy, srednie=dict(dodatnie=float(au[dod].mean()), ujemne=float(au[uj].mean())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
