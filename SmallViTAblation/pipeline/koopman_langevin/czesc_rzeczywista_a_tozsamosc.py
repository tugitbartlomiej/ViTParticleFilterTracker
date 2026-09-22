#!/usr/bin/env python3
"""Czy tozsamosc narzedzia siedzi w CZESCI RZECZYWISTEJ modow Koopmana (hipoteza autora, 2026-09-22).

Operator translacyjny (token narzedzia w jednym polozeniu -> ten sam token po przesunieciu narzedzia)
liczony BEZ symetryzacji ma wartosci wlasne zespolone. Kazda mode daje dla tokenu liczbe zespolona

    c = (profil - srodek) . u

Pytanie: ktora czesc tej liczby niesie tozsamosc narzedzia. Sprawdzane postacie:

    Re        sama czesc rzeczywista
    Im        sama czesc urojona
    |c|       modul pary sprzezonej (teoria: dla pary obrotowej niezmiennikiem jest dopiero dlugosc)
    Re^2      kwadrat (teoria: dla mody o ujemnej wartosci wlasnej znak zmienia sie po kroku,
              wiec niezmiennikiem jest kwadrat, a nie sama wartosc)
    Re i Im   obie naraz, czyli to, czego uzywal dotychczasowy kod tozsamosci

Dla kazdej postaci i kazdej rodziny modow mierzone sa trzy rzeczy:

    odroznienie  AUC tokenu narzedzia wobec tokenu tla, uczenie i ocena na ROZNYCH filmach
                 (prawdziwe ramki uzyte WYLACZNIE do oceny)
    przezycie    zgodnosc wartosci przed i po przesunieciu narzedzia (im blizej jedynki, tym trwalsza)
    wydluzenie   ile procent proporcji prawdziwej ramki da sie odczytac z kodu klatki

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.czesc_rzeczywista_a_tozsamosc \\
        --recording .../CZASTKI_b0500 --package .../detector_probe_clips --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .patch_analysis import grid_masks, load_recording, translational_pairs
from .widmo_tozsamosci import operator, skill


def postac(c: np.ndarray, kind: str) -> np.ndarray:
    """Zespolone wspolrzedne -> liczby rzeczywiste w zadanej postaci."""
    if kind == 'Re':
        return c.real
    if kind == 'Im':
        return c.imag
    if kind == '|c|':
        return np.abs(c)
    if kind == 'Re^2':
        return c.real ** 2
    if kind == 'Re i Im':
        return np.hstack([c.real, c.imag])
    raise ValueError(kind)


def auc_tokenow(x: np.ndarray, y: np.ndarray, g: np.ndarray, lam: float = 1.0) -> float:
    """Ridge z podzialem po filmach; AUC tokenu narzedzia wobec tla."""
    names = sorted(set(g.tolist()))
    fold = {v: i % 2 for i, v in enumerate(names)}
    part = np.array([fold[v] for v in g])
    x = np.hstack([(x - x.mean(0)) / np.maximum(x.std(0), 1e-9), np.ones((len(x), 1))])
    pred = np.zeros(len(y))
    for f in (0, 1):
        tr, te = part != f, part == f
        if not tr.any() or not te.any():
            continue
        a = x[tr].T @ x[tr] + lam * np.eye(x.shape[1])
        pred[te] = x[te] @ np.linalg.solve(a, x[tr].T @ y[tr])
    pos, neg = pred[y == 1], pred[y == 0]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(len(order)) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(len(pos) * len(neg), 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modes', type=int, default=5, help='ile modow z kazdej rodziny')
    ap.add_argument('--frames', type=int, default=600, help='ile klatek na pomiar po tokenach')
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
    tol = 1e-8 * max(1.0, float(np.abs(lam).max()))
    rzecz = np.abs(lam.imag) < tol
    rodziny = {
        'trwale (Re>0, Im=0)': np.flatnonzero(rzecz & (lam.real > 0))[:args.modes],
        'przemienne (Re<0, Im=0)': np.flatnonzero(rzecz & (lam.real < 0))[:args.modes],
        'obrotowe (Im>0, po jednej z pary)': np.flatnonzero((~rzecz) & (lam.imag > 0))[:args.modes],
    }
    print(f'par {len(x0)}; ' + ', '.join(f'{k}: {len(v)}' for k, v in rodziny.items()), flush=True)

    # prawdziwe ramki: tylko do oceny
    box = np.full((len(fids), 4), np.nan)
    for clip, t in tracks.items():
        for i, n in enumerate(t['frame']):
            j = fids.index(f'cataract_lmm/tracking/{clip}/{n:08d}') if f'cataract_lmm/tracking/{clip}/{n:08d}' in fids else None
            if j is not None and t['box'][i]:
                box[j] = t['box'][i]
    ma_ramke = np.isfinite(box).all(1)

    rng = np.random.default_rng(args.seed)
    rows = np.flatnonzero(truth.min(1) > -2)
    wybor = rows[rng.choice(len(rows), min(args.frames, len(rows)), replace=False)]

    raport = {'pary': int(len(x0)), 'rodziny': {}}
    print(f'\n{"rodzina":36s} {"postac":10s} {"|lambda|":>8s} {"odroznienie":>11s} {"przezycie":>10s} {"wydluzenie":>11s}')
    for nazwa, cols in rodziny.items():
        if not len(cols):
            continue
        v = u[:, cols]
        zespolone = np.abs(lam[cols].imag).max() > tol
        # wspolrzedne tokenow wybranych klatek + prawda do oceny
        feats_c, y, g = [], [], []
        for t in wybor:
            keep = truth[t] != -1
            c = (prof[t][keep] - mu) @ v
            feats_c.append(c)
            y.append((truth[t][keep] == 1).astype(float))
            g.append(np.full(int(keep.sum()), clips[t]))
        feats_c = np.vstack(feats_c)
        y, g = np.concatenate(y), np.concatenate(g)
        # wspolrzedne par (przezycie)
        a0, a1 = (x0 - mu) @ v, (x1 - mu) @ v
        # kod klatki (wydluzenie)
        kod_c = np.full((len(fids), len(cols)), np.nan, dtype=complex)
        for t in range(len(fids)):
            if score[t] > .5 and maska[t].sum() >= 4:
                kod_c[t] = ((prof[t, maska[t]] - mu) @ v).mean(0)
        ok = ma_ramke & np.isfinite(kod_c).all(1)
        cel = np.log(box[ok, 2] / np.maximum(box[ok, 3], 1))

        raport['rodziny'][nazwa] = dict(mody=len(cols), lam=[[float(lam[c].real), float(lam[c].imag)] for c in cols], postacie={})
        for kind in (('Re', 'Im', '|c|', 'Re i Im', 'Re^2') if zespolone else ('Re', 'Re^2')):
            f_tok = postac(feats_c, kind)
            auc = auc_tokenow(f_tok, y, g)
            p0, p1 = postac(a0, kind), postac(a1, kind)
            przezycie = float(np.mean([np.corrcoef(p0[:, j], p1[:, j])[0, 1] for j in range(p0.shape[1])]))
            wyd = skill(postac(kod_c[ok], kind), cel, clips[ok])
            raport['rodziny'][nazwa]['postacie'][kind] = dict(odroznienie_auc=auc, przezycie=przezycie, wydluzenie=wyd)
            print(f'{nazwa:36s} {kind:10s} {abs(lam[cols[0]]):8.3f} {auc:11.3f} {przezycie:10.2f} {wyd:10.1f}%', flush=True)

    # laczenie: czy modul par obrotowych doklada cos do samych trwalych
    trw = rodziny['trwale (Re>0, Im=0)']
    obr = rodziny['obrotowe (Im>0, po jednej z pary)']
    prz = rodziny['przemienne (Re<0, Im=0)']
    zestawy = {
        'trwale Re': [(trw, 'Re')],
        'trwale Re + obrotowe Re': [(trw, 'Re'), (obr, 'Re')],
        'trwale Re + obrotowe |c|': [(trw, 'Re'), (obr, '|c|')],
        'trwale Re + przemienne Re^2': [(trw, 'Re'), (prz, 'Re^2')],
        'trwale Re + obrotowe |c| + przemienne Re^2': [(trw, 'Re'), (obr, '|c|'), (prz, 'Re^2')],
        'trwale Re + obrotowe Re i Im': [(trw, 'Re'), (obr, 'Re i Im')],
        'trwale Re + obrotowe Re i Im + przemienne Re^2': [(trw, 'Re'), (obr, 'Re i Im'), (prz, 'Re^2')],
        'wszystko liniowo (jak dotad)': [(trw, 'Re'), (obr, 'Re i Im'), (prz, 'Re')],
    }
    print()
    raport['zestawy'] = {}
    for nazwa, czesci in zestawy.items():
        f_tok, kod_parts, ok_all = [], [], None
        for cols, kind in czesci:
            v = u[:, cols]
            blok = []
            for t in wybor:
                keep = truth[t] != -1
                blok.append(postac((prof[t][keep] - mu) @ v, kind))
            f_tok.append(np.vstack(blok))
            kod_c = np.full((len(fids), len(cols)), np.nan, dtype=complex)
            for t in range(len(fids)):
                if score[t] > .5 and maska[t].sum() >= 4:
                    kod_c[t] = ((prof[t, maska[t]] - mu) @ v).mean(0)
            kod_parts.append((kod_c, kind))
            ok_all = np.isfinite(kod_c).all(1) if ok_all is None else (ok_all & np.isfinite(kod_c).all(1))
        y2, g2 = [], []
        for t in wybor:
            keep = truth[t] != -1
            y2.append((truth[t][keep] == 1).astype(float))
            g2.append(np.full(int(keep.sum()), clips[t]))
        auc = auc_tokenow(np.hstack(f_tok), np.concatenate(y2), np.concatenate(g2))
        ok = ma_ramke & ok_all
        kod = np.hstack([postac(k[ok], kind) for k, kind in kod_parts])
        wyd = skill(kod, np.log(box[ok, 2] / np.maximum(box[ok, 3], 1)), clips[ok])
        raport['zestawy'][nazwa] = dict(odroznienie_auc=auc, wydluzenie=wyd, kolumny=int(np.hstack(f_tok).shape[1]))
        print(f'{nazwa:44s} kolumn {np.hstack(f_tok).shape[1]:3d}  odroznienie {auc:.3f}  wydluzenie {wyd:5.1f}%', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
