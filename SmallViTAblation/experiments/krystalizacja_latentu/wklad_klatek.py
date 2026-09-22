#!/usr/bin/env python3
"""Ktore klatki model zyskuje, a ktore traci w czasie uczenia i czy widac to w latencie (2026-09-22).

Na tych samych klatkach walidacyjnych mamy 11 checkpointow. Dla kazdej klatki liczymy pokrycie
przewidzianej ramki z prawdziwa (IoU) w kazdym checkpoincie. Stad dwie wielkosci na klatke:

  trafnosc w szczycie   IoU w checkpoincie o najlepszej jakosci na walidacji
  zmiana po szczycie    IoU na koncu minus IoU w szczycie; ujemna = model te klatke traci

Pytanie: czy te dwie wielkosci da sie przewidziec z samego latentu, bez etykiet. Porownujemy
kilka opisow klatki, wszystkie o tej samej liczbie liczb:

  wolne mody       wspolrzedne w najtrwalszych modach operatora czasu (czesc rzeczywista)
  skladowe glowne  kierunki najwiekszej zmiennosci (kontrola)
  kierunki losowe  podloga
  caly latent      sufit dla tej reprezentacji

Uczenie i ocena zawsze na ROZNYCH filmach, wiec wynik mowi o przenoszalnosci, nie o pamieci.

    py -3.11 experiments/krystalizacja_latentu/wklad_klatek.py \
        --latents results/koopman_langevin_epochs/latents_epochs \
        --package feature_cache/detector_probe_clips \
        --map results/koopman_langevin_epochs/epoch_val_map.json \
        --out experiments/krystalizacja_latentu/wyniki/REPORT_wklad_klatek.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within, ridge_cv, whiten


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0, y0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y1 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-9)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--latents', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--map', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--blok', default='box_pool')
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--max-gap', type=int, default=6)
    args = ap.parse_args(argv)

    def klucz(p: Path) -> int:
        m = re.search(r'epoch_(\d+)', p.name)
        return int(m.group(1)) if m else -1

    pliki = sorted(args.latents.glob('LATENT_*.npz'), key=klucz)
    tagi = ['coco' if klucz(p) < 0 else f'ep{klucz(p):03d}' for p in pliki]
    epoki = [klucz(p) for p in pliki]
    dane = [np.load(p, allow_pickle=True) for p in pliki]
    fids = list(dane[0]['frame_ids'])
    video = np.array([f.split('/')[-2] for f in fids])
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)

    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    at = {f: i for i, f in enumerate(fids)}
    box = np.full((len(fids), 4), np.nan)
    for clip, t in tracks.items():
        for i, n in enumerate(t['frame']):
            j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
            if j is not None and t['box'][i]:
                box[j] = t['box'][i]
    ok = np.flatnonzero(np.isfinite(box).all(1))

    mapa = json.loads(args.map.read_text(encoding='utf-8'))['per_epoch_map_50_95_val']
    po_coco = [i for i, e in enumerate(epoki) if e >= 0]
    szczyt = max(po_coco, key=lambda i: mapa[str(epoki[i])])
    koniec = po_coco[-1]
    print(f'checkpointow {len(pliki)}, szczyt jakosci: {tagi[szczyt]}, koniec: {tagi[koniec]}', flush=True)

    krzywa = np.zeros((len(pliki), len(fids)))
    for k, d in enumerate(dane):
        krzywa[k] = iou(d['pred_box'].astype(np.float64), box)
    print(f'{"ckpt":7s} {"mAP":>6s} {"IoU":>6s} {"udzial IoU<0.5":>15s}')
    for k in po_coco:
        v = krzywa[k, ok]
        print(f'{tagi[k]:7s} {mapa[str(epoki[k])]:6.3f} {v.mean():6.3f} {100 * (v < .5).mean():14.1f}%', flush=True)

    trafnosc = krzywa[szczyt, ok]
    zmiana = krzywa[koniec, ok] - krzywa[szczyt, ok]
    vid_ok = video[ok]
    print(f'\nklatek z ramka {len(ok)}')
    print(f'traci po szczycie (IoU spada o ponad 0.05): {100 * (zmiana < -.05).mean():.1f}% klatek')
    print(f'zyskuje po szczycie (rosnie o ponad 0.05):  {100 * (zmiana > .05).mean():.1f}% klatek')
    print(f'srednia zmiana {zmiana.mean():+.4f}, mediana {np.median(zmiana):+.4f}')
    print('\nfilm po filmie (srednia zmiana IoU po szczycie):')
    for v in sorted(set(vid_ok.tolist())):
        m = vid_ok == v
        print(f'   {v:18s} klatek {int(m.sum()):4d}   IoU w szczycie {trafnosc[m].mean():.3f}   '
              f'zmiana {zmiana[m].mean():+.4f}', flush=True)

    # ---- opisy klatki liczone w checkpoincie szczytowym, bez etykiet
    zw = whiten(dane[szczyt][args.blok].astype(np.float64), args.dim)
    par = pairs_within(video, frame, args.max_gap)
    lam, u, _ = edmd(zw[par[:, 0]], zw[par[:, 1]])
    baza = u[:, :args.modes].real
    baza = baza / np.maximum(np.linalg.norm(baza, axis=0, keepdims=True), 1e-9)
    rng = np.random.default_rng(0)
    opisy = {f'wolne mody ({args.modes})': (zw @ baza)[ok],
             f'skladowe glowne ({args.modes})': zw[ok][:, :args.modes],
             f'kierunki losowe ({args.modes})': (zw @ rng.normal(size=(zw.shape[1], args.modes)))[ok],
             f'caly latent ({args.dim})': zw[ok]}
    cele = {'trafnosc w szczycie': trafnosc, 'zmiana po szczycie': zmiana}
    print(f'\nprzewidywanie z latentu (R2 w procentach, uczenie i ocena na roznych filmach)')
    print(f'{"opis klatki":26s} ' + ' '.join(f'{k:>22s}' for k in cele))
    raport = {'szczyt': tagi[szczyt], 'koniec': tagi[koniec], 'klatek': int(len(ok)),
              'Re_wolnych': [float(v) for v in lam.real[:args.modes]],
              'udzial_traci': float((zmiana < -.05).mean()), 'udzial_zyskuje': float((zmiana > .05).mean()),
              'przewidywanie': {}}
    for nazwa, x in opisy.items():
        r = {k: ridge_cv(x, y, vid_ok) for k, y in cele.items()}
        raport['przewidywanie'][nazwa] = r
        print(f'{nazwa:26s} ' + ' '.join(f'{r[k]:21.1f}%' for k in cele), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
