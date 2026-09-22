#!/usr/bin/env python3
"""Jak latent detektora zmienia sie w trakcie uczenia i czy sie krystalizuje (pytanie autora, 2026-09-22).

Wejscie: po jednym `LATENT_<tag>.npz` na checkpoint, wszystkie na TYCH SAMYCH klatkach,
uszeregowane wzdluz zegara treningu, plus prawdziwe ramki narzedzia (tracks.json) i krzywa
jakosci na walidacji (epoch_val_map.json).

Dla kazdego checkpointu (bez etykiet):
  PR            ile kierunkow latent naprawde uzywa (uczestnictwo)
  statyka       jaka czesc zmiennosci to roznice MIEDZY filmami, a nie ruch wewnatrz filmu
  wolne mody    ile modow operatora czasu ma czesc rzeczywista powyzej progu
  trwalosc      srednia czesc rzeczywista najwolniejszych modow
  statyka wolnych  ile statyki lapie podprzestrzen wolnych modow

Dla kazdej pary sasiednich checkpointow:
  przesuniecie  srednia droga klatki po najlepszym obrocie (Procrustes)
  CKA           czy geometria calego latentu jest ta sama
  CCA wolnych   czy WOLNA podprzestrzen opisuje to samo co krok wczesniej

Z etykietami WYLACZNIE do oceny (uczenie i ocena na roznych filmach):
  odczyt polozenia, wielkosci i wydluzenia narzedzia z latentu

Na koniec: korelacja kazdej krzywej z jakoscia na walidacji, czyli czy ktorakolwiek wielkosc
liczona BEZ etykiet mowi, kiedy przestac uczyc.

    py -3.11 experiments/krystalizacja_latentu/krystalizacja.py \
        --latents results/koopman_langevin_epochs/latents_epochs \
        --package feature_cache/detector_probe_clips \
        --map results/koopman_langevin_epochs/epoch_val_map.json \
        --out experiments/krystalizacja_latentu/wyniki/REPORT_krystalizacja.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- narzedzia

def whiten(z: np.ndarray, dim: int) -> np.ndarray:
    """Pierwsze `dim` kierunkow glownych, kazdy o tej samej sile (zeby skala nie mylila porownan)."""
    a = z - z.mean(0)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    d = min(dim, int((s > 1e-8 * s[0]).sum()))
    return a @ (vt[:d].T / (s[:d] / np.sqrt(len(a))))


def participation(z: np.ndarray) -> float:
    a = z - z.mean(0)
    ev = np.clip(np.linalg.eigvalsh(a.T @ a / len(a)), 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-20))


def pairs_within(video: np.ndarray, frame: np.ndarray, max_gap: int) -> np.ndarray:
    out = []
    for v in np.unique(video):
        idx = np.flatnonzero(video == v)
        idx = idx[np.argsort(frame[idx])]
        d = np.diff(frame[idx])
        for i in np.flatnonzero(d <= max_gap):
            out.append((idx[i], idx[i + 1]))
    return np.array(out)


def edmd(x0: np.ndarray, x1: np.ndarray, reg: float = 1e-3):
    mu = x0.mean(0)
    a, b = x0 - mu, x1 - mu
    c00 = a.T @ a / len(a) + reg * np.eye(a.shape[1])
    k = np.linalg.solve(c00, a.T @ b / len(a))
    lam, u = np.linalg.eig(k)
    o = np.argsort(-lam.real)
    return lam[o], u[:, o], mu


def cca_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Srednia korelacja kanoniczna: czy dwa opisy tych samych klatek mowia to samo."""
    qa, _ = np.linalg.qr(a - a.mean(0))
    qb, _ = np.linalg.qr(b - b.mean(0))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def cka(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    n = np.linalg.norm(a.T @ b, 'fro') ** 2
    return float(n / max(np.linalg.norm(a.T @ a, 'fro') * np.linalg.norm(b.T @ b, 'fro'), 1e-20))


def procrustes_move(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Droga kazdej klatki miedzy checkpointami po najlepszym obrocie i skali."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    a0 = a0 / np.linalg.norm(a0) * np.sqrt(len(a0))
    b0 = b0 / np.linalg.norm(b0) * np.sqrt(len(b0))
    u, _, vt = np.linalg.svd(a0.T @ b0)
    return np.linalg.norm(a0 @ (u @ vt) - b0, axis=1)


def ridge_cv(x: np.ndarray, y: np.ndarray, group: np.ndarray, lam: float = 1.0, folds: int = 4, seed: int = 0) -> float:
    names = np.array(sorted(set(group.tolist())))
    fold = {v: i % folds for i, v in enumerate(np.random.default_rng(seed).permutation(names))}
    part = np.array([fold[v] for v in group])
    xx = np.hstack([(x - x.mean(0)) / np.maximum(x.std(0), 1e-9), np.ones((len(x), 1))])
    pred = np.zeros_like(y)
    for f in range(folds):
        tr, te = part != f, part == f
        if not tr.any() or not te.any():
            continue
        beta = np.linalg.solve(xx[tr].T @ xx[tr] + lam * np.eye(xx.shape[1]), xx[tr].T @ y[tr])
        pred[te] = xx[te] @ beta + (y[tr].mean(0) - (xx[tr] @ beta).mean(0))
    return float(100 * (1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()))


# --------------------------------------------------------------------------- glowne

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--latents', type=Path, required=True, help='katalog z LATENT_*.npz')
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--map', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--blok', default='box_pool', choices=['box_pool', 'query', 'global_pool', 'features'])
    ap.add_argument('--dim', type=int, default=32)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--prog', type=float, default=0.9, help='prog czesci rzeczywistej dla wolnego modu')
    ap.add_argument('--max-gap', type=int, default=6)
    args = ap.parse_args(argv)

    def klucz(p: Path) -> int:
        m = re.search(r'epoch_(\d+)', p.name)
        return int(m.group(1)) if m else -1

    pliki = sorted(args.latents.glob('LATENT_*.npz'), key=klucz)
    tagi = ['coco' if klucz(p) < 0 else f'ep{klucz(p):03d}' for p in pliki]
    epoki = [klucz(p) for p in pliki]
    print(f'checkpointow {len(pliki)}: {", ".join(tagi)}', flush=True)

    mapa = json.loads(args.map.read_text(encoding='utf-8'))['per_epoch_map_50_95_val']
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))

    dane = [np.load(p, allow_pickle=True) for p in pliki]
    fids = list(dane[0]['frame_ids'])
    for d in dane[1:]:
        assert list(d['frame_ids']) == fids, 'checkpointy licza inne klatki'
    video = np.array([f.split('/')[-2] for f in fids])
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)

    box = np.full((len(fids), 4), np.nan)
    at = {f: i for i, f in enumerate(fids)}
    for clip, t in tracks.items():
        for i, n in enumerate(t['frame']):
            j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
            if j is not None and t['box'][i]:
                box[j] = t['box'][i]
    ok = np.isfinite(box).all(1)
    print(f'klatek {len(fids)}, filmow {len(set(video.tolist()))}, z ramka {int(ok.sum())}', flush=True)

    cele = {'polozenie': box[ok, 0] + box[ok, 2] / 2,
            'wielkosc': np.log(np.maximum(box[ok, 2] * box[ok, 3], 1.0)),
            'wydluzenie': np.log(np.maximum(box[ok, 2], 1.0) / np.maximum(box[ok, 3], 1.0))}
    par = pairs_within(video, frame, args.max_gap)
    print(f'par klatka-do-klatki {len(par)}', flush=True)

    raport = {'blok': args.blok, 'dim': args.dim, 'modes': args.modes, 'prog_Re': args.prog,
              'klatek': len(fids), 'par': int(len(par)), 'checkpointy': {}, 'zegar_treningu': []}
    kody, wolne, wiersze = [], [], []
    for tag, ep, d in zip(tagi, epoki, dane):
        z = d[args.blok].astype(np.float64)
        zw = whiten(z, args.dim)
        lam, u, _ = edmd(zw[par[:, 0]], zw[par[:, 1]])
        ile_wolnych = int((lam.real > args.prog).sum())
        baza = u[:, :args.modes].real
        baza = baza / np.maximum(np.linalg.norm(baza, axis=0, keepdims=True), 1e-9)
        kod_wolny = zw @ baza
        q, _ = np.linalg.qr(baza)
        a = zw - zw.mean(0)
        mid = {v: a[video == v].mean(0) for v in set(video.tolist())}
        stat = np.array([mid[v] for v in video])
        st_all = float((stat ** 2).sum() / (a ** 2).sum())
        st_wolne = float(((stat @ q) ** 2).sum() / (stat ** 2).sum())
        odczyt = {k: ridge_cv(zw[ok], y, video[ok]) for k, y in cele.items()}
        w = dict(epoka=ep, mAP=(float(mapa[str(ep)]) if str(ep) in mapa else None),
                 PR=participation(z), statyka=st_all, wolnych=ile_wolnych,
                 trwalosc=float(lam.real[:args.modes].mean()), statyka_wolnych=st_wolne,
                 odczyt=odczyt)
        raport['checkpointy'][tag] = w
        wiersze.append((tag, w))
        kody.append(zw)
        wolne.append(kod_wolny)

    print()
    print(f'{"ckpt":7s} {"mAP":>6s} {"PR":>6s} {"statyka":>8s} {"wolnych":>8s} {"trwalosc":>9s} '
          f'{"stat.wolne":>11s} {"polozenie":>10s} {"wielkosc":>9s} {"wydluzenie":>11s}')
    for tag, w in wiersze:
        mp = f'{w["mAP"]:.3f}' if w['mAP'] is not None else '  -  '
        print(f'{tag:7s} {mp:>6s} {w["PR"]:6.2f} {100 * w["statyka"]:7.0f}% {w["wolnych"]:8d} '
              f'{w["trwalosc"]:9.3f} {100 * w["statyka_wolnych"]:10.0f}% '
              f'{w["odczyt"]["polozenie"]:9.1f}% {w["odczyt"]["wielkosc"]:8.1f}% '
              f'{w["odczyt"]["wydluzenie"]:10.1f}%', flush=True)

    print()
    print(f'{"krok":18s} {"przesuniecie":>13s} {"CKA":>7s} {"CCA wolnych":>12s} {"CCA calosci":>12s}')
    for i in range(1, len(tagi)):
        ruch = procrustes_move(kody[i - 1], kody[i])
        w = dict(od=tagi[i - 1], do=tagi[i], przesuniecie=float(ruch.mean()),
                 cka=cka(kody[i - 1], kody[i]), cca_wolne=cca_mean(wolne[i - 1], wolne[i]),
                 cca_calosc=cca_mean(kody[i - 1], kody[i]))
        raport['zegar_treningu'].append(w)
        print(f'{tagi[i - 1] + " -> " + tagi[i]:18s} {w["przesuniecie"]:13.3f} {w["cka"]:7.3f} '
              f'{w["cca_wolne"]:12.3f} {w["cca_calosc"]:12.3f}', flush=True)

    ep_ok = [i for i, w in enumerate(wiersze) if w[1]['mAP'] is not None]
    y = np.array([wiersze[i][1]['mAP'] for i in ep_ok])
    krzywe = {'PR': np.array([wiersze[i][1]['PR'] for i in ep_ok]),
              'statyka': np.array([wiersze[i][1]['statyka'] for i in ep_ok]),
              'wolnych': np.array([wiersze[i][1]['wolnych'] for i in ep_ok], float),
              'trwalosc': np.array([wiersze[i][1]['trwalosc'] for i in ep_ok]),
              'statyka_wolnych': np.array([wiersze[i][1]['statyka_wolnych'] for i in ep_ok])}
    print('\nkorelacja krzywej z jakoscia na walidacji (checkpointy po fine-tuningu):')
    raport['korelacja_z_mAP'] = {}
    for k, v in krzywe.items():
        r = float(np.corrcoef(v, y)[0, 1]) if v.std() > 1e-12 else 0.0
        raport['korelacja_z_mAP'][k] = r
        print(f'   {k:18s} {r:6.2f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
