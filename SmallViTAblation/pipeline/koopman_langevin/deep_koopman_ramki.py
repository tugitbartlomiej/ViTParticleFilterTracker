#!/usr/bin/env python3
"""Deep Koopman na opisie WYCINKA narzedzia: nieliniowe podniesienie, w ktorym krok klatki jest liniowy.

Pomysl autora (2026-09-22): skoro opis ramki klatka po klatce daje niezmienniki, poszukac ich
autoenkoderem Koopmana zamiast jednym liniowym operatorem.

  wejscie     opis wycinka z prawdziwej ramki narzedzia (siatka `--cells`, wybielony do `--dim`)
  podniesienie enkoder g (siec gesta) przenosi opis do `--lift` liczb
  krok        w tych liczbach nastepna klatka to MNOZENIE przez jedna macierz: g(x1) = g(x0) W
  powrot      dekoder h wraca z podniesienia do opisu, wiec podniesienie nie moze zgubic tresci
  niezmiennik wektor wlasny W o module najblizszym jedynki: wielkosc, ktora po klatce wraca prawie ta sama

Uczciwosc pomiaru:
  podzial po FILMACH          model uczy sie na jednych operacjach, sprawdza na innych
  punkt odniesienia TRWALOSC  "nic sie nie zmienia" (x1 = x0); umiejetnosc liczona wzgledem niego
  punkt odniesienia LINIOWY   ten sam podzial, zwykly operator EDMD bez podniesienia
  podloga CZAS POTASOWANY     te same klatki, pary bez kolejnosci; tyle daje sam bezwlad obrazu
  kosinusy glowne             czy kierunki deep pokrywaja sie z liniowymi i z translacyjnymi

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.deep_koopman_ramki \\
        --recording .../CZASTKI_pool_b0500 --pool .../pool_manifest.jsonl --lift 8 16 32 --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .niezmiennik_czasowy import crop_descriptor, edmd, skill
from .patch_analysis import generalised_koopman, load_recording, principal_cosines, translational_pairs


def build(recording: Path, package: Path | None, pool: Path | None, cells: int, dim: int, max_gap: int):
    rec = load_recording(recording)
    grid = tuple(rec['meta']['grid'])
    fids = rec['frame_ids']
    at = {f: i for i, f in enumerate(fids)}
    box = np.full((len(fids), 4), np.nan)
    if package:
        tracks = json.loads((package / 'tracks.json').read_text(encoding='utf-8'))
        for clip, t in tracks.items():
            for i, n in enumerate(t['frame']):
                j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
                if j is not None and t['box'][i]:
                    box[j] = t['box'][i]
    else:
        for line in pool.open(encoding='utf-8'):
            r = json.loads(line)
            j = at.get(r['frame_id'])
            ann = r.get('annotations') or []
            if j is not None and ann and r.get('split', 'train') == 'train':
                box[j] = ann[0]['bbox']
    prof = rec['profil_red'].astype(np.float64)
    x = crop_descriptor(prof, box, grid, cells)
    video = np.array([f.split('/')[-2] for f in fids])
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)
    ok = np.isfinite(x).all(1) & np.isfinite(box).all(1)
    xm = x[ok].astype(np.float64)
    mean = xm.mean(0)
    _, s, vt = np.linalg.svd(xm - mean, full_matrices=False)
    base = vt[:dim].T / np.maximum(s[:dim] / np.sqrt(len(xm)), 1e-9)
    z = np.full((len(fids), dim), np.nan)
    z[ok] = (xm - mean) @ base
    rows = np.flatnonzero(ok)
    pairs = []
    for v in np.unique(video[rows]):
        idx = rows[video[rows] == v]
        idx = idx[np.argsort(frame[idx])]
        d = np.diff(frame[idx])
        for i in np.flatnonzero(d <= max_gap):
            pairs.append((idx[i], idx[i + 1]))
    return dict(rec=rec, grid=grid, z=z, box=box, video=video, rows=rows, pairs=np.array(pairs), base=base, cells=cells)


def split_videos(names: np.ndarray, seed: int) -> tuple[set, set]:
    """Jedna trzecia filmow na ocene, jedna szosta na wczesne zatrzymanie, reszta na uczenie."""
    order = np.random.default_rng(seed).permutation(names)
    n_te = max(1, len(names) // 3)
    n_va = max(1, len(names) // 6)
    return set(order[:n_te].tolist()), set(order[n_te:n_te + n_va].tolist())


def train_deep(z: np.ndarray, pairs: np.ndarray, video: np.ndarray, lift: int, seed: int, steps: int,
               hidden: int, device: str, eval_pairs: np.ndarray | None = None):
    """`pairs` sluzy do UCZENIA, `eval_pairs` (domyslnie te same) do oceny; podzial zawsze po filmach."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    ev = pairs if eval_pairs is None else eval_pairs
    names = np.array(sorted(set(video[pairs[:, 0]].tolist())))
    test_names, val_names = split_videos(names, seed)
    tr = np.array([v not in test_names and v not in val_names for v in video[pairs[:, 0]]])
    va = np.array([v in val_names for v in video[pairs[:, 0]]])
    te_ev = np.array([v in test_names for v in video[ev[:, 0]]])
    d = z.shape[1]
    x0 = torch.tensor(z[pairs[:, 0]], dtype=torch.float32, device=device)
    x1 = torch.tensor(z[pairs[:, 1]], dtype=torch.float32, device=device)
    e0 = torch.tensor(z[ev[te_ev, 0]], dtype=torch.float32, device=device)
    e1 = torch.tensor(z[ev[te_ev, 1]], dtype=torch.float32, device=device)
    v0 = torch.tensor(z[pairs[va, 0]], dtype=torch.float32, device=device)
    v1 = torch.tensor(z[pairs[va, 1]], dtype=torch.float32, device=device)
    mtr = torch.tensor(tr, device=device)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Sequential(nn.Linear(d, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, lift))
            self.h = nn.Sequential(nn.Linear(lift, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, d))
            self.w = nn.Parameter(torch.eye(lift) + 0.01 * torch.randn(lift, lift))

    model = Model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    a0, a1 = x0[mtr], x1[mtr]
    best_val, best_state, patience = np.inf, None, 0
    for step in range(steps):
        opt.zero_grad()
        g0, g1 = model.g(a0), model.g(a1)
        pred = g0 @ model.w
        loss = ((model.h(g0) - a0) ** 2).mean() + ((model.h(g1) - a1) ** 2).mean() \
            + ((pred - g1) ** 2).mean() + ((model.h(pred) - a1) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if step % 25 == 0 and len(v0):                              # wczesne zatrzymanie na filmach walidacyjnych
            with torch.no_grad():
                val = float(((model.h(model.g(v0) @ model.w) - v1) ** 2).mean())
            if val < best_val - 1e-6:
                best_val, patience = val, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        err_model = float(((model.h(model.g(e0) @ model.w) - e1) ** 2).sum())
        err_persist = float(((e0 - e1) ** 2).sum())
    lam = np.linalg.eigvals(model.w.detach().cpu().numpy().T)
    lam = lam[np.argsort(-np.abs(lam))]
    # kierunki wrazliwosci funkcji wlasnych w przestrzeni wejscia (srednia po danych)
    w_np = model.w.detach().cpu().numpy().T
    vals, vecs = np.linalg.eig(w_np)
    order = np.argsort(-np.abs(vals))
    u = torch.tensor(np.real(vecs[:, order[:8]]), dtype=torch.float32, device=device)
    xs = x0[:512].clone().requires_grad_(True)
    gg = model.g(xs) @ u
    dirs = []
    for k in range(u.shape[1]):
        grad = torch.autograd.grad(gg[:, k].sum(), xs, retain_graph=True)[0]
        dirs.append(grad.mean(0).detach().cpu().numpy())
    kod_all = None
    with torch.no_grad():
        finite = np.isfinite(z).all(1)
        zz = torch.tensor(z[finite], dtype=torch.float32, device=device)
        kod_all = (model.g(zz) @ u).cpu().numpy()
    return dict(lam=lam, skill=float(100 * (1 - err_model / err_persist)), dirs=np.array(dirs).T,
                code=kod_all, finite=finite, test_videos=sorted(test_names))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path)
    ap.add_argument('--pool', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--max-gap', type=int, default=6)
    ap.add_argument('--lift', type=int, nargs='+', default=[8, 16, 32])
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args(argv)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = build(args.recording, args.package, args.pool, args.cells, args.dim, args.max_gap)
    z, pairs, video, rows = data['z'], data['pairs'], data['video'], data['rows']
    print(f'par {len(pairs)}, filmow {len(set(video[pairs[:, 0]].tolist()))}, wymiar wejscia {z.shape[1]}, urzadzenie {device}', flush=True)

    # liniowy punkt odniesienia na tym samym podziale
    rng = np.random.default_rng(0)
    names = np.array(sorted(set(video[pairs[:, 0]].tolist())))
    test_names = set(rng.permutation(names)[:max(1, len(names) // 3)].tolist())
    te = np.array([v in test_names for v in video[pairs[:, 0]]])
    lam_lin, modes_lin, mu_lin = edmd(z[pairs[~te, 0]], z[pairs[~te, 1]])
    a = z[pairs[~te, 0]] - mu_lin
    b = z[pairs[~te, 1]] - mu_lin
    k_lin = np.linalg.solve(a.T @ a / len(a) + 1e-3 * np.eye(z.shape[1]), a.T @ b / len(a))
    pred = (z[pairs[te, 0]] - mu_lin) @ k_lin + mu_lin
    lin_skill = float(100 * (1 - ((pred - z[pairs[te, 1]]) ** 2).sum() / ((z[pairs[te, 0]] - z[pairs[te, 1]]) ** 2).sum()))

    shuffled = pairs.copy()
    for v in np.unique(video[pairs[:, 0]]):
        m = np.flatnonzero(video[pairs[:, 0]] == v)
        shuffled[m, 1] = pairs[np.random.default_rng(1).permutation(m), 1]

    b = data['box'][rows]
    cele = {'wydluzenie ramki': np.log(b[:, 2] / np.maximum(b[:, 3], 1)),
            'wielkosc ramki': np.log(b[:, 2] * b[:, 3]),
            'polozenie w kadrze (poziomo)': b[:, 0] + b[:, 2] / 2}
    t0, t1 = translational_pairs(data['rec'], data['grid'], per_clip=200)
    _, tmod, _ = generalised_koopman(t0, t1)
    k_tok = data['rec']['profil_red'].shape[-1]

    raport = dict(pairs=int(len(pairs)), dim=int(z.shape[1]), cells=args.cells, linear_skill=lin_skill,
                  linear_lam=[float(abs(v)) for v in lam_lin[:8]], lifts={})
    print(f'\nLINIOWY punkt odniesienia: umiejetnosc nad trwaloscia {lin_skill:.1f}%, |lambda| {np.round(np.abs(lam_lin[:4]), 3).tolist()}\n')
    print(f'{"podniesienie":>13s} {"umiejetnosc":>12s} {"podloga tasowana":>17s} {"|lambda| max":>13s} '
          f'{"wydluzenie":>11s} {"polozenie":>10s} {"kos. z liniowym":>16s} {"kos. z translacja":>18s}')
    for lift in args.lift:
        got = [train_deep(z, pairs, video, lift, s, args.steps, args.hidden, device) for s in args.seeds]
        # podloga: uczenie na parach bez kolejnosci, ocena na PRAWDZIWYCH parach tych samych filmow
        floor = [train_deep(z, shuffled, video, lift, s, args.steps, args.hidden, device, eval_pairs=pairs)
                 for s in args.seeds[:1]]
        niesie_all, kos_lin_all, kos_tr_all = [], [], []
        for g in got:                                              # usredniamy po ziarnach, bez wybierania po wyniku
            sel = np.isin(rows, np.flatnonzero(g['finite']))
            kod = g['code'][np.isin(np.flatnonzero(g['finite']), rows)]
            niesie_all.append({k: skill(kod, y[sel], video[rows][sel]) for k, y in cele.items()})
            kos_lin_all.append(principal_cosines(g['dirs'], modes_lin[:, :8].real))
            w_tok = (data['base'] @ g['dirs']).reshape(args.cells * args.cells, k_tok, g['dirs'].shape[1]).mean(0)
            kos_tr_all.append(principal_cosines(w_tok, tmod[:, :8].real))
        niesie = {k: float(np.mean([n[k] for n in niesie_all])) for k in cele}
        kos_lin = np.mean(kos_lin_all, axis=0)
        kos_tr = np.mean(kos_tr_all, axis=0)
        raport['lifts'][lift] = dict(skill=[g['skill'] for g in got], skill_shuffled=[f['skill'] for f in floor],
                                     lam_abs=[[float(abs(v)) for v in g['lam'][:8]] for g in got], carries=niesie,
                                     carries_per_seed=niesie_all, cosines_vs_linear=kos_lin.tolist(),
                                     cosines_vs_translational=kos_tr.tolist(), test_videos=got[0]['test_videos'])
        print(f'{lift:13d} {np.mean([g["skill"] for g in got]):11.1f}% {np.mean([f["skill"] for f in floor]):16.1f}% '
              f'{np.mean([abs(g["lam"][0]) for g in got]):13.3f} {niesie["wydluzenie ramki"]:10.1f}% '
              f'{niesie["polozenie w kadrze (poziomo)"]:9.1f}% '
              f'{np.round(kos_lin[:2], 2).tolist()!s:>16s} {np.round(kos_tr[:2], 2).tolist()!s:>18s}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
