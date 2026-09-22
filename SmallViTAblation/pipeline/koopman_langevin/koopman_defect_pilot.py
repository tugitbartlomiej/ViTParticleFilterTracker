#!/usr/bin/env python3
"""Can Koopman on the DETECTOR's own latent tell what the network still lacks?

The question of document 39 was "which frames to buy". This one is upstream: at a given
checkpoint, is there a LABEL-FREE signal, built from the dynamics of the network's latent, that
marks the frames where the network's understanding of the instrument is not yet formed AND
where more training data would fix it?

Every frame of the probe clips is scored at the small-data checkpoint (L0, 500 frames) with:
  conf          1 - detector confidence                      (the standard active-learning signal)
  e_lat         residual energy of the extended Langevin equation on the latent (surprise + defect)
  defect        the part of e_lat that the motion of the PREDICTED box does not explain:
                the latent jumped although the network's own box says the tool did not
                (cross-fitted ridge of log e_lat on box-motion features, residual)
  koop_defect   one-step Koopman residual of the latent measured in the slow-mode subspace only
                (the "state" part of the latent broke, not the appearance part)
  unsettled     how far the frame's particle moved between checkpoints (Procrustes), i.e.
                the training clock: frames whose representation still drifts

Ground truth enters only in the OUTCOME: for each frame, was the L0 detector wrong (IoU < 0.5),
and did the detector trained on five times more data (2500 frames) get it right? A frame that L0
misses and the big detector hits carries REMOVABLE error; one that both miss carries error that
these labels do not remove. The pilot asks which score separates removable from irremovable
error, because that is exactly "what the network lacks that training can give".

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.koopman_defect_pilot \
        --package feature_cache/detector_probe_clips --latents results/koopman_langevin_epochs/latents_budget \
        --displacement results/koopman_langevin_epochs/REPORT_budget_axis.displacement.npz --out .../REPORT_defect_pilot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..latent_dynamics.gle import _folds, _ridge_fit_predict
from . import epoch_dynamics as ed
from .kalman_state import auc
from .latent_langevin import LatentLangevin


def cross_fitted_prediction(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    fold = _folds(groups, 4, 0)
    out = np.full_like(y, np.nan)
    for f in sorted(set(fold.tolist())):
        fit, new = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        out[new] = _ridge_fit_predict(x[fit], y[fit], x[new], alpha)
    return out


def koopman_slow_residual(z: np.ndarray, model: LatentLangevin, modes: int = 8, alpha: float = 1e-2) -> np.ndarray:
    """One-step residual of the lifted linear operator, projected on its slowest eigenfunctions."""
    w = model.windows
    x0 = np.hstack([z[w[:, -2]], z[w[:, -3]]])
    x1 = np.hstack([z[w[:, -1]], z[w[:, -2]]])
    gram = x0.T @ x0 / len(x0)
    gram.flat[::len(gram) + 1] += alpha
    k = np.linalg.solve(gram, x0.T @ x1 / len(x0))
    values, right = np.linalg.eig(k)
    order = np.argsort(-np.abs(values))[:modes]
    coords_true, coords_pred = x1 @ right[:, order], (x0 @ k) @ right[:, order]
    return (np.abs(coords_true - coords_pred) ** 2).sum(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--latents', type=Path, required=True)
    ap.add_argument('--displacement', type=Path, required=True)
    ap.add_argument('--small', default='ut_b0500_best')
    ap.add_argument('--big', default='ut_b2500_best')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    small = {k: v for k, v in np.load(args.latents / f'LATENT_{args.small}.npz').items()}
    big = {k: v for k, v in np.load(args.latents / f'LATENT_{args.big}.npz').items()}
    ids = [str(f) for f in small['frame_ids'].tolist()]
    assert ids == [str(f) for f in big['frame_ids'].tolist()]
    state = ed.physical_state(tracks, ids)
    labelled = np.isfinite(state['box']).all(1)
    val = state['split'] == 'val'
    iou_small, iou_big = ed.iou(small['pred_box'], state['box']), ed.iou(big['pred_box'], state['box'])
    miss_small, miss_big = iou_small < .5, iou_big < .5
    gain = iou_big - iou_small

    z = ed.pca(small['box_pool'].astype(np.float64), 24)[0]
    rows = ed.rows_of(state)
    stride = float(next(iter(tracks.values()))['stride'])
    model = LatentLangevin(rows, z, seed=42, dim=24, lags=8, folds=4, max_gap_frames=1.5 * stride)
    n = len(ids)
    e_lat = np.full(n, np.nan)
    e_lat[model.target] = model.err_pool
    koop = np.full(n, np.nan)
    koop[model.windows[:, -1]] = koopman_slow_residual(z, model)

    # motion of the network's OWN box between consecutive frames (label-free)
    prev = state['previous']
    pc = small['pred_box'][:, :2] + small['pred_box'][:, 2:] / 2
    step = np.full((n, 2), np.nan)
    size_step = np.full(n, np.nan)
    has = prev >= 0
    step[has] = pc[has] - pc[prev[has]]
    area = np.log(np.maximum(small['pred_box'][:, 2] * small['pred_box'][:, 3], 1))
    size_step[has] = area[has] - area[prev[has]]
    motion = np.column_stack([np.linalg.norm(step, axis=1), (step ** 2).sum(1), np.abs(size_step), size_step ** 2])
    ok = np.isfinite(e_lat) & np.isfinite(motion).all(1)
    explained = np.full(n, np.nan)
    explained[ok] = cross_fitted_prediction(motion[ok], np.log(e_lat[ok, None] + 1e-9), state['group'][ok])[:, 0]
    defect = np.log(e_lat + 1e-9) - explained                    # latent jumped more than its own box motion warrants

    disp = np.load(args.displacement)
    assert [str(f) for f in disp['frame_ids'].tolist()] == ids
    unsettled, key = None, None
    for k in disp.files:
        if k.endswith('__box_pool') and k.startswith('b0500__'):
            unsettled, key = disp[k].astype(np.float64), k
            break

    scores = {'conf (1 - pewnosc)': 1 - small['score'].astype(np.float64), 'e_lat (reszta Langevina)': e_lat,
              'defect (reszta minus ruch ramki)': defect, 'koop_defect (reszta w wolnych modach)': koop,
              'unsettled (' + str(key) + ')': unsettled}
    report = dict(small=args.small, big=args.big, frames=int(n), val_frames=int((val & labelled).sum()),
                  miss_rate_small=float(miss_small[val & labelled].mean()), miss_rate_big=float(miss_big[val & labelled].mean()),
                  removable_share_of_small_misses=float((~miss_big)[val & labelled & miss_small].mean()), scores={})
    print('walidacja: chybienia L0 %.3f, duzy detektor %.3f; usuwalne = %.2f chybien L0' % (
        report['miss_rate_small'], report['miss_rate_big'], report['removable_share_of_small_misses']))
    print('%-40s %16s %24s %18s %26s' % ('wynik', 'AUC chybienie L0', 'AUC usuwalne | chybione', 'AUC zysk IoU>0.1', 'AUC usuwalne | wszystkie'))
    for name, s in scores.items():
        m = val & labelled & np.isfinite(s)
        a1 = auc(s[m], miss_small[m])
        m2 = m & miss_small
        a2 = auc(s[m2], ~miss_big[m2])
        a3 = auc(s[m], gain[m] > .1)
        a4 = auc(s[m], miss_small[m] & ~miss_big[m])
        report['scores'][name] = dict(auc_miss_small=a1, auc_removable_among_misses=a2, auc_gain=a3, auc_removable_all=a4,
                                      frames=int(m.sum()))
        print('%-40s %16.3f %24.3f %18.3f %26.3f' % (name, a1, a2, a3, a4))
    names = list(scores)
    m = val & labelled & np.all([np.isfinite(scores[k]) for k in names], axis=0)
    corr = np.corrcoef(np.vstack([scores[k][m] for k in names]))
    report['score_correlations'] = {names[i]: {names[j]: float(corr[i, j]) for j in range(len(names))} for i in range(len(names))}
    print('korelacje miedzy wynikami (walidacja):')
    for i, a in enumerate(names):
        print('   ', a[:22].ljust(22), ' '.join('%+.2f' % corr[i, j] for j in range(len(names))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
