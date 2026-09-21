#!/usr/bin/env python3
"""Instrument state (position, velocity, acceleration) as a Langevin model filtered from the latent.

The hand of the surgeon is an input nobody observes, so an autonomous Koopman model cannot
predict it. The linear part that CAN be written down is inertia; the rest is forcing. Per image
axis the state is x = [p, v, a] and

    p' = p + v
    v' = f_v * v + a
    a' = f_a * a + w          w ~ N(0, q)     acceleration = coloured forcing (Ornstein-Uhlenbeck)
    y  = p + n                n ~ N(0, r)     y = position read out of the latent, with its error

which is an extended Langevin equation with observation noise, and its optimal estimator is the
Kalman filter. The four numbers (f_v, f_a, q, r) are fitted by maximum likelihood of the
innovations on OTHER operation groups than the ones scored. No network is trained here: this
is the reference any Koopman autoencoder on the same latent has to beat.

Observation sources compared on the same frames:
  latent box_pool   cross-fitted linear read-out of position from the detector latent
  detector box      centre of the predicted box (no latent; control)

Scored against the ground-truth track (labels are used for scoring only):
  velocity          correlation of the filtered (causal) and smoothed velocity with the local
                    slope of the true track, next to the plain finite-difference slope
  innovations       lag-1 correlation (white = model is complete) and excess kurtosis
                    (heavy tails = forcing comes in rare bursts, as HAVOK assumes)
  miss detection    does the size of the normalised innovation flag frames where the detector
                    box misses the instrument (IoU < 0.5), against the detector's own confidence

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.kalman_state \
        --package SmallViTAblation/feature_cache/detector_probe_clips \
        --latent .../LATENT_ut_b2500_best.npz --out .../REPORT_kalman_b2500.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..latent_dynamics.gle import _folds
from . import epoch_dynamics as ed
from .latent_velocity import local_slope, out_of_fold_position


CLIP = 2.5      # innovations beyond 2.5 standard deviations are treated as bursts, not as Gaussian noise


def matrices(theta: np.ndarray):
    theta = np.clip(theta, -12, 12)
    f_v, f_a = 1 / (1 + np.exp(-theta[0])), 1 / (1 + np.exp(-theta[1]))
    a = np.array([[1.0, 1.0, 0.0], [0.0, f_v, 1.0], [0.0, 0.0, f_a]])
    q = np.zeros((3, 3))
    q[2, 2] = np.exp(theta[2])
    return a, q, float(np.exp(theta[3]))


def kalman(y: np.ndarray, theta: np.ndarray, smooth: bool = False):
    """y: [sequences, time]; every sequence shares the model, so the covariances are computed once."""
    a, q, r = matrices(theta)
    h = np.array([1.0, 0.0, 0.0])
    n, t_len = y.shape
    x = np.zeros((n, 3))
    x[:, 0] = y[:, 0]
    p = np.diag([r, 25.0, 4.0])
    xs_f, ps_f, xs_p, ps_p = [], [], [], []
    innovation = np.zeros((n, t_len))
    variance = np.zeros(t_len)
    for t in range(t_len):
        if t:
            x, p = x @ a.T, a @ p @ a.T + q
        xs_p.append(x.copy()); ps_p.append(p.copy())
        s = float(h @ p @ h + r)
        gain = p @ h / s
        innovation[:, t] = y[:, t] - x[:, 0]
        variance[t] = s
        # Robust update (Masreliez): a burst moves the state by at most CLIP standard deviations.
        # The first Gaussian fit collapsed to "no inertia, 18 px noise" because the innovations
        # have an excess kurtosis of 18 to 65: forcing and detector misses come in rare jumps.
        limited = np.clip(innovation[:, t], -CLIP * np.sqrt(s), CLIP * np.sqrt(s))
        x = x + limited[:, None] * gain[None]
        p = p - np.outer(gain, h @ p)
        xs_f.append(x.copy()); ps_f.append(p.copy())
    z = np.abs(innovation[:, 1:]) / np.sqrt(variance[1:])
    huber = np.where(z <= CLIP, .5 * z ** 2, CLIP * z - .5 * CLIP ** 2)           # Huber instead of the square
    loglik = -float((huber + .5 * np.log(variance[1:])).sum())
    filtered = np.stack(xs_f, 1)
    if not smooth:
        return filtered, innovation, variance, loglik
    smoothed = filtered.copy()
    for t in range(t_len - 2, -1, -1):
        c = ps_f[t] @ a.T @ np.linalg.inv(ps_p[t + 1])
        smoothed[:, t] = filtered[:, t] + (smoothed[:, t + 1] - xs_p[t + 1]) @ c.T
    return filtered, innovation, variance, loglik, smoothed


def fit(y: np.ndarray) -> np.ndarray:
    best = None
    for start in ([2.0, 0.0, 0.0, 2.0], [0.0, 2.0, 1.0, 4.0], [3.0, -1.0, -1.0, 1.0]):
        result = minimize(lambda th: -kalman(y, th)[3], np.array(start), method='Nelder-Mead',
                          options=dict(maxiter=400, xatol=1e-3, fatol=1e-2))
        if best is None or result.fun < best.fun:
            best = result
    return best.x


def sequences(values: np.ndarray, state: dict):
    """[clips, time, 2] for clips of equal length with a constant step; plus the row index of each cell."""
    index = []
    for name in np.unique(state['clip']):
        w = np.flatnonzero(state['clip'] == name)
        w = w[np.argsort(state['number'][w])]
        index.append(w)
    length = min(len(w) for w in index)
    index = np.stack([w[:length] for w in index])
    return values[index], index


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float('nan')
    return float((rank[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--latent', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dim', type=int, default=24)
    ap.add_argument('--window', type=int, default=2, help='half-width of the reference slope of the true track')
    args = ap.parse_args(argv)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    with np.load(args.latent, allow_pickle=False) as data:
        d = {k: data[k] for k in ('frame_ids', 'box_pool', 'pred_box', 'score')}
    state = ed.physical_state(tracks, [str(f) for f in d['frame_ids'].tolist()])
    truth = state['targets']['position']
    v_true = local_slope(truth, state['clip'], state['number'], args.window)
    speed = np.linalg.norm(v_true, axis=1)
    keep = np.isfinite(speed) & (speed <= np.nanpercentile(speed, 99))
    missed = ed.iou(d['pred_box'], state['box']) < .5
    labelled = np.isfinite(state['box']).all(1)
    sources = {'latent box_pool': out_of_fold_position(ed.pca(d['box_pool'].astype(np.float64), args.dim)[0], state),
               'detector box': (d['pred_box'][:, :2] + d['pred_box'][:, 2:] / 2).astype(np.float64)}
    report = dict(latent=str(args.latent), reference_window=args.window, sources={},
                  miss_rate=float(missed[labelled].mean()),
                  auc_confidence_flags_miss=auc(-d['score'][labelled].astype(np.float64), missed[labelled]))
    for name, observed in sources.items():
        y, index = sequences(observed, state)                               # [clips, time, 2]
        groups = np.array([state['group'][w[0]] for w in index])
        fold = _folds(groups, 4, 0)
        v_filtered, v_smoothed, surprise = (np.full_like(truth, np.nan) for _ in range(3))
        thetas, lag1, kurt = [], [], []
        for f in sorted(set(fold.tolist())):
            fit_rows, new_rows = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            theta = fit(np.concatenate([y[fit_rows, :, 0], y[fit_rows, :, 1]]))
            thetas.append(theta)
            for axis in range(2):
                filtered, innovation, variance, _, smoothed = kalman(y[new_rows, :, axis], theta, smooth=True)
                v_filtered[index[new_rows], axis] = filtered[:, :, 1]
                v_smoothed[index[new_rows], axis] = smoothed[:, :, 1]
                z = innovation / np.sqrt(variance)[None]
                surprise[index[new_rows], axis] = z
                body = z[:, 5:]
                lag1.append(float(np.corrcoef(body[:, :-1].ravel(), body[:, 1:].ravel())[0, 1]))
                kurt.append(float((body ** 4).mean() / (body ** 2).mean() ** 2 - 3))
        slope = local_slope(observed, state['clip'], state['number'], args.window)

        def agreement(v):
            m = keep & np.isfinite(v).all(1)
            corr = float(np.mean([np.corrcoef(v[m][:, j], v_true[m][:, j])[0, 1] for j in range(2)]))
            r2 = float(1 - ((v[m] - v_true[m]) ** 2).sum() / ((v_true[m] - v_true[m].mean(0)) ** 2).sum())
            return dict(correlation=corr, r2=r2)

        size = np.linalg.norm(surprise, axis=1)
        m = labelled & np.isfinite(size)
        a_mat = [matrices(t) for t in thetas]
        report['sources'][name] = dict(
            velocity=dict(finite_difference_slope=agreement(slope), kalman_filtered_causal=agreement(v_filtered),
                          kalman_smoothed=agreement(v_smoothed)),
            innovations=dict(lag1=float(np.mean(lag1)), excess_kurtosis=float(np.mean(kurt))),
            parameters=dict(f_v=float(np.mean([a[0][1, 1] for a in a_mat])), f_a=float(np.mean([a[0][2, 2] for a in a_mat])),
                            forcing_std=float(np.mean([np.sqrt(a[1][2, 2]) for a in a_mat])),
                            observation_noise_std_px=float(np.mean([np.sqrt(a[2]) for a in a_mat]))),
            auc_innovation_flags_miss=auc(size[m], missed[m]))
        v = report['sources'][name]['velocity']
        print(f'{name:16s} predkosc: roznica {v["finite_difference_slope"]["correlation"]:+.2f} (R2 {v["finite_difference_slope"]["r2"]:+.2f})  '
              f'filtr {v["kalman_filtered_causal"]["correlation"]:+.2f} (R2 {v["kalman_filtered_causal"]["r2"]:+.2f})  '
              f'wygladzacz {v["kalman_smoothed"]["correlation"]:+.2f} (R2 {v["kalman_smoothed"]["r2"]:+.2f})  '
              f'| szum odczytu {report["sources"][name]["parameters"]["observation_noise_std_px"]:.1f} px  '
              f'AUC chybien {report["sources"][name]["auc_innovation_flags_miss"]:.2f}', flush=True)
    print(f'AUC chybien z pewnosci detektora {report["auc_confidence_flags_miss"]:.2f}, udzial chybien {report["miss_rate"]:.3f}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
