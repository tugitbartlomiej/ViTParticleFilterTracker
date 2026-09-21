#!/usr/bin/env python3
"""Koopman (DMD) separation of a moving instrument from a slowly varying scene.

Idea (Grosek and Kutz 2014, DMD background/foreground separation), applied to
DINOv2 patch tokens instead of pixels. Inside a short window the token video is
decomposed into Koopman modes. Modes whose eigenvalue stays near 1 do not
change from frame to frame: they are the scene (eye, drape, lighting). What is
left after removing them is whatever moves, which in cataract tracking clips is
mostly the instrument. From that foreground we read, without labels:

  saliency   (T, P)   where the moving object is on the patch grid
  centroid   (T, 2)   instrument position in native pixels
  spread     (T, 2)   spatial extent of the foreground
  descriptor (T, c)   saliency-weighted mean token = appearance of the object

Every baseline that a reviewer would ask for is computed in the same pass:
frame difference, temporal-median and temporal-mean background subtraction.
Ground-truth boxes are read only by `validate`.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.koopman_tool \
        --tokens SmallViTAblation/feature_cache/koopman_tokens \
        --out SmallViTAblation/results/koopman_langevin_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ('dmd_slow_modes', 'temporal_median', 'temporal_mean', 'frame_difference')


def valid_patches(geometry: dict, grid: int) -> np.ndarray:
    """Patches whose centre lies inside the letterboxed image, not in the padding."""
    size = geometry['size']
    cell = size / grid
    centres = (np.arange(grid) + .5) * cell
    w = geometry['native_width'] * geometry['scale']
    h = geometry['native_height'] * geometry['scale']
    ok_x = (centres >= geometry['left']) & (centres <= geometry['left'] + w)
    ok_y = (centres >= geometry['top']) & (centres <= geometry['top'] + h)
    return (ok_y[:, None] & ok_x[None, :]).reshape(-1)


def dmd_background(x: np.ndarray, rank: int, slow_tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact DMD of one window x (W, n). Returns the slow-mode reconstruction and eigenvalues."""
    x1, x2 = x[:-1].T, x[1:].T                                  # (n, W-1)
    gram = x1.T @ x1                                            # method of snapshots: n >> W
    evals, v = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1][:rank]
    s = np.sqrt(np.maximum(evals[order], 1e-12))
    v = v[:, order]
    u = x1 @ v / s                                              # (n, r)
    a_tilde = u.T @ x2 @ v / s                                  # (r, r)
    lam, w = np.linalg.eig(a_tilde)
    phi = x2 @ v / s @ w                                        # (n, r) exact DMD modes
    b = np.linalg.lstsq(phi, x[0].astype(np.complex128), rcond=None)[0]
    slow = np.abs(np.log(lam + 1e-12)) < slow_tol               # |log lambda| small: no decay, no rotation
    if not slow.any():
        slow[np.argmin(np.abs(np.log(lam + 1e-12)))] = True
    steps = np.arange(len(x))
    dynamics = (lam[slow][None, :] ** steps[:, None]) * b[slow][None, :]      # (W, n_slow)
    return np.real(dynamics @ phi[:, slow].T), lam


def saliency_maps(tokens: np.ndarray, *, window: int = 24, hop: int = 12, rank: int = 8,
                  slow_tol: float = .05) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Foreground energy per patch for DMD and three simple controls. tokens: (T, P, c)."""
    t_len, p, c = tokens.shape
    x = tokens.reshape(t_len, p * c).astype(np.float64)
    acc = {m: np.zeros((t_len, p)) for m in METHODS if m != 'frame_difference'}
    count = np.zeros(t_len)
    spectra = []
    starts = list(range(0, max(1, t_len - window + 1), hop))
    if starts[-1] + window < t_len:
        starts.append(t_len - window)
    for s0 in starts:
        seg = x[s0:s0 + window]
        if len(seg) < 4:
            continue
        background, lam = dmd_background(seg, min(rank, len(seg) - 2), slow_tol)
        spectra.append(lam)
        parts = {'dmd_slow_modes': seg - background,
                 'temporal_median': seg - np.median(seg, axis=0),
                 'temporal_mean': seg - seg.mean(axis=0)}
        for name, fg in parts.items():
            acc[name][s0:s0 + len(seg)] += np.linalg.norm(fg.reshape(len(seg), p, c), axis=2)
        count[s0:s0 + len(seg)] += 1
    maps = {name: value / np.maximum(count, 1)[:, None] for name, value in acc.items()}
    diff = np.zeros((t_len, p))
    diff[1:] = np.linalg.norm((x[1:] - x[:-1]).reshape(t_len - 1, p, c), axis=2)
    diff[0] = diff[1] if t_len > 1 else 0
    maps['frame_difference'] = diff
    return maps, spectra


def tool_state(saliency: np.ndarray, tokens: np.ndarray, geometry: dict, grid: int,
               valid: np.ndarray, power: float = 2.0) -> dict[str, np.ndarray]:
    """Centroid and spread in native pixels, plus the appearance descriptor of the foreground."""
    cell = geometry['size'] / grid
    ys, xs = np.divmod(np.arange(grid * grid), grid)
    px = ((xs + .5) * cell - geometry['left']) / geometry['scale']
    py = ((ys + .5) * cell - geometry['top']) / geometry['scale']
    w = np.where(valid[None, :], saliency, 0.0) ** power
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    cx, cy = w @ px, w @ py
    sx = np.sqrt(np.maximum(w @ px ** 2 - cx ** 2, 0))
    sy = np.sqrt(np.maximum(w @ py ** 2 - cy ** 2, 0))
    descriptor = np.einsum('tp,tpc->tc', w, tokens.astype(np.float64))
    return dict(centroid=np.stack([cx, cy], 1), spread=np.stack([sx, sy], 1), descriptor=descriptor,
                concentration=(w ** 2).sum(1) * valid.sum())        # 1 = flat map, large = peaked map


def _boxes(labels: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    box = np.full((len(labels), 4), np.nan)
    for i, row in enumerate(labels):
        text = (row.get('cystotome_bbox') or '').strip()
        if text:
            box[i] = json.loads(text)
    return box, np.isfinite(box).all(1)


def validate(maps: dict[str, np.ndarray], states: dict[str, dict], labels: list[dict], geometry: dict,
             grid: int, valid: np.ndarray, stride: int) -> dict:
    """Ground truth enters only here. Pointing game, saliency lift and velocity agreement."""
    box, has = _boxes(labels)
    cell = geometry['size'] / grid
    ys, xs = np.divmod(np.arange(grid * grid), grid)
    px = ((xs + .5) * cell - geometry['left']) / geometry['scale']
    py = ((ys + .5) * cell - geometry['top']) / geometry['scale']
    half = .5 * cell / geometry['scale']
    inside = ((px[None] + half >= box[:, None, 0]) & (px[None] - half <= box[:, None, 0] + box[:, None, 2])
              & (py[None] + half >= box[:, None, 1]) & (py[None] - half <= box[:, None, 1] + box[:, None, 3])
              & valid[None])                                        # (T, P) patch overlaps the box
    centre = np.stack([box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2], 1)
    pair = has[1:] & has[:-1]
    v_true = (centre[1:] - centre[:-1])[pair] / stride
    chance = float((inside[has].sum(1) / valid.sum()).mean()) if has.any() else float('nan')
    report = dict(frames=len(labels), frames_with_box=int(has.sum()), chance_pointing=chance)
    for name, sal in maps.items():
        s = np.where(valid[None], sal, -np.inf)
        hit = inside[np.arange(len(s)), np.argmax(s, 1)][has]
        mass = np.where(valid[None], sal, 0.0)
        share = (mass * inside).sum(1)[has] / np.maximum(mass.sum(1)[has], 1e-12)
        area = inside[has].sum(1) / valid.sum()
        v_est = (states[name]['centroid'][1:] - states[name]['centroid'][:-1])[pair] / stride
        err = np.linalg.norm(states[name]['centroid'][has] - centre[has], axis=1)
        corr = [float(np.corrcoef(v_est[:, k], v_true[:, k])[0, 1]) if len(v_true) > 3 else float('nan')
                for k in (0, 1)]
        report[name] = dict(pointing_accuracy=float(hit.mean()) if len(hit) else float('nan'),
                            saliency_lift=float((share / np.maximum(area, 1e-12)).mean()) if len(hit) else float('nan'),
                            centroid_error_px_median=float(np.median(err)) if len(err) else float('nan'),
                            velocity_corr_x=corr[0], velocity_corr_y=corr[1])
    return report


def analyse_clip(path: Path, **kw) -> tuple[dict, dict, dict]:
    data = np.load(path, allow_pickle=False)
    tokens = data['tokens'].astype(np.float32)
    grid = int(data['grid'])
    geometry = json.loads(str(data['geometry']))
    valid = valid_patches(geometry, grid)
    maps, spectra = saliency_maps(tokens, **kw)
    states = {name: tool_state(sal, tokens, geometry, grid, valid) for name, sal in maps.items()}
    labels = json.loads(path.with_suffix('').with_suffix('.labels.json').read_text(encoding='utf-8'))
    report = validate(maps, states, labels, geometry, grid, valid, int(data['stride']))
    lam = np.concatenate(spectra) if spectra else np.zeros(0, complex)
    report['spectrum'] = dict(modes=int(len(lam)), slow_fraction=float((np.abs(np.log(lam + 1e-12)) < kw.get('slow_tol', .05)).mean()),
                              median_modulus=float(np.median(np.abs(lam))) if len(lam) else float('nan'))
    meta = dict(video_id=str(data['video_id']), group_id=str(data['group_id']),
                frame_numbers=data['frame_numbers'], valid=valid, grid=grid, geometry=geometry)
    return report, states, dict(maps=maps, meta=meta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokens', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--window', type=int, default=24)
    ap.add_argument('--hop', type=int, default=12)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--slow-tol', type=float, default=.05)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for path in sorted(args.tokens.glob('*.npz')):
        report, states, extra = analyse_clip(path, window=args.window, hop=args.hop, rank=args.rank,
                                             slow_tol=args.slow_tol)
        reports[path.stem] = report
        best = states['dmd_slow_modes']
        np.savez_compressed(args.out / f'{path.stem}.state.npz', frame_numbers=extra['meta']['frame_numbers'],
                            centroid=best['centroid'], spread=best['spread'], descriptor=best['descriptor'],
                            concentration=best['concentration'],
                            saliency=extra['maps']['dmd_slow_modes'].astype(np.float32),
                            video_id=extra['meta']['video_id'], group_id=extra['meta']['group_id'])
        print(path.stem, {m: {k: round(v, 3) for k, v in report[m].items()} for m in METHODS}, flush=True)
    summary = {m: {k: float(np.nanmean([r[m][k] for r in reports.values()])) for k in
                   ('pointing_accuracy', 'saliency_lift', 'centroid_error_px_median', 'velocity_corr_x', 'velocity_corr_y')}
               for m in METHODS}
    summary['chance_pointing'] = float(np.nanmean([r['chance_pointing'] for r in reports.values()]))
    summary['clips'] = len(reports)
    (args.out / 'koopman_tool_report.json').write_text(json.dumps(dict(summary=summary, clips=reports,
        config=vars(args) | {'tokens': str(args.tokens), 'out': str(args.out)},
        labels_used_for_saliency=False, labels_used_for_validation_only=True), indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps({k: ({a: round(b, 3) for a, b in v.items()} if isinstance(v, dict) else v)
                                 for k, v in summary.items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
