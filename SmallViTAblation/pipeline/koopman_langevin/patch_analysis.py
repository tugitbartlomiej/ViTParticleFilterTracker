#!/usr/bin/env python3
"""Analysis of patch-token particle recordings (`patch_particles.py`) along training.

Three questions from the grant theory (`Research/DiagnosticTool/test/`), transplanted from
DINOv2-on-DAVIS to the RF-DETR backbone on cataract clips, plus the training clock:

  guards       radius = loudness before normalisation (T24c), dead direction (T3c),
               determinism (T35); registers (T6a): grid places with anomalous radius that stay
               anomalous whatever the frame
  basins (03)  micro-states (k-means on direction-normalised profiles of one clip) matched to
               the INSTRUMENT mask of the grid by a greedy union; floor = smooth random
               partitions (Voronoi of random grid seeds, same k, 100 draws, quantile 95).
               Test of the GEOMETRY of the embedding, not of dynamics
  Koopman      two operators on the same particles (DiagnosticTool D65):
               TRANSLATIONAL  pairs = patches of the instrument at two different positions of
                              the frame (found with the detector's OWN box, label-free), matched
                              by their relative place inside the box; slow modes = invariants of
                              translation = candidates for the identity code of the instrument
               TEMPORAL       pairs = the same grid place in consecutive frames (Euler, T15a);
                              slow modes = slowly varying quantities of the scene
               Identity test: do the slow translational modes separate instrument from
               background patches (labels for scoring only) as well as the full profile,
               better than the same number of leading PCA directions?

Every number is reported per checkpoint, so the training clock is the outer axis.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.patch_analysis \
        --package feature_cache/detector_probe_clips --recordings results/.../czastki \
        --tags coco b0500 b2500 ep005 ep029 --out REPORT_patch_particles.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..latent_dynamics.gle import _ridge_fit_predict
from .kalman_state import auc


# --------------------------------------------------------------------------- loading

def load_recording(directory: Path) -> dict:
    with np.load(directory / 'czesc000.npz', allow_pickle=False) as data:
        rec = {k: data[k] for k in data.files}
    with np.load(directory / 'baza.npz', allow_pickle=False) as data:
        rec['baza'] = {k: data[k] for k in data.files}
    rec['meta'] = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
    rec['frame_ids'] = [str(f) for f in rec['frame_ids'].tolist()]
    return rec


def grid_masks(tracks: dict, frame_ids: list[str], grid: tuple[int, int], native=(720, 480)) -> np.ndarray:
    """Instrument mask on the patch grid from the ground-truth box: 1 inside, -1 on the boundary band,
    0 background, -2 when the frame has no annotation. Labels: scoring only."""
    h, w = grid
    ys = (np.arange(h) + .5) / h
    xs = (np.arange(w) + .5) / w
    out = np.full((len(frame_ids), h * w), -2, dtype=np.int8)
    at = {(c, n): (c, i) for c, t in tracks.items() for i, n in enumerate(t['frame'])}
    for k, fid in enumerate(frame_ids):
        clip, n = fid.split('/')[-2], int(fid.split('/')[-1])
        c, i = at[(clip, n)]
        box = tracks[c]['box'][i]
        if box is None:
            continue
        x0, y0 = box[0] / native[0], box[1] / native[1]
        x1, y1 = x0 + box[2] / native[0], y0 + box[3] / native[1]
        inside = (xs[None, :] >= x0) & (xs[None, :] <= x1) & (ys[:, None] >= y0) & (ys[:, None] <= y1)
        band = ((np.abs(xs[None, :] - x0) < .5 / w) | (np.abs(xs[None, :] - x1) < .5 / w)) & (ys[:, None] >= y0 - .5 / h) & (ys[:, None] <= y1 + .5 / h)
        band |= ((np.abs(ys[:, None] - y0) < .5 / h) | (np.abs(ys[:, None] - y1) < .5 / h)) & (xs[None, :] >= x0 - .5 / w) & (xs[None, :] <= x1 + .5 / w)
        m = np.where(band, -1, np.where(inside, 1, 0)).astype(np.int8)
        out[k] = m.ravel()
    return out


# --------------------------------------------------------------------------- basins (test 03)

def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = (x ** 2).sum(1)[:, None] - 2 * x @ centres.T + (centres ** 2).sum(1)[None]
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                centres[j] = x[m].mean(0)
    return lab


def greedy_union_iou(labels: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Sort micro-states by precision, add them while the IoU with the instrument mask grows."""
    valid = truth >= 0
    lab, t = labels[valid], truth[valid] == 1
    prec = np.array([(t[lab == j].mean() if (lab == j).any() else 0.0) for j in range(k)])
    best, chosen = 0.0, np.zeros(len(lab), bool)
    for j in np.argsort(-prec):
        trial = chosen | (lab == j)
        iou = (trial & t).sum() / max((trial | t).sum(), 1)
        if iou > best:
            best, chosen = iou, trial
        else:
            break
    return float(best)


def basins_for_clip(profiles: np.ndarray, truth: np.ndarray, grid: tuple[int, int], k: int, seed: int,
                    draws: int = 100) -> dict:
    """profiles [T, N, r], truth [T, N]. Micro-states over ALL frames of the clip."""
    t_len, n, r = profiles.shape
    x = profiles.reshape(-1, r).astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)          # direction only
    lab = kmeans(x, k, seed)
    iou = greedy_union_iou(lab, truth.ravel(), k)
    # smooth random floor: Voronoi of random seeds on the grid, the same for every frame
    h, w = grid
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), -1).reshape(-1, 2).astype(float)
    rng = np.random.default_rng(seed + 1)
    floor = []
    for _ in range(draws):
        seeds = coords[rng.choice(len(coords), k, replace=False)] + rng.normal(scale=.3, size=(k, 2))
        vor = ((coords[:, None, :] - seeds[None]) ** 2).sum(2).argmin(1)
        floor.append(greedy_union_iou(np.tile(vor, t_len), truth.ravel(), k))
    return dict(iou=iou, floor_q95=float(np.quantile(floor, .95)), floor_median=float(np.median(floor)),
                instrument_share=float((truth == 1).mean()))


# --------------------------------------------------------------------------- Koopman (D65)

def generalised_koopman(x0: np.ndarray, x1: np.ndarray, reg: float = 1e-3):
    """Symmetrised generalised eigenproblem C01_sym v = lambda C00 v, solved after whitening."""
    mu = x0.mean(0)
    a, b = x0 - mu, x1 - mu
    c00 = a.T @ a / len(a) + reg * np.eye(a.shape[1])
    c01 = (a.T @ b + b.T @ a) / (2 * len(a))
    vals, vecs = np.linalg.eigh(c00)
    white = vecs @ np.diag(1 / np.sqrt(vals)) @ vecs.T
    lam, v = np.linalg.eigh(white @ c01 @ white)
    order = np.argsort(-lam)
    return lam[order], white @ v[:, order], mu


def translational_pairs(rec: dict, grid: tuple[int, int], min_shift: float = 3.0, per_clip: int = 400, seed: int = 0):
    """Instrument patches at two positions, matched by relative place inside the detector's own box."""
    h, w = grid
    rng = np.random.default_rng(seed)
    clips = np.array([f.split('/')[-2] for f in rec['frame_ids']])
    box = rec['pred_box'].astype(float)
    score = rec['score'].astype(float)
    x0, x1 = [], []
    for clip in np.unique(clips):
        idx = np.flatnonzero((clips == clip) & (score > .5) & (box[:, 2] > 40) & (box[:, 3] > 20))
        if len(idx) < 10:
            continue
        for _ in range(per_clip):
            i, j = rng.choice(idx, 2, replace=False)
            ci = (box[i, 0] + box[i, 2] / 2) / 720 * w, (box[i, 1] + box[i, 3] / 2) / 480 * h
            cj = (box[j, 0] + box[j, 2] / 2) / 720 * w, (box[j, 1] + box[j, 3] / 2) / 480 * h
            if np.hypot(ci[0] - cj[0], ci[1] - cj[1]) < min_shift:
                continue
            for gi in np.flatnonzero(rec['maska_pred'][i]):
                ry, rx = divmod(int(gi), w)
                u = ((rx + .5) / w * 720 - box[i, 0]) / max(box[i, 2], 1)          # relative place inside box i
                v = ((ry + .5) / h * 480 - box[i, 1]) / max(box[i, 3], 1)
                px = int((box[j, 0] + u * box[j, 2]) / 720 * w)
                py = int((box[j, 1] + v * box[j, 3]) / 480 * h)
                if 0 <= px < w and 0 <= py < h and rec['maska_pred'][j, py * w + px]:
                    x0.append(rec['profil_red'][i, gi]); x1.append(rec['profil_red'][j, py * w + px])
    return np.asarray(x0, float), np.asarray(x1, float)


def temporal_pairs(rec: dict, max_pairs: int = 200000, seed: int = 0):
    ids = rec['frame_ids']
    clips = np.array([f.split('/')[-2] for f in ids])
    nums = np.array([int(f.split('/')[-1]) for f in ids])
    step = np.diff(nums)
    follow = np.flatnonzero((clips[1:] == clips[:-1]) & (step == np.median(step)))
    rng = np.random.default_rng(seed)
    n = rec['profil_red'].shape[1]
    frames = rng.choice(follow, min(len(follow), max_pairs // n), replace=False)
    x0 = rec['profil_red'][frames].reshape(-1, rec['profil_red'].shape[2]).astype(float)
    x1 = rec['profil_red'][frames + 1].reshape(-1, rec['profil_red'].shape[2]).astype(float)
    return x0, x1


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def identity_auc(rec: dict, truth: np.ndarray, features: np.ndarray, folds: int = 2) -> float:
    """Cross-validated by clip: ridge read-out of instrument (1) vs background (0) patches."""
    clips = np.array([f.split('/')[-2] for f in rec['frame_ids']])
    names = sorted(set(clips.tolist()))
    fold_of = {c: i % folds for i, c in enumerate(names)}
    rng = np.random.default_rng(0)
    t_idx, n_idx = np.nonzero(truth >= 0)
    keep = rng.choice(len(t_idx), min(len(t_idx), 120000), replace=False)
    t_idx, n_idx = t_idx[keep], n_idx[keep]
    x = features[t_idx, n_idx]
    y = (truth[t_idx, n_idx] == 1).astype(float)
    fold = np.array([fold_of[c] for c in clips[t_idx]])
    scores = np.full(len(y), np.nan)
    for f in range(folds):
        fit, new = fold != f, fold == f
        scores[new] = _ridge_fit_predict(x[fit], y[fit, None], x[new], 1e-2)[:, 0]
    return auc(scores, y == 1)


# --------------------------------------------------------------------------- driver

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--recordings', type=Path, required=True)
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--k', type=int, default=96)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--basin-clips', type=int, default=6)
    args = ap.parse_args(argv)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    report = {}
    for tag in args.tags:
        rec = load_recording(args.recordings / f'CZASTKI_{tag}')
        grid = tuple(rec['meta']['grid'])
        truth = grid_masks(tracks, rec['frame_ids'], grid)
        clips = np.array([f.split('/')[-2] for f in rec['frame_ids']])
        val = np.array([tracks[c]['split'] == 'val' for c in clips])
        g = rec['meta']['guards']
        # registers: places whose radius is anomalous in most frames
        radius = rec['promien'].astype(np.float32)
        z = (radius - radius.mean(1, keepdims=True)) / np.maximum(radius.std(1, keepdims=True), 1e-6)
        constant_anomaly = (np.abs(z) > 3).mean(0)
        out = dict(guards=g, registers=dict(places_anomalous_in_over_half_of_frames=int((constant_anomaly > .5).sum()),
                                            top_place_share=float(constant_anomaly.max())))
        # basins on validation clips
        basins = {}
        for clip in sorted(set(clips[val].tolist()))[:args.basin_clips]:
            m = clips == clip
            basins[clip] = basins_for_clip(rec['profil_red'][m], truth[m], grid, args.k, seed=0)
        out['basins'] = dict(per_clip=basins, iou_mean=float(np.mean([b['iou'] for b in basins.values()])),
                             floor_q95_mean=float(np.mean([b['floor_q95'] for b in basins.values()])),
                             clips_above_floor=int(sum(b['iou'] > b['floor_q95'] for b in basins.values())))
        # Koopman: translational and temporal
        t0, t1 = translational_pairs(rec, grid)
        lam_tr, modes_tr, mu_tr = generalised_koopman(t0, t1)
        s0, s1 = temporal_pairs(rec)
        lam_tm, modes_tm, mu_tm = generalised_koopman(s0, s1)
        rng = np.random.default_rng(1)
        perm = rng.permutation(len(t1))
        lam_floor, _, _ = generalised_koopman(t0, t1[perm])                      # pairs broken: floor of eigenvalues
        cos = principal_cosines(modes_tr[:, :args.modes], modes_tm[:, :args.modes])
        # identity read-out: slow translational modes vs leading PCA vs full reduced profile
        prof = rec['profil_red'].astype(np.float32)
        slow = (prof - mu_tr) @ modes_tr[:, :args.modes]
        fast = (prof - mu_tr) @ modes_tr[:, -args.modes:]
        pca = prof[:, :, :args.modes]
        out['koopman'] = dict(translational_pairs=int(len(t0)), temporal_pairs=int(len(s0)),
                              lambda_translational=[float(v) for v in lam_tr[:args.modes]],
                              lambda_translational_floor=[float(v) for v in lam_floor[:args.modes]],
                              lambda_temporal=[float(v) for v in lam_tm[:args.modes]],
                              slow_subspace_cosines=[float(c) for c in cos],
                              identity_auc=dict(slow_translational=identity_auc(rec, truth, slow),
                                                fast_translational=identity_auc(rec, truth, fast),
                                                leading_pca=identity_auc(rec, truth, pca),
                                                full_reduced=identity_auc(rec, truth, prof)))
        report[tag] = out
        k = out['koopman']
        print(f"{tag:6s} promien med {g['radius_median']:.3f} martwy {g['dead_direction_share_mean']:.1e} determinizm {g['determinism_max_abs_diff']:.1e} | "
              f"baseny IoU {out['basins']['iou_mean']:.3f} podloga {out['basins']['floor_q95_mean']:.3f} ({out['basins']['clips_above_floor']}/{len(basins)} klipow nad) | "
              f"lambda transl {k['lambda_translational'][0]:.2f}/{k['lambda_translational'][args.modes - 1]:.2f} podloga {k['lambda_translational_floor'][0]:.2f} | "
              f"cos podprzestrzeni {np.mean(cos):.2f} | tozsamosc AUC wolne {k['identity_auc']['slow_translational']:.3f} szybkie {k['identity_auc']['fast_translational']:.3f} "
              f"PCA {k['identity_auc']['leading_pca']:.3f} pelny {k['identity_auc']['full_reduced']:.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
