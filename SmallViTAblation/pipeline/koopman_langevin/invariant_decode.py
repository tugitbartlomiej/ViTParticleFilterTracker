#!/usr/bin/env python3
"""From the latent back to the image: what the Koopman invariants of the instrument encode, and
which frames encode them well.

Three steps on a patch-particle recording (`patch_particles.py`), labels for scoring only:

1. IDENTITY CODE. Translational Koopman operator on instrument patches (found with the detector's
   own box, label-free); its slow eigenfunctions are the invariants of translation. Every frame
   gets the mean invariant code of its instrument patches, a prototype is the mean code over
   confident frames, and the CLOSURE error of a frame is the distance of its code from the
   prototype (how well this frame "closes" the identity). Checked against the ground truth: does
   closure error mark detector misses, and how does it compare with confidence?

2. DECODING. A linear pixel decoder (ridge) from the FULL 384-dim profile of a patch token to its
   16x16x3 pixels, fitted on held-in frames, scored on held-out frames. The decoder is then applied
   to the token projected on (a) the slow invariant subspace, (b) the fast subspace, (c) the leading
   PCA subspace, which shows in pixels what each subspace keeps. The decoded prototype is the
   "instrument the network believes in". Montages are written as PNG.

3. RETRIEVAL. Frames ranked by closure error: the best-closed and worst-closed instrument crops are
   written as montages; the ranking is the label-free query for "frames like the ones that encode
   the identity well" in a larger pool.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.invariant_decode \
        --package feature_cache/detector_probe_clips --recording results/.../czastki/CZASTKI_b2500 \
        --out results/koopman_langevin_epochs/decode_b2500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import epoch_dynamics as ed
from .kalman_state import auc
from .patch_analysis import generalised_koopman, grid_masks, load_recording, translational_pairs


def frame_pixels(path: Path, side: int, grid: int) -> np.ndarray:
    """Patches [grid*grid, 16*16*3] of the frame resized the way the detector sees it (square)."""
    im = np.asarray(Image.open(path).convert('RGB').resize((side, side), Image.BILINEAR), dtype=np.float32) / 255
    p = side // grid
    return im.reshape(grid, p, grid, p, 3).transpose(0, 2, 1, 3, 4).reshape(grid * grid, p * p * 3)


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float):
    mx, my = x.mean(0), y.mean(0)
    xc = x - mx
    gram = xc.T @ xc + lam * np.eye(x.shape[1])
    w = np.linalg.solve(gram, xc.T @ (y - my))
    return mx, my, w


def montage(rows: list[list[np.ndarray]], cell: int, out: Path):
    canvas = np.ones((len(rows) * cell, len(rows[0]) * cell, 3), np.uint8) * 255
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            tile = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((cell, cell), Image.NEAREST))
            canvas[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = tile
    Image.fromarray(canvas).save(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--recording-small', type=Path, help='recording of the small-data detector; enables the removable-error check')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--lam', type=float, default=10.0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    rec = load_recording(args.recording)
    grid = tuple(rec['meta']['grid'])
    g = grid[0]
    side = rec['meta']['resolution']
    truth = grid_masks(tracks, rec['frame_ids'], grid)
    clips = np.array([f.split('/')[-2] for f in rec['frame_ids']])
    val = np.array([tracks[c]['split'] == 'val' for c in clips])
    prof = rec['profil_red'].astype(np.float64)                          # [T, N, r]
    score, mask = rec['score'].astype(float), rec['maska_pred']

    # ---- 1. identity code and closure
    x0, x1 = translational_pairs(rec, grid)
    lam, modes, mu = generalised_koopman(x0, x1)
    slow, fast = modes[:, :args.modes], modes[:, -args.modes:]
    code = np.full((len(prof), args.modes), np.nan)
    spread = np.full(len(prof), np.nan)
    for t in range(len(prof)):
        if score[t] > .5 and mask[t].sum() >= 4:
            c = (prof[t, mask[t]] - mu) @ slow
            code[t], spread[t] = c.mean(0), float(np.linalg.norm(c - c.mean(0), axis=1).mean())
    confident = (score > .8) & np.isfinite(code).all(1) & ~val
    prototype = code[confident].mean(0)
    closure = np.linalg.norm(code - prototype, axis=1)
    box_truth = np.isfinite(rec['pred_box']).all(1)
    gt_box = np.full((len(prof), 4), np.nan)
    at = {(c, n): (c, i) for c, t in tracks.items() for i, n in enumerate(t['frame'])}
    for k, fid in enumerate(rec['frame_ids']):
        c, i = at[(fid.split('/')[-2], int(fid.split('/')[-1]))]
        if tracks[c]['box'][i] is not None:
            gt_box[k] = tracks[c]['box'][i]
    iou = ed.iou(rec['pred_box'], gt_box)
    labelled = np.isfinite(gt_box).all(1)
    m = val & labelled & np.isfinite(closure)
    report = dict(pairs=int(len(x0)), lambda_slow=[float(v) for v in lam[:args.modes]], frames_with_code=int(np.isfinite(closure).sum()),
                  closure=dict(auc_miss_from_closure=auc(closure[m], iou[m] < .5), auc_miss_from_spread=auc(spread[m], iou[m] < .5),
                               auc_miss_from_confidence=auc(-score[m], iou[m] < .5),
                               corr_closure_iou=float(np.corrcoef(closure[m], iou[m])[0, 1])))
    print(f"kod tozsamosci: {report['frames_with_code']} klatek; chybienie z domkniecia AUC {report['closure']['auc_miss_from_closure']:.3f}, "
          f"z rozrzutu {report['closure']['auc_miss_from_spread']:.3f}, z pewnosci {report['closure']['auc_miss_from_confidence']:.3f}", flush=True)
    if args.recording_small:
        small = load_recording(args.recording_small)
        assert small['frame_ids'] == rec['frame_ids']
        iou_s = ed.iou(small['pred_box'], gt_box)
        removable = (iou_s < .5) & (iou >= .5)
        ps, ms, sc_s = small['profil_red'].astype(np.float64), small['maska_pred'], small['score'].astype(float)
        xs0, xs1 = translational_pairs(small, grid)
        _, modes_s, mu_s = generalised_koopman(xs0, xs1)
        code_s = np.full((len(ps), args.modes), np.nan)
        for t in range(len(ps)):
            if sc_s[t] > .5 and ms[t].sum() >= 4:
                code_s[t] = ((ps[t, ms[t]] - mu_s) @ modes_s[:, :args.modes]).mean(0)
        proto_s = code_s[(sc_s > .8) & np.isfinite(code_s).all(1) & ~val].mean(0)
        closure_s = np.linalg.norm(code_s - proto_s, axis=1)
        mm = val & labelled & np.isfinite(closure_s)
        report['small_detector'] = dict(auc_removable_from_closure=auc(closure_s[mm], removable[mm]),
                                        auc_removable_from_confidence=auc(-sc_s[mm], removable[mm]),
                                        auc_miss_from_closure=auc(closure_s[mm], iou_s[mm] < .5))
        print(f"maly detektor: usuwalny blad z domkniecia AUC {report['small_detector']['auc_removable_from_closure']:.3f}, "
              f"z pewnosci {report['small_detector']['auc_removable_from_confidence']:.3f}", flush=True)

    # ---- 2. pixel decoder on full profiles
    full_nr = rec['profil_pelny_nr']
    full = rec['profil_pelny'].astype(np.float32)                         # [F, N, K]
    K = full.shape[2]
    kier = rec['baza']['kier']                                            # [K, r]
    q_slow, _ = np.linalg.qr(kier @ slow)                                 # orthonormal basis in the full space
    q_fast, _ = np.linalg.qr(kier @ fast)
    q_pca = kier[:, :args.modes]
    fit_frames = np.array([i for i, t in enumerate(full_nr) if not val[t]])
    test_frames = np.array([i for i, t in enumerate(full_nr) if val[t]])
    rng = np.random.default_rng(0)
    fit_sel = rng.choice(fit_frames, min(len(fit_frames), 400), replace=False)
    xs, ys = [], []
    for i in fit_sel:
        t = full_nr[i]
        fid = rec['frame_ids'][t]
        path = args.package / 'frames' / fid.split('/')[-2] / f"{int(fid.split('/')[-1]):08d}.jpg"
        xs.append(full[i]); ys.append(frame_pixels(path, side, g))
    x, y = np.concatenate(xs), np.concatenate(ys)
    mx, my, w = ridge_fit(x, y, args.lam)
    mu_full = mx

    def decode(a, q=None):
        a = a - mu_full
        if q is not None:
            a = a @ q @ q.T
        return (a @ w) + my

    r2 = {}
    tool_r2 = {}
    ys_t, pred = {k: [] for k in ('full', 'slow', 'fast', 'pca')}, {k: [] for k in ('full', 'slow', 'fast', 'pca')}
    tool_flags = []
    for i in test_frames[:120]:
        t = full_nr[i]
        fid = rec['frame_ids'][t]
        path = args.package / 'frames' / fid.split('/')[-2] / f"{int(fid.split('/')[-1]):08d}.jpg"
        pix = frame_pixels(path, side, g)
        tool_flags.append(truth[t] == 1)
        for name, q in (('full', None), ('slow', q_slow), ('fast', q_fast), ('pca', q_pca)):
            pred[name].append(decode(full[i], q)); ys_t[name].append(pix)
    tool_flags = np.concatenate(tool_flags)
    for name in pred:
        p_, y_ = np.concatenate(pred[name]), np.concatenate(ys_t[name])
        r2[name] = float(1 - ((p_ - y_) ** 2).sum() / ((y_ - y_.mean(0)) ** 2).sum())
        tool_r2[name] = float(1 - ((p_[tool_flags] - y_[tool_flags]) ** 2).sum() / ((y_[tool_flags] - y_[tool_flags].mean(0)) ** 2).sum())
    report['pixel_decoder'] = dict(fit_frames=int(len(fit_sel)), r2_all_patches=r2, r2_instrument_patches=tool_r2)
    print('dekoder pikseli R2 (wszystkie / narzedzie): ' + ', '.join(f'{k} {r2[k]:.3f} / {tool_r2[k]:.3f}' for k in r2), flush=True)

    # montage: one validation frame, whole image decoded through each subspace
    show = [i for i in test_frames if score[full_nr[i]] > .8][:1]
    if show:
        i = show[0]; t = full_nr[i]
        fid = rec['frame_ids'][t]
        path = args.package / 'frames' / fid.split('/')[-2] / f"{int(fid.split('/')[-1]):08d}.jpg"
        pix = frame_pixels(path, side, g)
        p = side // g
        def assemble(patches):
            return patches.reshape(g, g, p, p, 3).transpose(0, 2, 1, 3, 4).reshape(side, side, 3)
        row = [assemble(pix), assemble(decode(full[i])), assemble(decode(full[i], q_slow)), assemble(decode(full[i], q_fast)), assemble(decode(full[i], q_pca))]
        montage([row], side, args.out / 'klatka_oryginal_pelny_wolne_szybkie_pca.png')
        # instrument patches of that frame: original | full | slow | fast
        rows = []
        for n in np.flatnonzero(truth[t] == 1)[:10]:
            rows.append([pix[n].reshape(p, p, 3), decode(full[i][n:n + 1])[0].reshape(p, p, 3), decode(full[i][n:n + 1], q_slow)[0].reshape(p, p, 3),
                         decode(full[i][n:n + 1], q_fast)[0].reshape(p, p, 3)])
        if rows:
            montage(rows, 64, args.out / 'patche_narzedzia_oryginal_pelny_wolne_szybkie.png')
    # the prototype decoded: the instrument the network believes in (slow subspace only)
    proto_full = mu_full + (kier @ (mu + prototype @ np.linalg.pinv(slow)) - mu_full) @ q_slow @ q_slow.T if False else None

    # ---- 3. retrieval by closure
    order = np.argsort(closure)
    good = [t for t in order if val[t] and np.isfinite(closure[t])][:8]
    bad = [t for t in order[::-1] if val[t] and np.isfinite(closure[t])][:8]
    def crop(t):
        fid = rec['frame_ids'][t]
        im = Image.open(args.package / 'frames' / fid.split('/')[-2] / f"{int(fid.split('/')[-1]):08d}.jpg").convert('RGB')
        x0_, y0_, w_, h_ = rec['pred_box'][t]
        box = (max(int(x0_) - 10, 0), max(int(y0_) - 10, 0), min(int(x0_ + w_) + 10, im.width), min(int(y0_ + h_) + 10, im.height))
        return np.asarray(im.crop(box).resize((128, 128)), dtype=np.float32) / 255
    montage([[crop(t) for t in good], [crop(t) for t in bad]], 128, args.out / 'wycinki_najlepiej_i_najgorzej_domkniete.png')
    report['retrieval'] = dict(best_closed=[dict(frame=rec['frame_ids'][t], closure=float(closure[t]), iou=float(iou[t]), score=float(score[t])) for t in good],
                               worst_closed=[dict(frame=rec['frame_ids'][t], closure=float(closure[t]), iou=float(iou[t]), score=float(score[t])) for t in bad],
                               prototype=[float(v) for v in prototype])
    np.savez_compressed(args.out / 'kod_tozsamosci.npz', frame_ids=np.array(rec['frame_ids']), code=code, closure=closure, spread=spread,
                        prototype=prototype, slow_modes=slow, mu=mu, slow_full_basis=q_slow)
    (args.out / 'REPORT_decode.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('najlepiej domkniete: IoU', [round(iou[t], 2) for t in good], ' najgorzej:', [round(iou[t], 2) for t in bad])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
