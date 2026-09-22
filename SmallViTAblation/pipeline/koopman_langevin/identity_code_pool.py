#!/usr/bin/env python3
"""Identity code of the instrument for every frame of the label-eligible pool (label-free).

Input: a patch-particle recording of the CURRENT detector (L0) on the sparse pool frames
(`patch_particles.py --frame-list`). The translational Koopman operator is fitted on pairs of
instrument patches at different positions inside the same video (positions from the detector's
own boxes), its slow eigenfunctions are the invariants of translation, and each frame gets

    code_f     mean invariant coordinates of its instrument patches      (dim = modes)
    closure_f  distance of code_f from the prototype (mean over confident frames)
    spread_f   within-frame spread of the instrument patches in the invariant subspace

Frames without a confident box get NaN. Output is small (frame_id + a few numbers), meant to
feed `selection_matrix`-style coverage in the identity-code space ("frames like the ones that
encode the identity", the author's idea of 2026-09-22).

    python -m SmallViTAblation.pipeline.koopman_langevin.identity_code_pool \
        --recording <dir>/CZASTKI_pool_b0500 --out <dir>/KOD_TOZSAMOSCI_pool_b0500.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .patch_analysis import generalised_koopman, load_recording, translational_pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--min-score', type=float, default=.5)
    args = ap.parse_args(argv)
    rec = load_recording(args.recording)
    grid = tuple(rec['meta']['grid'])
    x0, x1 = translational_pairs(rec, grid, per_clip=600)
    lam, modes, mu = generalised_koopman(x0, x1)
    slow = modes[:, :args.modes]
    prof, score, mask = rec['profil_red'].astype(np.float64), rec['score'].astype(float), rec['maska_pred']
    n = len(prof)
    code = np.full((n, args.modes), np.nan)
    spread = np.full(n, np.nan)
    for t in range(n):
        if score[t] > args.min_score and mask[t].sum() >= 4:
            c = (prof[t, mask[t]] - mu) @ slow
            code[t], spread[t] = c.mean(0), float(np.linalg.norm(c - c.mean(0), axis=1).mean())
    ok = np.isfinite(code).all(1)
    prototype = code[ok & (score > .8)].mean(0)
    closure = np.linalg.norm(code - prototype, axis=1)
    np.savez_compressed(args.out, frame_ids=np.array(rec['frame_ids']), code=code.astype(np.float32), closure=closure.astype(np.float32),
                        spread=spread.astype(np.float32), score=score.astype(np.float32), prototype=prototype,
                        lambda_slow=lam[:args.modes], pairs=len(x0))
    print(json.dumps(dict(frames=int(n), with_code=int(ok.sum()), pairs=int(len(x0)), lambda_slow=[round(float(v), 3) for v in lam[:args.modes]],
                          closure_median=float(np.nanmedian(closure)), out=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
