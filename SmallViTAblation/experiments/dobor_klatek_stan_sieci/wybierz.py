"""Czy zadaniowy zysk z historii daje użyteczne klatki? Zapisuje zamrożony wybór.

Lepsza predykcja latentu nie potwierdza selektora: hipotezę obala brak poprawy
AP po identycznym treningu względem readout_loss oraz placebo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

from .dane import load_data, sha256, write_json
from .dopasowanie import fit_cross_fitted
from .selekcja import recoverable_scores, placebo_scores, rank_select


def build_variants(pool, l0, e_obs, e_dyn, out, *, budget=1500, seed=42, min_gap_frames=5, large_budget=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    missing = ~np.isfinite(e_dyn)
    safe_dyn = np.where(missing, e_obs, e_dyn)
    scores = recoverable_scores(e_obs, safe_dyn)
    placebo, placebo_meta = placebo_scores(pool, e_obs, safe_dyn, seed=seed, eligible=~missing)
    names = dict(random=None, readout_loss=e_obs, koopman_langevin=scores, placebo=placebo)
    if large_budget is not None:
        if large_budget <= budget:
            raise ValueError('large_budget must exceed the primary budget')
        names['random_large'] = None
    arms = {}
    for name, score in names.items():
        arm_budget = large_budget if name == 'random_large' else budget
        added, meta = rank_select(pool, score, l0, arm_budget-len(l0), seed, min_gap_frames=min_gap_frames)
        ids = list(l0)+list(added)
        if len(ids) != arm_budget or len(set(ids)) != arm_budget:
            raise ValueError(f'{name}: invalid selector budget')
        vid = f'stan_{name}_b{arm_budget}_sel{seed}'
        write_json(out/f'{vid}.json', dict(schema_version=1, variant_id=vid, run_id=vid,
            strategy=name, scope='labelled_latent_task_gap_pilot', seed=seed,
            selection_seed=seed, budget=arm_budget, frame_ids=ids, labels_used_for_ranking=score is not None,
            selector_meta=meta, initial_count=len(l0), test_used_for_selection=False))
        arms[name] = dict(file=f'{vid}.json', selected=len(ids), metadata=meta,
            sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
    for name in arms:
        chosen = json.loads((out/arms[name]['file']).read_text())['frame_ids'][len(l0):]
        for other in arms:
            ids = json.loads((out/arms[other]['file']).read_text())['frame_ids'][len(l0):]
            arms[name].setdefault('overlap_added', {})[other] = len(set(chosen)&set(ids))/len(chosen)
    return dict(arms=arms, missing_temporal_windows=int(missing.sum()),
                positive_gain=int((scores>0).sum()), placebo=placebo_meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('pool','cache','initial-selection','out'):
        ap.add_argument('--'+arg, type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--budget', type=int, default=1500)
    ap.add_argument('--context', type=int, default=8)
    ap.add_argument('--step', type=float, default=5)
    ap.add_argument('--dim', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--max-windows', type=int, default=20000)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Use a new immutable output directory: {args.out}')
    started = time.monotonic()
    pool, rows, arrays, l0, feature_meta = load_data(args.pool,args.cache,args.initial_selection)
    args.out.mkdir(parents=True)
    result = fit_cross_fitted(rows,arrays['features'],arrays['inputs'],arrays['observed'],pool,l0,
        context=args.context,step=args.step,dim=args.dim,epochs=args.epochs,
        max_windows=args.max_windows,seed=args.seed,device=args.device,artifact_dir=args.out/'models')
    e_obs,e_dyn=result['e_obs'],result['e_dyn']
    selection = build_variants(pool,l0,e_obs,e_dyn,args.out/'variants',budget=args.budget,seed=args.seed)
    np.savez_compressed(args.out/'SCORES.npz',frame_ids=np.array([r['frame_id'] for r in pool]),
        e_obs=e_obs,e_dyn=e_dyn,pred_obs_box=result['pred_obs_box'],pred_dyn_box=result['pred_dyn_box'])
    source = {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    write_json(args.out/'BIEG_selekcja.json',dict(status='complete',question='task_gap_frame_acquisition',
        selection=selection,fit=result['audit'],source_sha256=source,feature_meta=feature_meta,
        pool_sha256=sha256(args.pool),cache_sha256=sha256(args.cache),l0_sha256=sha256(args.initial_selection),
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        elapsed_seconds=time.monotonic()-started,labels_used_for_ranking=True,test_used=False,
        interpretation='Hypothesis of training usefulness; not proof of epistemic uncertainty'))
    print(json.dumps(dict(out=str(args.out),positive_gain=selection['positive_gain'],
        missing=selection['missing_temporal_windows'],elapsed_seconds=time.monotonic()-started)),flush=True)


if __name__ == '__main__':
    main()
