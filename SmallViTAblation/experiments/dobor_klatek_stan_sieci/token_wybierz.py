"""Czy dobór z błędu pojedynczych patchy poprawi uczenie na mniejszej puli?

Test obala brak zysku AP względem straty i placebo. Ten etap zamraża pięć
wyborów przed treningiem; nie korzysta z AP walidacji do strojenia selektora.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .dane import read_jsonl, sha256, write_json
from .wybierz import build_variants


def load_tokens(pool_path, cache, initial_selection):
    cache = Path(cache)
    all_pool = read_jsonl(pool_path)
    pool = [r for r in all_pool if r.get('split') == 'train']
    evaluation_groups = {str(r['group_id']) for r in all_pool if r.get('split') != 'train'}
    rows = read_jsonl(cache/'rows.jsonl')
    meta = json.loads((cache/'meta.json').read_text(encoding='utf-8'))
    if meta['spatial_pooling'] or meta['feature'] != 'last_raw_vit_patch':
        raise ValueError('particles must be individual raw ViT patches')
    if any(r.get('split') != 'train' or str(r['group_id']) in evaluation_groups for r in rows):
        raise ValueError('non-training data in selector cache')
    index = {str(r['frame_id']): r for r in rows}
    if len(index) != len(rows) or len({r['frame_id'] for r in pool}) != len(pool):
        raise ValueError('duplicate frames')
    for r in pool:
        cached = index.get(str(r['frame_id']))
        if cached is None or any(str(cached[k]) != str(r[k]) for k in ('group_id', 'video_id')):
            raise ValueError('candidate missing or group mismatch in token cache')
    values = np.load(cache/'tokens.npy', mmap_mode='r', allow_pickle=False)
    if sha256(cache/'tokens.npy') != meta['tokens_sha256']:
        raise ValueError('token contents differ from extraction manifest')
    inputs = np.load(cache/'inputs.npy', allow_pickle=False)
    if values.shape != (len(rows), int(np.prod(meta['grid_shape'])), meta['projected_channels']):
        raise ValueError('invalid token cache shape')
    if inputs.shape != (len(rows), 6) or not np.isfinite(inputs).all():
        raise ValueError('invalid forcing inputs')
    initial = json.loads(Path(initial_selection).read_text(encoding='utf-8'))
    l0 = initial.get('frame_ids', initial.get('selected_frame_ids'))
    if not l0 or len(set(l0)) != len(l0) or not set(l0) <= {r['frame_id'] for r in pool}:
        raise ValueError('invalid initial selection')
    return pool, rows, values, inputs, l0, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pool', 'cache', 'initial-selection', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-windows', type=int, default=20000)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('selection output must be new')
    start = time.monotonic()
    pool, rows, tokens, inputs, l0, meta = load_tokens(args.pool, args.cache, args.initial_selection)
    from .token_dopasowanie import fit_token_cross_fitted
    args.out.mkdir(parents=True)
    result = fit_token_cross_fitted(rows, tokens, inputs, pool, l0, grid_shape=tuple(meta['grid_shape']),
        context=8, step=5, folds=3, epochs=args.epochs, max_windows=args.max_windows,
        seed=42, device=args.device, artifact_dir=args.out/'models')
    selection = build_variants(pool, l0, result['e_obs'], result['e_dyn'], args.out/'variants',
                                budget=1500, large_budget=3000, seed=42)
    np.savez_compressed(args.out/'SCORES.npz', frame_ids=np.array([r['frame_id'] for r in pool]),
                        e_obs=result['e_obs'], e_dyn=result['e_dyn'], e_persistence=result['e_persistence'])
    with (args.out/'DIAGNOSTYKA_KLATEK.jsonl').open('x', encoding='utf-8') as stream:
        for row in result['diagnostics']:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    write_json(args.out/'BIEG_selekcja.json', dict(status='complete', particle='one patch token',
        selection=selection, fit=result['audit'], feature_meta=meta, labels_used_for_ranking=True,
        source_sha256={f.name:sha256(f) for f in sorted(Path(__file__).parent.glob('*.py'))},
        pool_sha256=sha256(args.pool), l0_sha256=sha256(args.initial_selection),
        elapsed_seconds=time.monotonic()-start, test_used=False))
    print(json.dumps(dict(status='complete', positive_gain=selection['positive_gain'],
                          missing=selection['missing_temporal_windows'], seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
