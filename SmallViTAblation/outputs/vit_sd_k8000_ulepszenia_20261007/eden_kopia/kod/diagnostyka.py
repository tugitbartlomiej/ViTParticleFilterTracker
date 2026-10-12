"""DDIM diagnosis on the verified step-7500 checkpoint, no training/media writes.

torchrun --nproc_per_node=4 diagnostyka.py --manifest ... --vit ...
    --checkpoint ... --output ... [--smoke]

VAL: 3 fixed films x fraction .33 x 2 seeds, 6 profiles distributed by rank.
TRAIN: 3 fixed films at the same fraction/seeds, selected profile ONLY.
Smoke: 1 film per split, 1 seed, 2 steps; does not select a production sampler.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import sys
import time
import uuid

sys.dont_write_bytecode = True

try:
    from .evaluation import aggregate_records, evaluate, fixed_samples
except ImportError:
    from evaluation import aggregate_records, evaluate, fixed_samples

BASE_SHA = '6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7'
PROFILES = tuple((steps, spacing) for steps in (50, 100, 200) for spacing in ('leading', 'trailing'))
SEEDS = (20261007, 20261008)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    """Only JSON scalar metrics/configuration; unique temp then atomic replace."""
    path = Path(path)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_profile(profiles, tolerance=.02):
    """Fastest measured profile within 2% of minimum film-weighted VAL error.

    Conservative fallback: a gain must exceed 2% overall, improve every paired
    film and every seed's film-weighted mean, on >=3 films and >=2 seeds. This is
    a repeatability heuristic for this small fixed cohort, NOT a significance
    test. Without that evidence retain 50/leading, even outside the 2% band.
    """
    if not isinstance(tolerance, Real) or not math.isfinite(tolerance) or not 0 <= tolerance <= .1:
        raise ValueError('tolerance must lie in [0,.1]')
    if not profiles:
        raise ValueError('No sampler profiles')
    prepared, seen = [], set()
    expected = None
    for profile in profiles:
        steps, spacing = profile['steps'], profile['spacing']
        if isinstance(steps, bool) or not isinstance(steps, Integral) or not 1 <= steps <= 1000:
            raise ValueError('Profile steps outside [1,1000]')
        if spacing not in ('leading', 'trailing') or (steps, spacing) in seen:
            raise ValueError('Invalid/duplicate sampler profile')
        seen.add((steps, spacing))
        elapsed = profile['elapsed_seconds']
        if not isinstance(elapsed, Real) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Measured elapsed_seconds must be positive and finite')
        aggregate = aggregate_records(profile['val']['frames'])
        identities = {(r['film'], r['frame'], r['seed']) for r in aggregate['frames']}
        if not identities:
            raise ValueError('Empty validation observations')
        if expected is None:
            expected = identities
        elif identities != expected:
            raise ValueError('Profiles must evaluate identical film/frame/seed observations')
        errors = [r.get('full_relative_mse') for r in aggregate['frames']]
        if any(e is None or not math.isfinite(e) or e < 0 for e in errors):
            raise ValueError('Each observation needs finite nonnegative full_relative_mse')
        prepared.append(dict(profile=profile, aggregate=aggregate,
                             score=aggregate['film_mean']['full_relative_mse']))
    baseline = next((p for p in prepared if (p['profile']['steps'], p['profile']['spacing']) == (50, 'leading')), None)
    if baseline is None:
        raise ValueError('Production selection requires the 50/leading baseline')
    best = min(prepared, key=lambda p: p['score'])
    band = best['score'] * (1 + tolerance)
    seeds = sorted({r['seed'] for r in baseline['aggregate']['frames']})
    film_count = baseline['aggregate']['films']

    def reliable(candidate):
        if film_count < 3 or len(seeds) < 2 or baseline['score'] <= 0:
            return False
        if 1 - candidate['score'] / baseline['score'] <= tolerance:
            return False
        if any(candidate['aggregate']['per_film'][f]['full_relative_mse'] >= metrics['full_relative_mse']
               for f, metrics in baseline['aggregate']['per_film'].items()):
            return False
        for seed in seeds:
            scores = [aggregate_records([r for r in p['aggregate']['frames'] if r['seed'] == seed])
                      ['film_mean']['full_relative_mse'] for p in (baseline, candidate)]
            if scores[1] >= scores[0]:
                return False
        return True

    candidates = [p for p in prepared if p['score'] <= band + 1e-12]
    has_reliable_gain = any(reliable(p) for p in candidates)
    if has_reliable_gain:
        chosen = min(candidates, key=lambda p: (p['profile']['elapsed_seconds'], p['profile']['steps'], p['profile']['spacing']))
        reason = 'Fastest measured profile within 2% of minimum VAL mean; paired film/seed robustness criterion met.'
        # A faster near-best candidate can itself lack consistent improvement.
        # In that case the conservative baseline takes precedence.
        if not reliable(chosen):
            chosen, has_reliable_gain = baseline, False
            reason = 'Fastest near-best profile lacks reliable paired improvement; retain 50/leading.'
    else:
        chosen = baseline
        reason = 'No reliable gain over baseline on paired VAL films/seeds; retain 50/leading.'
    return dict(steps=int(chosen['profile']['steps']), spacing=chosen['profile']['spacing'], reason=reason,
                reliable_gain=has_reliable_gain, selection_basis='VAL only', tolerance=tolerance,
                selected_val_full_relative_mse=chosen['score'], lowest_val_full_relative_mse=best['score'],
                baseline_val_full_relative_mse=baseline['score'], within_best_two_percent=chosen['score'] <= band + 1e-12,
                elapsed_seconds=chosen['profile']['elapsed_seconds'], films=film_count, seeds=seeds,
                reliability_rule='>2% mean gain, all paired films and all seed means improve, >=3 films and >=2 seeds; heuristic, not significance',
                timing_note='Synchronized wall time for full evaluation, including wrong controls and metrics; same cohort, warmup excluded',
                train_used_for_selection=False)


def _cohort(samples):
    return [dict(film=str(s['film']), frame=int(s['frame']), has_tool=bool(s['has_tool'])) for s in samples]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'vit', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    for key in ('USE_TF', 'USE_FLAX'):
        os.environ.setdefault(key, '0')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

    import torch
    import torch.distributed as dist
    import diffusers
    if __package__:
        from .dane import load_manifest
        from .model import Denoiser, VIT_SHA, new_vae, teacher
    else:
        from dane import load_manifest
        from model import Denoiser, VIT_SHA, new_vae, teacher

    rank, world, local = (int(os.getenv(k, default)) for k, default in
                          (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    if not 0 <= rank < world:
        raise ValueError('Invalid torchrun rank/world size')
    torch.set_num_threads(2)
    device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=2))

    def broadcast(value):
        values = [value]
        if distributed:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    output = Path(args.output)
    run_id = broadcast(uuid.uuid4().hex if rank == 0 else None)
    started = time.perf_counter()
    try:
        # Never publish a result from another invocation or smoke as production.
        # Profile filenames include run_id; aggregate reads only current files.
        actual_sha = sha256(args.checkpoint)
        if actual_sha != BASE_SHA:
            raise ValueError(f'Checkpoint SHA256 mismatch: {actual_sha}')
        actual_vit_sha = sha256(args.vit)
        if actual_vit_sha != VIT_SHA:
            raise ValueError(f'Teacher SHA256 mismatch: {actual_vit_sha}')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        signature = checkpoint.get('signature', {})
        if signature.get('teacher_sha256') != VIT_SHA or signature.get('smoke', True) or checkpoint.get('step') != 7500:
            raise ValueError('Checkpoint teacher, smoke flag or step provenance mismatch')
        scale = float(checkpoint['scale'])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Checkpoint scale is not finite and positive')
        rows = load_manifest(args.manifest)
        nfilms, seeds = (1, SEEDS[:1]) if args.smoke else (3, SEEDS)
        val_samples, val_rejected = fixed_samples(rows, 'val', films=nfilms, fractions=(.33,))
        if len({s['film'] for s in val_samples}) != nfilms:
            raise ValueError(f'Require exactly {nfilms} valid selected VAL films: {val_rejected}')
        cohort = _cohort(val_samples)
        manifest_sha = sha256(args.manifest)
        code_hashes = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        provenance = dict(checkpoint_path=str(Path(args.checkpoint).resolve()), checkpoint_sha256=actual_sha,
                          checkpoint_step=checkpoint['step'], checkpoint_signature=signature,
                          teacher_path=str(Path(args.vit).resolve()), teacher_sha256=actual_vit_sha,
                          manifest_path=str(Path(args.manifest).resolve()), manifest_sha256=manifest_sha,
                          manifest_matches_checkpoint_bytes=manifest_sha == signature.get('manifest_sha256'),
                          manifest_note='Input may relocate archive paths; byte hash comparison is reported, not silently substituted',
                          code_sha256=code_hashes, vae_scale=scale,
                          selected_val_rows=[r for r in rows if r['video_id'] in {s['film'] for s in val_samples}],
                          source_weights='Own random-origin VAE/UNet/adapter trained to step 7500',
                          runtime=dict(python=platform.python_version(), torch=torch.__version__, diffusers=diffusers.__version__,
                                       cuda=torch.version.cuda, world_size=world),
                          media_or_latents_saved=False, frame_cache_saved=False)
        check = dict(cohort=cohort, manifest=manifest_sha, code=code_hashes)
        checks = [None] * world
        if distributed:
            dist.all_gather_object(checks, check)
        else:
            checks[0] = check
        if any(c != check for c in checks):
            raise ValueError('Ranks disagree about source/cohort/code provenance')

        profiles = ((2, 'leading'),) if args.smoke else PROFILES
        assigned = [(i, p) for i, p in enumerate(profiles) if i % world == rank]
        vae = net = vit = None
        if assigned or rank == 0:
            vae, net = new_vae().to(device), Denoiser().to(device)
            vae.load_state_dict(checkpoint['vae'], strict=True)
            net.load_state_dict(checkpoint['net'], strict=True)
            vae.eval().requires_grad_(False)
            net.eval().requires_grad_(False)
            vit = teacher(args.vit, device).eval().requires_grad_(False)
        del checkpoint
        if assigned:
            # One full forward path initializes kernels before profile timing.
            evaluate(vae, net, vit, val_samples[:1], scale, device, steps=1, seeds=seeds[:1])
        if distributed:
            dist.barrier()

        def timed(samples, steps, spacing):
            sync()
            before = time.perf_counter()
            result = evaluate(vae, net, vit, samples, scale, device,
                              steps=steps, spacing=spacing, seeds=seeds)
            sync()
            return result, time.perf_counter() - before

        for index, (steps, spacing) in assigned:
            val, elapsed = timed(val_samples, steps, spacing)
            profile = dict(run_id=run_id, rank=rank, steps=steps, spacing=spacing,
                           seeds=list(seeds), fractions=[.33], smoke=args.smoke,
                           elapsed_seconds=elapsed, val=val, val_rejected=val_rejected,
                           provenance=provenance, train_evaluation='selected profile only')
            atomic_json(output / f'SAMPLER_PROFILE_{run_id}_{index:02d}.json', profile)
            print(json.dumps(dict(event='sampler_profile_complete', rank=rank, steps=steps,
                                  spacing=spacing, val=val['film_mean'], elapsed_seconds=elapsed), allow_nan=False), flush=True)
        if distributed:
            dist.barrier()

        selection = None
        if rank == 0:
            results = [json.loads((output / f'SAMPLER_PROFILE_{run_id}_{i:02d}.json').read_text(encoding='utf-8'))
                       for i in range(len(profiles))]
            if any(r['run_id'] != run_id or r['provenance'] != provenance for r in results):
                raise ValueError('Profile provenance mismatch')
            if args.smoke:
                selection = dict(steps=50, spacing='leading', smoke=True, production_selection_valid=False,
                                 reason='Smoke exercised 2/leading on one film and one seed; production default retained, no gain established',
                                 evaluated_steps=2, evaluated_spacing='leading', reliable_gain=False,
                                 selection_basis='smoke only', train_used_for_selection=False)
                selected = results[0]
            else:
                selection = select_profile(results)
                selection.update(smoke=False, production_selection_valid=True)
                selected = next(r for r in results if (r['steps'], r['spacing']) == (selection['steps'], selection['spacing']))
            train_samples, train_rejected = fixed_samples(rows, 'train', films=nfilms, fractions=(.33,))
            if len({s['film'] for s in train_samples}) != nfilms:
                raise ValueError(f'Require exactly {nfilms} valid selected TRAIN films: {train_rejected}')
            train, train_elapsed = timed(train_samples, selected['steps'], selected['spacing'])
            selection.update(run_id=run_id, checkpoint_sha256=actual_sha, teacher_sha256=actual_vit_sha,
                             manifest_sha256=manifest_sha, code_sha256=code_hashes)
            matched = dict(steps=selected['steps'], spacing=selected['spacing'], seeds=list(seeds), fractions=[.33],
                           train=train, val=selected['val'], train_elapsed_seconds=train_elapsed,
                           train_rejected=train_rejected,
                           full_relative_mse_val_minus_train=selected['val']['film_mean']['full_relative_mse'] - train['film_mean']['full_relative_mse'],
                           note='Different disjoint TRAIN/VAL films, matched film count, fractions, seeds and sampler; TRAIN evaluated only after VAL selection')
            report = dict(run_id=run_id, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                          smoke=args.smoke, provenance=provenance, selection=selection,
                          profiles=results, matched_train_val=matched,
                          selected_train_rows=[r for r in rows if r['video_id'] in {s['film'] for s in train_samples}],
                          train_profiles_evaluated=1, val_profiles_evaluated=len(results),
                          total_elapsed_seconds=time.perf_counter() - started,
                          limitations=['Small fixed VAL cohort; heuristic selection, not statistical significance',
                                       'TRAIN evaluated for selected profile only',
                                       'Timing includes metrics/controls and can vary across GPUs',
                                       'Reference-layout IoU is not generated-image ground truth'])
            atomic_json(output / 'DIAGNOSTYKA.json', report)
            # Publish selection last, after matched TRAIN and all reports succeed.
            atomic_json(output / 'SAMPLER_SELECTION.json', selection)
            print(json.dumps(dict(event='sampler_selected', **selection), allow_nan=False), flush=True)
        broadcast(selection)
        if distributed:
            dist.barrier()
        return 0
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
