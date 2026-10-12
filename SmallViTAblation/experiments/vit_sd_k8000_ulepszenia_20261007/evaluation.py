"""Natural K8000 evaluation. Only JSON metrics leave this module; tensors stay in RAM.

evaluate(...) and aggregate_records(...) return film_mean, frames, films (count),
per_film and metric_film_counts. Frames are individual frame/seed observations.
Seeds are averaged within frames, frames within films, then films equally.
wrong_condition_mse is the WRONG condition's full *relative* latent MSE against
the natural target. Missing GT/controls are None, never a successful zero error.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import math
from numbers import Integral, Real
import zipfile

import torch
from torch.nn import functional as F

try:
    from .dane import iter_video, load_manifest, read_annotations
except ImportError:
    from dane import iter_video, load_manifest, read_annotations

STABLE_METRICS = ('full_relative_mse', 'cosine', 'roi_relative_mse',
                  'image_std', 'wrong_condition_mse', 'condition_image_mse')
METADATA_KEYS = {'film', 'frame', 'seed', 'has_tool', 'roi_tokens',
                 'wrong_condition_film', 'wrong_condition_frame', 'wrong_condition_available'}


def fixed_samples(rows, split, films=None, fractions=(.33, .67)):
    """Fully drain iter_video for each selected film before accepting any sample.

    films=None selects all; an integer selects evenly spaced sorted film IDs,
    including endpoints (one film selects the midpoint). No replacement after a
    rejected film: callers can require the intended cohort size. Every call uses
    fresh full-video preflight, with no persistent frame or latent cache.
    Returns (RAM samples, JSON-safe rejection records).
    """
    fractions = tuple(fractions)
    if not fractions or any(isinstance(f, bool) or not isinstance(f, Real)
                            or not math.isfinite(f) or not 0 <= f <= 1 for f in fractions):
        raise ValueError('fractions must be nonempty finite values in [0,1]')
    if films is not None and (isinstance(films, bool) or not isinstance(films, Integral) or films < 1):
        raise ValueError('films must be None or a positive integer limit')
    selected = [r for r in load_manifest(rows) if r['split'] == str(split).lower()]
    if films is not None and films < len(selected):
        positions = ([len(selected) // 2] if films == 1 else
                     [round(i * (len(selected) - 1) / (films - 1)) for i in range(films)])
        selected = [selected[i] for i in positions]
    csv.field_size_limit(32 * 1024 * 1024)
    samples, rejected = [], []
    for row in selected:
        try:
            count = row.get('frame_count', row.get('annotation_count'))
            if count is None:
                count = len(read_annotations(row['annotation_zip'], row['video_id']))
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ValueError('film frame count must be a positive integer')
            wanted = {max(1, min(count, int(count * f))) for f in fractions}
            # list() deliberately exhausts the second pass, including validation
            # that happens AFTER the final selected frame was yielded.
            found = list(iter_video(row, selected_frames=wanted))
            if (len(found) != len(wanted) or {s['frame'] for s in found} != wanted
                    or any(s['film'] != row['video_id'] for s in found)):
                raise ValueError('selected frame/film identities do not match manifest')
            samples.extend(sorted(found, key=lambda s: s['frame']))
        except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
            rejected.append(dict(film=row['video_id'], split=row['split'],
                                 reason=str(error), error_type=type(error).__name__))
    return samples, rejected


def roi_mask(bbox, has_tool=True, *, device=None):
    """Normalized valid xyxy -> 600 row-major tokens of the 20x30 grid.

    Mark cells with positive-area overlap. Boundaries within 1e-6 grid units of
    an integer are snapped to avoid an extra cell from float32 normalization.
    Tiny nonempty boxes always retain at least their containing cell.
    Invalid/missing GT returns None; it is never interpreted as a negative label.
    """
    if not has_tool:
        return None
    try:
        b = torch.as_tensor(bbox, dtype=torch.float64).reshape(-1).tolist()
    except (TypeError, ValueError, RuntimeError):
        return None
    if (len(b) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in b)
            or b[2] <= b[0] or b[3] <= b[1]):
        return None
    def snap(x):
        return round(x) if abs(x - round(x)) < 1e-6 else x
    x0, y0, x1, y1 = [snap(v * n) for v, n in zip(b, (30, 20, 30, 20))]
    left, top = min(29, math.floor(x0)), min(19, math.floor(y0))
    right, bottom = min(30, max(left + 1, math.ceil(x1))), min(20, max(top + 1, math.ceil(y1)))
    mask = torch.zeros((20, 30), dtype=torch.bool, device=device)
    mask[top:bottom, left:right] = True
    return mask.flatten()


def wrong_condition_indices(samples):
    """Deterministic derangement of films; align frame ordinal within each film."""
    by_film = defaultdict(list)
    for i, sample in enumerate(samples):
        by_film[str(sample['film'])].append(i)
    films = sorted(by_film)
    if len(films) < 2:
        return [None] * len(samples)
    for indices in by_film.values():
        indices.sort(key=lambda i: samples[i]['frame'])
    wrong = [None] * len(samples)
    for n, film in enumerate(films):
        others = by_film[films[(n + 1) % len(films)]]
        for ordinal, index in enumerate(by_film[film]):
            wrong[index] = others[ordinal % len(others)]
    return wrong


def aggregate_records(records):
    """Merge evaluated shards without weighting long films or extra seeds more.

    Does not rerun controls: their original within-shard pairing is preserved in
    each record. Reject duplicate (film,frame,seed), nonfinite values and malformed
    scalar metrics. Optional metrics average only the films with valid evidence;
    metric_film_counts makes that coverage explicit. Empty shards are allowed.
    """
    records = list(records)
    groups = defaultdict(lambda: defaultdict(list))
    seen, keys = set(), set(STABLE_METRICS)
    for record in records:
        identity = (str(record['film']), int(record['frame']), record.get('seed'))
        if identity in seen:
            raise ValueError(f'Duplicate evaluated frame/seed: {identity}')
        seen.add(identity)
        for key, value in record.items():
            if key in METADATA_KEYS:
                continue
            if value is None:
                keys.add(key)
            elif isinstance(value, Real) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite metric {key} at {identity}')
                keys.add(key)
            else:
                raise ValueError(f'Metric {key} must be a JSON scalar or None')
        groups[identity[0]][identity[1]].append(record)

    def mean(values):
        values = [float(v) for v in values if v is not None]
        return math.fsum(values) / len(values) if values else None

    per_film = {}
    for film, frame_groups in sorted(groups.items()):
        per_film[film] = {key: mean([mean([r.get(key) for r in rs]) for rs in frame_groups.values()])
                          for key in sorted(keys)}
    counts = {key: sum(metrics[key] is not None for metrics in per_film.values()) for key in sorted(keys)}
    controls = sum(bool(r.get('wrong_condition_available')) for r in records)
    return dict(film_mean={key: mean([p[key] for p in per_film.values()]) for key in sorted(keys)},
                frames=records, films=len(groups), per_film=per_film, metric_film_counts=counts,
                frame_count=sum(len(g) for g in groups.values()), observations=len(records),
                controls=dict(available_observations=controls, total_observations=len(records),
                              status=('available' if controls == len(records) and controls else
                                      'partial' if controls else 'unavailable')))


def _finite(tensor, label):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f'Nonfinite {label}')
    return tensor


def latent_metrics(actual, target, mask=None):
    if actual.shape != target.shape or actual.ndim != 3 or actual.shape[1:] != (600, 768):
        raise ValueError('Evaluation requires complete [B,600,768] teacher latents')
    actual, target = _finite(actual.float(), 'actual latent'), _finite(target.float(), 'target latent')
    mse = (actual - target).square().mean()
    result = dict(full_mse=mse.item(),
                  full_relative_mse=(mse / target.square().mean().clamp_min(1e-6)).item(),
                  cosine=F.cosine_similarity(actual.flatten(1), target.flatten(1)).mean().item(),
                  roi_latent_mse=None, roi_relative_mse=None)
    if mask is not None and mask.any():
        error = (actual[:, mask] - target[:, mask]).square().mean()
        result.update(roi_latent_mse=error.item(),
                      roi_relative_mse=(error / target[:, mask].square().mean().clamp_min(1e-6)).item())
    return result


def ssim(x, y):
    """11x11 uniform local-window RGB SSIM, valid windows, data_range=1."""
    window = min(11, x.shape[-2], x.shape[-1])
    pool = lambda t: F.avg_pool2d(t, window, 1)
    mx, my = pool(x), pool(y)
    vx, vy = (pool(x.square()) - mx.square()).clamp_min(0), (pool(y.square()) - my.square()).clamp_min(0)
    cov = pool(x * y) - mx * my
    return (((2 * mx * my + .0001) * (2 * cov + .0009)) /
            ((mx.square() + my.square() + .0001) * (vx + vy + .0009))).mean()


def _image_metrics(image, source):
    _finite(image, 'image')
    mse = (image - source).square().mean().item()
    return dict(psnr=-10 * math.log10(max(mse, 1e-12)), ssim=ssim(image, source).item(),
                image_std=image.flatten(2).std(dim=-1, unbiased=False).mean().item())


def _teacher(vit, image):
    output, latent = vit(image)
    vit._latent = None
    if latent is None or latent.shape[1:] != (600, 768):
        raise ValueError('Teacher must expose the full K8000 [B,600,768] latent')
    return _finite(output.float(), 'detector output'), _finite(latent.float(), 'teacher latent')


def _detection(output, bbox, mask):
    # Identical peak/box decoder to the verified model_snapshot; imported lazily.
    if __package__:
        from .model_snapshot import dekoduj
    else:
        from model_snapshot import dekoduj
    boxes, scores = dekoduj(output, k=1)
    box = _finite(boxes[0, 0].float(), 'decoded box')
    result = dict(natural_detector_confidence=float(scores[0, 0]),
                  reference_layout_iou_not_generated_ground_truth=None)
    if mask is not None:
        target = torch.as_tensor(bbox, device=box.device, dtype=box.dtype) * box.new_tensor([480, 320, 480, 320])
        inter = (torch.minimum(box[2:], target[2:]) - torch.maximum(box[:2], target[:2])).clamp_min(0).prod()
        union = (box[2:] - box[:2]).clamp_min(0).prod() + (target[2:] - target[:2]).prod() - inter
        result['reference_layout_iou_not_generated_ground_truth'] = float(inter / union.clamp_min(1e-12))
    return result


def _make_scheduler(steps, spacing, device):
    from diffusers import DDIMScheduler
    if __package__:
        from .model import schedule
    else:
        from model import schedule
    scheduler = DDIMScheduler.from_config(schedule().config, timestep_spacing=spacing)
    scheduler.set_timesteps(steps, device=device)
    return scheduler


def _check_stop(stop_check):
    if stop_check():
        raise InterruptedError('Evaluation interrupted for checkpoint')


def _generate(vae, net, condition, shape, scale, device, steps, spacing, seed, stop_check):
    scheduler = _make_scheduler(steps, spacing, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    h = torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * scheduler.init_noise_sigma
    amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    for t in scheduler.timesteps:
        _check_stop(stop_check)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            prediction = net(scheduler.scale_model_input(h, t), t, condition)
        h = _finite(scheduler.step(prediction.float(), t, h, eta=0.0).prev_sample, 'DDIM state')
    _check_stop(stop_check)
    decoded = _finite(vae.decode(h / scale).sample.float(), 'VAE decode')
    return ((decoded + 1) / 2).clamp(0, 1)


@torch.no_grad()
def evaluate(vae, net, vit, samples, scale, device, *, steps=50, spacing='leading',
             seeds=(20261007, 20261008), stop_check=lambda: False):
    """Natural Gaussian-start generation, paired wrong-film control and VAE floor.

    Caller owns requires_grad flags. All original per-submodule training flags
    are restored even on cancellation/errors. DDP wrappers are bypassed during
    unequal per-rank evaluation. No distributed collective occurs here.
    """
    if isinstance(steps, bool) or not isinstance(steps, Integral) or not 1 <= steps <= 1000:
        raise ValueError('steps must be an integer in [1,1000]')
    if spacing not in ('leading', 'trailing'):
        raise ValueError('spacing must be leading or trailing')
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    seeds = tuple(seeds)
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s < 2**63 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must be distinct nonnegative integers below 2**63')
    device, samples = torch.device(device), list(samples)
    vae, net, vit = (getattr(m, 'module', m) for m in (vae, net, vit))
    modes = {m: m.training for root in (vae, net, vit) for m in root.modules()}
    records = []
    try:
        vae.eval(); net.eval(); vit.eval()
        wrong_indices = wrong_condition_indices(samples)
        for index, sample in enumerate(samples):
            _check_stop(stop_check)
            source = _finite(sample['image'][None].to(device).float(), 'source image')
            reference = _finite(sample['vit_image'][None].to(device).float(), 'source teacher image')
            _, target = _teacher(vit, reference)
            mask = roi_mask(sample.get('bbox'), sample.get('has_tool', False), device=device)
            encoded = _finite(vae.encode(source * 2 - 1).latent_dist.mode(), 'VAE encoding')
            reconstruction = ((_finite(vae.decode(encoded).sample.float(), 'VAE reconstruction') + 1) / 2).clamp(0, 1)
            _, reconstructed_latent = _teacher(vit, F.interpolate(reconstruction, (320, 480), mode='bilinear', align_corners=False))
            reconstruction_metrics = {'vae_reconstruction_' + k: v
                                      for k, v in latent_metrics(reconstructed_latent, target, mask).items()}
            reconstruction_metrics.update({'vae_reconstruction_' + k: v for k, v in _image_metrics(reconstruction, source).items()})
            other = samples[wrong_indices[index]] if wrong_indices[index] is not None else None
            wrong_target = _teacher(vit, other['vit_image'][None].to(device).float())[1] if other is not None else None
            for seed in seeds:
                image = _generate(vae, net, target, encoded.shape, scale, device, steps, spacing, seed, stop_check)
                output, actual = _teacher(vit, F.interpolate(image, (320, 480), mode='bilinear', align_corners=False))
                record = dict(film=str(sample['film']), frame=int(sample['frame']), seed=int(seed),
                              has_tool=mask is not None, roi_tokens=int(mask.sum()) if mask is not None else 0,
                              wrong_condition_available=other is not None,
                              wrong_condition_film=str(other['film']) if other else None,
                              wrong_condition_frame=int(other['frame']) if other else None,
                              wrong_condition_mse=None, condition_image_mse=None,
                              condition_image_mean_absolute_change=None, relative_gain_vs_wrong_condition=None)
                record.update(latent_metrics(actual, target, mask))
                record.update(_image_metrics(image, source))
                record.update(_detection(output, sample.get('bbox'), mask))
                record.update(reconstruction_metrics)
                if wrong_target is not None:
                    wrong_image = _generate(vae, net, wrong_target, encoded.shape, scale, device, steps, spacing, seed, stop_check)
                    _, wrong_latent = _teacher(vit, F.interpolate(wrong_image, (320, 480), mode='bilinear', align_corners=False))
                    wrong_error = latent_metrics(wrong_latent, target)['full_relative_mse']
                    record.update(wrong_condition_mse=wrong_error,
                                  condition_image_mse=(image - wrong_image).square().mean().item(),
                                  condition_image_mean_absolute_change=(image - wrong_image).abs().mean().item(),
                                  relative_gain_vs_wrong_condition=1 - record['full_relative_mse'] / max(wrong_error, 1e-12))
                records.append(record)
        result = aggregate_records(records)
        result['config'] = dict(steps=int(steps), spacing=spacing, seeds=list(map(int, seeds)), scale=scale,
                                latent_shape=[600, 768], start='Gaussian noise', eta=0,
                                images_or_latents_saved=False, aggregation='seeds -> frames -> equal films',
                                ssim='11x11 uniform valid windows, RGB, data_range=1',
                                psnr='RGB MSE with data_range=1; exact matches capped at 120 dB',
                                roi='positive-area overlap with normalized bbox on row-major 20x30 grid',
                                wrong_condition_mse='full relative latent MSE against natural target',
                                iou_note='Reference-layout diagnostic only; not generated-image ground truth')
        if samples:
            sampler = _make_scheduler(steps, spacing, device)
            result['config']['scheduler_config'] = dict(sampler.config)
            result['config']['sampler_timesteps'] = sampler.timesteps.cpu().tolist()
        return result
    finally:
        vit._latent = None
        for module, training in modes.items():
            module.training = training
