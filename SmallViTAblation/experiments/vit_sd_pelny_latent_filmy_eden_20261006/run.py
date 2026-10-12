"""DDPM z naturalnym pełnym latentem; smoke obala brak gradientu lub zależności od warunku.

Online RGB/VAE/ViT only. No image, video, embedding or latent is saved.
Run under torchrun; deployment and submission belong to the parent task.
"""
import argparse
import contextlib
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
import torch
import torch.distributed as dist
from torch.nn import functional as F
from model import (Denoiser, load_sd, load_vit, vit_forward, auxiliary_loss,
                   trainable_state, restore_trainable)

STOP_REQUESTED = False
STATUS_OUTPUT = None
STATUS_RANK = 0


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def weights_digest(root):
    files = sorted(Path(root).glob('*.safetensors')) or sorted(Path(root).glob('*.bin'))
    if not files:
        raise ValueError(f'No pretrained weights in {root}')
    return {p.name: digest(p) for p in files}


def collective(value, device, operation=dist.ReduceOp.SUM):
    tensor = torch.as_tensor(value, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=operation)
    return tensor


def collate(samples):
    # Keep bbox lists so multiple tools / empty bbox do not break default_collate.
    return dict(image=torch.stack([s['image'] for s in samples]),
                vit_image=torch.stack([s['vit_image'] for s in samples]),
                bbox=[s['bbox'] for s in samples], film=[s['film'] for s in samples],
                frame=[s['frame'] for s in samples], has_tool=[s['has_tool'] for s in samples])


def loader(path, split, a, rank, world, repeat):
    from dane import VideoFrames, load_manifest
    rows = [r for r in load_manifest(path) if r['split'] == split]
    if repeat and len(rows) < world * max(1, a.workers):
        raise ValueError('Every TRAIN rank/worker needs a film; reduce workers or ranks')
    # VideoFrames shards films itself. Never add DistributedSampler.
    dataset = VideoFrames(path, split=split, rank=rank, world_size=world, seed=a.seed, repeat=repeat)
    generator = torch.Generator().manual_seed(a.seed + rank)
    return torch.utils.data.DataLoader(dataset, batch_size=a.batch_size, num_workers=a.workers,
                                      collate_fn=collate, pin_memory=torch.cuda.is_available(),
                                      persistent_workers=False, generator=generator,
                                      **({'prefetch_factor': 1} if a.workers else {}))


def validate_manifests(a):
    from dane import load_manifest
    rows = load_manifest(a.data_manifest)
    train = {r['video_id'] for r in rows if r['split'] == 'train'}
    evaluation = load_manifest(a.eval_manifest or a.data_manifest)
    val = {r['video_id'] for r in evaluation if r['split'] == 'val'}
    assignments = {}
    for row in rows + evaluation:
        film, split = row['video_id'], row['split']
        if film in assignments and assignments[film] != split:
            raise ValueError(f'Film crosses splits: {film}')
        assignments[film] = split
    if not train or not val or train & val:
        raise ValueError('Require disjoint TRAIN and VAL films; use --eval-manifest for a TRAIN-only manifest')
    if a.expected_train_films and len(train) != a.expected_train_films:
        raise ValueError(f'Expected {a.expected_train_films} TRAIN films, got {len(train)}')
    return dict(train_films=sorted(train), val_films=sorted(val),
                manifest_sha256=digest(a.data_manifest),
                eval_manifest_sha256=digest(a.eval_manifest or a.data_manifest))


def prepare(batch, vit, vae, device):
    image = batch['image'].to(device, non_blocking=True)
    vi = batch['vit_image'].to(device, non_blocking=True)
    if image.shape[1:] != (3, 384, 576) or vi.shape[1:] != (3, 320, 480):
        raise ValueError('Dataset resize contract violated')
    for x in (image, vi):
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('Expected finite RGB tensors in [0,1]')
    with torch.no_grad():
        # Frozen VAE encodes in FP32. Natural frozen ViT forward, never a synthetic hook target.
        clean = vae.encode(image * 2 - 1).latent_dist.sample() * float(vae.config.scaling_factor)
        _, latent = vit_forward(vit, vi)
    if clean.shape[1:] != (4, 48, 72):
        raise ValueError('Expected VAE latent Bx4x48x72')
    return clean, latent.detach()


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


@torch.no_grad()
def sensitivity(model, clean, latent, device):
    model.eval()
    noisy = torch.randn_like(clean[:1])
    t = torch.full((1,), 100, device=device, dtype=torch.long)
    z = latent[:1]
    with amp(device):
        original = model(noisy, t, z).float()
        zero = model(noisy, t, torch.zeros_like(z)).float()
        shuffled = model(noisy, t, z.roll(1, dims=1)).float()
    results = dict(zero_condition_rms=float((original-zero).square().mean().sqrt()),
                   shuffled_token_rms=float((original-shuffled).square().mean().sqrt()))
    if not all(v > 0 and torch.isfinite(torch.tensor(v)) for v in results.values()):
        raise RuntimeError(f'UNet condition sensitivity failed: {results}')
    model.train()
    return results, (noisy, t, z, original)


@torch.no_grad()
def evaluate(model, vit, vae, scheduler, a, rank, world, device):
    model.eval()
    per_film = {}
    # Evaluation should not consume the TRAIN noise RNG sequence.
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(a.seed + 50000 + rank)
        for i, batch in enumerate(loader(a.eval_manifest or a.data_manifest, 'val', a, rank, world, False)):
            if a.eval_batches and i >= a.eval_batches:
                break
            clean, latent = prepare(batch, vit, vae, device)
            noise = torch.randn_like(clean)
            t = torch.randint(0, 1000, (len(clean),), device=device)
            with amp(device):
                pred = model(scheduler.add_noise(clean, noise, t), t, latent)
            losses = (pred.float() - noise).square().flatten(1).mean(1)
            for film, loss in zip(batch['film'], losses.tolist()):
                item = per_film.setdefault(film, [0., 0])
                item[0] += loss
                item[1] += 1
    gathered = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(gathered, per_film)
    else:
        gathered[0] = per_film
    merged = {}
    for local in gathered:
        for film, (total, count) in local.items():
            item = merged.setdefault(film, [0., 0])
            item[0] += total
            item[1] += count
    if not merged:
        raise RuntimeError('No held-out evaluation samples')
    scores = {film: total / count for film, (total, count) in merged.items()}
    model.train()
    return dict(val_mse=sum(v[0] for v in merged.values()) / sum(v[1] for v in merged.values()),
                val_film_mean_mse=sum(scores.values()) / len(scores), val_films=len(scores),
                val_samples=sum(v[1] for v in merged.values()), per_film_mse=scores,
                truncated_per_rank=bool(a.eval_batches))


def rng_state(consumed):
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                python=random.getstate(), consumed_batches=consumed)


def save_checkpoint(out, step, model, optimizer, signature, consumed, rank, world, device):
    states = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(states, rng_state(consumed))
    else:
        states[0] = rng_state(consumed)
    if rank == 0:
        target = out / f'checkpoint_{step:08d}.pt'
        temp = target.with_suffix('.tmp')
        # Remove stale incomplete save from an interrupted process before allocating another.
        for stale in out.glob('checkpoint_*.tmp'):
            stale.unlink()
        payload = dict(format_version=1, step=step, trainable=trainable_state(model),
                       optimizer=optimizer.state_dict(), signature=signature, ranks=states)
        with temp.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        # Two committed checkpoints, at most three copies temporarily during atomic publication.
        for old in sorted(out.glob('checkpoint_*.pt'))[:-2]:
            old.unlink()
        atomic_json(out / 'LATEST.json', dict(step=step, checkpoint=target.name, bytes=target.stat().st_size))
    if dist.is_initialized():
        dist.barrier()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--data-manifest', type=Path)
    p.add_argument('--eval-manifest', type=Path)
    p.add_argument('--sd-root', type=Path, help='Direct pretrained EMA020000 UNet directory')
    p.add_argument('--vae-root', type=Path, help='Direct sd-vae-ft-mse directory')
    p.add_argument('--vit-checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--steps', type=int, default=20000, help='Optimizer updates, not microbatches')
    p.add_argument('--batch-size', type=int, default=2, help='Per GPU microbatch')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--train-unet-mode', choices=['cross-attention', 'full'], default='cross-attention')
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--adapter-lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--input-scale', type=float, default=1.0)
    p.add_argument('--aux-every', type=int, default=8)
    p.add_argument('--aux-weight', type=float, default=.01)
    p.add_argument('--detector-weight', type=float, default=.1)
    p.add_argument('--aux-max-t', type=int, default=250)
    p.add_argument('--aux-batch-size', type=int, default=1)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=32, help='Per rank; 0 = full VAL pass')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261006)
    p.add_argument('--expected-train-films', type=int, default=118, help='0 disables count assertion for local tests')
    p.add_argument('--max-hours', type=float, default=23.5)
    p.add_argument('--resume', default='auto', help='auto, none, or checkpoint path')
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Two full pretrained updates on real data + reload/sensitivity')
    return p


def main():
    global STATUS_OUTPUT, STATUS_RANK
    run_started = time.monotonic()
    p = parser()
    initial, _ = p.parse_known_args()
    if initial.config:
        defaults = json.loads(initial.config.read_text(encoding='utf-8-sig'))
        unknown = set(defaults) - {action.dest for action in p._actions}
        if unknown:
            p.error(f'Unknown config keys: {unknown}')
        p.set_defaults(**defaults)
    a = p.parse_args()
    for name in ('data_manifest', 'sd_root', 'vae_root', 'vit_checkpoint', 'output'):
        value = getattr(a, name)
        if not value:
            p.error(f'--{name.replace("_", "-")} is required')
        setattr(a, name, Path(value))
    if a.eval_manifest:
        a.eval_manifest = Path(a.eval_manifest)
    for name in ('steps', 'batch_size', 'grad_accum', 'save_every', 'eval_every', 'log_every', 'aux_batch_size'):
        if getattr(a, name) < 1:
            p.error(f'{name} must be positive')
    if a.workers < 0 or a.eval_batches < 0 or a.aux_every < 0 or not 0 <= a.aux_max_t <= 250:
        p.error('Invalid worker/eval/aux settings (aux_max_t must be <=250)')
    if a.smoke:
        a.steps, a.save_every, a.eval_every, a.log_every = 2, 1, 2, 1
    rank, world, local = (int(os.environ.get(n, d)) for n, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{local}' if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16-capable GPU required')
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo')
    torch.set_num_threads(4)
    random.seed(a.seed + rank)
    torch.manual_seed(a.seed)  # Same initial adapter on every rank, even before DDP broadcast.
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    STATUS_OUTPUT, STATUS_RANK = out, rank
    if rank == 0:
        atomic_json(out / 'STATUS.json', dict(state='running', phase='loading', step=0,
                    target_steps=a.steps, world_size=world, pid=os.getpid(), started_unix=time.time()))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, request_stop)
    split_info = validate_manifests(a)
    provenance = None
    if rank == 0:
        provenance = dict(vit_sha256=digest(a.vit_checkpoint), unet_weights=weights_digest(a.sd_root),
                          vae_weights=weights_digest(a.vae_root), splits=split_info,
                          unet_config_sha256=digest(a.sd_root / 'config.json'),
                          vae_config_sha256=digest(a.vae_root / 'config.json'))
    if world > 1:
        values = [provenance]
        dist.broadcast_object_list(values, src=0)
        provenance = values[0]
    unet, vae, scheduler = load_sd(a.sd_root, a.vae_root, a.train_unet_mode, device,
                                  not a.no_gradient_checkpointing)
    vit = load_vit(a.vit_checkpoint, device)
    model = Denoiser(unet, a.input_scale).to(device)
    unet_params = [p for p in model.unet.parameters() if p.requires_grad]
    if not unet_params:
        raise RuntimeError('No UNet training parameters selected')
    params = list(model.adapter.parameters()) + unet_params
    optimizer = torch.optim.AdamW([{'params': model.adapter.parameters(), 'lr': a.adapter_lr},
                                   {'params': unet_params, 'lr': a.lr}], weight_decay=.01)
    signature = dict(mode=a.train_unet_mode, input_scale=a.input_scale, world_size=world,
                     batch_size=a.batch_size, grad_accum=a.grad_accum, workers=a.workers, seed=a.seed,
                     aux_every=a.aux_every, aux_weight=a.aux_weight, detector_weight=a.detector_weight,
                     aux_max_t=a.aux_max_t, aux_batch_size=a.aux_batch_size, lr=a.lr,
                     adapter_lr=a.adapter_lr, warmup=a.warmup, provenance=provenance)
    checkpoints = sorted(out.glob('checkpoint_*.pt'))
    path = checkpoints[-1] if a.resume == 'auto' and checkpoints else None
    if a.resume not in ('auto', 'none'):
        path = Path(a.resume)
    if checkpoints and (a.resume == 'none' or a.smoke):
        raise ValueError('Use a fresh output for --resume none or --smoke')
    step, consumed, state = 0, 0, None
    if path:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['signature'] != signature:
            raise ValueError('Resume changes data/weights/training configuration; use the original settings')
        restore_trainable(model, checkpoint['trainable'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        state = checkpoint['ranks'][rank]
        consumed = state['consumed_batches']
        del checkpoint
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None,
                                                    broadcast_buffers=False) if world > 1 else model
    if rank == 0:
        import diffusers, timm
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        config.update(global_batch=a.batch_size * world * a.grad_accum, world_size=world,
                      trainable_parameters=sum(p.numel() for p in params),
                      unet_trainable_parameters=sum(p.numel() for p in unet_params),
                      frozen_vit=True, frozen_vae=True, ema=False, resume_step=step,
                      torch=torch.__version__, diffusers=diffusers.__version__, timm=timm.__version__,
                      python=sys.version, provenance=provenance)
        config['checkpoint_estimate_gb'] = sum(p.numel() for p in params) * 12 / 1e9
        config['atomic_checkpoint_peak_estimate_gb'] = config['checkpoint_estimate_gb'] * 3
        atomic_json(out / 'CONFIG.json', config)
        atomic_json(out / 'STATUS.json', dict(state='running', phase='replay' if consumed else 'training',
                    step=step, target_steps=a.steps, world_size=world, consumed_batches_per_rank=consumed,
                    elapsed_seconds=time.monotonic()-run_started))
    iterator = iter(loader(a.data_manifest, 'train', a, rank, world, True))
    # Exact deterministic replay of the streaming cursor: no disk cache, potentially expensive.
    for _ in range(consumed):
        next(iterator)
    torch.manual_seed(a.seed + 1000 + rank)
    if state:
        torch.set_rng_state(state['torch'])
        random.setstate(state['python'])
        if cuda:
            torch.cuda.set_rng_state(state['cuda'], device)
    start = run_started
    if rank == 0:
        atomic_json(out / 'STATUS.json', dict(state='running' if step < a.steps else 'complete',
                    phase='training', step=step, target_steps=a.steps, world_size=world,
                    elapsed_seconds=time.monotonic()-start))
    optimizer.zero_grad(set_to_none=True)
    smoke_stats = []
    while step < a.steps:
        model.train()
        warmup = min(1., (step + 1) / max(1, a.warmup))
        optimizer.param_groups[0]['lr'] = a.adapter_lr * warmup
        optimizer.param_groups[1]['lr'] = a.lr * warmup
        denoise_sum, aux_sum, consistency_sum, detection_sum = 0., 0., 0., 0.
        for micro in range(a.grad_accum):
            batch = next(iterator)
            consumed += 1
            clean, latent = prepare(batch, vit, vae, device)
            noise = torch.randn_like(clean)
            t = torch.randint(0, 1000, (len(clean),), device=device)
            noisy = scheduler.add_noise(clean, noise, t)
            use_aux = micro == a.grad_accum - 1 and a.aux_every > 0 and (
                (step + 1) % a.aux_every == 0 or a.smoke)
            aux_n = min(a.aux_batch_size, len(clean)) if use_aux else 0
            if use_aux:
                low_t = torch.randint(0, a.aux_max_t + 1, (aux_n,), device=device)
                low_noisy = scheduler.add_noise(clean[:aux_n], torch.randn_like(clean[:aux_n]), low_t)
                all_noisy = torch.cat((noisy, low_noisy))
                all_t = torch.cat((t, low_t))
                all_latent = torch.cat((latent, latent[:aux_n]))
            else:
                all_noisy, all_t, all_latent = noisy, t, latent
            sync = net.no_sync() if world > 1 and micro < a.grad_accum - 1 else contextlib.nullcontext()
            with sync:
                with amp(device):
                    predicted = net(all_noisy, all_t, all_latent)
                    loss_denoise = F.mse_loss(predicted[:len(clean)].float(), noise)
                    loss_aux = loss_denoise.new_zeros(())
                    if use_aux:
                        alpha = scheduler.alphas_cumprod.to(device)[low_t].reshape(-1, 1, 1, 1)
                        estimate = (low_noisy - (1-alpha).sqrt() * predicted[len(clean):].float()) / alpha.sqrt()
                        loss_aux, cons, det = auxiliary_loss(estimate, latent[:aux_n], vae, vit,
                                                           batch['bbox'][:aux_n], batch['has_tool'][:aux_n],
                                                           float(vae.config.scaling_factor), a.detector_weight)
                        consistency_sum += float(cons)
                        detection_sum += float(det)
                    # Auxiliary branch once per update: retain its specified weight across accumulation.
                    loss = loss_denoise / a.grad_accum + a.aux_weight * loss_aux
                if not bool(collective(float(torch.isfinite(loss.detach())), device, dist.ReduceOp.MIN)):
                    raise FloatingPointError('Nonfinite loss on a rank')
                loss.backward()
            denoise_sum += float(loss_denoise.detach()) / a.grad_accum
            aux_sum += float(loss_aux.detach())
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if a.smoke:
            adapter_grad = sum(float(p.grad.float().square().sum()) for p in model.adapter.parameters() if p.grad is not None)
            unet_grad = sum(float(p.grad.float().square().sum()) for p in unet_params if p.grad is not None)
            if adapter_grad <= 0 or unet_grad <= 0:
                raise RuntimeError('Smoke requires nonzero adapter and UNet gradients')
            if any(p.grad is not None for p in list(vit.parameters()) + list(vae.parameters())):
                raise RuntimeError('Frozen encoder has parameter gradients')
            smoke_stats.append(dict(step=step+1, adapter_grad_energy=adapter_grad, unet_grad_energy=unet_grad))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        stop = collective(int(STOP_REQUESTED or (out / 'STOP').exists() or
                              time.monotonic()-start >= a.max_hours * 3600), device, dist.ReduceOp.MAX).item() > 0
        end = stop or step >= a.steps
        if step % a.log_every == 0 or end:
            totals = collective([denoise_sum, aux_sum, consistency_sum, detection_sum, float(norm)], device) / world
            entry = dict(step=step, denoise_mse=totals[0].item(), aux_loss=totals[1].item(),
                         latent_consistency=totals[2].item(), detection_loss=totals[3].item(),
                         grad_norm=totals[4].item(), elapsed_seconds=time.monotonic()-start)
            if rank == 0:
                with (out / 'METRICS.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(entry, allow_nan=False)+'\n')
                print(json.dumps(entry), flush=True)
                atomic_json(out / 'STATUS.json', dict(state='running', phase='training', target_steps=a.steps,
                            world_size=world, consumed_batches_per_rank=consumed, **entry))
        if step % a.eval_every == 0 or end:
            metrics = evaluate(model, vit, vae, scheduler, a, rank, world, device)
            if rank == 0:
                atomic_json(out / 'EVAL.json', dict(step=step, **metrics))
        if step % a.save_every == 0 or end:
            save_checkpoint(out, step, model, optimizer, signature, consumed, rank, world, device)
        if a.smoke and end:
            results, probe = sensitivity(model, clean, latent, device)
            noisy_probe, timestep_probe, latent_probe, reference = probe
            ck = torch.load(out / f'checkpoint_{step:08d}.pt', map_location='cpu', weights_only=True)
            if ck['step'] != 2 or not ck['optimizer']['state']:
                raise RuntimeError('Smoke checkpoint has no completed updates / optimizer state')
            with torch.no_grad():
                next(model.adapter.parameters()).add_(1.)
            restore_trainable(model, ck['trainable'])
            optimizer.load_state_dict(ck['optimizer'])
            model.eval()
            with torch.no_grad(), amp(device):
                reloaded = model(noisy_probe, timestep_probe, latent_probe).float()
            error = float((reloaded-reference).abs().max())
            if error > 1e-5:
                raise RuntimeError(f'Checkpoint reload changed prediction: {error}')
            local_stats = dict(rank=rank, gradients=smoke_stats, **results, reload_max_error=error,
                               actual_pretrained=True, actual_dataset=True)
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, local_stats)
            else:
                gathered[0] = local_stats
            if rank == 0:
                atomic_json(out / 'SMOKE.json', dict(passed=True, steps=step, ranks=gathered))
        if end:
            if rank == 0:
                atomic_json(out / 'STATUS.json', dict(step=step, target_steps=a.steps,
                                                     state='complete' if step >= a.steps else 'stopped'))
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if STATUS_OUTPUT is not None and STATUS_RANK == 0:
            previous = {}
            status_path = STATUS_OUTPUT / 'STATUS.json'
            if status_path.exists():
                previous = json.loads(status_path.read_text(encoding='utf-8'))
            atomic_json(status_path, dict(previous, state='failed', error=traceback.format_exc()))
        raise
