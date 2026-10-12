"""Dluzszy trening zatwierdzonego wariantu FastGAN z checkpointami i statusem.

Wagi od zera; podglady EMA nie sa dowodem jakosci masek ani odwracania RF-DETR.
Wznowienie tylko z wlasnych checkpointow tego skryptu. Pilot pozostaje bez zmian.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from pilot_lokalny import Crops, crop_part

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class Batches:
    """Jedna permutacja na epoke; stan zachowany dla wznowienia bez przeskokow."""
    def __init__(self, dataset, size):
        if len(dataset) % size:
            raise ValueError('W tym biegu liczba obrazow musi byc podzielna przez batch')
        self.dataset, self.size = dataset, size
        self.generator = torch.Generator().manual_seed(1942)
        self.order = torch.randperm(len(dataset), generator=self.generator).tolist()
        self.cursor = 0

    def next(self):
        if self.cursor == len(self.order):
            self.order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
            self.cursor = 0
        ids = self.order[self.cursor:self.cursor + self.size]
        self.cursor += self.size
        return torch.stack([self.dataset[i] for i in ids]).cuda()

    def state(self):
        return dict(order=self.order, cursor=self.cursor, generator=self.generator.get_state())

    def restore(self, state):
        if sorted(state['order']) != list(range(len(self.dataset))):
            raise ValueError('Checkpoint ma inny zbior danych')
        self.order, self.cursor = state['order'], state['cursor']
        self.generator.set_state(state['generator'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--checkpoint-every', type=int, default=5000)
    p.add_argument('--preview-every', type=int, default=1000)
    p.add_argument('--memory-fraction', type=float, default=.5)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed, previous_elapsed = 0, 0.
    durations, latest_checkpoint = [], None
    state = 'initializing'
    write_json(args.out / 'STATUS.json', dict(status=state, pid=os.getpid(), updated_at=datetime.now().astimezone().isoformat()))
    try:
        if min(args.steps, args.batch, args.preview_every, args.checkpoint_every) <= 0:
            raise ValueError('Budzet i interwaly musza byc dodatnie')
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cudnn.benchmark = False
        free, _ = torch.cuda.mem_get_info()
        if free < 2500 * 1024**2:
            raise RuntimeError('Za malo wolnej pamieci na start; cudzych procesow nie zatrzymano')
        source = HERE / 'vendor/deepfake_gi_fastGAN/models.py'
        spec = importlib.util.spec_from_file_location('fastgan_training_vendor', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dataset = Crops(HERE / 'dane/pilot_1000')
        batches = Batches(dataset, args.batch)
        g = module.Generator(ngf=64, nz=256, nc=4, im_size=256).cuda()
        d = module.Discriminator(ndf=64, nc=4, im_size=256).cuda()
        g.apply(module.weights_init)
        d.apply(module.weights_init)
        # Kopiowanie modulow ze spectral_norm jest problematyczne po pierwszym forward.
        ema = copy.deepcopy(g).eval().requires_grad_(False)
        percept = lpips.LPIPS(net='vgg').eval().cuda().requires_grad_(False)
        opt_g = torch.optim.Adam(g.parameters(), lr=.0002, betas=(.5, .999))
        opt_d = torch.optim.Adam(d.parameters(), lr=.0002, betas=(.5, .999))
        fixed = torch.randn(8, 256, device='cuda')
        manifest_sha = digest(HERE / 'dane/pilot_1000/MANIFEST.json')
        config = dict(steps=args.steps, batch=args.batch, checkpoint_every=args.checkpoint_every,
                      preview_every=args.preview_every, memory_fraction=args.memory_fraction,
                      frames=len(dataset), size=256, channels=4, width=64, seed=42,
                      torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), ema_decay=.999,
                      initial_weights='random' if not args.resume else str(args.resume.resolve()),
                      architecture_sha256=digest(source), data_manifest_sha256=manifest_sha,
                      code_sha256=digest(Path(__file__)), dataset_code_sha256=digest(HERE/'pilot_lokalny.py'),
                      augmentation='joint RGB/mask horizontal flip, no color augmentation',
                      loss='hinge + LPIPS VGG RGB + L1 mask reconstruction',
                      note='Generator only, no detector training. No guarantee of fiber matching.')
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
            old = checkpoint['config']
            for key in ('data_manifest_sha256', 'architecture_sha256', 'batch', 'ema_decay', 'loss', 'augmentation', 'code_sha256', 'dataset_code_sha256'):
                if old[key] != config[key]:
                    raise ValueError(f'Niezgodnosc wznowienia: {key}')
            g.load_state_dict(checkpoint['g'])
            d.load_state_dict(checkpoint['d'])
            ema.load_state_dict(checkpoint['ema'])
            opt_g.load_state_dict(checkpoint['opt_g'])
            opt_d.load_state_dict(checkpoint['opt_d'])
            fixed = checkpoint['fixed'].cuda()
            batches.restore(checkpoint['batches'])
            completed = checkpoint['step']
            previous_elapsed = checkpoint['elapsed_seconds']
            torch.set_rng_state(checkpoint['torch_rng'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng'])
            random.setstate(checkpoint['python_rng'])
            if completed >= args.steps:
                raise ValueError('Checkpoint osiagnal juz zadany budzet')
        write_json(args.out / 'CONFIG.json', config)
        # Kopia wykonywanego kodu chroni odtwarzalnosc przy pozniejszych zmianach plikow.
        (args.out / 'trening_lokalny.py.snapshot').write_bytes(Path(__file__).read_bytes())
        (args.out / 'pilot_lokalny.py.snapshot').write_bytes((HERE/'pilot_lokalny.py').read_bytes())
        print(json.dumps(config), flush=True)
        torch.cuda.reset_peak_memory_stats()
        training_start = time.monotonic()

        def status(name, losses=None):
            elapsed = previous_elapsed + time.monotonic() - started
            speed = float(np.median(durations[-1000:])) if durations else None
            result = dict(status=name, pid=os.getpid(), step=completed, target_steps=args.steps,
                          approximate_epochs=completed*args.batch/len(dataset), elapsed_seconds=elapsed,
                          median_step_seconds=speed,
                          remaining_hours=speed*(args.steps-completed)/3600 if speed else None,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                          latest_checkpoint=latest_checkpoint, losses=losses,
                          updated_at=datetime.now().astimezone().isoformat())
            write_json(args.out / 'STATUS.json', result)
            return result

        def save_checkpoint():
            nonlocal latest_checkpoint
            filename = args.out / f'checkpoint_{completed:06d}.pth'
            temporary = filename.with_suffix('.pth.tmp')
            torch.save(dict(step=completed, config=config, g=g.state_dict(), d=d.state_dict(),
                            ema=ema.state_dict(), opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(),
                            fixed=fixed, batches=batches.state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
                            elapsed_seconds=previous_elapsed + time.monotonic() - started), temporary)
            os.replace(temporary, filename)
            latest_checkpoint = str(filename)

        def preview():
            # Generator wstrzykuje szum takze w eval; podglad ma stale ziarno i nie
            # zmienia strumienia losowego treningu ani stanow normalizacji G.
            with torch.random.fork_rng(devices=[0]), torch.no_grad():
                torch.manual_seed(1977)
                image = ema(fixed)[0].clamp(-1, 1).cpu()
            save_image((image[:, :3]+1)/2, args.out/f'RGB_{completed:06d}.png', nrow=4)
            save_image((image[:, 3:]+1)/2, args.out/f'MASKI_{completed:06d}.png', nrow=4)

        state = 'running'
        status(state)
        initial_step = completed
        with (args.out / 'KROKI.jsonl').open('w', encoding='utf-8') as log:
            while completed < args.steps:
                if (args.out / 'STOP').exists():
                    state = 'stopped_requested'
                    break
                free, _ = torch.cuda.mem_get_info()
                if free < 512*1024**2:
                    state = 'stopped_low_free_vram'
                    break
                tick = time.monotonic()
                real = batches.next()
                d.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                part = random.randrange(4)
                pred, recon = d(real, 'real', part)
                loss_d = F.relu(torch.rand_like(pred)*.2 + .8 - pred).mean()
                for rec, target in zip(recon, [real, real, crop_part(real, part)]):
                    rgb = F.interpolate(target[:, :3], rec.shape[-2:], mode='bilinear', align_corners=False)
                    mask = F.interpolate(target[:, 3:], rec.shape[-2:], mode='nearest')
                    loss_d = loss_d + percept(rec[:, :3], rgb).mean() + F.l1_loss(rec[:, 3:], mask)
                if not torch.isfinite(loss_d):
                    raise RuntimeError('Niefinitywna strata D-real')
                loss_d.backward()
                with torch.no_grad():
                    fake = g(torch.randn(args.batch, 256, device='cuda'))
                pred_fake = d(fake, 'fake')
                loss_fake = F.relu(torch.rand_like(pred_fake)*.2 + .8 + pred_fake).mean()
                if not torch.isfinite(loss_fake):
                    raise RuntimeError('Niefinitywna strata D-fake')
                loss_fake.backward()
                opt_d.step()
                d.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                fake_g = g(torch.randn(args.batch, 256, device='cuda'))
                loss_g = -d(fake_g, 'fake').mean()
                if not torch.isfinite(loss_g):
                    raise RuntimeError('Niefinitywna strata G')
                loss_g.backward()
                opt_g.step()
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), g.parameters()):
                        a.lerp_(b, .001)
                    for a, b in zip(ema.buffers(), g.buffers()):
                        a.copy_(b)
                torch.cuda.synchronize()
                seconds = time.monotonic()-tick
                completed += 1
                if completed > initial_step + 10:
                    durations.append(seconds)
                losses = dict(g=float(loss_g.detach()), d=float((loss_d+loss_fake).detach()))
                log.write(json.dumps(dict(step=completed, seconds=seconds, **losses))+'\n')
                log.flush()
                del real, pred, recon, rec, target, rgb, mask, loss_d, fake, pred_fake, loss_fake, fake_g, loss_g
                if completed % args.preview_every == 0 or completed == args.steps:
                    preview()
                if completed % args.checkpoint_every == 0 or completed == args.steps:
                    save_checkpoint()
                if completed % 100 == 0 or completed == initial_step + 1:
                    print(json.dumps(status('running', losses)), flush=True)
        if completed == args.steps:
            state = 'completed'
        elif completed and completed % args.checkpoint_every:
            save_checkpoint()
            preview()
        report = status(state)
        report['training_seconds_this_process'] = time.monotonic()-training_start
        report['quality'] = 'not independently evaluated; generated masks are not ground truth'
        write_json(args.out/'REPORT.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        failure = dict(status='failed', pid=os.getpid(), step=completed, target_steps=args.steps,
                       latest_checkpoint=latest_checkpoint, error=str(error),
                       updated_at=datetime.now().astimezone().isoformat())
        write_json(args.out/'STATUS.json', failure)
        write_json(args.out/'REPORT.json', failure)
        (args.out/'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
