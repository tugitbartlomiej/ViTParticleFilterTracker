"""Ograniczony pilot FastGAN: czas i VRAM, bez wnioskow o jakosci generatora.

Architektura HF, losowe wagi, RGB+segmentacja. LPIPS-VGG tylko RGB, dodatkowe
L1 maski w samorekonstrukcji D. Nie uruchamia ani nie zatrzymuje RF-DETR.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import time
import traceback
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent


class Crops(Dataset):
    def __init__(self, root):
        self.root = root
        self.audit = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
        self.records = self.audit['records']
        if any(r['split'] != 'train' for r in self.records):
            raise ValueError('W treningu sa dane inne niz train')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        a = np.array(Image.open(self.root / self.records[i]['file']).convert('RGBA'))
        if random.random() < .5:
            a = a[:, ::-1].copy()
        return torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 127.5 - 1


def crop_part(x, part):
    half = x.shape[-1] // 2
    y, z = divmod(part, 2)
    return x[:, :, y*half:(y+1)*half, z*half:(z+1)*half]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--memory-fraction', type=float, default=.5)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.backends.cudnn.benchmark = False
    source = HERE / 'vendor/deepfake_gi_fastGAN/models.py'
    spec = importlib.util.spec_from_file_location('fastgan_vendor_models', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = Crops(HERE / 'dane/pilot_1000')
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    g = module.Generator(ngf=64, nz=256, nc=4, im_size=256).cuda()
    d = module.Discriminator(ndf=64, nc=4, im_size=256).cuda()
    g.apply(module.weights_init)
    d.apply(module.weights_init)
    percept = lpips.LPIPS(net='vgg').eval().cuda()
    percept.requires_grad_(False)
    opt_g = torch.optim.Adam(g.parameters(), lr=.0002, betas=(.5, .999))
    opt_d = torch.optim.Adam(d.parameters(), lr=.0002, betas=(.5, .999))
    config = dict(**vars(args), frames=len(data), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  initialization='random; HF polyp weights not loaded', width=64, size=256, channels=4,
                  loss='hinge + LPIPS VGG RGB reconstruction + L1 mask reconstruction (weight 1)',
                  augmentation='joint RGB/mask horizontal flip; no color on mask',
                  purpose='timing pilot only', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    config['out'] = str(args.out)
    (args.out / 'CONFIG.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config), flush=True)
    fixed = torch.randn(4, 256, device='cuda')
    it = iter(loader)
    durations, completed = [], 0
    status = 'completed'
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        with (args.out / 'KROKI.jsonl').open('w', encoding='utf-8') as log:
            for step in range(args.steps):
                # Zostawiamy margines innym procesom, bez ich przerywania.
                free, total = torch.cuda.mem_get_info()
                if free < 512 * 1024**2:
                    status = 'stopped_low_free_vram'
                    break
                tick = time.perf_counter()
                try:
                    real = next(it).cuda()
                except StopIteration:
                    it = iter(loader)
                    real = next(it).cuda()
                d.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                part = random.randrange(4)
                pred, recon = d(real, 'real', part)
                targets = [real, real, crop_part(real, part)]
                loss_d = F.relu(torch.rand_like(pred) * .2 + .8 - pred).mean()
                for rec, target in zip(recon, targets):
                    rgb = F.interpolate(target[:, :3], rec.shape[-2:], mode='bilinear', align_corners=False)
                    mask = F.interpolate(target[:, 3:], rec.shape[-2:], mode='nearest')
                    loss_d = loss_d + percept(rec[:, :3], rgb).mean() + F.l1_loss(rec[:, 3:], mask)
                loss_d.backward()
                with torch.no_grad():
                    fake = g(torch.randn(args.batch, 256, device='cuda'))
                fake_pred = d(fake, 'fake')
                loss_fake = F.relu(torch.rand_like(fake_pred) * .2 + .8 + fake_pred).mean()
                loss_fake.backward()
                opt_d.step()
                d.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                fake_g = g(torch.randn(args.batch, 256, device='cuda'))
                loss_g = -d(fake_g, 'fake').mean()
                loss_g.backward()
                opt_g.step()
                if not (torch.isfinite(loss_g) and torch.isfinite(loss_d) and torch.isfinite(loss_fake)):
                    raise RuntimeError('Niefinitywna strata')
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                completed = step + 1
                if step >= 10:
                    durations.append(seconds)
                row = dict(step=completed, seconds=seconds, loss_g=float(loss_g.detach()),
                           loss_d=float((loss_d + loss_fake).detach()),
                           allocated_mib=torch.cuda.memory_allocated()/1024**2,
                           reserved_mib=torch.cuda.memory_reserved()/1024**2)
                log.write(json.dumps(row) + '\n')
                log.flush()
                if completed % 20 == 0 or step == 0:
                    print(json.dumps(row), flush=True)
                del real, pred, recon, targets, loss_d, fake, fake_pred, loss_fake, fake_g, loss_g
        if completed:
            g.eval()
            with torch.no_grad():
                generated = g(fixed)[0].clamp(-1, 1).cpu()
            save_image((generated[:, :3] + 1)/2, args.out / 'PODGLAD_RGB.png', nrow=2)
            save_image((generated[:, 3:] + 1)/2, args.out / 'PODGLAD_MASKI.png', nrow=2)
            torch.save(dict(g=g.state_dict(), d=d.state_dict(), opt_g=opt_g.state_dict(),
                            opt_d=opt_d.state_dict(), steps=completed, config=config,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
                       args.out / 'pilot.pth')
    except Exception as error:
        status = 'failed'
        (args.out / 'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(f'ERROR: {error}', flush=True)
    report = dict(status=status, steps=completed, elapsed_seconds=time.perf_counter()-started,
                  median_step_seconds=float(np.median(durations)) if durations else None,
                  p90_step_seconds=float(np.quantile(durations, .9)) if durations else None,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                  projected_50000_steps_hours=float(np.median(durations)*50000/3600) if durations else None,
                  quality='not evaluated; timing extrapolation does not guarantee convergence')
    (args.out / 'REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if status == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
