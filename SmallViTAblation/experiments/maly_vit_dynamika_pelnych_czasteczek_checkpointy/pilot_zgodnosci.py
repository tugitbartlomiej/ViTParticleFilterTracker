"""Test whether local checkpoint forwards reproduce the recorded full tokens.

A wrong layer, altered BN, nonrepeatable output or relative L2 > 1e-4 rejects
the planned reuse of this local forward path. No ViT weights are optimized.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import json
import platform
import shutil
from datetime import datetime, timezone
import numpy as np
import psutil
import torch
import timm
from dane import HERE, SOURCE, config, load_verified, sha
from model_snapshot import MalyViT


def main():
    cfg = config()
    a, _, meta = load_verified()
    assert sha(HERE / 'model_snapshot.py') == next(r['sha256'] for r in meta['code'] if r['path'] == 'model.py')
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    hardware = dict(free_disk_bytes=shutil.disk_usage(HERE).free,
                    ram_available_bytes=psutil.virtual_memory().available,
                    gpu_free_bytes=free_gpu, gpu_total_bytes=total_gpu,
                    gpu_name=torch.cuda.get_device_name(0))
    assert hardware['free_disk_bytes'] > cfg['minimum_free_disk_bytes']
    assert hardware['ram_available_bytes'] > 4 * 1024**3 and free_gpu > 2 * 1024**3
    x = np.load(SOURCE / 'dane' / 'obrazy.npy', allow_pickle=False)
    assert x.dtype == np.uint8 and x.shape == (12, 320, 480, 3)
    X = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().float().cuda() / 255.
    net = MalyViT().cuda().eval()
    net.zapisuj_latent(True)
    results = []
    with torch.no_grad():
        for step in cfg['forward_checkpoints']:
            path = SOURCE / 'wagi' / f'k{step:04d}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert checkpoint['krok'] == step
            net.load_state_dict(checkpoint['net'], strict=True)
            bn0 = {n: b.clone() for n, b in net.named_buffers()}
            _, z1 = net(X)
            saved = z1.detach().clone()
            _, z2 = net(X)
            repeat = float((z2 - saved).abs().max())
            arr = saved.cpu().numpy()
            assert arr.shape == (12, 600, 768) and np.isfinite(arr).all()
            reference = np.asarray(a[step])
            error = arr.astype(np.float64) - reference
            rel = float(np.linalg.norm(error) / max(np.linalg.norm(reference.astype(np.float64)), 1e-30))
            max_abs = float(np.abs(error).max())
            bn_unchanged = all(torch.equal(b, bn0[n]) for n, b in net.named_buffers())
            row = dict(step=step, checkpoint_sha256=sha(path), shape=list(arr.shape),
                       repeated_forward_max_abs_error=repeat, reference_relative_l2_error=rel,
                       reference_max_abs_error=max_abs, buffers_unchanged=bn_unchanged,
                       passed=repeat <= 1e-6 and rel <= cfg['forward_relative_l2_tolerance'] and bn_unchanged)
            results.append(row)
            print(json.dumps(row), flush=True)
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='passed' if all(r['passed'] for r in results) else 'failed',
                  hardware_before=hardware, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  process_peak_working_set_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None),
                  versions=dict(python=platform.python_version(), torch=torch.__version__, timm=timm.__version__, numpy=np.__version__),
                  recorded_versions=meta['versions'], model_sha256=sha(HERE / 'model_snapshot.py'),
                  input_normalization='RGB uint8 /255 then model ImageNet mean/std',
                  measurement='last block mlp.act GELU, eval, no_grad, fp32, autocast off, TF32 off',
                  training_performed=False, results=results)
    (HERE / 'AUDYT_ZGODNOSCI.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    assert report['status'] == 'passed', 'Forward audit failed; inspect AUDYT_ZGODNOSCI.json before forecasting.'


if __name__ == '__main__':
    main()
