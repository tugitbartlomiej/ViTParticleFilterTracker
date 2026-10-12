"""Eden runtime preflight; deliberately never run locally for this delivery."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from .dane import sha256

OFFICIAL_NANO_MD5 = 'fb6504cce7fbdc783f7a46991f07639f'


def check(initial_weights):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('runtime checks must run inside the Eden Slurm job')
    import torch
    if importlib.metadata.version('rfdetr') != '1.10.1':
        raise RuntimeError('this adapter is pinned to RF-DETR 1.10.1')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one visible CUDA GPU is required')
    initial_weights = Path(initial_weights)
    digest = hashlib.md5()
    with initial_weights.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            digest.update(block)
    if digest.hexdigest() != OFFICIAL_NANO_MD5:
        raise RuntimeError('initial weights do not match the official generic COCO Nano asset; old L0 is forbidden')
    provenance = json.loads(Path(__file__).with_name('vendor').joinpath('PROVENANCE.json').read_text(encoding='utf-8'))
    for row in provenance['files']:
        destination = row['destination']
        if '/' not in destination:
            if sha256(Path(__file__).with_name('vendor')/destination) != row['destination_sha256']:
                raise RuntimeError(f'frozen vendor source changed: {destination}')
    return dict(rfdetr=importlib.metadata.version('rfdetr'), torch=torch.__version__,
                lightning=importlib.metadata.version('pytorch-lightning'), numpy=importlib.metadata.version('numpy'),
                gpu=torch.cuda.get_device_name(0), initial_weights_sha256=sha256(initial_weights),
                official_coco_asset_md5=digest.hexdigest(), job_id=os.environ['SLURM_JOB_ID'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-weights', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(check(a.initial_weights), indent=2))


if __name__ == '__main__':
    main()
