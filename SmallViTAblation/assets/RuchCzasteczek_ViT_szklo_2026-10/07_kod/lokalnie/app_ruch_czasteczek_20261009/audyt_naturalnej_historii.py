"""Read-only audit of the downloaded run and exact 1500-image evaluation manifest."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/app_ruch_czasteczek_20261009'
sys.path.insert(0, str(ROOT / 'app'))
import torch
import ruch_zbioru as R


def audit():
    family = ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
    downloaded = json.loads((family / 'pobranie_wag_2026-10-01/MANIFEST_LOKALNY_20261001_163035.json').read_text(encoding='utf-8'))
    records = []
    for entry in downloaded['files']:
        path = family / 'wagi' / entry['name']
        sha = R._sha(path)
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        assert sha == entry['local_sha256'] == entry['remote_sha256']
        assert ckpt['krok'] == entry['step']
        records.append({'path': str(path), 'step': ckpt['krok'], 'sha256': sha, 'bytes': path.stat().st_size})
    assert [r['step'] for r in records] == list(range(0, 1001, 10))
    assert records[-1]['sha256'] == R._sha(R.DEFAULT_CHECKPOINT)
    manifest = [json.loads(line) for line in R.DEFAULT_MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
    coco_path = ROOT / 'experiments/maly_vit_od_zera/eden/wejscia/pretrain/train/_annotations.coco.json'
    coco = json.loads(coco_path.read_text(encoding='utf-8'))
    names = [str(r['file_name']).replace('\\', '/').rsplit('/', 1)[-1] for r in coco['images']]
    assert names == [r['image_name'] for r in manifest] and len(names) == 1500
    logs = []
    for name in ['LOG.json', 'OCENY.json', 'metadata.json']:
        path = family / 'dane' / name
        logs.append({'path': str(path), 'sha256': R._sha(path), 'bytes': path.stat().st_size})
    result = {'status': 'verified', 'count': len(records), 'steps': [r['step'] for r in records],
              'checkpoints': records, 'endpoint_matches_assets_k1000': True,
              'image_count': len(manifest), 'point_count': len(manifest) * 600,
              'manifest': str(R.DEFAULT_MANIFEST), 'manifest_sha256': R._sha(R.DEFAULT_MANIFEST),
              'image_order_matches_training_coco': True, 'training_coco_sha256': R._sha(coco_path),
              'logs': logs, 'training_run': '2026-10-01_maly_vit_ruch_czastek_gesta_historia',
              'training_seed': 47, 'stored_weight_stride': 10,
              'dense_historical_probe_images': 12, 'retraining': False}
    OUT.mkdir(parents=True, exist_ok=True)
    R._json(OUT / 'AUDYT_NATURALNEJ_HISTORII.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checkpoints', 'steps')}, ensure_ascii=False))


if __name__ == '__main__':
    audit()
