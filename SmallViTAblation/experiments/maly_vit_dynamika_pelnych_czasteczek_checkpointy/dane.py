"""Load the verified trajectory without copying the full 22 GB array.

All time indices are updates of one Eden run. Original image/token indices
remain fixed. The older 60/120 local run and other s47/s48 runs are not input.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'maly_vit_ruch_czastek_gesta_historia' / 'wyniki' / 'pelny'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def config():
    return json.loads((HERE / 'CONFIG.json').read_text(encoding='utf-8'))


def load_verified():
    receipt = json.loads((SOURCE / 'pobranie_latentu_2026-10-01' / 'MANIFEST_LOKALNY_2026-10-01.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'complete_verified' and receipt['sha256_match']
    meta = json.loads((SOURCE / 'dane' / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'complete' and meta['slurm_job'] == '1818732'
    assert receipt['remote_sha256'] == meta['latent_sha256']
    p = SOURCE / 'dane' / 'LATENT_GESTY.npy'
    assert p.stat().st_size == receipt['bytes'] == meta['latent_bytes']
    a = np.load(p, mmap_mode='r', allow_pickle=False)
    assert isinstance(a, np.memmap) and a.shape == (1001, 12, 600, 768)
    assert a.dtype == np.float32 and np.array_equal(meta['kroki'], np.arange(1001))
    small_manifest = json.loads((SOURCE / 'pobranie_latentu_2026-10-01' / 'METADANE_ZDALNE_2026-10-01.json').read_text(encoding='utf-8'))
    for item in small_manifest:
        assert sha(SOURCE / 'dane' / item['name']) == item['sha256']
    masks = dict(np.load(SOURCE / 'dane' / 'maski.npz', allow_pickle=False))
    assert masks['coverage'].shape == (12, 600)
    assert np.array_equal(masks['tool'], masks['coverage'] > 0)
    images = json.loads((SOURCE / 'dane' / 'images.json').read_text(encoding='utf-8'))
    assert [r['frame_id'] for r in images] == [r['frame_id'] for r in meta['images']]
    return a, masks, meta


def masked_trajectory():
    a, masks, meta = load_verified()
    mi, ti = np.nonzero(masks['tool'])
    assert len(mi) == 218
    steps = np.arange(0, 1001, config()['sample_every_updates'])
    z = np.empty((len(steps), len(mi), 768), dtype=np.float32)
    for j, t in enumerate(steps):
        z[j] = a[t, mi, ti]
    assert np.isfinite(z).all()
    coverage = masks['coverage'][mi, ti].astype(np.float64)
    weights = np.zeros_like(coverage)
    for i in range(12):
        keep = mi == i
        weights[keep] = coverage[keep] / coverage[keep].sum() / 12
    assert np.isclose(weights.sum(), 1)
    return z, steps, mi, ti, coverage, weights, meta
