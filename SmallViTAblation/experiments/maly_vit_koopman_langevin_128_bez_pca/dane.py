"""Zapewnia identyczne natywne kanały i dane dla porównania; różnica SHA unieważnia test."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'maly_vit_tozsamosc90_bez_pca_langevin'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf-8')

def load_bank(dt):
    meta = read(SOURCE / f'MODELE_DT{dt}.json')
    with np.load(SOURCE / f'MACIERZE_DT{dt}.npz') as arr:
        for key in arr.files:
            parts = key.split('__')
            if len(parts) == 3: meta[parts[0]][parts[1]][parts[2]] = arr[key]
            else: meta[parts[0]][parts[1]] = arr[key]
    return meta

def load_data():
    source_hashes = {}
    with np.load(SOURCE / 'DYNAMIKA_NATYWNE_J.npz') as arr:
        data = {key: arr[key] for key in arr.files}
    choice = read(SOURCE / 'WYBOR_WALIDACJA.json')
    assert np.array_equal(data['channels0'], choice['channels0'])
    assert len(data['channels0']) == len(set(data['channels0'].tolist())) == 128
    for dt in (1, 10):
        record = read(SOURCE / f'WYBOR_DYNAMIKA_DT{dt}.json')
        for name, expected in [('DYNAMIKA_NATYWNE_J.npz', record['data_sha']),
                               ('KANALY_MODEL.npz', record['channels_model_sha']),
                               (f'MACIERZE_DT{dt}.npz', record['matrices_sha'])]:
            actual = sha(SOURCE / name)
            assert actual == expected, name
            source_hashes[name] = actual
    # Subtraction is performed only after converting original float32 measurements to float64.
    for p in (2, 8):
        data[f's_{p}'] = data[f's_{p}'].astype(np.float64)
        assert data[f's_{p}'].shape[-1] == 128
        assert np.isclose(data[f'w_{p}'].sum(), 1)
    return data, source_hashes
