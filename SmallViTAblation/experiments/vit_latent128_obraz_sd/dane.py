"""Wczytanie jednego rzeczywistego celu: J128 dotyczy 600 tokenów pełnej klatki.

Nie używamy uśrednionego latentu wycinka 176 px z dawnej pętli SD.
Cel pochodzi z TARGETS_FULL400, nie z odrzuconej próby REFINED400.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'outputs/vit_optymalizacja_latentu128/input'
TARGET_DIR = ROOT / 'outputs/vit_latent128_maksymalizacja'
OUT = ROOT / 'outputs/vit_latent128_obraz_sd'
SOURCE = Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation')
CKPT = SOURCE / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k1000.pt'
SD = Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele')
ANNOTATIONS = SOURCE.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(index=0):
    rows = json.loads((INPUT / 'manifest.json').read_text(encoding='utf-8'))
    row = next(r for r in rows if r['index'] == index)
    path = INPUT / row['image']
    assert sha(path) == row['sha256']
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    target_path = TARGET_DIR / 'TARGETS_FULL400.npz'
    with np.load(target_path) as data:
        k = int(np.flatnonzero(data['source_indices'] == index)[0])
        selected = {name: data[name][k].copy() for name in ('base', 'target', 'delta_raw')}
        selected['channels0'] = data['channels0'].copy()
    expected = json.loads((TARGET_DIR / 'RESULT.json').read_text(encoding='utf-8'))
    assert sha(CKPT) == expected['checkpoint_sha']
    assert selected['target'].shape == (600, 128)
    assert len(np.unique(selected['channels0'])) == 128
    return row, image, selected, target_path
