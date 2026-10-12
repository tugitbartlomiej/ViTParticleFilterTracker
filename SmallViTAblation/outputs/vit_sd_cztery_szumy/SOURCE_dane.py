"""Wspólne wejścia; after jest sztucznym celem, a channels0 indeksem od zera.
Nie wczytujemy fotografii. Ocena używa naturalnego latentu z zapisanego PNG.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/vit_sd_cztery_szumy'
TARGET = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90'
DIAGNOSTIC = ROOT / 'outputs/vit_latent768_pelny_zbior/DIAGNOSTYKA_OSIAGALNOSCI.json'
SD = Path('F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele')
SEEDS = (2026100511, 2026100523, 2026100537, 2026100549)

def load_target():
    with np.load(TARGET / 'LATENT_PELNY.npz') as f:
        target, channels = f['after'].copy(), f['channels0'].copy()
    assert target.shape == (600, 768) and np.isfinite(target).all()
    assert channels.shape == (128,) and len(np.unique(channels)) == 128
    assert channels.min() >= 0 and channels.max() < 768
    return target, channels
