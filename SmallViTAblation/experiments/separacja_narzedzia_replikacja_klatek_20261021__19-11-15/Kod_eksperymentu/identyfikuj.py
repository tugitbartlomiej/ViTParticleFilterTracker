"""Która klatka stoi za którą historią w cache_uczenia/lasery (porównanie rgb z obrazami w app/zdjeciatestowe i atlasie)."""
import sys, json
from pathlib import Path
import numpy as np
APP = Path(__file__).resolve().parents[3] / 'app'
sys.path.insert(0, str(APP))
import lasery_dane as LD
kand = sorted((APP / 'zdjeciatestowe').glob('*.jpg'))
obrazy = {}
for p in kand:
    try:
        obrazy[p.name] = np.asarray(LD.wczytaj_zdjecie(p)[0] if isinstance(LD.wczytaj_zdjecie(p), tuple) else LD.wczytaj_zdjecie(p), np.float32)
    except Exception as e:
        print('pomijam', p.name, e)
for h in ['140854f7a6dbcbe5', '2cca2743e1c74b14', 'c584ef4647149c7c']:
    rgb = np.load(APP / f'cache_uczenia/lasery/{h}/dane.npz')['rgb'].astype(np.float32)
    best = sorted(((np.abs(v - rgb).mean(), n) for n, v in obrazy.items() if v.shape == rgb.shape))[:2]
    print(h, best, 'uczacy' if best and LD.czy_w_zbiorze_uczacym(best[0][1]) else 'nie w zbiorze uczacym / ?')
