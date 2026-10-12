"""Przygotowanie filmu 5 (co trzęsie cząsteczkami): gęsty zapis biegu 35 000 (osobny bieg z obserwatorem gęstym, ziarno 47), sonda 0
(wiersz puli 144, TR_0001_S1_P03 klatka 838), pełny latent 768D co 1 krok dla kroków 0–1094 (1. epoka), z PACZKI.npz: długość kroku wag
|δθ|, lr, które klatki były w każdej paczce. Miniatury klatek paczek z kroków WOLNE (z zipów zbioru). Zapis: drgania_X.npy (1095, 600, 768)
f16 w scratchpadzie (1 GB, za duży do outputs) i drgania.npz obok filmu.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 drgania_prep.py <katalog_na_X>"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
A = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis/lokalnie'
G = A / 'gesty/gesty'
OUTX = Path(sys.argv[1]) / 'drgania_X.npy'
WOLNE = list(range(300, 315))                                                      # przejścia x_t → x_{t+1} (krok t+1, paczka idx[t])
X, K = [], []
for i in range(11):
    d = np.load(G / f'pelne768_{i:03d}.npz'); k = d['kroki']; m = k <= 1094
    X.append(d['latent'][m, 0]); K.append(k[m]); print('plik', i, flush=True)
X = np.concatenate(X); K = np.concatenate(K)
assert X.shape == (1095, 600, 768) and np.all(K == np.arange(1095))
np.save(OUTX, X)
p = np.load(G / 'PACZKI.npz')
assert np.all(p['krok'][:1094] == np.arange(1, 1095))
man = open(A / 'wejscia/MANIFEST_35000.jsonl', encoding='utf-8').readlines()
sonda = json.loads(open(A / 'wejscia/SONDY_100.jsonl', encoding='utf-8').readline())
assert sonda['pool_row'] == 144


def obraz(wiersz, rozm):
    r = json.loads(man[wiersz])
    with zipfile.ZipFile(r['archive']) as z:
        im = Image.open(io.BytesIO(z.read(r['member']))).convert('RGB')
    return np.asarray(im.resize(rozm, Image.LANCZOS)), r['film']


zdj, _ = obraz(144, (480, 320))
mini, filmy = [], []
for t in WOLNE:
    rz = [obraz(int(w), (72, 48)) for w in p['idx'][t]]
    mini.append(np.stack([a for a, _ in rz])); filmy.append([f for _, f in rz]); print('paczka', t, flush=True)
np.savez_compressed(HERE / 'drgania.npz', dth=p['norma_aktualizacji'][:1094], lr=p['lr'][:1094], strata=p['strata_przed'][:1094],
                    idx=p['idx'][:1094], wolne=np.array(WOLNE), mini=np.stack(mini), filmy=np.array(filmy), zdjecie=zdj,
                    bbox480=np.array(sonda['bbox_xyxy_480']), frame=sonda['frame_id'])
print('zapisano', OUTX, HERE / 'drgania.npz')
