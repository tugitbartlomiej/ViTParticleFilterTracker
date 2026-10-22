"""Nowa ogniskowa: czas policz (1 token) i policz_wiele (3 tokeny), zgodność ze starą wersją (pulapka_stara.pkl)."""
import os, sys, time, tempfile, pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import lasery_dane as LD, ogniskowa as OGN
stara = pickle.load(open(sys.argv[1], 'rb'))
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
n = 20


def hs():
    return SimpleNamespace(katalog=Path(tempfile.mkdtemp()), kroki=h.kroki[200:200 + n], wagi=h.wagi, rgb=h.rgb, kol=h.kol,
                           pos=h.pos[200:200 + n], r99=h.r99[200:200 + n])


OGN.policz(hs(), 5, G=17)  # rozgrzewka
t0 = time.perf_counter(); d1 = OGN.policz(hs(), 370, G=17); print('1 token', f'{(time.perf_counter()-t0)/n*1000:.0f} ms/ckpt', flush=True)
t0 = time.perf_counter(); dw = OGN.policz_wiele(hs(), [370, 12, 101], G=17); print('3 tokeny', f'{(time.perf_counter()-t0)/n*1000:.0f} ms/ckpt', flush=True)
for tok, nazwa, d in ((370, 'policz', d1), (370, 'wiele', dw[370]), (12, 'wiele', dw[12])):
    gorsze = []
    for kk, v in stara[tok].items():
        a = np.asarray(v); b = np.asarray(d[kk])
        if kk in ('wek', 'wek0'):
            continue                                                       # wektory własne: dowolny znak/faza
        if a.dtype.kind in 'fc':
            r = np.abs(a.astype(np.complex128) - b.astype(np.complex128)).max() / max(np.abs(a).max(), 1e-30)
            if r > 1e-3:
                gorsze.append((kk, float(r)))
        elif not np.array_equal(a, b):
            gorsze.append((kk, 'różne', int((a != b).sum())))
    print(tok, nazwa, 'różnice > 1e-3:', gorsze, flush=True)
