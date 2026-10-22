"""Mapa pułapek 600 cząsteczek na 20 checkpointach: czas (float32 i float64), zgodność z ogniskowa.policz (370, 12) i między dokładnościami."""
import os, sys, time, tempfile, pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import lasery_dane as LD, mapa_pulapek as MP
stara = pickle.load(open(sys.argv[1], 'rb'))
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
n = 20
hs = lambda: SimpleNamespace(katalog=Path(tempfile.mkdtemp()), kroki=h.kroki[200:200 + n], wagi=h.wagi, rgb=h.rgb, kol=h.kol, pos=h.pos[200:200 + n], r99=h.r99[200:200 + n])
MP.policz(hs())
w = {}
for dok in ('float32', 'float64'):
    t0 = time.perf_counter(); w[dok] = MP.policz(hs(), dokladnosc=dok); print(dok, f'{(time.perf_counter()-t0)/n*1000:.0f} ms/ckpt (600 cząsteczek)', flush=True)
for tok in (370, 12):
    s = stara[tok]; d = w['float32']
    print(tok, 'zb zgodne', np.array_equal(s['zb'], d['zb'][:, tok]),
          'ogn maks. różnica / r_kom', float((np.linalg.norm(s['ogn'] - d['ogn'][:, tok], axis=1) / s['r_kom'])[s['zb']].max()) if s['zb'].any() else '-',
          'lam maks. wzgl.', float((np.abs(s['lam'] - d['lam'][:, tok]) / np.abs(s['lam']).max(1, keepdims=True)).max()))
a, b = w['float32'], w['float64']
print('float32 vs float64: zb różne', int((a['zb'] != b['zb']).sum()), 'z', a['zb'].size,
      '; ogn maks. / r_kom (oba zb)', float((np.linalg.norm(a['ogn'] - b['ogn'], axis=2) / b['r_kom'])[a['zb'] & b['zb']].max()),
      '; lam maks. wzgl.', float((np.abs(a['lam'] - b['lam']) / np.abs(b['lam']).max(2, keepdims=True)).max()))
ta, _ = MP.charakter(a['lam'], a['zb']); tb, _ = MP.charakter(b['lam'], b['zb'])
print('typy różne', int((ta != tb).sum()), '; zb udział', float(a['zb'].mean()), MP.statystyki(a, 0))
