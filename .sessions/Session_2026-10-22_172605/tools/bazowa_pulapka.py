"""Wynik i czas starej ogniskowa.policz na 20 checkpointach (tok 370 i 12, G=17) do porównania z nową wersją."""
import os, sys, time, tempfile, pickle
from pathlib import Path
from types import SimpleNamespace
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import lasery_dane as LD, ogniskowa as OGN
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
n = 20; tmp = Path(tempfile.mkdtemp())
hs = SimpleNamespace(katalog=tmp, kroki=h.kroki[200:200 + n], wagi=h.wagi, rgb=h.rgb, kol=h.kol, pos=h.pos[200:200 + n], r99=h.r99[200:200 + n])
out = {}
for tok in (370, 12):
    t0 = time.perf_counter(); out[tok] = OGN.policz(hs, tok, G=17, zasieg=1.0); print(tok, f'{(time.perf_counter()-t0)/n*1000:.0f} ms/ckpt', flush=True)
pickle.dump(out, open(sys.argv[1], 'wb'))
