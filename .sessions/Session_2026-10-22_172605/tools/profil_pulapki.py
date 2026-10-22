"""Czas ogniskowa.policz na 20 checkpointach (zapis do katalogu tymczasowego): z polem na siatce i bez."""
import cProfile
import io
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import lasery_dane as LD  # noqa: E402
import ogniskowa as OGN  # noqa: E402

h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
n = 20
tmp = Path(tempfile.mkdtemp())
hs = SimpleNamespace(katalog=tmp, kroki=h.kroki[200:200 + n], wagi=h.wagi, rgb=h.rgb, kol=h.kol, pos=h.pos[200:200 + n], r99=h.r99[200:200 + n])
for G in (1, 17):
    pr = cProfile.Profile()
    t0 = time.perf_counter(); pr.enable()
    OGN.policz(hs, 370, G=G, zasieg=1.0)
    pr.disable(); dt = time.perf_counter() - t0
    print(f'G={G}: {dt / n * 1000:.0f} ms na checkpoint → {dt / n * 806 / 60:.1f} min na 806', flush=True)
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(10); print(s.getvalue()[:2500])
