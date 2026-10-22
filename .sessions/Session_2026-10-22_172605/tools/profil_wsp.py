"""Koszt części wspólnej (wagi + sieć + rzut) i pełnego policz na 40 checkpointach, z cProfile."""
import os, sys, time, tempfile, cProfile, pstats, io
from pathlib import Path
from types import SimpleNamespace
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import torch, lasery_dane as LD, ogniskowa as OGN
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
n = 40
hs = lambda: SimpleNamespace(katalog=Path(tempfile.mkdtemp()), kroki=h.kroki[300:300 + n], wagi=h.wagi, rgb=h.rgb, kol=h.kol, pos=h.pos[300:300 + n], r99=h.r99[300:300 + n])
for _ in OGN.checkpointy(hs(), torch.device('cuda')): pass
t0 = time.perf_counter()
for _ in OGN.checkpointy(hs(), torch.device('cuda')): pass
torch.cuda.synchronize(); print('wspólne', f'{(time.perf_counter()-t0)/n*1000:.1f} ms/ckpt', flush=True)
pr = cProfile.Profile(); pr.enable(); t0 = time.perf_counter()
OGN.policz(hs(), 370, G=17)
pr.disable(); print('policz 1', f'{(time.perf_counter()-t0)/n*1000:.1f} ms/ckpt')
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(14); print(s.getvalue()[:3000])
