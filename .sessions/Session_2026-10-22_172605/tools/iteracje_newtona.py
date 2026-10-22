"""Ile iteracji Newtona potrzeba: zb i u dla 600 cząsteczek przy 8/15/30/60 iteracjach, na 30 checkpointach z całej historii."""
import os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import torch, lasery_dane as LD, ogniskowa as OGN
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
ix = np.linspace(0, h.K - 1, 30).astype(int)
hs = SimpleNamespace(katalog=None, kroki=h.kroki[ix], wagi=h.wagi, rgb=h.rgb, kol=h.kol, pos=h.pos[ix], r99=h.r99[ix])
dev = torch.device('cuda'); tk = torch.arange(600, device=dev)
wyn = {n: [] for n in (8, 15, 30, 60)}; czasy = {n: 0.0 for n in wyn}
for i, k, kt in OGN.checkpointy(hs, dev):
    _, _, Li = OGN.komorki(kt['pc'], tk, 1.0)
    for n in wyn:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        u, zb, *_ = OGN.newton_wsadowo(kt, tk, Li, iteracje=n)
        torch.cuda.synchronize(); czasy[n] += time.perf_counter() - t0
        wyn[n].append(zb.cpu().numpy())
for n in wyn:
    z = np.array(wyn[n]); print(n, 'iteracji: zera w komórce', int(z.sum()), 'różne od 60:', int((z != np.array(wyn[60])).sum()), f'{czasy[n]/30*1000:.1f} ms/ckpt')
