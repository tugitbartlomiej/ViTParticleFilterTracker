"""Profil pulapka_obliczenia.policz_historie (pomiar pułapki w 12 podkrokach) na 6 checkpointach."""
import os, sys, time, cProfile, pstats, io, tempfile
from pathlib import Path
from types import SimpleNamespace
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
import lasery_dane as LD, pulapka_obliczenia as PO, przejscia_zapis as PZ
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
kat = PZ.katalog(h); print('katalog przejść', kat, 'pliki:', len(list(Path(kat).glob('*'))) if Path(kat).exists() else 0)
n = 12
hs = SimpleNamespace(**{k: getattr(h, k) for k in ('wagi', 'rgb', 'kol', 'pos', 'r99')}); hs.katalog = Path(tempfile.mkdtemp()); hs.kroki = h.kroki[400:400 + n]; hs.K = n
import pulapka_obliczenia
hs2 = SimpleNamespace(**vars(hs)); hs2.kroki = h.kroki[300:302]; hs2.K = 2; PO.policz_historie(hs2, tok=226, co=1, G=5)  # rozgrzewka importów
pr = cProfile.Profile(); pr.enable(); t0 = time.perf_counter()
p = PO.policz_historie(hs, tok=226, co=1, G=5)
pr.disable(); print(f'{(time.perf_counter()-t0)/n*1000:.0f} ms/ckpt', p)
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(8); print(s.getvalue()[:4500])
