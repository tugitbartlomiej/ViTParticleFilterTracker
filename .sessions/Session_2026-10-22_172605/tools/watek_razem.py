import os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
from PyQt6.QtWidgets import QApplication
a = QApplication(sys.argv)
import lasery_dane as LD, ogniskowa as OGN
from zakladka_laserow import OgniskowaWatek
h = LD.Historia(Path(APP) / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14')
hs = SimpleNamespace(katalog=Path(tempfile.mkdtemp()), kroki=h.kroki[:8], wagi=h.wagi, rgb=h.rgb, kol=h.kol, pos=h.pos[:8], r99=h.r99[:8])
w = OgniskowaWatek(hs, 370, 17, 1.0, razem=[12, 370, 101])
wynik = {}
w.gotowe.connect(lambda args, W: (wynik.update(args=args, toki=sorted(W)), a.quit()))
w.blad.connect(lambda t: (print('błąd', t), a.quit()))
w.postep.connect(lambda i, n, o: print(o) if i == n else None)
w.start(); a.exec(); w.wait()
print(wynik, [OGN.plik(hs, t, 17, 1.0).exists() for t in (370, 12, 101)])
