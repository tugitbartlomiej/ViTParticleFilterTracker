"""Profil LaserLatentSpaceApp: czas budowy okna, liczba widoków OpenGL i czas klatki osi czasu (CPU w ustaw_czas + rysowanie)."""
import cProfile
import io
import os
import pstats
import sys
import time

APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
os.environ['EDYTOR_BEZ_USTAWIEN'] = os.environ.get('EDYTOR_BEZ_USTAWIEN', '1')
from PyQt6.QtCore import QCoreApplication, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
t0 = time.perf_counter()
import app  # noqa: E402
from styl import STYL  # noqa: E402
a.setStyleSheet(STYL)
t1 = time.perf_counter()
pr = cProfile.Profile(); pr.enable()
o = app.Okno()
pr.disable()
t2 = time.perf_counter()
o.show(); a.processEvents()
t3 = time.perf_counter()
from widok_gl import WidokGL  # noqa: E402
print(f'import {t1 - t0:.2f} s, Okno() {t2 - t1:.2f} s, show {t3 - t2:.2f} s, widoków GL: {len(list(WidokGL._instancje))}', flush=True)
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(18); print(s.getvalue()[:3500])
z = o.zakladka_laserow


def klatki():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', os.environ.get('HIST', '2cca2743e1c74b14')))
    a.processEvents()
    widoczne = [k for k, d in z.doki.items() if d.isVisible() and not d.visibleRegion().isEmpty()]
    print('widoczne okienka:', widoczne, flush=True)
    for nazwa, ust in (('bez pułapki', {}), ('z pułapką', {'og': True})):
        if ust.get('og'):
            z.ch_og.setChecked(True)
            for _ in range(60):
                time.sleep(0.25); a.processEvents()
                if getattr(z, '_og', None) is not None:
                    break
        K = z.h.K
        ts = [K * 0.3 + i * 0.37 for i in range(120)]
        pr = cProfile.Profile()
        cpu, calk = [], []
        for t in ts:
            c0 = time.perf_counter(); pr.enable(); z.ustaw_czas(t); pr.disable(); c1 = time.perf_counter()
            for v in WidokGL._instancje:
                if v.isVisible():
                    v.repaint()                                           # synchroniczne rysowanie widocznych widoków
            a.processEvents(); c2 = time.perf_counter()
            cpu.append(c1 - c0); calk.append(c2 - c0)
        import numpy as np
        print(f'[{nazwa}] ustaw_czas: mediana {np.median(cpu) * 1e3:.1f} ms, p90 {np.percentile(cpu, 90) * 1e3:.1f} ms;  '
              f'klatka z rysowaniem: mediana {np.median(calk) * 1e3:.1f} ms, p90 {np.percentile(calk, 90) * 1e3:.1f} ms', flush=True)
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(22); print(s.getvalue()[:5000], flush=True)
    a.quit()


QTimer.singleShot(1500, klatki)
a.exec()
