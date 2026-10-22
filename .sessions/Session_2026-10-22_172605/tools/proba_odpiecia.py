"""Odpięte okienka: zwykłe okna (nie Qt.Tool), sfera po odpięciu dalej rysuje, F2/F3 chowają i pokazują, przypinanie z powrotem."""
import os
import sys
import time

APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP); os.chdir(APP)
os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app  # noqa: E402

o = app.Okno(); o.show()
z = o.zakladka_laserow


def p(n=20):
    for _ in range(n):
        a.processEvents(); time.sleep(0.02)


def start():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')); p()
    for d in (z.dok_panel, z.dok_podkroki, z.doki['s']):
        d.setFloating(True); p(15)
        print(d.objectName(), 'pływa', d.isFloating(), 'typ okna', d.windowType().name, 'widoczne', d.isVisible(), flush=True)
    z.ustaw_czas(300.0); p(20)
    img = z.doki['s'].widget().grab(); img.save(os.path.join(OUT, 'sfera_odpieta.png'))
    print('obraz sfery odpiętej', img.width(), img.height(), flush=True)
    o.raise_(); o.activateWindow(); p(10)
    for a_ in o.menuBar().actions():
        if a_.text() == 'Widok':
            for b in a_.menu().actions():
                if b.shortcut().toString() in ('F2', 'F3'):
                    b.trigger(); p(5); print(b.shortcut().toString(), 'po 1. naciśnięciu widoczne:', z.dok_panel.isVisible() if b.shortcut().toString() == 'F2' else z.dok_podkroki.isVisible())
                    b.trigger(); p(5); print(b.shortcut().toString(), 'po 2. naciśnięciu widoczne:', z.dok_panel.isVisible() if b.shortcut().toString() == 'F2' else z.dok_podkroki.isVisible(),
                                                  'typ', (z.dok_panel if b.shortcut().toString() == 'F2' else z.dok_podkroki).windowType().name, flush=True)
    for d in (z.dok_panel, z.dok_podkroki, z.doki['s']):
        d.setFloating(False); p(5)
    print('przypięte z powrotem:', [d.isFloating() for d in (z.dok_panel, z.dok_podkroki, z.doki['s'])], flush=True)
    a.quit()


QTimer.singleShot(1500, start)
a.exec()
