import os
import sys

APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\app'
sys.path.insert(0, APP); os.chdir(APP)
os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app  # noqa: E402

o = app.Okno(); o.show()
print('karty:', [o.zakladki_glowne.tabText(i) for i in range(o.zakladki_glowne.count())], flush=True)


def buduj():
    for nazwa, *_ in app.ZAKLADKI_LENIWE:
        z = o.zbuduj_zakladke(nazwa)
        print(nazwa, 'OK' if z is not None else 'NIEDOSTĘPNA', flush=True)
    a.quit()


QTimer.singleShot(1500, buduj)
a.exec()
