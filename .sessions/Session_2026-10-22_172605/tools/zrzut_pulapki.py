import os, sys, time
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app
o = app.Okno(); o.resize(1400, 900); o.show(); z = o.zakladka_laserow
def p(n=10):
    for _ in range(n): a.processEvents(); time.sleep(0.02)
def start():
    s = z._sek['pulapka']; s.setParent(None); s.resize(560, s.sizeHint().height()); s.show(); p()
    im = s.grab(); h = im.height()
    im.copy(0, 0, 560, h // 2).save(os.path.join(OUT, 'pulapka_1.png')); im.copy(0, h // 2, 560, h - h // 2).save(os.path.join(OUT, 'pulapka_2.png'))
    print(h); a.quit()
QTimer.singleShot(1000, start); a.exec()
