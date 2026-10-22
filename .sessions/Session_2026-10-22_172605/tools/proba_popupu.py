"""Okienko ustawień z prawego kliku: od razu osobne okno, drugi klik przywołuje to samo, ⊟ zwija grupy."""
import os, sys, time
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt, QTimer, QPoint
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app
from panel_ustawien import PopupUstawien
o = app.Okno(); o.resize(1400, 900); o.show(); z = o.zakladka_laserow
def p(n=10):
    for _ in range(n): a.processEvents(); time.sleep(0.02)
def start():
    z.menu_okienka('11', QPoint(300, 200)); p()
    pp = PopupUstawien.otwarte[-1]
    print('typ okna', pp.windowType().name, 'otwarte', len(PopupUstawien.otwarte))
    o.activateWindow(); p(5); print('po kliknięciu w okno główne widoczne:', pp.isVisible())
    z.menu_okienka('11', QPoint(600, 400)); p(); print('drugi prawy klik: otwarte', len(PopupUstawien.otwarte), 'to samo', PopupUstawien.otwarte[-1] is pp)
    pp.zwin_wszystko(True); p(); pp.resize(560, 700); p()
    pp.grab().save(os.path.join(OUT, 'popup_zwiniety.png'))
    pp.zwin_wszystko(False); p()
    pp.close(); p(); print('po zamknięciu otwarte', len(PopupUstawien.otwarte))
    a.quit()
QTimer.singleShot(1200, start); a.exec()
