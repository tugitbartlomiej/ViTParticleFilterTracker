"""Stereo SBS we wszystkich okienkach 3D: lewa i prawa połówka mają treść, różnią się (paralaksa), bez wyjątków."""
import os, sys, time, traceback
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
bledy = []
sys.excepthook = lambda t, v, tb: bledy.append(''.join(traceback.format_exception(t, v, tb)))
import numpy as np
import laser_app  # noqa
from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
from laser_app.main_window import MainWindow
from laser_app.ui.gl_view import GLView
o = MainWindow(); o.resize(1600, 950); o.show(); z = o.zakladka_laserow
def p(n=10):
    for _ in range(n): a.processEvents(); time.sleep(0.02)
def tablica(img):
    img = img.convertToFormat(img.Format.Format_RGB888)
    w, h = img.width(), img.height()
    b = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(h, img.bytesPerLine())[:, :3 * w].reshape(h, w, 3)
    return b.astype(np.int16)
def start():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')); p()
    z.ch_og.setChecked(True); z._og_lista_gotowych(); z.cb_og_gotowe.count() and z._og_wybierz_gotowa(0)
    z.ch_mp.setChecked(True); z.ch_dol.setChecked(True)
    z.ch_stereo.setChecked(True); p(5)
    for k, d in z.doki.items():
        gl_ = [w for w in d.widget().findChildren(GLView)] if d.widget() else []
        if not gl_:
            continue
        d.show(); d.raise_(); z.ustaw_czas(400.3); p(25)
        v = gl_[0]
        if v.width() < 40:
            continue
        A = tablica(v.grabFramebuffer()); h, w, _ = A.shape
        L, R = A[:, :w // 2], A[:, w - w // 2:]
        tresc = lambda X: float((np.abs(X - X[0, 0]).sum(2) > 30).mean())
        roz = float((np.abs(L - R).sum(2) > 30).mean())
        print(f'{k:4s} lewa treść {tresc(L):.3f} prawa {tresc(R):.3f} różnica oczu {roz:.3f}', flush=True)
    v = z.vs; A = v.grabFramebuffer(); A.save(os.path.join(OUT, 'stereo_sfera.png'))
    z.ch_stereo.setChecked(False)
    print('BŁĘDY', len(bledy)); [print(b[-800:]) for b in bledy[:3]]
    a.quit()
QTimer.singleShot(1500, start); a.exec()
