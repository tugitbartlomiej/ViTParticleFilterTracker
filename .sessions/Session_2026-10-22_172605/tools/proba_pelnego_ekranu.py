"""Pełny ekran okienka sfery ze stereo: tryb dla nakładek (okno o 2 px większe od ekranu, dla Odyssey 3D Hub) i klasyczny.
Każdy tryb w osobnym procesie: python proba_pelnego_ekranu.py 0|1. Wypisuje prostokąt okna i obszar klienta (piksele fizyczne),
czy widok OpenGL rysuje stereo (grabFramebuffer, bez zrzutu ekranu) i SHQueryUserNotificationState (2 = zwykły pełny ekran,
3 = Direct3D na wyłączność, 5 = pulpit). Błąd kończy próbę (bez zawieszenia)."""
import ctypes, os, sys, time, traceback
from ctypes import wintypes
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
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
from laser_app.ui import dock_windows as dw
o = MainWindow(); o.resize(1600, 950); o.show(); z = o.zakladka_laserow
tryb = sys.argv[1] if len(sys.argv) > 1 else '1'


def p(n=10):
    for _ in range(n):
        a.processEvents(); time.sleep(0.02)


def stan():
    s = ctypes.c_int(0); ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(s)); return s.value


def stereo(v):
    img = v.grabFramebuffer().convertToFormat(v.grabFramebuffer().Format.Format_RGB888)
    w, h = img.width(), img.height()
    A = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(h, img.bytesPerLine())[:, :3 * w].reshape(h, w, 3)
    A = A.astype(np.int16); L, R = A[:, :w // 2], A[:, w - w // 2:]
    tresc = lambda X: float((np.abs(X - X[0, 0]).sum(2) > 30).mean())
    return f'{w}x{h}, treść L {tresc(L):.3f} P {tresc(R):.3f}, różnica oczu {float((np.abs(L - R).sum(2) > 30).mean()):.3f}'


def start():
    try:
        z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')); p()
        z.ch_stereo.setChecked(True); d = z.doki['s']; d.show(); d.raise_(); z.ustaw_czas(400.3); p(15)
        z.ch_pelny_nakladki.setChecked(tryb == '1'); dw._pelny_dok(d); p(40)
        hwnd = wintypes.HWND(int(d.winId()))
        r = wintypes.RECT(); ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
        c = wintypes.RECT(); ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(c))
        st = []
        for _ in range(6):
            p(15); st.append(stan())
        print(f'tryb {"dla nakładek" if tryb == "1" else "klasyczny"}: okno {(r.left, r.top, r.right, r.bottom)}, klient {c.right}x{c.bottom}, '
              f'stan Qt {int(d.windowState().value)}, powiadomienia {st}, widok {stereo(z.vs)}', flush=True)
        dw._pelny_dok(d); p(5)
        z.ch_pelny_nakladki.setChecked(True); z.ch_stereo.setChecked(False)
    except Exception:
        bledy.append(traceback.format_exc())
    print('BLEDY', len(bledy)); [print(b[-1200:]) for b in bledy[:3]]
    o.close(); p(3); a.quit()


QTimer.singleShot(1500, start); a.exec()
