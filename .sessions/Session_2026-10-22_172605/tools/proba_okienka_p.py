"""Okienko P: przełączniki widoczności (wszystkie wiązki, zestawy, filtry laserów) zmieniają obraz; zrzuty widżetu vp, bez ekranu."""
import os, sys, time, traceback
from pathlib import Path
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
HIST = Path(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
bledy = []
sys.excepthook = lambda t, v, tb: bledy.append(''.join(traceback.format_exception(t, v, tb)))
przed = {p for p in (HIST / 'przejscia').rglob('*')} if (HIST / 'przejscia').exists() else set()
bylo_prz = (HIST / 'przejscia').exists()
import numpy as np
import laser_app  # noqa
from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
from laser_app.main_window import MainWindow
o = MainWindow(); o.resize(1600, 950); o.show(); z = o.zakladka_laserow


def p(n=10):
    for _ in range(n):
        a.processEvents(); time.sleep(0.02)


def tablica(img):
    img = img.convertToFormat(img.Format.Format_RGB888)
    w, h = img.width(), img.height()
    b = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(h, img.bytesPerLine())[:, :3 * w].reshape(h, w, 3)
    return b.astype(np.int16)


def zrzut(nazwa):
    z.ep.update(); p(8)
    img = z.vp.grabFramebuffer(); img.save(os.path.join(OUT, f'p_{nazwa}.png'))
    A = tablica(img)
    return A, float((np.abs(A - A[0, 0]).sum(2) > 30).mean())


def start():
    z.ch_prz_zapis.setChecked(False)
    z._gotowe(str(HIST)); p()
    d = z.doki['p']; d.show(); d.raise_(); z.ustaw_czas(400.0); p(10)
    t0 = time.time()
    while z.d_pod is None and time.time() - t0 < 180:
        p(5)
    print('podkroki gotowe', z.d_pod is not None, f'{time.time() - t0:.1f} s', flush=True)
    z.vp.setCameraPosition(distance=z.vp.opts['distance'] / 7); z.sp_skala_l.setValue(30); z.sp_p_grub.setValue(2)
    z.ustaw_podkrok(1.5); p(5)                               # MLP bloku 1: lasery
    wyniki = {}
    z._p_wlacz_wszystkie(); wyniki['wszystko'] = zrzut('wszystko')
    z.ch_p_wiazki.setChecked(False); wyniki['bez_wiazek'] = zrzut('bez_wiazek')
    z._p_zestaw(lasery=True); wyniki['tylko_lasery'] = zrzut('tylko_lasery')
    z.sp_p_top_l.setValue(20); wyniki['lasery_top20'] = zrzut('lasery_top20')
    z.sp_p_top_l.setValue(0); z.cb_p_znak_l.setCurrentIndex(2); wyniki['lasery_ujemne'] = zrzut('lasery_ujemne')
    z.cb_p_znak_l.setCurrentIndex(0)
    z.ch_p_czastki.setChecked(False); z.ch_p_wybrana.setChecked(False); z.ch_p_wiazki.setChecked(False); wyniki['pusto'] = zrzut('pusto')
    z.ch_p_czastki.setChecked(True); z.ch_p_wybrana.setChecked(True)
    z.ustaw_podkrok(0.5); z._p_zestaw(uwaga=True); wyniki['uwaga'] = zrzut('uwaga')
    z.cb_p_glowa.setCurrentIndex(2); wyniki['uwaga_glowa2'] = zrzut('uwaga_glowa2')
    z.cb_p_glowa.setCurrentIndex(0); z._p_wlacz_wszystkie()
    for k, (A, t) in wyniki.items():
        print(f'{k:14s} treść {t:.4f}', flush=True)
    r = lambda x, y: float((np.abs(wyniki[x][0] - wyniki[y][0]).sum(2) > 30).mean())
    print('różnica top20 vs wszystkie lasery', r('lasery_top20', 'tylko_lasery'), ' głowa 2 vs średnia', r('uwaga_glowa2', 'uwaga'))
    print('BŁĘDY', len(bledy)); [print(b[-1500:]) for b in bledy[:3]]
    o.close(); p(5)
    po = {q for q in (HIST / 'przejscia').rglob('*')} if (HIST / 'przejscia').exists() else set()
    nowe = sorted(po - przed, key=lambda q: -len(q.parts))
    for q in nowe:
        q.unlink() if q.is_file() else q.rmdir()
    if not bylo_prz and (HIST / 'przejscia').exists():
        (HIST / 'przejscia').rmdir()
    print('usunięte pliki próby', len(nowe))
    a.quit()


QTimer.singleShot(1500, start); a.exec()
