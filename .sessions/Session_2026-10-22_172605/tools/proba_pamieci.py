"""Pamięć dyskowa w oknie: dolina, mechanika, pole w podkrokach – pierwsze liczenie i drugie z dysku (czas, brak wyjątków);
okno „Obliczenia” buduje się i widzi wpisy."""
import os, sys, time, traceback
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
bledy = []
sys.excepthook = lambda t, v, tb: bledy.append(''.join(traceback.format_exception(t, v, tb)))
import laser_app  # noqa
from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
from laser_app.main_window import MainWindow
o = MainWindow(); o.resize(1600, 950); o.show(); z = o.zakladka_laserow
def p(n=10):
    for _ in range(n): a.processEvents(); time.sleep(0.02)
def czekaj(war, s=120):
    t0 = time.time()
    while not war() and time.time() - t0 < s: p(3)
    return time.time() - t0
def start():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')); p()
    z.doki['12'].raise_(); z.ch_dol.setChecked(True)
    for powt in (1, 2):
        z._dol = None; z._dol_zlecony = None; z.ustaw_czas(300.0); z._dol_zlec()
        print(f'dolina, raz {powt}: {czekaj(lambda: z._dol is not None):.2f} s', flush=True)
    z.doki['13'].raise_()
    for powt in (1, 2):
        z._mech = None; z._mech_zlecony = None; z.ustaw_czas(300.0); z._mech_zlec()
        print(f'mechanika, raz {powt}: {czekaj(lambda: z._mech is not None):.2f} s', flush=True)
    z.doki['11'].raise_(); z.ch_og.setChecked(True); z.ch_pod.setChecked(True)
    czekaj(lambda: z.d_pod is not None, 120)
    z.cb_og_czas.setCurrentIndex(1)
    for powt in (1, 2):
        z._pp_cache.clear(); z._pp_klucz = None; z._pp_pole = None
        z.ustaw_podkrok(5.5)
        print(f'pole w podkrokach, raz {powt}: {czekaj(lambda: z._pp_pole is not None):.2f} s', flush=True)
        p(10)
    z.ustaw_podkrok(6.2); p(20)
    from laser_app.ui.computations_dialog import ComputationsDialog
    d = ComputationsDialog(z, o); d.show(); p(10)
    print('okno obliczeń:', d.lb_hist.text().split(chr(10))[0], '|', d.lb_stan.text()[:200], flush=True)
    d._zaznacz_czastki(); print('zaznaczone:', [(w['rodzaj'], w['tok']) for w in d._zaznaczone()], flush=True)
    d.close()
    print('BŁĘDY', len(bledy)); [print(b[-1200:]) for b in bledy[:3]]
    a.quit()
QTimer.singleShot(1500, start); a.exec()
