"""Okienko 14 z mapą pułapek (liczenie całej historii w tle), lista gotowych pułapek, podgląd etykiety: zrzuty widżetów."""
import os, sys, time
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app
o = app.Okno(); o.resize(1600, 950); o.show(); z = o.zakladka_laserow
def p(n=10):
    for _ in range(n): a.processEvents(); time.sleep(0.02)
def start():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')); p()
    z._og_lista_gotowych(); print('gotowe pułapki:', [z.cb_og_gotowe.itemText(i) for i in range(z.cb_og_gotowe.count())][:6], flush=True)
    z.doki['14'].raise_(); p(5)
    z.ch_mp.setChecked(True); p(5)
    t0 = time.time()
    while z._mp is None and time.time() - t0 < 240:
        p(25)
    print('mapa po', round(time.time() - t0), 's;', z.lb_mp.text(), flush=True)
    for t in (100.0, 100.5, 700.0):
        z.ustaw_czas(t); p(8)
        print(t, z.lb14.text()[:220], flush=True)
    z.doki['14'].widget().grab().save(os.path.join(OUT, 'mapa14.png'))
    z.cb_mp_kolor.setCurrentIndex(0); z.ch_mp_skok.setChecked(True); z.ch_mp_sfera.setChecked(True); z.ustaw_czas(700.0); p(10)
    z.doki['s'].raise_(); p(5); z.ustaw_czas(700.2); p(10)
    z.doki['s'].widget().grab().save(os.path.join(OUT, 'mapa_sfera.png'))
    z.cb_etyk.setCurrentIndex(1); p(5); print('etykieta:', z.lb_etyk.text(), '|', z.lb_etyk_tryby.text(), flush=True)
    z.podglad_etyk.resize(480, 320); z.podglad_etyk.grab().save(os.path.join(OUT, 'podglad_etyk.png'))
    if z.cb_og_gotowe.count():
        z._og_wybierz_gotowa(0); p(10); print('wybrano gotową:', z.lb_og.text()[:160], flush=True)
    a.quit()
QTimer.singleShot(1500, start); a.exec()
