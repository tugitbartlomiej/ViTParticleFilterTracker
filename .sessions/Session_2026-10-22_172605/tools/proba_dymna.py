"""Próba dymna po refaktorze: okno, historia, okienka 1–14, sfera, rzuty, pułapka i mapa z pliku, podkroki, dolina, mechanika,
etykiety, popupy, układy. Każdy wyjątek (także w slotach Qt) jest liczony i wypisywany."""
import os, sys, time, traceback
APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP); os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
bledy = []
def hak(t, v, tb):
    bledy.append(''.join(traceback.format_exception(t, v, tb)))
sys.excepthook = hak
import laser_app  # noqa
from PyQt6.QtCore import QCoreApplication, Qt, QTimer, QPoint
from PyQt6.QtWidgets import QApplication
QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
from laser_app.main_window import MainWindow
from laser_app.ui.settings_panel import SettingsPopup
o = MainWindow(); o.resize(1600, 950); o.show(); z = o.zakladka_laserow
def p(n=10):
    for _ in range(n): a.processEvents(); time.sleep(0.02)
def czekaj(war, s=120):
    t0 = time.time()
    while not war() and time.time() - t0 < s: p(5)
    return war()
def krok(opis, fn):
    try:
        fn(); p(8); print('ok ', opis, flush=True)
    except Exception:
        bledy.append(opis + '\n' + traceback.format_exc()); print('BŁĄD', opis, flush=True)
def start():
    krok('historia', lambda: z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', '2cca2743e1c74b14')))
    for t in (0.0, 120.5, 400.25, 805.0):
        krok(f'ustaw_czas {t}', lambda t=t: z.ustaw_czas(t))
    for k in list(z.doki):
        krok(f'okienko {k}', lambda k=k: (z.doki[k].show(), z.doki[k].raise_(), z.ustaw_czas(300.3)))
    krok('pułapka z pliku', lambda: (z._og_lista_gotowych(), z.cb_og_gotowe.count() and z._og_wybierz_gotowa(0)))
    krok('pułapka czas', lambda: (z.doki['11'].raise_(), z.ch_og_obrys.setChecked(True), z.ustaw_czas(301.5)))
    krok('mapa z pliku', lambda: (z.doki['14'].raise_(), z.ch_mp.setChecked(True), z.ch_mp_skok.setChecked(True), z.ustaw_czas(302.2)))
    for i in range(z.cb_mp_kolor.count()):
        krok(f'mapa kolor {i}', lambda i=i: z.cb_mp_kolor.setCurrentIndex(i))
    for i in range(z.cb_etyk.count()):
        krok(f'etykieta {i}', lambda i=i: z.cb_etyk.setCurrentIndex(i))
    krok('dolina', lambda: (z.doki['12'].raise_(), z.ch_dol.setChecked(True), z._dol_zlec()))
    czekaj(lambda: getattr(z, '_dol', None) is not None, 90)
    krok('dolina czas', lambda: z.ustaw_czas(303.0))
    krok('mechanika', lambda: (z.doki['13'].raise_(), z._mech_zlec()))
    czekaj(lambda: getattr(z, '_mech', None) is not None, 90)
    krok('mechanika czas', lambda: z.ustaw_czas(303.6))
    krok('podkroki wł.', lambda: z.ch_pod.setChecked(True))
    czekaj(lambda: z.d_pod is not None, 120)
    for s_ in (0.0, 3.5, 7.2, 11.9):
        krok(f'podkrok {s_}', lambda s_=s_: z.ustaw_podkrok(s_))
    krok('pułapka w podkrokach', lambda: (z.doki['11'].raise_(), z.cb_og_czas.setCurrentIndex(1), z.ustaw_podkrok(5.5)))
    p(60)
    krok('podkroki wył.', lambda: (z.cb_og_czas.setCurrentIndex(0), z.ch_pod.setChecked(False)))
    for k in ('s', '11', '14', 'R1', 'R12', '6'):
        krok(f'popup {k}', lambda k=k: z.menu_okienka(k, QPoint(400, 300)))
    krok('zwiń/rozwiń', lambda: [(pp.zwin_wszystko(True), pp.zwin_wszystko(False)) for pp in list(SettingsPopup.otwarte)])
    krok('zamknij popupy', lambda: [pp.close() for pp in list(SettingsPopup.otwarte)])
    for u in ('pulapka', 'rzuty', 'sfera'):
        krok(f'układ {u}', lambda u=u: z._uklad_gotowy(u))
    krok('odtwarzanie', lambda: (z.bt_play.setChecked(True), p(60), z.bt_play.setChecked(False)))
    krok("zestaw: aktywne", lambda: z._zestaw_aktywne(0))
    krok('lista zdjęć testowych', lambda: print('   zdjęć testowych:', len([i for i in range(z.cb_zdjecie.count())])))
    krok('etykieta GT zdjęcia', lambda: print('   ', __import__('laser_app.compute.tool_labels', fromlist=['x']).etykieta_dla_zdjecia('sample_images/TR_0001_S1_P03_klatka_000870.jpg')))
    print('BŁĘDY:', len(bledy), flush=True)
    for b in bledy[:8]:
        print(b[-1500:], flush=True)
    a.quit()
QTimer.singleShot(1500, start); a.exec()
