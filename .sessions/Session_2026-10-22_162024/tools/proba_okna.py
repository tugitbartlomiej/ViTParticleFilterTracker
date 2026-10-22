"""Próba okna LaserLatentSpaceApp: buduje okno, czeka, opcjonalnie wczytuje historię z cache, robi zrzuty i zamyka."""
import os
import sys
import time
import traceback

APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP)
os.chdir(APP)
os.environ['EDYTOR_BEZ_USTAWIEN'] = os.environ.get('EDYTOR_BEZ_USTAWIEN', '1')

from PyQt6.QtCore import QCoreApplication, QPoint, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app  # noqa: E402
from styl import STYL  # noqa: E402

a.setStyleSheet(STYL)
t0 = time.perf_counter()
o = app.Okno()
o.show()
print(f'okno w {time.perf_counter() - t0:.1f} s', flush=True)
z = o.zakladka_laserow
kroki = []


def zrzut(nazwa):
    QApplication.primaryScreen().grabWindow(0).save(os.path.join(OUT, nazwa))
    print('zrzut', nazwa, flush=True)


def krok():
    if not kroki:
        a.quit(); return
    fn = kroki.pop(0)
    try:
        fn()
    except Exception:
        traceback.print_exc()
    QTimer.singleShot(int(os.environ.get('ODSTEP', '2500')), krok)


kroki.append(lambda: zrzut('01_start.png'))
if os.environ.get('WCZYTAJ') == '1':
    kroki.append(lambda: z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', os.environ.get('HIST', '2cca2743e1c74b14'))))
    kroki += [lambda: None] * int(os.environ.get('CZEKAJ', '6'))
    kroki.append(lambda: print('historia:', z.h is not None and z.h.katalog, z.lb_stan.text(), flush=True))
    kroki.append(lambda: zrzut('02_historia.png'))
    kroki.append(lambda: print('etykieta:', z.lb_etyk.text(), flush=True))
    kroki.append(lambda: (z.ch_og_obrys.setChecked(True), z.ch_og.setChecked(True), z.ustaw_czas(z.h.K * 0.6)))
    kroki += [lambda: None] * int(os.environ.get('CZEKAJ_OG', '8'))
    kroki.append(lambda: print('pułapka:', z.lb_og.text()[:300], '| lb11:', z.lb11.text()[:300], flush=True))
    kroki.append(lambda: zrzut('06_pulapka_obrys.png'))
    kroki.append(lambda: z.cb_etyk.setCurrentIndex(1))
    kroki.append(lambda: print('pokrycie:', z.lb_etyk.text(), flush=True))
    kroki.append(lambda: z.cb_etyk.setCurrentIndex(0))
    kroki.append(lambda: z.cb_kol_sciezki.setCurrentIndex(5))
    kroki.append(lambda: z.cb_kolor_cz.setCurrentIndex(3))
    kroki.append(lambda: z._zaznacz_obszar(2))
    kroki.append(lambda: print('źrenica:', z.ed_tok.text()[:80], flush=True))
kroki.append(lambda: z.menu_okienka('s', o.mapToGlobal(QPoint(700, 300))))
kroki.append(lambda: zrzut('03_popup_sfera.png'))
kroki.append(lambda: (QApplication.activePopupWidget() and QApplication.activePopupWidget().close()))
kroki.append(lambda: z.menu_okienka('11', o.mapToGlobal(QPoint(900, 300))))
kroki.append(lambda: (QApplication.activePopupWidget().przypnij() if QApplication.activePopupWidget() else print('brak popupu')))
kroki.append(lambda: zrzut('04_przypiety_11.png'))
kroki.append(lambda: [p.close() for p in list(__import__('panel_ustawien').PopupUstawien.otwarte)])
kroki.append(lambda: z.menu_okienka('R1', o.mapToGlobal(QPoint(900, 300))))
kroki.append(lambda: (QApplication.activePopupWidget() and QApplication.activePopupWidget().close()))
for n in ('pulapka', 'rzuty', 'sfera'):
    kroki.append(lambda n=n: z._uklad_gotowy(n))
    kroki.append(lambda n=n: zrzut(f'05_uklad_{n}.png'))
kroki.append(lambda: z._przywroc_uklad())
kroki.append(lambda: print('ustawienia:', len(__import__('ustawienia').stan(o)), flush=True))
QTimer.singleShot(3000, krok)
a.exec()
print('koniec', flush=True)

