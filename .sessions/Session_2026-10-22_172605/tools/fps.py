"""FPS odtwarzania osi czasu (▶) w LaserLatentSpaceApp: klatki sfery na sekundę i czas ustaw_czas, w kilku układach."""
import os
import sys
import time

APP = r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\LaserLatentSpaceApp'
sys.path.insert(0, APP); os.chdir(APP)
os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'
import numpy as np  # noqa: E402
from PyQt6.QtCore import QCoreApplication, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
a = QApplication(sys.argv)
import app  # noqa: E402

o = app.Okno(); o.show()
z = o.zakladka_laserow
licznik = {'s': 0}
stara = z.vs.paintGL


def paintGL():
    licznik['s'] += 1; stara()


z.vs.paintGL = paintGL
cz = []
stare_uc = z.ustaw_czas


def ustaw_czas(t):
    c0 = time.perf_counter(); stare_uc(t); cz.append(time.perf_counter() - c0)


z.ustaw_czas = ustaw_czas
etapy = []
if os.environ.get('ROZMIARY') == '1':                                    # kto zmienia rozmiar widoków w czasie odtwarzania
    import zakladka_laserow as ZL
    stary_re = ZL.Widok2D.resizeEvent
    zmiany = []

    def resizeEvent(self, ev):
        if z.bt_play.isChecked() and len(zmiany) < 12:
            k = next((k for k, d in z.doki.items() if d.isAncestorOf(self)), '?')
            if k == '1':
                print('   szerokości doków:', {kk: (z.doki[kk].width(), z.doki[kk].widget().minimumSizeHint().width())
                                               for kk in ('1', '2', '3', '4', '6', 's', '11')},
                      'panel', z.dok_panel.width(), 'min/hint', z.dok_panel.minimumSizeHint().width(), z.dok_panel.sizeHint().width(),
                      z.dok_panel.minimumWidth(), z.panel.minimumWidth(), z.panel.strony.minimumSizeHint().width(),
                      z.panel.strony.currentWidget().minimumSizeHint().width(), 'podkroki', z.dok_podkroki.width(), z.dok_podkroki.height(),
                      'okna', z.okna.width(), 'zakładka', z.width(), 'okno', o.width(), flush=True)
            zmiany.append(f'okienko {k}: {ev.oldSize().width()}x{ev.oldSize().height()} -> {ev.size().width()}x{ev.size().height()}')
            print(zmiany[-1], flush=True)
        stary_re(self, ev)
    ZL.Widok2D.resizeEvent = resizeEvent
    from collections import Counter
    from PyQt6.QtCore import QEvent, QObject
    zdarzenia = Counter()

    class Filtr(QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Type.LayoutRequest and z.bt_play.isChecked():
                rodzic = obj.parent() if hasattr(obj, 'parent') else None
                zdarzenia[f'{type(obj).__name__}:{obj.objectName()} <- {type(rodzic).__name__ if rodzic else "-"}'] += 1
            return False
    FILTR = Filtr(); a.installEventFilter(FILTR)
    import atexit
    atexit.register(lambda: print('LayoutRequest:', zdarzenia.most_common(15), flush=True))


def etap(nazwa, przygotuj, sek=5.0):
    def start():
        przygotuj(); licznik['s'] = 0; cz.clear(); z.cb_tempo.setCurrentIndex(1); z.sp_tempo.setValue(20.0)
        z.ustaw_czas(z.h.K * 0.2); z.bt_play.setChecked(True); t0 = time.perf_counter()

        def stop():
            z.bt_play.setChecked(False)
            print(f'[{nazwa}] sfera {licznik["s"] / (time.perf_counter() - t0):.0f} kl./s, ustaw_czas mediana '
                  f'{np.median(cz) * 1e3:.1f} ms, p90 {np.percentile(cz, 90) * 1e3:.1f} ms ({len(cz)} wywołań)', flush=True)
            nastepny()
        QTimer.singleShot(int(sek * 1000), stop)
    etapy.append(start)


def nastepny():
    if etapy:
        QTimer.singleShot(300, etapy.pop(0))
    else:
        a.quit()


def wczytaj():
    z._gotowe(os.path.join(APP, 'cache_uczenia', 'lasery', os.environ.get('HIST', '2cca2743e1c74b14')))
    t0 = time.perf_counter()
    while not z.es.duzy_gotowy('ZH') and time.perf_counter() - t0 < 60:
        z.vs.update(); a.processEvents(); time.sleep(0.005)
    print('ZH gotowe', z.es.duzy_gotowy('ZH'), flush=True)
    nastepny()


import cProfile, pstats, io  # noqa: E402,E401
PR = cProfile.Profile()


def z_profilem():
    PR.enable()


def wypisz():
    PR.disable()
    s = io.StringIO(); pstats.Stats(PR, stream=s).sort_stats('tottime').print_stats(25); print(s.getvalue()[:6000], flush=True)


if os.environ.get('PROFIL') == '1':
    uk = os.environ.get('UKLAD', '')
    etap(f'układ {uk or "domyślny"} (profil)', lambda: (z._uklad_gotowy(uk) if uk else None, z_profilem()))
    etapy.append(lambda: (wypisz(), nastepny()))
else:
    etap('układ domyślny', lambda: None)
    etap('sfera na całe okno', lambda: z._uklad_gotowy('sfera'))
    etap('rzuty R1–R7 obok sfery', lambda: z._uklad_gotowy('rzuty'))
    etap('domyślny + pułapka 11', lambda: (z._przywroc_uklad(), z.ch_og.setChecked(True)), 8.0)
    etap('sfera + pułapka obok', lambda: z._uklad_gotowy('pulapka'))
    etap('pułapka + obrys 3D', lambda: z.ch_og_obrys.setChecked(True), 6.0)
    etap('pułapka + obrys 3D, bez linii wiązek', lambda: z.ch_og_wiazki.setChecked(False), 6.0)
    etapy.append(lambda: (print('obrys policzony:', z.eog.n_of_tr > 0, 'info:', (z.eog.info_wiazek or {}).get('obrys') is not None,
                                flush=True), nastepny()))
QTimer.singleShot(1500, wczytaj)
a.exec()
