"""Wspólne narzędzia testów GUI (test_gui.py): jedna QApplication na moduł, budowa okna bez autowczytywania, czekanie na zdarzenia,
zbieranie wyjątków ze slotów Qt (PyQt6 inaczej przerywa proces) i sprzątanie wątków po oknie.

Kolejność importów ma znaczenie: `conftest` (zmienne środowiska i AA_ShareOpenGLContexts przed QApplication), potem `app`
(ustawia OpenGL.ERROR_CHECKING, zanim cokolwiek zaimportuje OpenGL)."""
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path

KAT = Path(__file__).resolve().parents[1]
if str(KAT) not in sys.path:
    sys.path.insert(0, str(KAT))

os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'          # testy nigdy nie czytają ani nie piszą prawdziwego settings.json
os.environ['EDYTOR_BEZ_AUTOSTARTU'] = '1'
import conftest  # noqa: E402,F401  (AA_ShareOpenGLContexts przed pierwszą QApplication)
from laser_app import main_window  # noqa: E402  (OpenGL.ERROR_CHECKING przed pierwszym importem OpenGL)

from PyQt6.QtCore import QThread, QTimer  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

HISTORIA = KAT / 'cache_uczenia' / 'lasery' / '2cca2743e1c74b14'
WYJATKI = []                                      # wyjątki zgłoszone w slotach i zdarzeniach (sys.excepthook)


def _hak(typ, wart, tb):
    WYJATKI.append(''.join(traceback.format_exception(typ, wart, tb)))


def aplikacja():
    """Jedna QApplication dla całego modułu (ze stylem aplikacji) i hak na wyjątki ze slotów."""
    from laser_app.ui.style import STYL
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(STYL)
    sys.excepthook = _hak
    return a


def przetworz(n=3):
    """Kilka obrotów pętli zdarzeń razem z odroczonymi usunięciami (deleteLater)."""
    for _ in range(n):
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, 52)     # QEvent.Type.DeferredDelete
    QApplication.processEvents()


def czekaj(warunek, timeout=10.0, odswiez=None):
    """Kręci pętlą zdarzeń, aż warunek() jest prawdą; zwraca ostatnią wartość warunku (False po przekroczeniu timeout)."""
    t0 = time.perf_counter()
    while True:
        QApplication.processEvents()
        if odswiez is not None:
            odswiez()
        w = warunek()
        if w or time.perf_counter() - t0 > timeout:
            return w
        QTest.qWait(10)


@contextlib.contextmanager
def ustawienia_startowe(plik):
    """Okno zbudowane w tym bloku wczyta `plik` jak przy prawdziwym starcie (U.PLIK + EDYTOR_BEZ_USTAWIEN=0); poza blokiem
    wszystko wraca (zamknięcie okna nic nie zapisze)."""
    from laser_app import settings
    stary_plik, stara_env = settings.PLIK, os.environ.get('EDYTOR_BEZ_USTAWIEN')
    stare_domyslne = settings.wczytaj.__defaults__, settings.zapisz.__defaults__        # plik=PLIK wiąże się w chwili importu, samo U.PLIK nie wystarcza
    settings.PLIK = Path(plik); os.environ['EDYTOR_BEZ_USTAWIEN'] = '0'
    settings.wczytaj.__defaults__ = (Path(plik),); settings.zapisz.__defaults__ = (Path(plik),)
    try:
        yield
    finally:
        settings.PLIK = stary_plik; os.environ['EDYTOR_BEZ_USTAWIEN'] = stara_env or '1'
        settings.wczytaj.__defaults__, settings.zapisz.__defaults__ = stare_domyslne


def zbuduj_okno(pokaz=True, plik_startowy=None):
    """Nowe `main_window.MainWindow` bez automatycznego wczytywania historii z cache (testy wczytują ją same przez `_gotowe`)."""
    from laser_app.lasers_tab import tab as lasers_tab
    stara = lasers_tab.LasersTab._wczytaj_z_cache
    lasers_tab.LasersTab._wczytaj_z_cache = lambda self: None
    try:
        if plik_startowy is not None:
            with ustawienia_startowe(plik_startowy):
                o = main_window.MainWindow()
        else:
            o = main_window.MainWindow()
    finally:
        lasers_tab.LasersTab._wczytaj_z_cache = stara
    if pokaz:
        pokaz_okno(o)
    return o


def pokaz_okno(o, ms=300):
    o.show(); o.raise_(); o.activateWindow()
    przetworz(10)
    QTest.qWait(ms)
    przetworz(5)


def watki(o):
    """Wszystkie QThread należące do okna (atrybuty zakładki i rzutów, dzieci Qt)."""
    z = o.zakladka_laserow
    wynik = []
    for obj in (z, getattr(z, 'rzuty', None)):
        if obj is not None:
            wynik += [v for v in vars(obj).values() if isinstance(v, QThread)]
    wynik += o.findChildren(QThread)
    return list({id(t): t for t in wynik}.values())


def zamknij_popupy():
    from laser_app.ui.settings_panel import SettingsPopup
    for p in list(SettingsPopup.otwarte):
        p.close()
    for w in QApplication.topLevelWidgets():
        if w.objectName() == 'popup_ustawien' and w.isVisible():
            w.close()
    przetworz(2)


def popup_otwarty():
    """Widoczny SettingsPopup albo None; najpierw nieprzypięty (świeżo otwarty), potem przypięty."""
    from laser_app.ui.settings_panel import SettingsPopup
    wid = [w for w in QApplication.topLevelWidgets() if w.objectName() == 'popup_ustawien' and w.isVisible()]
    swieze = [w for w in wid if w not in SettingsPopup.otwarte]
    return (swieze or wid or [None])[-1]


def zamknij_okno(o):
    """Zatrzymuje odtwarzanie, zamyka popupy i okno, czeka na wątki w tle i usuwa okno."""
    z = o.zakladka_laserow
    with contextlib.suppress(Exception):
        z.bt_play.setChecked(False); z.timer.stop()
    zamknij_popupy()
    o.close()
    for t in watki(o):
        with contextlib.suppress(Exception):
            if hasattr(t, 'stop'):
                t.stop = True
            t.quit(); t.wait(60000)
    przetworz(3)
    o.deleteLater()
    przetworz(3)


def stan_dokow(z):
    """{klucz: (otwarte, odpiete, obszar)} dla wszystkich doków zakładki (doki widoków i doki sterujące)."""
    wynik = {}
    for k, d in list(z.doki.items()) + [(f'ster_{k}', d) for k, d in z.doki_ster.items()]:
        wynik[k] = (d.isVisible(), d.isFloating(), z.okna.dockWidgetArea(d).value if not d.isFloating() else -1)
    return wynik


def napisy_zastepnika(panel):
    """Zastępniki wypożyczonych sekcji (napis z settings_panel._SectionLoan) nadal żyjące w panelu."""
    return [l for l in panel.findChildren(QLabel) if l.text().startswith('↗ Ustawienia otwarte')]
