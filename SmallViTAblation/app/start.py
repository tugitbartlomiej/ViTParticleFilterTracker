"""Start edytora latentu z okienkiem ładowania.

Pierwsze uruchomienie importuje torch; na zimnym dysku trwa to kilkadziesiąt sekund i bez tego okienka przez ten
czas nic nie widać. Aplikacja startuje pusta: nie wczytuje sekwencji, historii uczenia ani modelu SD (patrz `app.py`).
Każdy błąd startu trafia do okna z komunikatem i do `start.log`.
Uruchomienie: python start.py (albo uruchom.bat).
"""
import sys
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox

LOG = Path(__file__).with_name('start.log')
# Tylko to, co potrzebuje samo okno. Ciężkie biblioteki zakładek (timm i torchvision w `detekcja`, OpenGL, wykresy)
# ładują się dopiero przy pierwszym wejściu na zakładkę, która ich używa (`Okno.zbuduj_zakladke`).
KROKI = [('torch', 'biblioteki PyTorch'), ('silnik', 'silnika SD'), ('app', 'okna aplikacji')]


def main():
    if sys.stdout is None or sys.stderr is None:          # pythonw (exe) nie ma konsoli: print i paski tqdm do pliku
        wyjscie = open(LOG.with_name('wyjscie.log'), 'w', encoding='utf-8', buffering=1)
        sys.stdout = sys.stdout or wyjscie
        sys.stderr = sys.stderr or wyjscie
    # wspólne konteksty OpenGL: widok 3D przeniesiony do osobnego okna (Odepnij) dostaje nowy kontekst,
    # a shadery i bufory pyqtgraph muszą w nim dalej istnieć (bez tego GL_INVALID_VALUE przy glUseProgram)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    aplikacja = QApplication(sys.argv)
    okienko = QLabel()
    okienko.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
    okienko.setStyleSheet('QLabel { background: #1e1f24; color: #e8e8e8; font-size: 15px; padding: 26px; '
                          'border: 1px solid #555; }')
    okienko.setMinimumWidth(460)
    t0 = time.perf_counter()
    dziennik = []

    def pokaz(tekst):
        dziennik.append(f'{time.perf_counter() - t0:6.1f} s  {tekst}')
        okienko.setText('Edytor latentu SD\n\n' + tekst + f'\n\n{time.perf_counter() - t0:.0f} s'
                        '\nPierwszy start po włączeniu komputera trwa dłużej (zimny dysk).')
        okienko.adjustSize()
        aplikacja.processEvents()

    okienko.show()
    try:
        import importlib
        for modul, opis in KROKI:
            pokaz(f'Ładowanie: {opis}...')
            importlib.import_module(modul)
        import app
        from styl import STYL
        aplikacja.setStyleSheet(STYL)
        pokaz('Budowanie okna...')
        okno = app.Okno()
        okno.show()
        pokaz('Gotowe.')
        okienko.close()
    except Exception:
        tekst = traceback.format_exc()
        dziennik.append(tekst)
        LOG.write_text('\n'.join(dziennik), encoding='utf-8')
        okienko.close()
        QMessageBox.critical(None, 'Błąd startu edytora', f'{tekst}\n\nZapisano w {LOG}')
        sys.exit(1)
    LOG.write_text('\n'.join(dziennik), encoding='utf-8')
    sys.exit(aplikacja.exec())


if __name__ == '__main__':
    main()
