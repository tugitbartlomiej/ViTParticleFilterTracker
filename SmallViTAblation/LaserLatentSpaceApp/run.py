"""Start aplikacji „Lasery i koloidy” z okienkiem ładowania.

Okno startuje bez liczenia: jeżeli wybrana w ustawieniach historia jest już w cache (cache_uczenia/lasery), wczytuje się
zaraz po pokazaniu okna. Każdy błąd startu trafia do okna z komunikatem i do `run.log`.
Uruchomienie: python run.py (albo run.bat, albo „Lasery i koloidy.exe”); bez okienka ładowania: python -m laser_app.
"""
import sys
import time
import traceback

import laser_app  # noqa: F401  (najpierw: wyłącza sprawdzanie błędów PyOpenGL przed pierwszym importem OpenGL.GL)
from laser_app.paths import LOG_FILE as LOG, OUTPUT_LOG_FILE
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox

KROKI = [('numpy', 'numpy'), ('pyqtgraph.opengl', 'OpenGL (pyqtgraph)'), ('laser_app.gpu.sphere_gl', 'shadery GPU'),
         ('laser_app.lasers_tab.tab', 'okienka')]


def main():
    if sys.stdout is None or sys.stderr is None:          # pythonw (exe) nie ma konsoli: print i paski tqdm do pliku
        wyjscie = open(OUTPUT_LOG_FILE, 'w', encoding='utf-8', buffering=1)
        sys.stdout = sys.stdout or wyjscie
        sys.stderr = sys.stderr or wyjscie
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
        okienko.setText('Lasery i koloidy\n\n' + tekst + f'\n\n{time.perf_counter() - t0:.0f} s'
                        '\nPierwszy start po włączeniu komputera trwa dłużej (zimny dysk).')
        okienko.adjustSize()
        aplikacja.processEvents()

    okienko.show()
    try:
        import importlib
        for modul, opis in KROKI:
            pokaz(f'Ładowanie: {opis}...')
            importlib.import_module(modul)
        from laser_app import main_window
        from laser_app.ui.style import STYL
        aplikacja.setStyleSheet(STYL)
        pokaz('Budowanie okna...')
        okno = main_window.MainWindow()
        okno.show()
        pokaz('Gotowe.')
        okienko.close()
    except Exception:
        tekst = traceback.format_exc()
        dziennik.append(tekst)
        LOG.write_text('\n'.join(dziennik), encoding='utf-8')
        okienko.close()
        QMessageBox.critical(None, 'Błąd startu: Lasery i koloidy', f'{tekst}\n\nZapisano w {LOG}')
        sys.exit(1)
    LOG.write_text('\n'.join(dziennik), encoding='utf-8')
    sys.exit(aplikacja.exec())


if __name__ == '__main__':
    main()
