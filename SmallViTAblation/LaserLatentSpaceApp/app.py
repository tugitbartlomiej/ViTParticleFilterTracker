"""Lasery i koloidy: osobna aplikacja (PyQt6) wydzielona z zakładki „Lasery i koloidy” edytora latentu (`../app`).

Oś czasu uczenia jednego zdjęcia przez checkpointy małego ViT: sfera reflektorów, pułapka optyczna, studnie, drgania, atencja
6 bloków, rzuty z 192 wymiarów, IoU i pewność (filmy 2, 12, 16). Całe okno to `zakladka_laserow.ZakladkaLaserow`:
  - u góry pasek osi czasu (odtwarzanie, A–B, tempo),
  - okienka-doki z widokami (każde można przesunąć, połączyć w karty, odpiąć jako osobne okno albo rozwinąć na ekran),
  - „Parametry”: zwarta lista sekcji i jedna strona naraz, też odpinana (F2 chowa i pokazuje),
  - „Podkroki sieci”: osobne okienko z suwakiem przejścia przez 6 bloków,
  - prawy klik w dowolne okienko: jego ustawienia przy kursorze (📌 przypina je jako małe okno).
Menu „Ustawienia” zapisuje stan kontrolek i układ okienek do `ustawienia.json` (wczytywany przy starcie).
Uruchomienie: python start.py (okienko ładowania), „Lasery i koloidy.exe” albo python app.py.
"""
import os
import sys
import traceback

import OpenGL

# PyOpenGL domyślnie woła glGetError po KAŻDYM wywołaniu GL: każde czeka na kartę (synchronizacja CPU–GPU). W odtwarzaniu to było
# ~50 tys. wywołań i połowa czasu klatki. Błędy shaderów i tak zgłasza kompilacja; LASERY_GL_DEBUG=1 włącza sprawdzanie z powrotem.
OpenGL.ERROR_CHECKING = os.environ.get('LASERY_GL_DEBUG') == '1'

from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402
from PyQt6.QtGui import QAction, QKeySequence  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox  # noqa: E402

from styl import STYL  # noqa: E402

SKROTY = """<b>Oś czasu</b>: spacja odtwarza / zatrzymuje, strzałki ← → krok, Home początek.<br>
<b>Widoki 3D</b> (okienko pod kursorem): LPM obrót, Ctrl+LPM albo ŚPM przesuwanie, kółko zbliża do kursora, PPM + przeciągnięcie zbliża,
W/S/A/D obrót, Q/E oddal/zbliż, I/J/K/L przesuwanie, Home albo 0 kadr startowy, Shift szybciej, Ctrl wolniej.<br>
<b>Prawy klik</b> (bez przeciągania) w okienko: jego ustawienia; 📌 przypina je jako osobne okno, ⇤ pokazuje je w panelu.<br>
<b>F2</b> panel parametrów, <b>F3</b> podkroki sieci, <b>F11</b> pełny ekran okienka pod kursorem (Esc wraca).<br>
<b>Ctrl+Shift+1..3</b> zapamiętuje układ okienek, <b>Ctrl+1..3</b> go przywraca."""


class Okno(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Lasery i koloidy — oś czasu uczenia małego ViT')
        self.resize(1680, 1000)
        from zakladka_laserow import ZakladkaLaserow
        self.zakladka_laserow = z = ZakladkaLaserow(self)       # nazwa atrybutu = klucze ustawień jak w dawnej zakładce
        self.setCentralWidget(z)
        from wygoda import ulatw
        ulatw(self); ulatw(z.panel)                             # kółko tylko w klikniętym polu, liczby zatwierdzane Enterem
        from okna import skrot_f11
        skrot_f11(self)
        self._menu()
        self.statusBar().showMessage('Wybierz historię i zdjęcie w „Parametry → Dane”, potem „Policz / wczytaj”. '
                                     'Prawy klik w okienko otwiera jego ustawienia.', 15000)
        self._ustawienia_startowe()

    # ------------------------------------------------------------------ menu
    def _menu(self):
        z = self.zakladka_laserow
        mb = self.menuBar()
        import ustawienia as U

        def akcja(menu, nazwa, fn, skrot=None, opis=''):
            a = QAction(nazwa, self)
            if skrot:
                a.setShortcut(QKeySequence(skrot))
            if opis:
                a.setToolTip(opis); a.setStatusTip(opis)
            a.triggered.connect(fn); menu.addAction(a)
            return a

        p = mb.addMenu('Plik'); p.setToolTipsVisible(True)
        akcja(p, 'Policz / wczytaj historię', lambda: (self._pokaz_dane(), z._licz()), 'Ctrl+O',
              'Historia i zdjęcie wybrane w „Parametry → Dane”: z cache od razu, inaczej liczenie na GPU.')
        akcja(p, 'Przerwij liczenie', z._przerwij, 'Ctrl+.')
        akcja(p, 'Inny katalog wag…', lambda: (self._pokaz_dane(), z._inny_katalog()))
        akcja(p, 'Zdjęcie z pliku…', lambda: (self._pokaz_dane(), z._plik_zdjecia()))
        p.addSeparator()
        akcja(p, 'Zapisz ustawienia i układ okienek', lambda: self._zapisz_ustawienia(U.PLIK), 'Ctrl+S',
              f'Stan list, pól, opcji, układ okienek i okna do {U.PLIK.name}; tak samo wystartuje następne uruchomienie.')
        akcja(p, 'Zapisz ustawienia jako…', lambda: self._zapisz_ustawienia(None), 'Ctrl+Shift+S', 'Zapis do wybranego pliku JSON.')
        akcja(p, 'Wczytaj ustawienia…', self._wczytaj_ustawienia, None, 'Listy, pola, opcje i układ z wybranego pliku JSON.')
        p.addSeparator()
        akcja(p, 'Zapisz obraz okienka pod kursorem (PNG)…', self._zapisz_obraz, 'Ctrl+E',
              'Okienko pod kursorem myszy (albo całe okno) do pliku PNG.')
        p.addSeparator()
        akcja(p, 'Zamknij', self.close, 'Ctrl+Q', 'Zamyka aplikację (z zapisem ustawień, gdy włączony).')

        m = mb.addMenu('Ustawienia'); m.setToolTipsVisible(True)
        self.akcja_autozapis = QAction('Zapisuj ustawienia i układ przy zamknięciu', self); self.akcja_autozapis.setCheckable(True)
        self.akcja_autozapis.setChecked(True); m.addAction(self.akcja_autozapis)
        akcja(m, 'Przywróć domyślne', self._przywroc_ustawienia, None,
              'Wartości z chwili uruchomienia programu bez pliku ustawień i domyślny układ okienek.')
        m.addSeparator()
        a = QAction('Stereo 3D (SBS) we wszystkich widokach', self); a.setCheckable(True)
        a.toggled.connect(z.ch_stereo.setChecked); z.ch_stereo.toggled.connect(a.setChecked); m.addAction(a)

        mb.addMenu(z.menu_okienek)
        w = mb.addMenu('Widok')
        for dok, skrot in ((z.dok_panel, 'F2'), (z.dok_podkroki, 'F3')):
            a = dok.toggleViewAction(); a.setShortcut(QKeySequence(skrot)); w.addAction(a)
        a = QAction('Panel parametrów jako osobne okno', self)
        a.triggered.connect(lambda: (z.dok_panel.show(), z.dok_panel.setFloating(not z.dok_panel.isFloating())))
        w.addAction(a)
        w.addSeparator()
        a = QAction('Układ domyślny', self); a.triggered.connect(z._przywroc_uklad); w.addAction(a)
        p = mb.addMenu('Pomoc')
        a = QAction('Skróty i obsługa myszy', self)
        a.triggered.connect(lambda: QMessageBox.information(self, 'Skróty', SKROTY)); p.addAction(a)

    # ------------------------------------------------------------------ ustawienia (plik konfiguracyjny)
    def _ustawienia_startowe(self):
        import ustawienia as U
        self._domyslne = U.stan(self)
        if U.PLIK.exists() and not _bez_ustawien():
            try:
                ok, pom = U.wczytaj(self)
                self.statusBar().showMessage(f'Wczytano ustawienia z {U.PLIK.name}: {ok} kontrolek'
                                             + (f', {pom} pominiętych (brak pozycji)' if pom else ''), 8000)
            except Exception as e:                                     # noqa: BLE001
                self.statusBar().showMessage(f'Nie wczytano {U.PLIK.name}: {e}', 8000)

    def _zapisz_ustawienia(self, plik):
        import ustawienia as U
        if plik is None:
            plik, _ = QFileDialog.getSaveFileName(self, 'Zapisz ustawienia', str(U.PLIK), 'Ustawienia (*.json)')
            if not plik:
                return
        n = U.zapisz(self, plik)
        self.statusBar().showMessage(f'Zapisano {n} ustawień do {plik}', 8000)

    def _pokaz_dane(self):
        z = self.zakladka_laserow
        z.dok_panel.show(); z.dok_panel.raise_(); z.panel.pokaz('dane')

    def _zapisz_obraz(self):
        """Okienko-dok pod kursorem (albo całe okno) do PNG; widoki OpenGL wchodzą w obraz (QWidget.grab)."""
        from datetime import datetime
        from pathlib import Path
        from PyQt6.QtGui import QCursor
        from PyQt6.QtWidgets import QApplication, QDockWidget
        w = QApplication.widgetAt(QCursor.pos())
        while w is not None and not isinstance(w, QDockWidget):
            w = w.parentWidget()
        cel = w.widget() if w is not None else self
        nazwa = (w.objectName().replace('dok_lasery_', '') if w is not None else 'okno') + datetime.now().strftime('_%Y%m%d_%H%M%S.png')
        kat = Path(__file__).resolve().parent / 'zrzuty'
        kat.mkdir(exist_ok=True)
        plik, _ = QFileDialog.getSaveFileName(self, 'Zapisz obraz', str(kat / nazwa), 'Obraz PNG (*.png)')
        if plik:
            ok = cel.grab().save(plik)
            self.statusBar().showMessage(f'Zapisano {plik}' if ok else f'Nie zapisano {plik}', 8000)

    def _wczytaj_ustawienia(self):
        import ustawienia as U
        plik, _ = QFileDialog.getOpenFileName(self, 'Wczytaj ustawienia', str(U.PLIK), 'Ustawienia (*.json)')
        if plik:
            ok, pom = U.wczytaj(self, plik)
            self.statusBar().showMessage(f'Wczytano {ok} ustawień z {plik}' + (f', {pom} pominiętych' if pom else ''), 8000)

    def _przywroc_ustawienia(self):
        import ustawienia as U
        ok, _ = U.ustaw(self, {k: v for k, v in self._domyslne.items() if not k.endswith('.uklad_okienek')})
        self.zakladka_laserow._przywroc_uklad()
        self.statusBar().showMessage(f'Przywrócono {ok} ustawień domyślnych i układ okienek.', 8000)

    def closeEvent(self, e):
        import ustawienia as U
        from panel_ustawien import PopupUstawien
        for p in list(PopupUstawien.otwarte):                   # przypięte popupy oddają sekcje panelowi przed zapisem
            p.close()
        if self.akcja_autozapis.isChecked() and not _bez_ustawien():
            try:
                U.zapisz(self)
            except Exception:                                   # noqa: BLE001
                traceback.print_exc()
        super().closeEvent(e)


def _bez_ustawien():
    import os
    return os.environ.get('EDYTOR_BEZ_USTAWIEN') == '1'


def main():
    # wspólne konteksty OpenGL: okienko odpięte do osobnego okna dostaje nowy kontekst, a shadery i bufory muszą w nim istnieć
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    aplikacja = QApplication(sys.argv)
    aplikacja.setStyleSheet(STYL)
    okno = Okno()
    okno.show()
    sys.exit(aplikacja.exec())


if __name__ == '__main__':
    main()
