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
import sys
import traceback

from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from styl import STYL

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
        m = mb.addMenu('Ustawienia')
        m.setToolTipsVisible(True)
        import ustawienia as U
        for nazwa, opis, fn in (
                ('Zapisz ustawienia', f'Stan list, pól, opcji i układ okienek do {U.PLIK.name} (wczytywany przy starcie).',
                 lambda: self._zapisz_ustawienia(U.PLIK)),
                ('Zapisz ustawienia jako…', 'Zapis do wybranego pliku JSON.', lambda: self._zapisz_ustawienia(None)),
                ('Wczytaj ustawienia…', 'Ustawia listy, pola, opcje i układ z wybranego pliku JSON.', self._wczytaj_ustawienia),
                ('Przywróć domyślne', 'Wartości z chwili uruchomienia programu bez pliku ustawień.', self._przywroc_ustawienia)):
            a = QAction(nazwa, self); a.setToolTip(opis); a.setStatusTip(opis); a.triggered.connect(fn); m.addAction(a)
        m.addSeparator()
        self.akcja_autozapis = QAction('Zapisuj przy zamknięciu', self); self.akcja_autozapis.setCheckable(True)
        self.akcja_autozapis.setChecked(True); m.addAction(self.akcja_autozapis)
        m.addSeparator()
        a = QAction('Zakończ', self); a.setShortcut(QKeySequence('Ctrl+Q')); a.triggered.connect(self.close); m.addAction(a)

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
