"""Edytor latentu SD i analizy ruchu cząsteczek ViT: główne okno aplikacji (PyQt6).

Okno składa się z zakładek. Pierwsza, „Edytor SD”, jest częścią okna (`edytor_sd.EdytorSD`); pozostałe są budowane
leniwie przy pierwszym wejściu na kartę (`ZAKLADKI_LENIWE`, `Okno.zbuduj_zakladke`). Tu jest też menu „Ustawienia”
(zapis stanu kontrolek do `ustawienia.json`) i zamykanie okna.
Uruchomienie: python start.py (okienko ładowania) albo python app.py.
"""
import sys
import traceback

from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow

from edytor_sd import JEDNOSTKI, LIMITY, NA_STRONE, STRON, TRYBY, ZAKRESY, EdytorSD  # noqa: F401 (dawne app.*)
from styl import STYL
from zakladki_glowne import ZakladkiGlowne, _Odnosnik


# Zakładki budowane przy pierwszym pokazaniu: (atrybut okna, krótki tytuł karty, pełny opis (dymek karty i komunikaty),
# moduł, klasa, zakładki, do których dostaje leniwy odnośnik w konstruktorze)
ZAKLADKI_LENIWE = (
    ('zakladka_detekcji', 'Detekcja', 'Detekcja ViT i auto-szukanie latentu', 'zakladka_detekcji', 'ZakladkaDetekcji', ()),
    ('zakladka_filmu', 'Film', 'Film: sekwencja klatek, cząsteczki 3D i makrowektory', 'zakladka_filmu', 'ZakladkaFilmu', ()),
    ('zakladka_uczenia', 'Uczenie ViT', 'Uczenie ViT: ruch cząsteczek po krokach', 'zakladka_uczenia', 'ZakladkaUczenia', ()),
    ('zakladka_pchniecia', 'Pchnięcie', 'Pchnięcie obrazów: SD, DINO i ruch cząstek w kroku', 'zakladka_pchniecia',
     'ZakladkaPchniecia', ('zakladka_uczenia',)),
    ('zakladka_korekty', 'Korekta latentu', 'Korekta latentu: cel i dynamika', 'zakladka_korekty', 'ZakladkaKorekty',
     ('zakladka_uczenia',)),
    ('zakladka_ruchu', 'Ruch zbioru', 'Ruch zbioru: cząsteczki i optymalizacja', 'zakladka_ruchu', 'ZakladkaRuchu', ()),
)
OPIS_EDYTORA = 'Edytor SD: latent 600 tokenów × 768 kanałów edytowany suwakami i generowanie obrazu z SD'


class Okno(EdytorSD, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Edytor latentu SD — 600 tokenów x 768 kanałów')
        self.resize(1420, 920)
        self._inicjuj_edytor()
        self._buduj_ui()
        from wygoda import ulatw
        ulatw(self)                           # kółko tylko w klikniętym polu, liczby zatwierdzane Enterem
        self._ustawienia_startowe()
        self._po_budowie_edytora()

    def __getattr__(self, nazwa):
        """Zakładki leniwe: pierwsze odczytanie `okno.zakladka_*` buduje zakładkę (dopiero wtedy jest w `__dict__`)."""
        if nazwa.startswith('zakladka_') and nazwa in self.__dict__.get('_leniwe', {}):
            return self.zbuduj_zakladke(nazwa)
        raise AttributeError(nazwa)

    def zakladka_gotowa(self, nazwa):
        """Zbudowana zakładka albo None (nie buduje jej)."""
        return self.__dict__.get(nazwa)

    def zbuduj_zakladke(self, nazwa):
        """Buduje zakładkę leniwą (raz), podmienia zastępnik na karcie, stosuje do niej wygodę pól i zapisane
        ustawienia. Błąd budowy zostaje w karcie jako komunikat (atrybut = None), reszta aplikacji działa."""
        if nazwa in self.__dict__:
            return self.__dict__[nazwa]
        zastepnik, tytul, modul, klasa, zaleznosci = self._leniwe[nazwa]
        import importlib
        import time
        import ustawienia as U
        from PyQt6.QtWidgets import QApplication
        t0 = time.perf_counter()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.statusBar().showMessage(f'Budowanie zakładki „{tytul}”...')
            QApplication.processEvents()
            try:
                argumenty = [_Odnosnik(self, z) for z in zaleznosci]
                zakladka = getattr(importlib.import_module(modul), klasa)(self, *argumenty)
            except Exception as e:                   # brak OpenGL / cv2 itp.: reszta aplikacji działa
                traceback.print_exc()
                komunikat = QLabel(f'Zakładka „{tytul}” niedostępna: {e}')
                komunikat.setWordWrap(True); komunikat.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.zakladki_glowne.podmien(zastepnik, komunikat)
                self.__dict__[nazwa] = None
                return None
            self.zakladki_glowne.podmien(zastepnik, zakladka)
            setattr(self, nazwa, zakladka)           # od teraz zwykły atrybut (widoczny dla ustawień)
            from wygoda import ulatw
            ulatw(zakladka)
            oczekujace = self.__dict__.setdefault('_ustawienia_oczekujace', {})
            moje = {k: oczekujace.pop(k) for k in [k for k in oczekujace if k.startswith(nazwa + '.')]}
            if self.__dict__.get('_domyslne') is not None:
                self._domyslne.update({k: v for k, v in U.stan(self).items() if k.startswith(nazwa + '.')})
            if moje:
                try:
                    U.ustaw(self, moje)
                except Exception:
                    traceback.print_exc()
            self.statusBar().showMessage(f'Zakładka „{tytul}” zbudowana w {time.perf_counter() - t0:.1f} s.', 6000)
            return zakladka
        finally:
            QApplication.restoreOverrideCursor()

    def _po_zmianie_zakladki(self, i):
        """Pierwsze wejście na kartę leniwej zakładki buduje ją (po chwili, żeby najpierw pojawiła się karta)."""
        w = self.zakladki_glowne.widget(i)
        for nazwa, (zastepnik, *_reszta) in self._leniwe.items():
            if w is zastepnik and nazwa not in self.__dict__:
                QTimer.singleShot(20, lambda n=nazwa: self.zbuduj_zakladke(n) if n not in self.__dict__ else None)
                return

    # ------------------------------------------------------------------ UI

    def _buduj_ui(self):
        self.zakladki_glowne = ZakladkiGlowne()
        self.zakladki_glowne.addTab(self._panel_edytora(), 'Edytor SD')
        self.zakladki_glowne.setTabToolTip(0, OPIS_EDYTORA)
        # Pozostałe zakładki to lekkie zastępniki; prawdziwa zakładka (importy, setki kontrolek, wykresy) powstaje
        # przy pierwszym wejściu na kartę albo przy pierwszym użyciu przez inną zakładkę (`zbuduj_zakladke`).
        self._leniwe = {}
        for atrybut, karta, tytul, modul, klasa, zaleznosci in ZAKLADKI_LENIWE:
            zastepnik = QLabel(f'Budowanie zakładki „{tytul}”...')
            zastepnik.setAlignment(Qt.AlignmentFlag.AlignCenter)
            i = self.zakladki_glowne.addTab(zastepnik, karta)
            self.zakladki_glowne.setTabToolTip(i, tytul)
            self._leniwe[atrybut] = (zastepnik, tytul, modul, klasa, zaleznosci)
        self.zakladki_glowne.currentChanged.connect(self._po_zmianie_zakladki)
        self.setCentralWidget(self.zakladki_glowne)

    # ------------------------------------------------------------------ ustawienia (plik konfiguracyjny)

    def _ustawienia_startowe(self):
        """Menu Ustawienia, zapamiętanie domyślnych i wczytanie ustawienia.json (gdy jest)."""
        import ustawienia as U
        from PyQt6.QtGui import QAction
        self._domyslne = U.stan(self)
        m = self.menuBar().addMenu('Ustawienia')
        for nazwa, opis, fn in (
                ('Zapisz ustawienia', f'Zapisuje stan wszystkich list, pól i opcji do {U.PLIK.name} (wczytywany przy starcie).',
                 lambda: self._zapisz_ustawienia(U.PLIK)),
                ('Zapisz ustawienia jako…', 'Zapis do wybranego pliku JSON (np. osobny zestaw do porównań).',
                 lambda: self._zapisz_ustawienia(None)),
                ('Wczytaj ustawienia…', 'Ustawia listy, pola i opcje z wybranego pliku JSON.', self._wczytaj_ustawienia),
                ('Przywróć domyślne', 'Wartości z chwili uruchomienia programu bez pliku ustawień.', self._przywroc_ustawienia)):
            akcja = QAction(nazwa, self); akcja.setToolTip(opis); akcja.setStatusTip(opis); akcja.triggered.connect(fn)
            m.addAction(akcja)
        m.setToolTipsVisible(True)
        m.addSeparator()
        self.akcja_autozapis = QAction('Zapisuj przy zamknięciu', self); self.akcja_autozapis.setCheckable(True)
        self.akcja_autozapis.setChecked(True); m.addAction(self.akcja_autozapis)
        if U.PLIK.exists() and not _bez_ustawien():
            try:
                ok, pom = U.wczytaj(self)
                self.statusBar().showMessage(f'Wczytano ustawienia z {U.PLIK.name}: {ok} kontrolek'
                                             + (f', {pom} pominiętych (brak pozycji)' if pom else ''), 8000)
            except Exception as e:
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
        self.__dict__['_ustawienia_oczekujace'] = {}                # zakładki niezbudowane wystartują z domyślnymi
        ok, _ = __import__('ustawienia').ustaw(self, self._domyslne)
        self.statusBar().showMessage(f'Przywrócono {ok} ustawień domyślnych (zapis do pliku przy zamknięciu albo z menu).', 8000)

    def closeEvent(self, e):
        ruch = self.zakladka_gotowa('zakladka_ruchu')          # niezbudowanej zakładki nie budujemy przy zamknięciu
        if ruch is not None and not ruch.shutdown():
            self.statusBar().showMessage('Zatrzymywanie obliczeń ruchu i zapisywanie wyniku. Zamknij okno ponownie po zakończeniu.')
            e.ignore()
            return
        import ustawienia as U
        if getattr(self, 'akcja_autozapis', None) is not None and self.akcja_autozapis.isChecked() and not _bez_ustawien():
            try:
                U.zapisz(self)
            except Exception:
                traceback.print_exc()
        super().closeEvent(e)


def _bez_ustawien():
    import os
    return os.environ.get('EDYTOR_BEZ_USTAWIEN') == '1'


def main():
    # wspólne konteksty OpenGL: widok 3D przeniesiony do osobnego okna (Odepnij) dostaje nowy kontekst,
    # a shadery i bufory pyqtgraph muszą w nim dalej istnieć (bez tego GL_INVALID_VALUE przy glUseProgram)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    aplikacja = QApplication(sys.argv)
    aplikacja.setStyleSheet(STYL)
    okno = Okno()
    okno.show()
    sys.exit(aplikacja.exec())


if __name__ == '__main__':
    main()
