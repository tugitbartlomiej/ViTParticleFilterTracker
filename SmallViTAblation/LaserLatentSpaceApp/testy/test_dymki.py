"""Testy dymków i nagłówków grup panelu parametrów (dymki.py, grupy_parametrow.py).

Uruchomienie (z katalogu aplikacji, prawdziwa karta GPU i pulpit):
    $env:PYTHONPATH='.'; W:\\Python\\3.11\\python.exe -m unittest testy.test_dymki -v
Okno budowane jak w test_gui.py (bez autowczytywania historii, bez pliku ustawień)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _wspolne_gui as W  # noqa: E402  (najpierw conftest i app, potem reszta)

from PyQt6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QPushButton,  # noqa: E402
                             QToolButton, QWidget)

import dymki  # noqa: E402
import grupy_parametrow as GP  # noqa: E402
from zakladka_laserow import Zwijana  # noqa: E402

KONTROLKI = (QComboBox, QAbstractSpinBox, QCheckBox, QLineEdit, QPushButton, QToolButton)
APP = OKNO = Z = None
ZMIENIONE = ZMIENIONE_2 = NAGLOWKI = NAGLOWKI_2 = None


def _kontrolki_sekcji(sek):
    """Kontrolki sekcji panelu bez wnętrzności złożonych pól (QLineEdit w spinboxie i liście) i bez nagłówków zwijania (Zwijana)."""
    naglowki = {s.naglowek for s in sek.findChildren(Zwijana)} | ({sek.naglowek} if isinstance(sek, Zwijana) else set())
    return [w for w in sek.findChildren(QWidget)
            if isinstance(w, KONTROLKI) and w not in naglowki and not isinstance(w.parentWidget(), (QAbstractSpinBox, QComboBox))]


def _naglowki_grup(sek):
    return [l for l in sek.findChildren(QLabel) if l.objectName() == GP.NAZWA_NAGLOWKA]


def setUpModule():
    global APP, OKNO, Z, ZMIENIONE, ZMIENIONE_2, NAGLOWKI, NAGLOWKI_2
    APP = W.aplikacja()
    OKNO = W.zbuduj_okno()
    Z = OKNO.zakladka_laserow
    # aplikacja sama woła oba zastosuj przy budowie (_buduj): tu stan okna, a liczby z pierwszego wywołania na świeżym drzewie
    ZMIENIONE = getattr(Z, '_dymki_zmienione', None)
    NAGLOWKI = {k: len(_naglowki_grup(s)) for k, s in Z._sek.items() if _naglowki_grup(s)}
    W.przetworz(3)
    ZMIENIONE_2 = dymki.zastosuj(Z)                         # drugie wywołanie: nic nie wolno zmienić ani zdublować
    NAGLOWKI_2 = GP.zastosuj(Z)
    W.przetworz(3)


def tearDownModule():
    W.zamknij_okno(OKNO)


class TestDymki(unittest.TestCase):
    def test_zmieniono_kontrolki_i_ponowne_wywolanie_nic_nie_zmienia(self):
        """Wywołanie przy budowie okna uzupełnia dymki wielu kontrolek, kolejne niczego już nie zmienia."""
        self.assertGreater(ZMIENIONE, 50)
        self.assertEqual(ZMIENIONE_2, 0)

    def test_kazda_kontrolka_sekcji_ma_dymek(self):
        """Każda kontrolka (lista, pole liczbowe, pole wyboru, pole tekstu, przycisk) w sekcjach panelu ma niepusty dymek."""
        brak = []
        for klucz, sek in Z._sek.items():
            for w in _kontrolki_sekcji(sek):
                if not w.toolTip().strip():
                    brak.append(f'{klucz}: {type(w).__name__} {getattr(w, "text", lambda: "")()!r}')
        self.assertEqual(brak, [], 'kontrolki bez dymku:\n' + '\n'.join(brak))

    def test_kazda_etykieta_wiersza_ma_dymek(self):
        """Etykieta każdego wiersza QFormLayout, który ma pole, ma dymek (po najechaniu na nazwę parametru widać objaśnienie)."""
        brak = []
        for klucz, sek in Z._sek.items():
            for f in sek.findChildren(QFormLayout):
                for r in range(f.rowCount()):
                    et = f.itemAt(r, QFormLayout.ItemRole.LabelRole)
                    pole = f.itemAt(r, QFormLayout.ItemRole.FieldRole)
                    if et is not None and pole is not None and et.widget() is not None and not et.widget().toolTip().strip():
                        brak.append(f'{klucz}: {et.widget().text()!r}')
        self.assertEqual(brak, [], 'etykiety bez dymku:\n' + '\n'.join(brak))

    def test_dymek_etykiety_to_dymek_pola(self):
        """Dymek etykiety jest tym samym tekstem co dymek jej pola (np. „Siatka (węzłów na bok)” i sp_og_G)."""
        f = Z._sek['pulapka'].layout()
        wiersz = GP.wiersz(f, Z.sp_og_G)
        self.assertGreaterEqual(wiersz, 0)
        self.assertEqual(f.itemAt(wiersz, QFormLayout.ItemRole.LabelRole).widget().toolTip(), Z.sp_og_G.toolTip())

    def test_lista_zachowuje_opis_po_zmianie_wyboru(self):
        """Dymek listy to opis parametru z dopiskiem o wyborze; zmiana wyboru nie podmienia opisu samym tekstem pozycji (jak robił to automat)."""
        cb = Z.cb_wiazki
        stary = cb.currentIndex()
        try:
            cb.setCurrentIndex(2)
            self.assertIn('Które wiązki rysować', cb.toolTip())
            self.assertIn(cb.currentText(), cb.toolTip())
        finally:
            cb.setCurrentIndex(stary)

    def test_dobre_dymki_zostaja(self):
        """Istniejący opis wieloliniowy albo długi nie jest nadpisywany."""
        self.assertIn('MANIFEST_35000', Z.cb_film.toolTip())
        self.assertIn('Pole wszystkich 768 wiązek', Z.ch_og.toolTip())

    def test_slownik_odnosi_sie_do_istniejacych_atrybutow(self):
        """Każdy klucz słownika DYMKI (z) i DYMKI_RZUTY (z.rzuty) to istniejący widżet: literówka w nazwie nie przechodzi niezauważona."""
        for nazwa, slownik, obiekt in (('z', dymki.DYMKI, Z), ('z.rzuty', dymki.DYMKI_RZUTY, Z.rzuty)):
            for klucz in slownik:
                self.assertIsInstance(getattr(obiekt, klucz, None), QWidget, f'{nazwa}.{klucz}')


class TestGrupy(unittest.TestCase):
    def test_wstawiono_naglowki(self):
        """Długie sekcje dostały nagłówki grup: liczba zgodna z wynikiem zastosuj i z drzewem widżetów."""
        for klucz in ('pulapka', 'sfera', 'dane'):
            self.assertGreaterEqual(NAGLOWKI[klucz], 3, klucz)
        for klucz, n in NAGLOWKI.items():
            self.assertEqual(len(_naglowki_grup(Z._sek[klucz])), n, klucz)

    def test_ponowne_wywolanie_nie_dubluje(self):
        """Drugie wywołanie nie dodaje nagłówków (są oznaczone objectName „naglowek_grupy”)."""
        self.assertTrue(all(n == 0 for n in NAGLOWKI_2.values()), NAGLOWKI_2)
        for klucz, n in NAGLOWKI.items():
            self.assertEqual(len(_naglowki_grup(Z._sek[klucz])), n, klucz)

    def test_naglowek_stoi_tuz_przed_pierwsza_kontrolka_grupy(self):
        """Nagłówek „Liczenie pułapki” leży w wierszu bezpośrednio przed wierszem kontrolki ch_og, a „Światło ogniska” przed ch_og_sw."""
        f = Z._sek['pulapka'].layout()
        for tytul, atr in (('Liczenie pułapki', 'ch_og'), ('Światło ogniska', 'ch_og_sw'), ('Lokalne pole: mgła i strzałki', 'ch_og_mgla')):
            r = GP.wiersz(f, getattr(Z, atr))
            naglowek = f.itemAt(r - 1, QFormLayout.ItemRole.SpanningRole)
            self.assertIsNotNone(naglowek, tytul)
            self.assertEqual(naglowek.widget().objectName(), GP.NAZWA_NAGLOWKA)
            self.assertEqual(naglowek.widget().text(), tytul)

    def test_kolejnosc_grup_zgodna_z_kolejnoscia_wierszy(self):
        """Nagłówki pułapki idą w kolejności wierszy z GRUPY (wstawianie od końca nie przestawia numerów)."""
        f = Z._sek['pulapka'].layout()
        teksty = []
        for r in range(f.rowCount()):
            el = f.itemAt(r, QFormLayout.ItemRole.SpanningRole)
            if el is not None and el.widget() is not None and el.widget().objectName() == GP.NAZWA_NAGLOWKA:
                teksty.append(el.widget().text())
        self.assertEqual(teksty, [t for t, _ in GP.GRUPY['pulapka']])

    def test_brakujacy_atrybut_jest_pomijany(self):
        """Grupa z nieistniejącym atrybutem nie zgłasza wyjątku i nic nie wstawia; nieznana sekcja też jest pomijana."""
        przed = len(_naglowki_grup(Z._sek['pulapka']))
        wynik = GP.zastosuj(Z, {'pulapka': [('Nic', 'nie_ma_takiego_atrybutu')], 'nie_ma_sekcji': [('X', 'ch_og')]})
        self.assertEqual(wynik, {'pulapka': 0})
        self.assertEqual(len(_naglowki_grup(Z._sek['pulapka'])), przed)


if __name__ == '__main__':
    unittest.main()
