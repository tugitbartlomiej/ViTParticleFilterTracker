"""Testy integracyjne GUI „Lasery i koloidy” (PyQt6 + OpenGL): okno, menu, panel, popupy ustawień, układ okienek, ustawienia,
etykiety, widoczność okienek, bufor GPU 'ZH', brak pełzania układu, budżet wydajności, zapis obrazu, prawy klik.

Uruchomienie (z katalogu aplikacji, prawdziwa karta GPU i pulpit):
    $env:PYTHONPATH='.'; W:\\Python\\3.11\\python.exe -m unittest testy.test_gui -v
Jedna QApplication na moduł. Ustawienia: tylko pliki tymczasowe (EDYTOR_BEZ_USTAWIEN=1), cache_uczenia tylko do odczytu."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _wspolne_gui as W  # noqa: E402  (najpierw conftest i app, potem reszta)

import numpy as np  # noqa: E402
from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QContextMenuEvent, QCursor, QImage, QKeySequence  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow  # noqa: E402

import etykiety_narzedzia as EN  # noqa: E402
import rzuty_okienka as RZO  # noqa: E402
import ustawienia as U  # noqa: E402
from panel_ustawien import PopupUstawien  # noqa: E402
from widok_gl import WidokGL  # noqa: E402

APP = None
METRYKI = {}                                         # liczby do raportu (wypisywane na końcu modułu)


def setUpModule():
    global APP
    APP = W.aplikacja()


def tearDownModule():
    if METRYKI:
        print('\n[METRYKI]', json.dumps(METRYKI, ensure_ascii=False, indent=1), file=sys.stderr)


class Baza(unittest.TestCase):
    """Po każdym teście: żadnych wyjątków ze slotów Qt i żadnych zostawionych popupów."""

    def setUp(self):
        W.WYJATKI.clear()

    def tearDown(self):
        W.zamknij_popupy()
        wyj, W.WYJATKI[:] = list(W.WYJATKI), []
        self.assertEqual(wyj, [], 'wyjątki w slotach Qt:\n' + '\n'.join(wyj))


class BazaOkna(Baza):
    """Jedno okno na klasę (bez historii); każdy test startuje od układu domyślnego."""

    @classmethod
    def setUpClass(cls):
        cls.okno = W.zbuduj_okno()
        cls.z = cls.okno.zakladka_laserow

    @classmethod
    def tearDownClass(cls):
        W.zamknij_okno(cls.okno)

    def setUp(self):
        super().setUp()
        self.z._przywroc_uklad()
        W.przetworz(5)


# ====================================================================== 1. okno, menu, skróty
class TestOknoMenu(BazaOkna):
    def test_01_okno_sie_buduje(self):
        """Okno główne powstaje, jego środkiem jest zakładka, okienka to doki wewnętrznego QMainWindow, a wszystkie okienka R* istnieją."""
        o, z = self.okno, self.z
        self.assertTrue(o.isVisible())
        self.assertIs(o.centralWidget(), z)
        self.assertIsInstance(z.okna, QMainWindow)
        oczekiwane = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 's', 'p'} | {k for k, _, _ in RZO.OKIENKA}
        self.assertTrue(oczekiwane <= set(z.doki), f'brak okienek: {oczekiwane - set(z.doki)}')
        self.assertEqual(set(z.doki_ster), {'parametry', 'podkroki'})
        for k, d in z.doki.items():
            self.assertIsInstance(d, QDockWidget, k)
            self.assertIs(d.parentWidget(), z.okna, k)
        self.assertTrue(z.doki['s'].isVisible())

    def test_02_pasek_menu(self):
        """Pasek menu ma Plik, Ustawienia, Okienka, Widok, Pomoc (w tej kolejności)."""
        nazwy = [a.text().replace('&', '') for a in self.okno.menuBar().actions()]
        self.assertEqual(nazwy, ['Plik', 'Ustawienia', 'Okienka', 'Widok', 'Pomoc'])

    def test_03_plik_skroty(self):
        """Menu Plik zawiera akcje ze skrótami Ctrl+O, Ctrl+S, Ctrl+Shift+S, Ctrl+E i Ctrl+Q (i nic się nie dubluje)."""
        plik = [a for a in self.okno.menuBar().actions() if a.text() == 'Plik'][0].menu()
        skroty = [a.shortcut().toString() for a in plik.actions() if not a.shortcut().isEmpty()]
        for s in ('Ctrl+O', 'Ctrl+S', 'Ctrl+Shift+S', 'Ctrl+E', 'Ctrl+Q'):
            self.assertIn(s, skroty)
        self.assertEqual(len(skroty), len(set(skroty)), f'zdublowane skróty: {skroty}')
        ctrl_e = [a for a in plik.actions() if a.shortcut().toString() == 'Ctrl+E'][0]
        self.assertIn('PNG', ctrl_e.text())

    def _klawisz_przelacza(self, klawisz, dok, akcja):
        """Naciska klawisz w oknie głównym; zwraca (zmieniło się, okno było aktywne)."""
        o = self.okno
        o.activateWindow(); o.raise_()
        W.czekaj(o.isActiveWindow, 3)
        przed = dok.isVisible()
        QTest.keyClick(o, klawisz)
        W.przetworz(5)
        if dok.isVisible() != przed:
            return True, o.isActiveWindow()
        return False, o.isActiveWindow()

    def test_04_f2_f3_przelaczaja_dok(self):
        """F2 chowa i pokazuje dok „Parametry”, F3 dok „Podkroki” (skróty akcji z menu Widok)."""
        z = self.z
        widok = [a for a in self.okno.menuBar().actions() if a.text() == 'Widok'][0].menu()
        skroty = {a.shortcut().toString(): a for a in widok.actions() if not a.shortcut().isEmpty()}
        self.assertIn('F2', skroty); self.assertIn('F3', skroty)
        for klawisz, nazwa, dok in ((Qt.Key.Key_F2, 'F2', z.dok_panel), (Qt.Key.Key_F3, 'F3', z.dok_podkroki)):
            self.assertTrue(dok.isVisible(), nazwa)
            zmiana, aktywne = self._klawisz_przelacza(klawisz, dok, skroty[nazwa])
            if not zmiana and not aktywne:                          # ograniczenie środowiska: pulpit nie oddał fokusu oknu testowemu
                skroty[nazwa].trigger(); W.przetworz(5)
                METRYKI['f2_f3_przez_trigger_bo_okno_nieaktywne'] = True
            self.assertFalse(dok.isVisible(), f'{nazwa} nie schował doku (okno aktywne: {aktywne})')
            zmiana, aktywne = self._klawisz_przelacza(klawisz, dok, skroty[nazwa])
            if not zmiana and not aktywne:
                skroty[nazwa].trigger(); W.przetworz(5)
            self.assertTrue(dok.isVisible(), f'{nazwa} nie pokazał doku (okno aktywne: {aktywne})')


# ====================================================================== 2. panel parametrów
class TestPanel(BazaOkna):
    def test_01_kazda_strona_istnieje(self):
        """Każdy klucz z STRONY jest stroną panelu (kolejność, liczba stron i pozycji listy zgodne)."""
        p = self.z.panel
        klucze = [s[0] for s in self.z.STRONY]
        self.assertEqual(p._klucze, klucze)
        self.assertEqual(p.strony.count(), len(klucze))
        self.assertEqual(p.lista.count(), len(klucze))

    def test_02_sekcje_stron_istnieja_i_leza_na_stronie(self):
        """Każda sekcja wymieniona w STRONY istnieje w z._sek i leży na swojej stronie panelu."""
        z = self.z
        for i, (klucz, _, sekcje, _) in enumerate(z.STRONY):
            for s in sekcje:
                with self.subTest(strona=klucz, sekcja=s):
                    self.assertIn(s, z._sek)
                    self.assertTrue(z.panel.strony.widget(i).isAncestorOf(z._sek[s]), 'sekcja nie leży na stronie')

    def test_03_sekcje_okienek_istnieja(self):
        """Każda sekcja z SEKCJE_OKIENEK istnieje w z._sek, a okienka R* mają niepuste podsekcje (Rzuty.sekcje_okienka)."""
        z = self.z
        for k, nazwy in z.SEKCJE_OKIENEK.items():
            self.assertIn(k, z.doki)
            for n in nazwy:
                with self.subTest(okienko=k, sekcja=n):
                    self.assertIn(n, z._sek)
        for k in z.doki:
            if k.startswith('R'):
                with self.subTest(okienko=k):
                    sek = z.rzuty.sekcje_okienka(k)
                    self.assertIsInstance(sek, list)
                    for s in sek:
                        self.assertTrue(z.panel.isAncestorOf(s), f'podsekcja {k} poza panelem')

    def test_04_pokaz_przelacza_strone(self):
        """panel.pokaz(klucz) przełącza stronę (lista, stos i biezaca()) i emituje zmiana_strony."""
        p = self.z.panel
        zgloszone = []
        p.zmiana_strony.connect(zgloszone.append)
        kolejnosc = [s[0] for s in self.z.STRONY]
        for k in kolejnosc[::-1] + kolejnosc:
            p.pokaz(k)
            self.assertEqual(p.biezaca(), k)
            self.assertEqual(p.strony.currentIndex(), kolejnosc.index(k))
        self.assertTrue(set(kolejnosc) <= set(zgloszone))
        p.pokaz('nie_ma_takiej')
        self.assertEqual(p.biezaca(), kolejnosc[-1])               # nieznany klucz nic nie zmienia


# ====================================================================== 3. wypożyczanie sekcji przez popupy
def _sekcje(z, k):
    """Sekcje, które okienko k wypożycza (jak z.menu_okienka)."""
    if k.startswith('R'):
        return list(z.rzuty.sekcje_okienka(k))
    return [z._sek[n] for n in z.SEKCJE_OKIENEK.get(k, ())]


def _zapis(s):
    """Stan sekcji w panelu: rodzic, liczba pozycji układu rodzica, zwinięcie."""
    r = s.parentWidget()
    uk = r.layout() if r is not None else None
    zw = s.zwinieta() if hasattr(s, 'zwinieta') else None
    return dict(rodzic=r, n=uk.count() if uk is not None else None, zwinieta=zw)


class TestPopupy(BazaOkna):
    def _sprawdz_wrocily(self, sekcje, przed, kontekst=''):
        W.przetworz(4)
        for s in sekcje:
            po = _zapis(s)
            self.assertIs(po['rodzic'], przed[id(s)]['rodzic'], f'{kontekst}: sekcja wróciła do innego rodzica')
            self.assertEqual(po['n'], przed[id(s)]['n'], f'{kontekst}: zmieniła się liczba pozycji układu rodzica')
            self.assertEqual(po['zwinieta'], przed[id(s)]['zwinieta'], f'{kontekst}: stan zwinięcia')
            self.assertFalse(s.isHidden(), f'{kontekst}: sekcja ukryta po powrocie')
            self.assertTrue(s.isVisibleTo(przed[id(s)]['rodzic']), f'{kontekst}: sekcja niewidoczna w układzie panelu')
        self.assertEqual(W.napisy_zastepnika(self.z.panel), [], f'{kontekst}: zastępnik został w panelu')
        self.assertEqual(PopupUstawien._u_kogo, {}, f'{kontekst}: _u_kogo niepuste')
        self.assertEqual(PopupUstawien.otwarte, [], f'{kontekst}: przypięte popupy zostały')

    def test_01_kazdy_dok_wypozycza_i_oddaje(self):
        """menu_okienka(k) dla każdego doku: sekcje przechodzą do popupu (z zastępnikami w panelu); po zamknięciu wracają do swojego
        rodzica (te same układy, widoczne), zastępników brak, PopupUstawien._u_kogo puste."""
        z = self.z
        klucze = list(z.doki) + list(z.doki_ster)
        for k in klucze:
            with self.subTest(okienko=k):
                sek = _sekcje(z, k)
                przed = {id(s): _zapis(s) for s in sek}
                z.menu_okienka(k, QPoint(400, 300))
                W.przetworz(3)
                p = W.popup_otwarty()
                self.assertIsNotNone(p, 'popup się nie otworzył')
                for s in sek:
                    self.assertIs(s.window(), p, 'sekcja nie jest w popupie')
                if sek:
                    self.assertEqual(len(W.napisy_zastepnika(z.panel)), len([s for s in sek if przed[id(s)]['n'] is not None]))
                p.close()
                self._sprawdz_wrocily(sek, przed, k)

    def test_02_przypiecie(self):
        """przypnij(): popup zostaje otwarty jako osobne okno (w PopupUstawien.otwarte) i trzyma sekcje; zamknięcie oddaje je panelowi."""
        z = self.z
        for k in ('s', 'R1', '11'):
            with self.subTest(okienko=k):
                sek = _sekcje(z, k)
                przed = {id(s): _zapis(s) for s in sek}
                z.menu_okienka(k, QPoint(300, 200))
                p = W.popup_otwarty()
                self.assertIsNotNone(p)
                p.przypnij(); W.przetworz(4)
                self.assertIn(p, PopupUstawien.otwarte)
                self.assertTrue(p.isVisible())
                self.assertFalse(p.windowFlags() & Qt.WindowType.Popup == Qt.WindowType.Popup and
                                 not (p.windowFlags() & Qt.WindowType.Tool), 'przypięty popup nadal jest Popup')
                for s in sek:
                    self.assertIs(s.window(), p)
                    self.assertEqual(PopupUstawien._u_kogo.get(id(s)), p)
                W.przetworz(10); QTest.qWait(100)
                self.assertTrue(p.isVisible(), 'przypięty popup zniknął po chwili')
                p.close()
                self._sprawdz_wrocily(sek, przed, k)

    def test_03_drugi_popup_ze_wspolna_sekcja_zamyka_pierwszy(self):
        """Popup R3 otwarty przy popupie R1 (wspólne sekcje) zamyka R1; sekcje R1, których R3 nie ma, wracają do panelu,
        wspólne są w R3; zamknięcie R3 oddaje wszystko. Dla R1 przypiętego i nieprzypiętego."""
        z = self.z
        s1, s3 = _sekcje(z, 'R1'), _sekcje(z, 'R3')
        wspolne = {id(s) for s in s1} & {id(s) for s in s3}
        self.assertTrue(wspolne, 'R1 i R3 nie mają wspólnych sekcji (założenie testu)')
        wszystkie = {id(s): s for s in s1 + s3}.values()
        for przypiety in (True, False):
            with self.subTest(pierwszy_przypiety=przypiety):
                przed = {id(s): _zapis(s) for s in wszystkie}
                z.menu_okienka('R1', QPoint(300, 200))
                p1 = W.popup_otwarty()
                self.assertIsNotNone(p1)
                if przypiety:
                    p1.przypnij(); W.przetworz(3)
                z.menu_okienka('R3', QPoint(500, 250))
                W.przetworz(4)
                self.assertTrue(sip.isdeleted(p1) or not p1.isVisible(), 'pierwszy popup nie został zamknięty')
                p3 = W.popup_otwarty()
                self.assertIsNotNone(p3)
                self.assertFalse(sip.isdeleted(p1) is False and p3 is p1)
                for s in s3:
                    self.assertIs(s.window(), p3)
                for s in s1:
                    if id(s) not in wspolne:
                        self.assertIsNot(s.window(), p3)
                        self.assertTrue(z.panel.isAncestorOf(s), 'sekcja tylko R1 nie wróciła do panelu')
                for s in wszystkie:
                    if id(s) in {id(x) for x in s3}:
                        self.assertIs(PopupUstawien._u_kogo.get(id(s)), p3)
                p3.close()
                self._sprawdz_wrocily(wszystkie, przed, f'R1->R3 (przypiety={przypiety})')

    def test_04_zamkniecie_okna_oddaje_sekcje(self):
        """Zamknięcie okna głównego zamyka przypięte popupy i oddaje sekcje panelowi (zapis ustawień nie widzi sekcji w popupie)."""
        o = W.zbuduj_okno()
        try:
            z = o.zakladka_laserow
            sek = _sekcje(z, 'R1') + _sekcje(z, 's')
            przed = {id(s): _zapis(s) for s in sek}
            for k in ('R1', 's'):
                z.menu_okienka(k, QPoint(300, 200))
                W.popup_otwarty().przypnij(); W.przetworz(3)       # popup_otwarty: najpierw nieprzypięty
            self.assertEqual(len(PopupUstawien.otwarte), 2)
            o.close(); W.przetworz(5)
            self.assertEqual(PopupUstawien.otwarte, [])
            self.assertEqual(PopupUstawien._u_kogo, {})
            for s in sek:
                self.assertIs(_zapis(s)['rodzic'], przed[id(s)]['rodzic'])
                self.assertFalse(s.isHidden())
            self.assertEqual(W.napisy_zastepnika(z.panel), [])
        finally:
            W.zamknij_okno(o)


# ====================================================================== 4. układ okienek
class TestUklad(BazaOkna):
    def test_01_struktura_slownika(self):
        """uklad_okienek(): słownik z wersja_ukladu == 2, okna (base64), sloty (dict) i okno (geometria okna głównego); serializowalny do JSON."""
        v = self.z.uklad_okienek()
        self.assertEqual(v['wersja_ukladu'], 2)
        for k in ('okna', 'sloty', 'okno'):
            self.assertIn(k, v)
        self.assertIsInstance(v['okna'], str); self.assertGreater(len(v['okna']), 50)
        self.assertIsInstance(v['sloty'], dict)
        self.assertIsInstance(v['okno'], str); self.assertGreater(len(v['okno']), 20)
        json.dumps(v)

    def _zmien_uklad(self):
        z = self.z
        z._uklad_gotowy('rzuty')
        for k in ('1', '2', '9'):
            z.doki[k].close()
        z.doki['3'].setFloating(True)
        z.doki_ster['podkroki'].close()
        W.przetworz(5)

    def test_02_zapis_zmiana_przywrocenie(self):
        """Zapis → zmiana układu (układ „rzuty”, zamknięte i odpięte doki) → ustaw_uklad_okienek(zapis) przywraca, które doki są
        otwarte, zamknięte i odpięte, oraz ich obszary."""
        z = self.z
        z.doki['4'].close(); z.doki['R5'].setFloating(True); W.przetworz(5)
        przed = W.stan_dokow(z)
        zapis = z.uklad_okienek()
        self._zmien_uklad()
        self.assertNotEqual(W.stan_dokow(z), przed, 'zmiana układu nic nie zmieniła (błąd testu)')
        z.ustaw_uklad_okienek(zapis); W.przetworz(8)
        po = W.stan_dokow(z)
        self.assertEqual({k: v[:2] for k, v in po.items()}, {k: v[:2] for k, v in przed.items()}, 'otwarte/odpięte po przywróceniu')
        self.assertEqual(po, przed)

    def test_03_slot_zapisz_i_przywroc(self):
        """_slot(1, zapisz=True) zapamiętuje układ, _slot(1) go przywraca (otwarte, odpięte, obszary); pusty slot nie zmienia układu."""
        z = self.z
        z.doki['5'].close(); z.doki['R2'].setFloating(True); W.przetworz(5)
        przed = W.stan_dokow(z)
        z._slot(1, zapisz=True)
        self.assertIn(1, z._sloty)
        self.assertIn('1', z.uklad_okienek()['sloty'])
        self._zmien_uklad()
        z._slot(1); W.przetworz(8)
        self.assertEqual(W.stan_dokow(z), przed)
        po_pustym = W.stan_dokow(z)
        z._slot(3); W.przetworz(3)
        self.assertEqual(W.stan_dokow(z), po_pustym)

    def test_04_przywroc_uklad_pokazuje_wszystkie_doki(self):
        """_przywroc_uklad() pokazuje wszystkie doki (także zamknięte) i przypina odpięte."""
        z = self.z
        self._zmien_uklad()
        z._przywroc_uklad(); W.przetworz(8)
        for k, d in list(z.doki.items()) + list(z.doki_ster.items()):
            with self.subTest(dok=k):
                self.assertTrue(d.isVisible(), 'dok niewidoczny po przywróceniu układu')
                self.assertFalse(d.isFloating(), 'dok odpięty po przywróceniu układu')

    def test_05_skroty_slotow_ctrl_shift_i_ctrl(self):
        """Akcje Ctrl+Shift+1 (zapamiętaj) i Ctrl+1 (przywróć) istnieją w zakładce i działają przez QAction.trigger()."""
        z = self.z
        akcje = {a.shortcut().toString(): a for a in z.actions() if not a.shortcut().isEmpty()}
        for s in ('Ctrl+Shift+1', 'Ctrl+Shift+2', 'Ctrl+Shift+3', 'Ctrl+1', 'Ctrl+2', 'Ctrl+3'):
            self.assertIn(s, akcje)
        z.doki['6'].close(); W.przetworz(3)
        przed = W.stan_dokow(z)
        akcje['Ctrl+Shift+2'].trigger()
        self._zmien_uklad()
        akcje['Ctrl+2'].trigger(); W.przetworz(8)
        self.assertEqual(W.stan_dokow(z), przed)

    def test_06_start_z_zapisanym_ukladem_przed_pierwszym_pokazaniem(self):
        """NOWE okno z plikiem ustawień, w którym część doków jest zamknięta, startuje z tymi dokami zamkniętymi już PRZED pierwszym
        show() (startowe wczytanie ustawień), a po pokazaniu zamknięte nie są widoczne, pozostałe tak."""
        z = self.z
        zamkniete = ('1', '2', 'R3', '12')
        for k in zamkniete:
            z.doki[k].close()
        z.doki_ster['podkroki'].close()
        W.przetworz(5)
        with tempfile.TemporaryDirectory() as kat:
            plik = Path(kat) / 'u.json'
            U.zapisz(self.okno, plik)
            zapisane = json.loads(plik.read_text(encoding='utf-8'))['ustawienia']
            self.assertIn('zakladka_laserow.uklad_okienek', zapisane)
            o2 = W.zbuduj_okno(pokaz=False, plik_startowy=plik)
            try:
                z2 = o2.zakladka_laserow
                self.assertFalse(o2.isVisible())
                self.assertTrue(o2.statusBar().currentMessage().startswith('Wczytano ustawienia'), o2.statusBar().currentMessage())
                for k, d in z2.doki.items():            # przed show(): zamknięte są jawnie ukryte, reszta nie
                    with self.subTest(przed_show=k):
                        self.assertEqual(d.isHidden(), k in zamkniete, 'stan doku przed pierwszym show()')
                self.assertTrue(z2.doki_ster['podkroki'].isHidden())
                self.assertFalse(z2.doki_ster['parametry'].isHidden())
                W.pokaz_okno(o2, 400)
                for k, d in z2.doki.items():
                    with self.subTest(po_show=k):
                        self.assertEqual(d.isVisible(), k not in zamkniete, 'widoczność doku po pierwszym show()')
                self.assertFalse(z2.doki_ster['podkroki'].isVisible())
                self.assertTrue(z2.doki_ster['parametry'].isVisible())
            finally:
                W.zamknij_okno(o2)


# ====================================================================== 5. ustawienia: zapis i odczyt
class TestUstawienia(BazaOkna):
    ZMIANY = (('sp_rozm', 7.5), ('sp_jas', 2.5), ('cb_wiazki', 2), ('ch_swiatlo', False), ('sp_gain', 0.5), ('cb_tempo', 1),
              ('sp_tempo', 12.0), ('cb_skala', 1), ('ch_petla', True), ('sp_okno', 450), ('cb_kolor_cz', 3), ('cb_kol_sciezki', 4),
              ('sp_etyk_prog', 0.25))

    def _wartosc(self, w):
        from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox
        if isinstance(w, QComboBox):
            return w.currentIndex()
        if isinstance(w, QAbstractSpinBox):
            return w.value()
        return w.isChecked()

    def _ustaw(self, w, v):
        from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox
        if isinstance(w, QComboBox):
            w.setCurrentIndex(v)
        elif isinstance(w, QAbstractSpinBox):
            w.setValue(v)
        else:
            w.setChecked(v)

    def test_01_zapis_zmiana_wczytanie(self):
        """U.zapisz(okno, plik tymczasowy) → zmiana kilkunastu kontrolek → U.wczytaj(okno, plik) przywraca je wszystkie."""
        z = self.z
        przed = {n: self._wartosc(getattr(z, n)) for n, _ in self.ZMIANY}
        with tempfile.TemporaryDirectory() as kat:
            plik = Path(kat) / 'u.json'
            n = U.zapisz(self.okno, plik)
            self.assertGreater(n, 100)
            dane = json.loads(plik.read_text(encoding='utf-8'))
            self.assertEqual(dane['wersja'], U.WERSJA)
            for nazwa, v in self.ZMIANY:
                self._ustaw(getattr(z, nazwa), v)
            zmienione = {n_: self._wartosc(getattr(z, n_)) for n_, _ in self.ZMIANY}
            for nazwa, v in self.ZMIANY:
                self.assertEqual(zmienione[nazwa], v, f'{nazwa}: test nie zdołał zmienić kontrolki')
                self.assertNotEqual(zmienione[nazwa], przed[nazwa], f'{nazwa}: zmiana zgodna ze stanem początkowym (błąd testu)')
            ok, pom = U.wczytaj(self.okno, plik)
            self.assertGreater(ok, 100)
        po = {n_: self._wartosc(getattr(z, n_)) for n_, _ in self.ZMIANY}
        for nazwa, _ in self.ZMIANY:
            self.assertEqual(po[nazwa], przed[nazwa], f'{nazwa} nie wrócił')
        METRYKI['ustawienia_ok_pominiete'] = (ok, pom)

    def test_02_caly_stan_okrąg(self):
        """stan(okno) po zapisie, ZMIANIE i wczytaniu jest identyczny z początkowym dla WSZYSTKICH kontrolek (poza układem okienek)."""
        z = self.z
        bez_ukladu = lambda s: {k: v for k, v in s.items() if not k.endswith('.uklad_okienek')}
        przed = bez_ukladu(U.stan(self.okno))
        with tempfile.TemporaryDirectory() as kat:
            plik = Path(kat) / 'u.json'
            U.zapisz(self.okno, plik)
            for nazwa, v in self.ZMIANY:
                self._ustaw(getattr(z, nazwa), v)
            U.wczytaj(self.okno, plik)
        po = bez_ukladu(U.stan(self.okno))
        self.assertEqual(set(po), set(przed))
        różne = {k: (przed[k], po[k]) for k in przed if przed[k] != po[k]}
        self.assertEqual(różne, {}, 'kontrolki, które nie wróciły po zapisie i wczytaniu')

    def test_04_wczytanie_wlasnego_zapisu_bez_pominietych(self):
        """Wczytanie pliku zapisanego przez to samo okno nie pomija żadnej kontrolki (puste listy z indeksem -1 też wracają bez „pominięcia”)."""
        with tempfile.TemporaryDirectory() as kat:
            plik = Path(kat) / 'u.json'
            U.zapisz(self.okno, plik)
            ok, pom = U.wczytaj(self.okno, plik)
        stan = U.stan(self.okno)
        puste = [k for k, v in stan.items() if isinstance(v, dict) and v.get('indeks') == -1]
        self.assertEqual(pom, 0, f'pominięte przy wczytaniu własnego zapisu (puste listy: {puste})')

    def test_05_plik_domyslny_nietkniety(self):
        """Testy nie zapisują prawdziwego ustawienia.json (zapis okna z EDYTOR_BEZ_USTAWIEN=1 omija plik)."""
        przed = U.PLIK.stat().st_mtime if U.PLIK.exists() else None
        o = W.zbuduj_okno()
        W.zamknij_okno(o)
        po = U.PLIK.stat().st_mtime if U.PLIK.exists() else None
        self.assertEqual(przed, po)


# ====================================================================== 11. zapis obrazu, 12. prawy klik (bez historii)
class TestObrazIPrawyKlik(BazaOkna):
    def test_01_zapisz_obraz_okienka_pod_kursorem(self):
        """Plik → „Zapisz obraz okienka pod kursorem”: okno._zapisz_obraz z podstawionym QFileDialog.getSaveFileName zapisuje niepusty,
        poprawny PNG o rozmiarze zawartości doku pod kursorem (albo całego okna, gdy pod kursorem nie ma doku)."""
        o, z = self.okno, self.z
        d = z.doki['s']
        cel = d.widget()
        srodek = cel.mapToGlobal(cel.rect().center())
        QCursor.setPos(srodek); W.przetworz(5); QTest.qWait(100)
        w = QApplication.widgetAt(QCursor.pos())
        while w is not None and not isinstance(w, QDockWidget):
            w = w.parentWidget()
        with tempfile.TemporaryDirectory() as kat:
            plik = str(Path(kat) / 'okienko.png')
            with mock.patch.object(QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (plik, ''))):
                o._zapisz_obraz()
            self.assertTrue(os.path.exists(plik), 'plik PNG nie powstał')
            self.assertGreater(os.path.getsize(plik), 500)
            im = QImage(plik)
            self.assertFalse(im.isNull())
            oczek = (w.widget() if w is not None else o).size()
            dpr = o.devicePixelRatio()
            self.assertEqual((im.width(), im.height()), (round(oczek.width() * dpr), round(oczek.height() * dpr)))
            if w is not None:
                METRYKI['obraz_okienka_px'] = (im.width(), im.height(), w.objectName())
                self.assertIs(w, d, 'kursor ustawiony na sferze, a obraz zapisano dla innego okienka (środowisko?)')
            self.assertIn('Zapisano', o.statusBar().currentMessage())

    def test_02_zapisz_obraz_anulowanie(self):
        """Anulowanie dialogu (pusta nazwa) nie zapisuje niczego i nie zgłasza błędu."""
        with mock.patch.object(QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: ('', ''))):
            self.okno._zapisz_obraz()

    def test_03_prawy_klik_w_widok_otwiera_popup(self):
        """Sygnał prawy_px widoku OpenGL (WidokGL) otwiera PopupUstawien właściwego okienka (podlacz_menu); zamknięcie zwraca sekcje."""
        z = self.z
        for widok, klucz, tytul in ((z.vs, 's', 'Sfera'), (z.vog, '11', '11. Pułapka'), (z.vdol, '12', '12. Dolina'),
                                    (z.vmech, '13', '13. Reflektor'), (z.v1, '1', '1. Ruch'), (z.vp, 'p', 'Przejście')):
            with self.subTest(okienko=klucz):
                sek = _sekcje(z, klucz)
                przed = {id(s): _zapis(s) for s in sek}
                widok.prawy_px.emit(10.0, 10.0)
                W.przetworz(4)
                p = W.popup_otwarty()
                self.assertIsNotNone(p, 'prawy klik nie otworzył popupu')
                self.assertTrue(p.windowTitle().startswith('Ustawienia: ' + tytul), p.windowTitle())
                for s in sek:
                    self.assertIs(s.window(), p)
                p.close(); W.przetworz(4)
                for s in sek:
                    self.assertIs(_zapis(s)['rodzic'], przed[id(s)]['rodzic'])
                    self.assertFalse(s.isHidden())

    def test_04_prawy_klik_przez_zdarzenie_menu_kontekstowego(self):
        """QContextMenuEvent wysłany do napisu pod widokiem (_Oslona) i do paska tytułu doku otwiera popup tego okienka
        (zdarzenie dociera do filtra doku przez propagację)."""
        z = self.z
        for klucz, cel in (('s', z.lbs), ('1', z.lb1), ('s', z.doki['s'].titleBarWidget()), ('11', z.doki['11'])):
            with self.subTest(okienko=klucz, cel=type(cel).__name__):
                if not cel.isVisible():
                    z.doki[klucz].raise_(); W.przetworz(4)
                gl = cel.mapToGlobal(QPoint(5, 5))
                ev = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(5, 5), gl)
                QApplication.sendEvent(cel, ev)
                W.przetworz(4)
                p = W.popup_otwarty()
                self.assertIsNotNone(p, 'menu kontekstowe nie otworzyło popupu')
                p.close(); W.przetworz(3)


# ====================================================================== testy z wczytaną historią (cache, GPU)
class BazaHistorii(Baza):
    """Okno z wczytaną prawdziwą historią (806 checkpointów, Z.npy przez mmap) — raz na klasę."""

    @classmethod
    def setUpClass(cls):
        if not (W.HISTORIA / 'dane.npz').exists():
            raise AssertionError(f'brak cache historii {W.HISTORIA}')
        cls.okno = W.zbuduj_okno()
        cls.z = z = cls.okno.zakladka_laserow
        t0 = time.perf_counter()
        z._gotowe(str(W.HISTORIA))
        W.przetworz(10); QTest.qWait(300)
        METRYKI['wczytanie_historii_s'] = round(time.perf_counter() - t0, 2)

    @classmethod
    def tearDownClass(cls):
        W.zamknij_okno(cls.okno)

    def setUp(self):
        super().setUp()
        z = self.z
        z.bt_play.setChecked(False)
        z._przywroc_uklad()
        z.doki['s'].raise_(); z.doki['11'].raise_()
        W.przetworz(6)


class TestHistoriaEtykiety(BazaHistorii):
    def test_01_historia_wczytana(self):
        """Po _gotowe: z.h ustawione (K > 100, Z jako mmap float16), suwak osi czasu ma pełny zakres, napis czasu wypełniony."""
        z = self.z
        self.assertIsNotNone(z.h)
        self.assertGreater(z.h.K, 100)
        self.assertEqual(z.h.Z.dtype, np.float16)
        self.assertEqual(z.suwak.maximum(), (z.h.K - 1) * 100)
        self.assertIn('krok', z.lb_czas.text())

    def test_02_tryby_etykiety_narzedzia(self):
        """Przełączenie cb_etyk przez wszystkie EN.TRYBY zmienia h.narz.sum() (dotknięte ≥ środek ≥ pokrycie), aktualizuje lb_etyk
        i nie zgłasza wyjątków."""
        z, h = self.z, self.z.h
        self.assertIsNotNone(h.udzial, 'historia bez obrysu narzędzia: test etykiet nie ma sensu')
        self.assertEqual(z.cb_etyk.count(), len(EN.TRYBY))
        ile, napisy = {}, {}
        for i, (kod, _) in enumerate(EN.TRYBY):
            z.cb_etyk.setCurrentIndex(i); W.przetworz(3)
            self.assertEqual(z.cb_etyk.currentData(), kod)
            ile[kod] = int(h.narz.sum()); napisy[kod] = z.lb_etyk.text()
            self.assertIn(h.etykieta, z.lb_etyk.text(), 'lb_etyk nie pokazuje opisu etykiety')
            self.assertEqual(h.tryb_etykiety, kod)
        METRYKI['etykiety_narz'] = ile
        self.assertGreaterEqual(ile['dotkniete'], ile['srodek'], ile)
        self.assertGreaterEqual(ile['srodek'], ile['pokrycie'], ile)
        self.assertGreater(ile['dotkniete'], ile['pokrycie'], f'tryby nie różnią się liczbą łat: {ile}')
        self.assertEqual(len(set(napisy.values())), len(napisy), f'lb_etyk nie zmienia się między trybami: {napisy}')
        z.cb_etyk.setCurrentIndex(0); W.przetworz(3)

    def test_03_zaznacz_obszar(self):
        """_zaznacz_obszar(kod) wpisuje do ed_tok dokładnie patche danego obszaru (2 = źrenica, 3 = narzędzie), -1 czyści pole."""
        z, h = self.z, self.z.h
        z.cb_etyk.setCurrentIndex(0); W.przetworz(3)
        k = h.obszary(z.sp_oko_prog.value())
        for kod in (2, 3, 1, 0):
            z._zaznacz_obszar(kod); W.przetworz(2)
            wpisane = [int(x) for x in z.ed_tok.text().split(',') if x.strip()]
            self.assertEqual(wpisane, list(np.flatnonzero(k == kod)), f'obszar {kod}')
        z._zaznacz_obszar(2)
        self.assertGreater(len(z.ed_tok.text()), 0, 'źrenica: pusto (brak obrysu oka dla zdjęcia?)')
        self.assertEqual(set(z._wybrane_tok()) >= set(np.flatnonzero(k == 2)), True)
        z._zaznacz_obszar(-1)
        self.assertEqual(z.ed_tok.text(), '')

    def test_04_tryby_kolorow(self):
        """cb_kol_sciezki = 5 (obszary oka) i cb_kolor_cz = 3 działają: kolory z 4 klas oka, bez wyjątków; pozostałe tryby też."""
        z = self.z
        for i in range(z.cb_kol_sciezki.count()):
            z.cb_kol_sciezki.setCurrentIndex(i); W.przetworz(2)
        z.cb_kol_sciezki.setCurrentIndex(3); W.przetworz(2)         # „narzędzie”: _ks != (5, h), więc 5 wyśle kolory ponownie
        with mock.patch.object(z.es, 'ustaw_kolory_sciezek', wraps=z.es.ustaw_kolory_sciezek) as sp:
            z.cb_kol_sciezki.setCurrentIndex(5); W.przetworz(3)
        self.assertEqual(z._ks, (5, id(z.h)))
        K = z._kolory_tryb(3)                                       # tryb „obszary oka” (indeks 5 ścieżek i 3 cząsteczek)
        sp.assert_called()
        np.testing.assert_allclose(sp.call_args[0][0], K, err_msg='kolory ścieżek 5 nie trafiły do sfery')
        for i in range(z.cb_kolor_cz.count()):
            z.cb_kolor_cz.setCurrentIndex(i); W.przetworz(2)
        z.cb_kolor_cz.setCurrentIndex(3); W.przetworz(3)
        self.assertEqual(K.shape, (600, 4))
        kol = {tuple(np.round(r[:3], 3)) for r in K}
        self.assertLessEqual(len(kol), 4)
        self.assertGreaterEqual(len(kol), 2, 'obszary oka: wszystkie patche tego samego koloru')
        kod = z.h.obszary(z.sp_oko_prog.value())
        for c in range(4):                                          # jedna klasa oka = jeden kolor
            self.assertEqual(len({tuple(r[:3]) for r in K[kod == c]}), 1 if (kod == c).any() else 0)
        for i in (0, 1, 2, 3):
            self.assertEqual(z._kolory_tryb(i).shape, (600, 4))
        z.cb_kol_sciezki.setCurrentIndex(0); z.cb_kolor_cz.setCurrentIndex(0); W.przetworz(3)


class TestHistoriaWidocznosc(BazaHistorii):
    def test_01_okienko_w_karcie_nie_jest_widac(self):
        """Dok schowany w karcie za innym: widac() False (choć isVisible() bywa True); po wyniesieniu na wierzch widac() True, a tamten False."""
        z = self.z
        for ukryty, wierzch in (('12', '11'), ('R1', 's')):
            with self.subTest(ukryty=ukryty, wierzch=wierzch):
                z.doki[wierzch].raise_(); W.przetworz(6)
                self.assertTrue(z.widac(wierzch))
                self.assertFalse(z.widac(ukryty), f'{ukryty} schowany w karcie, a widac() True')
                METRYKI[f'isVisible_{ukryty}_za_{wierzch}'] = z.doki[ukryty].isVisible()
                z.doki[ukryty].raise_(); W.przetworz(6)
                self.assertTrue(z.widac(ukryty), f'{ukryty} wyniesiony na wierzch, a widac() False')
                self.assertFalse(z.widac(wierzch), f'{wierzch} przykryty, a widac() True')
                z.doki[wierzch].raise_(); W.przetworz(4)

    def test_02_zamkniete_i_odpiete(self):
        """Zamknięty dok: widac() False; odpięty dok otwarty: widac() True."""
        z = self.z
        z.doki['2'].close(); W.przetworz(4)
        self.assertFalse(z.widac('2'))
        z.doki['3'].setFloating(True); z.doki['3'].show(); W.przetworz(6)
        self.assertTrue(z.widac('3'))
        self.assertFalse(z.widac('nie_ma'))

    def test_03_ustaw_czas_pomija_ukryte_widoki_2d(self):
        """ustaw_czas nie woła e.ustaw_czas dla zamkniętego okienka (1), ale woła dla widocznych (2, 3, sfera); dla okienka w karcie
        pod spodem (R1 za sferą) też nie."""
        z = self.z
        z.doki['1'].close(); W.przetworz(4)
        self.assertFalse(z.widac('1')); self.assertTrue(z.widac('2'))
        with mock.patch.object(z.e1, 'ustaw_czas', wraps=z.e1.ustaw_czas) as s1, \
                mock.patch.object(z.e2, 'ustaw_czas', wraps=z.e2.ustaw_czas) as s2, \
                mock.patch.object(z.es, 'ustaw_czas', wraps=z.es.ustaw_czas) as ss:
            z.ustaw_czas(5.0); W.przetworz(2)
            s1.assert_not_called()
            s2.assert_called()
            ss.assert_called()
            z.doki['s'].setFloating(False)
            z.doki['R1'].raise_(); W.przetworz(6)
            self.assertFalse(z.widac('s'))
            ss.reset_mock(); s2.reset_mock()
            z.ustaw_czas(6.0)
            ss.assert_not_called()
            s2.assert_called()

    def test_04_odslonienie_odswieza_biezaca_chwile(self):
        """Po wyniesieniu schowanego okienka (sfera spod R1) z karty na wierzch dostaje ono bieżącą chwilę (_odslonieto), bez ruchu osi czasu."""
        z = self.z
        z.ustaw_czas(10.0)
        z.doki['R1'].raise_(); W.przetworz(6)
        self.assertFalse(z.widac('s'))
        with mock.patch.object(z.es, 'ustaw_czas', wraps=z.es.ustaw_czas) as ss:
            z.doki['s'].raise_()
            W.czekaj(lambda: ss.called, 3)
            self.assertTrue(ss.called, 'odsłonięta sfera nie dostała ustaw_czas')
            self.assertAlmostEqual(ss.call_args[0][0], 10.0)


class _NaEkranie:
    """Testy, które potrzebują prawdziwego rysowania OpenGL: w procesie z platformą offscreen (np. razem z test_moduly, który
    ją ustawia dla całego procesu) widoki GL nie dostają klatek, więc te testy są wtedy pomijane. Uruchamiaj test_gui osobno."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtGui import QGuiApplication
        W.aplikacja()
        if QGuiApplication.platformName() == 'offscreen':
            raise unittest.SkipTest('platforma offscreen (proces z test_moduly): uruchom testy.test_gui osobno')
        super().setUpClass()


class TestHistoriaGPU(_NaEkranie, BazaHistorii):
    def test_01_zh_trafia_na_karte_bit_w_bit(self):
        """Bufor GPU 'ZH' (całe Z historii, float16) jest wysłany w całości w ciągu 90 s, a odczyt kilku checkpointów z karty jest
        bitowo równy np.asarray(h.Z[k])."""
        from OpenGL import GL
        z, h = self.z, self.z.h
        self.assertTrue(z.widac('s'))
        t0 = time.perf_counter()
        ok = W.czekaj(lambda: z.es.duzy_gotowy('ZH'), 90, odswiez=z.vs.update)
        METRYKI['zh_na_karcie_s'] = round(time.perf_counter() - t0, 2)
        self.assertTrue(ok, 'ZH nie został w pełni wysłany na kartę w 90 s')
        W.przetworz(3)
        K, TOK, NK = h.Z.shape
        nb = TOK * NK * 2
        z.vs.makeCurrent()
        try:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, z.es._buf['ZH'])
            rozm = GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_BUFFER_SIZE)
            self.assertEqual(int(rozm), K * nb, 'rozmiar bufora ZH na karcie')
            for k in (0, 1, K // 2, K - 2, K - 1):
                with self.subTest(checkpoint=k):
                    out = np.zeros(nb, np.uint8)                 # bajty: PyOpenGL nie zna float16
                    GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, k * nb, nb, out)
                    ref = np.asarray(h.Z[k])
                    self.assertTrue(np.array_equal(out.view(np.uint16).reshape(TOK, NK), ref.view(np.uint16)), f'checkpoint {k}: różnice z GPU')
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        finally:
            z.vs.doneCurrent()


class TestHistoriaWydajnosc(_NaEkranie, BazaHistorii):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        W.czekaj(lambda: cls.z.es.duzy_gotowy('ZH'), 90, odswiez=cls.z.vs.update)    # wysyłka ZH nie fałszuje pomiarów
        W.przetworz(5)

    def test_01_brak_pelzania_ukladu(self):
        """60 wywołań ustaw_czas + processEvents nie zmienia rozmiarów widocznych widoków GL i generuje co najwyżej 2 zdarzenia
        LayoutRequest dla wewnętrznego QMainWindow (napisy co klatkę nie przeliczają układu)."""
        z = self.z

        class Licznik(QObject):
            n = 0

            def eventFilter(self, obj, e):
                if e.type() == QEvent.Type.LayoutRequest:
                    self.n += 1
                return False
        for t in np.linspace(0, 50, 8):                                     # rozgrzewka: pierwsze wywołania mogą budować zasoby
            z.ustaw_czas(float(t)); W.przetworz(2)
        QTest.qWait(200)
        widoki = [v for v in list(WidokGL._instancje) if not sip.isdeleted(v) and v.isVisible()]
        self.assertGreaterEqual(len(widoki), 4)
        rozm0 = {id(v): (v.width(), v.height()) for v in widoki}
        rozm = lambda k: (z.doki[k].width(), z.doki[k].height())
        dok0 = {k: rozm(k) for k in z.doki if z.doki[k].isVisible()}
        f = Licznik(); z.okna.installEventFilter(f)
        zmiany = []
        try:
            for i in range(60):
                z.ustaw_czas(100.0 + 7.3 * i)
                W.przetworz(1)
                for v in widoki:
                    if (v.width(), v.height()) != rozm0[id(v)]:
                        zmiany.append((i, id(v), rozm0[id(v)], (v.width(), v.height())))
        finally:
            z.okna.removeEventFilter(f)
        METRYKI['layout_request_w_60_klatkach'] = f.n
        self.assertEqual(zmiany[:3], [], 'rozmiar widoku GL zmienił się w trakcie ustaw_czas')
        self.assertEqual({k: rozm(k) for k in dok0}, dok0, 'rozmiary doków pełzną')
        self.assertLessEqual(f.n, 2, f'{f.n} zdarzeń LayoutRequest dla z.okna w 60 klatkach')

    def test_02_ustaw_czas_mediana(self):
        """Mediana czasu ustaw_czas < 10 ms w układzie domyślnym z wczytaną historią (liczby trafiają do raportu)."""
        z = self.z
        ts = np.random.default_rng(1).uniform(0, z.h.K - 1, 120)
        for t in ts[:10]:
            z.ustaw_czas(float(t)); W.przetworz(1)
        czasy = []
        for t in ts[10:]:
            t0 = time.perf_counter(); z.ustaw_czas(float(t)); czasy.append((time.perf_counter() - t0) * 1000)
            W.przetworz(1)
        med, p95, mx = (float(np.median(czasy)), float(np.percentile(czasy, 95)), float(max(czasy)))
        METRYKI['ustaw_czas_ms'] = dict(mediana=round(med, 2), p95=round(p95, 2), max=round(mx, 2))
        self.assertLess(med, 10.0, f'mediana ustaw_czas {med:.1f} ms (p95 {p95:.1f}, max {mx:.1f})')

    def test_03_odtwarzanie_klatki_sfery(self):
        """Odtwarzanie ▶ ze stałym tempem daje ≥ 20 rysowań sfery na sekundę przez 3 s (liczone przez opakowanie paintGL)."""
        z = self.z
        licznik = [0]
        oryg = WidokGL.paintGL

        def opakowany(self_):
            if self_ is z.vs:
                licznik[0] += 1
            return oryg(self_)
        z.cb_tempo.setCurrentIndex(1); z.sp_tempo.setValue(5.0)
        z.ch_petla.setChecked(True)
        z.ustaw_czas(0.0); W.przetworz(5); QTest.qWait(200)
        with mock.patch.object(WidokGL, 'paintGL', opakowany):
            licznik[0] = 0
            t0 = time.perf_counter(); t_start = z.t
            z.bt_play.setChecked(True)
            QTest.qWait(3000)
            dt = time.perf_counter() - t0
            z.bt_play.setChecked(False)
            n = licznik[0]; przesuniecie = z.t - t_start
        fps = n / dt
        METRYKI['odtwarzanie_sfera_fps'] = round(fps, 1)
        METRYKI['odtwarzanie_przesuniecie_ckpt'] = round(przesuniecie, 1)
        self.assertGreater(przesuniecie, 5, 'oś czasu prawie nie ruszyła podczas odtwarzania')
        self.assertGreaterEqual(fps, 20.0, f'{fps:.1f} klatek sfery/s w odtwarzaniu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
