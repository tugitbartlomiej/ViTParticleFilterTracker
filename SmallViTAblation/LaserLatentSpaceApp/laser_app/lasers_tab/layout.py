"""Okienka-doki zakładki: tworzenie, widoczność (liczymy tylko to, co widać), układy domyślne i gotowe, sloty, zapis układu w ustawieniach, panel parametrów i okienka ustawień pod prawym klikiem."""
from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import QByteArray, Qt, QTimer
from PyQt6.QtWidgets import QDockWidget, QLabel, QVBoxLayout, QWidget

from laser_app.gpu.planar_gl import PlanarMotion
from laser_app.lasers_tab.widgets import View2D
from laser_app.ui.dock_windows import bez_ukladu
from laser_app.ui.widgets import _staly_uklad


class LayoutMixin:
    """Okienka-doki zakładki: tworzenie, widoczność (liczymy tylko to, co widać), układy domyślne i gotowe, sloty, zapis układu w ustawieniach, panel parametrów i okienka ustawień pod prawym klikiem.
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    # ------------------------------------------------------------------ panel parametrów i ustawienia okienek pod prawym klikiem
    # okienko → sekcje panelu z jego ustawieniami (prawy klik w okienko wypożycza je do popupu); R* pyta rzuty_okienka
    SEKCJE_OKIENEK = {'1': ('plaskie',), '2': ('plaskie',), '3': ('plaskie',), 's': ('sfera', 'wybor'), '4': ('mapy',), '5': ('mapy',),
                      '11': ('pulapka',), '14': ('mapa_pul',), '12': ('dolina',), '13': ('mech',), 'p': ('przejscie',),
                      '6': ('dane',), '7': ('dane',), '8': ('dane',), '9': ('dane',), '10': ('dane',)}

    # strony panelu: (klucz, nazwa na liście, sekcje, dymek)
    STRONY = (('dane', 'Dane, etykiety', ('dane', 'etykiety'), 'Historia uczenia, zdjęcie, etykieta narzędzia i obszary oka'),
              ('zestaw', 'Kilka zdjęć', ('zestaw',), 'Kilka zdjęć naraz na sferze reflektorów (600 koloidów na zdjęcie)'),
              ('sfera', 'Sfera', ('wybor', 'sfera'), 'Sfera reflektorów, wybór cząsteczek i reflektorów'),
              ('pulapka', 'Pułapka (11)', ('pulapka',), 'Pułapka optyczna: ogniskowa wiązek wokół koloidu, obrys ogniska 3D'),
              ('mapa_pul', 'Mapa pułapek (14)', ('mapa_pul',), 'Pułapki wszystkich 600 cząsteczek: kształt pola, zera, wiry'),
              ('podkroki', 'Podkroki (P)', ('przejscie', 'zapis'), 'Okienko P (przejście przez sieć) i zapis przejść na dysk'),
              ('rzuty', 'Rzuty R0–R12', ('rzuty',), 'Rzuty z 192 wymiarów do 3D'),
              ('plaskie', 'Ruch 1–3', ('plaskie',), 'Rzut płaski: ruch globalny, studnie, drgania'),
              ('mapy', 'Mapy 4–5', ('mapy',), 'Atencja i mapy na zdjęciu'),
              ('dolina', 'Dolina (12)', ('dolina',), 'Dolina pewności i IoU'),
              ('mech', 'Mechanika (13)', ('mech',), 'Mechanika reflektora'),
              ('widok', 'Widok', ('stereo',), 'Stereo 3D'))

    def _z_napisem(self, widok, nazwa):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(widok, 1)
        lb = QLabel(''); lb.setStyleSheet('color:#ffd166; padding:2px 6px; background:#0b1014'); _staly_uklad(lb)
        v.addWidget(bez_ukladu(lb))                                   # napis zmieniany co klatkę nie przelicza układu okna
        setattr(self, nazwa, lb)
        return w

    def _okno_2d(self, klucz, tytul, tryb):
        v = View2D(); e = PlanarMotion(tryb); v.addItem(e)
        v.klik_px.connect(lambda x, y, v=v, e=e: self._klik_2d(v, e, x, y))
        w = self._z_napisem(v, f'lb{klucz}')
        self._dok(klucz, tytul, w)
        return v, e, getattr(self, f'lb{klucz}')

    def _dok(self, klucz, tytul, widget):
        d = QDockWidget(tytul, self.okna); d.setObjectName(f'dok_lasery_{klucz}'); d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        d.toggleViewAction().setText(tytul)
        from laser_app.ui.dock_windows import pasek_doku
        from laser_app.ui.settings_panel import podlacz_menu
        pasek_doku(d)                                   # ⛶: maksymalizacja na ekranie okienka (też dwuklik, F11)
        podlacz_menu(d, lambda pos, k=klucz: self.menu_okienka(k, pos))     # prawy klik: ustawienia tego okienka
        d.visibilityChanged.connect(lambda on: self._odslonieto() if on else None)
        self.doki[klucz] = d
        return d

    def widac(self, klucz):
        """Czy okienko naprawdę widać: otwarte i (odpięte albo z niepustym widocznym obszarem). QDockWidget.isVisible() jest
        prawdą także dla okienka w nieaktywnej karcie, więc samo isVisible() kazało liczyć i rysować schowane okienka."""
        d = self.doki.get(klucz)
        return d is not None and d.isVisible() and (d.isFloating() or not d.visibleRegion().isEmpty())

    def _odslonieto(self):
        """Okienko odsłonięte (karta, otwarcie, odpięcie): po chwili jedno odświeżenie bieżącej chwili (zbiorcze dla wielu naraz)."""
        if self.h is None:
            return
        if not hasattr(self, '_odsl_timer'):
            self._odsl_timer = QTimer(self); self._odsl_timer.setSingleShot(True); self._odsl_timer.setInterval(0)
            self._odsl_timer.timeout.connect(self._po_odslonieciu)
        self._odsl_timer.start()

    def _po_odslonieciu(self):
        if self.h is None:
            return
        if not self.timer.isActive():
            self.ustaw_czas(self.t)
        self._og_zlec()                                                  # pułapka czekała na widoczne okienko 11
        self._mp_zlec()                                                  # mapa pułapek czekała na widoczne okienko 14

    def _dok_sterowania(self, klucz, tytul, widget):
        """Okienko sterujące (parametry, podkroki): jak okienka widoków można je odpiąć i przesunąć, ale nie należy do self.doki
        (nie jest widokiem osi czasu)."""
        from laser_app.ui.dock_windows import ScrollWrapper
        d = QDockWidget(tytul, self.okna); d.setObjectName(f'dok_lasery_ster_{klucz}'); d.setWidget(ScrollWrapper(widget))
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        d.toggleViewAction().setText(tytul)
        from laser_app.ui.dock_windows import pasek_doku
        pasek_doku(d)
        if not hasattr(self, 'doki_ster'):
            self.doki_ster = {}
        self.doki_ster[klucz] = d
        return d

    def _buduj_panel(self):
        from laser_app.ui.settings_panel import ParameterPanel
        self._buduj_etykiety()
        self._buduj_wybor()
        self._buduj_zestaw()
        self.panel = ParameterPanel()
        self._strona_sekcji = {}
        for klucz, nazwa, sekcje, dymek in self.STRONY:
            self.panel.dodaj_strone(klucz, nazwa, [self._sek[s] for s in sekcje], dymek)
            for s in sekcje:
                self._strona_sekcji.setdefault(s, klucz)
        self.panel.setMinimumWidth(330)
        self.dok_panel = self._dok_sterowania('parametry', 'Parametry', self.panel)
        self.dok_panel.setToolTip('F2 chowa i pokazuje, ❐ odpina jako osobne okno. Prawy klik w okienko: jego ustawienia przy kursorze.')

    def menu_okienka(self, klucz, pos):
        """Prawy klik w okienko: popup z jego ustawieniami (sekcje panelu wypożyczone na czas otwarcia) i akcjami okienka."""
        from laser_app.ui.dock_windows import przelacz, przelacz_dok, resetuj_widoki, widoki_3d
        from laser_app.ui.gl_view import GLView
        from laser_app.ui.settings_panel import SettingsPopup
        d = self.doki.get(klucz) or self.doki_ster.get(klucz)
        if klucz.startswith('R') and hasattr(self.rzuty, 'sekcje_okienka'):
            sekcje = self.rzuty.sekcje_okienka(klucz)
            strona = 'rzuty'
        else:
            nazwy = self.SEKCJE_OKIENEK.get(klucz, ())
            sekcje = [self._sek[n] for n in nazwy]
            strona = self._strona_sekcji.get(nazwy[0]) if nazwy else None
        akcje = []
        if d is not None:
            akcje = [('⧉', 'Odepnij okienko jako osobne okno albo przypnij z powrotem.', lambda: d.setFloating(not d.isFloating())),
                     ('⛶', 'Maksymalizuj okienko na jego ekranie (drugi raz wraca).', lambda: przelacz_dok(d)),
                     ('🖵', 'Pełny ekran okienka (Esc wraca), np. stereo na monitorze 3D.', lambda: przelacz(d.widget())),
                     ('🗙', 'Zamknij okienko (wraca z menu „Okienka”).', d.close)]
            if widoki_3d(d):
                akcje[:0] = [('⟲', 'Resetuj widok 3D okienka (kamera sprzed obracania i przybliżania; też Home albo 0).',
                              lambda: resetuj_widoki(d))] + [
                    (o, f'Widok {opis} (w widoku: klawisz {o[-1]}{", Shift" if o.startswith("−") else ""}).',
                     lambda o=o: [v.ustaw_os(o) for v in widoki_3d(d)])
                    for o, opis in GLView.NAZWY_OSI.items() if o != 'izo']
        p = SettingsPopup.istniejacy(klucz, sekcje)
        if p is not None:                                                # to samo okienko drugi raz: przywołanie, bez przestawiania
            p.show(); p.raise_(); p.activateWindow()
            return
        tytul = d.windowTitle() if d is not None else klucz
        p = SettingsPopup(tytul.split(':')[0], sekcje, akcje,
                          na_panel=(lambda: (self.dok_panel.show(), self.dok_panel.raise_(), self.panel.pokaz(strona))) if strona else None)
        p.klucz = klucz
        from laser_app.ui.input_comfort import ulatw
        ulatw(p)
        p.pokaz_przy(pos)

    def _wykres(self, klucz, tytul, os_y):
        p = pg.PlotWidget(); p.setBackground('#0b1014'); p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel('left', os_y); p.setMouseEnabled(x=True, y=False)
        kur = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('w', width=1.5))
        p.addItem(kur)
        kur.sigPositionChanged.connect(lambda l: None if self._blokada else self.ustaw_czas(float(l.value())))
        p.scene().sigMouseClicked.connect(lambda ev, p=p: self._klik_wykres(p, ev))
        p.kursor = kur
        self._dok(klucz, tytul, p)
        return p

    def _uklad_domyslny(self):
        """Główne miejsce mają widoki: panel parametrów wąską kolumną po lewej (na całą wysokość), podkroki cienkim paskiem
        u góry, pod nim duża sfera (z rzutami R w kartach) i obok pułapka optyczna (z doliną, mechaniką i okienkiem P w kartach),
        na dole niższy wiersz: 1 | 2 | 3 | mapy 4–5 | wykresy 6–10."""
        d, s = self.doki, self.doki_ster
        A, Hz, Vt = Qt.DockWidgetArea, Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        o = self.okna
        o.setCorner(Qt.Corner.TopLeftCorner, A.LeftDockWidgetArea); o.setCorner(Qt.Corner.BottomLeftCorner, A.LeftDockWidgetArea)
        o.addDockWidget(A.LeftDockWidgetArea, s['parametry'])
        o.addDockWidget(A.TopDockWidgetArea, s['podkroki'], Vt)
        o.addDockWidget(A.TopDockWidgetArea, d['s'], Vt)
        o.splitDockWidget(d['s'], d['11'], Hz)
        for a, b in (('11', '14'), ('14', '12'), ('12', '13'), ('13', 'p')):
            o.tabifyDockWidget(d[a], d[b])
        d['11'].raise_()
        o.addDockWidget(A.BottomDockWidgetArea, d['1'], Hz)
        for a, b in (('1', '2'), ('2', '3'), ('3', '4'), ('4', '6')):
            o.splitDockWidget(d[a], d[b], Hz)
        o.tabifyDockWidget(d['4'], d['5']); d['4'].raise_()
        for a, b in (('6', '7'), ('7', '8'), ('8', '9'), ('9', '10')):
            o.tabifyDockWidget(d[a], d[b])
        d['6'].raise_()

    def showEvent(self, ev):
        """Przy pierwszym pokazaniu: rozmiary doków według prawdziwej wielkości okna (przed pokazaniem Qt je ignoruje
        i sfera, główny widok, zostawała paskiem szerokości kilku centymetrów)."""
        super().showEvent(ev)
        if not getattr(self, '_rozmiary_ok', False):
            self._rozmiary_ok = True
            QTimer.singleShot(0, self._rozmiary_domyslne)

    def _rozmiary_domyslne(self, zastosuj_zapis=True):
        if zastosuj_zapis:
            self._rozmiary_gotowe = True
            if getattr(self, '_uklad_ok', False):                   # zapisany układ był już zastosowany przed pokazaniem okna:
                self._zastosuj_uklad()                              # jeszcze raz, teraz z prawdziwą wielkością okna (rozmiary doków)
                return
        d, s = self.doki, self.doki_ster
        Vt, Hz = Qt.Orientation.Vertical, Qt.Orientation.Horizontal
        W, H = max(self.okna.width() - 440, 700), max(self.okna.height(), 600)
        self.okna.resizeDocks([s['parametry']], [440], Hz)
        self.okna.resizeDocks([s['podkroki'], d['s'], d['1']], [s['podkroki'].sizeHint().height(), int(H * 0.66), int(H * 0.30)], Vt)
        self.okna.resizeDocks([d['s'], d['11']], [int(W * 0.60), int(W * 0.40)], Hz)
        self.okna.resizeDocks([d[k] for k in ('1', '2', '3', '4', '6')], [W // 5] * 5, Hz)

    def _uklad_gotowy(self, nazwa):
        """Gotowe układy z menu „Okienka → Układy okienek” (zaczynają od domyślnego)."""
        self._przywroc_uklad()
        d, o = self.doki, self.okna
        Vt, Hz, A = Qt.Orientation.Vertical, Qt.Orientation.Horizontal, Qt.DockWidgetArea
        W, H = max(o.width() - 440, 700), max(o.height(), 600)
        if nazwa == 'pulapka':
            d['s'].raise_(); d['11'].raise_()
            o.resizeDocks([d['s'], d['11']], [W // 2, W // 2], Hz)
            o.resizeDocks([d['s'], d['1']], [int(H * 0.78), int(H * 0.2)], Vt)
        elif nazwa == 'rzuty':
            r = [k for k in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7') if k in d]
            for k in r:                                          # z kart sfery do osobnej kolumny po prawej
                o.removeDockWidget(d[k])
            o.addDockWidget(A.RightDockWidgetArea, d[r[0]])
            for a, b in zip(r, r[1:]):
                o.tabifyDockWidget(d[a], d[b])
            for k in r:
                d[k].show()
            d[r[0]].raise_(); d['s'].raise_()
            o.resizeDocks([d[r[0]]], [int(W * 0.42)], Hz)
            o.resizeDocks([d['s'], d['1']], [int(H * 0.75), int(H * 0.22)], Vt)
        elif nazwa == 'sfera':
            d['s'].raise_()
            o.resizeDocks([d['s'], d['11']], [int(W * 0.86), int(W * 0.14)], Hz)
            o.resizeDocks([d['s'], d['1']], [int(H * 0.86), int(H * 0.12)], Vt)

    def _slot(self, i, zapisz=False):
        """Ctrl+Shift+1..3 zapamiętuje bieżący układ okienek, Ctrl+1..3 go przywraca (sloty zapisują się w settings.json)."""
        if zapisz:
            self._sloty[i] = bytes(self.okna.saveState().toBase64()).decode('ascii')
            self._komunikat(f'Zapamiętano układ okienek {i} (Ctrl+{i} przywraca).')
        elif i in self._sloty:
            for d in list(self.doki.values()) + list(self.doki_ster.values()):
                d.setFloating(False)
            self.okna.restoreState(QByteArray.fromBase64(self._sloty[i].encode('ascii')))
        else:
            self._komunikat(f'Układ {i} nie jest zapamiętany: Ctrl+Shift+{i} zapisuje bieżący.')

    def _komunikat(self, tekst, ms=6000):
        okno = self.window()
        if hasattr(okno, 'statusBar'):
            okno.statusBar().showMessage(tekst, ms)

    def uklad_okienek(self):
        """Układ okienek (doki: położenie, odpięcie, rozmiary) i zapamiętane sloty, do settings.json (settings.stan)."""
        okno = self.window()
        return dict(okna=bytes(self.okna.saveState().toBase64()).decode('ascii'), wersja_ukladu=2,
                    sloty={str(k): v for k, v in self._sloty.items()}, zestaw=[list(z) for z in getattr(self, '_zestaw', [])],
                    okno=bytes(okno.saveGeometry().toBase64()).decode('ascii') if okno is not self else None)

    def ustaw_uklad_okienek(self, v):
        """Przywraca zapisany układ okienek od razu, także przed pierwszym pokazaniem okna: startują tylko okienka, które były
        otwarte i widoczne przy zapisie (widoki w zamkniętych i schowanych w kartach okienkach nie tworzą kontekstu OpenGL ani
        nie liczą). Po pokazaniu okna układ stosuje się jeszcze raz, żeby rozmiary doków pasowały do prawdziwej wielkości."""
        self._uklad_zapisany = v
        if isinstance(v, dict) and v.get('zestaw') and not self._zestaw:     # lista zdjęć zestawu (liczy się dopiero na żądanie)
            self._zestaw = [(str(a), str(b)) for a, b in v['zestaw']][:32]
            self._zestaw_odswiez_liste()
        self._zastosuj_uklad()

    def _zastosuj_uklad(self):
        v = getattr(self, '_uklad_zapisany', None)
        if not isinstance(v, dict) or v.get('wersja_ukladu') != 2:          # układ z dawnej zakładki (lewa kolumna): domyślny
            return
        try:
            self._sloty.update({int(k): s for k, s in (v.get('sloty') or {}).items()})
            okno = self.window()
            if v.get('okno') and okno is not self and not getattr(self, '_rozmiary_gotowe', False):
                okno.restoreGeometry(QByteArray.fromBase64(v['okno'].encode('ascii')))   # wielkość i ekran głównego okna
            self._uklad_ok = bool(self.okna.restoreState(QByteArray.fromBase64(v['okna'].encode('ascii'))))
        except Exception:                                                # noqa: BLE001 (stary albo uszkodzony zapis: układ domyślny)
            import traceback
            traceback.print_exc()

    def _przywroc_uklad(self):
        for d in list(self.doki.values()) + list(self.doki_ster.values()):
            d.setFloating(False); d.show()
        if self._stan_domyslny is not None:
            self.okna.restoreState(self._stan_domyslny)
            self._rozmiary_domyslne(zastosuj_zapis=False)
