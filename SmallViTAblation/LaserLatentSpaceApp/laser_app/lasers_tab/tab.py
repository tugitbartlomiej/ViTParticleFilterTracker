"""Zakładka „Lasery i koloidy”: układ z filmów 2, 12 i 16 (`outputs/porownanie_szklo_20261010/film_krysztal`) na żywo, na GPU.

Wybierasz historię checkpointów (katalog k<krok>.pt), zakres kroków i zdjęcie; `history.policz` liczy na GPU (raz, potem z cache)
latent 600 × 768, kierunki fc2, atencję 6 bloków, ramkę i pewność w każdym checkpoincie oraz rozkład ruchu na trzy składowe.
Okienka (każde można zamknąć, odpiąć jako osobne okno albo przesunąć; menu „Okienka”):
  1. Ruch globalny (układ inercjalny)          2. Odkształcenie lokalne: studnie i strzałki od adresów
  3. Drgania wokół centrów (zbliżenie)        Sfera reflektorów (768 reflektorów, wiązki, załamanie na sferze — film 12)
  4. Atencja 6 bloków (mozaika)               5. Mapa na zdjęciu (centra, drgania, atencja, natężenie reflektora)
  6. Wielkość trzech składowych ruchu         7. IoU    8. Pewność    9. Pary przeciwnych reflektorów
  10. Obrót igieł reflektorów (gradient, film 23); na sferze strzałki obrotu („Strzałki obrotu” w sekcji sfery)
  11. Pułapka optyczna (ogniskowa wiązek)     12. Dolina pewności i IoU wybranej cząsteczki (valley.py; też na sferze)
  13. Reflektor: mechanika (reflector_mechanics.py): odrzut wiązek i wykonany ruch igły, celownika i progu
  R1–R8. Rzuty z 192 wymiarów z miernikiem wierności (sekcja „Rzuty z 192 wymiarów”, projections_view.py, projections.py):
      mapa klas, siły, wiązki ważone, rama siły, rama Schura, wędrówka w czasie uczenia, 7 widoków Fano i panel bez rzutu
Wszystkie okienka pokazują tę samą chwilę osi czasu (suwak, odtwarzanie, klik w wykres). Rysują shadery (pakiet laser_app/gpu).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                             QMenu, QPushButton, QToolButton, QVBoxLayout, QWidget)

from laser_app.compute import history as hist
from laser_app.gpu import trap_map_gl
from laser_app.gpu.focal_field_gl import FocalField
from laser_app.gpu.image_maps_gl import ImageMaps
from laser_app.gpu.schematic_gl import SchematicGPU
from laser_app.gpu.sphere_gl import ReflectorSphere
from laser_app.gpu.valley_gl import ValleyGPU
from laser_app.lasers_tab.data_view import DataMixin
from laser_app.lasers_tab.image_set_view import ImageSetMixin
from laser_app.lasers_tab.labels_view import LabelsMixin
from laser_app.lasers_tab.layout import LayoutMixin
from laser_app.lasers_tab.mechanics_view import MechanicsMixin
from laser_app.lasers_tab.playback import PlaybackMixin
from laser_app.lasers_tab.shared import EP, pl, TLO
from laser_app.lasers_tab.substeps_view import SubstepsMixin
from laser_app.lasers_tab.trap_map_view import TrapMapMixin
from laser_app.lasers_tab.trap_view import TrapMixin
from laser_app.lasers_tab.valley_view import ValleyMixin
from laser_app.lasers_tab.widgets import ABSlider, View2D
from laser_app.lasers_tab.workers import MechanicsWorker, SubstepFieldWorker, ValleyWorker
from laser_app.ui import projections_view
from laser_app.ui.dock_windows import bez_ukladu
from laser_app.ui.gl_view import GLView
from laser_app.ui.widgets import CollapsibleSection, lista_liczb, NarrowCombo, _spin, _staly_uklad


class LasersTab(LayoutMixin, DataMixin, ImageSetMixin, LabelsMixin, SubstepsMixin, TrapMixin, TrapMapMixin, ValleyMixin, MechanicsMixin, PlaybackMixin, QWidget):
    """Zakładka „Lasery i koloidy”: okno aplikacji bez menu. Metody są pogrupowane w mixinach (pliki obok);
    tu: budowa (sekcje panelu i okienka), wspólne opcje rysowania i ustawienie chwili osi czasu we wszystkich widocznych okienkach."""

    def __init__(self, okno=None):
        super().__init__()
        self.okno = okno
        self.h: hist.History | None = None
        self.t = 0.0
        self.krok_odtw = 0.0
        self.watek = None
        self._blokada = False
        self._buduj()
        self._lista_historii()
        self._lista_zdjec()
        self.timer = QTimer(self); self.timer.setInterval(16); self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._klatka)
        QTimer.singleShot(50, self._wczytaj_z_cache)

    def _buduj(self):
        """Układ: u góry pasek osi czasu, pod nim okienka (doki). Parametry nie zajmują już stałej kolumny: są w okienku
        „Parametry” (lista sekcji + jedna strona naraz), które można odpiąć jak każde okienko albo schować (F2), a ustawienia
        konkretnego okienka otwiera prawy klik w nie (settings_panel.SettingsPopup). Sekcje budują się jak dawniej
        (Zwijana + QFormLayout) i trafiają do self._sek; strony panelu składa _buduj_panel."""
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 2, 2, 2); lay.setSpacing(2)
        self._sek = {}

        wiersz_st = QHBoxLayout()
        self.ch_stereo = QCheckBox('Stereo 3D (SBS)')
        self.ch_stereo.setToolTip('Przełącznik 2D ↔ 3D dla wszystkich widoków OpenGL (sfera, R1–R7, dolina…): lewe i prawe oko obok siebie,\n'
                                  'każde zgniecione do połowy szerokości (half side-by-side) dla monitora 3D, np. Samsung Odyssey 3D.\n'
                                  'Okno z widokiem rozwiń na cały ekran monitora 3D (⛶ albo F11). Napisy 3D są w stereo ukryte, a klikanie\n'
                                  'w cząsteczki w trybie stereo nie jest dokładne (wyłącz stereo do zaznaczania).')
        self.sp_stereo = QDoubleSpinBox(); self.sp_stereo.setRange(0.0, 0.2); self.sp_stereo.setSingleStep(0.005); self.sp_stereo.setDecimals(3)
        self.sp_stereo.setValue(GLView.stereo_odstep); self.sp_stereo.setPrefix('odstęp ')
        self.sp_stereo.setToolTip('Odstęp oczu jako ułamek odległości kamery od środka widoku (tam jest zerowy rozrzut). Większy = głębsza scena.')
        self.ch_stereo_zam = QCheckBox('zamień oczy'); self.ch_stereo_zam.setToolTip('Gdy głębia jest odwrócona (przód i tył zamienione).')
        self.ch_stereo.toggled.connect(lambda on: GLView.ustaw_stereo(wlacz=on))
        self.sp_stereo.valueChanged.connect(lambda v: GLView.ustaw_stereo(odstep=v))
        self.ch_stereo_zam.toggled.connect(lambda on: GLView.ustaw_stereo(zamiana=on))
        for w_ in (self.ch_stereo, self.sp_stereo, self.ch_stereo_zam):
            wiersz_st.addWidget(w_)
        wiersz_st.addStretch(1)
        from laser_app.ui import dock_windows
        self.ch_pelny_nakladki = QCheckBox('pełny ekran pozwala nakładkom (Odyssey 3D Hub: SBS → 3D)')
        self.ch_pelny_nakladki.setChecked(dock_windows.PELNY_DLA_NAKLADEK)
        self.ch_pelny_nakladki.setToolTip('Włączone: okienko w pełnym ekranie (🖵, F11) zajmuje dokładnie cały monitor (Odyssey 3D Hub widzi pełny ekran\n'
                                          'i pokazuje swoje okienko SBS → 3D), ale ma ramkę 1 px, więc sterownik OpenGL nie bierze go na wyłączność\n'
                                          'i okno Huba może się nad nim położyć, tak jak przy YouTube.\n'
                                          'Wyłączone: klasyczny pełny ekran bez ramki; sterownik może go wziąć na wyłączność i nic się nad nim nie pokaże.')
        self.ch_pelny_nakladki.toggled.connect(dock_windows.ustaw_pelny_dla_nakladek)
        g = CollapsibleSection('Stereo 3D (monitor 3D, wszystkie widoki)'); f = QFormLayout(g); f.addRow(wiersz_st)
        f.addRow(self.ch_pelny_nakladki)
        self._sek['stereo'] = g


        self._buduj_sek_dane()
        self._buduj_sek_rzuty()
        self._buduj_sek_plaskie()
        self._buduj_sek_sfera()
        self._buduj_sek_zapis()
        self._buduj_sek_pulapka()
        self._buduj_sek_mapa_pul()
        self._buduj_sek_mapy()
        self._buduj_sek_dolina()
        self._buduj_sek_mech()
        self._buduj_okienka()

    def _buduj_sek_rzuty(self):
        """Sekcja rzutów z 192 wymiarów (okienka R0–R12, projections_view)."""
        # --- rzuty z 192 wymiarów (okienka R1–R8, projections_view.py)
        self.rzuty = projections_view.ProjectionWindows(self)
        self._sek['rzuty'] = self.rzuty

    def _buduj_sek_plaskie(self):
        """Sekcja okienek 1–3 (rzut płaski: ruch globalny, studnie, drgania)."""
        # --- rzuty 2D
        g = CollapsibleSection('Okienka 1–3: rzut płaski (film 2)'); f = QFormLayout(g)
        self.sp_rozm = _spin(1, 20, 3.0, 0.5, 1); self.sp_jas = _spin(0.05, 5, 1.0, 0.1, 2); self.sp_sig = _spin(0.05, 2, 0.26, 0.02, 2)
        self.sp_strz = _spin(0.1, 5, 1.0, 0.1, 1); self.sp_slad = _spin(0, 20000, 200, 50); self.sp_zbl = _spin(2, 30, 7.2, 0.5, 1)
        self.ch_kamera = QCheckBox('kamera okienka 3 idzie za narzędziem'); self.ch_kamera.setChecked(True)
        self.ch_narz2 = QCheckBox('pierścienie na narzędziu'); self.ch_narz2.setChecked(True)
        for nazwa, w, dymek in (('Wielkość tokenów [px]', self.sp_rozm, ''), ('Jasność studni', self.sp_jas, ''),
                                ('Szerokość studni σ [kom.]', self.sp_sig, 'Gauss studni w komórkach siatki (film: 0,26).'),
                                ('Skala strzałek', self.sp_strz, 'Strzałka adres → centrum mnożona przez tę liczbę.'),
                                ('Ślad drgań [kroki]', self.sp_slad, 'Długość śladu atomu w okienku 3 w krokach uczenia (film 16: 200).'),
                                ('Zbliżenie okienka 3 [kom.]', self.sp_zbl, 'Szerokość kadru okienka 3 w komórkach siatki.')):
            w.setToolTip(dymek); f.addRow(nazwa, w); w.valueChanged.connect(self._opcje)
        f.addRow(self.ch_kamera); f.addRow(self.ch_narz2)
        self.ch_kamera.toggled.connect(self._opcje); self.ch_narz2.toggled.connect(self._opcje)
        self._sek['plaskie'] = g

    def _buduj_sek_sfera(self):
        """Sekcja sfery reflektorów: wiązki, koloidy, wektory, ścieżki, wygładzanie."""
        # --- sfera
        g = CollapsibleSection('Sfera reflektorów (film 12)'); f = QFormLayout(g)
        self.ch_swiatlo = QCheckBox('reflektory świecą (wiązki i linie uwagi)'); self.ch_swiatlo.setChecked(True)
        self.ch_swiatlo.setToolTip('Odznaczone: gasną wszystkie wiązki laserów i linie uwagi na sferze; wybór „Wiązki” zostaje zapamiętany.')
        self.ch_swiatlo.toggled.connect(self._opcje)
        f.addRow(self.ch_swiatlo)
        self.cb_wiazki = NarrowCombo(); self.cb_wiazki.addItems(['brak', 'wszystkie (768 × 600)', 'do wybranych cząsteczek', 'z wybranych reflektorów'])
        self.cb_wiazki.setCurrentIndex(1)
        self.sp_gain = _spin(0.0005, 20, 0.05, 0.01, 4); self.sp_prog = _spin(0, 5, 0.02, 0.01, 3)
        self.sp_gain.setToolTip('Jasność wiązki = z_jc / 2 × wzmocnienie (film 12: 0,05).')
        self.sp_eksp = _spin(0.05, 20, 1.0, 0.1, 2)
        self.sp_grub_w = _spin(0.5, 10, 1.0, 0.5, 1, ' px'); self.sp_grub_w.setToolTip('Grubość linii wiązek laserów i linii uwagi na sferze.')
        self.ch_kier = QCheckBox('wypadkowy kierunek wybranych reflektorów (średnica d_c)'); self.ch_kier.setChecked(True)
        self.ch_kier.setToolTip('Linia przez całą sferę wzdłuż d_c: kierunek, w który pchają wszystkie wiązki wybranych reflektorów '
                                '(tylko przy „Wiązki: z wybranych reflektorów”).')
        self.ch_sfera = QCheckBox(); self.ch_sfera.setChecked(True); self.ch_sfera.setToolTip('Pokaż siatkę szklanej sfery.')
        self.sp_eksp.setToolTip('Światło wiązek sumowane w buforze float i tonowane 1 − exp(−ekspozycja · światło), jak w filmie 12.')
        self.sp_prog.setToolTip('Wiązki ciemniejsze niż próg nie są rysowane (mniej szumu, szybciej).')
        self.cb_skala = NarrowCombo(); self.cb_skala.addItems(['chmura dopasowana do kroku', 'stała skala (z końca)'])
        self.sp_tok = _spin(0, 599, 0); self.sp_refl = _spin(0, 767, 0)
        bt_tok = QPushButton('narzędzie'); bt_tok.setToolTip('Cząsteczka w środku ramki narzędzia.'); bt_tok.clicked.connect(self._tok_narzedzia)
        bt_refl = QPushButton('najzmienniejszy'); bt_refl.setToolTip('Reflektor o największej wariancji natężeń na końcu.')
        bt_refl.clicked.connect(self._refl_pokaz)
        h1 = QHBoxLayout(); h1.addWidget(self.sp_tok); h1.addWidget(bt_tok)
        h2 = QHBoxLayout(); h2.addWidget(self.sp_refl); h2.addWidget(bt_refl)
        self.ed_tok = QLineEdit(); self.ed_refl = QLineEdit()
        self.ed_tok.setPlaceholderText('np. 12, 40, 100-105')
        self.ed_refl.setPlaceholderText('np. 51, 121, 300-310')
        self.ed_tok.setToolTip('Dodatkowe wybrane cząsteczki (oprócz tej powyżej): liczby po przecinku albo spacji, zakres a-b. '
                               'Działa w „Wiązki: do wybranych cząsteczek”, „Wektory dla: wybranych cząsteczek” i liniach uwagi.')
        self.ed_refl.setToolTip('Dodatkowe wybrane reflektory (oprócz tego powyżej): liczby po przecinku albo spacji, zakres a-b. '
                                'Wiązki „z wybranych reflektorów” świecą ze wszystkich, a mapa „natężenie wybranych reflektorów” '
                                'w okienku 5 pokazuje łączne oświetlenie Σ_c max(z_jc, 0).')
        for e_ in (self.ed_tok, self.ed_refl):
            e_.editingFinished.connect(self._opcje)
        self.sp_sfera = _spin(0, 1, 0.10, 0.02, 2); self.ch_refl = QCheckBox('pokaż reflektory'); self.ch_refl.setChecked(True)
        self.sp_rozm3 = _spin(1, 30, 5, 0.5, 1, ' px'); self.sp_rozm3.setToolTip('Wielkość punktu koloidu (tokenu) na ekranie.')
        self.sp_rozm_pul = _spin(1, 80, 13, 1, 1, ' px'); self.sp_rozm_pul.setToolTip('Średnica poświaty średniej skoków koloidu na ekranie.')
        self.sp_jas_pul = _spin(0.05, 3, 0.55, 0.05, 2); self.sp_jas_pul.setToolTip('Jasność poświaty średnich skoków.')
        self.sp_skala_chm = _spin(0.1, 1.8, 1.0, 0.05, 2, ' ×')
        self.sp_skala_chm.setToolTip('Rozmiar całej chmury koloidów i średnich skoków względem szklanej sfery (1 = jak film 12: 99 % chmury w 0,55 promienia). '
                                     'Wiązki, punkty załamania, wektory i drogi przeliczają się razem z chmurą.')
        self.cb_widok_s = NarrowCombo(); self.cb_widok_s.addItems(['koloidy (tokeny)', 'średnie skoków koloidów (±okno)', 'koloidy i średnie skoków'])
        self.cb_widok_s.setToolTip('Średnia skoków koloidu = jego średnie położenie z ± okna kroków (jak centra studni w filmie 2; dawniej „pułapka”), '
                                   'w tych samych współrzędnych sfery. Liczona z ruchu koloidu i z przyszłych kroków, więc NIE jest przyczyną ruchu. '
                                   'Pułapka optyczna z laserów: sekcja „Pułapka optyczna” i okienko 11. W trybie podkroków średnia skoków = stan cząsteczki.')
        WEK = ['brak', 'z kroku na krok', 'od kroku 0 (adres startu)', 'cała historia: droga od kroku 0']
        self.cb_wek_kol = NarrowCombo(); self.cb_wek_kol.addItems(WEK)
        self.cb_wek_pul = NarrowCombo(); self.cb_wek_pul.addItems(WEK)
        for c_ in (self.cb_wek_kol, self.cb_wek_pul):
            c_.setToolTip('Z kroku na krok: od położenia w poprzednim checkpoincie (w podkrokach: poprzednim podkroku) do teraz. '
                          'Od kroku 0: od położenia w pierwszym checkpoincie (adres startu) do teraz. Cała historia: droga przez '
                          'wszystkie checkpointy od 0 do teraz. Koloidy żółte, średnie skoków zielone.')
        self.cb_wek_dla = NarrowCombo(); self.cb_wek_dla.addItems(['wszystkich cząsteczek', 'narzędzia', 'wybranych cząsteczek'])
        self.cb_wek_dla.setCurrentIndex(1)
        self.sp_skala_wek = _spin(0.1, 50, 1.0, 0.5, 1, ' ×'); self.sp_skala_wek.setToolTip('Długość wektorów (droga zawsze w skali 1).')
        for nazwa, w in (('Wiązki', self.cb_wiazki), ('Grubość wiązek', self.sp_grub_w), ('Wzmocnienie', self.sp_gain), ('Ekspozycja światła', self.sp_eksp), ('Próg', self.sp_prog),
                         ('Skala chmury', self.cb_skala)):
            f.addRow(nazwa, w)
        self._wiersze_wyboru = (('Cząsteczka (patch)', h1), ('  + cząsteczki', self.ed_tok), ('Reflektor (neuron)', h2),
                                ('  + reflektory', self.ed_refl))                # osobna sekcja „wybor” (_buduj_wybor)
        self.cb_obrot = NarrowCombo(); self.cb_obrot.addItems(['brak', 'dryf netto (2000 kroków)', 'drganie (100 kroków)', 'dryf i drganie'])
        self.cb_obrot.setToolTip('Gdzie w układzie jest gradient (film 23). Przy każdym reflektorze igła = kierunek d_c, w który pchają jego wiązki; '
                                 'strzałka na końcu igły = dokąd i jak szybko igła się obraca w czasie uczenia. Kierunek = rzut obrotu na 3D, '
                                 'długość = prędkość kątowa v̂_c w 192 wymiarach (°/100 kroków, skala log). Dryf netto: z checkpointów odległych '
                                 'o ≥ 2000 kroków (kolor plasma, grot); drganie: o ≥ 100 kroków (biała kreska). To rzeczywisty krok uczenia '
                                 '(gradient Σ_j z_jc δ_j przez AdamW); weight decay tylko skraca v_c, nie obraca go. Wybrany reflektor grubo.')
        self.sp_dl_obr = _spin(0.2, 5, 1.0, 0.1, 1, ' ×'); self.sp_dl_obr.setToolTip('Długość igieł i strzałek obrotu.')
        h3 = QHBoxLayout(); h3.addWidget(self.cb_obrot, 1); h3.addWidget(self.sp_dl_obr)
        f.addRow('Strzałki obrotu', h3); f.addRow(self.ch_kier)
        self.cb_obrot.currentIndexChanged.connect(self._opcje); self.sp_dl_obr.valueChanged.connect(self._opcje)
        h4 = QHBoxLayout(); h4.addWidget(self.ch_sfera); h4.addWidget(self.sp_sfera, 1)
        self.ch_sfera.toggled.connect(self.sp_sfera.setEnabled)
        f.addRow('Sfera (krycie)', h4); f.addRow('Wielkość koloidów', self.sp_rozm3); f.addRow('Wielkość średnich skoków', self.sp_rozm_pul)
        f.addRow('Jasność średnich skoków', self.sp_jas_pul); f.addRow('Skala chmury w sferze', self.sp_skala_chm); f.addRow(self.ch_refl)
        for w_ in (self.sp_rozm_pul, self.sp_jas_pul, self.sp_skala_chm):
            w_.valueChanged.connect(self._opcje)
        f.addRow('Pokazuj', self.cb_widok_s); f.addRow('Wektory koloidów', self.cb_wek_kol); f.addRow('Wektory średnich skoków', self.cb_wek_pul)
        f.addRow('Wektory dla', self.cb_wek_dla); f.addRow('Skala wektorów', self.sp_skala_wek)
        self.cb_kol_sciezki = NarrowCombo()
        self.cb_kol_sciezki.addItems(['jednolity (koloidy żółte, średnie skoków zielone)', 'kolor patcha ze zdjęcia', 'położenie patcha na obrazie (siatka 30 × 20: barwa = kolumna, jasność = wiersz)', 'narzędzie',
                                      'odpływ od adresu (jak okienko 2, plasma)', 'obszary oka: narzędzie, źrenica, rogówka, tło'])
        self.cb_kol_sciezki.setToolTip('Kolor drogi i wektorów koloidów i średnich skoków. „Odpływ od adresu”: każdy punkt drogi i koniec wektora ma kolor '
                                       'według odległości od położenia w kroku 0 (adres), skala plasma jak strzałki w okienku 2: '
                                       'ciemny fiolet = przy adresie, żółty = 99. percentyl odpływu.')
        self.cb_wyg = NarrowCombo(); self.cb_wyg.addItems(['brak', 'średnia ważona krocząca', 'EMA', 'średnia krocząca + EMA'])
        self.cb_wyg.setToolTip('Wygładzanie średnich skoków wzdłuż checkpointów (tylko przeszłość, wagi rosnące ku teraźniejszości / wykładnicze). '
                               'Dotyczy drogi, wektorów i poświaty średnich skoków. Nakłada się na już symetryczną średnią ±okno, więc przyszłe kroki zostają w środku. Uwaga: odstępy checkpointów to 10 kroków do kroku 1000, potem 100.')
        self.sp_wyg_okno = _spin(2, 300, 10, 1, sufiks=' ckpt'); self.sp_wyg_okno.setToolTip('Okno średniej ważonej (wagi okno, okno−1, …, 1).')
        self.sp_wyg_alfa = _spin(0.005, 1.0, 0.2, 0.05, 3); self.sp_wyg_alfa.setToolTip('EMA: y = α·x + (1 − α)·y_poprzednie (na checkpoint).')
        self.sp_zanik = _spin(0, 70000, 0, 100, sufiks=' kroków'); self.sp_zanik.setSpecialValueText('bez zaniku')
        self.sp_zanik.setToolTip('Droga gaśnie z wiekiem: jasność exp(−3·wiek/zanik), wiek w krokach uczenia (w podkrokach: w podkrokach). 0 = bez zaniku.')
        f.addRow('Kolor ścieżek', self.cb_kol_sciezki); f.addRow('Wygładzanie średnich skoków', self.cb_wyg)
        f.addRow('Okno średniej', self.sp_wyg_okno); f.addRow('EMA α', self.sp_wyg_alfa); f.addRow('Zanik historii', self.sp_zanik)
        for w_ in (self.cb_kol_sciezki, self.cb_wyg):
            w_.currentIndexChanged.connect(self._opcje)
        for w_ in (self.sp_wyg_okno, self.sp_wyg_alfa, self.sp_zanik):
            w_.valueChanged.connect(self._opcje)
        for w_ in (self.cb_widok_s, self.cb_wek_kol, self.cb_wek_pul, self.cb_wek_dla):
            w_.currentIndexChanged.connect(self._opcje)
        self.sp_skala_wek.valueChanged.connect(self._opcje)
        for w in (self.cb_wiazki, self.cb_skala):
            w.currentIndexChanged.connect(self._opcje)
        for w in (self.sp_gain, self.sp_eksp, self.sp_prog, self.sp_tok, self.sp_refl, self.sp_sfera, self.sp_rozm3, self.sp_grub_w):
            w.valueChanged.connect(self._opcje)
        for w in (self.ch_kier, self.ch_sfera):
            w.toggled.connect(self._opcje)
        self.ch_refl.toggled.connect(self._opcje)
        self._sek['sfera'] = g

    def _buduj_sek_mapy(self):
        """Sekcja okienek 4–5: mapy na zdjęciu."""
        # --- mapy
        g = CollapsibleSection('Okienka 4–5: mapy na zdjęciu'); f = QFormLayout(g)
        self.cb_mapa = NarrowCombo()
        self.cb_mapa.addItems(['gdzie centra odpłynęły najdalej', 'drgania (token od centrum)', 'średnia atencja 6 bloków']
                              + [f'atencja blok {b + 1}' for b in range(6)] + ['łączne natężenie wybranych reflektorów Σ z_jc'])
        self.sp_krycie = _spin(0, 1, 0.6, 0.05, 2); self.ch_pierw = QCheckBox('atencja: pierwiastek (jak film 16)'); self.ch_pierw.setChecked(True)
        self.ch_rm = QCheckBox('ramka modelu (błękitna)'); self.ch_rm.setChecked(True)
        self.ch_rg = QCheckBox('ramka prawdziwa (biała)'); self.ch_rg.setChecked(True)
        f.addRow('Mapa w okienku 5', self.cb_mapa); f.addRow('Krycie mapy', self.sp_krycie)
        f.addRow(self.ch_pierw); f.addRow(self.ch_rm); f.addRow(self.ch_rg)
        self.cb_mapa.currentIndexChanged.connect(self._opcje); self.sp_krycie.valueChanged.connect(self._opcje)
        for w in (self.ch_pierw, self.ch_rm, self.ch_rg):
            w.toggled.connect(self._opcje)
        self._sek['mapy'] = g

    def _buduj_okienka(self):
        """Pasek osi czasu, okienka-doki z widokami (1–14, sfera, R0–R12, P), panel i menu „Okienka”."""
        # --- pasek osi czasu
        os_ = QWidget(); ho = QHBoxLayout(os_); ho.setContentsMargins(4, 2, 4, 2)
        self.bt_okna = QToolButton(); self.bt_okna.setText('Okienka ▾'); self.bt_okna.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.bt_start = QPushButton('⏮'); self.bt_play = QPushButton('▶'); self.bt_play.setCheckable(True)
        self.bt_start.setToolTip('Na początek osi czasu.'); self.bt_play.setToolTip('Odtwarzanie / pauza (spacja).')
        for b_ in (self.bt_start, self.bt_play):
            b_.setMaximumWidth(48)                                           # ikony: miejsce zostaje dla licznika czasu
        self.suwak = ABSlider(); self.suwak.setRange(0, 0)
        self.cb_tempo = NarrowCombo(); self.cb_tempo.addItems(['tempo jak w filmie 16 (wolno na początku)', 'stałe: checkpointy na sekundę'])
        self.cb_tempo.setToolTip('Film 16: v = min(0,8 + krok/250, 150) kroków na klatkę; stałe: tyle checkpointów na sekundę.')
        self.sp_tempo = _spin(0.05, 200, 1.0, 0.25, 2, ' ×'); self.sp_tempo.setToolTip('Mnożnik tempa (albo checkpointy/s w trybie stałym).')
        self.lb_czas = QLabel('—'); _staly_uklad(self.lb_czas)               # bez sztywnego minimum: dzieli miejsce z suwakiem
        f_ = self.lb_czas.font(); f_.setPointSize(f_.pointSize() + 2); self.lb_czas.setFont(f_)
        self.ch_petla = QCheckBox('pętla'); self.ch_petla.setToolTip('Po dojściu do końca uczenia odtwarzanie wraca do kroku 0 (także z podkrokami).')
        self.ab = [None, None]                                               # punkty A i B (indeks checkpointu, None = brak)
        self.bt_a = QPushButton('A'); self.bt_b = QPushButton('B'); self.bt_ab0 = QPushButton('✕AB')
        self.bt_a.setToolTip('Punkt A = bieżący krok: odtwarzanie zaczyna się od A (z pętlą: wraca do A). Kolor błękitny na suwaku.')
        self.bt_b.setToolTip('Punkt B = bieżący krok: odtwarzanie kończy się w B (z pętlą: skacze do A). Kolor pomarańczowy na suwaku.')
        self.bt_ab0.setToolTip('Usuń punkty A i B (cała oś czasu).')
        for b_ in (self.bt_a, self.bt_b, self.bt_ab0):
            b_.setMaximumWidth(44)
        self.bt_a.clicked.connect(lambda: self._ustaw_ab(0)); self.bt_b.clicked.connect(lambda: self._ustaw_ab(1))
        self.bt_ab0.clicked.connect(lambda: self._ustaw_ab(None))
        self.ch_petla.toggled.connect(lambda *_: self._ustaw_ab(-1))
        for w in (self.bt_okna, self.bt_start, self.bt_play, self.ch_petla, self.bt_a, self.bt_b, self.bt_ab0):
            ho.addWidget(w)
        ho.addWidget(self.suwak, 3); ho.addWidget(self.cb_tempo); ho.addWidget(self.sp_tempo); ho.addWidget(bez_ukladu(self.lb_czas), 2)
        self.suwak.valueChanged.connect(lambda v: None if self._blokada else self.ustaw_czas(v / 100.0))
        self.bt_play.toggled.connect(self._graj); self.bt_start.clicked.connect(lambda: self.ustaw_czas(self._zakres_ab()[0]))

        # --- okienka (doki)
        self.okna = QMainWindow(); self.okna.setDockNestingEnabled(True)
        self.okna.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks
                                 | QMainWindow.DockOption.AllowTabbedDocks)
        self.layout().addWidget(os_); self.layout().addWidget(self.okna, 1)
        self._pl = self.layout()
        self.doki = {}
        self.v1, self.e1, self.lb1 = self._okno_2d('1', '1. Ruch globalny (układ inercjalny): cała sieć naraz', 'globalny')
        self.v2, self.e2, self.lb2 = self._okno_2d('2', '2. Odkształcenie lokalne: studnie odpływają od adresów', 'lokalny')
        self.v3, self.e3, self.lb3 = self._okno_2d('3', '3. Drgania: tokeny wokół centrów studni', 'drgania')
        self.vs = GLView(); self.vs.setBackgroundColor(TLO); self.vs.setCameraPosition(distance=135, elevation=24, azimuth=-10)
        self.es = ReflectorSphere(hist.R_SFERY); self.vs.addItem(self.es)
        self.vs.klik_px.connect(self._klik_sfera)
        self._dok('s', 'Sfera reflektorów: 768 neuronów, 600 wiązek każdy (film 12)', self._z_napisem(self.vs, 'lbs'))
        self.v4 = View2D(); self.e4 = ImageMaps(); self.v4.addItem(self.e4)
        self._dok('4', '4. Atencja 6 bloków (ile uwagi dostaje patch)', self._z_napisem(self.v4, 'lb4'))
        self.v5 = View2D(); self.e5 = ImageMaps(); self.v5.addItem(self.e5)
        self._dok('5', '5. Mapa na zdjęciu', self._z_napisem(self.v5, 'lb5'))
        self.v4.kadruj(47, 21, 98, 48); self.v5.kadruj(15, 10, 32, 22)
        self.e4.kafelki = [dict(mapa=b, x=(b % 3) * 32, y=-(b // 3) * 24, norma=0, gamma=0.5, lut=2, alfa=0.62) for b in range(6)]
        self.p6 = self._wykres('6', '6. Wielkość trzech składowych ruchu', 'średnio [komórki siatki]')
        self.p7 = self._wykres('7', '7. IoU z checkpointu na checkpoint', 'IoU')
        self.p8 = self._wykres('8', '8. Pewność z checkpointu na checkpoint', 'pewność')
        self.p9 = self._wykres('9', '9. Pary przeciwnych reflektorów (cos < −0,5)', 'odsetek kanałów')
        self.p10 = self._wykres('10', '10. Obrót igieł reflektorów (gradient): mediana', '°/100 kroków')
        self.p10.setLogMode(y=True)
        # okienko 11: pułapka optyczna i koloid skaczący w polu jej wiązek (focal_trap.py); ta sama pułapka może być na dużej sferze
        self.es_og = FocalField(); self.es_og.setVisible(False); self.vs.addItem(self.es_og)
        self.vog = GLView(); self.vog.setBackgroundColor(TLO); self.vog.setCameraPosition(distance=30, elevation=22, azimuth=-35)
        self.eog = FocalField(); self.vog.addItem(self.eog)
        self.eog_pp = FocalField(); self.eog_pp.opcje.update(tylko_pole=True, slad=0)   # pole w komórce w podkrokach
        self.eog_pp.setVisible(False); self.eog_pp.setDepthValue(1); self.vog.addItem(self.eog_pp)   # nad wiązkami
        self._pp_pole = None; self._pp_klucz = None; self._pp_cache = {}
        self.watek_pp = SubstepFieldWorker(); self.watek_pp.gotowe.connect(self._pole_pod_gotowe)
        self._dok('11', '11. Pułapka optyczna: koloid skacze w polu wiązek (blok 6)', self._z_napisem(self.vog, 'lb11'))
        # okienko 12: dolina pewności i IoU wybranej cząsteczki (valley.py); ta sama dolina wokół cząsteczki na dużej sferze
        self.es_dol = ValleyGPU(); self.es_dol.setVisible(False); self.es_dol.setDepthValue(2); self.vs.addItem(self.es_dol)
        self.vdol = GLView(); self.vdol.setBackgroundColor(TLO); self.vdol.setCameraPosition(distance=42, elevation=20, azimuth=-40)
        self.edol = ValleyGPU(); self.vdol.addItem(self.edol)
        self.watek_dol = ValleyWorker(); self.watek_dol.gotowe.connect(self._dol_gotowe)
        self._dol_obrot = QTimer(self); self._dol_obrot.setInterval(33)
        self._dol_obrot.timeout.connect(lambda: self.vdol.orbit(0.25, 0) if self.widac('12') else None)
        self._dok('12', '12. Dolina pewności i IoU: gdzie cząsteczka daje dobrą odpowiedź (192 wymiary, przekrój 3D)',
                  self._z_napisem(self.vdol, 'lb12'))
        self.doki['12'].visibilityChanged.connect(lambda on: self._dol_zlec() if on else None)
        # okienko 13: rysunek techniczny wybranego reflektora (reflector_mechanics.py, reflector_drawing.py)
        self.vmech = GLView(); self.vmech.setBackgroundColor(TLO)
        self.vmech.setCameraPosition(distance=50, elevation=36, azimuth=-62)
        from PyQt6.QtGui import QVector3D
        self.vmech.opts['center'] = QVector3D(0.0, 0.0, 0.5)
        self.emech = SchematicGPU(); self.vmech.addItem(self.emech)
        self._mech_napisy = []
        self.watek_mech = MechanicsWorker(); self.watek_mech.gotowe.connect(self._mech_gotowe)
        self._dok('13', '13. Reflektor: mechanika — odrzut wiązek, sprężyny i wykonany ruch (blok 6)', self._z_napisem(self.vmech, 'lb13'))
        self.doki['13'].visibilityChanged.connect(lambda on: self._mech_zlec() if on else None)
        # okienko 14: mapa pułapek wszystkich cząsteczek (trap_map.py); ta sama mapa może być na dużej sferze
        self.vmp = GLView(); self.vmp.setBackgroundColor(TLO); self.vmp.setCameraPosition(distance=95, elevation=18, azimuth=-50)
        self.gl_mp = trap_map_gl.TrapMapGL(); self.gl_mp.dodaj_do(self.vmp); self.gl_mp.widoczna(False)
        self.gl_mp_s = trap_map_gl.TrapMapGL(); self.gl_mp_s.dodaj_do(self.vs); self.gl_mp_s.widoczna(False)
        self._dok('14', '14. Mapa pułapek: wszystkie 600 cząsteczek, kształt pola i zera pola w komórkach (blok 6)',
                  self._z_napisem(self.vmp, 'lb14'))
        self.doki['14'].visibilityChanged.connect(lambda on: self._mp_zlec() if on else None)
        self.rzuty.buduj_okienka()                                      # okienka R1–R8: rzuty z 192 wymiarów
        self._buduj_przejscie()
        self._buduj_panel()
        self._uklad_domyslny()
        self.rzuty.uklad_domyslny()
        self.menu_okienek = m = QMenu('Okienka', self)
        m.setToolTipsVisible(True)
        for d in self.doki_ster.values():
            m.addAction(d.toggleViewAction())
        m.addSeparator()
        grupy = (('Ruch i mapy (1–10)', ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')),
                 ('Sfera, pułapka, dolina, mechanika', ('s', '11', '14', '12', '13', 'p')),
                 ('Rzuty z 192 wymiarów (R0–R12)', tuple(k for k in self.doki if k.startswith('R'))))
        for nazwa, klucze in grupy:
            pm = m.addMenu(nazwa)
            for k in klucze:
                if k in self.doki:
                    pm.addAction(self.doki[k].toggleViewAction())
        m.addSeparator()
        a = QAction('Resetuj wszystkie widoki 3D', self); a.setShortcut('Ctrl+Shift+R')
        a.setToolTip('Kamera każdego widoku OpenGL wraca sprzed obracania, przesuwania i przybliżania (jedno okienko: ⟲ albo Home).')
        a.triggered.connect(GLView.resetuj_wszystkie); m.addAction(a); self.addAction(a)
        from laser_app.ui.dock_windows import menu_osi
        mo = menu_osi(self); mo.setTitle('Wszystkie widoki 3D wzdłuż osi X, Y, Z'); m.addMenu(mo)
        mu = m.addMenu('Układy okienek')
        for nazwa, fn in (('Domyślny: sfera duża, rzuty i wykresy w kartach', self._przywroc_uklad),
                          ('Sfera + pułapka optyczna obok siebie', lambda: self._uklad_gotowy('pulapka')),
                          ('Sfera + rzuty R1–R7 obok siebie', lambda: self._uklad_gotowy('rzuty')),
                          ('Sfera na całe okno (reszta w kartach)', lambda: self._uklad_gotowy('sfera'))):
            a = QAction(nazwa, self); a.triggered.connect(fn); mu.addAction(a)
        mu.addSeparator()
        self._sloty = {}
        for i in (1, 2, 3):
            a = QAction(f'Zapamiętaj bieżący układ jako {i}', self); a.setShortcut(f'Ctrl+Shift+{i}')
            a.triggered.connect(lambda _=False, i=i: self._slot(i, zapisz=True)); mu.addAction(a); self.addAction(a)
        for i in (1, 2, 3):
            a = QAction(f'Przywróć układ {i}', self); a.setShortcut(f'Ctrl+{i}')
            a.triggered.connect(lambda _=False, i=i: self._slot(i)); mu.addAction(a); self.addAction(a)
        self.bt_okna.setMenu(m)
        self._stan_domyslny = self.okna.saveState()                     # układ domyślny bez rozmiarów (rozmiary: _rozmiary_domyslne)
        # listy to NarrowCombo: długie pozycje nie wymuszają szerokości panelu ani okienek (wcześniej pasek „Przejścia”
        # trzymał 1700 px i ściskał sferę do 80 px); w rozwiniętej liście tekst jest cały
        for cb in self.findChildren(QComboBox):
            if not cb.toolTip():                                         # bez własnego dymka: dymek = pełny wybrany tekst
                cb.setToolTip(cb.currentText())
                cb.currentTextChanged.connect(cb.setToolTip)
        from laser_app.ui import tooltips
        from laser_app.ui import parameter_groups
        self._dymki_zmienione = tooltips.zastosuj(self)   # dymki: co robi każdy parametr (także po najechaniu na nazwę wiersza)
        parameter_groups.zastosuj(self)           # nagłówki podgrup w długich sekcjach (pułapka, sfera, dane, …)

    # ------------------------------------------------------------------ opcje
    def _opcje(self, *_):
        for e, v in ((self.e1, self.v1), (self.e2, self.v2), (self.e3, self.v3)):
            e.opcje.update(rozm=self.sp_rozm.value(), jas=self.sp_jas.value(), sig=self.sp_sig.value(), strz=self.sp_strz.value(),
                           narz=self.ch_narz2.isChecked(), tok=self.sp_tok.value() if self.h else -1,
                           pxc=v.px_na_komorke())
            e.update()
        o = self.es.opcje
        self.es.ustaw_wybor(self._wybrane_refl(), [self._tok_sfery(j) for j in self._wybrane_tok()])   # numery na sferze
        swiatlo = self.ch_swiatlo.isChecked()
        o.update(wiazki=self.cb_wiazki.currentIndex() if swiatlo else 0, gain=self.sp_gain.value(), prog=self.sp_prog.value(), tok=self._tok_sfery(self.sp_tok.value()),
                 refl=self.sp_refl.value(), sfera=self.sp_sfera.value() if self.ch_sfera.isChecked() else 0.0,
                 grub_w=self.sp_grub_w.value(), kierunek=self.ch_kier.isChecked(), ekspozycja=self.sp_eksp.value(),
                 widok=self.cb_widok_s.currentIndex(), wek_kol=self.cb_wek_kol.currentIndex(), wek_pul=self.cb_wek_pul.currentIndex(),
                 filtr=(0, 1, 2)[self.cb_wek_dla.currentIndex()], skala_wek=self.sp_skala_wek.value(),
                 kol_sciezki=self.cb_kol_sciezki.currentIndex(), zanik=float(self.sp_zanik.value()),
                 sily=(self.cb_sily.currentIndex() if hasattr(self, 'cb_sily') else 1) if swiatlo else 0, gain_a=self.sp_gain_a.value() if hasattr(self, 'sp_gain_a') else 1.0, reflektory=self.ch_refl.isChecked(), rozm=self.sp_rozm3.value(),
                 rozm_pul=self.sp_rozm_pul.value(), jas_pul=self.sp_jas_pul.value(), skala_chmury=self.sp_skala_chm.value(),
                 obrot=self.cb_obrot.currentIndex(), dl_obr=self.sp_dl_obr.value())
        if self.h is not None:
            wyg = (self.cb_wyg.currentIndex(), self.sp_wyg_okno.value(), self.sp_wyg_alfa.value())
            if getattr(self.es, '_wyg', None) != wyg:
                self.es.ustaw_wygladzanie(*wyg)
            ks = self.cb_kol_sciezki.currentIndex()
            if getattr(self, '_ks', None) != (ks, id(self.h)) and ks in (1, 2, 3, 5):
                self._ks = (ks, id(self.h)); self.es.ustaw_kolory_sciezek(self._kolory_sfery(ks - 1 if ks < 4 else 3))
        if self.h is not None and getattr(self, '_skala', None) != self.cb_skala.currentIndex():
            self._skala = self.cb_skala.currentIndex(); self.es.ustaw_skale(self._h_sfery(), self._skala == 1)
            if getattr(self, '_og', None) is not None:                     # pułapka w tej samej skali co sfera
                self._og_zastosuj(self._og_klucz, self._og, kamera=False)
        if hasattr(self, 'eog'):
            self.eog.opcje['skala_chmury'] = self.es_og.opcje['skala_chmury'] = self.sp_skala_chm.value()
        if getattr(self, '_dol', None) is not None:
            self._dol_przeksztalcenie()
        self.es.update()
        al = self.sp_krycie.value(); gam = 0.5 if self.ch_pierw.isChecked() else 1.0
        for k in self.e4.kafelki:
            k.update(alfa=max(al, 0.05), gamma=gam)
        mapa = self.cb_mapa.currentIndex()
        if self.h is not None:
            kody = {0: dict(mapa=7, norma=1, vmax=self.h.LOK99, gamma=1.0, lut=1), 1: dict(mapa=8, norma=1, vmax=self.h.VIB97, gamma=1.0, lut=0),
                    2: dict(mapa=6, norma=0, gamma=gam, lut=2),
                    9: dict(mapa=9, norma=1, gamma=1.0, lut=3, vmax=self._vmax_oswietlenia())}
            k = kody.get(mapa, dict(mapa=mapa - 3, norma=0, gamma=gam, lut=2))
            self.e5.kafelki = [dict(x=0, y=0, alfa=al, **k)]
        for e in (self.e4, self.e5):
            e.ramki = dict(model=self.ch_rm.isChecked(), gt=self.ch_rg.isChecked()); e.update()
        if hasattr(self, 'ep'):
            self._opcje_przejscia()
        if self.h is not None:
            self.ustaw_czas(self.t)

    def _wybrane_refl(self):
        return lista_liczb(f'{self.sp_refl.value()} {self.ed_refl.text()}', 768)

    def _wybrane_tok(self):
        return lista_liczb(f'{self.sp_tok.value()} {self.ed_tok.text()}', 600)

    def _vmax_oswietlenia(self):
        """Skala mapy łącznego oświetlenia: 99,5. percentyl sumy z wybranych reflektorów, po wszystkich checkpointach (co ~20.)."""
        c = tuple(self._wybrane_refl())
        if getattr(self, '_vmax_osw', (None,))[0] != (id(self.h), c):
            ks = range(0, self.h.K, max(1, self.h.K // 20))
            v = np.concatenate([ImageMaps.oswietlenie(self.h.Z[k], c) for k in ks])
            self._vmax_osw = ((id(self.h), c), max(float(np.percentile(v, 99.5)), 1e-6))
        return self._vmax_osw[1]

    def _tok_narzedzia(self):
        if self.h is not None:
            self.sp_tok.setValue(self.h.fok)

    def _refl_pokaz(self):
        if self.h is not None:
            self.sp_refl.setValue(self.h.refl_pokaz)

    # ------------------------------------------------------------------ czas
    def ustaw_czas(self, t):
        h = self.h
        if h is None:
            return
        self.t = t = float(np.clip(t, 0, h.K - 1))
        tryb_pod = self.ch_pod.isChecked() and getattr(self, '_pp', None) is not None
        # tylko okienka, które naprawdę widać (okienko w nieaktywnej karcie albo zamknięte nie liczy ani nie wysyła nic na GPU;
        # po odsłonięciu dostaje bieżącą chwilę w _odslonieto)
        para = (('4', self.e4), ('5', self.e5)) if tryb_pod else (('1', self.e1), ('2', self.e2), ('3', self.e3), ('4', self.e4),
                                                                  ('5', self.e5), ('s', self.es))
        for k, e in para:
            if self.widac(k):
                e.ustaw_czas(t)
        if self.es.opcje['wiazki'] and not tryb_pod and self.widac('s'):
            self.es.ustaw_natezenia(self._h_sfery(), t)
        if self.e5.kafelki and self.e5.kafelki[0]['mapa'] == 9 and self.widac('5'):
            self.e5.ustaw_reflektor(h, t, self._wybrane_refl())
        self._blokada = True
        self.suwak.setValue(int(round(t * 100)))
        for k, p in (('6', self.p6), ('7', self.p7), ('8', self.p8), ('9', self.p9), ('10', self.p10)):
            if self.widac(k):
                p.kursor.setValue(t)
        self._blokada = False
        krok = h.krok(t)
        if not self.bt_play.isChecked():
            self.krok_odtw = krok
        iou_ = h.interp(np.nan_to_num(h.iou), t); pew = h.interp(h.pew, t)
        self.lb_czas.setText(f'krok {krok:,.0f}   epoka {pl(krok / EP)}   IoU {pl(iou_)}   pewność {pl(pew)}'.replace(',', ' ', 1)
                             if krok >= 1000 else f'krok {krok:.0f}   epoka {pl(krok / EP)}   IoU {pl(iou_)}   pewność {pl(pew)}')
        par = h.interp(h.par, t)
        self.lb1.setText(f'przesunięcie {pl(float(np.hypot(*par[:2])))} kom.   obrót {par[2]:+.1f}°   skala ×{pl(par[3], 3)}')
        self.lb2.setText(f'średnie przesunięcie centrów {pl(float(h.interp(h.lok_m, t)))} kom.;  narzędzie/reszta ×{pl(float(h.interp(h.narz_lok, t)))}')
        self.lb3.setText(f'średnie drgania {pl(float(h.interp(h.vib_m, t)))} kom.   (centrum = średnia ±{int(h.okno)} kroków)')
        self._zlec_podkroki()
        obr = ''
        if self.cb_obrot.currentIndex() and getattr(h, 'obr_dryf', None) is not None:
            i_ = int(round(t)); c_ = self.sp_refl.value()
            obr = (f';  obrót igieł (mediana): drganie {pl(float(np.median(h.obr_drg[i_, :, 3])), 1)}, dryf {pl(float(np.median(h.obr_dryf[i_, :, 3])), 2)} '
                   f'°/100 kroków, od startu {pl(float(h.obr_kat0[i_]), 0)}°;  reflektor {c_}: dryf {pl(float(h.obr_dryf[i_, c_, 3]), 2)}')
        self.lbs.setText(f'pary przeciwnych reflektorów {h.interp(h.pary, t):.0%};  wiązki: {self.cb_wiazki.currentText()} '
                         f'(kolor = laser, błękit = ujemna jasność z < 0)' + obr)
        self.lb4.setText('bloki 1 2 3 (góra), 4 5 6 (dół); ' + ('pierwiastek z mapy, ' if self.ch_pierw.isChecked() else '') + 'normowana w każdym bloku;  biała = prawdziwa ramka, błękitna = ramka modelu')
        self.lb5.setText(f'{self.cb_mapa.currentText()}')
        self._og_czas(t)
        self._mp_czas(t)
        self._dol_czas(tryb_pod)
        if not tryb_pod and getattr(self, '_mech', None) is not None and self.widac('13'):
            self._mech_zlec(); self._mech_rysuj()                   # ruch w obrębie checkpointu: u = t − i
        self.rzuty.ustaw_czas(t)                                     # okienka R1–R8 (tylko widoczne)
        if tryb_pod:
            return
        if self.ch_kamera.isChecked() and hasattr(self, '_cz'):
            cz = h.interp(self._cz, t)
            self.v3.kadruj(cz[0], cz[1], self.sp_zbl.value(), self.sp_zbl.value() * 2 / 3)
        zb = self.sp_zbl.value()
        if hasattr(self, '_cz'):
            cz = h.interp(self._cz, t)
            self.e2.opcje['box'] = (cz[0] - zb / 2, cz[1] - zb / 3, cz[0] + zb / 2, cz[1] + zb / 3) if self.ch_kamera.isChecked() else None
        L = int(np.floor(t)) - int(np.floor(h.indeks(max(krok - self.sp_slad.value(), float(h.kroki[0])))))
        for e, v in ((self.e1, self.v1), (self.e2, self.v2), (self.e3, self.v3)):
            e.opcje['pxc'] = v.px_na_komorke(); e.opcje['L'] = int(np.clip(L, 0, 400)) if self.sp_slad.value() > 0 else 0
