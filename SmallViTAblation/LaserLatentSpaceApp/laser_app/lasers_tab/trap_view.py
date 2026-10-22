"""Okienko 11: pułapka optyczna wybranej cząsteczki (focal_trap), jej wiązki, światło, obrys 3D, pole w podkrokach, gotowe pułapki z dysku i pomiar w podkrokach dla całej historii."""
from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QProgressBar,
                             QPushButton, QToolButton, QWidget)
import numpy as np

from laser_app.compute import focal_trap
from laser_app.lasers_tab.shared import pl
from laser_app.lasers_tab.workers import FocalTrapWorker, TrapMeasurementsWorker
from laser_app.ui.layout_widgets import NarrowCombo
from laser_app.ui.widgets import _spin, CollapsibleSection, lista_liczb


class TrapMixin:
    """Okienko 11: pułapka optyczna wybranej cząsteczki (focal_trap), jej wiązki, światło, obrys 3D, pole w podkrokach, gotowe pułapki z dysku i pomiar w podkrokach dla całej historii.
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_pulapka(self):
        """Sekcja pułapki optycznej (okienko 11)."""
        # --- pułapka optyczna (ogniskowa wiązek bloku 6)
        g = CollapsibleSection('Pułapka optyczna: ogniskowa wiązek wokół koloidu (okienko 11)'); f = QFormLayout(g)
        self.ch_og = QCheckBox('pokaż podgląd ogniskowej wybranej cząsteczki (wczytaj albo policz)'); self.ch_og.setChecked(False)
        self.ch_og.setToolTip('Pole wszystkich 768 wiązek bloku 6 wokół wybranej cząsteczki (sekcja sfery: „Cząsteczka”), w przekroju 3D '
                              'rzutu sfery, w każdym checkpoincie, liczone z wag (app/focal_trap.py). Pułapka = punkt, w którym pchnięcia '
                              'wszystkich wiązek się znoszą (zero pola). Liczy się w tle i zostaje obok historii w cache. '
                              'Po uruchomieniu programu wyłączone: nic się nie wczytuje ani nie liczy, dopóki tego nie zaznaczysz '
                              '(albo nie klikniesz „Wczytaj / policz”).')
        self.cb_og_czas = NarrowCombo(); self.cb_og_czas.addItems(['oś uczenia: blok 6 w każdym checkpoincie',
                                                                  'podkroki: przejście przez 6 bloków (suwak podkroków)'])
        self.cb_og_czas.setToolTip('Oś uczenia: ognisko 768 wiązek bloku 6 w każdym checkpoincie (skok cząsteczki w bloku 6). Podkroki: '
                                   'w podkrokach MLP ognisko 768 wiązek danego bloku, w podkrokach uwagi 600 wkładów od innych cząsteczek '
                                   '(tam cząsteczkę przesuwają inne cząsteczki, nie lasery); w każdym podkroku suma wkładów = prawdziwy ruch.')
        self.ch_og_pole = QCheckBox('licz też lokalne pole na siatce i szukaj zera (wolniej)'); self.ch_og_pole.setChecked(False)
        self.ch_og_pole.setToolTip('Bez tego liczone są tylko wiązki wybranej cząsteczki. Z tym: pole na siatce G³, zero pola (Newton, jakobian\n'
                                   'analityczny) i kształt pola w koloidzie: około 0,07 s na checkpoint przy siatce 17 (RTX 3070), czyli około\n'
                                   'minuty na 806 checkpointów; każda dodatkowa cząsteczka „Licz razem z” około 0,04 s na checkpoint.')
        self.sp_og_G = _spin(9, 33, 17, 2); self.sp_og_G.setToolTip('Węzłów siatki pola na bok (G³ punktów).')
        self.cb_og_lupa = QComboBox(); self.cb_og_lupa.addItems(['lupa: kula pola = ułamek długości wiązek', 'prawdziwa skala (bez lupy)'])
        self.cb_og_lupa.setToolTip('Podkroki: komórka tokenu ma tylko 1–15 % długości wiązek, więc pole w prawdziwej skali chowa się w jasnym\n'
                                   'środku ogniska. Lupa pokazuje kulę komórki powiększoną wokół cząsteczki do wybranego ułamka długości\n'
                                   'wiązek (powiększenie w opisie okienka); kierunki i proporcje strzałek bez zmian.')
        self.sp_og_lupa = _spin(0.1, 1.0, 0.6, 0.05, 2, ' × wiązek')
        self.sp_og_lupa.setToolTip('Promień pokazanej kuli pola (z lupą) jako ułamek długości wiązek okienka 11.')
        for w_ in (self.cb_og_lupa, self.sp_og_lupa):
            (w_.currentIndexChanged if w_ is self.cb_og_lupa else w_.valueChanged).connect(lambda *_: self._og_opcje())
        self.cb_og_gotowe = NarrowCombo(); self.cb_og_gotowe.setPlaceholderText('— policzone pułapki tej historii —')
        self.cb_og_gotowe.setToolTip('Pułapki już policzone dla tej historii (pliki ogniskowa_t…npz obok historii): wybór ustawia cząsteczkę,\n'
                                     'siatkę i zasięg i wczytuje pułapkę z dysku od razu, bez liczenia. Lista odświeża się przy otwarciu.')
        self.cb_og_gotowe.przed_otwarciem = self._og_lista_gotowych
        self.cb_og_gotowe.activated.connect(self._og_wybierz_gotowa)
        self.cb_og_dokl = NarrowCombo(); self.cb_og_dokl.addItems(['float32 na GPU (szybko; reszta pola w float64)', 'float64 (dokładnie, wolniej)'])
        self.cb_og_dokl.setToolTip('Dokładność liczenia pułapki i mapy pułapek. float32: jakobian, siatka pola i kolory w float32 (karta GeForce liczy\n'
                                   'float64 kilkadziesiąt razy wolniej), ale reszta pola f w Newtonie w float64, bo w pułapce pchnięcia 768 wiązek\n'
                                   'się znoszą (mała różnica dużych liczb). Pomiar: te same zera pola co float64, wartości własne zgodne do 2·10⁻⁵;\n'
                                   'pole i tak jest zapisywane w float16. Zmiana dotyczy nowych liczeń (gotowe pliki zostają).')
        self.ed_og_razem = QLineEdit(); self.ed_og_razem.setPlaceholderText('np. 12, 101 (najwyżej 8)')
        self.ed_og_razem.setToolTip('Pułapki tych cząsteczek liczą się RAZEM z wybraną, w jednym przejściu przez checkpointy: wczytanie wag\n'
                                    'i przejście sieci są wspólne, osobno liczy się tylko pole w komórce każdej cząsteczki. Trzy cząsteczki\n'
                                    'zajmują około 1,7 × czasu jednej. Każda zostaje w osobnym pliku obok historii, więc przełączenie\n'
                                    '„Cząsteczka” na którąś z nich (albo „▾ policzone”) pokazuje jej pułapkę od razu, bez liczenia.')
        self.bt_og_razem = QToolButton(); self.bt_og_razem.setText('▾ policzone')
        self.bt_og_razem.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.bt_og_razem.setToolTip('Przejście do cząsteczki, której pułapka jest już policzona (z tej listy i wybrana), z tą samą siatką i zasięgiem.')
        m_ = QMenu(self.bt_og_razem); m_.aboutToShow.connect(lambda m_=m_: self._og_menu_policzone(m_)); self.bt_og_razem.setMenu(m_)
        self.sp_og_zas = _spin(0.05, 1.0, 1.0, 0.05, 2, ' × komórki')
        self.sp_og_zas.setToolTip('Pole liczy się tylko w komórce tokenu: połowa boku sześcianu = ułamek r_kom, gdzie r_kom = połowa odległości '
                                  'do najbliższego innego tokenu w rzucie 3D (w każdym checkpoincie). Przy 1 żaden inny token nie leży w sześcianie. '
                                  'Zero pola (pułapka) jest szukane tylko w środku komórki; siatka nie rośnie.')
        hb = QHBoxLayout(); self.bt_og = QPushButton('Wczytaj / policz'); self.bt_og_stop = QPushButton('Przerwij')
        self.bt_og.clicked.connect(lambda: self._og_zlec(wymus=True) if self.ch_og.isChecked() else self.ch_og.setChecked(True))
        self.bt_og_stop.clicked.connect(self._og_przerwij)
        self.bt_og_stop.setEnabled(False); hb.addWidget(self.bt_og); hb.addWidget(self.bt_og_stop)
        self.pasek_og = QProgressBar(); self.pasek_og.setTextVisible(True)
        self.bt_pom = QPushButton('Policz pułapkę w podkrokach dla całej historii → plik'); self.bt_pom_stop = QPushButton('Przerwij')
        self.bt_pom.setToolTip('Dla wybranej cząsteczki, w każdym checkpoincie (co n) i każdym z 12 podkroków: skok, komórka, pole w komórce\n'
                               'i jego część liniowa J (sztywność κ w 3 osiach, dywergencja, wir, gdzie jest zero pola i czy pułapka trzyma),\n'
                               'kształt ogniska (tensor kierunków, wydłużenie, koherencja wiązek), ile laserów świeci, ilu tokenów słucha.\n'
                               'Zapis: katalog historii / pulapka_podkrokow_t<cząsteczka>_G<siatka>.npz i .csv (trap_measurements.py).')
        self.sp_pom_co = _spin(1, 200, 1, 1, sufiks=' ckpt'); self.sp_pom_co.setToolTip('Co ile checkpointów liczyć (ostatni zawsze).')
        self.sp_pom_G = _spin(3, 17, 5, 2); self.sp_pom_G.setToolTip('Węzłów siatki pola na bok do dopasowania J (5 wystarcza: pole jest liniowe, R² > 0,99).')
        self.pasek_pom = QProgressBar(); self.pasek_pom.setTextVisible(True); self.lb_pom = QLabel(''); self.lb_pom.setWordWrap(True)
        self.bt_pom_stop.setEnabled(False)
        self.bt_pom.clicked.connect(self._pomiary_start); self.bt_pom_stop.clicked.connect(self._pomiary_stop)
        self.ch_og_wiazki = QCheckBox('768 wiązek zbiegających się w cząsteczce (zbliżenie)'); self.ch_og_wiazki.setChecked(True)
        self.ch_og_wiazki.setToolTip('Każdy reflektor wysyła do wybranej cząsteczki jedną wiązkę z kierunku −d_c (d_c = P v_c w rzucie sfery), '
                                     'w kolorze reflektora; błękit = ujemna jasność. Wszystkie 768 wiązek zbiega się w cząsteczce: to jej ognisko.')
        self.cb_og_jasn = NarrowCombo(); self.cb_og_jasn.addItems(['wkład do skoku: wiązka z kierunku, w który pcha, jasność |z − z̄|·|P v|', 'jak na sferze: kierunek v_c, jasność z, błękit = ujemna dioda'])
        self.cb_og_jasn.setToolTip('Wkład do skoku liczony względem średniej po tokenach, jak położenia na sferze: suma wkładów = rysowany skok (Z20).')
        self.sp_og_gw = _spin(0.05, 20, 0.6, 0.1, 2, ' ×'); self.sp_og_gw.setToolTip('Jasność wiązek (najjaśniejsza = 1).')
        self.sp_og_dlw = _spin(0.1, 10, 1.0, 0.1, 1, ' ×'); self.sp_og_dlw.setToolTip('Długość wiązek w zbliżeniu.')
        self.cb_og_wk = NarrowCombo(); self.cb_og_wk.addItems(['wkłady: brak', 'wkłady: strzałki od cząsteczki', 'wkłady: łańcuch głowa–ogon (koniec = skok)'])
        self.cb_og_wk.setCurrentIndex(1)
        self.cb_og_wk.setToolTip('Wkład wiązki c do skoku = (z_jc − z̄_c)·P v_c. Strzałki: każdy wkład od cząsteczki (długości przeskalowane, '
                                 'kierunki prawdziwe). Łańcuch: wkłady dodawane jeden za drugim, od najdłuższego; koniec łańcucha to dokładnie skok, '
                                 'a długość drogi pokazuje, ile pchnięć się znosi (Z18).')
        self.sp_og_dlwk = _spin(0.1, 10, 1.0, 0.1, 1, ' ×'); self.sp_og_dlwk.setToolTip('Długość strzałek wkładów.')
        self.ch_og_sw = QCheckBox('światło ogniska (wiązki gaussowskie)'); self.ch_og_sw.setChecked(True)
        self.ch_og_sw.setToolTip('Każda wiązka to wiązka gaussowska z przewężeniem w cząsteczce, w kolorze reflektora; światło wszystkich wiązek '
                                 'sumuje się i daje kształt ogniska. Ognisko jest wydłużone wzdłuż kierunku, z którego przychodzi najwięcej światła '
                                 '(osie: białe odcinki; wartości w opisie okienka).')
        self.sp_og_jsw = _spin(0.01, 20, 0.35, 0.05, 2, ' ×'); self.sp_og_jsw.setToolTip('Jasność światła ogniska.')
        self.sp_og_szer = _spin(0.002, 0.3, 0.02, 0.005, 3, ' × L'); self.sp_og_szer.setToolTip('Przewężenie wiązki w ognisku (ułamek długości wiązki).')
        self.sp_og_rozb = _spin(0.02, 1.0, 0.15, 0.01, 2, ' rad'); self.sp_og_rozb.setToolTip('Kąt rozbieżności wiązki: jak szybko rozszerza się poza ogniskiem.')
        self.ch_og_linie = QCheckBox('cienkie linie wiązek'); self.ch_og_linie.setChecked(True)
        self.ch_og_refl = QCheckBox('reflektory: wiązki od reflektorów na sferze, skala jak sfera (film 12)'); self.ch_og_refl.setChecked(False)
        self.ch_og_refl.setToolTip('Okienko 11 w skali sfery reflektorów: 768 reflektorów na zewnętrznej sferze (kolory jak w filmie 12), '
                                   'szklana sfera i wiązka każdego reflektora do wybranej cząsteczki: reflektor → punkt załamania na szkle → '
                                   'cząsteczka (ostatni odcinek wzdłuż kierunku wiązki). Kamera przejmuje odległość i kąt widoku sfery i przestaje '
                                   'śledzić cząsteczkę. W podkrokach uwagi źródłami są inne cząsteczki.')
        self.sp_og_jrefl = _spin(0.05, 20, 1.0, 0.1, 2, ' ×'); self.sp_og_jrefl.setToolTip('Jasność wiązek od reflektorów.')
        self.ch_og_osieo = QCheckBox('osie kształtu ogniska'); self.ch_og_osieo.setChecked(True)
        self.ch_og_obrys = QCheckBox('obrys ogniska 3D (powierzchnia równej jasności światła)'); self.ch_og_obrys.setChecked(False)
        self.ch_og_obrys.setToolTip('Powierzchnia, na której suma światła wszystkich wiązek gaussowskich spada do wybranego ułamka jasności '
                                    'w cząsteczce (jak objętość ogniska w optyce). Kształt pochodzi z danych: z kierunków i jasności wiązek; '
                                    'ognisko wydłuża się wzdłuż kierunku, z którego przychodzi najwięcej światła. Rozmiar zależy od przewężenia '
                                    'i rozbieżności wiązek (parametry rysunku), więc porównywać można kształty, nie rozmiary bezwzględne.')
        self.sp_og_poz = _spin(0.02, 0.98, 0.5, 0.05, 2, ' szczytu')
        self.sp_og_poz.setToolTip('Poziom obrysu: 0,5 = połowa jasności w cząsteczce (FWHM), 0,135 = 1/e².')
        self.sp_og_jobr = _spin(0.02, 1.0, 0.15, 0.05, 2); self.sp_og_jobr.setToolTip('Krycie powierzchni obrysu.')
        self.sp_og_pobr = _spin(1, 200, 20, 1, 0, ' ×')
        self.sp_og_pobr.setToolTip('Lupa: obrys rysowany powiększony wokół cząsteczki (prawdziwy ma około 1–2 przewężeń wiązki, czyli kilka '
                                   'pikseli). Kształt i proporcje bez zmian; 1 = prawdziwy rozmiar. Warstwice: złote = przekroje wzdłuż najdłuższej '
                                   'osi obrysu, białe = wzdłuż drugiej osi.')
        self.ch_og_mgla = QCheckBox('lokalne pole: mgła (kolor = wiązki)'); self.ch_og_mgla.setChecked(False)
        self.ch_og_mgla.setToolTip('Punkt siatki ma kolor wiązek o największym wkładzie |z_c − z̄_c|·|P v_c| (kolory reflektorów jak na sferze, '
                                   'mieszanka trzech największych; nasycenie = udział największej). Jasny tam, gdzie pchnięcia się znoszą: '
                                   'exp(−|f|² / 2σ²), σ = ostrość × mediana |f| na siatce.')
        self.sp_og_jas = _spin(0.005, 5, 0.25, 0.02, 3); self.sp_og_ostr = _spin(0.02, 3, 0.35, 0.05, 2)
        self.sp_og_tlo = _spin(0, 1, 0.05, 0.01, 2); self.sp_og_rozm = _spin(2, 120, 22, 1, 0, ' px')
        self.sp_og_ostr.setToolTip('Ostrość ogniska: σ = ostrość × mediana |f|. Mniej = węższe, jaśniejsze ognisko.')
        self.sp_og_tlo.setToolTip('Jasność pola poza ogniskiem (0 = widać tylko ognisko).')
        self.sp_og_rozm.setToolTip('Rozmycie: średnica plamki każdego punktu siatki na ekranie.')
        self.cb_og_strz = NarrowCombo(); self.cb_og_strz.addItems(['brak', 'pole wiązek (pełne pchnięcie)', 'kształt pułapki (pole względem skoku koloidu)'])
        self.cb_og_strz.setCurrentIndex(1)                                   # widać od razu po policzeniu pola (oś uczenia)
        self.cb_og_strz.setToolTip('Pełne pole: dokąd wiązki pchają cząstkę postawioną w danym punkcie. Kształt pułapki: to samo minus skok '
                                   'samego koloidu, czyli jak wiązki rozpychają albo ściągają i obracają otoczenie koloidu (siodło z wirem).')
        self.sp_og_strz = _spin(0.1, 10, 0.8, 0.1, 1, ' ×'); self.sp_og_pods = _spin(1, 8, 3)
        self.sp_og_strz.setToolTip('Długość strzałek (1 = typowa strzałka długości odstępu między strzałkami).')
        self.ch_og_plyn = QCheckBox('strzałki pola płyną wzdłuż pola (animacja, także na pauzie)'); self.ch_og_plyn.setChecked(True)
        self.ch_og_plyn.setToolTip('Każda strzałka przesuwa się wzdłuż swojego kierunku, pojawia się i gaśnie (faza losowa w każdym węźle), '
                                   'więc widać, dokąd pole niesie. Długość i kierunek bez zmian.')
        self.sp_og_tempo = _spin(0.05, 5, 0.6, 0.05, 2, ' /s'); self.sp_og_tempo.setToolTip('Tempo płynięcia strzałek (przebiegi na sekundę).')
        self._og_anim = QTimer(self); self._og_anim.setInterval(33); self._og_anim.timeout.connect(self._og_anim_tik)
        self.sp_og_pods.setToolTip('Strzałka co tyle węzłów siatki.')
        self.ch_og_osie = QCheckBox('osie siodła i wir (w koloidzie i w pułapce)'); self.ch_og_osie.setChecked(True)
        self.ch_og_osie.setToolTip('Wartości własne Jakobianu 3 × 3 pola: błękit = ściąga (groty do środka), czerwień = wypycha (groty na zewnątrz), '
                                   'fiolet = wir (okrąg z grotami w kierunku obrotu). Długość ∝ |λ|.')
        self.ch_og_skok = QCheckBox('skok koloidu w bloku 6 (pchnięcie MLP)'); self.ch_og_skok.setChecked(True)
        self.ch_og_tlo = QCheckBox('pozostałe cząsteczki w tle (przyciemnione)'); self.ch_og_tlo.setChecked(True)
        self.ch_og_tlo.setToolTip('Pozostałe 599 cząsteczek w kolorach cząsteczek, przyciemnione; w podkrokach płyną razem z przejściem '
                                  'przez sieć, w osi uczenia z checkpointami (te same współrzędne co sfera).')
        self.sp_og_jtlo = _spin(0.02, 1.0, 0.35, 0.05, 2, ' ×'); self.sp_og_jtlo.setToolTip('Jasność cząsteczek w tle.')
        self.ch_og_ruch = QCheckBox('wypadkowy ruch koloidu w nadchodzącym kroku (zielony)'); self.ch_og_ruch.setChecked(True)
        self.ch_og_ruch.setToolTip('Zielona strzałka: dokąd koloid przejdzie z bieżącego checkpointu do następnego (położenie następne − bieżące, '
                                   'wypadkowa całego kroku uczenia). Pomarańczowy skok to pchnięcie wiązek w tym checkpoincie; w opisie kąt '
                                   'między nimi. W kadrze komórki długość jest przycięta do ścianki sześcianu, kierunek prawdziwy.')
        self.cb_og_slad = NarrowCombo(); self.cb_og_slad.addItems(['brak', 'koloid', 'koloid i pułapka']); self.cb_og_slad.setCurrentIndex(1)
        self.sp_og_slad = _spin(1, 2000, 30, 5, sufiks=' ckpt'); self.sp_og_slad.setToolTip('Długość śladu w checkpointach; kropki = skoki między checkpointami.')
        self.cb_og_uklad = NarrowCombo(); self.cb_og_uklad.addItems(['ślad w układzie laboratorium', 'ślad względem pułapki (pułapka stoi)'])
        self.cb_og_sledz = NarrowCombo(); self.cb_og_sledz.addItems(['kamera nie śledzi', 'środek kadru: pułapka', 'środek kadru: koloid'])
        self.cb_og_sledz.setCurrentIndex(2)
        self.cb_og_sledz.setToolTip('Okienko 11: wybrany punkt jest zawsze na środku okienka (obrót myszą zostaje). Gdy zera pola nie ma '
                                    'w komórce, „pułapka” to najsłabsze pole w komórce tokenu.')
        self.cb_og_skala = NarrowCombo(); self.cb_og_skala.addItems(['skala: swobodna (przybliżenie myszą)',
                                                                   'skala: sześcian zawsze tej samej wielkości na ekranie',
                                                                   'skala: stała przez całe uczenie (jedna odległość kamery)'])
        self.cb_og_skala.setCurrentIndex(1)
        self.cb_og_skala.setToolTip('Sześcian tej samej wielkości: odległość kamery dopasowana w każdym checkpoincie tak, że sześcian komórki '
                                    'zawsze tak samo wypełnia okienko (rozmiar komórki w ten sposób znika z obrazu). Stała: jedna odległość '
                                    'na całe uczenie (90. percentyl wielkości komórki), więc widać, jak komórka rośnie i maleje; w nielicznych '
                                    'checkpointach z największą komórką sześcian wychodzi poza okienko. W obu wiązki i wkłady są przycięte '
                                    'do sześcianu, a przybliżenie myszą nie zostaje.')
        self.ch_og_kolej = QCheckBox('odtwarzanie: najpierw lasery, potem koloid'); self.ch_og_kolej.setChecked(True)
        self.ch_og_kolej.setToolTip('Okienko 11: w pierwszej połowie przejścia między checkpointami zmienia się pole wiązek, pułapka i skok, '
                                    'w drugiej koloid przechodzi na nowe miejsce. Obie zmiany pochodzą z tego samego kroku uczenia; kolejność '
                                    'pokazuje łańcuch: zmienione lasery → nowe pole → nowe położenie.')
        self.ch_og_sfera = QCheckBox('pokaż pułapkę także na dużej sferze'); self.ch_og_sfera.setChecked(False)
        self.lb_og = QLabel('—'); self.lb_og.setWordWrap(True)
        f.addRow(self.ch_og); f.addRow('Gotowe (bez liczenia)', self.cb_og_gotowe); f.addRow('Czas okienka 11', self.cb_og_czas)
        f.addRow(self.ch_og_pole); f.addRow('Dokładność', self.cb_og_dokl)
        f.addRow('Siatka (węzłów na bok)', self.sp_og_G); f.addRow('Zasięg siatki', self.sp_og_zas)
        w_ = QWidget(); l_ = QHBoxLayout(w_); l_.setContentsMargins(0, 0, 0, 0); l_.addWidget(self.ed_og_razem, 1); l_.addWidget(self.bt_og_razem)
        f.addRow('Licz razem z', w_)
        w_ = QWidget(); l_ = QHBoxLayout(w_); l_.setContentsMargins(0, 0, 0, 0); l_.addWidget(self.cb_og_lupa, 1); l_.addWidget(self.sp_og_lupa)
        f.addRow('Pole w podkrokach', w_)
        f.addRow(hb); f.addRow(self.pasek_og)
        w_ = QWidget(); l_ = QHBoxLayout(w_); l_.setContentsMargins(0, 0, 0, 0)
        for x_ in (self.bt_pom, QLabel('co'), self.sp_pom_co, QLabel('siatka'), self.sp_pom_G, self.bt_pom_stop):
            l_.addWidget(x_, 1 if x_ is self.bt_pom else 0)
        f.addRow(w_); f.addRow(self.pasek_pom); f.addRow(self.lb_pom)
        f.addRow(self.ch_og_wiazki); f.addRow('Jasność wiązek', self.cb_og_jasn); f.addRow('  wzmocnienie', self.sp_og_gw)
        f.addRow('  długość wiązek', self.sp_og_dlw); f.addRow(self.ch_og_refl); f.addRow('  jasność od reflektorów', self.sp_og_jrefl)
        f.addRow(self.ch_og_linie); f.addRow(self.ch_og_sw)
        f.addRow('  jasność światła', self.sp_og_jsw); f.addRow('  przewężenie', self.sp_og_szer); f.addRow('  rozbieżność', self.sp_og_rozb)
        f.addRow(self.ch_og_osieo); f.addRow(self.ch_og_obrys); f.addRow('  poziom obrysu', self.sp_og_poz)
        f.addRow('  krycie obrysu', self.sp_og_jobr); f.addRow('  powiększenie obrysu', self.sp_og_pobr); f.addRow('Wkłady', self.cb_og_wk); f.addRow('  długość wkładów', self.sp_og_dlwk)
        f.addRow(self.ch_og_mgla); f.addRow('Jasność mgły', self.sp_og_jas); f.addRow('Ostrość ogniska', self.sp_og_ostr)
        f.addRow('Tło pola', self.sp_og_tlo); f.addRow('Rozmycie', self.sp_og_rozm)
        f.addRow('Rodzaj strzałek', self.cb_og_strz); f.addRow('  długość', self.sp_og_strz); f.addRow('  gęstość: co', self.sp_og_pods)
        f.addRow(self.ch_og_plyn); f.addRow('  tempo płynięcia', self.sp_og_tempo)
        f.addRow(self.ch_og_osie); f.addRow(self.ch_og_skok); f.addRow(self.ch_og_ruch)
        f.addRow(self.ch_og_tlo); f.addRow('  jasność tła', self.sp_og_jtlo)
        f.addRow('Ślad', self.cb_og_slad); f.addRow('Długość śladu', self.sp_og_slad); f.addRow('Układ śladu', self.cb_og_uklad)
        f.addRow('Kadr', self.cb_og_sledz); f.addRow('Skala kadru', self.cb_og_skala); f.addRow(self.ch_og_kolej); f.addRow(self.ch_og_sfera); f.addRow(self.lb_og)
        for w in (self.ch_og_wiazki, self.ch_og_mgla, self.ch_og_osie, self.ch_og_skok, self.ch_og_ruch, self.ch_og_plyn, self.ch_og_tlo, self.ch_og_kolej, self.ch_og_sfera,
                  self.ch_og_sw, self.ch_og_linie, self.ch_og_osieo, self.ch_og_obrys):
            w.toggled.connect(self._og_opcje)
        for w in (self.sp_og_jas, self.sp_og_ostr, self.sp_og_tlo, self.sp_og_rozm, self.sp_og_strz, self.sp_og_pods, self.sp_og_slad,
                  self.sp_og_gw, self.sp_og_dlw, self.sp_og_dlwk, self.sp_og_jsw, self.sp_og_szer, self.sp_og_rozb, self.sp_og_jrefl,
                  self.sp_og_poz, self.sp_og_jobr, self.sp_og_pobr, self.sp_og_jtlo):
            w.valueChanged.connect(self._og_opcje)
        for w in (self.cb_og_strz, self.cb_og_slad, self.cb_og_uklad, self.cb_og_sledz, self.cb_og_skala, self.cb_og_jasn, self.cb_og_wk,
                  self.cb_og_czas):
            w.currentIndexChanged.connect(self._og_opcje)
        self.ch_og_refl.toggled.connect(self._og_kamera_refl); self.ch_og_refl.toggled.connect(self._og_opcje)
        self._og_timer = QTimer(self); self._og_timer.setSingleShot(True); self._og_timer.setInterval(500)
        self._og_timer.timeout.connect(self._og_zlec)
        for w in (self.sp_og_G, self.sp_og_zas, self.sp_tok):
            w.valueChanged.connect(lambda *_: self._og_timer.start())
        self.ch_og.toggled.connect(lambda *_: self._og_zlec())
        self.ch_og.toggled.connect(lambda *_: self._og_czas(self.t) if self.h is not None else None)
        self.ch_og_pole.toggled.connect(lambda *_: self._og_zlec())
        self.cb_og_czas.currentIndexChanged.connect(lambda *_: self._zlec_podkroki())
        self.cb_og_czas.currentIndexChanged.connect(self._og_czas_zmiana)
        self._og = None; self._og_watek = None; self._og_klucz = None
        self._sek['pulapka'] = g

    def _og_klucz_teraz(self):
        h = self.h
        G = int(self.sp_og_G.value()) if self.ch_og_pole.isChecked() else 1        # G = 1: tylko wiązki wybranej cząsteczki
        return (str(h.katalog), int(self.sp_tok.value()), G, round(float(self.sp_og_zas.value()), 2))

    def _og_zlec(self, wymus=False):
        """Pułapka dla wybranej cząsteczki: z cache od razu, inaczej liczenie w tle (stare liczenie innej cząsteczki jest przerywane)."""
        h = self.h
        if h is None or not self.ch_og.isChecked():
            return
        klucz = self._og_klucz_teraz()
        if klucz == self._og_klucz and self._og is not None and not wymus:
            return
        if not wymus and not (self.widac('11') or (self.ch_og_sfera.isChecked() and self.widac('s'))):
            self.lb_og.setText('Pułapka policzy się, gdy okienko 11 (albo pułapka na sferze) będzie widoczne.')
            return                                                       # nic nie liczymy dla niewidocznego okienka (_odslonieto)
        w = self._og_watek
        if w is not None and w.isRunning():
            if getattr(w, 'klucz', None) == klucz:
                return
            k_ = getattr(w, 'klucz', None)
            if k_ is not None and (k_[0], k_[2], k_[3]) == (klucz[0], klucz[2], klucz[3]) and klucz[1] in w.toki:
                self.lb_og.setText(f'Pułapka cząsteczki {klucz[1]} liczy się razem z {", ".join(map(str, w.toki))}; pokaże się po skończeniu.')
                return                                                   # ta cząsteczka już się liczy razem z inną: czekamy
            w.stop = True
            self._og_stare = [x for x in getattr(self, '_og_stare', []) if x.isRunning()] + [w]
        d = focal_trap.wczytaj(h, *klucz[1:])
        if d is not None:
            self._og_zastosuj(klucz, d)
            return
        razem = [t for t in lista_liczb(self.ed_og_razem.text(), self._n_tok()) if t != klucz[1]][:8]
        w = FocalTrapWorker(h, *klucz[1:], razem=razem, dokladnosc=self._dokladnosc()); w.klucz = klucz
        w.postep.connect(self._og_postep)
        w.gotowe.connect(lambda args, W, k=klucz: self._og_gotowe(k, W))
        w.blad.connect(self._og_blad)
        self._og_watek = w; self.bt_og_stop.setEnabled(True); self.pasek_og.setValue(0)
        self.lb_og.setText(f'Liczę pułapkę cząsteczki {klucz[1]}' + (f' (razem z {", ".join(map(str, razem))})' if razem else '')
                           + f' we wszystkich {h.K} checkpointach (w tle)…')
        w.start()

    def _og_przerwij(self):
        if self._og_watek is not None:
            self._og_watek.stop = True

    def _og_postep(self, i, n, opis):
        self.pasek_og.setMaximum(n); self.pasek_og.setValue(i); self.pasek_og.setFormat(opis)

    def _og_blad(self, tekst):
        self.bt_og_stop.setEnabled(False)
        self.lb_og.setText(f'Pułapka: {tekst}')

    def _dokladnosc(self):
        return 'float64' if self.cb_og_dokl.currentIndex() == 1 else 'float32'

    def _og_lista_gotowych(self):
        """Lista „Gotowe (bez liczenia)”: pułapki tej historii na dysku (cząsteczka, siatka, zasięg); wybrana pozycja zostaje."""
        cb = self.cb_og_gotowe
        cb.blockSignals(True); cb.clear()
        if self.h is not None:
            for tok, G, zas in focal_trap.gotowe(self.h):
                opis = f'cząsteczka {tok}: ' + (f'pole na siatce {G}³, zasięg {pl(zas)}' if G > 1 else f'tylko wiązki, zasięg {pl(zas)}')
                cb.addItem(opis, (tok, G, zas))
        if self.h is not None:
            teraz = self._og_klucz_teraz()[1:]
            for i in range(cb.count()):
                tok, G, zas = cb.itemData(i)
                if (tok, G, round(zas, 2)) == (teraz[0], teraz[1], teraz[2]):
                    cb.setCurrentIndex(i)
                    break
            else:
                cb.setCurrentIndex(-1)
        cb.blockSignals(False)

    def _og_wybierz_gotowa(self, i):
        dane = self.cb_og_gotowe.itemData(i)
        if not dane:
            return
        tok, G, zas = dane
        for w_ in (self.sp_tok, self.sp_og_G, self.sp_og_zas, self.ch_og_pole):
            w_.blockSignals(True)
        self.ch_og_pole.setChecked(G > 1)
        if G > 1:
            self.sp_og_G.setValue(G)
        self.sp_og_zas.setValue(zas)
        for w_ in (self.sp_og_G, self.sp_og_zas, self.ch_og_pole):
            w_.blockSignals(False)
        self.sp_tok.blockSignals(False)
        if not self.ch_og.isChecked():
            self.ch_og.blockSignals(True); self.ch_og.setChecked(True); self.ch_og.blockSignals(False)
        self.sp_tok.setValue(tok)                                        # zmiana cząsteczki odświeża sferę, wybór i okienka
        self._og_zlec(wymus=True)                                        # plik jest na dysku: wczytanie, bez liczenia

    def _og_gotowe(self, klucz, W):
        """W = {cząsteczka: wynik} (wybrana i liczone razem z nią); pokazujemy tę, która jest wybrana TERAZ (mogła się zmienić na
        jedną z liczonych razem w trakcie liczenia)."""
        self.bt_og_stop.setEnabled(False)
        if self.h is None:
            return
        teraz = self._og_klucz_teraz()
        self._og_lista_gotowych()                                        # nowe pliki pułapek na liście „Gotowe”
        if (teraz[0], teraz[2], teraz[3]) == (klucz[0], klucz[2], klucz[3]) and teraz[1] in W:
            self._og_zastosuj(teraz, W[teraz[1]])
        elif teraz != self._og_klucz:
            self._og_zlec()                                              # wybrano coś innego w trakcie: teraz to

    def _n_tok(self):
        return int(self.sp_tok.maximum()) + 1

    def _og_menu_policzone(self, m):
        """Menu „▾ policzone”: wybrana cząsteczka i cząsteczki z „Licz razem z”, ✓ = pułapka gotowa na dysku (ta sama siatka i zasięg)."""
        m.clear()
        h = self.h
        if h is None:
            m.addAction('najpierw wczytaj historię').setEnabled(False)
            return
        _, tok, G, zas = self._og_klucz_teraz()
        toki = list(dict.fromkeys([tok] + lista_liczb(self.ed_og_razem.text(), self._n_tok())))
        for t in toki:
            gotowa = focal_trap.plik(h, t, G, zas).exists()
            a = m.addAction(f'{"✓" if gotowa else "…"} cząsteczka {t}' + (' (wybrana)' if t == tok else ''))
            a.triggered.connect(lambda _=False, t=t: self.sp_tok.setValue(t))

    def _og_zastosuj(self, klucz, d, kamera=True):
        h = self.h
        if h is None or len(d['kroki']) != h.K or np.any(np.asarray(d['kroki']) != np.asarray(h.kroki)):
            return
        self._og, self._og_klucz = d, klucz
        r = h.r99[-1] * np.ones(h.K) if self.cb_skala.currentIndex() == 1 else h.r99
        s = 0.55 * self.es.R / np.maximum(np.asarray(r, np.float64), 1e-9)
        for e in (self.eog, self.es_og):
            e.ustaw_dane(d, s); e.ustaw_kolory(h.kol)
        self.eog.ustaw_reflektory(h.rfix)
        Lb = np.maximum(1.5 * np.linalg.norm(d['skok'], axis=1), 0.5 * d['L']) * s
        self._og_dist_zbl = 4.0 * float(np.median(Lb)) * self.sp_skala_chm.value()
        if kamera and not self.ch_og_refl.isChecked():                   # zbliżenie: kadr na wiązki i skok cząsteczki
            self.vog.opts['distance'] = self._og_dist_zbl
            if int(d['G']) > 1 and self.cb_og_sledz.currentIndex() == 0:
                self.cb_og_sledz.setCurrentIndex(2)                          # pole tokenu: koloid na środku kadru
        # kadr o stałej skali: jedna odległość na całe uczenie, z 90. percentyla promienia kuli komórki
        # (wokół koloidu: L; wokół pułapki: L + odległość pułapka–koloid), w jednostkach ekranu
        Ls = np.asarray(d['L'], np.float64) * s
        off = np.linalg.norm(np.asarray(d['ogn'], np.float64) - np.asarray(d['srodek'], np.float64), axis=1) * s
        self._og_R_stale = (float(np.percentile(Ls + off, 90)), float(np.percentile(Ls, 90)))
        zb = np.asarray(d['zb'])
        rk = np.asarray(d['r_kom']) / np.maximum(np.asarray(d['r99']), 1e-12)
        self.lb_og.setText(f'Pułapka cząsteczki {klucz[1]}: komórka tokenu (połowa odległości do sąsiada) {pl(float(np.median(rk)), 3)} r99 '
                           f'(mediana; {pl(float(rk.min()), 3)}–{pl(float(rk.max()), 3)}); zero pola w komórce w {zb.mean():.0%} checkpointów '
                           f'({int(zb.sum())} z {len(zb)}). Zgodność z siecią: jasności {float(np.max(d["blad_z"])):.1e}, '
                           f'położenia {float(np.max(d["blad_pos"])):.1e}.')
        self._og_opcje()

    def _og_kamera_refl(self, on):
        """Reflektory w okienku 11: kamera jak na sferze reflektorów (odległość, pole widzenia, kąty, środek), bez śledzenia;
        po wyłączeniu powrót do zbliżenia i śledzenia koloidu."""
        from PyQt6.QtGui import QVector3D
        if on:
            self._og_sledz_przed = (self.cb_og_sledz.currentIndex(), self.cb_og_skala.currentIndex())
            self.cb_og_sledz.setCurrentIndex(0); self.cb_og_skala.setCurrentIndex(0)
            for k in ('fov', 'elevation', 'azimuth'):
                self.vog.opts[k] = self.vs.opts[k]
            # pole widzenia pyqtgraph jest poziome: ta sama liczba pikseli na jednostkę sceny przy innej szerokości okienka
            self.vog.opts['distance'] = self.vs.opts['distance'] * max(self.vog.width(), 1) / max(self.vs.width(), 1)
            c = self.vs.opts['center']; self.vog.opts['center'] = QVector3D(c.x(), c.y(), c.z())
        else:
            self.vog.opts['distance'] = getattr(self, '_og_dist_zbl', 30.0)
            sl_, sk_ = getattr(self, '_og_sledz_przed', (2, 1))
            self.cb_og_sledz.setCurrentIndex(sl_); self.cb_og_skala.setCurrentIndex(sk_)
        self.vog.update()

    def _og_opcje(self, *_):
        sk = self.sp_skala_chm.value()
        wspolne = dict(mgla=self.ch_og_mgla.isChecked(), jas=self.sp_og_jas.value(), ostr=self.sp_og_ostr.value(),
                       tlo=self.sp_og_tlo.value(), rozm=self.sp_og_rozm.value(), strzalki=self.cb_og_strz.currentIndex(),
                       dl_strz=self.sp_og_strz.value(), pods=self.sp_og_pods.value(), osie=self.ch_og_osie.isChecked(),
                       skok=self.ch_og_skok.isChecked(), skala_chmury=sk)
        self.eog.opcje.update(wspolne, slad=self.cb_og_slad.currentIndex(), n_slad=self.sp_og_slad.value(),
                              wsp=self.cb_og_uklad.currentIndex(), pudlo=self.ch_og_mgla.isChecked() or self.cb_og_strz.currentIndex() > 0,
                              wiazki=self.ch_og_wiazki.isChecked(), jasnosc=self.cb_og_jasn.currentIndex(), gain_w=self.sp_og_gw.value(),
                              dl_wiazki=self.sp_og_dlw.value(), wklady=self.cb_og_wk.currentIndex(), dl_wkladow=self.sp_og_dlwk.value(),
                              swiatlo=self.ch_og_sw.isChecked(), jas_sw=self.sp_og_jsw.value(), szer_sw=self.sp_og_szer.value(),
                              rozb_sw=self.sp_og_rozb.value(), linie_wiazek=self.ch_og_linie.isChecked(), osie_ogniska=self.ch_og_osieo.isChecked(),
                              refl=self.ch_og_refl.isChecked(), jas_refl=self.sp_og_jrefl.value(), R=float(self.es.R),
                              sfera=float(self.es.opcje.get('sfera', 0.10)), obrys=self.ch_og_obrys.isChecked(),
                              poziom_obr=self.sp_og_poz.value(), jas_obr=self.sp_og_jobr.value(), pow_obr=self.sp_og_pobr.value(),
                              przytnij=self.cb_og_skala.currentIndex() > 0, pulapka_zast=self.cb_og_sledz.currentIndex() == 1,
                              ruch_nast=self.ch_og_ruch.isChecked(), plyn=self.ch_og_plyn.isChecked(), tlo_czastki=self.ch_og_tlo.isChecked())
        for k_ in ('mgla', 'jas', 'ostr', 'tlo', 'rozm', 'strzalki', 'dl_strz', 'pods', 'plyn', 'skala_chmury'):
            self.eog_pp.opcje[k_] = self.eog.opcje[k_]                   # pole podkroków: te same opcje mgły i strzałek
        anim = (self.ch_og.isChecked() and self.ch_og_plyn.isChecked() and self.cb_og_strz.currentIndex() > 0
                and ((self._og is not None and int(self._og['G']) > 1) or self._og_tryb_pod()))
        if anim and not self._og_anim.isActive():
            self._og_anim.start()
        elif not anim and self._og_anim.isActive():
            self._og_anim.stop()
        self.es_og.opcje.update(wspolne, slad=0, wsp=0, pudlo=False, wiazki=False, swiatlo=False, obrys=False, wklady=0)  # na sferze bez ogniska
        if self.h is not None:
            self._og_czas(self.t)

    @staticmethod
    def _og_opis_obrysu(iw):
        ob = (iw or {}).get('obrys')
        if not ob:
            return ''
        a = ob['osie']
        return (f';  obrys ogniska ({pl(ob["poziom"])} szczytu, lupa ×{ob["pow"]:.0f}): osie {pl(a[0])}/{pl(a[1])}/{pl(a[2])} w0, '
                f'wydłużenie ×{pl(a[0] / max(a[2], 1e-12))}')

    def _og_anim_tik(self):
        """Płynące strzałki pola: faza rośnie z czasem rzeczywistym; przerysowanie tylko okienka 11, gdy jest widoczne."""
        import time as _t
        if not self.vog.isVisible() or self.vog.visibleRegion().isEmpty():
            return
        self.eog.opcje['faza'] = self.eog_pp.opcje['faza'] = (_t.perf_counter() * self.sp_og_tempo.value()) % 1000.0
        self.vog.update()

    def _og_czas_zmiana(self, i):
        """Podkroki: reflektory w skali sfery włączane same (ta sama skala w uwadze i MLP, reflektory zawsze widoczne);
        powrót do osi uczenia wyłącza je, jeśli włączyło je to przełączenie."""
        if i == 1 and not self.ch_og_refl.isChecked():
            self._og_refl_auto = True; self.ch_og_refl.setChecked(True)
        elif i == 0 and getattr(self, '_og_refl_auto', False):
            self._og_refl_auto = False; self.ch_og_refl.setChecked(False)

    def _og_tytul(self, tekst):
        """Tytuł okienka 11 zależny od trybu (oś uczenia: blok 6; podkroki: bieżący blok i podkrok); zmiana tylko gdy inny."""
        dk = self.doki.get('11')
        if dk is not None and dk.windowTitle() != tekst:
            dk.setWindowTitle(tekst)

    def _og_tryb_pod(self):
        return hasattr(self, 'cb_og_czas') and self.cb_og_czas.currentIndex() == 1 and self.ch_og.isChecked()

    def _og_podkrok(self, s):
        """Okienko 11 w podkrokach: ognisko wybranej cząsteczki w podkroku s (dane przejścia przez sieć, układ '_n', skala sfery).
        MLP bloku b: 768 wiązek bloku, wkład c = (z_jc − z̄_c)·P v_c. Uwaga bloku b: 600 wkładów od innych cząsteczek,
        Σ_h (A^h_jk − ā^h_k)·u^h_k. W obu: suma wkładów porównana z prawdziwym ruchem stanu j w tym podkroku."""
        d = self.d_pod
        self.eog.ustaw_tryb_pod(True)
        if d is None or self.h is None or 'pos_n' not in d:
            self.eog._ognisko_czysc()
            self.lb11.setText('Podkroki: liczę przejście przez sieć dla tego checkpointu…'); self._zlec_podkroki(); return
        from laser_app.compute.common import kolory_kanalow
        j = int(self.sp_tok.value()); pos = d['pos_n']
        sc = 0.55 * self.es.R / max(float(np.percentile(np.linalg.norm(pos, axis=2), 99)), 1e-9)
        s = float(np.clip(s, 0, 12)); k = int(np.clip(np.floor(s), 0, 11)); f = s - k if s < 12 else 1.0
        if getattr(self, '_og_lb_pod', (None,))[0] != (id(d), j, sc):     # jedna długość wiązek na całe przejście (12 podkroków)
            ruchy = np.linalg.norm(np.diff(pos[:, j], axis=0), axis=1) * sc
            self._og_lb_pod = ((id(d), j, sc), 1.5 * float(ruchy.max()))
        Lst = self._og_lb_pod[1]
        kolej = self.ch_og_kolej.isChecked()
        ff, fc = (min(2 * f, 1.0), max(2 * f - 1, 0.0)) if kolej else (1.0, f)
        p0, p1 = pos[k, j] * sc, pos[k + 1, j] * sc; ruch = p1 - p0
        c = p0 + fc * ruch; b = k // 2
        if self.ch_og_tlo.isChecked():                                    # pozostałe cząsteczki w tle, w tej samej chwili podkroku
            kc = getattr(self, '_kol_cz', None)
            self.eog.ustaw_tlo(((1 - fc) * pos[k] + fc * pos[k + 1]) * sc, None if kc is None else kc[:, :3],
                               self.sp_og_jtlo.value(), bez=j)
        from laser_app.gpu.base import przenikanie, reflektory_przejscia
        fp = self._plynnie()
        kier_r, kol_r_, kl = reflektory_przejscia(d, s, fp)               # reflektory na sferze (jak sfera w podkrokach)
        if getattr(self, '_og_rf_pod', (None, None))[0] != kl:
            self._og_rf_pod = (kl, 50.0 * kier_r)
        kol_r = 0.25 + 0.75 * kol_r_
        wm, bm, ba = przenikanie(s, fp)
        if 1e-3 < wm < 0.999:                                             # płynne przejście: wiązki MLP i wkłady uwagi naraz
            self._og_przenikanie(d, j, sc, c, wm, bm, ba, kol_r, Lst)
            co = (f'przejście: blok {bm + 1} MLP {wm:.0%} ↔ blok {ba + 1} uwaga {1 - wm:.0%} '
                  '(wiązki jednego podkroku gasną, następnego się rozjaśniają)')
        elif k % 2 == 1 or wm >= 0.999:                                   # MLP bloku b: 768 wiązek
            b = bm
            pv = d['pv_n'][b] * sc; zc = d['z_c'][b, j].astype(np.float64)
            kol = kol_r
            wk = zc[:, None] * pv
            if self.cb_og_jasn.currentIndex() == 0:                         # wiązka wchodzi z kierunku, w który pcha
                kier, jas = wk, np.linalg.norm(wk, axis=1)
            else:                                                           # jak na sferze: kierunek v_c, błękit = ujemna dioda
                zz = d['z'][b, j].astype(np.float64); kier, jas = pv, np.abs(zz) * np.linalg.norm(pv, axis=1)
                kol = np.where((zz < 0)[:, None], np.array([0.30, 0.85, 1.0]), kol)
            self.eog.ustaw_ognisko(c, kier, wk, jas, kol, ruch=ruch, zrodla=self._og_rf_pod[1], Lb_staly=Lst,
                                   zapalone=float((d['z'][b, j] > 0.05).mean()), ujemne=float((d['z'][b, j] < -0.05).mean()))
            co = f'blok {b + 1}, MLP: 768 wiązek bloku zbiega się w cząsteczce'
        else:                                                             # uwaga bloku b: 600 wkładów od innych cząsteczek
            b = ba
            wk = np.einsum('hk,hkd->kd', d['ah_c'][b, :, j, :].astype(np.float64), d['uh_n'][b].astype(np.float64)) * sc
            kol = getattr(self, '_kol_cz', None)
            kol = kol[:, :3] if kol is not None else np.tile([0.35, 0.8, 1.0], (wk.shape[0], 1))
            self.eog.ustaw_ognisko(c, wk, wk, np.linalg.norm(wk, axis=1), kol, ruch=ruch, zrodla=pos[k] * sc, prosto=True,
                                   Lb_staly=Lst, refl_tlo=(self._og_rf_pod[1], kol_r))
            co = f'blok {b + 1}, uwaga: 600 wkładów od innych cząsteczek (tu przesuwają ją cząsteczki, nie lasery)'
        co += self._og_pole_podkroku(d, j, sc, s, fp, Lst)
        b = k // 2
        self._og_tytul(f'11. Pułapka optyczna: koloid w polu wiązek, podkroki — blok {b + 1}, {"MLP (768 laserów)" if k % 2 else "uwaga (600 cząsteczek)"}')
        if self.cb_og_sledz.currentIndex():
            from PyQt6.QtGui import QVector3D
            q = c * self.sp_skala_chm.value()
            self.vog.opts['center'] = QVector3D(float(q[0]), float(q[1]), float(q[2]))
            if self.cb_og_skala.currentIndex() and not self.ch_og_refl.isChecked():   # stały kadr: kula o promieniu długości wiązek
                R = Lst * self.sp_og_dlw.value() * self.sp_skala_chm.value() * 1.1
                w_, h_ = max(self.vog.width(), 1), max(self.vog.height(), 1)
                self.vog.opts['distance'] = R / (np.tan(np.radians(self.vog.opts['fov']) / 2) * min(1.0, h_ / w_))
            self.vog.update()
        iw = self.eog.info_wiazek or {}
        lam = iw.get('ksztalt', np.full(3, np.nan))
        blad = iw.get('blad')
        self.lb11.setText(f'krok {int(d["krok"])}, podkrok {pl(s, 2)}/12, cząsteczka {j}: {co};  suma wkładów = ruch w tym podkroku '
                          f'(błąd względny {blad:.1e});  skok = {iw.get("wypadkowa", 0) / max(iw.get("suma", 1), 1e-12):.0%} sumy długości wkładów;  '
                          f'kształt ogniska: rozkład kierunków {pl(lam[0])}/{pl(lam[1])}/{pl(lam[2])} (izotropowo 0,33), '
                          f'wydłużenie ×{pl(iw.get("wydluzenie", 1.0))}{self._og_opis_obrysu(iw)}' if blad is not None
                          else f'podkrok {pl(s, 2)}: brak danych')

    def _pomiary_start(self):
        if self.h is None:
            self.lb_pom.setText('Najpierw wczytaj historię uczenia.'); return
        w = getattr(self, '_pom_watek', None)
        if w is not None and w.isRunning():
            return
        w = TrapMeasurementsWorker(self.h, self.sp_tok.value(), self.sp_pom_co.value(), self.sp_pom_G.value())
        w.postep.connect(lambda i, n, t: (self.pasek_pom.setMaximum(n), self.pasek_pom.setValue(i), self.lb_pom.setText(t)))
        w.koniec.connect(lambda t: (self.lb_pom.setText(t), self.bt_pom.setEnabled(True), self.bt_pom_stop.setEnabled(False)))
        self._pom_watek = w; self.bt_pom.setEnabled(False); self.bt_pom_stop.setEnabled(True)
        self.lb_pom.setText(f'Liczę pułapkę cząsteczki {self.sp_tok.value()} w podkrokach…'); w.start()

    def _pomiary_stop(self):
        w = getattr(self, '_pom_watek', None)
        if w is not None:
            w.stop = True

    def _og_pole_podkroku(self, d, j, sc, s, fp, Lst):
        """Pole sił w kuli komórki cząsteczki j w podkrokach (mgła i strzałki jak w osi uczenia): zleca policzenie dla tego
        checkpointu (pole_podkrokow, w tle, ~0,3 s), a gotowe pokazuje w chwili s; w płynnym przejściu pole bieżącego i
        następnego podkroku przenikają się razem z wiązkami. Lupa (domyślnie): kula komórki pokazana w promieniu ułamka długości
        wiązek Lst (jedna skala ekranu dla kuli w każdym podkroku). Zwraca dopisek do opisu okienka."""
        if not (self.cb_og_strz.currentIndex() > 0 or self.ch_og_mgla.isChecked()):
            self.eog_pp.setVisible(False); return ''
        if getattr(self, '_baza_pod', None) is None:
            self.eog_pp.setVisible(False); return ''
        klucz = (str(self.h.wagi), int(d['krok']), j, int(self.sp_og_G.value()), round(float(self.sp_og_zas.value()), 2))
        if klucz != self._pp_klucz:
            self._pp_klucz = klucz
            if klucz in self._pp_cache:
                self._pp_pole = self._pp_cache[klucz]
            else:
                self._pp_pole = None
                dd = dict(d); dd['baza'] = self._baza_pod
                kc = getattr(self, '_kol_cz', None)
                self.watek_pp.zlec(klucz, self.h.wagi, int(d['krok']), self.h.rgb, j, dd, G=klucz[3], zasieg=klucz[4],
                                   kol_cz=None if kc is None else kc[:, :3])
        if self._pp_pole is None:
            self.eog_pp.setVisible(False)
            return '  [pole w komórce: liczę…]'
        w = self._pp_pole
        lupa = self.cb_og_lupa.currentIndex() == 0
        pow_ = (self.sp_og_lupa.value() * Lst / np.maximum(w['L'] * sc, 1e-30)) if lupa else np.ones(12)
        kl = (id(w), lupa, round(self.sp_og_lupa.value(), 3), round(float(Lst), 9), round(float(sc), 9))
        if kl != getattr(self, '_pp_pokaz', None):                       # kula pokazana w promieniu L · powiększenie (wokół x_k)
            self.eog_pp.ustaw_dane(dict(w, L=(w['L'] * pow_).astype(np.float32)), np.full(12, sc, np.float32))
            self._pp_pokaz = kl
        n = int(round(s)); t = float(min(np.floor(s), 11))              # pole podkroku k (stan przed podkrokiem), środek w x_k
        if fp > 0 and 1 <= n <= 11 and abs(s - n) < fp / 2:              # przenikanie pól na granicy podkroków (jak wiązki)
            u = (s - n + fp / 2) / fp; t = (n - 1) + u * u * (3 - 2 * u)
        self.eog_pp.ustaw_chwile(t, False); self.eog_pp.setVisible(True)
        k = int(t)
        return (f';  pole w komórce (promień {pl(float(w["L"][k]) / max(float(np.linalg.norm(w["skok"][k])), 1e-30))} skoku'
                + (f', pokazane w lupie ×{pow_[k]:.0f}' if lupa else '') +
                '): strzałki = dokąd wiązki pchnęłyby cząsteczkę postawioną w tym miejscu')

    def _pole_pod_gotowe(self, klucz, w):
        if 'blad_watku' in w:
            self.lb11.setText(f'Pole w podkrokach: błąd {w["blad_watku"]}'); return
        self._pp_cache[klucz] = w
        while len(self._pp_cache) > 8:
            self._pp_cache.pop(next(iter(self._pp_cache)))
        if klucz == self._pp_klucz and self.d_pod is not None:
            self._pp_pole = w
            self._og_opcje()

    def _og_przenikanie(self, d, j, sc, c, wm, bm, ba, kol_r, Lst):
        """Okienko 11, płynne przejście podkroków: 768 wiązek MLP bloku bm (jasność × wm) i 600 wkładów uwagi bloku ba
        (× 1 − wm) naraz; wkłady i skok ważone tak samo, więc suma wkładów = ważony ruch obu podkroków."""
        pos = d['pos_n']
        pv = d['pv_n'][bm] * sc; zc = d['z_c'][bm, j].astype(np.float64); wkM = zc[:, None] * pv
        if self.cb_og_jasn.currentIndex() == 0:
            kierM, jasM, kolM = wkM, np.linalg.norm(wkM, axis=1), kol_r
        else:
            zz = d['z'][bm, j].astype(np.float64); kierM, jasM = pv, np.abs(zz) * np.linalg.norm(pv, axis=1)
            kolM = np.where((zz < 0)[:, None], np.array([0.30, 0.85, 1.0]), kol_r)
        wkA = np.einsum('hk,hkd->kd', d['ah_c'][ba, :, j, :].astype(np.float64), d['uh_n'][ba].astype(np.float64)) * sc
        kolA = getattr(self, '_kol_cz', None)
        kolA = kolA[:, :3] if kolA is not None else np.tile([0.35, 0.8, 1.0], (wkA.shape[0], 1))
        jasA = np.linalg.norm(wkA, axis=1)
        ruchM = (pos[2 * bm + 2, j] - pos[2 * bm + 1, j]) * sc; ruchA = (pos[2 * ba + 1, j] - pos[2 * ba, j]) * sc
        nM, nA = len(wkM), len(wkA)
        self.eog.ustaw_ognisko(
            c, np.concatenate([kierM, wkA]), np.concatenate([wm * wkM, (1 - wm) * wkA]),
            np.concatenate([jasM / max(float(jasM.max()), 1e-12) * wm, jasA / max(float(jasA.max()), 1e-12) * (1 - wm)]),
            np.concatenate([np.asarray(kolM, np.float64)[:, :3], np.asarray(kolA, np.float64)]),
            ruch=wm * ruchM + (1 - wm) * ruchA, zrodla=np.concatenate([self._og_rf_pod[1], pos[2 * ba] * sc]),
            prosto=np.r_[np.zeros(nM, bool), np.ones(nA, bool)], Lb_staly=Lst, jas_wzgl=True)

    def _og_czas(self, t):
        tryb_pod = self.ch_pod.isChecked() and getattr(self, '_pp', None) is not None
        self.eog.setVisible(self.ch_og.isChecked())
        if not self.ch_og.isChecked():                                   # podgląd wyłączony: nic nie wczytujemy ani nie rysujemy
            self.es_og.setVisible(False)
            self.lb11.setText('Podgląd ogniskowej wyłączony. Zaznacz „pokaż podgląd ogniskowej” albo kliknij „Wczytaj / policz” '
                              '(sekcja „Pułapka optyczna”).')
            return
        if not self.widac('11') and not (self.ch_og_sfera.isChecked() and self.widac('s')):
            return                                                        # pułapki nigdzie nie widać: nic nie liczymy (_odslonieto)
        if self._og_tryb_pod():
            self.es_og.setVisible(False)
            self._og_podkrok(self.s_pod)
            return
        self.eog_pp.setVisible(False)
        self.eog.ustaw_tryb_pod(False)
        self._og_tytul('11. Pułapka optyczna: koloid w polu wiązek, oś uczenia — blok 6 (podkroki: „Czas okienka 11”)')
        d = self._og
        self.es_og.setVisible(bool(self.ch_og_sfera.isChecked() and d is not None and not tryb_pod))
        if d is None:
            self.lb11.setText('Pułapka optyczna: włącz „licz pułapkę wybranej cząsteczki” (sekcja „Pułapka optyczna”).')
            return
        self.eog.ustaw_chwile(t, self.ch_og_kolej.isChecked())
        if self.ch_og_tlo.isChecked():                                    # pozostałe cząsteczki w tle (jak koloid: ułamek fc)
            e_, h_ = self.eog, self.h
            P = ((1 - e_.fc) * h_.pos[e_.ka] * float(e_.skala_k[e_.ka]) + e_.fc * h_.pos[e_.kb] * float(e_.skala_k[e_.kb]))
            kc = getattr(self, '_kol_cz', None)
            self.eog.ustaw_tlo(P, None if kc is None else kc[:, :3], self.sp_og_jtlo.value(), bez=int(self.sp_tok.value()))
        else:
            self.eog.ustaw_tlo(None)
        if self.es_og.visible():
            self.es_og.ustaw_chwile(t, False)
        sl = self.cb_og_sledz.currentIndex()
        if sl:
            from PyQt6.QtGui import QVector3D
            sk = self.sp_skala_chm.value()
            p = (self.eog.pulapka, self.eog.koloid)[sl - 1]() * sk
            self.vog.opts['center'] = QVector3D(float(p[0]), float(p[1]), float(p[2]))
            skala = self.cb_og_skala.currentIndex()
            if skala and int(d['G']) > 1:                                   # kula opisana na sześcianie (wokół środka kadru) w okienku
                if skala == 1:                                               # sześcian zawsze tej samej wielkości
                    R = self.eog.pol_boku() + (np.linalg.norm(self.eog.pulapka() - self.eog.koloid()) if sl == 1 else 0.0)
                else:                                                        # jedna odległość kamery na całe uczenie
                    R = self._og_R_stale[0 if sl == 1 else 1]
                R *= sk * 1.05
                w_, h_ = max(self.vog.width(), 1), max(self.vog.height(), 1)
                self.vog.opts['distance'] = R / (np.tan(np.radians(self.vog.opts['fov']) / 2) * min(1.0, h_ / w_))
            self.vog.update()
        k = self.eog.k_teraz(); r99 = float(d['r99'][k]); zb = bool(d['zb'][k])
        top, topu = (d['top_ogn'][k], d['top_ogn_u'][k]) if zb else (d['top_kol'][k], d['top_kol_u'][k])
        wiazki = ', '.join(f'#{int(c)} {u:.1%}'.replace('.', ',') for c, u in zip(top[:3], topu[:3]))
        skok = float(np.linalg.norm(d['skok'][k])) / r99
        rk = float(d['r_kom'][k])
        if zb:
            odl = float(np.linalg.norm(d['ogn'][k] - d['srodek'][k])) / rk
            o_p = f'pułapka w komórce, {pl(odl)} r_kom od koloidu: {focal_trap.opis_lambda(d["lam"][k])}'
        else:
            lin = float(np.linalg.norm(d['ogn_lin'][k] - d['srodek'][k])) / rk
            o_p = (f'brak zera pola w komórce tokenu (oszacowanie liniowe {pl(lin, 1)} r_kom, szara strzałka do brzegu); '
                   f'„pułapka” = najsłabsze pole w komórce')
        o_p += f';  komórka r_kom = {pl(rk / r99, 3)} r99 (najbliższy sąsiad: token {int(d["sasiad"][k])})'
        iw = getattr(self.eog, 'info_wiazek', None) if self.ch_og_wiazki.isChecked() else None
        o_w = ''
        if iw:
            top_w = ', '.join(f'#{int(c)} {u:.1%}'.replace('.', ',') for c, u in zip(iw['top'], iw['udzial']))
            o_w = (f'768 wiązek: zapalonych {iw["zapalone"]:.0%}, ujemnych {iw["ujemne"]:.0%}; skok = {iw["wypadkowa"] / max(iw["suma"], 1e-12):.0%} '
                   f'sumy długości wkładów (reszta się znosi); najsilniejsze {top_w}{self._og_opis_obrysu(iw)};  ')
        ir = getattr(self.eog, 'info_ruch', None)
        if ir and self.ch_og_ruch.isChecked():
            kat = float(np.degrees(np.arccos(np.clip(ir['cos'], -1, 1))))
            o_w += (f'nadchodzący krok (zielony): ruch {pl(ir["dl"] / r99)} r99, kąt do skoku {kat:.0f}°, '
                    f'{pl(ir["stosunek"])} × długość skoku;  ')
        self.lb11.setText(f'krok {int(d["kroki"][k])}, cząsteczka {int(d["tok"])}:  {o_w}skok {pl(skok)} r99 (pomarańczowy);  '
                          f'kształt pola w koloidzie: {focal_trap.opis_lambda(d["lam0"][k])};  {o_p};  wiązki {"w pułapce" if zb else "w koloidzie"}: {wiazki}')
