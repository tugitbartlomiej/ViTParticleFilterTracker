"""Zakładka „Uczenie ViT: ruch cząsteczek po krokach”: oś czasu to krok uczenia, nie klatka filmu.

Ta sama klatka próbna, latent z każdego kroku optymalizatora (gęsta historia, analiza_uczenia), widoki
3D nad obrazem i przestrzeń makrowektorów (te same kierunki i kolory co w zakładce filmu) oraz wykresy
ruchu cząsteczek pod osią kroków: strata i mAP, prędkość tokenów w latencie i cząstek w przestrzeni makro,
odległość od stanu końcowego, rozdzielenie narzędzie / tło, efektywny rząd i mapa CKA krok x krok.
Detekcja i atencja pochodzą z najbliższego zapisanego checkpointu wag (co 10 kroków), co jest podpisane.

Baza PCA i środek układu liczone są raz z puli wszystkich kroków, a nie osobno w każdym kroku:
w jednym biegu kanał c w kroku t i t+1 to ten sam neuron, a PCA na krok dawałaby przeskoki osi,
które wyglądają jak ruch cząstek, a nim nie są (Vernier i in. 2020).
"""
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QProgressBar, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget,
                             QVBoxLayout, QWidget)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import analiza_uczenia as AU
import rownanie_czastki as RC
from wspolne import (ctrl_wcisniety, paleta_zaznaczenia, polacz_zaznaczenie, przelacz_w_zaznaczeniu, rysuj_zaznaczenie,
                     tokeny_w_prostokacie)
import przyczyny_ruchu as PR
from styl import rola
from uklad import Przeplyw, dodaj
from zakladka_detekcji import PlotnoCiemne
from zakladka_filmu import KOLOR_VIT, WatekFilmu, prostokat

KOL_N, KOL_T, KOL_P = '#ff5a4e', '#8fa3b8', '#ffffff'      # narzędzie, tło, wybrany patch


class ZakladkaUczenia(QWidget):
    def __init__(self, okno):
        super().__init__()
        self.okno = okno
        self.j128 = [w['kanal'] for w in okno.narzedzie]
        self.biegi = AU.znajdz_biegi()
        self.opis = self.Z = self.an = self.dety = None
        self.i_kl, self.t, self.token, self.watek = 0, 0, None, None
        self._ckpt_det = None
        lay = QVBoxLayout(self)

        # ---------------- góra: źródło historii
        g = QGroupBox('Historia uczenia (latent zapisany w każdym kroku optymalizatora)')
        w = Przeplyw(); g.setLayout(w); lay.addWidget(g)
        self.combo_bieg = QComboBox(); self.combo_bieg.setMinimumWidth(320)
        for b in self.biegi:
            K, N = b['ksztalt'][:2]
            self.combo_bieg.addItem(f"{b['nazwa']}  ({K} kroków, {N} klatek)")
        dodaj(w, 'Bieg:', self.combo_bieg, 'Biegi z plikiem LATENT_GESTY.npy [kroki, klatki, 600, 768] w experiments/*/wyniki/*/dane.\n'
              'Latent ostatniego bloku (MLP po GELU) w trybie eval, ten sam co w reszcie aplikacji.')
        self.combo_klatka = QComboBox(); self.combo_klatka.setMinimumWidth(300)
        dodaj(w, 'Klatka próbna:', self.combo_klatka, 'Stała klatka, na której zapisano latent w każdym kroku uczenia.\n'
              'W nawiasie: zbiór; obok liczba patchy narzędzia.')
        self.spin_co = QSpinBox(); self.spin_co.setRange(1, 50); self.spin_co.setValue(1)
        dodaj(w, 'Co ile kroków:', self.spin_co, 'Animacja i przestrzeń makro co tyle kroków (1 = każdy krok).\n'
              'Wykresy zawsze pokazują wszystkie kroki.')
        self.btn_wczytaj = rola(QPushButton('Wczytaj historię'), 'glowny')
        self.btn_wczytaj.setToolTip('Wczytuje latent klatki ze wszystkich kroków, liczy miary ruchu i detekcję z wag.\n'
                                    'Pierwszy raz około 30-50 s (kopia float16 trafia do app/cache_uczenia), potem kilka sekund.')
        w.addWidget(self.btn_wczytaj)
        self.pasek = QProgressBar(); self.pasek.setMaximumWidth(220); self.pasek.setTextVisible(False); w.addWidget(self.pasek)
        self.btn_przyczyny = rola(QPushButton('Licz przyczyny ruchu'), 'narzedzie')
        self.btn_przyczyny.setToolTip('Rozkłada ruch cząsteczek tej klatki w każdym przedziale między checkpointami wag\n'
                                      '(co 10 kroków) na przyczyny: grupy wag (łodyga, atencja i MLP każdego bloku),\n'
                                      'gradient straty względem latentu i atencję. Około 30 s. Wyniki w zakładce\n'
                                      '„Przyczyny ruchu” pod klatką i jako strzałki w przestrzeni makro.')
        w.addWidget(self.btn_przyczyny)
        self.etykieta_biegu = rola(QLabel('' if self.biegi else 'Brak biegów z gęstą historią w experiments/*/wyniki/*/dane.'),
                                   'opis')
        w.addWidget(self.etykieta_biegu)

        # ---------------- środek: klatka i wykresy | 3D | makro
        podzial = QSplitter(Qt.Orientation.Horizontal)
        lewa = QWidget(); ll = QVBoxLayout(lewa); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(4)
        self.fig = Figure(figsize=(5, 3.6)); self.ax = self.fig.add_axes([0, 0, 1, .9])
        self.canvas = PlotnoCiemne(self.fig); self.canvas.setFixedHeight(300)
        self.canvas.setToolTip('Klatka próbna: patche narzędzia (czerwone kropki), ramka ViT z wag najbliższego\n'
                               'checkpointu (zielona) i wybrany patch (biały). Klik wybiera patch.\n'
                               'Prawy przycisk + przeciągnij: zaznacz prostokąt patchy; ich cząstki w przestrzeni\n'
                               'makro zostają jasne, reszta blednie. Prawy klik bez przeciągania czyści zaznaczenie.\n'
                               'Ctrl + klik: dodaj albo usuń jeden patch; Ctrl + prawy przycisk + przeciągnij: dołącz prostokąt.\n'
                               'W przestrzeni makro to samo: Shift + przeciągnij lewym przyciskiem, Ctrl + klik w cząstkę.')
        ll.addWidget(self.canvas)
        self.zakl_wyk = QTabWidget(); ll.addWidget(self.zakl_wyk, 1)
        self.fig_r = Figure(figsize=(5, 7)); self.canvas_r = PlotnoCiemne(self.fig_r)
        self.canvas_r.setToolTip('Przebieg uczenia; pionowa linia = bieżący krok. Klik przechodzi do kroku.')
        self.zakl_wyk.addTab(self.canvas_r, 'Ruch cząsteczek')
        self.fig_c = Figure(figsize=(5, 5)); self.canvas_c = PlotnoCiemne(self.fig_c)
        self.canvas_c.setToolTip('Liniowe CKA tokenów klatki między krokami (1 = ta sama geometria 600 tokenów,\n'
                                 'niezależnie od obrotu i skali). Kwadratowe bloki = fazy uczenia. Klik: krok z osi x.')
        self.zakl_wyk.addTab(self.canvas_c, 'Mapa CKA krok × krok')
        strona = QWidget(); ls = QVBoxLayout(strona); ls.setContentsMargins(0, 0, 0, 0)
        wp = Przeplyw(); ls.addLayout(wp)
        self.combo_zbior_p = QComboBox(); self.combo_zbior_p.addItems(['patche narzędzia', 'patche tła', 'wybrany patch'])
        dodaj(wp, 'Wykresy dla:', self.combo_zbior_p, 'Po których cząstkach uśredniać wykresy przyczyn ruchu.')
        self.fig_p = Figure(figsize=(5, 7)); self.canvas_p = PlotnoCiemne(self.fig_p)
        self.canvas_p.setToolTip('Przyczyny ruchu cząsteczek w przedziałach między checkpointami wag. Klik: krok.')
        ls.addWidget(self.canvas_p, 1)
        self.zakl_wyk.addTab(strona, 'Przyczyny ruchu')
        self.fig_z = Figure(figsize=(5, 7)); self.canvas_z = PlotnoCiemne(self.fig_z)
        self.canvas_z.setToolTip('Każdy zaznaczony patch osobno przez całe uczenie (kolory jak na obrazie i w przestrzeni\n'
                                 'makro); biała przerywana = średnia zaznaczonych. Pionowa linia = bieżący krok, klik: krok.')
        self.zakl_wyk.addTab(self.canvas_z, 'Zaznaczone patche')
        podzial.addWidget(lewa)

        from czasteczki3d import WidokCzasteczek3D
        from przestrzen_makro import WidokPrzestrzeniMakro
        self.w3d = WidokCzasteczek3D(self.j128)
        self.makro = WidokPrzestrzeniMakro(self.j128)
        self.makro.ustaw_wzor(self.w3d)
        from PyQt6.QtWidgets import QStackedWidget
        self._stos = QStackedWidget(); self._stos.addWidget(self.w3d)      # główny panel: widok 3D, atlas albo krajobraz
        podzial.addWidget(self._stos); podzial.addWidget(self.makro)
        self.makro.osadz = self._osadz
        for b, opis in ((self.makro.btn_atlas, 'atlas kanałów'), (self.makro.btn_krajobraz, 'krajobraz i tunele')):
            b.setCheckable(True)
            b.setToolTip(b.toolTip() + f'\nW tej zakładce: przełącza główny panel (zamiast widoku 3D) na {opis};\n'
                                       'ponowny klik wraca do widoku 3D.')
        from panel_rownania import PanelRownania
        self.panel = PanelRownania(self)
        podzial.addWidget(self.panel)
        podzial.setStretchFactor(0, 0); podzial.setStretchFactor(1, 1); podzial.setStretchFactor(2, 1)
        podzial.setSizes([460, 520, 600, 520])
        self._podzial, self._podzielono = podzial, False
        lay.addWidget(podzial, 1)

        # ---------------- odtwarzanie po krokach
        w = QHBoxLayout(); lay.addLayout(w)
        self.btn_pocz = QPushButton('⏮ Krok 0'); self.btn_wstecz = QPushButton('◀ Krok')
        self.btn_graj = rola(QPushButton('▶ Ucz (odtwarzaj)'), 'glowny'); self.btn_graj.setCheckable(True)
        self.btn_graj.setMinimumWidth(150)
        self.btn_dalej = QPushButton('Krok ▶'); self.btn_kon = QPushButton('Koniec ⏭')
        for b, opis in ((self.btn_pocz, 'Pierwszy krok (wagi losowe).'), (self.btn_wstecz, 'Poprzedni krok.'),
                        (self.btn_graj, 'Odtwarza uczenie krok po kroku.'), (self.btn_dalej, 'Następny krok.'),
                        (self.btn_kon, 'Ostatni krok uczenia.')):
            b.setToolTip(opis); w.addWidget(b)
        self.suwak = QSlider(Qt.Orientation.Horizontal); self.suwak.setRange(0, 0); w.addWidget(self.suwak, 1)
        self.spin_fps = QDoubleSpinBox(); self.spin_fps.setRange(.5, 10000); self.spin_fps.setDecimals(1)
        self.spin_fps.setValue(10)
        dodaj(w, 'kroków/s', self.spin_fps, 'Szybkość odtwarzania uczenia (pozycji suwaka na sekundę), do 10 000.\n'
              'Ekran odświeża się najwyżej kilkadziesiąt razy na sekundę; przy większej szybkości odtwarzanie\n'
              'przeskakuje kroki tak, żeby zgadzał się rzeczywisty czas (np. 300 kroków/s: całe uczenie w około 3 s).\n'
              'Przy szybkim odtwarzaniu warto zmniejszyć „Przelot” w przestrzeni makro.')
        self.chk_petla = QCheckBox('pętla'); self.chk_petla.setChecked(True); w.addWidget(self.chk_petla)

        w = QHBoxLayout(); lay.addLayout(w)
        self.combo_por = QComboBox(); self.combo_por.addItems(['brak', 'z poprzednim krokiem', 'z krokiem 0', 'ze stanem końcowym'])
        dodaj(w, 'Porównanie w 3D:', self.combo_por, 'Co jest „PRZED” w widoku 3D; „PO” = bieżący krok.')
        self.combo_srodek = QComboBox()
        self.combo_srodek.addItems(['średnia ze wszystkich kroków', 'stan końcowy (cząstki zbiegają do środka)', 'krok 0'])
        dodaj(w, 'Środek makro:', self.combo_srodek,
              'Względem czego liczone jest położenie cząstki w przestrzeni makrowektorów:\n'
              '- średnia ze wszystkich kroków (i tokenów): zwykły widok;\n'
              '- stan końcowy: cząstka = s · (z_t − z_końcowy tego patcha) · D, więc na końcu uczenia wszystkie\n'
              '  cząstki są w środku układu; widać, jak każdy patch dochodzi do swojego stanu końcowego;\n'
              '- krok 0: cząstki startują ze środka i rozchodzą się w miarę uczenia.')
        self.combo_sledz = QComboBox()
        self.combo_sledz.addItems(['wyłączone', 'czubek narzędzia (z etykiety)', 'najpewniejszy patch ViT', 'najszybszy patch w kroku'])
        self.combo_sledz.setCurrentIndex(1)
        dodaj(w, 'Śledź patch:', self.combo_sledz,
              'Automatyczny wybór patcha w każdym kroku; ręczny klik wyłącza.\n'
              '- czubek narzędzia: patch, w którym leży cystotome_tip z annotation.csv tej klatki (czerwone kółko\n'
              '  na obrazie); klatka jest stała, więc to ten sam patch w każdym kroku; bez etykiety: najpewniejszy ViT;\n'
              '- najpewniejszy patch ViT: z wag najbliższego checkpointu (środek wykrytego narzędzia);\n'
              '- najszybszy patch: największa zmiana latentu w tym kroku.')
        self.chk_strzalki = QCheckBox('Strzałki przyczyn w makro'); self.chk_strzalki.setChecked(True)
        self.chk_strzalki.setToolTip('Dla wybranej cząstki w przestrzeni makro, z jej bieżącego położenia (po „Licz przyczyny”):\n'
                                     '- biała: rzeczywisty ruch w bieżącym przedziale wag (np. krok 500 -> 510);\n'
                                     '- kolorowe: wkład każdej grupy wag (łodyga, atencja i MLP bloków; kolory jak w\n'
                                     '  wykresie przyczyn), ich suma ≈ biała strzałka;\n'
                                     '- żółta przerywana: kierunek −gradient straty względem latentu tej klatki, tej\n'
                                     '  samej długości co biała (dokąd strata „chciałaby” przesunąć cząstkę).')
        w.addWidget(self.chk_strzalki)
        self.spin_strz = QDoubleSpinBox(); self.spin_strz.setRange(.1, 100); self.spin_strz.setValue(3)
        dodaj(w, 'x', self.spin_strz, 'Mnożnik długości strzałek przyczyn (ruch w przedziale 10 kroków bywa krótki).')
        self.chk_widok3d = QCheckBox('widok 3D'); self.chk_widok3d.setChecked(True)
        self.chk_widok_makro = QCheckBox('makrowektory'); self.chk_widok_makro.setChecked(True)
        self.chk_widok_rown = QCheckBox('równanie'); self.chk_widok_rown.setChecked(True)
        self.chk_widok_rown.setToolTip('Czwarty panel: równanie Langevina, GLE i Koopmana wybranych cząstek z ich własnej historii.')
        w.addWidget(QLabel('Widoki:')); w.addWidget(self.chk_widok3d); w.addWidget(self.chk_widok_makro)
        w.addWidget(self.chk_widok_rown)
        self.etykieta_kroku = QLabel(''); self.etykieta_kroku.setWordWrap(True); w.addWidget(self.etykieta_kroku, 1)

        g = QGroupBox('Prognoza Langevina i ślad toru w przestrzeni makro')
        w = Przeplyw(); g.setLayout(w); lay.addWidget(g)
        self.chk_prognoza = QCheckBox('Prognoza Langevina'); self.chk_prognoza.setChecked(True)
        self.chk_prognoza.setToolTip(
            'Liniowy model ruchu cząstki z bezwładnością (dyskretny Langevin, oscylator):\n'
            '  v(t+1) = beta v(t) − kappa (z(t) − m(t)),  z(t+1) = z(t) + v(t+1),  m = średnia ostatnich 50 kroków.\n'
            'beta i kappa dopasowane na krokach 300-700 tej historii (dla średniej: na torze średniej).\n'
            'Kolory w przestrzeni makro:\n'
            '- pomarańczowa przerywana: prognoza z kroku startu na h kroków (tylko ze stanów do startu);\n'
            '- bledsze pomarańczowe przerywane: poprzednie prognozy;\n'
            '- pomarańczowa kropka: gdzie prognoza stawiała cząstkę w bieżącym kroku;\n'
            '- turkusowa ciągła: rzeczywisty ślad cząstki (przeszłe kroki);\n'
            '- pomarańczowa ciągła: ślad prognoz z wyprzedzeniem (gdzie model stawiał cząstkę w każdym kroku);\n'
            '- jasnoniebieska blada: rzeczywisty dalszy tor do końca prognozy (podgląd do porównania).')
        w.addWidget(self.chk_prognoza)
        self.combo_czastka = QComboBox()
        self.combo_czastka.addItems(['wybranego patcha', 'średniej cząstek narzędzia (jedna kropka)', 'średniej cząstek tła (jedna kropka)',
                                     'cząstek z listy (panel równania)', 'zaznaczonych patchy (każdy osobno)',
                                     'średniej zaznaczonych patchy (jedna kropka)'])
        dodaj(w, 'dla', self.combo_czastka,
              'Której cząstki dotyczą prognoza i ślad:\n'
              '- wybrany patch (kliknięty albo śledzony, np. czubek);\n'
              '- średnia cząstek narzędzia: średni latent wszystkich patchy narzędzia w każdym kroku, rysowany jako\n'
              '  jedna duża czerwona kropka (rzut jest liniowy, więc to dokładnie środek ciężkości czerwonych cząstek);\n'
              '  dla niej osobno dopasowany oscylator (beta, kappa średniej);\n'
              '- średnia cząstek tła: to samo dla patchy tła (szara kropka);\n'
              '- cząstki z listy panelu równania (kolory z panelu);\n'
              '- zaznaczone patche, każdy w swoim kolorze (najwyżej tyle, ile „prognoza najwyżej dla”);\n'
              '- średnia zaznaczonych patchy jako jedna żółta kropka.')
        self.spin_h = QSpinBox(); self.spin_h.setRange(2, 500); self.spin_h.setValue(30)
        dodaj(w, 'na', self.spin_h, 'Na ile kroków do przodu prognoza.')
        self.spin_nowa = QSpinBox(); self.spin_nowa.setRange(1, 500); self.spin_nowa.setValue(10)
        dodaj(w, 'nowa co', self.spin_nowa,
              'Co ile kroków startuje nowa prognoza. Między startami prognoza stoi w miejscu, a cząstka idzie\n'
              'dalej, więc widać, czy podąża za przewidzianym torem. 1 = nowa prognoza w każdym kroku.')
        self.spin_stare = QSpinBox(); self.spin_stare.setRange(0, 30); self.spin_stare.setValue(3)
        dodaj(w, 'stare prognozy:', self.spin_stare, 'Ile poprzednich prognoz zostawić na ekranie (coraz bledsze).')
        self.chk_przyszlosc = QCheckBox('rzeczywisty dalszy tor'); self.chk_przyszlosc.setChecked(True)
        self.chk_przyszlosc.setToolTip('Blada jasnoniebieska linia: prawdziwy tor od bieżącego kroku do końca prognozy.\n'
                                       'Tylko do porównania na ekranie; prognoza go nie używa.')
        w.addWidget(self.chk_przyszlosc)
        self.spin_slad = QSpinBox(); self.spin_slad.setRange(0, 1000); self.spin_slad.setValue(100)
        dodaj(w, 'Ślad (kroków):', self.spin_slad, 'Długość śladu rzeczywistego i śladu prognoz. 0 = bez śladu.')
        self.chk_slad_caly = QCheckBox('od kroku 0'); self.chk_slad_caly.setToolTip('Ślad przez całe uczenie od kroku 0.')
        w.addWidget(self.chk_slad_caly)
        self.spin_wyprz = QSpinBox(); self.spin_wyprz.setRange(0, 200); self.spin_wyprz.setValue(10)
        dodaj(w, 'ślad prognoz z wyprzedzeniem', self.spin_wyprz,
              'Pomarańczowa ciągła linia: dla każdego kroku t śladu położenie, które model przewidział z kroku\n'
              't − tyle kroków. Obok turkusowego śladu rzeczywistego widać, gdzie model się myli. 0 = bez.')
        self.chk_ema = QCheckBox('EMA'); self.chk_ema.setChecked(False)
        self.chk_ema.setToolTip('Wygładza oba ślady średnią wykładniczą po krokach (przyczynowo, od kroku 0):\n'
                                'ślad(t) = α · tor(t) + (1 − α) · ślad(t − 1). Tylko rysunek; prognoza liczy się z surowych stanów.\n'
                                'Wygładzony ślad kończy się trochę za cząstką (opóźnienie około 1/α kroków).')
        w.addWidget(self.chk_ema)
        self.spin_ema = QDoubleSpinBox(); self.spin_ema.setRange(.01, 1); self.spin_ema.setSingleStep(.05)
        self.spin_ema.setValue(.2)
        dodaj(w, 'α', self.spin_ema, 'Siła wygładzania: 0,5 słabo, 0,2 średnio (pamięć około 5 kroków), 0,05 mocno (około 20).')
        self.combo_zanik = QComboBox(); self.combo_zanik.addItems(['liniowe', 'wykładnicze', 'brak (stała jasność)'])
        dodaj(w, 'Zanikanie śladu:', self.combo_zanik, 'Jak blednie starsza część śladu.')
        self.spin_krycie = QDoubleSpinBox(); self.spin_krycie.setRange(0, 1); self.spin_krycie.setSingleStep(.05)
        self.spin_krycie.setValue(.15)
        dodaj(w, 'krycie najstarszego', self.spin_krycie, 'Krycie najstarszego odcinka śladu (najnowszy: 1).')

        g = QGroupBox('Zaznaczone patche: śledzenie wybranych cząstek przez uczenie')
        w = Przeplyw(); g.setLayout(w); lay.addWidget(g)
        self.etykieta_zazn = QLabel('Brak zaznaczenia. Ctrl + klik na obrazie albo w przestrzeni makro dodaje patch.')
        w.addWidget(self.etykieta_zazn)
        self.chk_tylko_zazn = QCheckBox('śledź tylko zaznaczone')
        self.chk_tylko_zazn.setToolTip('Przestrzeń makro rysuje tylko zaznaczone cząstki i ich ślady (reszta znika), a prognoza\n'
                                       'Langevina przełącza się na „zaznaczonych patchy (każdy osobno)”. Odtwarzaj uczenie i\n'
                                       'patrz, jak zmieniają się tylko te patche; ich przebiegi są w karcie „Zaznaczone patche”.')
        w.addWidget(self.chk_tylko_zazn)
        self.chk_kol_zazn = QCheckBox('osobne kolory'); self.chk_kol_zazn.setChecked(True)
        self.chk_kol_zazn.setToolTip('Każdy zaznaczony patch w swoim kolorze, tym samym na obrazie, w przestrzeni makro i na\n'
                                     'wykresach. Wyłączone: żółta nakładka i kolory wg „Kolor cząstek” w makro.')
        w.addWidget(self.chk_kol_zazn)
        self.btn_zazn_wybrany = QPushButton('+ wybrany patch')
        self.btn_zazn_wybrany.setToolTip('Dodaje do zaznaczenia bieżący wybrany (albo śledzony) patch.')
        self.btn_zazn_narz = QPushButton('patche narzędzia')
        self.btn_zazn_narz.setToolTip('Zaznacza wszystkie patche w obrysie narzędzia tej klatki (z Ctrl: dołącza).')
        self.btn_zazn_czysc = QPushButton('wyczyść')
        for b in (self.btn_zazn_wybrany, self.btn_zazn_narz, self.btn_zazn_czysc):
            w.addWidget(b)
        self.spin_zazn_prog = QSpinBox(); self.spin_zazn_prog.setRange(1, 60); self.spin_zazn_prog.setValue(8)
        dodaj(w, 'prognoza najwyżej dla', self.spin_zazn_prog,
              'Dla ilu pierwszych zaznaczonych patchy liczyć prognozę i ślad Langevina (każdy ma osobny model,\n'
              'więc przy wielu cząstkach krok się wydłuża). Ślady przestrzeni makro mają wszystkie zaznaczone.')
        self.timer = QTimer(self); self.timer.timeout.connect(self._nastepny)

        # ---------------- sygnały
        self.combo_bieg.currentIndexChanged.connect(self._zmiana_biegu)
        self.btn_wczytaj.clicked.connect(self._wczytaj)
        self.suwak.valueChanged.connect(self.idz)
        self.btn_pocz.clicked.connect(lambda: self.suwak.setValue(0))
        self.btn_kon.clicked.connect(lambda: self.suwak.setValue(self.suwak.maximum()))
        self.btn_wstecz.clicked.connect(lambda: self.suwak.setValue(self.suwak.value() - 1))
        self.btn_dalej.clicked.connect(lambda: self.suwak.setValue(self.suwak.value() + 1))
        self.btn_graj.toggled.connect(self._graj)
        self.spin_fps.valueChanged.connect(lambda *_: self._start_zegara() if self.timer.isActive() else None)
        self.combo_por.currentIndexChanged.connect(lambda *_: self.idz(self.t))
        self.combo_sledz.currentIndexChanged.connect(lambda *_: self.idz(self.t))
        self.combo_srodek.currentIndexChanged.connect(lambda *_: self._ustaw_makro())
        self.spin_co.valueChanged.connect(lambda *_: self._po_wczytaniu(None) if self.Z is not None else None)
        self.chk_widok3d.toggled.connect(lambda v: (self._stos.setVisible(v), self.idz(self.t)))
        self.chk_widok_makro.toggled.connect(lambda v: (self.makro.setVisible(v), self.idz(self.t)))
        self.chk_widok_rown.toggled.connect(lambda v: self.panel.setVisible(v))
        for sg in (self.panel.combo_model.currentIndexChanged, self.panel.edit.editingFinished, self.panel.spin_L.valueChanged,
                   self.panel.spin_d.valueChanged, self.panel.spin_r.valueChanged, self.panel.spin_od.valueChanged,
                   self.panel.spin_do.valueChanged):
            sg.connect(lambda *_: self.idz(self.t) if self.Z is not None else None)
        self.w3d.widok.klik.connect(lambda *_: self._wybierz_token(self.w3d.token))
        self.makro.token_wybrany.connect(self._wybierz_token)     # klik w cząstkę: ten sam patch na obrazie i w 3D
        self.canvas.mpl_connect('button_press_event', self._klik_klatki)
        self.canvas.mpl_connect('button_release_event', self._puszczenie_klatki)
        self.makro.zaznaczono.connect(self._zaznacz)               # Shift + przeciągnij w przestrzeni makro
        self.zaznaczenie = None
        self.chk_tylko_zazn.toggled.connect(self._tylko_zazn)
        self.chk_kol_zazn.toggled.connect(lambda *_: self._zaznacz(self.zaznaczenie))
        self.btn_zazn_wybrany.clicked.connect(
            lambda: self._zaznacz(polacz_zaznaczenie(self.zaznaczenie, [self.token], True)) if self.token is not None else None)
        self.btn_zazn_narz.clicked.connect(
            lambda: self._zaznacz(polacz_zaznaczenie(self.zaznaczenie, np.flatnonzero(self.narz), ctrl_wcisniety()))
            if self.Z is not None else None)
        self.btn_zazn_czysc.clicked.connect(lambda: self._zaznacz([]))
        self.spin_zazn_prog.valueChanged.connect(lambda *_: self.idz(self.t) if self.combo_czastka.currentIndex() == 4 else None)
        self.canvas_z.mpl_connect('button_press_event', self._klik_wykresu)
        self._zazn_miary = (None, None)
        self.canvas_r.mpl_connect('button_press_event', self._klik_wykresu)
        self.canvas_c.mpl_connect('button_press_event', self._klik_cka)
        self.canvas_p.mpl_connect('button_press_event', self._klik_cka)
        self.btn_przyczyny.clicked.connect(self._licz_przyczyny)
        self.combo_zbior_p.currentIndexChanged.connect(lambda *_: self._rysuj_przyczyny())
        self.chk_strzalki.toggled.connect(lambda *_: self.idz(self.t))
        self.spin_strz.valueChanged.connect(lambda *_: self.idz(self.t))
        self.chk_prognoza.toggled.connect(lambda *_: self.idz(self.t))
        for sg in (self.spin_h.valueChanged, self.combo_czastka.currentIndexChanged, self.spin_nowa.valueChanged,
                   self.spin_stare.valueChanged, self.chk_przyszlosc.toggled, self.spin_slad.valueChanged,
                   self.chk_slad_caly.toggled, self.spin_wyprz.valueChanged, self.chk_ema.toggled, self.spin_ema.valueChanged,
                   self.combo_zanik.currentIndexChanged, self.spin_krycie.valueChanged):
            sg.connect(lambda *_: self.idz(self.t))
        self._tory = {}; self.czubek = None; self._opis_lang = ''
        # „Animuj” w widoku 3D: bez porównania PRZED / PO odtwarza uczenie (sam widok 3D nie miałby czego animować)
        self.w3d.btn_anim.toggled.disconnect()
        self.w3d.btn_anim.toggled.connect(self._animuj_3d)
        self.w3d.btn_anim.setToolTip('Bez „Porównania w 3D”: odtwarza uczenie krok po kroku (jak „Ucz”).\n'
                                     'Z porównaniem: płynne przejście PRZED -> PO w pętli.')
        self.btn_graj.toggled.connect(lambda v: (self.w3d.btn_anim.blockSignals(True), self.w3d.btn_anim.setChecked(
            v or self.w3d.timer.isActive()), self.w3d.btn_anim.blockSignals(False)))
        self.prz = self.P_prz = None; self._delty = (None, None)
        self._zmiana_biegu()

    # ------------------------------------------------------------ wczytanie
    def showEvent(self, e):
        super().showEvent(e)
        if not self._podzielono:
            self._podzielono = True
            QTimer.singleShot(0, lambda: self._podzial.setSizes(
                [420, (self._podzial.width() - 420) * 3 // 10, (self._podzial.width() - 420) * 4 // 10,
                 (self._podzial.width() - 420) * 3 // 10]))

    def _status(self, t, ms=8000):
        self.okno.statusBar().showMessage(t, ms)

    def _bieg(self):
        i = self.combo_bieg.currentIndex()
        return self.biegi[i] if 0 <= i < len(self.biegi) else None

    def _zmiana_biegu(self, *_):
        b = self._bieg()
        self.combo_klatka.clear()
        if b is None:
            return
        o = AU.opis_biegu(b)
        for i, op in enumerate(o['opisy']):
            self.combo_klatka.addItem(f'{i}: {op}, {int(o["narzedzie"][i].sum())} patchy narzędzia')
        self.etykieta_biegu.setText(f'kroki {int(o["kroki"][0])}..{int(o["kroki"][-1])}, wagi co '
                                    f'{int(np.diff(o["kroki_wag"]).min()) if len(o["kroki_wag"]) > 1 else "?"} kroków '
                                    f'({len(o["wagi"])} checkpointów), ziarno {o["ziarno"]}')

    def _wczytaj(self):
        b = self._bieg()
        if b is None or self.watek:
            return
        i = max(0, self.combo_klatka.currentIndex())
        self.btn_graj.setChecked(False)

        def zadanie(wt):
            o = AU.opis_biegu(b)
            Z = AU.wczytaj_klatke(b, i, lambda k, n: wt.postep.emit(int(40 * k / n), f'latent: {k}/{n} kroków'))
            an = AU.analiza(Z, o['narzedzie'][i], o['tlo'][i],
                            lambda k, n: wt.postep.emit(40 + int(30 * k / n), f'miary ruchu: {k}/{n} kroków'))
            rgb = np.ascontiguousarray(o['obrazy'][i])
            d = AU.detekcje(o, rgb, lambda k, n: wt.postep.emit(70 + int(30 * k / n), f'detekcja z wag: {k}/{n}'))
            return dict(opis=o, Z=Z, an=an, dety=d, i=i, rgb=rgb, osc=AU.oscylator(Z), czubki=AU.czubki_biegu(b))

        self.watek = WatekFilmu(zadanie, self)
        self.pasek.setRange(0, 100)
        self.watek.postep.connect(lambda k, t: (self.pasek.setValue(k), self._status(t, 3000)))
        self.watek.gotowy.connect(lambda w: (setattr(self, 'watek', None), self._po_wczytaniu(w)))
        self.watek.blad.connect(lambda t: (setattr(self, 'watek', None), QMessageBox.critical(self, 'Błąd', t[-2000:])))
        self._status(f'wczytuję historię uczenia klatki {i}...')
        self.watek.start()

    def _po_wczytaniu(self, w):
        if w is not None:
            self.opis, self.Z, self.an, self.dety, self.i_kl, self.rgb = w['opis'], w['Z'], w['an'], w['dety'], w['i'], w['rgb']
            self.osc = w['osc']; self._tory = {}
            cz = w.get('czubki') or []
            self.czubek = cz[self.i_kl] if self.i_kl < len(cz) else None
            self.det = self.dety.pop('det'); self._ckpt_det = len(self.opis['wagi']) - 1
            self.narz = self.opis['narzedzie'][self.i_kl]
            self.token = None
            self.prz = self.P_prz = None; self._delty = (None, None); self.fig_p.clear(); self.canvas_p.draw_idle()
        co = self.spin_co.value()
        self.idx = np.arange(0, len(self.Z), co)
        if self.idx[-1] != len(self.Z) - 1:
            self.idx = np.append(self.idx, len(self.Z) - 1)                  # zawsze z krokiem końcowym
        self.kroki = self.opis['kroki']
        Zs = self.Z[self.idx]
        self.w3d.ustaw_baze_pca(Zs[::max(1, len(Zs) // 200)])                 # wspólna baza PCA z puli kroków
        self.makro.ckpt = self.opis['wagi'][-1] if self.opis['wagi'] else None
        self._Zs = Zs
        self._ustaw_makro(rysuj=False)
        self.suwak.blockSignals(True); self.suwak.setRange(0, len(self.idx) - 1); self.suwak.setValue(0)
        self.suwak.blockSignals(False)
        self._rysuj_wykresy(); self._rysuj_cka()
        self._zazn_miary = (None, None); self._rysuj_zazn()
        self.idz(0)
        a = self.an
        self._status(f'Historia: {len(self.Z)} kroków, klatka {self.i_kl}; animacja co {co} krok(ów). '
                     f'Względna zmiana latentu na krok: mediana {np.median(a["wzgl"][1:]) * 100:.2f} %.', 15000)

    def _ustaw_makro(self, rysuj=True):
        if self.Z is None:
            return
        i = self.combo_srodek.currentIndex()
        m = self.makro
        m.odniesienie = None if i == 0 else np.asarray(self.Z[-1 if i == 1 else 0], np.float32)
        if i and m.chk_srodek.isChecked():
            m.chk_srodek.blockSignals(True); m.chk_srodek.setChecked(False); m.chk_srodek.blockSignals(False)
        maski = np.repeat(self.narz[None], len(self.idx), 0)
        pew = self.dety['mapy_pew'][self._ckpt_kroku(self.idx)]
        m.ustaw_sekwencje(self._Zs, maski, pew)
        dP = np.linalg.norm(np.diff(m.P, axis=0), axis=-1) / np.maximum(np.diff(self.idx), 1)[:, None]
        self.v_makro = np.concatenate([np.zeros((1, 600), np.float32), dP]).astype(np.float32)   # [len(idx), 600]
        if rysuj:
            self._rysuj_wykresy(); self._rysuj_zazn(); self.idz(self.t)

    def _ckpt_kroku(self, k):
        """Indeks najbliższego checkpointu wag dla kroku (albo tablicy kroków) k."""
        kw = self.opis['kroki_wag']
        k = np.asarray(self.kroki)[np.asarray(k)]
        return np.abs(kw[None, :] - np.atleast_1d(k)[:, None]).argmin(1) if np.ndim(k) else int(np.abs(kw - k).argmin())

    def _atencja(self, k_ckpt, blok):
        """Atencja klatki próbnej z wag checkpointu k_ckpt (ładowanie wag tylko przy zmianie checkpointu)."""
        import torch
        if self._ckpt_det != k_ckpt:
            ck = torch.load(self.opis['wagi'][k_ckpt], map_location='cpu', weights_only=True)
            self.det.net.load_state_dict(ck.get('net', ck), strict=True)
            self._ckpt_det = k_ckpt
        return self.det.atencja(self.rgb, blok)

    # ------------------------------------------------------------ odtwarzanie
    def _graj(self, wl):
        if wl and self.Z is not None:
            self.btn_graj.setText('⏸ Pauza')
            self._start_zegara()
        else:
            self.btn_graj.setText('▶ Ucz (odtwarzaj)'); self.timer.stop()
            if wl:
                self.btn_graj.setChecked(False)

    def _animuj_3d(self, wl):
        if self.combo_por.currentIndex() and self.w3d.po is not None:
            self.w3d._animuj(wl)
            return
        self.w3d.timer.stop()
        if self.Z is None:
            self.w3d.btn_anim.blockSignals(True); self.w3d.btn_anim.setChecked(False); self.w3d.btn_anim.blockSignals(False)
            self._status('Najpierw „Wczytaj historię”.')
            return
        self.btn_graj.setChecked(wl)

    def _start_zegara(self):
        import time
        self._t_start, self._poz_start, self._ostatni = time.perf_counter(), self.t, self.t
        self.timer.setInterval(max(15, int(1000 / self.spin_fps.value()))); self.timer.start()

    def _osadz(self, nazwa, widzet):
        """Przyciski przestrzeni makro „Atlas kanałów” i „Krajobraz i tunele”: pokazują widok w głównym panelu
        zakładki zamiast widoku 3D; ponowny klik (albo klik drugiego) wraca do widoku 3D / przełącza."""
        if self._stos.indexOf(widzet) < 0:
            self._stos.addWidget(widzet)
        wraca = self._stos.currentWidget() is widzet
        self._stos.setCurrentWidget(self.w3d if wraca else widzet)
        self.makro.btn_atlas.setChecked(not wraca and nazwa == 'atlas')
        self.makro.btn_krajobraz.setChecked(not wraca and nazwa == 'krajobraz')
        if not self.chk_widok3d.isChecked():
            self.chk_widok3d.setChecked(True)
        if wraca:
            if nazwa == 'atlas':
                self.makro.podswietl_kanal(None)
            self.idz(self.t)
        else:
            widzet.odswiez(True)

    def _nastepny(self):
        import time
        if getattr(self, '_ostatni', None) != self.t:                      # ręczne przesunięcie suwaka: zegar od nowa
            self._t_start, self._poz_start = time.perf_counter(), self.t
        cel = self._poz_start + int((time.perf_counter() - self._t_start) * self.spin_fps.value())   # wg zegara
        t = max(self.t + 1, cel)
        if t >= len(self.idx):
            if not self.chk_petla.isChecked():                              # koniec: pokaż ostatni krok i stop
                self.btn_graj.setChecked(False); self.suwak.setValue(len(self.idx) - 1); return
            if self.t < len(self.idx) - 1:                                  # w pętli zawsze też ostatni krok
                t = len(self.idx) - 1
            else:
                self._t_start, self._poz_start = time.perf_counter(), 0
                t = 0
        self._ostatni = t
        self.suwak.setValue(t)

    def _wybierz_token(self, tok):
        if tok is None:
            return
        if self.combo_sledz.currentIndex():
            self.combo_sledz.blockSignals(True); self.combo_sledz.setCurrentIndex(0); self.combo_sledz.blockSignals(False)
        self.token = int(tok); self.w3d.token = self.token
        self._rysuj_wykresy()
        if self.prz is not None and self.combo_zbior_p.currentIndex() == 2:
            self._rysuj_przyczyny()
        self.idz(self.t)

    def idz(self, t):
        if self.Z is None:
            return
        self.t = t = int(np.clip(t, 0, len(self.idx) - 1))
        if self.suwak.value() != t:
            self.suwak.blockSignals(True); self.suwak.setValue(t); self.suwak.blockSignals(False)
        k = int(self.idx[t])                                     # indeks kroku w historii
        kc = self._ckpt_kroku(k)
        s = self.combo_sledz.currentIndex()
        if s == 1 and self.czubek is not None:
            x, y = self.czubek
            self.token = int(np.clip(y // 16, 0, 19)) * 30 + int(np.clip(x // 16, 0, 29))
        elif s in (1, 2):
            self.token = int(np.argmax(self.dety['mapy_pew'][kc]))
        elif s == 3:
            self.token = int(np.argmax(self.an['v'][k]))
        self.w3d.token = self.token
        z = np.asarray(self.Z[k], np.float32)
        por = self.combo_por.currentIndex()
        przed = None if por == 0 else np.asarray(self.Z[(int(self.idx[max(0, t - 1)]), 0, -1)[por - 1]], np.float32)
        fn = lambda b, kc=kc: self._atencja(kc, b)
        if self.chk_widok3d.isChecked() and self._stos.currentWidget() is self.w3d:
            obrysy = [(prostokat(self.dety['ramka'][kc]), KOLOR_VIT)]
            self.w3d.ustaw_obrysy(obrysy, self.narz)
            if przed is None:
                self.w3d.pokaz(z, None, self.rgb, fn)
            else:
                if self.w3d.combo_tryb.currentIndex() == 0:
                    self.w3d.combo_tryb.blockSignals(True); self.w3d.combo_tryb.setCurrentIndex(1)
                    self.w3d.combo_tryb.blockSignals(False)
                self.w3d.pokaz(przed, z, self.rgb, fn)
        self._opis_lang = ''
        if self.chk_widok_makro.isChecked():
            self.makro.ustaw_atencje(fn)
            self.makro.ustaw_obraz(self.rgb)
            self.makro.pokaz_klatke(t, self.token)
            self._strzalki_przyczyn(k)
        self.panel.przy_kroku()
        self._rysuj_klatke(k, kc)
        self._kursor(k)
        a = self.an
        kr = int(self.kroki[k])
        tekst = (f'krok {kr} ({t + 1}/{len(self.idx)}); detekcja i atencja z wag kroku {int(self.opis["kroki_wag"][kc])}: '
                 f'pewność {self.dety["pewnosc"][kc] * 100:.1f} %; zmiana latentu w tym kroku {a["wzgl"][k] * 100:.2f} %, '
                 f'prędkość narzędzie / tło {a["v"][k][self.narz].mean():.3f} / {a["v"][k][~self.narz].mean():.3f}')
        if self.token is not None:
            tekst += f'; patch {self.token}: prędkość {a["v"][k][self.token]:.3f}, do stanu końcowego {a["do_konca"][k][self.token]:.2f}'
        self.etykieta_kroku.setText(tekst + self._opis_lang)

    # ------------------------------------------------------------ rysunki
    def _rysuj_klatke(self, k, kc):
        ax = self.ax
        ax.clear(); ax.imshow(self.rgb, extent=(0, 480, 320, 0))
        xy = np.stack(np.divmod(np.arange(600), 30), 1)[:, ::-1] * 16 + 8                # środki patchy (x, y)
        n = self.narz
        ax.scatter(xy[n, 0], xy[n, 1], s=6, c=KOL_N, alpha=.8)
        x0, y0, x1, y1 = self.dety['ramka'][kc]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=KOLOR_VIT[:3], lw=1.6))
        if self.czubek is not None:
            ax.plot(*self.czubek, 'o', ms=7, mfc='none', mec='#ff3030', mew=1.6)
        if self.token is not None:
            r, c = divmod(self.token, 30)
            ax.add_patch(Rectangle((c * 16, r * 16), 16, 16, fill=False, ec='white', lw=1.5))
        rysuj_zaznaczenie(ax, self.zaznaczenie, self._kolory_zazn())
        ax.set_xlim(0, 480); ax.set_ylim(320, 0); ax.axis('off')
        ax.set_title(f'krok {int(self.kroki[k])}: wagi kroku {int(self.opis["kroki_wag"][kc])}, pewność '
                     f'{self.dety["pewnosc"][kc] * 100:.0f} %', fontsize=9, color='white')
        self.canvas.draw_idle()

    def _rysuj_wykresy(self):
        if self.an is None:
            return
        a, kr, n = self.an, self.kroki, self.narz
        f = self.fig_r; f.clear()
        osie = f.subplots(5, 1, sharex=True)
        f.subplots_adjust(left=.13, right=.86, top=.97, bottom=.06, hspace=.28)
        ax = osie[0]
        ks, st = self.opis['strata']
        ax.plot(ks, st, color='#f0c040', lw=1.2, label='strata'); ax.set_ylabel('strata', fontsize=8)
        ax2 = ax.twinx(); km, mp = self.opis['mAP']
        ax2.plot(km, mp * 100, 'o-', color='#40d0ff', ms=3, lw=1, label='mAP@0,5'); ax2.set_ylabel('mAP %', fontsize=8)
        ax.set_title('strata uczenia (żółta) i mAP walidacji (niebieska)', fontsize=8)

        def trzy(ax, Y, x, tytul, log=True):
            ax.plot(x, Y[:, n].mean(1), color=KOL_N, lw=.9, label='narzędzie')
            ax.plot(x, Y[:, ~n].mean(1), color=KOL_T, lw=.9, label='tło')
            if self.token is not None:
                ax.plot(x, Y[:, self.token], color=KOL_P, lw=.7, alpha=.8, label=f'patch {self.token}')
            if log:
                ax.set_yscale('log')
            ax.set_title(tytul, fontsize=8)
            ax.legend(fontsize=7, loc='upper right', ncol=3, framealpha=.3)

        v = a['v'].copy(); v[0] = np.nan
        trzy(osie[1], v, kr, 'prędkość tokenu w latencie ||z_t − z_(t−1)|| (na krok)')
        vm = self.v_makro.copy(); vm[0] = np.nan
        trzy(osie[2], vm, kr[self.idx], 'prędkość cząstki w przestrzeni makro ||P_t − P_(t−1)|| (na krok)')
        trzy(osie[3], a['do_konca'], kr, 'odległość od stanu końcowego ||z_t − z_końcowy||', log=False)
        ax = osie[4]
        ax.plot(kr, a['separacja'], color='#7CFC00', lw=1, label='rozdzielenie narzędzie / tło')
        ax.set_ylabel('rozdz.', fontsize=8)
        ax2 = ax.twinx(); ax2.plot(kr, a['rzad'], color='#d080ff', lw=1); ax2.set_ylabel('rząd ef.', fontsize=8)
        ax.set_title('rozdzielenie narzędzie / tło (zielone) i efektywny rząd tokenów (fioletowy)', fontsize=8)
        ax.set_xlabel('krok uczenia', fontsize=8)
        for ax in f.axes:
            ax.tick_params(labelsize=7)
        self._linie = [ax.axvline(0, color='white', lw=.8, alpha=.7) for ax in osie]
        self.canvas_r.draw_idle()

    def _rysuj_cka(self):
        a = self.an
        f = self.fig_c; f.clear(); ax = f.add_axes([.12, .1, .8, .82])
        kr = self.kroki[a['cka_kroki']]
        ax.imshow(a['cka'], origin='lower', cmap='magma', vmin=max(0., float(np.percentile(a['cka'], 2))), vmax=1,
                  extent=(kr[0], kr[-1], kr[0], kr[-1]), aspect='auto')
        ax.set_title('CKA tokenów między krokami (jasne = podobna geometria)', fontsize=8)
        ax.set_xlabel('krok', fontsize=8); ax.set_ylabel('krok', fontsize=8); ax.tick_params(labelsize=7)
        self._linie_c = [ax.axvline(kr[0], color='cyan', lw=.8), ax.axhline(kr[0], color='cyan', lw=.8)]
        self.canvas_c.draw_idle()

    # ------------------------------------------------------------ przyczyny ruchu
    def _licz_przyczyny(self):
        if self.Z is None or self.watek:
            return
        b, o, i, rgb = self._bieg(), self.opis, self.i_kl, self.rgb

        def zadanie(wt):
            P = PR.Przyczyny(b, o, i, rgb)
            return P, P.licz(lambda k, n: wt.postep.emit(int(100 * k / n), f'przyczyny ruchu: checkpoint {k}/{n}'))

        def gotowe(w):
            self.P_prz, self.prz = w
            self._delty = (None, None)
            self._rysuj_przyczyny(); self.zakl_wyk.setCurrentIndex(2); self.idz(self.t)
            r = self.prz
            self._status(f'Przyczyny ruchu: {len(r["kroki_wag"]) - 1} przedziałów; średni cos(−gradient, ruch) '
                         f'{np.nanmean(r["cos_grad"]):.3f}; ruch razem z atencją {np.nanmean(r["cos_atencji"]):.2f} '
                         f'(kontrola: ze średnią {np.nanmean(r["cos_rowno"]):.2f})', 20000)

        self.watek = WatekFilmu(zadanie, self)
        self.watek.postep.connect(lambda k, t: (self.pasek.setValue(k), self._status(t, 3000)))
        self.watek.gotowy.connect(lambda w: (setattr(self, 'watek', None), gotowe(w)))
        self.watek.blad.connect(lambda t: (setattr(self, 'watek', None), QMessageBox.critical(self, 'Błąd', t[-2000:])))
        self.watek.start()

    def _maska_p(self):
        i = self.combo_zbior_p.currentIndex()
        if i == 2 and self.token is not None:
            m = np.zeros(600, bool); m[self.token] = True
            return m, f'patch {self.token}'
        return (self.narz, 'narzędzie') if i == 0 else (~self.narz, 'tło')

    def _rysuj_przyczyny(self):
        r = self.prz
        f = self.fig_p; f.clear()
        if r is None:
            self.canvas_p.draw_idle(); return
        import matplotlib
        from matplotlib.colors import TwoSlopeNorm
        m, nazwa = self._maska_p()
        kw = r['kroki_wag']; x0, x1 = float(kw[0]), float(kw[-1])
        osie = f.subplots(4, 1, sharex=True)
        f.subplots_adjust(left=.2, right=.86, top=.96, bottom=.06, hspace=.35)
        U = r['udzialy'][:, :, m].mean(-1).T                                        # [G, przedziały]
        v = max(.05, float(np.percentile(np.abs(U), 98)))
        ax = osie[0]
        ax.imshow(U, aspect='auto', cmap='coolwarm', norm=TwoSlopeNorm(0, -v, v), extent=(x0, x1, len(PR.GRUPY) - .5, -.5),
                  interpolation='nearest')
        ax.set_yticks(range(len(PR.GRUPY))); ax.set_yticklabels(PR.GRUPY, fontsize=6)
        ax.set_title(f'udział grup wag w ruchu cząstek ({nazwa}); czerwone = pcha w stronę ruchu', fontsize=8)
        kol = matplotlib.colormaps['tab20'](np.arange(len(PR.GRUPY)) % 20)
        for g in range(len(PR.GRUPY)):
            ax.add_patch(matplotlib.patches.Rectangle((x0 - (x1 - x0) * .03, g - .5), (x1 - x0) * .025, 1, color=kol[g],
                                                      clip_on=False))
        ax = osie[1]
        ax.imshow(np.log10(np.maximum(r['zmiana_wag'].T, 1e-6)), aspect='auto', cmap='viridis',
                  extent=(x0, x1, len(PR.GRUPY) - .5, -.5), interpolation='nearest')
        ax.set_yticks(range(len(PR.GRUPY))); ax.set_yticklabels(PR.GRUPY, fontsize=6)
        ax.set_title('zmiana wag grupy w przedziale log10(||ΔW|| / ||W||)', fontsize=8)
        ax = osie[2]
        xs = kw[:-1]
        ax.plot(xs, r['cos_grad'][:, m].mean(1), color='#ffd84a', lw=1, label='cos(−∂L/∂z, Δz)')
        ax.axhline(0, color='gray', lw=.6)
        ax.set_ylabel('cos', fontsize=8)
        ax2 = ax.twinx(); ax2.plot(kw, r['grad_norma'][:, m].mean(1), color='#ff8c40', lw=.9); ax2.set_ylabel('||∂L/∂z||', fontsize=8)
        ax.set_title('czy cząstki ruszają się wzdłuż −gradientu straty tej klatki (żółta) i jego siła (pomarańczowa)', fontsize=8)
        ax = osie[3]
        ax.plot(xs, r['cos_atencji'][:, m].mean(1), color='#40d0ff', lw=1, label='z tym, na co patrzy (atencja)')
        ax.plot(xs, r['cos_rowno'][:, m].mean(1), color='#8fa3b8', lw=1, ls='--', label='kontrola: ze średnią wszystkich')
        ax.set_ylabel('cos', fontsize=8)
        ax2 = ax.twinx(); ax2.plot(kw[1:], r['zmiana_atencji'][:, m].mean(1), color='#d080ff', lw=.9)
        ax2.set_ylabel('zmiana atencji', fontsize=8)
        ax.legend(fontsize=6, loc='lower left', framealpha=.3)
        ax.set_title('ruch razem z tokenami, na które patrzy (niebieska) i zmiana atencji (fioletowa)', fontsize=8)
        ax.set_xlabel('krok uczenia (początek przedziału wag)', fontsize=8)
        for ax in f.axes:
            ax.tick_params(labelsize=7)
        self._linie_p = [ax.axvline(0, color='white', lw=.8, alpha=.7) for ax in osie]
        self.canvas_p.draw_idle()

    def _strzalki_przyczyn(self, k):
        seg, kolory = self._odcinki_przyczyn(k)
        s2, k2, pkt = self._odcinki_prognozy(k)
        seg, kolory = seg + s2, kolory + k2
        self.makro.rysuj_przyczyny(np.array(seg) if seg else None, np.array(kolory) if kolory else None)
        if pkt:
            self.makro.rysuj_znaczniki(*(np.array(c) for c in zip(*pkt)))
        else:
            self.makro.rysuj_znaczniki(None)

    def _wpisy(self):
        """Cząstki do prognozy: [(klucz, maska [600], nazwa, kolor RGB albo None = kolory domyślne)]."""
        i = self.combo_czastka.currentIndex()
        if i == 0:
            if self.token is None:
                return []
            m = np.zeros(600, bool); m[self.token] = True
            return [(('patch', self.token), m, f'patch {self.token}', None)]
        if i in (1, 2):
            return [((('narzedzie',) if i == 1 else ('tlo',)), self.narz if i == 1 else ~self.narz,
                     'średnia narzędzia' if i == 1 else 'średnia tła', None)]
        if i in (4, 5):
            sel = self.zaznaczenie
            if sel is None:
                return []
            m = np.zeros(600, bool); m[sel] = True
            if i == 5:
                return [(('zazn', tuple(sorted(int(t) for t in sel))), m, f'średnia {len(sel)} zaznaczonych', None)]
            wynik = []
            for t, c in list(zip(sel, paleta_zaznaczenia(len(sel))))[:self.spin_zazn_prog.value()]:
                m = np.zeros(600, bool); m[t] = True
                wynik.append((('patch', int(t)), m, f'patch {t}', tuple(float(x) for x in c[:3])))
            return wynik
        from matplotlib.colors import to_rgb
        wynik = []
        for t in self.panel.tokeny():
            m = np.zeros(600, bool); m[t] = True
            wynik.append((('patch', t), m, f'patch {t}', to_rgb(self.panel.kolor(t))))
        return wynik

    def _model(self, klucz, maska):
        """Dopasowany model (rownanie_czastki) wybranego w panelu typu dla cząstki albo średniej; pamiętany."""
        mod = self.panel.combo_model.currentIndex(); par = self.panel.parametry()
        kl = (klucz, mod, tuple(sorted(par.items())))
        if kl not in self._tory:
            if klucz[0] == 'patch':
                x = np.asarray(self.Z[:, klucz[1]], np.float32)
            else:
                kx = ('x',) + klucz
                if kx not in self._tory:
                    self._tory[kx] = AU.tor_sredni(self.Z, maska)
                x = self._tory[kx]
            self._tory[kl] = RC.dopasuj(x, mod, **par)
        return self._tory[kl]

    def _zanik(self, n):
        """Krycie n kolejnych odcinków śladu, od najstarszego do najnowszego."""
        if n <= 0:
            return np.zeros(0, np.float32)
        u = np.linspace(0, 1, n)
        a0 = self.spin_krycie.value()
        i = self.combo_zanik.currentIndex()
        if i == 2:
            return np.ones(n, np.float32)
        f = u if i == 0 else np.exp(-4 * (1 - u))
        return (a0 + (1 - a0) * f).astype(np.float32)

    def _odcinki_prognozy(self, k):
        """Odcinki (prognozy, ślady) i znaczniki [(pozycja, kolor, rozmiar)] dla cząstek z _wpisy."""
        if not self.chk_prognoza.isChecked() or self.Z is None:
            return [], [], []
        seg, kol, pkt, opisy = [], [], [], []
        for klucz, maska, nazwa, barwa in self._wpisy():
            s2, k2, p2, op = self._prognoza_wpisu(k, klucz, maska, nazwa, barwa)
            seg += s2; kol += k2; pkt += p2; opisy.append(op)
        model = RC.MODELE[self.panel.combo_model.currentIndex()].split(' (')[0]
        self._opis_lang = (f'; {model}: ' + '; '.join(opisy)) if opisy else ''
        return seg, kol, pkt

    def _prognoza_wpisu(self, k, klucz, maska, nazwa, barwa):
        mdl = self._model(klucz, maska)
        x = mdl['x']
        K = len(x)
        m = self.makro
        o = m.P[m.t][maska].mean(0)                                   # bieżące położenie cząstki (albo średniej)
        rzut = lambda X: o + m.rzut_delty(np.asarray(X, np.float32) - x[k])
        seg, kol, pkt = [], [], []
        if barwa is None:                                             # jedna cząstka: turkus / pomarańcz
            k_slad, k_prog, k_przysz = (.2, .9, 1), (1, .55, .1), (.65, .8, 1)
        else:                                                         # kilka cząstek: kolor cząstki z panelu
            k_prog = barwa; k_slad = tuple(.45 * c + .55 for c in barwa); k_przysz = tuple(.3 * c + .7 for c in barwa)

        def linia(P, c, a=1., alfy=None, przerywana=False):
            for j in range(len(P) - 1):
                if przerywana and j % 2:
                    continue
                al = a if alfy is None else a * alfy[j]
                seg.extend([P[j], P[j + 1]]); kol.extend([(c[0], c[1], c[2], al)] * 2)

        ema = self.chk_ema.isChecked()
        a_ema = self.spin_ema.value()
        L = k if self.chk_slad_caly.isChecked() else min(k, self.spin_slad.value())
        t0 = k - L
        co = max(1, L // 400)                                         # do 400 odcinków na ślad
        if L > 0:                                                     # ślad rzeczywisty
            if ema:
                if mdl.get('_ema', (None,))[0] != a_ema:
                    mdl['_ema'] = (a_ema, AU.ema(x, a_ema))
                y = mdl['_ema'][1]
            else:
                y = x
            P = rzut(y[t0:k + 1][::-1][::co][::-1])
            linia(P, k_slad, 1., self._zanik(len(P) - 1))
        w = self.spin_wyprz.value()
        if L > 0 and w > 0 and k - w >= 0:                            # ślad prognoz z wyprzedzeniem w
            ts = np.arange(max(t0, w), k + 1)[::-1][::co][::-1]
            pred = RC.prognozy(mdl, ts - w, w)[:, -1]
            if ema:
                pred = AU.ema(pred, a_ema)
            P = rzut(pred)
            linia(P, k_prog, .9, self._zanik(len(P) - 1))
        h, N = self.spin_h.value(), self.spin_nowa.value()
        k0 = (k // N) * N
        starty = [k0 - j * N for j in range(self.spin_stare.value() + 1) if k0 - j * N >= 0]
        tory = RC.prognozy(mdl, starty, h)                            # [n, h + 1, 768]
        for j, tor in enumerate(tory):
            linia(rzut(tor), k_prog, 1. if j == 0 else .55 * (1 - j / (len(tory) + 1)), przerywana=True)
        opis = f'{nazwa} od kroku {int(self.kroki[k0])}'
        if 0 < k - k0 <= h:                                           # gdzie prognoza stawiała cząstkę teraz
            pkt.append((rzut(tory[0][k - k0][None])[0], (*k_prog, 1), 11))
            e = float(np.linalg.norm(tory[0][k - k0] - x[k])); ep = float(np.linalg.norm(x[k0] - x[k]))
            opis += f', po {k - k0} kr. błąd / bez ruchu {e / max(ep, 1e-9):.2f}'
        if self.chk_przyszlosc.isChecked():                           # rzeczywisty dalszy tor (podgląd)
            kon = min(K - 1, max(k0 + h, k))
            if kon > k:
                linia(rzut(x[k:kon + 1]), k_przysz, .45)
        if klucz[0] != 'patch':                                       # średnia jako jedna duża kropka
            barwa_sr = {'narzedzie': (1, .3, .25, 1), 'zazn': (1, .9, .1, 1)}.get(klucz[0], (.75, .8, .9, 1))
            pkt.append((o, barwa_sr, m.spin_rozmiar.value() + 12))
        elif barwa is not None:
            pkt.append((o, (*barwa, 1), m.spin_rozmiar.value() + 8))
        return seg, kol, pkt, opis

    def _odcinki_przyczyn(self, k):
        if self.prz is None or self.token is None or not self.chk_strzalki.isChecked():
            return [], []
        import matplotlib
        kw = self.prz['kroki_wag']
        i = int(np.clip(np.searchsorted(kw, self.kroki[k], 'right') - 1, 0, len(kw) - 2))
        if self._delty[0] != i:
            z0, dz, D = self.P_prz.delty(i)
            g = self.P_prz.gradient(i)[0]
            self._delty = (i, (dz.cpu().numpy(), D.cpu().numpy(), g.cpu().numpy()))
        dz, D, g = self._delty[1]
        tok = self.token
        mn = self.spin_strz.value()
        o = self.makro.P[self.makro.t][tok]
        wek = self.makro.rzut_delty(np.concatenate([dz[tok][None], D[:, tok]], 0)) * mn          # [1 + G, 3]
        gn = self.makro.rzut_delty(-g[tok][None])[0]
        gn = gn / max(float(np.linalg.norm(gn)), 1e-12) * float(np.linalg.norm(wek[0]))
        kol = np.concatenate([[[1, 1, 1, 1]], matplotlib.colormaps['tab20'](np.arange(len(PR.GRUPY)) % 20)]).astype(np.float32)
        seg = [o, o + wek[0]] + sum(([o, o + v] for v in wek[1:]), [])
        kolory = [kol[0], kol[0]] + sum(([c, c] for c in kol[1:]), [])
        for a in range(6):                                                          # przerywana żółta: −gradient
            seg += [o + gn * a / 6, o + gn * (a + .5) / 6]; kolory += [(1, .85, .2, 1)] * 2
        return seg, [tuple(c) for c in kolory]

    def _kursor(self, k):
        x = float(self.kroki[k])
        for l in getattr(self, '_linie_p', []):
            l.set_xdata([x, x])
        if getattr(self, '_linie_p', None) and self.zakl_wyk.currentIndex() == 2:
            self.canvas_p.draw_idle()
        for l in getattr(self, '_linie', []):
            l.set_xdata([x, x])
        self.canvas_r.draw_idle()
        for l in getattr(self, '_linie_z', []):
            l.set_xdata([x, x])
        if getattr(self, '_linie_z', None) and self.zakl_wyk.currentWidget() is self.canvas_z:
            self.canvas_z.draw_idle()
        if getattr(self, '_linie_c', None):
            self._linie_c[0].set_xdata([x, x]); self._linie_c[1].set_ydata([x, x])
            if self.zakl_wyk.currentIndex() == 1:
                self.canvas_c.draw_idle()

    def _idz_do_kroku(self, x):
        if self.Z is None or x is None:
            return
        k = int(np.abs(self.kroki - x).argmin())
        self.suwak.setValue(int(np.abs(self.idx - k).argmin()))

    def _klik_wykresu(self, e):
        if e.inaxes is not None:
            self._idz_do_kroku(e.xdata)

    def _klik_cka(self, e):
        if e.inaxes is not None:
            self._idz_do_kroku(e.xdata)

    def _klik_klatki(self, e):
        if e.inaxes is None or self.Z is None or e.xdata is None:
            return
        if e.button == 3:
            self._start_zazn = (e.xdata, e.ydata); return
        c = int(np.clip(e.xdata // 16, 0, 29)); r = int(np.clip(e.ydata // 16, 0, 19))
        if ctrl_wcisniety():                                          # Ctrl + klik: dodaj / usuń patch z zaznaczenia
            self._zaznacz(przelacz_w_zaznaczeniu(self.zaznaczenie, r * 30 + c)); return
        self._wybierz_token(r * 30 + c)

    # ------------------------------------------------------------ zaznaczanie patchy (obraz <-> przestrzeń makro)
    def _puszczenie_klatki(self, e):
        st, self._start_zazn = getattr(self, '_start_zazn', None), None
        if st is None or e.button != 3:
            return
        x1, y1 = (e.xdata, e.ydata) if e.xdata is not None else st
        duzy = abs(x1 - st[0]) + abs(y1 - st[1]) > 6
        if duzy:
            self._zaznacz(polacz_zaznaczenie(self.zaznaczenie, tokeny_w_prostokacie(st[0], st[1], x1, y1), ctrl_wcisniety()))
        elif not ctrl_wcisniety():
            self._zaznacz([])

    def _kolory_zazn(self):
        """Osobne kolory zaznaczonych patchy [n, 4] (w kolejności zaznaczania) albo None."""
        if self.zaznaczenie is None or not self.chk_kol_zazn.isChecked():
            return None
        return paleta_zaznaczenia(len(self.zaznaczenie))

    def _zaznacz(self, tokeny):
        """Zaznaczone patche: kolorowe (albo żółte) na obrazie, jasne w przestrzeni makro (reszta blada albo ukryta
        przy „śledź tylko zaznaczone”), przebiegi w karcie „Zaznaczone patche”; pusty = wyczyść."""
        self.zaznaczenie = np.asarray(tokeny, int) if tokeny is not None and len(tokeny) else None
        self.makro.tylko_zaznaczone = self.chk_tylko_zazn.isChecked()
        self.makro.ustaw_zaznaczenie(self.zaznaczenie, self._kolory_zazn())
        n = 0 if self.zaznaczenie is None else len(self.zaznaczenie)
        self.etykieta_zazn.setText('Brak zaznaczenia. Ctrl + klik na obrazie albo w przestrzeni makro dodaje patch.' if not n else
                                   f'Zaznaczone: {n} patchy' + (f' ({", ".join(str(t) for t in self.zaznaczenie[:12])}'
                                                                f'{", ..." if n > 12 else ""})'))
        self._rysuj_zazn()
        if n and self.zakl_wyk.currentWidget() is not self.canvas_z and self.chk_tylko_zazn.isChecked():
            self.zakl_wyk.setCurrentWidget(self.canvas_z)
        self.idz(self.t)
        self._status('Zaznaczenie wyczyszczone.' if not n else
                     f'Zaznaczono {n} patchy (Ctrl + klik: dodaj / usuń; prawy klik bez przeciągania na obrazie: wyczyść).')

    def _tylko_zazn(self, wl):
        """Śledź tylko zaznaczone: makro rysuje same zaznaczone cząstki, prognoza Langevina przechodzi na nie."""
        if wl and self.combo_czastka.currentIndex() == 0:
            self.combo_czastka.blockSignals(True); self.combo_czastka.setCurrentIndex(4); self.combo_czastka.blockSignals(False)
        elif not wl and self.combo_czastka.currentIndex() == 4:
            self.combo_czastka.blockSignals(True); self.combo_czastka.setCurrentIndex(0); self.combo_czastka.blockSignals(False)
        if wl and self.zaznaczenie is None:
            self._status('Najpierw zaznacz patche: Ctrl + klik na obrazie albo w przestrzeni makro, prawy przycisk + przeciągnij.')
        self._zaznacz(self.zaznaczenie)

    def _miary_zazn(self):
        """Miary zaznaczonych patchy po krokach (liczone raz na zaznaczenie): cos do stanu końcowego [K, n] i rozrzut
        zaznaczonych, czyli średnia odległość od ich wspólnego środka [K]."""
        sel = self.zaznaczenie
        klucz = tuple(int(t) for t in sel)
        if self._zazn_miary[0] != klucz:
            K = len(self.Z)
            kon = np.asarray(self.Z[-1][sel], np.float32)
            nk = np.linalg.norm(kon, axis=-1)
            cos = np.zeros((K, len(sel)), np.float32); roz = np.zeros(K, np.float32)
            for a in range(0, K, 100):                                    # paczkami, bez kopii całej historii
                X = np.asarray(self.Z[a:a + 100][:, sel], np.float32)       # [b, n, 768]
                cos[a:a + len(X)] = (X * kon).sum(-1) / np.maximum(np.linalg.norm(X, axis=-1) * nk, 1e-12)
                roz[a:a + len(X)] = np.linalg.norm(X - X.mean(1, keepdims=True), axis=-1).mean(1)
            self._zazn_miary = (klucz, (cos, roz))
        return self._zazn_miary[1]

    def _rysuj_zazn(self):
        """Karta „Zaznaczone patche”: każdy zaznaczony patch osobno przez całe uczenie."""
        f = self.fig_z; f.clear(); self._linie_z = []
        sel = self.zaznaczenie
        if self.an is None or sel is None:
            ax = f.add_axes([0, 0, 1, 1]); ax.axis('off')
            ax.text(.5, .5, 'Zaznacz patche, żeby zobaczyć, jak każdy z nich zmienia się podczas uczenia:\n'
                    'Ctrl + klik na obrazie albo w przestrzeni makro (dodaj / usuń),\n'
                    'prawy przycisk + przeciągnij na obrazie (prostokąt; z Ctrl dołącza).',
                    ha='center', va='center', color='#c8c8c8', fontsize=9, transform=ax.transAxes)
            self.canvas_z.draw_idle(); return
        a, kr = self.an, self.kroki
        n = len(sel)
        kol = paleta_zaznaczenia(n)
        cos, roz = self._miary_zazn()
        osie = f.subplots(5, 1, sharex=True)
        f.subplots_adjust(left=.13, right=.97, top=.97, bottom=.06, hspace=.3)
        lw, al = (1., .95) if n <= 20 else (.5, .4)

        def rysuj(ax, x, Y, tytul, log=False):
            for j in range(n):
                ax.plot(x, Y[:, j], color=kol[j][:3], lw=lw, alpha=al, label=f'patch {sel[j]}' if n <= 10 else None)
            if n > 1:
                ax.plot(x, Y.mean(1), color='white', lw=1.4, ls='--', label='średnia')     # NaN tylko w kroku 0
            if log:
                ax.set_yscale('log')
            ax.set_title(tytul, fontsize=8)

        v = a['v'][:, sel].copy(); v[0] = np.nan
        rysuj(osie[0], kr, v, 'prędkość w latencie ||z_t − z_(t−1)|| (na krok)', log=True)
        osie[0].legend(fontsize=6.5, loc='upper right', ncol=4, framealpha=.3)
        vm = self.v_makro[:, sel].copy(); vm[0] = np.nan
        rysuj(osie[1], kr[self.idx], vm, 'prędkość w przestrzeni makro ||P_t − P_(t−1)|| (na krok)', log=True)
        rysuj(osie[2], kr, a['do_konca'][:, sel], 'odległość od stanu końcowego ||z_t − z_końcowy||')
        rysuj(osie[3], kr, cos, 'podobieństwo do stanu końcowego cos(z_t, z_końcowy)')
        ax = osie[4]
        ax.plot(kr, roz, color='#ffd030', lw=1.2)
        ax.set_title(f'rozrzut {n} zaznaczonych: średnia odległość od ich środka (maleje = zbiegają się)', fontsize=8)
        ax.set_xlabel('krok uczenia', fontsize=8)
        for ax in f.axes:
            ax.tick_params(labelsize=7)
        x = float(kr[int(self.idx[min(getattr(self, 't', 0), len(self.idx) - 1)])])
        self._linie_z = [ax.axvline(x, color='white', lw=.8, alpha=.7) for ax in osie]
        self.canvas_z.draw_idle()
