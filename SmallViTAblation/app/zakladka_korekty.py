"""Zakładka „Korekta latentu: cel i dynamika” (SPECYFIKACJA_DLA_CLAUDE_CODE.md w
experiments/stanowisko_korekcji_latentu_gui).

Jaka mała zmiana reprezentacji tokenów wystarcza do poprawy lokalizacji narzędzia, które jej składowe widzi detektor
(fc2) i czy podobną zmianę daje rzeczywiste uczenie. Korekta z etykietą to pomiar oracle (korekta_latentu.py), ruch
uczenia to osobna oś czasu (tor uczenia obrazu, gałęzie uczenia w galezie_uczenia.py), modele dynamiki i obserwator
w dynamika_korekty.py. Nic nie liczy się przy otwarciu zakładki; każde obliczenie po kliknięciu, w tle, z przyciskiem
„Zatrzymaj”. Pomiary w outputs/stanowisko_korekcji_latentu_gui/<znacznik czasu>_<opis>/.
"""
import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QLabel, QLineEdit, QProgressBar,
                             QPushButton, QScrollArea, QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import korekta_latentu as KL
from styl import rola
from uklad import Przeplyw, dodaj
from zakladka_detekcji import PlotnoCiemne

ZRODLA = ['manifest J128 (400 obrazów z etykietą)', 'klatka próbna z zakładki Uczenie (12 klatek gęstej historii)']
KANALY = ['J128 (historyczne 128 kanałów)', 'wszystkie 768', 'wpisane niżej']
SEKWENCJE = ['tor korekty oracle (latent_optimization)', 'tor uczenia obrazu (network_training, co 10 kroków)',
             'klatka zakładki Uczenie: tokeny narzędzia (network_training, co krok)']
PRZESTRZENIE = ['nośnik korekty (tokeny × kanały interwencji)', 'tokeny obiektu × 768', 'średnia tokenów obiektu (768)']
KOL = dict(gt='#00e5ff', przed='#ffd030', po='#30ff70', diag='#ff60ff', det='#bbbbbb')


def _lista_int(t, domyslnie):
    try:
        v = [int(float(x)) for x in str(t).replace(';', ',').split(',') if x.strip()]
        return v or domyslnie
    except ValueError:
        return domyslnie


def _lista_float(t, domyslnie):
    try:
        v = [float(x) for x in str(t).replace(';', ',').split(',') if x.strip()]
        return v or domyslnie
    except ValueError:
        return domyslnie


def _f(v, n=3):
    return '-' if v is None else f'{v:.{n}f}'


def _plotno(figsize=(9, 6)):
    f = Figure(figsize=figsize); c = PlotnoCiemne(f)
    c.setMinimumHeight(420)
    return f, c


def _naglowek(tekst):
    """Pogrubiony nagłówek na początku wiersza opcji (grupa parametrów)."""
    e = QLabel(f'<b>{tekst}:</b>')
    e.setToolTip(f'Grupa opcji: {tekst.lower()}.')
    return e


class ZakladkaKorekty(QWidget):
    def __init__(self, okno, uczenie=None):
        super().__init__()
        self.okno, self.u = okno, uczenie
        self.st = None; self.watek = None
        self.wynik = None; self.hist = None; self.tor = None; self.por = None
        self.dyn = None; self.gal = None; self.kal = None; self.od = None; self.G = None
        self.rgb = None; self.ramki = None; self.opis_obr = {}
        self.ckpty = KL.checkpointy()
        self._rows = None
        lay = QVBoxLayout(self)

        # ------------------------------------------------------------ obraz i model
        w = Przeplyw(); lay.addLayout(w)
        self.btn_ust = rola(QPushButton('Opcje ▾'), 'narzedzie'); self.btn_ust.setCheckable(True); self.btn_ust.setChecked(True)
        self.btn_ust.setToolTip('Pokaż albo schowaj wiersze opcji: interwencja, optymalizacja, ocena (więcej miejsca na wykresy).')
        w.addWidget(self.btn_ust)
        w.addWidget(_naglowek('Obraz'))
        self.combo_ckpt = QComboBox(); self.combo_ckpt.addItems([o for o, _ in self.ckpty])
        if self.ckpty:
            self.combo_ckpt.setCurrentIndex(len(self.ckpty) - 1)
        dodaj(w, 'checkpoint', self.combo_ckpt, 'Zamrożony ViT, przy którym liczona jest korekta (gęsta historia co 10 '
                                                'kroków; k1000 = checkpoint historycznej maksymalizacji J128, ten sam hash).')
        self.combo_zr = QComboBox(); self.combo_zr.addItems(ZRODLA)
        dodaj(w, 'obraz z', self.combo_zr, 'Źródło obrazu i etykiety: manifest maksymalizacji J128 (ramka narzędzia) '
                                           'albo klatka próbna gęstej historii (ramka z obrysu cystotomu).')
        self.spin_obr = QSpinBox(); self.spin_obr.setRange(0, 399)
        dodaj(w, 'nr', self.spin_obr, 'Indeks obrazu w manifeście (0-399) albo numer klatki próbnej (0-11).')
        self.chk_z_ucz = QCheckBox('klatka, krok i zaznaczenie z zakładki Uczenie')
        self.chk_z_ucz.setToolTip('Przy źródle „klatka próbna” bierze bieżącą klatkę i najbliższy zapisany checkpoint '
                                  '(<= bieżący krok) z zakładki „Uczenie ViT”.')
        w.addWidget(self.chk_z_ucz)
        self.etykieta_obr = QLabel(''); w.addWidget(self.etykieta_obr)

        self.ust = QWidget(); ul = QVBoxLayout(self.ust); ul.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.ust)
        w = Przeplyw(); ul.addLayout(w)
        w.addWidget(_naglowek('Interwencja'))
        self.combo_tryb = QComboBox(); self.combo_tryb.addItems(list(KL.TRYBY.values()))
        self.combo_tryb.setCurrentIndex(1)
        dodaj(w, 'interwencja', self.combo_tryb, 'Poziom kontroli osiągalności. Swobodne: diagnostyka. Zakresy: każdy kanał '
                                                 'w [h_min, h_max] z wag checkpointu (konieczne, nie wystarczające). Dolny '
                                                 'GELU: jak historyczna maksymalizacja. Wejście MLP: zmiana wejścia LayerNorm '
                                                 'ostatniego MLP przez prawdziwe LN, fc1 i GELU; „gałąź” zostawia residual, '
                                                 '„wspólne” zmienia też residual. Żaden tryb nie gwarantuje istnienia zdjęcia.')
        self.combo_kan = QComboBox(); self.combo_kan.addItems(KANALY)
        dodaj(w, 'kanały', self.combo_kan, 'Które kanały latentu wolno zmieniać (tylko tryby latentu; wejście MLP zmienia '
                                           'wszystkie 768 przez fc1).')
        self.edit_kan = QLineEdit(); self.edit_kan.setPlaceholderText('np. 5, 17, 300'); self.edit_kan.setMaximumWidth(160)
        dodaj(w, 'wpisane', self.edit_kan, 'Lista kanałów 0-767 rozdzielona przecinkami (przy „wpisane niżej”).')
        self.combo_tok = QComboBox(); self.combo_tok.addItems(KL.TOKENY)
        dodaj(w, 'tokeny', self.combo_tok, 'Które tokeny zmieniać. Obiekt, granica i tło wyznacza ramka GT: wtedy wybór '
                                           'tokenów jest częścią interwencji oracle (zapisane w pomiarze).')
        self.combo_cel = QComboBox(); self.combo_cel.addItems(list(KL.CELE.values()))
        dodaj(w, 'cel', self.combo_cel, 'Strata: ogniskowa CenterNet + 2 (1 − GIoU). Lokalizacja: ogniskowa pozytywna + '
                                        '2 (1 − GIoU) + λ_fp · suma (p − 0,3)_+^2 poza obiektem. Najmniejsza: siatka '
                                        'promieni normy i bisekcja do progu jakości (najmniejsza znaleziona, nie '
                                        'najmniejsza możliwa).')
        w = Przeplyw(); ul.addLayout(w)
        w.addWidget(_naglowek('Optymalizacja'))
        self.combo_norma = QComboBox(); self.combo_norma.addItems(list(KL.NORMY.values()))
        dodaj(w, 'norma', self.combo_norma, 'Ograniczenie normy korekty (rzut po każdym kroku).')
        self.spin_rho = QDoubleSpinBox(); self.spin_rho.setRange(0, 1e5); self.spin_rho.setValue(50); self.spin_rho.setDecimals(2)
        dodaj(w, 'ρ', self.spin_rho, 'Promień ograniczenia normy.')
        self.spin_lam = QDoubleSpinBox(); self.spin_lam.setRange(0, 1e3); self.spin_lam.setDecimals(4); self.spin_lam.setValue(0)
        dodaj(w, 'λ', self.spin_lam, 'Regularyzacja: λ · średnia kwadratów korekty dodana do celu.')
        self.spin_lamfp = QDoubleSpinBox(); self.spin_lamfp.setRange(0, 1e3); self.spin_lamfp.setValue(1.0)
        dodaj(w, 'λ_fp', self.spin_lamfp, 'Waga kary za fałszywe szczyty w celu „lokalizacja”.')
        self.spin_kroki = QSpinBox(); self.spin_kroki.setRange(1, 20000); self.spin_kroki.setValue(200)
        dodaj(w, 'kroki', self.spin_kroki, 'Liczba kroków optymalizatora.')
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setRange(1e-5, 100); self.spin_lr.setDecimals(4); self.spin_lr.setValue(0.3)
        dodaj(w, 'krok lr', self.spin_lr, 'Krok optymalizacji (historycznie Adam 0,3 dla J128; dla wejścia MLP mniejszy, '
                                          'np. 0,05).')
        self.combo_opt = QComboBox(); self.combo_opt.addItems(['adam', 'sgd'])
        dodaj(w, 'optymalizator', self.combo_opt, 'Adam (jego momenty należą do stanu tej optymalizacji) albo SGD.')
        self.spin_ziarno = QSpinBox(); self.spin_ziarno.setRange(0, 10 ** 6)
        dodaj(w, 'ziarno', self.spin_ziarno, 'Ziarno losowego startu i kontroli.')
        self.combo_start = QComboBox(); self.combo_start.addItems(['zero', 'losowy'])
        dodaj(w, 'start', self.combo_start, 'Początek optymalizacji: zero albo losowy N(0, σ²) (porównanie początków).')
        self.spin_sigma = QDoubleSpinBox(); self.spin_sigma.setRange(0, 100); self.spin_sigma.setDecimals(4); self.spin_sigma.setValue(0.01)
        dodaj(w, 'σ', self.spin_sigma, 'Odchylenie losowego startu.')
        self.spin_co = QSpinBox(); self.spin_co.setRange(1, 1000); self.spin_co.setValue(1)
        dodaj(w, 'zapis co', self.spin_co, 'Co ile kroków zapisać stan korekty i gradient (rzeczywisty interwał zapisu; '
                                           'modele z wejściem optymalizatora wymagają zapisu co 1).')
        w = Przeplyw(); ul.addLayout(w)
        w.addWidget(_naglowek('Ocena'))
        self.spin_tau = QDoubleSpinBox(); self.spin_tau.setRange(0.01, 1); self.spin_tau.setValue(0.5)
        dodaj(w, 'τ detekcji', self.spin_tau, 'Protokół obrazu: detekcja = szczyt z pewnością >= τ; dopasowanie przy '
                                              'IoU >= 0,5; reszta to fałszywe detekcje.')
        self.spin_pi = QDoubleSpinBox(); self.spin_pi.setRange(0.05, 1); self.spin_pi.setValue(0.7)
        dodaj(w, 'próg IoU', self.spin_pi, 'Cel „najmniejsza”: IoU najpewniejszej ramki >= próg.')
        self.spin_pf = QSpinBox(); self.spin_pf.setRange(0, 10)
        dodaj(w, 'max FP', self.spin_pf, 'Cel „najmniejsza”: najwyżej tyle fałszywych detekcji.')
        self.spin_rmin = QDoubleSpinBox(); self.spin_rmin.setRange(1e-3, 1e5); self.spin_rmin.setValue(1.0)
        dodaj(w, 'ρ min', self.spin_rmin, 'Cel „najmniejsza”: najmniejszy promień siatki.')
        self.spin_rmax = QDoubleSpinBox(); self.spin_rmax.setRange(1e-2, 1e6); self.spin_rmax.setValue(300.0)
        dodaj(w, 'ρ max', self.spin_rmax, 'Cel „najmniejsza”: największy promień siatki.')
        self.spin_nr = QSpinBox(); self.spin_nr.setRange(2, 30); self.spin_nr.setValue(8)
        dodaj(w, 'promieni', self.spin_nr, 'Cel „najmniejsza”: liczba promieni siatki geometrycznej (potem 3 bisekcje).')
        self.etykieta_koszt = QLabel(''); w.addWidget(self.etykieta_koszt)

        w = Przeplyw(); lay.addLayout(w)
        self.btn_hist = QPushButton('Wczytaj historyczny wynik'); w.addWidget(self.btn_hist)
        self.btn_hist.setToolTip('Para przed/po z maksymalizacji J128 (outputs/vit_latent128_maksymalizacja) dla obrazu '
                                 'manifestu. Tylko końce korekty, bez trajektorii; ostrzeżenie przy niezgodnym checkpointcie.')
        self.btn_licz = rola(QPushButton('Oblicz korektę'), 'glowny'); w.addWidget(self.btn_licz)
        self.btn_licz.setToolTip('Liczy korektę latentu dla wybranego obrazu i checkpointu przy ustawionej interwencji i celu,\n'
                                 'w tle (nic nie liczy się samo przy otwarciu zakładki). „Zatrzymaj” przerywa obliczenie.')
        self.btn_stop = QPushButton('Zatrzymaj'); w.addWidget(self.btn_stop); self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip('Przerywa bieżące obliczenie; częściowe wyniki zostają zapisane ze statusem cancelled.')
        self.btn_por = QPushButton('Porównaj z ruchem uczenia'); w.addWidget(self.btn_por)
        self.btn_por.setToolTip('Latent tego obrazu przy 101 checkpointach gęstej historii (oś uczenia, co 10 kroków) '
                                'i porównanie z korektą policzoną przy wybranym checkpoincie.')
        self.btn_zapisz = QPushButton('Zapisz pomiar'); w.addWidget(self.btn_zapisz)
        self.btn_zapisz.setToolTip('Rysunki PNG i analizy (porównanie, modele, obserwator) do folderu bieżącego pomiaru.')
        self.btn_wczytaj = QPushButton('Wczytaj pomiar'); w.addWidget(self.btn_wczytaj)
        self.btn_wczytaj.setToolTip('Ponowne wczytanie zapisanego pomiaru korekty z jego folderu.')
        self.pasek = QProgressBar(); self.pasek.setMaximumWidth(220); w.addWidget(self.pasek)
        self.etykieta = QLabel('Korekta z etykietą (oracle) przy zamrożonym ViT. Wybierz obraz i kliknij „Oblicz korektę” '
                               'albo „Wczytaj historyczny wynik”. Nic nie liczy się samo.')
        self.etykieta.setWordWrap(True); lay.addWidget(self.etykieta)

        # ------------------------------------------------------------ podzakładki
        self.tabs = QTabWidget(); lay.addWidget(self.tabs, 1)
        self.fig_o, self.can_o = self._strona('Obraz i detekcje', (11, 6))
        self.fig_p, self.can_p = self._strona('Przebieg korekty', (11, 7))
        self.fig_f, self.can_f = self._strona('fc2: część niewidoczna', (11, 6))
        # porównanie z uczeniem i 3D
        s = QWidget(); sl = QVBoxLayout(s)
        w = Przeplyw(); sl.addLayout(w)
        self.combo_baza = QComboBox(); self.combo_baza.addItems(['kierunek korekty + 2 losowe (bez przyszłości)',
                                                                 'PCA ruchu uczenia do k_c (tylko przeszłość)',
                                                                 'PCA obu torów (retrospektywna)'])
        dodaj(w, 'baza 3D', self.combo_baza, 'Stała baza rzutu dla obu torów. Baza dopasowana z przyszłych stanów jest '
                                             'retrospektywna (tylko opis).')
        self.sl_kor = QSlider(Qt.Orientation.Horizontal); self.sl_kor.setMinimumWidth(220)
        dodaj(w, 'krok korekty', self.sl_kor, 'Położenie na torze korekty (zapisane kroki).')
        self.sl_ucz = QSlider(Qt.Orientation.Horizontal); self.sl_ucz.setMinimumWidth(220)
        dodaj(w, 'krok uczenia', self.sl_ucz, 'Położenie na torze uczenia (checkpointy co 10 kroków).')
        self.btn_graj = QPushButton('▶'); self.btn_graj.setCheckable(True); w.addWidget(self.btn_graj)
        self.btn_graj.setToolTip('Odtwarzanie obu torów (każdy po swojej osi).')
        self.etykieta_sl = QLabel(''); w.addWidget(self.etykieta_sl)
        self.fig_u = Figure(figsize=(11, 7)); self.can_u = PlotnoCiemne(self.fig_u); self.can_u.setMinimumHeight(560)
        sl.addWidget(self.can_u, 1)
        self.tabs.addTab(self._przew(s), 'Uczenie a korekta (3D)')
        # modele dynamiki
        s = QWidget(); sl = QVBoxLayout(s)
        w = Przeplyw(); sl.addLayout(w)
        self.combo_seq = QComboBox(); self.combo_seq.addItems(SEKWENCJE)
        dodaj(w, 'sekwencja', self.combo_seq, 'Rzeczywiście zapisana sekwencja i jej oś czasu.')
        self.combo_prz = QComboBox(); self.combo_prz.addItems(PRZESTRZENIE)
        dodaj(w, 'przestrzeń', self.combo_prz, 'Jawnie nazwana podprzestrzeń latentu dla toru uczenia (tor korekty: '
                                               'zawsze zmienna optymalizacji).')
        self.spin_fit = QDoubleSpinBox(); self.spin_fit.setRange(.2, .9); self.spin_fit.setValue(.6); self.spin_fit.setSingleStep(.05)
        dodaj(w, 'część do dopasowania', self.spin_fit, 'Dopasowanie (PCA, parametry) tylko na pierwszej części sekwencji; '
                                                       'prognozy od jej końca.')
        self.spin_L = QSpinBox(); self.spin_L.setRange(1, 30); self.spin_L.setValue(5)
        dodaj(w, 'L', self.spin_L, 'Długość pamięci GLE (liczba opóźnionych prędkości).')
        self.spin_okno = QSpinBox(); self.spin_okno.setRange(2, 200); self.spin_okno.setValue(20)
        dodaj(w, 'okno', self.spin_okno, 'Okno średniej kroczącej (położenie równowagi sprężyny).')
        self.spin_r = QSpinBox(); self.spin_r.setRange(1, 40); self.spin_r.setValue(8)
        dodaj(w, 'r', self.spin_r, 'Liczba modów PCA (Koopman, obserwator).')
        self.spin_d = QSpinBox(); self.spin_d.setRange(1, 20); self.spin_d.setValue(4)
        dodaj(w, 'd', self.spin_d, 'Długość okna Hankela (Koopman).')
        self.edit_h = QLineEdit('1, 5, 20'); self.edit_h.setMaximumWidth(90)
        dodaj(w, 'horyzonty', self.edit_h, 'Horyzonty prognozy w krokach zapisu tej sekwencji.')
        self.btn_dyn = rola(QPushButton('Oceń modele'), 'glowny'); w.addWidget(self.btn_dyn)
        self.btn_dyn.setToolTip('Dopasowuje modele dynamiki do wybranej sekwencji i ocenia ich prognozy na podanych horyzontach.\n'
                                'Wynik na wykresie poniżej; liczy się w tle.')
        self.fig_d = Figure(figsize=(11, 7)); self.can_d = PlotnoCiemne(self.fig_d); self.can_d.setMinimumHeight(560)
        sl.addWidget(self.can_d, 1)
        self.tabs.addTab(self._przew(s), 'Modele dynamiki')
        # gałęzie uczenia
        s = QWidget(); sl = QVBoxLayout(s)
        w = Przeplyw(); sl.addLayout(w)
        self.spin_gk = QSpinBox(); self.spin_gk.setRange(0, 979); self.spin_gk.setValue(500)
        dodaj(w, 'krok startu k', self.spin_gk, 'Stan uczenia gęstej historii w kroku k (wagi, BN, AdamW z powtórki).')
        self.combo_gi = QComboBox(); self.combo_gi.addItems(['waga obrazu w slocie p: 1 + ε', 'podmiana slotu p wybranym obrazem',
                                                             'wzrost małych zaburzeń wag (skale ε)'])
        dodaj(w, 'interwencja', self.combo_gi, 'Zdefiniowane wejście: zmiana wagi przykładu w paczce, podmiana jednego '
                                               'obrazu przy stałej liczebności paczki albo zaburzenie wag.')
        self.spin_slot = QSpinBox(); self.spin_slot.setRange(0, 31)
        dodaj(w, 'slot p', self.spin_slot, 'Pozycja w paczce kroku k.')
        self.spin_geps = QDoubleSpinBox(); self.spin_geps.setRange(-1, 100); self.spin_geps.setDecimals(3); self.spin_geps.setValue(0.5)
        dodaj(w, 'ε', self.spin_geps, 'Zmiana wagi składnika straty obrazu w slocie p.')
        self.edit_skale = QLineEdit('1e-4, 1e-3, 1e-2'); self.edit_skale.setMaximumWidth(140)
        dodaj(w, 'skale ε', self.edit_skale, 'Względne skale zaburzenia wag (wzrost zaburzeń).')
        self.spin_gH = QSpinBox(); self.spin_gH.setRange(1, 200); self.spin_gH.setValue(20)
        dodaj(w, 'H', self.spin_gH, 'Liczba kroków gałęzi.')
        self.edit_gh = QLineEdit('1, 2, 5, 10, 20'); self.edit_gh.setMaximumWidth(120)
        dodaj(w, 'pomiary po', self.edit_gh, 'Po ilu krokach od interwencji mierzyć (1 = sam krok z interwencją).')
        self.spin_gp = QSpinBox(); self.spin_gp.setRange(0, 10); self.spin_gp.setValue(2)
        dodaj(w, 'inne przyszłości', self.spin_gp, 'Ile dodatkowych przyszłych sekwencji paczek (oprócz prawdziwej).')
        self.chk_lin = QCheckBox('liniowość (2ε)'); self.chk_lin.setChecked(True); w.addWidget(self.chk_lin)
        self.chk_lin.setToolTip('Dodatkowa gałąź z interwencją 2ε: sprawdza, czy odpowiedź na 2ε jest równa 2 × odpowiedź na ε (liniowość).')
        self.chk_sup = QCheckBox('superpozycja (sloty p, p+1)'); w.addWidget(self.chk_sup)
        self.chk_sup.setToolTip('Dodatkowa gałąź z interwencją w dwóch sąsiednich slotach paczki (p i p+1): sprawdza, czy odpowiedź na oba\n'
                                'razem jest sumą odpowiedzi na każdy osobno (superpozycja). Dotyczy interwencji typu „waga”.')
        self.chk_det = QCheckBox('deterministycznie'); self.chk_det.setChecked(True); w.addWidget(self.chk_det)
        self.chk_det.setToolTip('Atencja MATH i deterministyczne algorytmy: dwie identyczne gałęzie dają różnicę 0. Bez '
                                'tego (jak w prawdziwym biegu) szum GPU po kilku krokach dorównuje odpowiedzi.')
        self.btn_gal = rola(QPushButton('Licz gałęzie'), 'glowny'); w.addWidget(self.btn_gal)
        self.btn_gal.setToolTip('Odtwarza uczenie od wybranego kroku w gałęziach (z interwencją, kontrolna bez niej) i mierzy różnice\n'
                                'po zadanych krokach. Liczy się w tle, przerywa „Zatrzymaj”.')
        self.fig_g = Figure(figsize=(11, 7)); self.can_g = PlotnoCiemne(self.fig_g); self.can_g.setMinimumHeight(560)
        sl.addWidget(self.can_g, 1)
        self.tabs.addTab(self._przew(s), 'Gałęzie uczenia')
        # obserwator
        s = QWidget(); sl = QVBoxLayout(s)
        w = Przeplyw(); sl.addLayout(w)
        self.combo_kseq = QComboBox(); self.combo_kseq.addItems(SEKWENCJE)
        dodaj(w, 'sekwencja', self.combo_kseq, 'Pełny zapis = prawda odniesienia; filtr widzi tylko część tokenów.')
        self.spin_kr = QSpinBox(); self.spin_kr.setRange(1, 30); self.spin_kr.setValue(6)
        dodaj(w, 'r', self.spin_kr, 'Liczba modów stanu (stan = [y(t), y(t−1)]).')
        self.spin_kfit = QDoubleSpinBox(); self.spin_kfit.setRange(.2, .9); self.spin_kfit.setValue(.6)
        dodaj(w, 'część do dopasowania', self.spin_kfit, 'Model, Q i R tylko z tej części.')
        self.spin_ku = QDoubleSpinBox(); self.spin_ku.setRange(.01, 1); self.spin_ku.setValue(.25)
        dodaj(w, 'ułamek tokenów', self.spin_ku, 'Jaka część tokenów jest mierzona.')
        self.spin_kco = QSpinBox(); self.spin_kco.setRange(1, 100); self.spin_kco.setValue(1)
        dodaj(w, 'pomiar co', self.spin_kco, 'Co ile kroków sekwencji jest pomiar.')
        self.spin_kz = QSpinBox(); self.spin_kz.setRange(0, 10 ** 6)
        dodaj(w, 'ziarno', self.spin_kz, 'Ziarno wyboru mierzonych tokenów.')
        self.btn_kal = rola(QPushButton('Licz obserwator'), 'glowny'); w.addWidget(self.btn_kal)
        self.btn_kal.setToolTip('Dopasowuje obserwator (filtr Kalmana) do wybranej sekwencji: widzi tylko część tokenów\n'
                                'i odtwarza resztę. Porównanie z pełnym zapisem na wykresie poniżej.')
        self.fig_k = Figure(figsize=(11, 7)); self.can_k = PlotnoCiemne(self.fig_k); self.can_k.setMinimumHeight(560)
        sl.addWidget(self.can_k, 1)
        self.tabs.addTab(self._przew(s), 'Obserwator (Kalman)')

        # ------------------------------------------------------------ sygnały
        self.btn_ust.toggled.connect(lambda v: (self.ust.setVisible(v), self.btn_ust.setText('Opcje ' + ('▾' if v else '▸'))))
        self.combo_zr.currentIndexChanged.connect(self._zmiana_zrodla)
        self.spin_obr.valueChanged.connect(lambda *_: self._wczytaj_obraz())
        self.chk_z_ucz.toggled.connect(lambda *_: self._wczytaj_obraz())
        for c in (self.combo_tryb, self.combo_kan, self.combo_tok, self.combo_cel):
            c.currentIndexChanged.connect(lambda *_: self._koszt())
        for c in (self.spin_kroki, self.spin_co):
            c.valueChanged.connect(lambda *_: self._koszt())
        self.btn_hist.clicked.connect(self.historyczny)
        self.btn_licz.clicked.connect(self.licz)
        self.btn_stop.clicked.connect(self.zatrzymaj)
        self.btn_por.clicked.connect(self.porownaj)
        self.btn_zapisz.clicked.connect(self.zapisz)
        self.btn_wczytaj.clicked.connect(self.wczytaj)
        self.btn_dyn.clicked.connect(self.licz_dynamike)
        self.btn_gal.clicked.connect(self.licz_galezie)
        self.btn_kal.clicked.connect(self.licz_obserwator)
        self.combo_baza.currentIndexChanged.connect(lambda *_: self._przelicz_rzut())
        self.sl_kor.valueChanged.connect(lambda *_: self.rysuj_uczenie())
        self.sl_ucz.valueChanged.connect(lambda *_: self.rysuj_uczenie())
        self.timer = QTimer(self); self.timer.setInterval(160); self.timer.timeout.connect(self._klatka)
        self.btn_graj.toggled.connect(lambda v: self.timer.start() if v else self.timer.stop())
        self._wczytaj_obraz()

    # ------------------------------------------------------------ układ
    def _przew(self, w):
        a = QScrollArea(); a.setWidget(w); a.setWidgetResizable(True); return a

    def _strona(self, nazwa, figsize):
        f = Figure(figsize=figsize); c = PlotnoCiemne(f); c.setMinimumHeight(520)
        self.tabs.addTab(self._przew(c), nazwa)
        return f, c

    def _status(self, t):
        self.etykieta.setText(t)

    # ------------------------------------------------------------ obraz
    def _zmiana_zrodla(self, *_):
        self.spin_obr.setRange(0, 399 if self.combo_zr.currentIndex() == 0 else 11)
        self._wczytaj_obraz()

    def _manifest(self):
        if self._rows is None:
            self._rows = KL.manifest()
        return self._rows

    def _wczytaj_obraz(self):
        """Obraz i etykieta (bez liczenia sieci). Brak zasobu lub błędna adnotacja: jasny status."""
        try:
            if self.combo_zr.currentIndex() == 0:
                i = self.spin_obr.value()
                row = self._manifest()[i]
                self.rgb, self.ramki = KL.obraz_manifestu(row)
                wb = row['frame_id'] in KL.ramki_bazy_uczenia()
                self.opis_obr = dict(id=f'm{i:04d}', zrodlo='manifest J128', indeks=i, frame_id=row['frame_id'],
                                     film=row['film'], split=row['split'], w_bazie_uczenia=wb,
                                     etykieta='bbox_xywh z manifestu (cystotom), piksele 480 x 320')
                self.etykieta_obr.setText(f'{row["film"]}, {row["split"]}, '
                                          f'{"W BAZIE UCZENIA (mógł uczyć ViT)" if wb else "poza bazą uczenia"}')
            else:
                import galezie_uczenia as GU
                b = GU.bieg_gestej_historii()
                if self.chk_z_ucz.isChecked() and self.u is not None and getattr(self.u, 'Z', None) is not None:
                    self.spin_obr.blockSignals(True); self.spin_obr.setValue(int(self.u.i_kl)); self.spin_obr.blockSignals(False)
                    krok = int(self.u.idx[self.u.t]); k10 = min((krok // 10) * 10, 1000)
                    j = [o for o, _ in self.ckpty].index(f'gęsta historia k{k10:04d}') if f'gęsta historia k{k10:04d}' in [o for o, _ in self.ckpty] else None
                    if j is not None:
                        self.combo_ckpt.setCurrentIndex(j)
                X, R, Mk, N = GU.sondy_gestej_historii(b)
                i = self.spin_obr.value()
                self.rgb = np.ascontiguousarray(X[i]); self.ramki = KL.sprawdz_ramki(R[i]) if len(R[i]) else np.zeros((0, 4), np.float32)
                self.opis_obr = dict(id=f'sonda{i:02d}', zrodlo='klatka próbna gęstej historii', indeks=i, opis=N[i],
                                     etykieta='ramka z obrysu cystotomu (images.json), piksele 480 x 320',
                                     w_bazie_uczenia=None)
                self.etykieta_obr.setText(N[i] + ('' if len(self.ramki) else ', BRAK obiektu w etykiecie'))
        except Exception as e:
            self.rgb = None; self.ramki = None
            self._status(f'Obraz niedostępny: {e}')
            return
        self._koszt()
        self.rysuj_obraz()

    def _zaznaczenie(self):
        u = self.u
        return None if u is None else getattr(u, 'zaznaczenie', None)

    def _kanaly(self):
        i = self.combo_kan.currentIndex()
        if i == 0:
            return KL.kanaly_j128()
        if i == 1:
            return np.arange(768)
        k = np.array(sorted({c for c in _lista_int(self.edit_kan.text(), []) if 0 <= c < 768}), np.int64)
        if not len(k):
            raise ValueError('nie wpisano poprawnych kanałów 0-767')
        return k

    def _spec(self):
        tryb = list(KL.TRYBY)[self.combo_tryb.currentIndex()]
        return dict(tryb=tryb, kanaly=self._kanaly(), tokeny=self.combo_tok.currentIndex(),
                    cel=list(KL.CELE)[self.combo_cel.currentIndex()], norma=list(KL.NORMY)[self.combo_norma.currentIndex()],
                    rho=self.spin_rho.value(), lam=self.spin_lam.value(), lam_fp=self.spin_lamfp.value(),
                    kroki=self.spin_kroki.value(), lr=self.spin_lr.value(), optymalizator=self.combo_opt.currentText(),
                    ziarno=self.spin_ziarno.value(), start=self.combo_start.currentText(), sigma_start=self.spin_sigma.value(),
                    zapis_co=self.spin_co.value(), tau=self.spin_tau.value(), prog_iou=0.5,
                    prog_iou_cel=self.spin_pi.value(), prog_fp=self.spin_pf.value(), rho_min=self.spin_rmin.value(),
                    rho_max=self.spin_rmax.value(), n_rho=self.spin_nr.value())

    def _koszt(self):
        try:
            sp = self._spec()
            t = self.combo_tok.currentIndex()
            if t == 0:
                nt = 600
            elif t == 1:
                z = self._zaznaczenie(); nt = len(z) if z is not None else 0
            else:
                nt = int(KL.maski_gt(self.ramki)[['obiekt', 'granica', 'tlo'][t - 2]].sum()) if self.ramki is not None and len(self.ramki) else 0
            nc = len(sp['kanaly']) if sp['tryb'] in KL.TRYBY_LATENTU else 192
            b = KL.koszt_zapisu((nt, nc), sp['kroki'], sp['zapis_co'])
            self.etykieta_koszt.setText(f'zapis trajektorii: {nt} tokenów × {nc}, około {b / 2 ** 20:.0f} MiB')
        except Exception as e:
            self.etykieta_koszt.setText(f'({e})')

    # ------------------------------------------------------------ wątek
    def _uruchom(self, zadanie, gotowe, przyciski=()):
        if self.watek is not None:
            self._status('Trwa inne obliczenie; poczekaj albo kliknij „Zatrzymaj”.'); return
        from zakladka_filmu import WatekFilmu
        self.watek = WatekFilmu(zadanie, self)
        self.watek.postep.connect(lambda a, t: (self.pasek.setValue(a), self._status(t)))
        self.watek.gotowy.connect(lambda w: (self._koniec_watku(), gotowe(w)))
        self.watek.blad.connect(lambda t: (self._koniec_watku(), self._status('Błąd: ' + t[-900:])))
        for b in (self.btn_licz, self.btn_hist, self.btn_por, self.btn_dyn, self.btn_gal, self.btn_kal):
            b.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.watek.start()

    def _koniec_watku(self):
        self.watek = None
        for b in (self.btn_licz, self.btn_hist, self.btn_por, self.btn_dyn, self.btn_gal, self.btn_kal):
            b.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def zatrzymaj(self):
        if self.watek is not None:
            self.watek.stop = True
            self._status('Zatrzymuję... (częściowe wyniki zostaną zapisane)')

    def _model(self, wt=None):
        """Własna kopia ViT dla wybranego checkpointu (ładowana w wątku przy pierwszym obliczeniu)."""
        if not self.ckpty:
            raise FileNotFoundError(f'brak checkpointów w {KL.WAGI_GESTE}')
        _, p = self.ckpty[self.combo_ckpt.currentIndex()]
        if self.st is None or self.st.ckpt != Path(p):
            if wt:
                wt.postep.emit(2, f'ładuję {Path(p).name} i liczę granice kanałów oraz SVD fc2...')
            self.st = KL.Stanowisko(p)
        return self.st

    # ------------------------------------------------------------ historyczny import
    def historyczny(self):
        if self.combo_zr.currentIndex() != 0:
            self._status('Historyczny wynik dotyczy tylko obrazów manifestu J128.'); return
        i = self.spin_obr.value()

        def zad(wt):
            st = self._model(wt)
            wt.postep.emit(30, 'import pary przed/po i AP zbioru 400 obrazów...')
            h = KL.import_historyczny(st, i)
            h['ap'] = KL.ap_historyczne()
            h['fc2_macierz'] = st.fc2
            return h

        def gotowe(h):
            self.hist = h; self.wynik = None
            o = h['ostrzezenia']
            ap = h['ap']['policzone']
            self._status(f'Historyczny wynik obrazu {i}: IoU najpewniejszej ramki {h["przed"]["iou_top"]:.3f} -> '
                         f'{h["po"]["iou_top"]:.3f}, strata {h["przed"]["strata"]:.3f} -> {h["po"]["strata"]:.3f}. Zbiór 400: '
                         f'AP50 {ap["baseline"]["AP50"]:.4f} -> {ap["optimized"]["AP50"]:.4f}, AP50:95 '
                         f'{ap["baseline"]["AP50_95"]:.4f} -> {ap["optimized"]["AP50_95"]:.4f} (oracle na optymalizowanej '
                         f'puli). ' + ('OSTRZEŻENIA: ' + '; '.join(o) if o else 'Checkpoint, kanały i bazowy latent zgodne.')
                         + ' ' + h['opis'])
            self.rysuj_obraz(); self.rysuj_fc2(); self.fig_p.clear(); self.can_p.draw_idle()
        self._uruchom(zad, gotowe)

    # ------------------------------------------------------------ korekta
    def licz(self):
        if self.rgb is None:
            self._status('Brak obrazu: nie ma czego korygować.'); return
        try:
            spec = self._spec()
        except Exception as e:
            self._status(f'Ustawienia: {e}'); return
        if spec['cel'] in ('lokalizacja', 'najmniejsza', 'detekcja') and (self.ramki is None or not len(self.ramki)):
            self._status('Brak etykiety obiektu: ten cel wymaga ramki GT.'); return
        rgb, ramki, opis = self.rgb, self.ramki, dict(self.opis_obr)
        zazn = self._zaznaczenie()

        def zad(wt):
            st = self._model(wt)
            return KL.oblicz_korekte(st, rgb, ramki, spec, opis, zazn, przerwij=lambda: wt.stop,
                                     postep=lambda a, t: wt.postep.emit(a, t))

        def gotowe(w):
            self.wynik = w; self.hist = None; self.tor = None; self.por = None
            m0, mb = w['przed'], w['najlepszy']
            spr = w['sprawdzenia']
            pf = mb['fc2']
            self._status(
                f'[{w["status"]}] {w["kat"]}. Krok {mb["krok"]} (najlepszy): strata {m0["strata"]:.4f} -> {mb["strata"]:.4f}, '
                f'IoU najpewniejszej ramki {_f(m0["iou_top"])} -> {_f(mb["iou_top"])}, pewność {m0["pewnosc_top"]:.3f} -> '
                f'{mb["pewnosc_top"]:.3f}, FP {m0["fp"]} -> {mb["fp"]}. ||Δz|| {mb["norma_dz"]:.3g}, część niewidoczna dla '
                f'fc2 ' + ('nieokreślona (korekta zerowa)' if pf['udzial_niewidocznej'] is None else
                           f'{100 * pf["udzial_niewidocznej"]:.1f} % sumy kwadratów') + '.' + f' Sprawdzenia: zerowa interwencja '
                f'{spr["zerowa_interwencja_blad"]:.1e}, wagi bez zmian {spr["wagi_bez_zmian"]}, poza interwencją bez '
                f'zmian {spr["poza_interwencja_bez_zmian"]}. Oracle z etykietą tego obrazu, nie detektor bez etykiet.'
                + (f' {w["uwaga"]}' if w.get('uwaga') else ''))
            self.pasek.setValue(100)
            self.rysuj_wszystko()
        self._uruchom(zad, gotowe)

    def wczytaj(self):
        kat = QFileDialog.getExistingDirectory(self, 'Folder pomiaru', str(KL.WYJSCIE))
        if not kat:
            return
        try:
            w = KL.wczytaj_pomiar(kat)
        except Exception as e:
            self._status(f'Nie wczytano: {e}'); return
        if 'najlepszy' not in w:
            self._status(f'Pomiar {kat}: status {w["status"]}, brak wyników końcowych '
                         f'{w.get("status_pelny", {}).get("blad", "")[-300:]}'); return
        w['spec_run'] = dict(w['spec_publiczna'], **({'norma': 'calosc', 'rho': w['rho_wybrane'], 'cel': 'strata'}
                                                     if w.get('rho_wybrane') else {}))
        if w.get('obraz') is not None:
            self.rgb = w['obraz']; self.ramki = np.asarray(w['konfiguracja'].get('ramki_gt') or np.zeros((0, 4)), np.float32)
        self.wynik = w; self.hist = None; self.tor = None; self.por = None
        sha = w['konfiguracja'].get('checkpoint_sha256')
        uw = '' if self.st is None or self.st.sha == sha else ' Uwaga: wybrany checkpoint różni się od checkpointu pomiaru.'
        self._status(f'Wczytano {kat} (status {w["status"]}).' + uw)
        self.rysuj_wszystko()

    # ------------------------------------------------------------ porównanie z uczeniem
    def _dz_korekty(self, st):
        """Δz korekty [600, 768] przy checkpoincie modelu i baza przebiegu."""
        import torch
        w = self.wynik
        baza = st.przebieg(self.rgb)
        if w is not None and 'koniec_npz' not in w:
            pass
        kn = w.get('koniec_npz')
        if kn is None:
            with np.load(Path(w['kat']) / 'KONIEC.npz') as f:
                kn = {n: f[n] for n in f.files}
        spec = KL.spec_z_zapisu(st, w['konfiguracja'], self.ramki)
        if w.get('rho_wybrane'):
            spec.update(norma='calosc', rho=w['rho_wybrane'], cel='strata')
        with torch.no_grad():
            zp, _, _ = st.wyjscie(torch.from_numpy(kn['d_najlepszy']).to(st.dev), baza, spec)
        return (zp[0] - baza['z']).cpu().numpy(), baza, spec

    def porownaj(self):
        if self.rgb is None:
            self._status('Brak obrazu.'); return
        if self.wynik is None and self.hist is None:
            self._status('Najpierw policz korektę (albo wczytaj historyczny wynik), żeby było z czym porównać.'); return
        rgb, ramki = self.rgb, self.ramki
        tor_stary = self.tor if self.tor is not None and self.tor.get('_id') == self.opis_obr.get('id') else None

        def zad(wt):
            st = self._model(wt)
            if self.wynik is not None:
                if self.wynik['konfiguracja'].get('checkpoint_sha256') != st.sha:
                    raise ValueError('pomiar korekty dotyczy innego checkpointu niż wybrany; wybierz jego checkpoint')
                dz, baza, spec = self._dz_korekty(st)
            else:
                h = self.hist
                baza = st.przebieg(rgb)
                dz = np.zeros((600, 768), np.float32); J = KL.kanaly_j128()
                H = KL._hist(); dz[:, J] = H['target'][h['pos']] - baza['z'][:, J].cpu().numpy()
                spec = None
            tor = tor_stary or KL.tor_uczenia(rgb, ramki, postep=lambda a, t: wt.postep.emit(int(a * .8), t),
                                              przerwij=lambda: wt.stop)
            tor['_id'] = self.opis_obr.get('id')
            if st.krok not in set(tor['kroki'].tolist()):
                raise ValueError(f'checkpoint korekty (krok {st.krok}) nie jest w torze uczenia gęstej historii')
            wt.postep.emit(85, 'porównanie i rzut 3D...')
            por = KL.porownaj_z_uczeniem(tor, st.krok, dz)
            return dict(tor=tor, por=por, dz=dz, spec=spec, baza=baza)

        def gotowe(r):
            self.tor, self.por = r['tor'], r['por']
            self._por_ctx = r
            p = self.por
            self._status(f'Tor uczenia: {len(self.tor["kroki"])} checkpointów (oś network_training, co 10 kroków). '
                         f'Prędkość uczenia tuż przed k_c = {p["k_c"]} (tylko przeszłość) ma kosinus '
                         f'{p.get("cos_predkosci_przed", float("nan")):+.3f} z kierunkiem korekty. Kosinusy dla k > k_c są '
                         f'retrospektywne (przyszłe stany uczenia, tylko wyjaśnienie).')
            self._przelicz_rzut()
            self.tabs.setCurrentIndex(3)
        self._uruchom(zad, gotowe)

    def _przelicz_rzut(self):
        if self.por is None:
            return
        r = self._por_ctx; st = self.st
        try:
            orac = None
            if self.combo_baza.currentIndex() == 2 and self.wynik is not None:
                orac = KL.probka_trajektorii(st, self.wynik['kat'], r['spec'], r['baza'])
            Q, opis = KL.baza_projekcji(self.combo_baza.currentIndex(), r['dz'], self.tor, st.krok, orac)
            Z = self.tor['z'].reshape(len(self.tor['kroki']), -1)
            ic = int(np.flatnonzero(self.tor['kroki'] == st.krok)[0])
            self._xyz_u = (Z - Z[ic]) @ Q.T
            vu = np.full(len(Z), np.nan); vu[1:] = np.linalg.norm(np.diff(Z, axis=0), axis=1) / np.diff(self.tor['kroki'])
            self._v_u = vu
            if self.wynik is not None:
                self._kr_k, self._xyz_k, self._v_k = KL.rzut_trajektorii(st, self.wynik['kat'], r['spec'], r['baza'], Q)
            else:
                self._kr_k = np.array([0, 1]); self._xyz_k = np.stack([np.zeros(3), Q @ r['dz'].reshape(-1)])
                self._v_k = None
            self._opis_bazy = opis
            self.sl_kor.setRange(0, max(len(self._kr_k) - 1, 0)); self.sl_kor.setValue(len(self._kr_k) - 1)
            self.sl_ucz.setRange(0, len(self.tor['kroki']) - 1); self.sl_ucz.setValue(ic)
        except Exception as e:
            self._status(f'Rzut 3D: {e}'); return
        self.rysuj_uczenie()

    def _klatka(self):
        for s in (self.sl_kor, self.sl_ucz):
            s.setValue(s.value() + 1 if s.value() < s.maximum() else 0)

    # ------------------------------------------------------------ modele dynamiki
    def _sekwencja(self, i, wt=None):
        """(X, opis osi, interwał, funkcja wyjścia albo None, gradienty albo None, kształt tokenów)."""
        import torch
        if i == 0:
            if self.wynik is None:
                raise ValueError('brak toru korekty: policz albo wczytaj korektę (import historyczny nie ma trajektorii)')
            kr, D = KL.wczytaj_trajektorie(self.wynik['kat'], 'd')
            if D is None or len(kr) < 10:
                raise ValueError('za krótki zapisany tor korekty')
            co = int(np.median(np.diff(kr))) if len(kr) > 1 else 1
            st = self._model(wt)
            if self.wynik['konfiguracja'].get('checkpoint_sha256') != st.sha:
                raise ValueError('tor korekty z innego checkpointu niż wybrany')
            baza = st.przebieg(self.rgb); spec = KL.spec_z_zapisu(st, self.wynik['konfiguracja'], self.ramki)
            if self.wynik.get('rho_wybrane'):
                spec.update(norma='calosc', rho=self.wynik['rho_wybrane'], cel='strata')
            cel = KL.cel_z_ramek(self.ramki, st.dev); ksz = D.shape[1:]

            def wyj(x):
                with torch.no_grad():
                    _, _, o = st.wyjscie(torch.from_numpy(np.asarray(x, np.float32).reshape(ksz)).to(st.dev), baza, spec)
                    m = st.ocena(o, self.ramki, cel)
                return m['strata'], m['iou_top']
            G = None
            if co == 1:
                _, G = KL.wczytaj_trajektorie(self.wynik['kat'], 'g')
                G = G.reshape(len(G), -1)
            return D, f'latent_optimization (krok korekty, zapis co {co})', co, wyj, G, ksz
        if i == 1:
            if self.tor is None:
                raise ValueError('brak toru uczenia: kliknij „Porównaj z ruchem uczenia”')
            Z = self.tor['z']; p = self.combo_prz.currentIndex()
            maska = KL.maski_gt(self.ramki)['obiekt'] if self.ramki is not None and len(self.ramki) else np.ones(600, bool)
            if p == 0 and self.wynik is not None:
                sp = self.wynik['konfiguracja']['spec']
                m = np.asarray(sp['maska_tokenow'], bool); kan = np.asarray(sp['kanaly'], int)
                X = Z[:, m][:, :, kan]
            elif p == 2:
                X = Z[:, maska].mean(1)[:, None, :]
            else:
                X = Z[:, maska]
            return X, 'network_training (krok uczenia, checkpointy co 10)', 10, None, None, X.shape[1:]
        u = self.u
        if u is None or getattr(u, 'Z', None) is None:
            raise ValueError('wczytaj historię w zakładce „Uczenie ViT”')
        import analiza_uczenia as AU
        b = u._bieg(); op = AU.opis_biegu(b)
        m = np.asarray(op['narzedzie'][u.i_kl], bool)
        if self.combo_prz.currentIndex() == 2:
            X = np.asarray(u.Z[:, m], np.float32).mean(1)[:, None, :]
        else:
            X = np.asarray(u.Z[:, m], np.float32)
        return X, f'network_training (krok uczenia, co 1; klatka {u.i_kl})', 1, None, None, X.shape[1:]

    def licz_dynamike(self):
        i = self.combo_seq.currentIndex()
        par = dict(frac_fit=self.spin_fit.value(), hs=tuple(_lista_int(self.edit_h.text(), [1, 5, 20])), L=self.spin_L.value(),
                   okno=self.spin_okno.value(), r=self.spin_r.value(), d=self.spin_d.value())

        def zad(wt):
            import dynamika_korekty as DK
            X, os_, co, wyj, G, ksz = self._sekwencja(i, wt)
            X2 = np.asarray(X, np.float64).reshape(len(X), -1)
            wt.postep.emit(10, f'modele na sekwencji {X2.shape} ({os_})...')
            r = DK.ocen_modele(X2, wyjscie=wyj, postep=lambda a, t: wt.postep.emit(10 + int(a * .7), t), **par)
            r['os'] = os_; r['interwal'] = co; r['ksztalt'] = list(X2.shape)
            try:
                r['blad'] = DK.blad_modelu(X2, par['frac_fit'], par['L'], par['okno'])
            except Exception as e:
                r['blad'] = None
            if G is not None:
                wt.postep.emit(85, 'modele z wejściem optymalizatora (h = 1)...')
                r['wejscie'] = {'gradient celu g(t)': DK.porownaj_wejscie(X2, G, par['frac_fit'], okno=par['okno'])}
                konf = self.wynik['konfiguracja']['spec']
                if konf.get('optymalizator', 'adam') == 'adam':
                    Ua = DK.krok_adama(G, float(konf.get('lr', .1)))
                    r['wejscie']['krok Adama z momentów u(t)'] = DK.porownaj_wejscie(X2, Ua, par['frac_fit'], okno=par['okno'])
            return r

        def gotowe(r):
            self.dyn = r
            self._status(f'Modele dyniki: {r["os"]}, sekwencja {r["ksztalt"]}, {r["opis"]}.'.replace('dyniki', 'dynamiki'))
            self.rysuj_dynamike(); self.pasek.setValue(100)
        self._uruchom(zad, gotowe)

    # ------------------------------------------------------------ gałęzie uczenia
    def licz_galezie(self):
        k = self.spin_gk.value(); typ = self.combo_gi.currentIndex()
        H = self.spin_gH.value(); hs = _lista_int(self.edit_gh.text(), [1, 5, 20])
        przysz = (None,) + tuple(range(1, self.spin_gp.value() + 1))
        rgb, ramki, opis = self.rgb, self.ramki, dict(self.opis_obr)
        det = self.chk_det.isChecked(); lin = self.chk_lin.isChecked(); sup = self.chk_sup.isChecked()
        slot, eps = self.spin_slot.value(), self.spin_geps.value()
        skale = _lista_float(self.edit_skale.text(), [1e-4, 1e-3, 1e-2])
        if typ == 1 and (rgb is None or ramki is None):
            self._status('Podmiana wymaga wybranego obrazu z etykietą.'); return

        def zad(wt):
            import torch
            import galezie_uczenia as GU
            from stan_uczenia import Odtworzenie
            if not torch.cuda.is_available():
                raise RuntimeError('brak GPU: gałęzie uczenia wymagają CUDA')
            b = GU.bieg_gestej_historii()
            if self.od is None:
                wt.postep.emit(2, 'baza uczenia i powtórka (pierwszy raz do 2 minut)...')
                self.od = Odtworzenie(b)
            self.od.przygotuj(lambda a, n: wt.postep.emit(2 + int(10 * a / n), f'powtórka uczenia {a}/{n}'))
            X, R, Mk, N = GU.sondy_gestej_historii(b)
            if rgb is not None:
                m = KL.maski_gt(ramki)['obiekt'] if ramki is not None and len(ramki) else np.zeros(600, bool)
                X = np.concatenate([X, rgb[None]]); R = R + [ramki if ramki is not None else np.zeros((0, 4))]
                Mk = np.concatenate([Mk, m[None]]); N = N + [f'wybrany obraz {opis.get("id")}']
            G = GU.Galezie(self.od, X, R, Mk, N)
            post = lambda a, t: wt.postep.emit(12 + int(a * .85), t)
            if typ == 2:
                w = GU.licz_wzrost(G, k, skale=skale, H=H, horyzonty=hs, przyszlosci=przysz, postep=post,
                                   przerwij=lambda: wt.stop, deterministyczne=det)
                w['typ'] = 'wzrost'
            else:
                if typ == 0:
                    iw = dict(rodzaj='waga', sloty=[slot], eps_slotow=[eps])
                else:
                    x1 = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().cuda() / 255
                    c = [t.cuda() for t in self.od.M.cele([np.asarray(ramki, np.float32).reshape(-1, 4)], 20, 30)]
                    iw = dict(rodzaj='podmiana', sloty=[slot], x=x1, cel=c, obraz=opis)
                w = GU.licz_odpowiedz(G, k, iw, H=H, horyzonty=hs, przyszlosci=przysz, liniowosc=lin, superpozycja=sup,
                                      postep=post, przerwij=lambda: wt.stop, deterministyczne=det,
                                      nazwa='waga' if typ == 0 else 'podmiana')
                w['typ'] = 'odpowiedz'
            w['sondy'] = N
            return w

        def gotowe(w):
            self.gal = w
            self._status(f'Gałęzie [{w["status"]}] w {w["czas_s"]:.0f} s, zapis {w["kat"]}. '
                         f'{"Deterministycznie: szum identycznych gałęzi powinien być 0." if det else "Bez determinizmu: szum GPU rośnie z krokami."}')
            self.rysuj_galezie(); self.pasek.setValue(100); self.tabs.setCurrentIndex(5)
        self._uruchom(zad, gotowe)

    # ------------------------------------------------------------ obserwator
    def licz_obserwator(self):
        i = self.combo_kseq.currentIndex()
        par = dict(frac_fit=self.spin_kfit.value(), r=self.spin_kr.value(), ulamek_tokenow=self.spin_ku.value(),
                   co_ile=self.spin_kco.value(), ziarno=self.spin_kz.value())

        def zad(wt):
            import dynamika_korekty as DK
            X, os_, co, _, _, ksz = self._sekwencja(i, wt)
            X = np.asarray(X, np.float64)
            if X.ndim == 2:
                X = X[:, None, :]
            X = X.reshape(len(X), X.shape[1], -1)
            wt.postep.emit(30, f'filtr na {X.shape} ({os_})...')
            r = DK.obserwator(X, **par)
            r['os'] = os_; r['interwal'] = co
            return r

        def gotowe(r):
            self.kal = r
            if 'brak' in r:
                self._status(r['brak']); return
            pk = r['pokrycie_90']
            self._status(f'Obserwator ({r["os"]}): tokeny mierzone {len(r["tokeny_obserwowane"])}/{r["n_tok"]}, koszt '
                         f'{100 * r["koszt"]["lacznie"]:.0f} % pełnego pomiaru, rząd obserwowalności '
                         f'{r["obserwowalnosc"]["rzad"]}/{r["obserwowalnosc"]["wymiar_stanu"]}, pokrycie przedziałów 90 %: '
                         f'prior {pk["prior"]:.2f}, posterior {pk["posterior"] if pk["posterior"] is None else round(pk["posterior"], 2)}.')
            self.rysuj_obserwator(); self.pasek.setValue(100)
        self._uruchom(zad, gotowe)

    # ------------------------------------------------------------ zapis
    def zapisz(self):
        kat = None
        if self.wynik is not None:
            kat = Path(self.wynik['kat'])
        else:
            from datetime import datetime
            kat = KL.WYJSCIE / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_analiza_{self.opis_obr.get('id', 'obraz')}"
            kat.mkdir(parents=True, exist_ok=True)
        an = {}
        if self.hist is not None:
            h = self.hist
            an['historyczny'] = dict(obraz=h['i'], przed=h['przed'], po=h['po'], ostrzezenia=h['ostrzezenia'],
                                     fc2={k: v for k, v in h['fc2'].items() if k != 'mapa_niewidocznej'},
                                     naruszenia=h['naruszenia'], ap=h['ap'], opis=h['opis'])
        if self.por is not None:
            an['porownanie_z_uczeniem'] = dict({k: v for k, v in self.por.items()}, time_axis='network_training',
                                              miary_uczenia=self.tor['miary'], kroki=self.tor['kroki'])
        if self.dyn is not None:
            an['modele_dynamiki'] = self.dyn
        if self.kal is not None:
            an['obserwator'] = self.kal
        if self.gal is not None:
            an['galezie'] = dict(folder=self.gal['kat'], status=self.gal['status'])
        KL._json(kat / 'ANALIZY.json', an)
        if self.tor is not None:
            np.save(kat / 'TOR_UCZENIA_float16.npy', self.tor['z'].astype(np.float16))
        for nazwa, f in (('obraz', self.fig_o), ('przebieg', self.fig_p), ('fc2', self.fig_f), ('uczenie_3d', self.fig_u),
                         ('dynamika', self.fig_d), ('galezie', self.fig_g), ('obserwator', self.fig_k)):
            if f.axes:
                f.savefig(kat / f'{nazwa}.png', dpi=110, facecolor=f.get_facecolor())
        self._status(f'Zapisano analizy i rysunki w {kat}' + (' (tor uczenia w float16; miary liczone w float32 przed '
                                                              'zapisem)' if self.tor is not None else ''))

    # ------------------------------------------------------------ rysunki
    def rysuj_wszystko(self):
        self.rysuj_obraz(); self.rysuj_przebieg(); self.rysuj_fc2()

    def _ramki_osi(self, ax, o, kolor, styl, etykieta, tau):
        from model_snapshot import dekoduj
        import torch
        bx, sc = dekoduj(torch.from_numpy(np.asarray(o))[None])
        bx, sc = bx[0].numpy(), sc[0].numpy()
        k0 = int(np.argmax(sc))
        for b, s in zip(bx, sc):
            if s >= tau:
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=KOL['det'], lw=.6, ls=':'))
        b = bx[k0]
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=kolor, lw=1.8, ls=styl,
                               label=f'{etykieta}: najpewniejsza ({sc[k0]:.2f})'))
        return bx, sc

    def rysuj_obraz(self):
        f = self.fig_o; f.clear()
        if self.rgb is None:
            self.can_o.draw_idle(); return
        src = self.wynik if self.wynik is not None else self.hist
        tau = self.spin_tau.value()
        ax1 = f.add_axes([.01, .08, .36, .82]); ax2 = f.add_axes([.38, .08, .36, .82]); ax3 = f.add_axes([.76, .02, .23, .96])
        for ax, tyt in ((ax1, 'przed korektą'), (ax2, 'po korekcie (najlepszy krok)')):
            ax.imshow(self.rgb, extent=(0, 480, 320, 0)); ax.set_xlim(0, 480); ax.set_ylim(320, 0); ax.axis('off')
            for b in (self.ramki if self.ramki is not None else []):
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=KOL['gt'], lw=1.6, label='GT'))
            ax.set_title(tyt, fontsize=9)
        ax3.axis('off')
        if src is None:
            ax3.text(0, .98, f'{self.opis_obr}\n\nBrak korekty: kliknij „Oblicz korektę”\nalbo „Wczytaj historyczny wynik”.',
                     va='top', fontsize=8, color='#e0e0e0', wrap=True)
            self.can_o.draw_idle(); return
        self._ramki_osi(ax1, src['o_przed'], KOL['przed'], '--', 'przed', tau)
        bx, sc = self._ramki_osi(ax2, src['o_po'], KOL['po'], '-', 'po', tau)
        mapa = np.asarray(src.get('mapa_dz'), np.float64).reshape(20, 30) if src.get('mapa_dz') is not None else None
        if mapa is not None:
            ax2.imshow(mapa, extent=(0, 480, 320, 0), cmap='magma', alpha=.45, interpolation='nearest')
        mpo = src['po'] if self.hist is not None and src is self.hist else src['najlepszy']
        if mpo.get('ramka_najlepsza_diag') is not None:
            b = mpo['ramka_najlepsza_diag']
            ax2.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=KOL['diag'], lw=1, ls='-.',
                                    label=f'diagnostyczna: wybrana przez GT (IoU {mpo["iou_najlepsze_diag"]:.2f})'))
        ax2.legend(fontsize=6.5, loc='lower left', framealpha=.5)
        ax1.legend(fontsize=6.5, loc='lower left', framealpha=.5)
        m0 = src['przed']
        wiersze = [('', 'przed', 'po')]
        def fm(v):
            return '-' if v is None else (f'{v:.4f}' if isinstance(v, float) else str(v))
        for k, n in (('strata', 'strata'), ('ogniskowa_poz', 'ogniskowa +'), ('ogniskowa_neg', 'ogniskowa −'),
                     ('ramka', '1 − GIoU'), ('iou_top', 'IoU najpewn.'), ('pewnosc_top', 'pewność'), ('tp', 'trafione'),
                     ('fp', 'fałszywe'), ('pominiete', 'pominięte')):
            wiersze.append((n, fm(m0.get(k)), fm(mpo.get(k))))
        txt = '\n'.join(f'{a:<13}{b:>9} {c:>9}' for a, b, c in wiersze)
        dod = []
        if self.hist is not None and src is self.hist:
            h = self.hist; ap = h['ap']['policzone']
            dod += ['', 'IMPORT HISTORYCZNY (tylko końce)', f'zbiór 400, {KL.EWALUATOR}:',
                    f'AP50  {ap["baseline"]["AP50"]:.4f} -> {ap["optimized"]["AP50"]:.4f}',
                    f'AP50:95 {ap["baseline"]["AP50_95"]:.4f} -> {ap["optimized"]["AP50_95"]:.4f}',
                    f'kontrola random {ap["random"]["AP50"]:.4f}', '(moduły z oracle: zależna od etykiet)',
                    f'poza zakresami ckpt: {100 * h["naruszenia"]["ulamek"]:.2f} % wsp.', f'  max przekr. {h["naruszenia"]["max"]:.2f}',
                    f'w bazie uczenia: {h["w_bazie_uczenia"]}'] + (['', 'OSTRZEŻENIA:'] + h['ostrzezenia'] if h['ostrzezenia'] else [])
        else:
            w = self.wynik
            spr = w.get('sprawdzenia', {})
            dod += ['', f'status {w.get("status")}', f'tryb {w["konfiguracja"]["tryb"]}',
                    f'||Δz|| {mpo.get("norma_dz", 0):.3g}, ||δ|| {mpo.get("norma_d", 0):.3g}',
                    f'niewidoczna dla fc2 {100 * (mpo.get("fc2", {}).get("udzial_niewidocznej") or 0):.1f} %',
                    f'poza zakresami {100 * mpo.get("naruszenia", {}).get("ulamek", 0):.2f} % wsp.',
                    f'zerowa interw. {spr.get("zerowa_interwencja_blad", float("nan")):.1e}',
                    f'wagi bez zmian {spr.get("wagi_bez_zmian")}', f'poza interw. bez zmian {spr.get("poza_interwencja_bez_zmian")}',
                    '', 'kontrole losowe (ta sama norma):']
            for c in w.get('kontrole', []):
                dod.append(f'{c["rodzaj"][:10]} {c["ziarno"]}: strata {c["strata"]:.3f}, IoU {fm(c["iou_top"])}, '
                           f'||δ|| {c["norma_d"]:.3g}')
            dod.append('ORACLE: etykieta tego obrazu')
        ax3.text(0, .99, txt + '\n' + '\n'.join(dod), va='top', family='monospace', fontsize=7, color='#e0e0e0')
        self.can_o.draw_idle()

    def rysuj_przebieg(self):
        f = self.fig_p; f.clear()
        w = self.wynik
        if w is None or not w.get('metryki'):
            self.can_p.draw_idle(); return
        M = w['metryki']; k = np.array([m['krok'] for m in M])
        g = lambda n: np.array([np.nan if m.get(n) is None else m[n] for m in M], np.float64)
        ax = f.subplots(2, 3); f.subplots_adjust(left=.06, right=.98, top=.93, bottom=.08, hspace=.45, wspace=.28)
        a = ax[0, 0]
        for n, c in (('cel', 'white'), ('strata', '#ffd030'), ('ogniskowa_poz', '#1f77b4'), ('ogniskowa_neg', '#9467bd'),
                     ('ramka', '#2ca02c'), ('kara_fp', '#d62728')):
            a.plot(k, g(n), color=c, lw=1, label=n)
        a.set_yscale('symlog', linthresh=1e-3); a.legend(fontsize=6); a.set_title('cel i składniki straty', fontsize=8)
        a = ax[0, 1]
        a.plot(k, g('iou_top'), color='#30ff70', label='IoU najpewniejszej'); a.plot(k, g('pewnosc_top'), color='#ffd030', label='pewność')
        a.plot(k, g('iou_najlepsze_diag'), color=KOL['diag'], ls=':', label='IoU najlepsze (diag., przez GT)')
        a.set_ylim(0, 1.02); a.legend(fontsize=6); a.set_title('jakość obrazu (protokół τ)', fontsize=8)
        a = ax[0, 2]
        a.plot(k, g('norma_dz'), color='white', label='||Δz||'); a.plot(k, g('przechodzaca'), color='#17becf', label='przechodząca przez fc2')
        a.plot(k, g('niewidoczna'), color='#ff7f0e', label='niewidoczna dla fc2')
        a.legend(fontsize=6); a.set_title('norma zmiany latentu i jej części', fontsize=8)
        a = ax[1, 0]
        a.plot(k, g('naruszenie'), color='#d62728', label='suma przekroczeń zakresów')
        a2 = a.twinx(); a2.plot(k, 100 * g('naruszenie_ulamek'), color='#ff9896', ls='--', label='% współrzędnych poza')
        a2.tick_params(labelsize=6)
        a.legend(fontsize=6, loc='upper left'); a2.legend(fontsize=6, loc='lower right')
        a.set_title('naruszenie zakresów z checkpointu', fontsize=8)
        a = ax[1, 1]
        a.plot(g('norma_dz'), g('iou_top'), color='#30ff70', lw=1, label='tor korekty')
        for c in w.get('kontrole', []):
            a.scatter([c['norma_dz']], [c['iou_top'] if c['iou_top'] is not None else np.nan], s=16,
                      c='#bbbbbb' if c['rodzaj'] == 'izotropowa' else '#8c564b')
        if w.get('krzywa_promieni'):
            kp = w['krzywa_promieni']
            a.plot([r['norma_dz'] for r in kp], [r['iou_top'] for r in kp], 'o-', color='#e377c2', ms=3, label='siatka promieni')
            if w.get('rho_wybrane'):
                a.set_title(f'jakość–norma (wybrane ρ = {w["rho_wybrane"]:.3g})', fontsize=8)
        else:
            a.set_title('jakość–norma (szare: losowe kontrole)', fontsize=8)
        a.axhline(self.spin_pi.value(), color='white', ls=':', lw=.7)
        a.set_xlabel('||Δz||', fontsize=7); a.set_ylabel('IoU najpewniejszej', fontsize=7); a.legend(fontsize=6)
        a = ax[1, 2]
        a.plot(k, g('tp'), color='#30ff70', label='trafione'); a.plot(k, g('fp'), color='#d62728', label='fałszywe')
        a.plot(k, g('pominiete'), color='#ffd030', label='pominięte')
        if 'adam_m' in M[-1]:
            a3 = a.twinx(); a3.plot(k, g('adam_m'), color='#9467bd', lw=.8, label='||m|| Adama'); a3.tick_params(labelsize=6)
            a3.set_yscale('log'); a3.legend(fontsize=6, loc='lower right')
        a.legend(fontsize=6, loc='upper left'); a.set_title('detekcje (τ) i moment Adama', fontsize=8)
        for a in f.axes:
            a.tick_params(labelsize=6.5)
        for a in ax.ravel():
            if a is not ax[1, 1]:
                a.set_xlabel('krok korekty (latent_optimization)', fontsize=7)
        f.suptitle(f'Prawdziwa trajektoria optymalizacji ({w["konfiguracja"]["tryb"]}, {w["konfiguracja"]["spec"].get("optymalizator")}, '
                   f'lr {w["konfiguracja"]["spec"].get("lr")}); metryki co krok', fontsize=9, color='white')
        self.can_p.draw_idle()

    def rysuj_fc2(self):
        f = self.fig_f; f.clear()
        st = self.st
        if st is None:
            self.can_f.draw_idle(); return
        ax = f.subplots(2, 2); f.subplots_adjust(left=.07, right=.98, top=.92, bottom=.08, hspace=.45, wspace=.25)
        a = ax[0, 0]
        S = st.fc2['S']; a.semilogy(S, color='#17becf'); a.axhline(st.fc2['tol'], color='#d62728', ls='--', lw=.8)
        a.set_title(f'W2 [192 × 768]: rząd {st.fc2["rzad"]} (tolerancja {st.fc2["tol"]:.1e}), jądro {st.fc2["jadro"]}', fontsize=8)
        a = ax[0, 1]
        J = KL.kanaly_j128(); pj = st.fc2_podzbioru(J)
        a.semilogy(pj['S'], color='#ffd030'); a.axhline(pj['tol'], color='#d62728', ls='--', lw=.8)
        a.set_title(f'W2[:, J128] [192 × 128]: rząd {pj["rzad"]}, jądro {pj["jadro"]}\n(podproblem 128 kanałów nie ma '
                    f'dużego jądra)', fontsize=8)
        a = ax[1, 0]
        sl = []
        if self.wynik is not None and 'najlepszy' in self.wynik:
            fb = self.wynik['najlepszy']['fc2']; sl.append(('korekta', fb))
            fk = self.wynik.get('koniec', {}).get('fc2')
            if fk:
                sl.append(('koniec', fk))
            for c in self.wynik.get('kontrole', [])[:3]:
                sl.append((f'losowa {c["ziarno"]}', c['fc2']))
        if self.hist is not None:
            sl.append(('historyczna J128', self.hist['fc2']))
        for i, (n, p) in enumerate(sl):
            u = p.get('udzial_niewidocznej')
            if u is None:
                a.text(i, .5, 'brak\n(zero)', ha='center', fontsize=6); continue
            a.bar(i, 1 - u, color='#17becf'); a.bar(i, u, bottom=1 - u, color='#ff7f0e')
        a.set_xticks(range(len(sl))); a.set_xticklabels([n for n, _ in sl], fontsize=6.5, rotation=20)
        a.set_ylim(0, 1); a.set_title('udział sumy kwadratów: przechodząca przez fc2 (niebieski) i niewidoczna (pomarańcz.)', fontsize=8)
        a = ax[1, 1]
        src = self.wynik if self.wynik is not None else self.hist
        mapa = None
        if self.wynik is not None:
            mapa = self.wynik.get('mapa_niewidocznej')
        elif self.hist is not None:
            mapa = self.hist['fc2']['mapa_niewidocznej']
        if mapa is not None and self.rgb is not None:
            a.imshow(self.rgb, extent=(0, 480, 320, 0))
            im = a.imshow(np.asarray(mapa, np.float64).reshape(20, 30), extent=(0, 480, 320, 0), cmap='viridis', vmin=0, vmax=1,
                          alpha=.6, interpolation='nearest')
            f.colorbar(im, ax=a, fraction=.03)
            a.set_title('udział części niewidocznej w każdym tokenie (puste = brak korekty)', fontsize=8)
        a.axis('off')
        for x in f.axes:
            x.tick_params(labelsize=6.5)
        f.suptitle('Ruch niewidoczny dla końcówki: zamrożone wagi, stały residual, dropout wyłączony; własność tego punktu, '
                   'w treningu W2 też się zmienia', fontsize=8.5, color='white')
        self.can_f.draw_idle()

    def rysuj_uczenie(self):
        f = self.fig_u; f.clear()
        if self.por is None or not hasattr(self, '_xyz_u'):
            self.can_u.draw_idle(); return
        p = self.por; kr = self.tor['kroki']
        iu = self.sl_ucz.value(); ik = self.sl_kor.value()
        a = f.add_subplot(2, 3, (1, 4), projection='3d')
        U = self._xyz_u; Kk = self._xyz_k
        a.plot(U[:, 0], U[:, 1], U[:, 2], color='#17becf', lw=1.2, label='uczenie z(k) − z(k_c)')
        a.plot(Kk[:, 0], Kk[:, 1], Kk[:, 2], color='#30ff70', lw=1.2, label='korekta Δz(t)')
        a.scatter(*U[iu], color='#17becf', s=40, edgecolors='white'); a.scatter(*Kk[min(ik, len(Kk) - 1)], color='#30ff70', s=40, edgecolors='white')
        a.scatter(0, 0, 0, color='white', s=20)
        a.set_title(self._opis_bazy, fontsize=7.5); a.legend(fontsize=6, loc='upper left')
        for os3 in (a.xaxis, a.yaxis, a.zaxis):
            os3.set_pane_color((.16, .17, .19, 1.))
        a.tick_params(labelsize=5.5)
        a = f.add_subplot(2, 3, 2)
        a.plot(kr, p['cos'], color='white', label='cos(D, Δz) pełny latent')
        a.plot(kr, p['cos_nosnik'], color='#ffd030', label='cos na nośniku korekty')
        a.axvline(p['k_c'], color='#d62728', ls='--', lw=.8); a.axhline(0, color='white', lw=.5, ls=':')
        a.axvspan(p['k_c'], kr.max(), color='#555555', alpha=.25, label='retrospektywnie (przyszłość)')
        a.axvline(kr[iu], color='#17becf', lw=.8)
        a.legend(fontsize=6); a.set_title('kierunek ruchu uczenia wobec korekty', fontsize=8)
        a = f.add_subplot(2, 3, 3)
        a.plot(kr, p['alfa'], color='#30ff70', label='α = <D, Δz>/||Δz||²'); a.plot(kr, p['stosunek'], color='#ff7f0e', label='||D||/||Δz||')
        a.axhline(1, color='white', ls=':', lw=.6); a.axvline(p['k_c'], color='#d62728', ls='--', lw=.8)
        a.set_yscale('symlog', linthresh=.1); a.legend(fontsize=6); a.set_title('postęp wzdłuż korekty', fontsize=8)
        a = f.add_subplot(2, 3, 5)
        mi = self.tor['miary']
        a.plot(kr, [m['iou_top'] if m['iou_top'] is not None else np.nan for m in mi], color='#17becf', label='IoU sieci w kroku k')
        a.plot(kr, [m['strata'] for m in mi], color='#ffd030', label='strata sieci w kroku k')
        src = self.wynik['najlepszy'] if self.wynik is not None else self.hist['po']
        a.axhline(src['iou_top'] or 0, color='#30ff70', ls='--', lw=.8, label='IoU po korekcie oracle')
        a.axvline(p['k_c'], color='#d62728', ls='--', lw=.8); a.set_yscale('symlog', linthresh=.1)
        a.legend(fontsize=6); a.set_title('ten obraz przy prawdziwych wagach', fontsize=8); a.set_xlabel('krok uczenia', fontsize=7)
        a = f.add_subplot(2, 3, 6)
        a.plot(kr, self._v_u, color='#17becf', label='uczenie: ||Δz|| na krok uczenia (z różnic co 10)')
        a.set_xlabel('krok uczenia (network_training)', fontsize=7, color='#17becf'); a.set_yscale('log')
        if self._v_k is not None:
            a2 = a.twiny()
            a2.plot(self._kr_k, self._v_k, color='#30ff70', lw=.9, label='korekta: ||Δz|| na krok korekty')
            a2.set_xlabel('krok korekty (latent_optimization)', fontsize=7, color='#30ff70'); a2.tick_params(labelsize=6)
            a2.legend(fontsize=6, loc='lower right')
        a.legend(fontsize=6, loc='upper right'); a.set_title('prędkość na właściwych osiach', fontsize=8)
        for x in f.axes:
            x.tick_params(labelsize=6.5)
        f.subplots_adjust(left=.03, right=.98, top=.93, bottom=.08, hspace=.5, wspace=.3)
        self.etykieta_sl.setText(f'korekta t = {int(self._kr_k[min(ik, len(self._kr_k) - 1)])}, uczenie k = {int(kr[iu])}')
        f.suptitle('Dwie osie czasu: krok korekty (wagi stoją) i krok uczenia (zmieniają się wagi, AdamW, BN). '
                   'Ruch korekty nie jest prawem uczenia; porównanie to pytanie badawcze.', fontsize=8.5, color='white')
        self.can_u.draw_idle()

    def rysuj_dynamike(self):
        f = self.fig_d; f.clear()
        r = self.dyn
        if r is None:
            self.can_d.draw_idle(); return
        ax = f.subplots(2, 2); f.subplots_adjust(left=.07, right=.98, top=.9, bottom=.1, hspace=.5, wspace=.25)
        hs = r['hs']; modele = list(r['tabela'])
        a = ax[0, 0]
        szer = .8 / len(hs)
        for j, h in enumerate(hs):
            for i, m in enumerate(modele):
                x = r['tabela'][m].get(h, {})
                if 'rel' in x:
                    a.bar(i + j * szer, min(x['rel'], 3), szer, color=['#1f77b4', '#ff7f0e', '#2ca02c'][j % 3],
                          label=f'h = {h}' if i == 0 else None)
                else:
                    a.text(i + j * szer, .05, 'brak\ndanych', fontsize=5, ha='center', rotation=90)
        a.axhline(1, color='white', ls=':', lw=.7)
        a.set_xticks(np.arange(len(modele)) + .4 - szer / 2); a.set_xticklabels(modele, fontsize=6.5, rotation=15)
        a.set_title('błąd prognozy / błąd persystencji (część testowa, obcięte do 3)', fontsize=8); a.legend(fontsize=6)
        a = ax[0, 1]
        if r.get('detektor'):
            for j, h in enumerate(hs):
                for i, m in enumerate(modele):
                    x = r['detektor'][m].get(h)
                    if x:
                        a.bar(i + j * szer, x['strata'], szer, color=['#1f77b4', '#ff7f0e', '#2ca02c'][j % 3])
            a.set_xticks(np.arange(len(modele)) + .4 - szer / 2); a.set_xticklabels(modele, fontsize=6.5, rotation=15)
            a.set_title('błąd wyniku detektora: |Δ strata| prognozy (końcówka zamrożona)', fontsize=8)
        else:
            a.axis('off')
            a.text(.02, .6, 'Błąd wyniku detektora niedostępny na osi uczenia:\nzmieniają się wagi i residual, więc mapy nie\n'
                   'da się policzyć z samego latentu.', fontsize=8, color='#e0e0e0')
        a = ax[1, 0]
        if r.get('wejscie'):
            k = 0
            for nazwa, tab in r['wejscie'].items():
                if 'brak' in tab:
                    continue
                for n, x in tab.items():
                    if 'rel' in x:
                        a.barh(k, min(x['rel'], 2), color='#30ff70' if 'wejście' in n else '#888888')
                        a.text(min(x['rel'], 2), k, f' {x["rel"]:.3f}', va='center', fontsize=6)
                    a.text(-.02, k, f'{nazwa[:18]}: {n}', ha='right', va='center', fontsize=6)
                    k += 1
            a.set_yticks([]); a.set_xlim(0, 2.3); a.set_position([.2, .08, .28, .33])
            a.set_title('h = 1: GLE bez wejścia i z wejściem optymalizatora (rel)', fontsize=8)
        else:
            a.axis('off'); a.text(.02, .6, 'Model z wejściem: tylko tor korekty zapisany co 1 krok\n(gradient i momenty Adama '
                                           'znane w chwili t).', fontsize=8, color='#e0e0e0')
        a = ax[1, 1]
        b = r.get('blad')
        if b:
            a.bar(range(len(b['acf'])), b['acf'], color='#9467bd')
            a.axhline(b['pasmo'], color='white', ls=':', lw=.7); a.axhline(-b['pasmo'], color='white', ls=':', lw=.7)
            a.set_title(f'błąd modelu GLE (1 krok, test, 1. składowa): autokorelacja, kurtoza {b["kurtoza"]:.2f}', fontsize=8)
        else:
            a.axis('off')
        for x in f.axes:
            x.tick_params(labelsize=6.5)
        par = r['parametry']
        f.suptitle(f'{r["os"]}; {r["opis"]}\nLangevin β={par["Langevin"].get("beta", float("nan")) if isinstance(par["Langevin"], dict) else "?"}'
                   f'; reszta = błąd modelu (nie biały szum ani temperatura bez dalszych testów)', fontsize=8, color='white')
        self.can_d.draw_idle()

    def rysuj_galezie(self):
        f = self.fig_g; f.clear()
        w = self.gal
        if w is None:
            self.can_g.draw_idle(); return
        N = w['sondy']
        if w['typ'] == 'wzrost':
            ax = f.subplots(1, 2); f.subplots_adjust(left=.07, right=.98, top=.85, bottom=.12, wspace=.25)
            for przysz, wr in w['wyniki'].items():
                for n, v in wr.items():
                    hs = sorted(v['d'], key=int)
                    st = '-' if przysz == 'prawdziwa' else '--'
                    kol = '#bbbbbb' if n == 'C' else None
                    ax[0].plot(hs, [v['d'][h] for h in hs], st, color=kol, marker='o', ms=3,
                               label=f'{"szum (C)" if n == "C" else n} {przysz}')
                    if n != 'C':
                        ax[1].plot(hs, [v['wzrost'][h] if v['wzrost'][h] is not None else np.nan for h in hs], st, marker='o', ms=3,
                                   label=f'{n} {przysz}')
            ax[0].set_yscale('log'); ax[0].set_xlabel('kroki od zaburzenia', fontsize=7)
            ax[0].set_title('d(h): odległość latentu sond od kontroli', fontsize=8); ax[0].legend(fontsize=6)
            ax[1].set_yscale('log'); ax[1].axhline(1, color='white', ls=':', lw=.7); ax[1].set_xlabel('kroki od zaburzenia', fontsize=7)
            ax[1].set_title('wzrost d(h)/d(1)', fontsize=8); ax[1].legend(fontsize=6)
            f.suptitle('Wzrost odchylenia reprezentacji w tokenach sond (nie wykładnik Lapunowa; krótkie okno nie dowodzi chaosu)',
                       fontsize=8.5, color='white')
        else:
            zb = w['podsumowanie']; hs = sorted(zb, key=int)
            ax = f.subplots(2, 2); f.subplots_adjust(left=.07, right=.98, top=.9, bottom=.08, hspace=.45, wspace=.25)
            a = ax[0, 0]
            for i, n in enumerate(N):
                y = [zb[h]['odpowiedz_dz']['srednia'][i] for h in hs]
                lo = [zb[h]['odpowiedz_dz']['min'][i] for h in hs]; hi = [zb[h]['odpowiedz_dz']['max'][i] for h in hs]
                wyr = i == len(N) - 1 and 'wybrany' in n
                a.plot(hs, y, color='#30ff70' if wyr else '#888888', lw=2 if wyr else .7, label=n if wyr else None)
                if wyr:
                    a.fill_between(hs, lo, hi, color='#30ff70', alpha=.2)
            sz = [np.mean(zb[h]['szum_dz']['srednia']) for h in hs]
            a.plot(hs, sz, color='#d62728', ls='--', label='szum identycznych gałęzi (średnio)')
            a.set_yscale('symlog', linthresh=1e-4); a.legend(fontsize=6)
            a.set_title('||Δz|| sond (szare: 12 klatek; pasmo: min-max po przyszłościach)', fontsize=8)
            a = ax[0, 1]
            for i, n in enumerate(N):
                y = [zb[h]['odpowiedz_dstrata']['srednia'][i] for h in hs]
                wyr = i == len(N) - 1 and 'wybrany' in n
                a.plot(hs, y, color='#30ff70' if wyr else '#888888', lw=2 if wyr else .7)
            a.plot(hs, [np.mean(np.abs(zb[h]['szum_dstrata']['srednia'])) for h in hs], color='#d62728', ls='--', label='|szum|')
            a.axhline(0, color='white', lw=.5, ls=':'); a.set_yscale('symlog', linthresh=1e-5); a.legend(fontsize=6)
            a.set_title('zmiana straty sond (gładkie wyjście; ujemna = pomaga)', fontsize=8)
            a = ax[1, 0]
            for k, c in (('liniowosc', '#ffd030'), ('superpozycja', '#e377c2')):
                if all(k in zb[h] for h in hs):
                    a.plot(hs, [zb[h][k]['srednia'] for h in hs], 'o-', color=c, label=k)
            a.axhline(.1, color='white', ls=':', lw=.6); a.legend(fontsize=6)
            a.set_title('odchylenie od liniowości ||R(2ε) − 2R(ε)||/||2R(ε)|| i superpozycji', fontsize=8)
            a = ax[1, 1]
            y = [zb[h]['odpowiedz_diou']['srednia'][-1] for h in hs]
            a.plot(hs, y, 'o-', color='#30ff70'); a.axhline(0, color='white', lw=.5, ls=':')
            a.set_title('zmiana IoU najpewniejszej ramki wybranego obrazu (wynik zadania, nie do identyfikacji)', fontsize=8)
            for x in ax.ravel():
                x.set_xlabel('kroki od interwencji (1 = krok z interwencją)', fontsize=7)
            k = w['konfiguracja']
            f.suptitle(f'Odpowiedź na interwencję {k["interwencja"].get("rodzaj")} w kroku {k["krok_startu"]}, '
                       f'{len(k["przyszlosci"])} przyszłe sekwencje, deterministycznie {k.get("deterministyczne")}',
                       fontsize=8.5, color='white')
        for x in f.axes:
            x.tick_params(labelsize=6.5)
        self.can_g.draw_idle()

    def rysuj_obserwator(self):
        f = self.fig_k; f.clear()
        r = self.kal
        if r is None or 'brak' in r:
            self.can_k.draw_idle(); return
        t = r['kroki_testu']
        ax = f.subplots(1, 2); f.subplots_adjust(left=.07, right=.98, top=.85, bottom=.12, wspace=.25)
        a = ax[0]
        for n, c in (('otwarta_petla', '#888888'), ('prior', '#ffd030'), ('posterior', '#30ff70'), ('sam_pomiar', '#e377c2')):
            a.plot(t, r['blad_modow'][n], color=c, lw=.9, label=n.replace('_', ' '))
        a.set_yscale('log'); a.legend(fontsize=6); a.set_title(f'błąd w {r["r"]} modach', fontsize=8)
        a = ax[1]
        for n, c in (('persystencja', '#bbbbbb'), ('otwarta_petla', '#888888'), ('prior', '#ffd030'), ('posterior', '#30ff70'),
                     ('rzut_r_modow', '#17becf')):
            a.plot(t, r['blad_latentu'][n], color=c, lw=.9, label=n.replace('_', ' '))
        a.legend(fontsize=6); a.set_title('błąd latentu (przyrostowo od ostatniego pełnego pomiaru)', fontsize=8)
        for x in ax:
            x.set_xlabel(f'krok sekwencji ({r["os"]})', fontsize=7); x.tick_params(labelsize=6.5)
        pk = r['pokrycie_90']
        f.suptitle(f'Obserwator: {len(r["tokeny_obserwowane"])}/{r["n_tok"]} tokenów, pomiar co {self.spin_kco.value()}, koszt '
                   f'{100 * r["koszt"]["lacznie"]:.0f} %; obserwowalność {r["obserwowalnosc"]["rzad"]}/{r["obserwowalnosc"]["wymiar_stanu"]}; '
                   f'pokrycie 90 %: prior {pk["prior"]:.2f}, posterior {pk["posterior"] if pk["posterior"] is None else round(pk["posterior"], 2)}; '
                   f'{r["r"]} mody = {100 * r["wariancja_r_modow"]:.0f} % wariancji części do dopasowania', fontsize=8, color='white')
        self.can_k.draw_idle()
