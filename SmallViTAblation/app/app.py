"""Edytor latentu SD — GUI (PyQt6).

Model warunkuje UNet pełnym latentem 600 tokenów x 768 kanałów. Suwaki edytują
jeden wektor 768 wartości; tryb decyduje, jak wektor nakłada się na latent bazowy:

- "Offset do wszystkich tokenów": latent = baza + wektor (najbliższe idei "768 wartości"),
- "Offset do jednego tokena":     latent[token] = baza[token] + wektor,
- "Ustaw token na wektor":        latent[token] = wektor.

Przycisk GENERUJ uruchamia DDIM z checkpointu i pokazuje obraz 384x576.
Uruchomienie: python app.py
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                             QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QScrollArea, QSpinBox,
                             QSplitter, QTabWidget, QVBoxLayout, QWidget)

import silnik
from styl import STYL, rola
from widzety_edytora import SUWAK_MAX, ListaKanalow, ObrazSiatka, WatekGeneracji, WatekModelu, WatekViT
from wspolne import KOLUMNY as KOLUMNY_SIATKI   # siatka patchy 20x30

NA_STRONE = 64
STRON = silnik.KANALY // NA_STRONE  # 12

TRYBY = ['Offset do wszystkich tokenów (baza + wektor)',
         'Offset do jednego tokena (token + wektor)',
         'Ustaw token na wektor (token = wektor)']

# Jednostki tylko zmieniają SPOSÓB WYŚWIETLANIA pola liczbowego; wektor w pamięci
# jest zawsze absolutny (takiego oczekuje model). % liczony od zakresu kanału
# [lo_c, hi_c] ze statystyk latentu bazowego; suwak w trybie % to dokładnie 0-100%.
JEDNOSTKI = ['Absolutne', '% zakresu kanału (0–100%)',
             '% zmiany kanału: −100% (do min) .. 0 .. +100% (do max)']

# Zakres suwaków: ze statystyk kanału (min..max bazy, poszerzone o 3 sigma)
# albo bipolarny -M..0..+M (M = max(|lo|, |hi|) kanału) — zero dokładnie na środku,
# wygodne dla offsetów dodatnich/ujemnych — albo limity makrowektora [min_c, max_c].
ZAKRESY = ['Statystyki kanału (min..max bazy)', 'Bipolarny (−M .. 0 .. +M)',
           'Limity kanału (min..max z grupy Limity)']

# Makrowektor = jeden z 768 kanałów; każdy ma własny przedział [min_c, max_c].
LIMITY = ['Matematyczne z wag ViT (checkpoint)',
          'Po T krokach uczenia AdamW (twarda granica)',
          'Statystyki latentu bazowego (min..max)',
          'Z pliku (NPZ / CSV / JSON)',
          'Ręczne (wspólne min / max)']



class Okno(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Edytor latentu SD — 600 tokenów x 768 kanałów')
        self.resize(1420, 920)

        self.trained = None
        self.baza, self.zrodlo_bazy = silnik.domyslna_baza()
        self.wektor = np.zeros(silnik.KANALY, np.float32)
        self.lo = self.hi = None
        self.obraz = None
        self.watek_generacji = None
        self.watek_vit = None
        # Generowanie na bieżąco: po każdej zmianie latentu czekamy, aż suwak przestanie się ruszać,
        # a zmiana w trakcie generacji jest liczona zaraz po niej (zawsze ostatni stan).
        self.timer_na_biezaco = QTimer(self)
        self.timer_na_biezaco.setSingleShot(True)
        self.timer_na_biezaco.timeout.connect(lambda: self._generuj(na_biezaco=True))
        self.oczekuje_na_biezaco = False
        self.start_rgb = None          # zdjęcie startowe img2img [3,384,576] w [0,1]
        self.start_nazwa = None
        self.sciezka_bazy = None
        self.narzedzie = silnik.kanaly_narzedzia()          # 128 kanałów J128, wg delta_rms malejąco
        self.ckpt_limitow = silnik.VIT_CHECKPOINT
        self.lim_lo, self.lim_hi = silnik.granice_z_wag(self.ckpt_limitow)
        self.zrodlo_limitow = f'matematyczne z wag {Path(self.ckpt_limitow).name}'

        self._buduj_ui()
        self.combo_jednostki.blockSignals(True)
        self.combo_jednostki.setCurrentIndex(2)   # domyślnie zmiana −100..+100% limitu
        self.combo_jednostki.blockSignals(False)
        self.combo_zakres.setEnabled(False)
        self._opisz_limity()
        self._przelicz_zakresy()
        self._odswiez_strone()
        self._aktualizuj_statystyki()

        self.statusBar().showMessage('Ładowanie modelu SD (weryfikacja wag + checkpoint, może potrwać ~1 min)...')
        self.watek_modelu = WatekModelu(self)
        self.watek_modelu.gotowy.connect(self._model_gotowy)
        self.watek_modelu.blad.connect(self._blad_krytyczny)
        self.watek_modelu.start()

    # ------------------------------------------------------------------ UI

    def _buduj_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._panel_edycji())
        splitter.addWidget(self._panel_obrazu())
        splitter.setSizes([820, 600])
        from zakladka_detekcji import ZakladkaDetekcji
        self.zakladki_glowne = QTabWidget()
        self.zakladki_glowne.addTab(splitter, 'Edytor SD')
        self.zakladka_detekcji = ZakladkaDetekcji(self)
        self.zakladki_glowne.addTab(self.zakladka_detekcji, 'Detekcja ViT i auto-szukanie latentu')
        try:
            from zakladka_filmu import ZakladkaFilmu
            self.zakladka_filmu = ZakladkaFilmu(self)
            self.zakladki_glowne.addTab(self.zakladka_filmu, 'Film: sekwencja klatek, cząsteczki 3D i makrowektory')
        except Exception as e:                       # brak OpenGL / cv2: reszta aplikacji działa
            import traceback; traceback.print_exc()
            self.zakladki_glowne.addTab(QLabel(f'Zakładka filmu niedostępna: {e}'), 'Film')
        self.setCentralWidget(self.zakladki_glowne)

    def _panel_edycji(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)

        grupa_baza = QGroupBox('Latent bazowy 600x768')
        grupa_baza.setToolTip('Punkt odniesienia edycji: pełny latent 600 tokenów x 768 kanałów.\n'
                              'Wektor z suwaków jest nakładany na tę bazę zgodnie z wybranym trybem.\n'
                              'Domyślnie: naturalny latent "before" z LATENT_PELNY.npz.')
        b = QVBoxLayout(grupa_baza)
        self.etykieta_bazy = QLabel(f'Źródło: {self.zrodlo_bazy}')
        self.etykieta_bazy.setWordWrap(True)
        b.addWidget(self.etykieta_bazy)
        wiersz = QHBoxLayout()
        self.btn_obraz = QPushButton('Z obrazu (ViT)...')
        self.btn_obraz.setToolTip('Wczytuje dowolny obraz (JPG/PNG...), skaluje go do 480x320\n'
                                  'i liczy naturalny latent zamrożonym ViT (MODEL_VIT_K1000.pt).\n'
                                  'ViT jest ładowany tylko na czas liczenia i zwalniany z VRAM.')
        self.btn_obraz.clicked.connect(self._wczytaj_baze_z_obrazu)
        self.btn_npz = QPushButton('Z pliku NPZ...')
        self.btn_npz.setToolTip('Wczytuje latent 600x768 z pliku NPZ.\n'
                                'Rozpoznawane klucze: latent / after / before, a gdy ich brak —\n'
                                'pierwsza tablica o kształcie 600x768 (np. latent zapisany w tej aplikacji).')
        self.btn_npz.clicked.connect(self._wczytaj_baze_z_npz)
        self.btn_baza_reset = QPushButton('Przywróć domyślną')
        self.btn_baza_reset.setToolTip('Przywraca domyślny latent bazowy: naturalny latent "before"\n'
                                       'z outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz.')
        self.btn_baza_reset.clicked.connect(self._przywroc_baze)
        for w in (self.btn_obraz, self.btn_npz, self.btn_baza_reset):
            wiersz.addWidget(w)
        wiersz.addStretch(1)
        b.addLayout(wiersz)
        lay.addWidget(grupa_baza)
        lay.addWidget(self._grupa_limitow())

        grupa_wektor = QGroupBox('Wektor 768 wartości (suwaki)')
        g = QVBoxLayout(grupa_wektor)
        wiersz_tryb = QHBoxLayout()
        etykieta_tryb = QLabel('Tryb:')
        etykieta_tryb.setToolTip('Jak wektor 768 wartości nakłada się na latent bazowy.')
        wiersz_tryb.addWidget(etykieta_tryb)
        self.combo_tryb = QComboBox()
        self.combo_tryb.addItems(TRYBY)
        self.combo_tryb.setToolTip('Sposób nałożenia wektora na bazę:\n'
                                   '- Offset do wszystkich tokenów: latent = baza + wektor\n'
                                   '  (te same 768 wartości dodane do każdego z 600 tokenów),\n'
                                   '- Offset do jednego tokena: latent[token] = baza[token] + wektor,\n'
                                   '- Ustaw token na wektor: latent[token] = wektor (nadpisuje token).\n'
                                   'Tokeny odpowiadają siatce 20x30 patchy obrazu (numeracja wierszami).')
        self.combo_tryb.currentIndexChanged.connect(self._zmiana_trybu)
        wiersz_tryb.addWidget(self.combo_tryb, 1)
        etykieta_token = QLabel('Token:')
        etykieta_token.setToolTip('Indeks tokena używanego w trybach "jeden token".')
        wiersz_tryb.addWidget(etykieta_token)
        self.spin_token = QSpinBox()
        self.spin_token.setRange(0, silnik.TOKENY - 1)
        self.spin_token.setEnabled(False)
        self.spin_token.setToolTip('Indeks tokena 0-599, aktywny w trybach "Offset do jednego tokena"\n'
                                   'i "Ustaw token na wektor". Token = jeden patch z siatki 20x30\n'
                                   '(token 0 = lewy górny róg obrazu, numeracja wierszami).')
        self.spin_token.valueChanged.connect(self._aktualizuj_statystyki)
        self.spin_token.valueChanged.connect(self._pokaz_token)
        wiersz_tryb.addWidget(self.spin_token)
        self.btn_zeruj = QPushButton('Zeruj wektor')
        self.btn_zeruj.setToolTip('Ustawia wszystkie 768 wartości wektora na 0.\n'
                                  'Latent efektywny staje się równy bazie.')
        self.btn_zeruj.clicked.connect(self._zeruj_wektor)
        wiersz_tryb.addWidget(self.btn_zeruj)
        g.addLayout(wiersz_tryb)

        wiersz_jednostki = QHBoxLayout()
        etykieta_jednostki = QLabel('Jednostki pól liczbowych:')
        etykieta_jednostki.setToolTip('Tylko sposób wyświetlania wartości przy suwakach.')
        wiersz_jednostki.addWidget(etykieta_jednostki)
        self.combo_jednostki = QComboBox()
        self.combo_jednostki.addItems(JEDNOSTKI)
        self.combo_jednostki.setToolTip('Jednostki pól liczbowych (NIE zmieniają wektora w pamięci):\n'
                                        '- Absolutne: surowe wartości latentu — takich oczekuje model\n'
                                        '  (naturalne wartości są zawsze >= -0,17; górna granica nie istnieje,\n'
                                        '  w latentcie bazowym max ok. 4,0),\n'
                                        '- % zakresu kanału: 0% = minimum kanału w bazie, 100% = maksimum;\n'
                                        '  można wpisać wartości poza 0-100%. Suwak zawsze odpowiada 0-100%.\n'
                                        'Generacja zawsze używa wartości absolutnych.')
        self.combo_jednostki.currentIndexChanged.connect(self._zmiana_zakresu)
        wiersz_jednostki.addWidget(self.combo_jednostki, 1)
        etykieta_zakres = QLabel('Zakres suwaków:')
        etykieta_zakres.setToolTip('Skala, po której chodzi suwak.')
        wiersz_jednostki.addWidget(etykieta_zakres)
        self.combo_zakres = QComboBox()
        self.combo_zakres.addItems(ZAKRESY)
        self.combo_zakres.setToolTip('Zakres wartości pokrywany przez suwak:\n'
                                     '- Statystyki kanału: [min - 3 sigma, max + 3 sigma] z latentu bazowego;\n'
                                     '  dla kanałów o małej zmienności zero może leżeć poza suwakiem,\n'
                                     '- Bipolarny: [-M, +M], gdzie M = max(|min|, |max|) kanału;\n'
                                     '  zero jest DOKŁADNIE na środku suwaka — w lewo wartości ujemne,\n'
                                     '  w prawo dodatnie. Wygodne dla offsetów dodatnich/ujemnych.\n'
                                     'W trybie % bipolarny 50% = 0. Zmiana zakresu nie zmienia wektora,\n'
                                     'a jedynie pozycje suwaków.')
        self.combo_zakres.currentIndexChanged.connect(self._zmiana_zakresu)
        wiersz_jednostki.addWidget(self.combo_zakres, 1)
        g.addLayout(wiersz_jednostki)

        self.zakladki = QTabWidget()
        self.zakladki.setToolTip('Obie zakładki edytują ten sam wektor 768 wartości.')
        strona_wszystkie = QWidget()
        lay_wszystkie = QVBoxLayout(strona_wszystkie)
        wiersz_stron = QHBoxLayout()
        wiersz_stron.addWidget(QLabel('Kanały:'))
        self.combo_strona = QComboBox()
        for s in range(STRON):
            self.combo_strona.addItem(f'{s*NA_STRONE}–{s*NA_STRONE+NA_STRONE-1}')
        self.combo_strona.setToolTip('Wybór strony 64 kanałów. Wszystkich kanałów jest 768 (12 stron).\n'
                                     'Najedź na etykietę K### przy suwaku, by zobaczyć zakres kanału.')
        self.combo_strona.currentIndexChanged.connect(self._odswiez_strone)
        wiersz_stron.addWidget(self.combo_strona, 1)
        btn_popr = QPushButton('<')
        btn_popr.setToolTip('Poprzednia strona kanałów.')
        btn_popr.clicked.connect(lambda: self.combo_strona.setCurrentIndex(max(0, self.combo_strona.currentIndex()-1)))
        btn_nast = QPushButton('>')
        btn_nast.setToolTip('Następna strona kanałów.')
        btn_nast.clicked.connect(lambda: self.combo_strona.setCurrentIndex(min(STRON-1, self.combo_strona.currentIndex()+1)))
        wiersz_stron.addWidget(btn_popr)
        wiersz_stron.addWidget(btn_nast)
        etykieta_skok = QLabel('Skok do:')
        etykieta_skok.setToolTip('Szybkie przejście do wybranego kanału.')
        wiersz_stron.addWidget(etykieta_skok)
        self.spin_skok = QSpinBox()
        self.spin_skok.setRange(0, silnik.KANALY - 1)
        self.spin_skok.setToolTip('Wpisz numer kanału (0-767), a lista przewinie się na właściwą stronę.')
        self.spin_skok.valueChanged.connect(lambda c: self.combo_strona.setCurrentIndex(c // NA_STRONE))
        wiersz_stron.addWidget(self.spin_skok)
        lay_wszystkie.addLayout(wiersz_stron)
        self.lista_wszystkie = ListaKanalow(self, NA_STRONE)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.lista_wszystkie)
        lay_wszystkie.addWidget(scroll, 1)
        self.zakladki.addTab(strona_wszystkie, 'Wszystkie 768 kanałów')

        strona_narzedzie = QWidget()
        lay_narz = QVBoxLayout(strona_narzedzie)
        opis = QLabel(f'{len(self.narzedzie)} kanałów tożsamości narzędzia (J128), posortowane wg siły zmiany '
                      'w sztucznej korekcie (delta RMS). Pozostałe 640 kanałów nie są tu pokazywane.')
        opis.setWordWrap(True)
        lay_narz.addWidget(opis)
        wiersz_narz = QHBoxLayout()
        btn_zeruj_narz = QPushButton('Zeruj kanały narzędzia')
        btn_zeruj_narz.setToolTip('Ustawia 0 tylko na 128 kanałach narzędzia; reszta wektora bez zmian.')
        btn_zeruj_narz.clicked.connect(self._zeruj_narzedzie)
        wiersz_narz.addWidget(btn_zeruj_narz)
        wiersz_narz.addWidget(QLabel('Wszystkie 128 o:'))
        self.spin_narz_procent = QDoubleSpinBox()
        self.spin_narz_procent.setRange(-100, 100)
        self.spin_narz_procent.setDecimals(1)
        self.spin_narz_procent.setSuffix(' %')
        self.spin_narz_procent.setValue(10)
        self.spin_narz_procent.setToolTip('Wspólna zmiana dla wszystkich 128 kanałów narzędzia,\n'
                                          'w procentach szerokości limitu każdego kanału (max − min).')
        wiersz_narz.addWidget(self.spin_narz_procent)
        btn_narz_procent = QPushButton('Ustaw')
        btn_narz_procent.setToolTip('Przesunięcie każdego z 128 kanałów = % × (max_c − min_c) jego limitu.\n'
                                    'Działa jak offset w trybach „Offset do …”.')
        btn_narz_procent.clicked.connect(self._narzedzie_procent)
        wiersz_narz.addWidget(btn_narz_procent)
        wiersz_narz.addStretch(1)
        lay_narz.addLayout(wiersz_narz)
        self.lista_narzedzie = ListaKanalow(self, len(self.narzedzie))
        self.lista_narzedzie.kanaly = [w['kanal'] for w in self.narzedzie]
        scroll_n = QScrollArea()
        scroll_n.setWidgetResizable(True)
        scroll_n.setWidget(self.lista_narzedzie)
        lay_narz.addWidget(scroll_n, 1)
        self.zakladki.addTab(strona_narzedzie, f'Narzędzie: {len(self.narzedzie)} kanałów')
        self.listy = (self.lista_wszystkie, self.lista_narzedzie)

        g.addWidget(self.zakladki, 1)
        lay.addWidget(grupa_wektor, 1)
        return panel

    def _grupa_limitow(self):
        grupa = QGroupBox('Limity makrowektorów: min / max każdego z 768 kanałów')
        grupa.setToolTip('Każdy kanał (makrowektor) ma własny przedział [min, max].\n'
                         'Limity wyznaczają zakres suwaków (Zakres suwaków = Limity kanału),\n'
                         'jednostkę „Zmiana o % limitu” i opcjonalne przycinanie latentu.\n'
                         'Wyprowadzenie granic: experiments/vit_granice_latentu/README.md.')
        v = QVBoxLayout(grupa)
        w1 = QHBoxLayout()
        w1.addWidget(QLabel('Źródło:'))
        self.combo_limity = QComboBox()
        self.combo_limity.addItems(LIMITY)
        self.combo_limity.setToolTip(
            '- Matematyczne z wag ViT: dokładny przedział dla ustalonych wag (min −0,17, max 7–10 dla k1000).\n'
            '- Po T krokach uczenia AdamW: twarda granica dla dowolnych danych po T krokach przepisu\n'
            '  maly_vit_od_zera (wspólna dla kanałów; bez limitu kroków granica nie istnieje).\n'
            '- Statystyki latentu bazowego: min..max obecnej bazy.\n'
            '- Z pliku: NPZ (h_min/h_max, min/max, lo/hi), CSV (kanal,min,max) albo JSON.\n'
            '- Ręczne: wspólne min i max dla wszystkich kanałów z pól poniżej.')
        self.combo_limity.activated.connect(self._zmiana_zrodla_limitow)
        w1.addWidget(self.combo_limity, 1)
        w1.addWidget(QLabel('T kroków:'))
        self.spin_T = QSpinBox()
        self.spin_T.setRange(1, 1_000_000)
        self.spin_T.setValue(1000)
        self.spin_T.setToolTip('Liczba kroków uczenia dla źródła „Po T krokach uczenia AdamW”.')
        self.spin_T.editingFinished.connect(lambda: self.combo_limity.currentIndex() == 1 and self._zmiana_zrodla_limitow(1))
        w1.addWidget(self.spin_T)
        btn_ckpt = QPushButton('Checkpoint ViT...')
        btn_ckpt.setToolTip('Wybór checkpointu ViT dla źródła „Matematyczne z wag ViT”\n'
                            '(np. wagi z innego kroku uczenia, k0000..k1000).')
        btn_ckpt.clicked.connect(self._wybierz_checkpoint_limitow)
        w1.addWidget(btn_ckpt)
        v.addLayout(w1)

        w2 = QHBoxLayout()
        w2.addWidget(QLabel('Ręczne: min'))
        self.spin_lim_min = QDoubleSpinBox()
        self.spin_lim_min.setRange(-1e6, 1e6); self.spin_lim_min.setDecimals(4)
        self.spin_lim_min.setValue(silnik.GELU_MIN)
        w2.addWidget(self.spin_lim_min)
        w2.addWidget(QLabel('max'))
        self.spin_lim_max = QDoubleSpinBox()
        self.spin_lim_max.setRange(-1e6, 1e6); self.spin_lim_max.setDecimals(4)
        self.spin_lim_max.setValue(4.43)
        self.spin_lim_max.setToolTip('Domyślnie 4,43: największa wartość latentu na 400 prawdziwych zdjęciach (k1000).')
        w2.addWidget(self.spin_lim_max)
        btn_wsp = QPushButton('Zastosuj wspólne')
        btn_wsp.setToolTip('Ustawia te same min i max dla wszystkich 768 kanałów.')
        btn_wsp.clicked.connect(lambda: (self.combo_limity.setCurrentIndex(4), self._zmiana_zrodla_limitow(4)))
        w2.addWidget(btn_wsp)
        w2.addSpacing(16)
        w2.addWidget(QLabel('Kanał'))
        self.spin_lim_kanal = QSpinBox()
        self.spin_lim_kanal.setRange(0, silnik.KANALY - 1)
        self.spin_lim_kanal.valueChanged.connect(self._pokaz_limit_kanalu)
        w2.addWidget(self.spin_lim_kanal)
        self.spin_k_min = QDoubleSpinBox(); self.spin_k_min.setRange(-1e6, 1e6); self.spin_k_min.setDecimals(4)
        self.spin_k_max = QDoubleSpinBox(); self.spin_k_max.setRange(-1e6, 1e6); self.spin_k_max.setDecimals(4)
        w2.addWidget(QLabel('min')); w2.addWidget(self.spin_k_min)
        w2.addWidget(QLabel('max')); w2.addWidget(self.spin_k_max)
        btn_k = QPushButton('Ustaw dla kanału')
        btn_k.setToolTip('Nadpisuje limit jednego kanału; pozostałe kanały bez zmian.')
        btn_k.clicked.connect(self._ustaw_limit_kanalu)
        w2.addWidget(btn_k)
        v.addLayout(w2)

        w3 = QHBoxLayout()
        self.chk_przytnij = QCheckBox('Przycinaj latent efektywny do limitów')
        self.chk_przytnij.setChecked(True)
        self.chk_przytnij.setToolTip('Każda wartość latentu efektywnego jest przycinana do [min_c, max_c]\n'
                                     'swojego kanału przed generowaniem i zapisem.')
        self.chk_przytnij.toggled.connect(self._aktualizuj_statystyki)
        w3.addWidget(self.chk_przytnij)
        btn_zapisz_lim = QPushButton('Zapisz limity')
        btn_zapisz_lim.setToolTip('Zapisuje limity 768 kanałów do outputs/latent_editor_sd_20261006/\n'
                                  'jako NPZ (h_min, h_max) i CSV (kanal,min,max).')
        btn_zapisz_lim.clicked.connect(self._zapisz_limity)
        w3.addWidget(btn_zapisz_lim)
        self.etykieta_limitow = QLabel('')
        self.etykieta_limitow.setWordWrap(True)
        w3.addWidget(self.etykieta_limitow, 1)
        v.addLayout(w3)
        return grupa

    def _panel_obrazu(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)

        self.etykieta_model = QLabel('Model: ładowanie...')
        self.etykieta_model.setWordWrap(True)
        self.etykieta_model.setToolTip('Wytrenowany Stable Diffusion: checkpoint krok 8500 (cross-attention:\n'
                                       'adapter 600x768 + wagi attn2), bazowy UNet ema_020000, VAE sd-vae-ft-mse.\n'
                                       'Przy starcie weryfikowane są hashe wag bazowych — stąd ~1 min ładowania.')
        lay.addWidget(self.etykieta_model)

        self.etykieta_obraz = ObrazSiatka()
        self.etykieta_obraz.tokenWybrany.connect(self._token_klikniety)
        lay.addWidget(self.etykieta_obraz, 1)
        wiersz_siatka = QHBoxLayout()
        self.chk_siatka = QCheckBox('Siatka patchy 20x30 (kliknij patch = wybór tokena)')
        self.chk_siatka.setChecked(True)
        self.chk_siatka.toggled.connect(self.etykieta_obraz.ustaw_siatke)
        wiersz_siatka.addWidget(self.chk_siatka)
        self.etykieta_tokena = QLabel('token 0 (wiersz 0, kolumna 0)')
        wiersz_siatka.addWidget(self.etykieta_tokena, 1)
        lay.addLayout(wiersz_siatka)

        grupa_start = QGroupBox('Start odszumiania')
        gs = QVBoxLayout(grupa_start)
        w1 = QHBoxLayout()
        self.combo_start = QComboBox()
        self.combo_start.addItems(['Czysty szum', 'Zaszumione zdjęcie (img2img)'])
        self.combo_start.setToolTip('Czysty szum: zwykła generacja od szumu Gaussa.\n'
                                    'Zaszumione zdjęcie: VAE koduje zdjęcie startowe, dostaje szum o wybranej sile,\n'
                                    'a ostatnie kroki DDIM odszumiają je sterowane latentem 600x768.')
        self.combo_start.currentIndexChanged.connect(lambda _: self._zaplanuj_na_biezaco())
        w1.addWidget(self.combo_start, 1)
        btn_start = QPushButton('Zdjęcie startowe...')
        btn_start.setToolTip('Dowolne zdjęcie; skalowane do 576x384 (INTER_AREA) jak w treningu SD.')
        btn_start.clicked.connect(self._wybierz_zdjecie_startowe)
        w1.addWidget(btn_start)
        btn_jak_baza = QPushButton('Jak baza')
        btn_jak_baza.setToolTip('Używa zdjęcia, z którego policzono latent bazowy (Z obrazu (ViT)...).')
        btn_jak_baza.clicked.connect(self._start_jak_baza)
        w1.addWidget(btn_jak_baza)
        gs.addLayout(w1)
        w2 = QHBoxLayout()
        w2.addWidget(QLabel('Siła szumu:'))
        self.spin_sila = QDoubleSpinBox()
        self.spin_sila.setRange(0.02, 1.0)
        self.spin_sila.setSingleStep(0.05)
        self.spin_sila.setDecimals(2)
        self.spin_sila.setValue(0.6)
        self.spin_sila.setToolTip('Ułamek kroków DDIM: 0,6 przy 50 krokach = szum do kroku czasowego ok. 600\n'
                                  'i 30 kroków odszumiania. Mało = blisko zdjęcia, 1,0 = jak czysty szum.')
        self.spin_sila.valueChanged.connect(lambda _: self._zaplanuj_na_biezaco())
        w2.addWidget(self.spin_sila)
        self.chk_podglad = QCheckBox('Podgląd kolejnych kroków co')
        self.chk_podglad.setToolTip('Pokazuje obraz w trakcie odszumiania (każdy podgląd to dodatkowy decode VAE).')
        w2.addWidget(self.chk_podglad)
        self.spin_podglad_co = QSpinBox()
        self.spin_podglad_co.setRange(1, 100)
        self.spin_podglad_co.setValue(2)
        w2.addWidget(self.spin_podglad_co)
        self.combo_podglad = QComboBox()
        self.combo_podglad.addItems(['bieżący stan (z szumem)', 'przewidywany obraz końcowy'])
        self.combo_podglad.setToolTip('Bieżący stan: zaszumiony obraz po danym kroku.\n'
                                      'Przewidywany: oszacowanie czystego obrazu z tego kroku (pred x0).')
        w2.addWidget(self.combo_podglad, 1)
        gs.addLayout(w2)
        self.etykieta_startu = QLabel('Zdjęcie startowe: brak')
        self.etykieta_startu.setWordWrap(True)
        gs.addWidget(self.etykieta_startu)
        lay.addWidget(grupa_start)

        wiersz_gen = QHBoxLayout()
        etykieta_kroki = QLabel('Kroki DDIM:')
        etykieta_kroki.setToolTip('Liczba kroków odszumiania.')
        wiersz_gen.addWidget(etykieta_kroki)
        self.spin_kroki = QSpinBox()
        self.spin_kroki.setRange(1, 200)
        self.spin_kroki.setValue(50)
        self.spin_kroki.setToolTip('Liczba kroków DDIM (1-200, domyślnie 50).\n'
                                   'Więcej kroków = wolniej, zwykle lepsza jakość.\n'
                                   'Scheduler: DDIM eta=0, 1000 kroków treningowych.')
        wiersz_gen.addWidget(self.spin_kroki)
        etykieta_ziarno = QLabel('Ziarno:')
        etykieta_ziarno.setToolTip('Ziarno szumu początkowego.')
        wiersz_gen.addWidget(etykieta_ziarno)
        self.spin_ziarno = QSpinBox()
        self.spin_ziarno.setRange(0, 2**31 - 1)
        self.spin_ziarno.setValue(20261006)
        self.spin_ziarno.setToolTip('Ziarno generatora szumu początkowego (Gaussa).\n'
                                    'To samo ziarno + ten sam latent = identyczny obraz.\n'
                                    'Zmień ziarno, by dostać inną realizację dla tego samego latentu.')
        wiersz_gen.addWidget(self.spin_ziarno)
        self.btn_generuj = rola(QPushButton('GENERUJ OBRAZ'), 'glowny')
        self.btn_generuj.setEnabled(False)
        self.btn_generuj.setToolTip('Uruchamia generację z aktualnego latentu efektywnego\n'
                                    '(baza + wektor zgodnie z trybem) w osobnym wątku.\n'
                                    'DDIM odszumia z czystego szumu; VAE dekoduje do RGB 384x576.\n'
                                    'Przycisk aktywny po załadowaniu modelu.')
        self.btn_generuj.clicked.connect(lambda: self._generuj())
        wiersz_gen.addWidget(self.btn_generuj, 1)
        lay.addLayout(wiersz_gen)

        wiersz_live = QHBoxLayout()
        self.chk_na_biezaco = QCheckBox('Generuj na bieżąco po zmianie')
        self.chk_na_biezaco.setToolTip('Po każdej zmianie suwaka, pola, limitu, trybu albo bazy obraz jest\n'
                                       'generowany automatycznie, gdy suwak przestanie się ruszać\n'
                                       '(opóźnienie poniżej). Zmiana w trakcie generacji jest liczona zaraz po niej.')
        self.chk_na_biezaco.toggled.connect(lambda on: on and self._zaplanuj_na_biezaco())
        wiersz_live.addWidget(self.chk_na_biezaco)
        wiersz_live.addWidget(QLabel('Kroki:'))
        self.spin_kroki_live = QSpinBox()
        self.spin_kroki_live.setRange(1, 200)
        self.spin_kroki_live.setValue(15)
        self.spin_kroki_live.setToolTip('Kroki DDIM dla generowania na bieżąco (mniej = szybciej).\n'
                                        'Przycisk GENERUJ OBRAZ używa pola „Kroki DDIM” powyżej.')
        wiersz_live.addWidget(self.spin_kroki_live)
        wiersz_live.addWidget(QLabel('Opóźnienie:'))
        self.spin_opoznienie = QSpinBox()
        self.spin_opoznienie.setRange(0, 5000)
        self.spin_opoznienie.setSingleStep(50)
        self.spin_opoznienie.setValue(300)
        self.spin_opoznienie.setSuffix(' ms')
        self.spin_opoznienie.setToolTip('Ile czekać od ostatniej zmiany, zanim ruszy generacja.')
        wiersz_live.addWidget(self.spin_opoznienie)
        wiersz_live.addStretch(1)
        lay.addLayout(wiersz_live)

        self.pasek = QProgressBar()
        self.pasek.setValue(0)
        self.pasek.setToolTip('Postęp odszumiania DDIM (krok / liczba kroków).')
        lay.addWidget(self.pasek)

        wiersz_zapis = QHBoxLayout()
        self.btn_zapisz_obraz = QPushButton('Zapisz obraz PNG')
        self.btn_zapisz_obraz.setEnabled(False)
        self.btn_zapisz_obraz.setToolTip('Zapisuje ostatnio wygenerowany obraz jako PNG\n'
                                         'do outputs/latent_editor_sd_20261006/ (nazwa z datą i godziną).')
        self.btn_zapisz_obraz.clicked.connect(self._zapisz_obraz)
        self.btn_zapisz_latent = QPushButton('Zapisz latent NPZ')
        self.btn_zapisz_latent.setToolTip('Zapisuje do NPZ w outputs/latent_editor_sd_20261006/:\n'
                                          'latent efektywny (klucz "latent"), bazę, wektor, tryb i token.\n'
                                          'Taki plik można wczytać jako bazę przyciskiem "Z pliku NPZ...".')
        self.btn_zapisz_latent.clicked.connect(self._zapisz_latent)
        wiersz_zapis.addWidget(self.btn_zapisz_obraz)
        wiersz_zapis.addWidget(self.btn_zapisz_latent)
        wiersz_zapis.addStretch(1)
        lay.addLayout(wiersz_zapis)

        self.etykieta_stats = QLabel('')
        lay.addWidget(self.etykieta_stats)
        lay.addStretch(1)
        return panel

    # ------------------------------------------------------------- edycja

    def _procent(self):
        return self.combo_jednostki.currentIndex() == 1

    def _zmiana_procentowa(self):
        return self.combo_jednostki.currentIndex() == 2

    def _szerokosc_limitu(self, c):
        return max(float(self.lim_hi[c]) - float(self.lim_lo[c]), 1e-9)

    def _przelicz_zakresy(self):
        """Aktywny zakres suwaków; wektor i wartości pozostają bez zmian.

        Jednostka „Zmiana o % limitu”: suwak to przesunięcie −W_c..+W_c, W_c = max_c − min_c.
        Inaczej wg combo_zakres: statystyki bazy, bipolarny albo limity kanału [min_c, max_c].
        """
        if self._zmiana_procentowa():
            w = (self.lim_hi - self.lim_lo).astype(np.float32)
            self.lo, self.hi = -w, w
            return
        if self.combo_zakres.currentIndex() == 2:
            self.lo, self.hi = self.lim_lo.copy(), self.lim_hi.copy()
            return
        lo, hi = silnik.zakresy_kanalow(self.baza)
        if self.combo_zakres.currentIndex() == 1:
            m = np.maximum(np.abs(lo), np.abs(hi))
            lo, hi = -m, m
        self.lo, self.hi = lo, hi

    def _zmiana_zakresu(self):
        # Przy % zmiany suwak to zawsze −100..+100% limitu, więc wybór zakresu suwaków nie ma znaczenia.
        self.combo_zakres.setEnabled(not self._zmiana_procentowa())
        self._przelicz_zakresy()
        self._odswiez_strone()

    def _na_wyswietlana(self, wartosc, c):
        if self._zmiana_procentowa():
            return float(wartosc) / self._szerokosc_limitu(c) * 100
        if self._procent():
            return (float(wartosc) - float(self.lo[c])) / (float(self.hi[c]) - float(self.lo[c])) * 100
        return float(wartosc)

    def _z_wyswietlanej(self, x, c):
        if self._zmiana_procentowa():
            return float(x) / 100 * self._szerokosc_limitu(c)
        if self._procent():
            return float(self.lo[c]) + float(x) / 100 * (float(self.hi[c]) - float(self.lo[c]))
        return float(x)

    def _opis_kanalu(self, lista, r, c):
        tekst = (f'kanał {c}\nlimit [{self.lim_lo[c]:.3f}, {self.lim_hi[c]:.3f}] ({self.zrodlo_limitow})\n'
                 f'zakres suwaka [{self.lo[c]:.3f}, {self.hi[c]:.3f}]')
        if lista is self.lista_narzedzie:
            n = self.narzedzie[r]
            tekst += (f'\nJ128 #{r + 1}: delta RMS {n["delta_rms"]:.2f}, średnia przed {n["before_mean"]:.3f}, '
                      f'po {n["after_mean"]:.3f}, maks. po {n["after_max"]:.2f}')
        return tekst

    def _odswiez_wiersz(self, lista, r):
        etykieta, suwak, spin = lista.wiersze[r]
        c = lista.kanaly[r]
        etykieta.setText(f'K{c:03d}' + (f'  #{r + 1}' if lista is self.lista_narzedzie else ''))
        etykieta.setToolTip(self._opis_kanalu(lista, r, c))
        suwak.blockSignals(True)
        suwak.setValue(self._pozycja_z_wartosci(self.wektor[c], c))
        suwak.blockSignals(False)
        spin.blockSignals(True)
        if self._zmiana_procentowa():
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(2)
            spin.setSuffix(' %')
        elif self._procent():
            spin.setRange(-999.9, 1999.9)
            spin.setDecimals(2)
            spin.setSuffix(' %')
        else:
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(4)
            spin.setSuffix('')
        spin.setValue(self._na_wyswietlana(self.wektor[c], c))
        spin.blockSignals(False)

    def _odswiez_strone(self):
        if self.lo is None:
            return
        if self._zmiana_procentowa():
            self._przelicz_zakresy()
        start = self.combo_strona.currentIndex() * NA_STRONE
        self.lista_wszystkie.kanaly = list(range(start, start + NA_STRONE))
        for lista in self.listy:
            for r in range(len(lista.wiersze)):
                self._odswiez_wiersz(lista, r)

    def _synchronizuj(self, c, pomin):
        """Odświeża wiersze pokazujące kanał c w drugiej liście (obie listy edytują ten sam wektor)."""
        for lista in self.listy:
            for r, k in enumerate(lista.kanaly):
                if k == c and (lista, r) != pomin:
                    self._odswiez_wiersz(lista, r)

    def _pozycja_z_wartosci(self, wartosc, c):
        lo, hi = float(self.lo[c]), float(self.hi[c])
        pos = round((float(wartosc) - lo) / max(hi - lo, 1e-12) * SUWAK_MAX)
        return max(0, min(SUWAK_MAX, pos))

    def _wartosc_z_pozycji(self, pos, c):
        return float(self.lo[c]) + (float(self.hi[c]) - float(self.lo[c])) * pos / SUWAK_MAX

    def _suwak_zmieniony(self, lista, r, pos):
        c = lista.kanaly[r]
        wartosc = self._wartosc_z_pozycji(pos, c)
        self.wektor[c] = wartosc
        spin = lista.wiersze[r][2]
        spin.blockSignals(True)
        spin.setValue(self._na_wyswietlana(wartosc, c))
        spin.blockSignals(False)
        self._synchronizuj(c, (lista, r))
        self._aktualizuj_statystyki()

    def _spin_zmieniony(self, lista, r, x):
        c = lista.kanaly[r]
        wartosc = self._z_wyswietlanej(x, c)
        self.wektor[c] = wartosc
        suwak = lista.wiersze[r][1]
        suwak.blockSignals(True)
        suwak.setValue(self._pozycja_z_wartosci(wartosc, c))
        suwak.blockSignals(False)
        self._synchronizuj(c, (lista, r))
        self._aktualizuj_statystyki()

    def _zmiana_trybu(self, indeks):
        self.spin_token.setEnabled(indeks in (1, 2))
        self._aktualizuj_statystyki()

    def _zeruj_wektor(self):
        self.wektor[:] = 0
        self._odswiez_strone()
        self._aktualizuj_statystyki()

    def _zeruj_narzedzie(self):
        self.wektor[self.lista_narzedzie.kanaly] = 0
        self._odswiez_strone()
        self._aktualizuj_statystyki()

    def _narzedzie_procent(self):
        """Przesunięcie każdego z 128 kanałów narzędzia o ten sam % szerokości jego limitu."""
        k = np.asarray(self.lista_narzedzie.kanaly)
        self.wektor[k] = self.spin_narz_procent.value() / 100 * (self.lim_hi[k] - self.lim_lo[k])
        self._odswiez_strone()
        self._aktualizuj_statystyki()

    # ------------------------------------------------------------ limity

    def _ustaw_limity(self, lo, hi, zrodlo):
        lo, hi = np.asarray(lo, np.float32).copy(), np.asarray(hi, np.float32).copy()
        if lo.shape != (silnik.KANALY,) or hi.shape != (silnik.KANALY,) or (hi <= lo).any():
            raise ValueError('Limity muszą mieć 768 wartości i max > min w każdym kanale')
        self.lim_lo, self.lim_hi, self.zrodlo_limitow = lo, hi, zrodlo
        self._opisz_limity()
        self._przelicz_zakresy()
        self._odswiez_strone()
        self._aktualizuj_statystyki()

    def _opisz_limity(self):
        self.etykieta_limitow.setText(
            f'{self.zrodlo_limitow}: min {self.lim_lo.min():.3f}..{self.lim_lo.max():.3f}, '
            f'max {self.lim_hi.min():.3f}..{self.lim_hi.max():.3f} (mediana {np.median(self.lim_hi):.3f})')
        self._pokaz_limit_kanalu(self.spin_lim_kanal.value())

    def _pokaz_limit_kanalu(self, c):
        self.spin_k_min.setValue(float(self.lim_lo[c]))
        self.spin_k_max.setValue(float(self.lim_hi[c]))

    def _zmiana_zrodla_limitow(self, indeks):
        try:
            if indeks == 0:
                lo, hi = silnik.granice_z_wag(self.ckpt_limitow)
                zrodlo = f'matematyczne z wag {Path(self.ckpt_limitow).name}'
            elif indeks == 1:
                lo, hi = silnik.granica_po_krokach(self.spin_T.value())
                zrodlo = f'AdamW po {self.spin_T.value()} krokach'
            elif indeks == 2:
                lo, hi = self.baza.min(axis=0), self.baza.max(axis=0)
                hi = np.where(hi - lo < 1e-3, lo + 1e-3, hi)
                zrodlo = f'statystyki bazy ({self.zrodlo_bazy})'
            elif indeks == 3:
                sciezka, _ = QFileDialog.getOpenFileName(self, 'Limity 768 kanałów', str(silnik.WYJSCIE),
                                                         'Limity (*.npz *.csv *.json)')
                if not sciezka:
                    return
                lo, hi = silnik.wczytaj_limity(sciezka)
                zrodlo = f'plik {Path(sciezka).name}'
            else:
                lo = np.full(silnik.KANALY, self.spin_lim_min.value(), np.float32)
                hi = np.full(silnik.KANALY, self.spin_lim_max.value(), np.float32)
                zrodlo = f'ręczne [{self.spin_lim_min.value():.3f}, {self.spin_lim_max.value():.3f}]'
            self._ustaw_limity(lo, hi, zrodlo)
        except Exception:
            self._blad_niekrytyczny(traceback.format_exc())

    def _wybierz_checkpoint_limitow(self):
        sciezka, _ = QFileDialog.getOpenFileName(self, 'Checkpoint ViT (słownik z kluczem net)',
                                                 str(Path(self.ckpt_limitow).parent), 'PyTorch (*.pt *.pth)')
        if not sciezka:
            return
        self.ckpt_limitow = Path(sciezka)
        self.combo_limity.setCurrentIndex(0)
        self._zmiana_zrodla_limitow(0)

    def _ustaw_limit_kanalu(self):
        c = self.spin_lim_kanal.value()
        lo, hi = self.lim_lo.copy(), self.lim_hi.copy()
        lo[c], hi[c] = self.spin_k_min.value(), self.spin_k_max.value()
        try:
            zrodlo = self.zrodlo_limitow if self.zrodlo_limitow.endswith('(zmienione ręcznie)') \
                else f'{self.zrodlo_limitow} (zmienione ręcznie)'
            self._ustaw_limity(lo, hi, zrodlo)
        except Exception:
            self._blad_niekrytyczny(traceback.format_exc())

    def _zapisz_limity(self):
        sciezka = silnik.zapisz_limity(self.lim_lo, self.lim_hi, self.zrodlo_limitow)
        self.statusBar().showMessage(f'Zapisano limity: {sciezka} (+ CSV)', 8000)

    # ------------------------------------------------------------ latent

    def efektywny_latent(self, przytnij=True):
        tryb = self.combo_tryb.currentIndex()
        if tryb == 0:
            eff = (self.baza + self.wektor[None, :]).astype(np.float32)
        else:
            eff = self.baza.copy()
            t = self.spin_token.value()
            if tryb == 1:
                eff[t] += self.wektor
            else:
                eff[t] = self.wektor
        if przytnij and self.chk_przytnij.isChecked():
            eff = np.clip(eff, self.lim_lo[None, :], self.lim_hi[None, :]).astype(np.float32)
        return eff

    def _aktualizuj_statystyki(self):
        surowy = self.efektywny_latent(przytnij=False)
        poza = int(((surowy < self.lim_lo[None]) | (surowy > self.lim_hi[None])).sum())
        eff = self.efektywny_latent()
        k = self.lista_narzedzie.kanaly
        self.etykieta_stats.setText(
            f'Latent efektywny: min {eff.min():.3f}, średnia {eff.mean():.3f}, max {eff.max():.3f} | '
            f'zmienionych kanałów: {int(np.count_nonzero(self.wektor))}/768 '
            f'(w tym narzędzia {int(np.count_nonzero(self.wektor[k]))}/{len(k)}) | '
            f'wartości poza limitami: {poza}' + (' (przycięte)' if poza and self.chk_przytnij.isChecked() else ''))
        self._zaplanuj_na_biezaco()

    def _zaplanuj_na_biezaco(self):
        if getattr(self, 'chk_na_biezaco', None) is not None and self.chk_na_biezaco.isChecked() \
                and self.trained is not None:
            self.timer_na_biezaco.start(self.spin_opoznienie.value())

    # --------------------------------------------------------------- baza

    def _ustaw_baze(self, baza, zrodlo):
        self.baza = baza.astype(np.float32)
        self.zrodlo_bazy = zrodlo
        self._przelicz_zakresy()
        self.etykieta_bazy.setText(f'Źródło: {zrodlo}')
        self._odswiez_strone()
        self._aktualizuj_statystyki()

    def _wczytaj_baze_z_obrazu(self):
        sciezka, _ = QFileDialog.getOpenFileName(self, 'Obraz dla naturalnego latentu ViT', '',
                                                 'Obrazy (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not sciezka:
            return
        self.statusBar().showMessage('ViT liczy naturalny latent z obrazu...')
        self.btn_obraz.setEnabled(False)
        self.sciezka_bazy = sciezka
        self.watek_vit = WatekViT(sciezka, self)
        self.watek_vit.gotowy.connect(self._baza_z_vit)
        self.watek_vit.blad.connect(self._blad_niekrytyczny)
        self.watek_vit.start()

    def _baza_z_vit(self, latent, nazwa):
        self.btn_obraz.setEnabled(True)
        self._ustaw_baze(latent, f'obraz: {nazwa} (naturalny latent ViT)')
        self.statusBar().showMessage('Latent bazowy z ViT gotowy.', 5000)

    def _wczytaj_baze_z_npz(self):
        sciezka, _ = QFileDialog.getOpenFileName(self, 'Latent NPZ 600x768', '', 'NPZ (*.npz)')
        if not sciezka:
            return
        try:
            latent, klucz = silnik.wczytaj_latent_npz(sciezka)
        except Exception:
            self._blad_niekrytyczny(traceback.format_exc())
            return
        self._ustaw_baze(latent, f'{Path(sciezka).name} [{klucz}]')

    def _przywroc_baze(self):
        baza, zrodlo = silnik.domyslna_baza()
        self._ustaw_baze(baza, zrodlo)

    # ---------------------------------------------------------- generowanie

    def _model_gotowy(self, trained):
        self.trained = trained
        urzadzenie = trained.device
        nazwa = torch.cuda.get_device_name(urzadzenie) if urzadzenie.type == 'cuda' else 'CPU'
        self.etykieta_model.setText(
            f'Model gotowy: checkpoint krok {trained.step}, tryb cross-attention, {nazwa}.')
        self.btn_generuj.setEnabled(True)
        self.statusBar().showMessage('Model załadowany. Ustaw wektor i kliknij GENERUJ OBRAZ.', 8000)

    def _generuj(self, na_biezaco=False):
        if self.trained is None:
            return
        if self.watek_generacji is not None:
            if na_biezaco:
                self.oczekuje_na_biezaco = True
            return
        latent = self.efektywny_latent()
        kroki = self.spin_kroki_live.value() if na_biezaco else self.spin_kroki.value()
        self.pasek.setMaximum(kroki)
        self.pasek.setValue(0)
        self.btn_generuj.setEnabled(False)
        self.statusBar().showMessage(f'Generowanie: {kroki} kroków DDIM...')
        img2img = self.combo_start.currentIndex() == 1
        if img2img and self.start_rgb is None:
            self.btn_generuj.setEnabled(True)
            self.statusBar().showMessage('Wybierz zdjęcie startowe (Zdjęcie startowe... albo Jak baza).', 8000)
            return
        opcje = dict(start=self.start_rgb if img2img else None, sila=self.spin_sila.value() if img2img else 1.0,
                     podglad_co=self.spin_podglad_co.value() if self.chk_podglad.isChecked() else 0,
                     podglad_x0=self.combo_podglad.currentIndex() == 1)
        if img2img:
            self.pasek.setMaximum(max(1, min(kroki, round(kroki * self.spin_sila.value()))))
        self.watek_generacji = WatekGeneracji(self.trained, latent, kroki,
                                              self.spin_ziarno.value(), self, **opcje)
        self.watek_generacji.postep.connect(lambda i, n: (self.pasek.setMaximum(n), self.pasek.setValue(i)))
        self.watek_generacji.podglad.connect(self._podglad_kroku)
        self.watek_generacji.gotowy.connect(self._generacja_gotowa)
        self.watek_generacji.blad.connect(self._blad_generacji)
        self.watek_generacji.start()

    def _generacja_gotowa(self, rgb):
        self.watek_generacji = None
        self.btn_generuj.setEnabled(True)
        self.obraz = rgb
        u8 = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0))
        h, w = u8.shape[:2]
        qimg = QImage(u8.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self.etykieta_obraz.ustaw_pixmap(QPixmap.fromImage(qimg))
        self.btn_zapisz_obraz.setEnabled(True)
        self.statusBar().showMessage('Obraz wygenerowany.', 5000)
        self._dokoncz_oczekujace()

    def _dokoncz_oczekujace(self):
        if self.oczekuje_na_biezaco:
            self.oczekuje_na_biezaco = False
            self._generuj(na_biezaco=True)

    def _blad_generacji(self, komunikat):
        self.watek_generacji = None
        self.btn_generuj.setEnabled(True)
        self.oczekuje_na_biezaco = False
        self.chk_na_biezaco.setChecked(False)   # bez pętli okien z błędem
        self._blad_niekrytyczny(komunikat)

    @staticmethod
    def _pixmapa(rgb):
        u8 = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0))
        h, w = u8.shape[:2]
        return QPixmap.fromImage(QImage(u8.data, w, h, w * 3, QImage.Format.Format_RGB888).copy())

    def _podglad_kroku(self, i, n, rgb):
        opis = 'zaszumiony start' if i == 0 else f'krok {i}/{n}'
        self.etykieta_obraz.ustaw_pixmap(self._pixmapa(rgb), opis)

    # ------------------------------------------------------------ tokeny i start

    def _token_klikniety(self, token):
        if self.combo_tryb.currentIndex() == 0:
            self.combo_tryb.setCurrentIndex(1)       # klik w patch = edycja jednego tokena
        self.spin_token.setValue(token)

    def _pokaz_token(self, token):
        wiersz, kolumna = divmod(int(token), KOLUMNY_SIATKI)
        self.etykieta_obraz.ustaw_token(token)
        self.etykieta_tokena.setText(f'token {token} (wiersz {wiersz}, kolumna {kolumna}; '
                                     f'piksele ViT x {kolumna * 16}–{kolumna * 16 + 15}, y {wiersz * 16}–{wiersz * 16 + 15})')

    def _ustaw_start(self, sciezka):
        try:
            self.start_rgb = silnik.obraz_startowy(sciezka)
        except Exception:
            self._blad_niekrytyczny(traceback.format_exc())
            return
        self.start_nazwa = Path(sciezka).name
        self.etykieta_startu.setText(f'Zdjęcie startowe: {self.start_nazwa}')
        self.combo_start.setCurrentIndex(1)
        if self.obraz is None:
            self.etykieta_obraz.ustaw_pixmap(self._pixmapa(self.start_rgb), 'zdjęcie startowe')
        self._zaplanuj_na_biezaco()

    def _wybierz_zdjecie_startowe(self):
        sciezka, _ = QFileDialog.getOpenFileName(self, 'Zdjęcie startowe (img2img)', '',
                                                 'Obrazy (*.png *.jpg *.jpeg *.bmp *.webp)')
        if sciezka:
            self._ustaw_start(sciezka)

    def _start_jak_baza(self):
        if not self.sciezka_bazy:
            QMessageBox.information(self, 'Brak zdjęcia bazy',
                                    'Latent bazowy nie pochodzi ze zdjęcia. Wczytaj bazę przyciskiem '
                                    '„Z obrazu (ViT)...” albo wybierz „Zdjęcie startowe...”.')
            return
        self._ustaw_start(self.sciezka_bazy)

    # --------------------------------------------------------------- zapis

    def _zapisz_obraz(self):
        if self.obraz is None:
            return
        from PIL import Image
        silnik.WYJSCIE.mkdir(parents=True, exist_ok=True)
        znacznik = datetime.now().strftime('%Y%m%d_%H%M%S')
        sciezka = silnik.WYJSCIE/f'obraz_{znacznik}.png'
        u8 = (np.clip(self.obraz, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(u8).save(sciezka)
        self.statusBar().showMessage(f'Zapisano: {sciezka}', 8000)

    def _zapisz_latent(self):
        silnik.WYJSCIE.mkdir(parents=True, exist_ok=True)
        znacznik = datetime.now().strftime('%Y%m%d_%H%M%S')
        sciezka = silnik.WYJSCIE/f'latent_{znacznik}.npz'
        np.savez(sciezka, latent=self.efektywny_latent(), baza=self.baza,
                 wektor=self.wektor, tryb=self.combo_tryb.currentIndex(),
                 token=self.spin_token.value(), h_min=self.lim_lo, h_max=self.lim_hi,
                 zrodlo_limitow=self.zrodlo_limitow, przyciete=self.chk_przytnij.isChecked(),
                 start=str(self.start_nazwa or '') if self.combo_start.currentIndex() == 1 else 'szum',
                 sila_szumu=self.spin_sila.value())
        self.statusBar().showMessage(f'Zapisano: {sciezka}', 8000)

    # --------------------------------------------------------------- błędy

    def _blad_krytyczny(self, komunikat):
        self.etykieta_model.setText('Błąd ładowania modelu — szczegóły w oknie dialogowym.')
        QMessageBox.critical(self, 'Błąd', komunikat)

    def _blad_niekrytyczny(self, komunikat):
        self.btn_obraz.setEnabled(True)
        self.statusBar().showMessage('Wystąpił błąd — szczegóły w oknie dialogowym.')
        QMessageBox.warning(self, 'Błąd', komunikat)


def main():
    aplikacja = QApplication(sys.argv)
    aplikacja.setStyleSheet(STYL)
    okno = Okno()
    okno.show()
    sys.exit(aplikacja.exec())


if __name__ == '__main__':
    main()
