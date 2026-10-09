"""Ruch całego zbioru w jednej bazie PCA; wybór ViT, obliczenia w tle i odtwarzanie liczone na GPU.

Odtwarzanie: wszystkie obliczenia klatki liczy karta (ruch_gpu.ChmuraRuchuGPU, OpenGL 4.3 compute shadery):
położenia i przesunięcia 900 000 cząsteczek, interpolację między zapisami, kolory, granice osi, mapę ruchu 20 × 30,
dokładną medianę normy 768D i wybór punktu kliknięciem. Zapisane stany osi czasu leżą w buforze karty (cała oś,
gdy mieści się w budżecie VRAM, inaczej pierścień stanów doczytywanych w tle). CPU przekazuje tylko parametry.

Nic nie jest wczytywane ani liczone przy tworzeniu zakładki: oś czasu wczytuje dopiero kliknięcie
„Wczytaj ruch”, „Gotowy wynik 1500 obrazów” albo „Wczytaj wynik naturalny”. Zegar odtwarzania działa tylko wtedy,
gdy coś się dzieje (odtwarzanie, obrót, doczytywanie stanów, oczekiwanie na wynik z karty).
"""
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget)

import ruch_zbioru as R
import ruch_naturalny as N
from ruch_gpu import ChmuraRuchuGPU
from styl import rola
from uklad import PanelOpcji, dodaj as _dodaj
from widok_gl import WidokGL

KOLOR_NIEZAPISANY = '#e6a35c'


def dodaj(row, label, widget, opis):
    """Etykieta + kontrolka z tym samym, konkretnym opisem w dymku (obie)."""
    _dodaj(row, label, widget, opis)


def przycisk(row, tekst, opis, glowny=False):
    b = QPushButton(tekst)
    if glowny:
        rola(b, 'glowny')
    b.setToolTip(opis)
    row.addWidget(b)
    return b


def pole_wyboru(row, tekst, opis, zaznaczone=False):
    c = QCheckBox(tekst)
    c.setChecked(zaznaczone)
    c.setToolTip(opis)
    row.addWidget(c)
    return c


class ZadanieRuchu(QThread):
    postep = pyqtSignal(dict)
    gotowe = pyqtSignal(object)
    blad = pyqtSignal(str)

    def __init__(self, funkcja, parent=None):
        super().__init__(parent)
        self.funkcja = funkcja

    def run(self):
        try:
            self.gotowe.emit(self.funkcja(self.postep.emit, self.isInterruptionRequested))
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.blad.emit(str(error))


class ZakladkaRuchu(QWidget):
    def __init__(self, okno=None):
        super().__init__()
        self.okno = okno
        self.timeline = None
        self.watek = None
        self._visual_step = 0.0
        self._playing = False
        self._last_tick = None
        self._last_stats = None
        self._last_stats_time = 0.0
        self._last_status_time = 0.0
        self._last_label_time = 0.0
        self._last_index = None
        self._pending_heat = None
        self._gpu_stats = None
        self._active_checkpoint = None
        self._training_logs = None
        self._shutting_down = False
        self._build()
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)          # start dopiero, gdy coś się dzieje (_budz)

    # ------------------------------------------------------------------ budowa formularza
    def _build(self):
        lay = QVBoxLayout(self)
        self.info = rola(QLabel('Ruch cząsteczek zbioru · 600 × 768 na obraz · korekta lub historia uczenia'), 'opis')
        lay.addWidget(self.info)
        self.opcje = PanelOpcji(lay, ['Checkpoint i dane', 'Optymalizacja', 'Uczenie naturalne', 'Animacja'])
        self.opcje.zakladki.setMinimumHeight(190)
        row = self.opcje.wiersz('Checkpoint i dane')
        self.edit_checkpoint = QLineEdit(str(R.DEFAULT_CHECKPOINT)); self.edit_checkpoint.setMinimumWidth(300)
        dodaj(row, 'ViT', self.edit_checkpoint, 'Ścieżka checkpointu detektora SmallViT (klucz net, wczytanie strict=True). '
              'Z niego liczy się nowy ruch optymalizacji; gotowe zapisy zachowują swój checkpoint.')
        self.btn_checkpoint = przycisk(row, 'Wybierz checkpoint', 'Otwiera okno wyboru pliku .pt/.pth z wagami ViT.')
        self.btn_load_checkpoint = przycisk(row, 'Sprawdź i wczytaj ViT',
                                            'Wczytuje checkpoint w tle na wybranym urządzeniu, sprawdza zgodność '
                                            'architektury i liczy SHA256 (do porównania z pochodzeniem zapisu ruchu).')
        self.model_status = QLabel('Model zostanie wczytany po kliknięciu.'); row.addWidget(self.model_status)
        row = self.opcje.wiersz('Checkpoint i dane')
        self.edit_manifest = QLineEdit(str(R.DEFAULT_MANIFEST)); self.edit_manifest.setMinimumWidth(300)
        dodaj(row, 'manifest obrazów', self.edit_manifest,
              'Manifest obrazów z etykietami (JSON/JSONL) albo adnotacje COCO. Bez etykiet nie liczymy IoU.')
        self.btn_manifest = przycisk(row, 'Wybierz manifest', 'Otwiera okno wyboru manifestu .json/.jsonl.')
        self.edit_image_root = QLineEdit(); self.edit_image_root.setMinimumWidth(200)
        dodaj(row, 'folder zdjęć', self.edit_image_root,
              'Opcjonalny katalog zdjęć: rozwiązuje względne ścieżki manifestu/COCO i ma pierwszeństwo przed '
              'ścieżkami bezwzględnymi z adnotacji.')
        self.btn_image_root = przycisk(row, 'Folder zdjęć', 'Otwiera okno wyboru katalogu ze zdjęciami.')
        row = self.opcje.wiersz('Checkpoint i dane')
        self.edit_timeline = QLineEdit(str(R.DEFAULT_TIMELINE)); self.edit_timeline.setMinimumWidth(300)
        dodaj(row, 'zapis ruchu', self.edit_timeline,
              'Folder z manifest.json i stanami rzeczywistych kroków (x, y, z, norma 768D). Format 16-bitowy (pliki .q16) '
              'zajmuje połowę miejsca i pamięci karty; dawny float32 (pliki .bin) czyta się bez zmian.')
        self.btn_timeline = przycisk(row, 'Wybierz zapis ruchu', 'Otwiera okno wyboru folderu zapisu ruchu.')
        self.btn_open = przycisk(row, 'Wczytaj ruch', 'Wczytuje zapis ruchu z pola „zapis ruchu”: weryfikuje manifest '
                                 'i pierwsze stany (SHA256), a pozostałe stany doczytuje w tle prosto do pamięci karty.',
                                 glowny=True)
        self.btn_q16 = przycisk(row, 'Zapisz kopię 16-bitową',
                                'Robi obok folderu z pola „zapis ruchu” kopię w formacie 16-bitowym (folder z końcówką _q16): '
                                'połowa miejsca na dysku i w pamięci karty, szybsze wczytywanie. Błąd położenia najwyżej pół '
                                'kroku kwantyzacji (rozpiętość osi w danym kroku / 131 070). Statystyki, krzywe i metryki '
                                'przepisane bez zmian. Oryginał zostaje; po zapisie kopia od razu się wczytuje.')
        self.btn_history = przycisk(row, 'Gotowy wynik 1500 obrazów',
                                    'Wstawia domyślny zapis optymalizacji 1500 obrazów (101 stanów, 900 000 cząsteczek) '
                                    'i od razu go wczytuje.')

        row = self.opcje.wiersz('Optymalizacja')
        self.spin_images = self._int(row, 'obrazy', 1, 100000, 1500, 'Liczba pierwszych obrazów manifestu w nowym przebiegu.')
        self.spin_steps = self._int(row, 'kroki', 1, 20000, 500, 'Liczba kroków optymalizacji latentu każdego obrazu.')
        self.spin_save = self._int(row, 'zapis co', 1, 20000, 5,
                                   'Co ile kroków zapisać stan wszystkich cząsteczek (rzut 3D i norma 768D). '
                                   'Mniejsza wartość = więcej stanów i więcej miejsca na dysku.')
        self.spin_lr = self._float(row, 'lr', .000001, 10, .05, 6, 'Stały współczynnik uczenia optymalizatora latentu.')
        self.combo_optimizer = QComboBox(); self.combo_optimizer.addItems(['adam', 'sgd'])
        dodaj(row, 'optymalizator', self.combo_optimizer, 'Optymalizator korekty latentu (Adam albo SGD); wagi ViT są zamrożone.')
        self.combo_constraint = QComboBox()
        self.combo_constraint.addItem('Dolna granica GELU', 'gelu')
        self.combo_constraint.addItem('Swobodna korekta', 'none')
        dodaj(row, 'ograniczenie', self.combo_constraint,
              'Dolna granica GELU: po każdym kroku latent jest przycinany do minimum funkcji GELU (wartości osiągalne '
              'przez sieć). Swobodna korekta: bez przycinania.')
        row = self.opcje.wiersz('Optymalizacja')
        self.spin_conf = self._float(row, 'cel pewności', .01, .999, .90, 3,
                                     'Próg pewności detekcji; obraz kończy optymalizację po spełnieniu obu celów.')
        self.spin_iou = self._float(row, 'cel IoU', .01, .999, .90, 3,
                                    'Próg IoU z etykietą; obraz kończy optymalizację po spełnieniu obu celów.')
        self.spin_regularization = self._float(row, 'kara za zmianę', 0, 1000, 0, 6,
                                               'Waga kary za średni kwadrat zmiany latentu względem stanu naturalnego.')
        self.spin_seed = self._int(row, 'ziarno', 0, 1000000, 42, 'Ziarno losowania próbki cząsteczek do wspólnej bazy PCA.')
        self.combo_device = QComboBox(); self.combo_device.addItems(['cuda', 'cpu'])
        dodaj(row, 'urządzenie', self.combo_device,
              'Urządzenie obliczeń w tle (optymalizacja i pomiar naturalny w PyTorch). Domyślnie CUDA; '
              'CPU tylko z wyboru, bez cichego przełączania.')
        torch = sys.modules.get('torch')                  # bez importu torch przy tworzeniu zakładki
        if torch is not None and not torch.cuda.is_available():
            self.combo_device.setCurrentText('cpu')
        self.btn_compute = przycisk(row, 'Oblicz nowy ruch', 'Uruchamia w tle optymalizację latentów wszystkich obrazów '
                                    'na wybranym urządzeniu i zapisuje nową oś czasu w outputs/ruch_czasteczek_gui.',
                                    glowny=True)
        self.btn_stop = przycisk(row, 'Zatrzymaj obliczenia', 'Przerywa obliczenie po bieżącym kroku; ukończone stany '
                                 'zostają zapisane i można je obejrzeć.')
        self.btn_stop.setEnabled(False)
        self.cost = QLabel(); self.opcje._strony['Optymalizacja'].addWidget(self.cost)
        note = QLabel('Każdy obraz ma osobną korektę wszystkich cząsteczek. Wagi pozostają zamrożone. '
                      'Po spełnieniu obu celów stan obrazu jest utrzymywany do końca wspólnej osi czasu. '
                      'Zapisujemy rzut 3D i metryki, nie pełne 768D w każdym kroku.')
        note.setWordWrap(True); self.opcje._strony['Optymalizacja'].addWidget(rola(note, 'opis'))

        row = self.opcje.wiersz('Uczenie naturalne')
        self.edit_checkpoint_dir = QLineEdit(str(R.ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi'))
        self.edit_checkpoint_dir.setMinimumWidth(330)
        dodaj(row, 'checkpointy treningu', self.edit_checkpoint_dir,
              'Folder z zapisanymi modelami treningu w postaci kNNNN.pt (krok w nazwie i w pliku muszą się zgadzać).')
        self.btn_checkpoint_dir = przycisk(row, 'Folder checkpointów', 'Otwiera okno wyboru folderu z checkpointami treningu.')
        self.btn_scan_natural = przycisk(row, 'Sprawdź historię', 'Sprawdza w tle, które checkpointy z zakresu istnieją, '
                                         'i weryfikuje ich zapisane kroki. Nie liczy ruchu.')
        self.btn_default_natural = przycisk(row, 'Historia prowadząca do K1000',
                                            'Ustawia lokalną historię 101 checkpointów k0000–k1000, jej logi, checkpoint '
                                            'K1000 i manifest 1500 obrazów, a potem sprawdza historię.')
        row = self.opcje.wiersz('Uczenie naturalne')
        self.edit_training_logs = QLineEdit(str(R.ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane'))
        self.edit_training_logs.setMinimumWidth(330)
        dodaj(row, 'logi treningu', self.edit_training_logs,
              'Plik JSON/JSONL albo folder logów (LOG.json i OCENY.json) z jawnym krokiem treningu.')
        self.btn_logs_file = przycisk(row, 'Plik logów', 'Otwiera okno wyboru pliku logów .json/.jsonl.')
        self.btn_logs_dir = przycisk(row, 'Folder logów', 'Otwiera okno wyboru folderu z LOG.json i OCENY.json.')
        self.btn_load_logs = przycisk(row, 'Wczytaj logi', 'Wczytuje w tle historyczną stratę i mAP50 z logów i rysuje '
                                      'je pod wykresem metryk (osobno od pomiaru IoU na obrazach).')
        row = self.opcje.wiersz('Uczenie naturalne')
        self.spin_natural_images = self._int(row, 'obrazy', 1, 100000, 1500, 'Liczba pierwszych obrazów manifestu mierzonych w każdym checkpoincie.')
        self.spin_natural_start = self._int(row, 'od kroku', 0, 10000000, 0, 'Pierwszy krok treningu (baza przesunięć i PCA).')
        self.spin_natural_end = self._int(row, 'do kroku', 0, 10000000, 1000, 'Ostatni krok treningu w pomiarze (włącznie).')
        self.spin_natural_stride = self._int(row, 'checkpoint co', 1, 1000000, 10,
                                             'Co ile kroków brać zapisany checkpoint (licząc od „od kroku”).')
        self.spin_natural_batch = self._int(row, 'paczka pomiarowa', 1, 128, 32,
                                            'Ile obrazów naraz przechodzi przez model w pomiarze. Obrazy są dekodowane raz '
                                            'i leżą na karcie, więc tempo ogranicza już tylko model (około 500 obrazów/s na '
                                            'RTX 3070 przy paczce 16–64). Ta sama paczka obowiązuje przy uzupełnianiu.')
        self.btn_compute_natural = przycisk(row, 'Oblicz ruch naturalny',
                                            'Uruchamia w tle naturalny forward każdego wybranego checkpointu na tych '
                                            'samych obrazach (bez treningu i bez podmiany aktywacji) i zapisuje oś czasu.',
                                            glowny=True)
        self.btn_latest_natural = przycisk(row, 'Wczytaj wynik naturalny',
                                           'Wczytuje najnowszy gotowy pomiar naturalny dla tej liczby obrazów, a gdy go nie '
                                           'ma, ukończone stany trwającego pomiaru. Kliknij ponownie, aby dołączyć nowe '
                                           'checkpointy.')
        self.btn_resume_natural = przycisk(row, 'Uzupełnij brakujące kroki',
                                           'Dolicza na GPU tylko brakujące checkpointy najnowszego przerwanego pomiaru '
                                           'naturalnego w tym samym folderze i zapisuje je na dysk. Najpierw liczy ponownie '
                                           'krok bazowy i sprawdza, czy zgadza się bitowo z zapisem (inaczej nic nie zmienia). '
                                           'Gdy nic nie jest przerwane, tylko to zgłasza.')
        self.natural_cost = QLabel(); self.natural_cost.setWordWrap(True)
        self.opcje._strony['Uczenie naturalne'].addWidget(self.natural_cost)
        note = QLabel('Te same obrazy i etykiety wybierz w „Checkpoint i dane”. Urządzenie CUDA/CPU i ziarno PCA '
                      'są w „Optymalizacja”. Każdy zapisany model wykonuje naturalny forward w trybie eval. '
                      'Nie trenujemy ponownie ani nie podmieniamy aktywacji. Brakujące kroki nie są rekonstruowane; '
                      'płynne przejścia służą tylko animacji.')
        note.setWordWrap(True); self.opcje._strony['Uczenie naturalne'].addWidget(rola(note, 'opis'))

        row = self.opcje.wiersz('Animacja')
        self.spin_tempo = self._float(row, 'tempo (kroki/s)', 1, 1000, 50, 1,
                                      'Szybkość odtwarzania w krokach osi czasu na sekundę. Animacja nie przeskakuje '
                                      'stanów, których jeszcze nie ma na karcie.')
        self.chk_smooth = pole_wyboru(row, 'Płynne przejścia GPU',
                                      'Włączone: między zapisami karta liczy liniowe przejście (stan NIEZAPISANY, '
                                      'oznaczony nad widokiem). Wyłączone: widać tylko zapisane stany.', True)
        self.chk_movement = pole_wyboru(row, 'Ruch względem kroku 0',
                                        'Włączone: położenie minus położenie w pierwszym kroku (przesunięcie). '
                                        'Wyłączone: bezwzględne współrzędne we wspólnej bazie PCA.', True)
        self.chk_spin = pole_wyboru(row, 'Obracaj', 'Powolny obrót kamery wokół chmury (12° na sekundę).')
        self.spin_opacity = self._float(row, 'widoczność', .01, 1, .20, 2,
                                        'Krycie pojedynczego punktu (0–1). Przy 900 000 punktów mała wartość pokazuje gęstość.')
        self.spin_size = self._float(row, 'rozmiar punktu', .5, 6, 1.5, 1, 'Średnica punktu w pikselach ekranu.')
        self.combo_scope = QComboBox(); self.combo_scope.addItems(['Cały zbiór', 'Jeden obraz'])
        dodaj(row, 'widok', self.combo_scope, 'Cały zbiór: wszystkie cząsteczki wszystkich obrazów. Jeden obraz: tylko '
              '600 cząsteczek obrazu o numerze z pola „nr obrazu” (te same indeksy co w zapisie).')
        self.spin_image = self._int(row, 'nr obrazu', 0, 1499, 0, 'Numer obrazu (kolejność manifestu) w widoku „Jeden obraz”.')
        row = self.opcje.wiersz('Animacja')
        self.combo_color = QComboBox()
        self.combo_color.addItem('Norma zmiany 768D (zapis)', 0)
        self.combo_color.addItem('Prędkość w PCA między zapisami', 1)
        self.combo_color.addItem('|Δ| w PCA od pierwszego kroku', 2)
        dodaj(row, 'kolor', self.combo_color,
              'Co koduje kolor punktu (liczy karta): zapisana norma zmiany w pełnych 768D; prędkość w rzucie PCA '
              'między sąsiednimi zapisami; długość przesunięcia w rzucie PCA od pierwszego kroku. Skala: maksimum '
              'po wszystkich zapisanych krokach na karcie.')
        self.combo_kadr = QComboBox()
        self.combo_kadr.addItem('Cały zbiór · wszystkie zapisane kroki', 0)
        self.combo_kadr.addItem('Widoczne cząsteczki · wszystkie zapisane kroki', 1)
        self.combo_kadr.addItem('Widoczne cząsteczki · bieżący stan', 2)
        dodaj(row, 'kadr osi', self.combo_kadr,
              'Granice osi liczone na karcie: stałe dla całego zbioru (jak w zapisie producenta), stałe dla widocznych '
              'cząsteczek, albo dopasowane do bieżącego stanu w każdej klatce (kamera podąża za chmurą).')
        self.spin_trail = self._int(row, 'ślad (zapisy)', 0, 60, 0,
                                    'Długość śladu w zapisanych krokach (0 = bez śladu). Rysowany dla widoku jednego '
                                    'obrazu; w widoku całego zbioru tylko dla klikniętej cząsteczki.')
        self.spin_vram = self._float(row, 'budżet VRAM (GB)', .25, 16, 2.0, 2,
                                     'Ile pamięci karty mogą zająć zapisane stany. Gdy cała oś się mieści, wszystkie stany '
                                     'są na karcie; inaczej działa pierścień stanów doczytywanych w tle.')
        self.btn_camera = przycisk(row, 'Ustaw kamerę', 'Przywraca domyślną odległość i kąty kamery.')
        self.btn_unpick = przycisk(row, 'Odznacz cząsteczkę', 'Usuwa wyróżnienie klikniętej cząsteczki i jej ślad.')
        self.btn_png = przycisk(row, 'Zapisz podgląd PNG', 'Zapisuje bieżący obraz widoku 3D do pliku PNG.')
        self.image_name = QLabel(); self.opcje._strony['Animacja'].addWidget(self.image_name)

        controls = QHBoxLayout()
        self.btn_play = rola(QPushButton('Odtwórz'), 'glowny'); self.btn_play.setEnabled(False)
        self.btn_play.setToolTip('Odtwarza albo zatrzymuje animację. Pauza zachowuje pozycję między zapisami.')
        self.btn_begin = QPushButton('Od początku'); self.btn_begin.setToolTip('Przechodzi do pierwszego zapisanego stanu.')
        self.btn_prev = QPushButton('Poprzedni zapis'); self.btn_prev.setToolTip('Przechodzi do poprzedniego zapisanego stanu.')
        self.btn_next = QPushButton('Następny zapis'); self.btn_next.setToolTip('Przechodzi do następnego zapisanego stanu.')
        self.step_label = QLabel('Krok —'); self.step_label.setMinimumWidth(165)
        self.suwak = QSlider(Qt.Orientation.Horizontal); self.suwak.setEnabled(False)
        self.suwak.setToolTip('Wybór zapisanego stanu osi czasu (tylko rzeczywiste zapisy, bez stanów pośrednich).')
        for widget in (self.btn_play, self.btn_begin, self.btn_prev, self.btn_next, self.step_label, self.suwak):
            controls.addWidget(widget)
        lay.addLayout(controls)
        self.frame_label = rola(QLabel('Wczytaj gotowy wynik lub oblicz nowe trajektorie.'), 'opis'); lay.addWidget(self.frame_label)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.view = WidokGL(); self.view.setMinimumHeight(480); self.view.setBackgroundColor('#10171c')
        self.view.setCameraPosition(distance=3.8, elevation=18, azimuth=35)
        self.view.setToolTip('Kliknij cząsteczkę, aby zobaczyć obraz i patch (wybór liczy karta). '
                             'Przeciąganie obraca kamerę, kółko przybliża.')
        self.cloud = ChmuraRuchuGPU(); self.view.addItem(self.cloud)
        self.cloud.statystyki_gotowe.connect(self._gpu_results, Qt.ConnectionType.QueuedConnection)
        self.cloud.wybrano.connect(self._picked, Qt.ConnectionType.QueuedConnection)
        grid = gl.GLGridItem(); grid.setSize(2.5, 2.5); grid.setSpacing(.5, .5); grid.translate(0, 0, -1.2)
        self.view.addItem(grid)
        self.view.klik_px.connect(self._pick)
        split.addWidget(self.view)
        right = QWidget(); rl = QVBoxLayout(right)
        self.metrics = QLabel('Pomiary pojawią się po wczytaniu.'); self.metrics.setWordWrap(True); rl.addWidget(self.metrics)
        self.curve = pg.PlotWidget(title='Rzeczywiste metryki detekcji'); self.curve.setMinimumWidth(280)
        self.curve.setMinimumHeight(140)
        self.curve.addLegend(); self.curve.setYRange(0, 1); self.curve.setLabel('bottom', 'krok optymalizacji')
        self.conf_curve = self.curve.plot(pen='#79d4c5', name='Pewność')
        self.iou_curve = self.curve.plot(pen='#e6c88f', name='IoU')
        self.cursor = pg.InfiniteLine(angle=90, pen='#ffffff'); self.curve.addItem(self.cursor)
        rl.addWidget(self.curve, 1)
        self.log_box = QWidget(); ll = QVBoxLayout(self.log_box); ll.setContentsMargins(0, 0, 0, 0)
        self.combo_log_metric = QComboBox()
        self.combo_log_metric.addItem('Strata · historyczny log treningu', 'loss')
        self.combo_log_metric.addItem('mAP50 · historyczna walidacja', 'map50')
        self.combo_log_metric.setToolTip('Która historyczna seria z logów treningu jest na wykresie.')
        ll.addWidget(self.combo_log_metric)
        self.log_curve = pg.PlotWidget(); self.log_curve.setMinimumHeight(100); self.log_curve.setMaximumHeight(150)
        self.log_curve.setLabel('bottom', 'rzeczywisty krok treningu')
        self.log_line = self.log_curve.plot(pen='#b69ee8', symbol='o', symbolSize=4)
        self.log_cursor = pg.InfiniteLine(angle=90, pen='#ffffff'); self.log_curve.addItem(self.log_cursor)
        ll.addWidget(self.log_curve)
        self.log_label = QLabel(); self.log_label.setWordWrap(True); ll.addWidget(self.log_label)
        self.log_box.hide(); rl.addWidget(self.log_box)
        self.heat = pg.ImageView(); self.heat.ui.roiBtn.hide(); self.heat.ui.menuBtn.hide()
        self.heat.setMinimumHeight(150)
        self.heat_label = QLabel('Ruch 768D per patch (mapa 20 × 30, liczona na karcie)')
        rl.addWidget(self.heat_label); rl.addWidget(self.heat, 1)
        split.addWidget(right); split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        self.gpu_status = rola(QLabel('GPU: oczekuje na dane.'), 'status'); lay.addWidget(self.gpu_status)
        self.pasek = QProgressBar(); self.pasek.setRange(0, 100); lay.addWidget(self.pasek)
        self.status = QLabel('Nic nie wczytuje się ani nie liczy przy otwieraniu zakładki.'); self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.btn_checkpoint.clicked.connect(lambda: self._choose_file(self.edit_checkpoint, 'Checkpoint (*.pt *.pth);;Wszystkie (*)'))
        self.btn_manifest.clicked.connect(lambda: self._choose_file(self.edit_manifest, 'Manifest (*.json *.jsonl);;Wszystkie (*)'))
        self.btn_image_root.clicked.connect(lambda: self._choose_dir(self.edit_image_root))
        self.btn_timeline.clicked.connect(lambda: self._choose_dir(self.edit_timeline))
        self.btn_load_checkpoint.clicked.connect(self._load_model)
        self.btn_compute.clicked.connect(self._compute)
        self.btn_stop.clicked.connect(self._cancel_job)
        self.btn_open.clicked.connect(self._load_timeline)
        self.btn_q16.clicked.connect(self._convert_q16)
        self.btn_history.clicked.connect(self._history)
        self.btn_checkpoint_dir.clicked.connect(lambda: self._choose_dir(self.edit_checkpoint_dir))
        self.btn_scan_natural.clicked.connect(self._scan_natural)
        self.btn_default_natural.clicked.connect(self._default_natural)
        self.btn_logs_file.clicked.connect(lambda: self._choose_file(self.edit_training_logs, 'Logi (*.json *.jsonl);;Wszystkie (*)'))
        self.btn_logs_dir.clicked.connect(lambda: self._choose_dir(self.edit_training_logs))
        self.btn_load_logs.clicked.connect(self._load_logs)
        self.btn_compute_natural.clicked.connect(self._compute_natural)
        self.btn_latest_natural.clicked.connect(self._latest_natural)
        self.btn_resume_natural.clicked.connect(self._resume_natural)
        self.combo_log_metric.currentIndexChanged.connect(self._plot_logs)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_begin.clicked.connect(lambda: self._seek_index(0))
        self.btn_prev.clicked.connect(lambda: self._seek_index(self.suwak.value() - 1))
        self.btn_next.clicked.connect(lambda: self._seek_index(self.suwak.value() + 1))
        self.suwak.valueChanged.connect(self._seek_index)
        for widget in (self.chk_movement, self.chk_smooth):
            widget.toggled.connect(self._display_options)
        for widget in (self.combo_scope, self.combo_color, self.combo_kadr):
            widget.currentIndexChanged.connect(self._display_options)
        for widget in (self.spin_image, self.spin_opacity, self.spin_size, self.spin_trail):
            widget.valueChanged.connect(self._display_options)
        self.chk_spin.toggled.connect(lambda _: self._budz())
        self.spin_vram.valueChanged.connect(self._vram_changed)
        self.btn_camera.clicked.connect(lambda: self.view.setCameraPosition(distance=3.8, elevation=18, azimuth=35))
        self.btn_unpick.clicked.connect(lambda: self._picked(-2))
        self.btn_png.clicked.connect(self._save_png)
        for widget in (self.spin_images, self.spin_steps, self.spin_save):
            widget.valueChanged.connect(self._estimate)
        for widget in (self.spin_natural_images, self.spin_natural_start, self.spin_natural_end, self.spin_natural_stride):
            widget.valueChanged.connect(self._estimate_natural)
        self._estimate()
        self._estimate_natural()

    @staticmethod
    def _int(row, label, lo, hi, value, opis):
        widget = QSpinBox(); widget.setRange(lo, hi); widget.setValue(value); dodaj(row, label, widget, opis); return widget

    @staticmethod
    def _float(row, label, lo, hi, value, decimals, opis):
        widget = QDoubleSpinBox(); widget.setDecimals(decimals); widget.setRange(lo, hi); widget.setValue(value)
        dodaj(row, label, widget, opis); return widget

    @staticmethod
    def _set_text(label, text):
        if label.text() != text:
            label.setText(text)

    def _mark_unsaved(self, unsaved):
        """Wyróżnienie stanu niezapisanego; styl zmieniany tylko przy zmianie (setStyleSheet przelicza wygląd)."""
        if getattr(self, '_unsaved_style', None) != unsaved:
            self._unsaved_style = unsaved
            self.frame_label.setStyleSheet(f'color: {KOLOR_NIEZAPISANY}; font-weight: bold;' if unsaved else '')

    def _choose_file(self, edit, filtr):
        name, _ = QFileDialog.getOpenFileName(self, 'Wybierz plik', edit.text(), filtr)
        if name:
            edit.setText(name)

    def _choose_dir(self, edit):
        name = QFileDialog.getExistingDirectory(self, 'Wybierz folder', edit.text())
        if name:
            edit.setText(name)

    def _estimate(self):
        frames = len(set([0, self.spin_steps.value(), *range(0, self.spin_steps.value() + 1, self.spin_save.value())]))
        size = self.spin_images.value() * 600 * 16 * frames / 1e9
        self.cost.setText(f'Zapis widoku: około {size:.2f} GB, {frames} stanów. Pełna optymalizacja odbywa się w 768 wymiarach.')

    def _estimate_natural(self):
        span = max(0, self.spin_natural_end.value() - self.spin_natural_start.value())
        frames = span // self.spin_natural_stride.value() + 1
        count = self.spin_natural_images.value() * 600
        output = count * 8 * frames / 1e9                    # zapis 16-bitowy (q16): 8 bajtów na cząsteczkę
        baseline = count * 768 * 4 / 1e9
        minuty = frames * self.spin_natural_images.value() / 500 / 60
        self.natural_cost.setText(f'Plan: do {frames} zapisanych modeli · {count:,} cząsteczek. '
                                 f'Zapis 16-bitowy około {output:.2f} GB na dysku. Stan początkowy {baseline:.2f} GB '
                                 f'trzymany w pamięci karty (albo RAM) tylko na czas pomiaru. Około {minuty:.0f} min na '
                                 'RTX 3070. Przed startem sprawdzamy dostępne checkpointy, dysk i pamięć.')

    # ------------------------------------------------------------------ zadania w tle (obliczenia, wczytywanie)
    def _job(self, function, on_done):
        if self.watek is not None:
            return
        self._pause()
        self.watek = ZadanieRuchu(function, self)
        self.watek.on_done = on_done
        self.watek.postep.connect(self._progress)
        self.watek.gotowe.connect(self._job_done)
        self.watek.blad.connect(self._error)
        self.watek.finished.connect(self._job_finished)
        self.btn_compute.setEnabled(False); self.btn_load_checkpoint.setEnabled(False)
        self.btn_open.setEnabled(False); self.btn_history.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_q16.setEnabled(False)
        for button in self._natural_buttons():
            button.setEnabled(False)
        self.watek.start()

    def _job_done(self, result):
        worker = self.sender()
        if worker is self.watek:
            worker.on_done(result)

    def _job_finished(self):
        worker = self.sender()
        if worker is not self.watek:
            if worker:
                worker.deleteLater()
            return
        self.watek = None
        if worker:
            worker.deleteLater()
        self.btn_compute.setEnabled(True); self.btn_load_checkpoint.setEnabled(True)
        self.btn_open.setEnabled(True); self.btn_history.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_q16.setEnabled(True)
        for button in self._natural_buttons():
            button.setEnabled(True)

    def _natural_buttons(self):
        return (self.btn_scan_natural, self.btn_default_natural, self.btn_load_logs,
                self.btn_compute_natural, self.btn_latest_natural, self.btn_resume_natural)

    def _error(self, message):
        self.status.setText('Nie udało się wykonać operacji: ' + message)

    def _progress(self, data):
        phase = data.get('phase')
        labels = {'validate': 'Sprawdzanie obrazów i etykiet', 'estimate': 'Sprawdzanie miejsca i pamięci',
                  'pca': 'Wyznaczanie wspólnej bazy PCA z naturalnych cząsteczek',
                  'baseline': 'Pomiar naturalnej detekcji', 'optimize': 'Optymalizacja cząsteczek',
                  'discover': 'Sprawdzanie checkpointów treningu', 'measure': 'Naturalny pomiar obrazów',
                  'frame': 'Zapisano stan wszystkich cząsteczek', 'decode': 'Dekodowanie obrazów (raz na pomiar, do pamięci karty)',
                  'finished': 'Obliczenia zakończone'}
        message = (labels[phase] if phase in ('discover', 'measure', 'frame') else
                   data.get('message') or labels.get(phase, str(data.get('status', 'Trwa obliczenie'))))
        if 'image' in data:
            message += f' · obraz {data["image"]}/{data.get("image_count", "?")}'
        if 'step' in data:
            message += f' · krok {data["step"]}'
        if 'confidence' in data:
            message += f' · pewność {data["confidence"]:.1%} · IoU {data["iou"]:.1%}'
        self.status.setText(message)
        if 'fraction' in data:
            self.pasek.setValue(round(float(data['fraction']) * 100))
        elif phase == 'optimize' and data.get('image_count'):
            cfg_steps = self.spin_steps.value()
            total = (data['image'] - 1 + data.get('step', 0) / cfg_steps) / data['image_count']
            self.pasek.setValue(round(total * 100))

    def _cancel_job(self):
        if self.watek:
            self.watek.requestInterruption()
            self.status.setText('Kończenie bieżącego kroku i zachowanie częściowego zapisu…')

    def _load_model(self):
        path, device = self.edit_checkpoint.text().strip(), self.combo_device.currentText()
        def load(progress, cancel):
            import hashlib
            from detekcja import Detektor
            progress({'message': 'Sprawdzanie checkpointu i zgodności architektury…'})
            model = Detektor(path, device)
            with open(path, 'rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            result = {'path': str(Path(path).resolve()), 'step': model.krok, 'sha256': digest, 'device': str(model.dev)}
            del model
            return result
        def loaded(result):
            self._active_checkpoint = result
            self.model_status.setText(f'ViT k{result["step"]} · {result["device"]} · SHA {result["sha256"][:12]}')
            self.status.setText('Checkpoint wczytany i sprawdzony. Obliczenia użyją osobnej zamrożonej kopii.')
            self._provenance()
        self._job(load, loaded)

    def _config(self):
        return R.Config(checkpoint=self.edit_checkpoint.text().strip(), manifest=self.edit_manifest.text().strip(),
                        image_root=self.edit_image_root.text().strip(), max_images=self.spin_images.value(),
                        steps=self.spin_steps.value(), save_every=self.spin_save.value(), lr=self.spin_lr.value(),
                        target_confidence=self.spin_conf.value(), target_iou=self.spin_iou.value(),
                        device=self.combo_device.currentText(), seed=self.spin_seed.value(),
                        constraint=self.combo_constraint.currentData(), regularization=self.spin_regularization.value(),
                        optimizer=self.combo_optimizer.currentText())

    def _compute(self):
        try:
            config = self._config()
            config.validate()
        except Exception as error:
            self._error(str(error)); return
        def done(path):
            self.edit_timeline.setText(str(path))
            result_status = json.loads((Path(path) / 'manifest.json').read_text(encoding='utf-8')).get('status')
            self.status.setText(f'{"Obliczenia zakończone" if result_status == "complete" else "Zachowano częściowy wynik"}: '
                                f'{path}. Kliknij „Wczytaj ruch”, żeby obejrzeć dostępne stany.')
            if result_status == 'complete':
                self.pasek.setValue(100)
        self._job(lambda p, c: R.run(config, progress=p, cancel=c), done)

    def _natural_config(self):
        return N.NaturalConfig(checkpoint_dir=self.edit_checkpoint_dir.text().strip(),
                               manifest=self.edit_manifest.text().strip(), image_root=self.edit_image_root.text().strip(),
                               max_images=self.spin_natural_images.value(), start_step=self.spin_natural_start.value(),
                               end_step=self.spin_natural_end.value(), checkpoint_stride=self.spin_natural_stride.value(),
                               batch_size=self.spin_natural_batch.value(), device=self.combo_device.currentText(),
                               seed=self.spin_seed.value(), logs=self.edit_training_logs.text().strip(),
                               reference_checkpoint=self.edit_checkpoint.text().strip())

    def _scan_natural(self):
        path = self.edit_checkpoint_dir.text().strip()
        start, end, stride = self.spin_natural_start.value(), self.spin_natural_end.value(), self.spin_natural_stride.value()
        def done(records):
            if not records:
                self._error('Nie ma checkpointów w wybranym zakresie.'); return
            self.status.setText(f'Historia: {len(records)} checkpointów · kroki {records[0]["step"]}–{records[-1]["step"]}. '
                                'Zapisane modele będą mierzone na tych samych obrazach. Pełny pomiar uruchamia „Oblicz ruch naturalny”.')
        self._job(lambda p, c: N.discover(path, start_step=start, end_step=end, checkpoint_stride=stride), done)

    def _default_natural(self):
        family = R.ROOT / 'experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny'
        self.edit_checkpoint_dir.setText(str(family / 'wagi'))
        self.edit_training_logs.setText(str(family / 'dane'))
        self.edit_checkpoint.setText(str(R.DEFAULT_CHECKPOINT))
        self.edit_manifest.setText(str(R.DEFAULT_MANIFEST))
        self.spin_natural_start.setValue(0); self.spin_natural_end.setValue(1000); self.spin_natural_stride.setValue(10)
        self._scan_natural()

    def _compute_natural(self):
        try:
            config = self._natural_config().validate()
        except Exception as error:
            self._error(str(error)); return
        if self._existing_natural_measurement(config):
            self.status.setText('Ten pomiar naturalnej historii już pracuje w tle. '
                                'Kliknij „Wczytaj wynik naturalny”, aby obejrzeć ukończone stany.')
            return
        def done(path):
            self.edit_timeline.setText(str(path))
            manifest = json.loads((Path(path) / 'manifest.json').read_text(encoding='utf-8'))
            complete = manifest.get('status') == 'complete'
            self.status.setText(f'{"Pomiar naturalny zakończony" if complete else "Zachowano częściowy pomiar naturalny"}: '
                                f'{path}. Kliknij „Wczytaj ruch”, żeby obejrzeć dostępne checkpointy.')
            if complete:
                self.pasek.setValue(100)
        self._job(lambda p, c: N.run(config, progress=p, cancel=c), done)

    def _resume_natural(self):
        """Uzupełnia najnowszy pomiar naturalny z brakującymi checkpointami (liczy GPU, zapis na dysk)."""
        lista = N.niekompletne()
        if not lista:
            self.status.setText('Nie ma przerwanych pomiarów naturalnych: wszystkie zapisane mają komplet checkpointów.')
            return
        folder, gotowe, wszystkie, stan = lista[0]
        self.status.setText(f'Uzupełniam {folder.name}: {gotowe}/{wszystkie} checkpointów ({stan}); '
                            f'brakujące {wszystkie - gotowe} liczy karta.')

        def done(path):
            self.edit_timeline.setText(str(path))
            m = json.loads((Path(path) / 'manifest.json').read_text(encoding='utf-8'))
            self.status.setText(f'Uzupełniono: {len(m["steps"])}/{len(m["checkpoints"])} checkpointów zapisanych w '
                                f'{path}. Kliknij „Wczytaj ruch”, żeby obejrzeć.')
            if m.get('status') == 'complete':
                self.pasek.setValue(100)
        device = self.combo_device.currentText() if hasattr(self, 'combo_device') else None
        self._job(lambda p, c: N.uzupelnij(folder, progress=p, cancel=c, device=device), done)

    @staticmethod
    def _existing_natural_measurement(config):
        """Recognize only this app's fixed-input measurement helper, never control other processes."""
        import psutil
        if (Path(config.checkpoint_dir).resolve() != Path(N.DEFAULT_CHECKPOINT_DIR).resolve()
                or Path(config.manifest).resolve() != Path(R.DEFAULT_MANIFEST).resolve()
                or config.start_step != 0):
            return False
        helper = R.ROOT / 'experiments/app_ruch_czasteczek_20261009/pomiar_naturalny.py'
        for process in psutil.process_iter(['cmdline']):
            try:
                args = process.info['cmdline'] or []
                if not any((Path(a) if Path(a).is_absolute() else R.ROOT / a).resolve() == helper.resolve()
                           for a in args[1:2]):
                    continue
                def option(name, default):
                    return int(args[args.index(name) + 1]) if name in args else default
                if (option('--images', 1500) == config.max_images and option('--end', 1000) == config.end_step
                        and option('--stride', 10) == config.checkpoint_stride):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError, IndexError):
                continue
        return False

    def _latest_natural(self):
        """Tylko po kliknięciu: szuka najnowszego pomiaru naturalnego i go wczytuje."""
        root = Path(N.OUTPUT_ROOT)
        partial = None
        for directory in sorted(root.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
            path = directory / 'manifest.json'
            if path.is_file():
                try:
                    m = json.loads(path.read_text(encoding='utf-8'))
                    if (m.get('time_axis') == 'training_checkpoint'
                            and m.get('image_count') == self.spin_natural_images.value() and m.get('steps')):
                        if m.get('status') == 'complete':
                            self.edit_timeline.setText(str(directory)); self._load_timeline(); return
                        if partial is None:
                            partial = directory
                except (OSError, ValueError):
                    continue
        if partial is not None:
            self.edit_timeline.setText(str(partial)); self._load_timeline(); return
        self._error('Brak gotowego naturalnego pomiaru dla tej liczby obrazów. Użyj „Oblicz ruch naturalny” lub wybierz zapis ręcznie.')

    def _load_logs(self):
        path = self.edit_training_logs.text().strip()
        def done(logs):
            self._set_logs(logs)
            self.status.setText('Wczytano historyczne logi. Ich strata i mAP są oddzielne od pomiaru IoU na wybranych obrazach.')
        self._job(lambda p, c: N.load_logs(path), done)

    def _set_logs(self, logs):
        self._training_logs = logs
        self.log_box.setVisible(bool(logs and logs.get('series')))
        self._plot_logs()

    def _plot_logs(self, *_):
        logs = self._training_logs or {}
        key = self.combo_log_metric.currentData()
        record = logs.get('series', {}).get(key, {})
        self.log_line.setData(record.get('steps', []), record.get('values', []))
        self.log_curve.enableAutoRange()
        self.log_curve.setLabel('left', 'strata' if key == 'loss' else 'mAP50')
        paths = [Path(f.get('path', '')).name for f in logs.get('files', [])]
        warning = '; '.join(logs.get('warnings', []))
        self.log_label.setText(f'Źródło: {", ".join(paths) or "brak logu"}. '
                               f'{len(record.get("steps", []))} rzeczywistych ocen. ' + warning)

    def _history(self):
        self.edit_timeline.setText(str(R.DEFAULT_TIMELINE)); self._load_timeline()

    def _load_timeline(self):
        """Tylko po kliknięciu. Wątek tła: manifest + pierwsze dwa stany (SHA256); resztę doczytuje strumień karty."""
        path = self.edit_timeline.text().strip()
        def load(progress, cancel):
            progress({'message': 'Weryfikacja zapisu i pierwszych stanów…'})
            timeline = R.Timeline(path)
            stan = (timeline.read_raw if getattr(timeline, 'frame_format', 'f32') == 'q16' else timeline.frame)
            first = np.array(stan(timeline.steps[0]))          # w formacie pliku: q16 idzie na kartę bez zamiany
            second = np.array(stan(timeline.steps[min(1, len(timeline.steps) - 1)]))
            return timeline, first, second
        self._job(load, self._loaded)

    def _convert_q16(self):
        """Kopia 16-bitowa osi czasu z pola „zapis ruchu” (w tle), potem jej wczytanie."""
        path = self.edit_timeline.text().strip()

        def done(cel):
            k = json.loads((Path(cel) / 'manifest.json').read_text(encoding='utf-8')).get('converted_from', {})
            if k:
                self.status.setText(f'Kopia 16-bitowa: {Path(cel).name}, {k["bytes_before"] / 1e9:.2f} GB -> '
                                    f'{k["bytes_after"] / 1e9:.2f} GB. Oryginał został. Wczytuję kopię.')
            else:
                self.status.setText('Ten zapis już jest 16-bitowy. Wczytuję go.')
            self.edit_timeline.setText(str(cel))
            self._load_timeline()
        self._job(lambda p, c: R.konwertuj_q16(path, progress=p, cancel=c), done)

    def _vram_budget(self):
        return int(self.spin_vram.value() * (1 << 30))

    def _loaded(self, result):
        timeline, first, second = result
        self.timeline = timeline
        steps = timeline.steps
        natural = timeline.metadata.get('time_axis') == 'training_checkpoint'
        self.info.setText('Naturalny ruch cząsteczek · kolejne zapisane modele · bez podmiany aktywacji'
                          if natural else 'Sztuczna korekta cząsteczek · przy zamrożonych wagach ViT')
        self.curve.setLabel('bottom', 'krok treningu zapisanej sieci' if natural else 'krok optymalizacji latentu')
        self.chk_movement.setText(f'Ruch względem pierwszego kroku ({steps[0]})')
        self._set_logs(timeline.metadata.get('training_logs') if natural else None)
        self.heat.clear()
        gotowe = {steps[0]: first}
        if len(steps) > 1:
            gotowe[steps[1]] = second
        self.cloud.ustaw_os(timeline, gotowe=gotowe, budzet=self._vram_budget())
        bounds = []
        for movement in (False, True):
            try:
                bounds.append(timeline.bounds(movement))
            except ValueError:
                bounds.append(None)
        self.cloud.ustaw_granice_manifestu(*bounds)
        self._visual_step = float(steps[0]); self._last_stats = None; self._last_index = None
        self._gpu_stats = None; self._pending_heat = None
        self.cloud.setVisible(True)
        self.suwak.blockSignals(True); self.suwak.setRange(0, len(steps) - 1); self.suwak.setValue(0)
        self.suwak.blockSignals(False); self.suwak.setEnabled(True)
        self.spin_image.setMaximum(max(timeline.image_count - 1, 0))
        self.btn_play.setEnabled(len(steps) > 1)
        curves = timeline.curves()                      # rzeczywiste metryki producenta: ustawiane raz
        for key, curve in [('confidence', self.conf_curve), ('iou', self.iou_curve)]:
            values = curves.get(key, [])
            x = curves.get('steps', steps if len(values) == len(steps) else list(range(len(values))))
            curve.setData(x, values)
        self._display_options()
        self._provenance()
        self._budz()

    def _provenance(self):
        if self.timeline is None:
            return
        sha = self.timeline.checkpoint_sha256
        text = f'{self.timeline.image_count} obrazów · {self.timeline.count:,} cząsteczek · {len(self.timeline.steps)} zapisanych stanów. '
        if self.timeline.metadata.get('time_axis') == 'training_checkpoint':
            text += (f'Naturalny trening, checkpointy {self.timeline.steps[0]}–{self.timeline.steps[-1]}. '
                     'Pomiary z niezmienionych modeli; interpolacja tylko w animacji.')
            state = self.timeline.metadata.get('status', 'unknown')
            if state != 'complete':
                state_label = {'running': 'pomiar w toku', 'cancelled': 'pomiar przerwany',
                               'failed': 'pomiar zakończony błędem', 'preparing': 'przygotowanie'}.get(state, state)
                text += (f' Zapis częściowy ({state_label}): {len(self.timeline.steps)} z '
                         f'{len(self.timeline.metadata.get("checkpoints", []))} wybranych modeli. '
                         'Wczytaj ponownie, aby odczytać nowo ukończone stany.')
            self.status.setText(text); return
        if self._active_checkpoint and sha and sha != self._active_checkpoint['sha256']:
            text += 'Uwaga: zapis ruchu pochodzi z innego checkpointu niż wczytany ViT. '
        else:
            text += 'Gotowy zapis zachowuje checkpoint swojego producenta. '
        self.status.setText(text + 'Zmiana checkpointu nie przelicza istniejących danych; użyj „Oblicz nowy ruch”.')

    # ------------------------------------------------------------------ odtwarzanie: tylko parametry dla karty
    def _indices(self):
        steps = self.timeline.steps
        i = min(max(bisect.bisect_right(steps, self._visual_step) - 1, 0), len(steps) - 1)
        return i, min(i + 1, len(steps) - 1)

    def _scope(self):
        tpi = self.cloud.tpi or 600
        if self.combo_scope.currentIndex() == 0:
            return 0, self.timeline.count
        return min(self.spin_image.value(), max(self.timeline.image_count - 1, 0)) * tpi, tpi

    def _pair_ready(self, a, b):
        return all(self.cloud.gotowy(s) for s in (self.timeline.steps[0], a, b))

    def _render(self):
        """Przekazuje karcie parametry klatki; etykiety i statystyki tylko przy zmianie. Zwraca: para gotowa."""
        if not self.timeline:
            return False
        steps = self.timeline.steps
        i, j = self._indices(); a, b = steps[i], steps[j]
        fraction = 0.0 if a == b else (self._visual_step - a) / (b - a)
        s = fraction if self.chk_smooth.isChecked() else 0.0
        first, count = self._scope()
        self.cloud.ustaw_klatke(a, b, s, self.chk_movement.isChecked(), self.combo_color.currentData(),
                                self.combo_kadr.currentData(), first, count, self.spin_size.value(),
                                self.spin_opacity.value(), self.spin_trail.value())
        if i != self._last_index:
            self.suwak.blockSignals(True); self.suwak.setValue(i); self.suwak.blockSignals(False)
            self._last_index = i
        if not self._pair_ready(a, b):
            self._set_text(self.frame_label, f'Wczytywanie zapisanych stanów {a} i {b} na kartę… '
                           f'({self.cloud.na_karcie()}/{len(steps)} stanów na karcie)')
            self._mark_unsaved(False)
            self._budz()
            return False
        now = time.perf_counter()
        etykiety = not self._playing or now - self._last_label_time > .05     # w ruchu napisy do 20 razy/s
        if etykiety:
            self._last_label_time = now
            self._set_text(self.step_label, f'Krok {self._visual_step:.2f} / {steps[-1]}')
        if 0 < s < 1:
            if etykiety:
                self._set_text(self.frame_label, f'STAN NIEZAPISANY · liniowe przejście na GPU między zapisami '
                               f'{a} → {b} (s = {s:.2f}). Zapisane są tylko stany {a} i {b}; metryki i mapa ze stanu {a}.')
            self._mark_unsaved(True)
        else:
            shown = b if s >= 1 else a
            extra = ' (płynne przejścia wyłączone)' if not self.chk_smooth.isChecked() and fraction > 0 else ''
            self._set_text(self.frame_label, f'Zapisany stan {shown}: położenia i normy bitowo z pliku{extra}.')
            self._mark_unsaved(False)
        if a != self._last_stats and (not self._playing or now - self._last_stats_time > .1):
            self._show_stats(a, first, count)
            self._last_stats_time = now
        return True

    def _show_stats(self, a, first, count):
        """Metryki producenta (zapis) od razu; mapa i mediana normy 768D przychodzą z karty (_gpu_results)."""
        stats = self.timeline.stats(a)
        confidence, iou = stats.get('confidence'), stats.get('iou')
        conf_text = 'brak pomiaru' if confidence is None else f'{confidence:.2%}'
        iou_text = 'brak pomiaru' if iou is None else f'{iou:.2%}'
        cfg = self.timeline.metadata.get('config', {})
        thresholds = f'{cfg.get("target_confidence", .9):.0%} / {cfg.get("target_iou", .9):.0%}'
        gpu = ''
        g = self._gpu_stats
        if g and g['krok'] == a and g['first'] == first and g['count'] == count:
            gpu = (f'\nRuch 768D w widoku (karta, stan {a}): mediana {g["mediana"]:.4g} · maks. {g["maksimum"]:.4g}')
        self.metrics.setText(f'Pomiary ze stanu {a}\nŚrednia pewność: {conf_text}\n'
                             f'Średnie IoU: {iou_text}\n'
                             f'Oba cele ({thresholds}): {stats.get("success_count", stats.get("both90", "—"))} / {self.timeline.image_count}\n'
                             + ('Naturalny forward checkpointu, bez podmiany aktywacji.'
                                if self.timeline.metadata.get('time_axis') == 'training_checkpoint'
                                else 'Korekta z etykietą, przy zamrożonym detektorze.') + gpu)
        self.cursor.setValue(a); self.log_cursor.setValue(a)
        self._last_stats = a
        self.cloud.zadaj_statystyki(a, first, count)
        self._budz()

    def _gpu_results(self, wynik):
        """Wyniki z karty: mapa ruchu (średnia po obrazach albo jeden obraz) i dokładna mediana normy 768D."""
        if self.timeline is None or wynik['klucz'][3] != self.cloud._generacja:
            return
        self._gpu_stats = wynik
        if wynik['krok'] == self._last_stats:
            self._last_stats = None                       # odśwież tekst metryk z liczbami z karty
            if not self._playing:
                self._render()
        mapa = np.asarray(wynik['mapa'], np.float32)
        if mapa.size == 600:
            self._pending_heat = (wynik['krok'], mapa.reshape(20, 30))
            self._apply_heat()

    def _apply_heat(self, force=False):
        if self._pending_heat is None:
            return
        now = time.perf_counter()
        if not force and self._playing and now - getattr(self, '_last_heat_time', 0.0) < .2:
            return
        krok, mapa = self._pending_heat
        self._pending_heat = None
        self._last_heat_time = now
        if self._playing:                                  # w ruchu tylko obraz (bez histogramu i osi ImageView)
            self.heat.getImageItem().setImage(mapa.T, levels=(float(mapa.min()), float(mapa.max()) or 1.0))
        else:
            self.heat.setImage(mapa.T, autoLevels=True)
        self._set_text(self.heat_label, f'Ruch 768D per patch · stan {krok} · '
                       + ('średnia po obrazach' if self.combo_scope.currentIndex() == 0 else 'jeden obraz')
                       + ' · liczone na karcie')

    def _display_options(self, *_):
        if self.timeline:
            self._last_stats = None
            names = self.timeline.image_names
            i = self.spin_image.value()
            self.image_name.setText(names[i] if i < len(names) else f'Obraz {i}')
            self._render()
            self._budz()

    def _vram_changed(self, *_):
        self.cloud.ustaw_budzet(self._vram_budget())
        self._budz()

    def _pause(self):
        self._playing = False; self._last_tick = None; self.btn_play.setText('Odtwórz')
        if self.timeline is not None:
            self._last_stats = None

    def _toggle_play(self):
        if not self.timeline:
            return
        if self._playing:
            self._pause(); self._render(); self._apply_heat(force=True); return
        if self._visual_step >= self.timeline.steps[-1]:
            self._seek_index(0)
        self._playing = True; self._last_tick = time.perf_counter(); self.btn_play.setText('Pauza')
        self._budz()

    def _seek_index(self, index):
        self._pause()
        if self.timeline:
            index = max(0, min(int(index), len(self.timeline.steps) - 1))
            self._visual_step = float(self.timeline.steps[index]); self._render()
            self._budz()

    def _advance(self, dt):
        """Przesuwa czas animacji; przekracza kolejne zapisy w jednej klatce tylko, gdy są już na karcie."""
        steps = self.timeline.steps
        remaining = dt * self.spin_tempo.value()
        while remaining > 0:
            i, j = self._indices()
            if i == j:
                break
            a, b = steps[i], steps[j]
            if not self._pair_ready(a, b):
                break
            gap = b - self._visual_step
            if remaining < gap:
                self._visual_step += remaining
                break
            remaining -= gap
            self._visual_step = float(b)
        if self._visual_step >= steps[-1]:
            self._visual_step = float(steps[-1])
            self._pause()

    def _budz(self):
        """Włącza zegar klatek, gdy jest coś do zrobienia; _tick sam go wyłącza w bezczynności."""
        if not self._shutting_down and self.timeline is not None and not self.timer.isActive():
            self._last_tick = None
            self.timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        self._budz()

    def _tick(self):
        if self._shutting_down or self.timeline is None or not self.isVisible():
            self.timer.stop(); self._last_tick = None
            return
        now = time.perf_counter()
        dt = min(now - (self._last_tick or now), .1); self._last_tick = now
        self.cloud.pompuj()
        if self.cloud.blad_odczytu:
            self._pause()
            self._error('odczyt stanu osi czasu: ' + self.cloud.blad_odczytu)
            self.cloud.blad_odczytu = ''
        if self._playing:
            self._advance(dt)
        if self.chk_spin.isChecked():
            self.view.orbit(dt * 12, 0)
        self._render()
        if self.cloud.blad:
            self._pause(); self._set_text(self.gpu_status, 'Błąd GPU: ' + self.cloud.blad)
        elif now - self._last_status_time > .25:
            self._last_status_time = now
            self._status_gpu()
        if not self._playing:
            self._apply_heat(force=True)
        if not (self._playing or self.chk_spin.isChecked() or self.cloud.zajety()):
            self.timer.stop(); self._last_tick = None
            self._status_gpu()

    def _status_gpu(self):
        c = self.cloud
        mode = 'cała oś na karcie' if c.cala_os else f'pierścień {c.n_slotow} slotów'
        self._set_text(self.gpu_status, f'GPU · {c.draw_count:,} punktów · stany na karcie {c.na_karcie()}/{c.T} '
                       f'({mode}) · {c.pamiec_karty() / 1e6:.0f} MB VRAM · {c.fps:.0f} kl./s · {c.renderer}')

    # ------------------------------------------------------------------ wybór cząsteczki (liczy karta)
    def _pick(self, x, y):
        if not self.timeline:
            return
        self.cloud.zadaj_wybor(x, y, self.view.width(), self.view.height())
        self._budz()

    def _picked(self, index):
        if index == -2:                                   # „Odznacz cząsteczkę”
            self.cloud.ustaw_wybrany(None)
            if self.timeline:
                self._display_options()
            return
        if index < 0 or self.timeline is None:
            self._set_text(self.status, 'Kliknięcie: brak cząsteczki w promieniu 12 px.')
            return
        tpi = self.cloud.tpi or 600
        image, token = divmod(index, tpi)
        name = self.timeline.image_names[image] if image < len(self.timeline.image_names) else f'obraz {image}'
        patch = f'({token // 30}, {token % 30})' if tpi == 600 else ''
        self.image_name.setText(f'Obraz {image} · patch {token} {patch} · {name}')
        self.cloud.ustaw_wybrany(index)

    def _save_png(self):
        name, _ = QFileDialog.getSaveFileName(self, 'Zapisz podgląd', str(R.OUTPUT_ROOT / 'podglad.png'), 'PNG (*.png)')
        if name:
            self.view.grabFramebuffer().save(name)
            self.status.setText(f'Zapisano podgląd: {name}')

    def shutdown(self):
        self._pause()
        if self.watek and self.watek.isRunning():
            self._cancel_job()
            return False
        self._shutting_down = True
        self.timer.stop()
        self.cloud.zamknij()
        if self.view.isValid():
            self.view.makeCurrent(); self.cloud.release(); self.view.doneCurrent()
        return True
