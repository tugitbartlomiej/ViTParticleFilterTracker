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
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget)

import ruch_zbioru as R
import ruch_naturalny as N
from ruch_gpu import ChmuraRuchuGPU
from styl import rola
from uklad import Odpinanie, PanelOpcji, dodaj as _dodaj, dodaj_nawigacje
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


def staly_napis(label, linie):
    """Napis o stałej wysokości (linie tekstu) i bez wpływu na szerokość układu: zmiana treści co klatkę
    odtwarzania nie przesuwa podziału okna ani nie zmienia wysokości wykresów pod spodem."""
    from PyQt6.QtWidgets import QSizePolicy
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    label.setFixedHeight(label.fontMetrics().lineSpacing() * linie + 4)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return label


def pole_wyboru(row, tekst, opis, zaznaczone=False):
    c = QCheckBox(tekst)
    c.setChecked(zaznaczone)
    c.setToolTip(opis)
    row.addWidget(c)
    return c


def okno_wyboru_zdjec(rodzic, zrodlo, wybrane):
    """Okno z listą zdjęć źródła (pola wyboru, filtr nazwy i filmu, losowanie, podgląd). Zwraca indeksy albo None."""
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QSpinBox, QVBoxLayout
    import zbiory_zdjec as Z
    d = QDialog(rodzic); d.setWindowTitle(f'Wybór zdjęć: {Z.ZRODLA[zrodlo]}'); d.resize(820, 620)
    lay = QVBoxLayout(d)
    gora = QHBoxLayout(); lay.addLayout(gora)
    filtr = QLineEdit(); filtr.setPlaceholderText('filtr: fragment nazwy albo filmu, np. TR_0005')
    filtr.setToolTip('Pokazuje tylko zdjęcia, których nazwa albo film zawiera wpisany tekst. Zaznaczenia ukrytych zostają.')
    gora.addWidget(filtr, 1)
    n_los = QSpinBox(); n_los.setRange(1, 5000); n_los.setValue(100)
    n_los.setToolTip('Ile zdjęć wylosować przyciskiem „Losuj” (spośród widocznych po filtrze).')
    b_los = QPushButton('Losuj'); b_los.setToolTip('Zastępuje zaznaczenie losowymi widocznymi zdjęciami (ziarno 2026).')
    b_zazn = QPushButton('Zaznacz widoczne'); b_zazn.setToolTip('Zaznacza wszystkie zdjęcia widoczne po filtrze.')
    b_odzn = QPushButton('Odznacz widoczne'); b_odzn.setToolTip('Odznacza wszystkie zdjęcia widoczne po filtrze.')
    for w in (QLabel('losowych'), n_los, b_los, b_zazn, b_odzn):
        gora.addWidget(w)
    srodek = QHBoxLayout(); lay.addLayout(srodek, 1)
    lista = QListWidget(); lista.setToolTip('Zaznacz pola zdjęć, dla których liczyć trajektorie cząsteczek.')
    srodek.addWidget(lista, 2)
    podglad = QLabel('Podgląd: kliknij zdjęcie.'); podglad.setMinimumSize(320, 214)
    podglad.setAlignment(Qt.AlignmentFlag.AlignCenter); podglad.setToolTip('Podgląd zaznaczonego wiersza listy.')
    srodek.addWidget(podglad, 1)
    licznik = QLabel(); lay.addWidget(licznik)
    przyciski = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    lay.addWidget(przyciski)
    wybrane = set(wybrane)
    opisy = Z.opisy(zrodlo)
    lista.setUpdatesEnabled(False)
    for i, nazwa, film in opisy:
        it = QListWidgetItem(f'{i:5d}  {nazwa}')
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked if i in wybrane else Qt.CheckState.Unchecked)
        it.setData(Qt.ItemDataRole.UserRole, i)
        lista.addItem(it)
    lista.setUpdatesEnabled(True)

    def policz():
        n = sum(lista.item(r).checkState() == Qt.CheckState.Checked for r in range(lista.count()))
        licznik.setText(f'Zaznaczone: {n} z {lista.count()} zdjęć ({n * 600:,} cząsteczek).')

    def filtruj(t):
        t = t.strip().lower()
        for r, (i, nazwa, film) in enumerate(opisy):
            lista.item(r).setHidden(bool(t) and t not in nazwa.lower() and t not in film.lower())

    def widoczne():
        return [r for r in range(lista.count()) if not lista.item(r).isHidden()]

    def ustaw(wiersze, stan):
        lista.blockSignals(True)
        for r in wiersze:
            lista.item(r).setCheckState(stan)
        lista.blockSignals(False); policz()

    def losuj():
        import numpy as np
        w = widoczne()
        los = set(np.random.default_rng(2026).choice(w, min(n_los.value(), len(w)), replace=False).tolist()) if w else set()
        ustaw(range(lista.count()), Qt.CheckState.Unchecked)
        ustaw(sorted(los), Qt.CheckState.Checked)

    def pokaz(it):
        if it is None:
            return
        try:
            i = it.data(Qt.ItemDataRole.UserRole)
            r = Z.wiersze(zrodlo)[i]
            if zrodlo == '1500':
                raw = (R._IMAGE_ROOT / r['image_name']).read_bytes()
            else:
                raw = Z._wypakuj(r).read_bytes()
            img = QImage.fromData(raw)
            podglad.setPixmap(QPixmap.fromImage(img).scaled(podglad.width(), podglad.height(),
                                                            Qt.AspectRatioMode.KeepAspectRatio,
                                                            Qt.TransformationMode.SmoothTransformation))
        except Exception as e:                       # brak pliku albo zipu: tylko komunikat w podglądzie
            podglad.setText(f'Podgląd niedostępny: {e}')

    filtr.textChanged.connect(filtruj)
    b_los.clicked.connect(losuj)
    b_zazn.clicked.connect(lambda: ustaw(widoczne(), Qt.CheckState.Checked))
    b_odzn.clicked.connect(lambda: ustaw(widoczne(), Qt.CheckState.Unchecked))
    lista.itemChanged.connect(lambda _: policz())
    lista.currentItemChanged.connect(lambda a, _b: pokaz(a))
    przyciski.accepted.connect(d.accept); przyciski.rejected.connect(d.reject)
    policz()
    if d.exec() != QDialog.DialogCode.Accepted:
        return None
    return [lista.item(r).data(Qt.ItemDataRole.UserRole) for r in range(lista.count())
            if lista.item(r).checkState() == Qt.CheckState.Checked]


def _napis_na_klatce(a, tekst):
    """Tekst (wiele linii) na półprzezroczystym pasku u góry klatki; czcionka Segoe UI, gdy jest w systemie."""
    if not tekst:
        return a
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(a).convert('RGBA')
    try:
        font = ImageFont.truetype('segoeui.ttf', max(14, im.width // 70))
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(im, 'RGBA')
    linie = str(tekst).split('\n')
    wys = sum(d.textbbox((0, 0), l, font=font)[3] + 6 for l in linie) + 12
    d.rectangle((0, 0, im.width, wys), fill=(10, 14, 18, 190))
    y = 8
    for l in linie:
        d.text((12, y), l, font=font, fill=(240, 240, 240, 255))
        y += d.textbbox((0, 0), l, font=font)[3] + 6
    return np.asarray(im.convert('RGB'))


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
        import historie_checkpointow as HC
        self.combo_model = QComboBox()
        self.combo_model.addItem('własny plik (pole „ViT” wyżej)', ('', None))
        for klucz, nazwa, plik, krok, opis in HC.MODELE:
            self.combo_model.addItem(nazwa, ('model', klucz))
        for h in HC.HISTORIE:
            self.combo_model.addItem(f'historia: {h.nazwa}', ('historia', h.klucz))
        dodaj(row, 'model', self.combo_model,
              'Gotowe modele z folderu assets zgodne z detektorem (MalyViT 600 × 768) albo checkpoint z zapisanej historii '
              'uczenia. Wybór wpisuje ścieżkę do pola „ViT”; „Sprawdź i wczytaj ViT” ładuje go i liczy SHA256. '
              + ' '.join(f'{n}: {o}' for _, n, _, _, o in HC.MODELE))
        self.combo_model_krok = QComboBox(); self.combo_model_krok.setMinimumWidth(190); self.combo_model_krok.setEnabled(False)
        dodaj(row, 'krok', self.combo_model_krok, 'Krok albo epoka checkpointu, gdy wybrano historię uczenia '
              '(K1000: co 10 kroków; 35 000 klatek: końce epok i siatka co 10/100 kroków).')
        self.model_opis = rola(QLabel(''), 'opis'); row.addWidget(self.model_opis)
        self.combo_model.currentIndexChanged.connect(lambda _: self._wybor_modelu())
        self.combo_model_krok.currentIndexChanged.connect(lambda _: self._wybor_kroku_modelu())
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
        self.combo_szklo = QComboBox(); self.combo_szklo.setMinimumWidth(260)
        self._wypelnij_osie_szkla()
        dodaj(row, 'oś szkła', self.combo_szklo,
              'Osie czasu „układu szkła” z outputs/ruch_czasteczek_gui (foldery ze „szklo” w nazwie): ruch tokenów względem '
              'klatki, czyli minus średnie przesunięcie 6 sąsiadów w latencie (z gęstego zapisu biegu 35 000, co 2 kroki do '
              '2188 i co 50 dalej), oraz symulacje szkła koloidalnego (dynamika browna miękkich kul: stała temperatura i '
              'stygnięcie własne jak tokeny). „Odśwież” czyta listę z dysku na nowo.')
        self.btn_szklo_odswiez = przycisk(row, 'Odśwież', 'Czyta na nowo listę osi szkła z dysku (po eksporcie albo symulacji '
                                          'uruchomionej, gdy aplikacja była otwarta).')
        self.btn_szklo = przycisk(row, 'Wczytaj oś szkła',
                                  'Wstawia wybraną oś szkła do pola „zapis ruchu” i od razu ją wczytuje (jak „Wczytaj ruch”).')

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
        self.btn_natural_35k = przycisk(row, 'Historia 35 000 · epoki',
                                        'Ustawia historię małego ViT uczonego od zera na 35 000 klatkach (21 epok, potem do 60) '
                                        '(experiments/maly_vit_od_zera_35000_21epok_20261009, bieg na Edenie): ten sam '
                                        'manifest 1500 obrazów, checkpoint co 1094 kroki = koniec każdej epoki (zmień '
                                        '„checkpoint co” na 100 albo 10, jeśli pobrano gęste kroki). Tylko wczytuje kolejne '
                                        'zapisane modele i mierzy cząsteczki, bez uczenia. Wagi leżą w '
                                        'assets/MalyViT_35000_od_zera_s47 (skrypt pobierz_z_edena.sh obok).')
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
        self._wiersze_trajektorii()
        note = QLabel('Te same obrazy i etykiety wybierz w „Checkpoint i dane”. Urządzenie CUDA/CPU i ziarno PCA '
                      'są w „Optymalizacja”. Każdy zapisany model wykonuje naturalny forward w trybie eval. '
                      'Nie trenujemy ponownie ani nie podmieniamy aktywacji. Brakujące kroki nie są rekonstruowane; '
                      'płynne przejścia służą tylko animacji.')
        note.setWordWrap(True); self.opcje._strony['Uczenie naturalne'].addWidget(rola(note, 'opis'))

        row = self.opcje.wiersz('Animacja')
        self.spin_tempo = self._float(row, 'tempo (kroki/s)', 1, 1000, 50, 1,
                                      'Szybkość odtwarzania w krokach osi czasu na sekundę. Animacja nie przeskakuje '
                                      'stanów, których jeszcze nie ma na karcie.')
        self.spin_fps = self._int(row, 'film: kl./s', 5, 60, 30, 'Liczba klatek na sekundę nagrywanego filmu MP4.')
        self.btn_film = przycisk(row, 'Nagraj film…',
                                 'Nagrywa odtwarzanie zakresu „od–do” do pliku MP4 z bieżącymi ustawieniami widoku '
                                 '(układ, kolory, kamera, wygładzanie, atencja) i tempem z pola „tempo (kroki/s)”. Klatki '
                                 'renderuje karta w rozdzielczości widoku; każda klatka czeka, aż jej stany są na karcie, '
                                 'więc film nie przeskakuje zapisów. Kodowanie H.264 przez ffmpeg (gdy jest w systemie), '
                                 'inaczej mp4v.')
        self.chk_smooth = pole_wyboru(row, 'Płynne przejścia GPU',
                                      'Włączone: między zapisami karta liczy liniowe przejście (stan NIEZAPISANY, '
                                      'oznaczony nad widokiem). Wyłączone: widać tylko zapisane stany.', True)
        self.combo_wyg = QComboBox()
        self.combo_wyg.addItem('bez wygładzania', 0)
        self.combo_wyg.addItem('średnia ruchoma (wyśrodkowana)', 1)
        self.combo_wyg.addItem('EMA (wygładzanie wykładnicze)', 2)
        dodaj(row, 'wygładzanie ruchu', self.combo_wyg,
              'Wygładza tory cząsteczek w czasie, po zapisanych stanach (liczy karta, we wszystkich układach, także w '
              'śladach, siatce zdjęcia, przestrzeni fazowej i Koopmanie):\n'
              '- średnia ruchoma: położenie w zapisie k = średnia z „okna” zapisów wokół k (wyśrodkowana, więc ruch się\n'
              '  nie opóźnia; na brzegach osi okno jest przycięte);\n'
              '- EMA: wygładzone(k) = α · stan(k) + (1 − α) · wygładzone(k − 1), od pierwszego zapisu (jak w Uczenie ViT);\n'
              '  α = 1 bez wygładzania, 0,3 pamięć około 3 zapisów; opóźnia ruch.\n'
              'Wygładzony stan nie jest zapisem: napis nad widokiem to zaznacza. Mapa ruchu, mediany i skale kolorów\n'
              'liczą się dalej z zapisanych stanów.')
        self.spin_wyg_okno = self._int(row, 'okno (zapisy)', 3, 41, 5,
                                       'Długość okna średniej ruchomej w zapisach (nieparzysta: tyle samo zapisów przed '
                                       'i po bieżącym; parzysta jest zaokrąglana w górę do nieparzystej).')
        self.spin_wyg_alfa = self._float(row, 'α EMA', .02, 1, .3, 2,
                                         'Współczynnik EMA: waga bieżącego zapisu. Mniejszy = mocniejsze wygładzenie '
                                         'i większe opóźnienie (pamięć około 1/α zapisów).')
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
        self.combo_uklad = QComboBox()
        self.combo_uklad.addItem('PCA 3D: wspólna baza cech', 0)
        self.combo_uklad.addItem('Siatka zdjęcia + cecha na osi Z', 1)
        self.combo_uklad.addItem('Przestrzeń fazowa: cecha, jej prędkość, norma 768D', 2)
        self.combo_uklad.addItem('Koopman: niezmienniki (X, Y) i mod najbardziej dynamiczny (Z)', 3)
        self.combo_uklad.addItem('Fizyka ruchu: temperatura, rodzaj dyfuzji, zmiana 768D (strona „Fizyka ruchu”)', 4)
        dodaj(row, 'układ', self.combo_uklad,
              'W jakiej przestrzeni oglądać cząsteczki (liczy karta, w każdej klatce):\n'
              '- PCA 3D: położenie w trzech głównych składowych wspólnej bazy (dotychczasowy widok);\n'
              '- siatka zdjęcia + cecha na Z: X i Y to kolumna i wiersz patcha na zdjęciu (stałe), a Z to wybrana\n'
              '  cecha, czyli rzut wszystkich 768 cech. Ruch cząsteczki widać jako wznoszenie i opadanie nad swoim\n'
              '  miejscem na obrazie, więc od razu wiadomo, które części zdjęcia się zmieniają;\n'
              '- przestrzeń fazowa: X = cecha, Y = jej prędkość między zapisami, Z = norma zmiany w 768D. Drgania\n'
              '  dają pętle, hamowanie spiralę do Y = 0, a stały dryf poziomą smugę (ślad wyłączony);\n'
              '- Koopman: X, Y = funkcje własne operatora Koopmana o wartości własnej bliskiej 1 (niezmienniki: prawie\n'
              '  stałe wzdłuż toru, grupują cząsteczki o podobnym losie), Z = funkcja własna modu najbardziej dynamicznego.\n'
              '  Operator dopasowuje karta (Hankel-DMD) do wszystkich cząsteczek; opcje na stronie „Koopman”.')
        self.combo_cecha = QComboBox()
        for nazwa, opis in (('PC1', 0), ('PC2', 1), ('PC3', 2), ('norma zmiany 768D', 3), ('odległość od środka PCA', 4)):
            self.combo_cecha.addItem(nazwa, opis)
        dodaj(row, 'cecha', self.combo_cecha,
              'Która liczba z 768 cech cząsteczki trafia na oś Z siatki (i na oś X przestrzeni fazowej):\n'
              '- PC1, PC2, PC3: rzut całego wektora 768 cech na daną oś wspólnej bazy PCA (PC1 = największy rozrzut);\n'
              '  przy „Ruch względem pierwszego kroku” liczona jest zmiana od pierwszego zapisu;\n'
              '- norma zmiany 768D: długość zmiany całego wektora cech od pierwszego zapisu (zapis producenta);\n'
              '- odległość od środka PCA: długość położenia w trzech składowych.')
        self.chk_bez_poz = pole_wyboru(row, 'Bez położenia patcha',
                                       'Od cechy odejmowana jest ta sama cecha średniej tego patcha po wszystkich obrazach '
                                       'w tym samym kroku (liczy karta). Usuwa to, co w latencie wynika z samego miejsca '
                                       'patcha na obrazie (kodowanie pozycji ViT, typowy układ sceny), bo w siatce zdjęcia '
                                       'położenie pokazują już osie X i Y. Zostaje to, czym obraz różni się od innych w tym '
                                       'miejscu. Działa w siatce zdjęcia i w przestrzeni fazowej. Przy „Ruch względem '
                                       'pierwszego kroku” od zmiany odejmowana jest średnia zmiana patcha.', True)
        self.combo_mozaika = QComboBox()
        self.combo_mozaika.addItem('Wszystkie obrazy na jednej siatce', False)
        self.combo_mozaika.addItem('Mozaika: każdy obraz w osobnym kaflu', True)
        dodaj(row, 'obrazy na siatce', self.combo_mozaika,
              'Tylko dla siatki zdjęcia. Jedna siatka: 1500 obrazów nakłada się na ten sam układ 20 × 30, więc nad każdym '
              'patchem stoi słup 1500 cząsteczek (rozrzut i ruch tego miejsca w całym zbiorze). Mozaika: każdy obraz '
              'dostaje własny kafel 30 × 20 obok pozostałych (krajobraz całego zbioru).')
        self.combo_color = QComboBox()
        self.combo_color.addItem('Norma zmiany 768D (zapis)', 0)
        self.combo_color.addItem('Prędkość w PCA między zapisami', 1)
        self.combo_color.addItem('|Δ| w PCA od pierwszego kroku', 2)
        self.combo_color.addItem('Położenie patcha (kolumna = barwa, wiersz = jasność)', 3)
        self.combo_color.addItem('Narzędzie (czerwone) i tło (szare) wg ramki GT', 4)
        self.combo_color.addItem('Pewność ViT obrazu w tym kroku', 5)
        self.combo_color.addItem('Numer obrazu (osobna barwa)', 6)
        self.combo_color.addItem('Klaster Koopmana (po „Osie pod klastry”; szare = outliery)', 7)
        self.combo_color.addItem('Temperatura: energia drgań własnych (inferno, okno ze strony „Fizyka ruchu”)', 8)
        self.combo_color.addItem('Reżim ruchu: balistyczny (żółty) / dyfuzyjny (niebieski) / uwięziony (fiolet)', 9)
        self.combo_color.addItem('Wykładnik MSD α (viridis: 0 uwięzienie … 1 dyfuzja)', 10)
        self.combo_color.addItem('Obszar oka z segmentacji: narzędzie / źrenica / rogówka / tło', 11)
        self.combo_color.addItem('Przyspieszenie chwilowe |v₁ − v₀| w PCA (zmiana prędkości wokół zapisu, inferno)', 12)
        self.combo_color.addItem('Prędkość chwilowa w PCA (różnica centralna wokół zapisu, inferno)', 13)
        self.combo_color.addItem('Prędkość chwilowa względem średniej tej cząsteczki (niebieski wolniej, czerwony zryw)', 14)
        self.combo_color.addItem('Energia cząsteczki: kinetyczna / potencjalna / całkowita (strona „Fizyka ruchu”)', 15)
        self.combo_color.addItem('Udział energii ruchu w paśmie okresów (strona „Fizyka ruchu”, inferno)', 16)
        self.combo_color.addItem('Zakresy właściwości: do 3 zakresów z kolorami (okres, temperatura, energia…)', 17)
        dodaj(row, 'kolor', self.combo_color,
              'Co koduje kolor punktu (liczy karta): zapisana norma zmiany w pełnych 768D; prędkość w rzucie PCA '
              'między sąsiednimi zapisami; długość przesunięcia w rzucie PCA od pierwszego kroku (skala: maksimum po '
              'zapisanych krokach); położenie patcha na zdjęciu (jak w przestrzeni makrowektorów); patche w ramce GT '
              'narzędzia; pewność detekcji ViT obrazu w bieżącym zapisanym kroku (viridis, z metrics.npy pomiaru); '
              'numer obrazu (cząsteczki jednego obrazu mają wspólną barwę); klaster Koopmana z przycisku „Osie pod klastry” '
              '(najbliższy środek klastra, szare = outliery; tylko w układzie Koopmana i tylko dla osi, dla których '
              'dobrano klastry); przyspieszenie chwilowe: |v₁ − v₀|, gdzie v₀ i v₁ to prędkości w rzucie PCA na '
              'odcinkach do i od zapisu najbliższego bieżącej chwili (na krok treningu, nierówne odstępy uwzględnione; '
              'z wygładzaniem, gdy włączone); skala: pierwiastek z ułamka 2 × maksymalnej prędkości. Prędkość chwilowa: '
              '|x(k+1) − x(k−1)| / odstęp kroków wokół zapisu k najbliższego bieżącej chwili. Prędkość względna: ta sama '
              'prędkość podzielona przez średnią prędkość tej cząsteczki po wszystkich odcinkach osi (log2: biały = typowo, '
              'czerwony = 2–4 razy szybciej niż zwykle, niebieski = wolniej).')
        self.spin_moc = self._float(row, 'moc koloru', 0, 3, 1, 2,
                                    'Siła kolorowania punktów (liczy shader, dla każdego trybu koloru): 0 = same '
                                    'odcienie szarości (tylko jasność barwy), 1 = barwy bez zmian, powyżej 1 = barwy '
                                    'mocniej nasycone (np. klastry wyraźniej odcinają się od szarych outlierów). '
                                    'Działa też na kolorach śladów i siatki zdjęcia.')
        self.spin_moc.setSingleStep(0.1)
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

        self.opcje.dodaj_strone('Koopman')
        row = self.opcje.wiersz('Koopman')
        self.spin_kd = self._int(row, 'opóźnienia', 1, 4, 3,
                                 'Zanurzenie opóźnieniowe (Hankel-DMD): stan kroku to d ostatnich zapisów (PC1, PC2, PC3, '
                                 'norma 768D), czyli 4 · d liczb. Więcej opóźnień = więcej modów i lepiej widoczne '
                                 'oscylacje, ale potrzeba d zapisów wstecz (na początku osi czasu powtarzany jest pierwszy).')
        self.combo_kxy = QComboBox()
        self.combo_kxy.addItem('zespolony niezmiennik najbliższy λ = 1: X = Re, Y = Im', 0)
        self.combo_kxy.addItem('dwa mody najbliższe λ = 1: części rzeczywiste', 1)
        self.combo_kxy.addItem('niezmiennik dodatni (λ ≈ +1) na X i ujemny (λ ≈ −1) na Y', 2)
        self.combo_kxy.setCurrentIndex(1)                     # w pomiarach naturalnych niezmienniki są rzeczywiste
        dodaj(row, 'X, Y', self.combo_kxy,
              'Które funkcje własne Koopmana trafiają na osie X i Y. Wartość własna bliska 1 oznacza wielkość prawie\n'
              'stałą wzdłuż toru cząsteczki (niezmiennik), więc położenie w X, Y mówi, do jakiej rodziny ruchów należy\n'
              'cząsteczka. Zespolony niezmiennik obraca się powoli: część rzeczywista i urojona to jego dwie osie.\n'
              'λ ≈ −1: wielkość, która co zapis zmienia znak, ale zachowuje wartość bezwzględną (niezmiennik ujemny).')
        self.combo_kz = QComboBox()
        self.combo_kz.addItem('najszybszy: największe |λ − 1|', 0)
        self.combo_kz.addItem('najsilniej oscylujący: największy |arg λ|', 1)
        self.combo_kz.addItem('najsilniej gasnący lub rosnący: największe |ln |λ||', 2)
        dodaj(row, 'Z', self.combo_kz, 'Który mod (spośród nieużytych na X, Y) trafia na oś Z jako część najbardziej '
              'dynamiczna: najdalej od λ = 1 (najszybsza zmiana na zapis), o największej częstotliwości albo najszybciej '
              'gasnący lub rosnący.')
        self.combo_kczesc = QComboBox()
        for tekst, v in (('część rzeczywista', 0), ('część urojona', 1), ('moduł |ψ|', 2)):
            self.combo_kczesc.addItem(tekst, v)
        dodaj(row, 'część Z', self.combo_kczesc, 'Jak pokazać zespoloną funkcję własną na osi Z: część rzeczywista, '
              'urojona albo moduł (amplituda modu, bez fazy). Dla modu rzeczywistego część urojona to 0.')
        self.btn_koop = przycisk(row, 'Przelicz Koopmana', 'Ponownie dopasowuje operator Koopmana na karcie do wszystkich '
                                 'stanów osi czasu (zwykle dzieje się to samo po wczytaniu osi i zmianie opóźnień).')
        self.btn_koop_auto = przycisk(row, 'Osie automatycznie', 'Wraca do automatycznego wyboru osi z pól „X, Y”, „Z” i '
                                      '„część Z” (tabela niżej pokaże, które składowe zostały wybrane).')
        # --- ręczny wybór składowych na osie
        row = self.opcje.wiersz('Koopman')
        self.combo_ksort = QComboBox()
        for tekst, v in (('bliskość do 1: |λ − 1| rosnąco (niezmienniki najpierw)', 0), ('|λ| malejąco (najwolniej gasnące)', 1),
                         ('|arg λ| rosnąco (najwolniejsze oscylacje)', 2), ('|arg λ| malejąco (najszybsze oscylacje)', 3),
                         ('Re λ malejąco', 4), ('Im λ malejąco', 5), ('|λ − 1| malejąco (najbardziej dynamiczne)', 6)):
            self.combo_ksort.addItem(tekst, v)
        dodaj(row, 'sortuj mody', self.combo_ksort,
              'Kolejność modów w tabeli i w szybkim przypisaniu („od”, „do” liczone w tej kolejności). Bliskość do 1: '
              'najpierw niezmienniki (λ ≈ 1, wielkości prawie stałe wzdłuż toru).')
        self.combo_kre = QComboBox()
        for tekst, v in (('wszystkie', 0), ('Re λ > 0', 1), ('Re λ < 0', 2)):
            self.combo_kre.addItem(tekst, v)
        dodaj(row, 'znak Re λ', self.combo_kre, 'Pokaż tylko mody o dodatniej (λ po prawej stronie zera, bez zmiany znaku '
              'co zapis) albo ujemnej części rzeczywistej wartości własnej (mod zmienia znak co zapis).')
        self.combo_kim = QComboBox()
        for tekst, v in (('wszystkie', 0), ('Im λ > 0', 1), ('Im λ < 0', 2), ('Im λ = 0 (rzeczywiste)', 3)):
            self.combo_kim.addItem(tekst, v)
        dodaj(row, 'znak Im λ', self.combo_kim, 'Pokaż mody o dodatniej albo ujemnej części urojonej (pary sprzężone '
              'oscylują w przeciwne strony; Im ψ ma u nich przeciwny znak) albo tylko mody rzeczywiste.')
        row = self.opcje.wiersz('Koopman')
        self.combo_kos = QComboBox()
        for tekst, v in (('X', 'x'), ('Y', 'y'), ('Z', 'z')):
            self.combo_kos.addItem(tekst, v)
        dodaj(row, 'oś', self.combo_kos, 'Na którą oś trafiają mody z szybkiego przypisania.')
        self.spin_kod = self._int(row, 'mody od', 1, 16, 1, 'Pierwszy mod zakresu (numer w tabeli, po sortowaniu i filtrach).')
        self.spin_kdo = self._int(row, 'do', 1, 16, 1, 'Ostatni mod zakresu (włącznie).')
        self.combo_kczesci = QComboBox()
        for tekst, v in (('część rzeczywista', ('re',)), ('część urojona', ('im',)), ('rzeczywista i urojona', ('re', 'im'))):
            self.combo_kczesci.addItem(tekst, v)
        dodaj(row, 'części', self.combo_kczesci, 'Które części funkcji własnych ψ wybranych modów rzutować na oś.')
        self.btn_kustaw = przycisk(row, 'Ustaw na osi', 'Zastępuje składowe osi wybranymi częściami modów z zakresu.')
        self.btn_kdodaj = przycisk(row, 'Dodaj do osi', 'Dokłada wybrane części modów z zakresu do składowych osi.')
        self.btn_kczysc = przycisk(row, 'Wyczyść oś', 'Usuwa wszystkie składowe osi (oś stoi wtedy w zerze).')
        row = self.opcje.wiersz('Koopman')
        self.combo_kkomb = {}
        for os_ in ('x', 'y', 'z'):
            c = QComboBox(); c.addItem('suma (rzut liniowy)', 0); c.addItem('moduł √Σ²', 1)
            dodaj(row, f'łączenie {os_.upper()}', c,
                  f'Jak łączyć składowe osi {os_.upper()}: suma = zwykły rzut liniowy na sumę wybranych części ψ (mody '
                  f'mogą się wzmacniać albo znosić); moduł = długość wektora składowych √Σ², zawsze ≥ 0 (np. Re i Im '
                  f'jednego modu dają jego amplitudę |ψ| bez fazy).')
            self.combo_kkomb[os_] = c
        from PyQt6.QtWidgets import QTableWidget, QAbstractItemView
        self.tab_koop = QTableWidget(0, 13)
        self.tab_koop.setHorizontalHeaderLabels(['nr', 'X Re', 'X Im', 'Y Re', 'Y Im', 'Z Re', 'Z Im',
                                                 'Re λ', 'Im λ', '|λ|', '|λ−1|', 'arg°', 'okres'])
        self.tab_koop.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tab_koop.setMinimumHeight(220)
        self.tab_koop.verticalHeader().setVisible(False)
        self.tab_koop.verticalHeader().setDefaultSectionSize(20)
        self.tab_koop.setToolTip('Widmo dopasowanego operatora Koopmana. Zaznacz, które części (Re, Im) funkcji własnych '
                                 'których modów rzutować na osie X, Y, Z (łączenie osi wyżej). Zmiana pola przełącza na '
                                 'ręczny wybór osi; „Osie automatycznie” wraca do wyboru z pól powyżej.')
        # tabela widma w prawej kolumnie (pod pomiarami), widoczna tylko w układzie Koopmana: tam mieści się całe widmo
        self._koop_mody, self._osie_reczne = [], None
        self.koop_label = rola(QLabel('Wybierz układ „Koopman”, aby dopasować operator i zobaczyć widmo.'), 'opis')
        self.koop_label.setWordWrap(True)
        self.koop_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.opcje._strony['Koopman'].addWidget(self.koop_label)
        # --- dobór osi pod klastry: wiersze na górze strony, zaraz pod pierwszym
        from uklad import Przeplyw
        lay_k = self.opcje._strony['Koopman']

        def wiersz_gora(n):
            w = Przeplyw(); lay_k.insertLayout(n, w)
            return w
        row = wiersz_gora(1)
        self.btn_kklastry = przycisk(row, 'Osie pod klastry',
                                     'Sam dobiera osie X, Y (i Z) spośród składowych Koopmana tak, żeby cząsteczki '
                                     'układały się w wyraźne klastry z jak najmniejszą liczbą outlierów. Karta zwraca '
                                     'próbkę stanów cząsteczek widoku, potem k-means (na karcie, torch) ocenia każdą parę '
                                     'Re ψ, Im ψ i |ψ| modów dla k = 2..k max: zwartość i rozdzielenie klastrów (sylwetka) '
                                     'minus kara za outliery. Wynik trafia do tabeli jak ręczne osie.', glowny=True)
        self.spin_kl_m = self._int(row, 'próbka', 500, 16384, 4096,
                                   'Liczba cząsteczek widoku (równomiernie po zakresie) użytych do oceny. Więcej = '
                                   'pewniej, ale wolniej (4096 to około 2-5 s na RTX 3070).')
        self.combo_kl_kiedy = QComboBox()
        self.combo_kl_kiedy.addItem('bieżący zapis', False); self.combo_kl_kiedy.addItem('cały ruch (8 zapisów)', True)
        dodaj(row, 'oceniaj w', self.combo_kl_kiedy,
              'Bieżący zapis: klastry takie, jakie widać teraz. Cały ruch: ta sama próbka cząsteczek w 8 zapisach '
              'rozłożonych po osi czasu i w bieżącym; każdy zapis oceniany osobno, wynik uśredniony. Wygrywają osie, na '
              'których cząsteczki tworzą klastry przez cały trening (a nie takie, które tylko oddzielają wczesny trening '
              'od późnego). Środki klastrów do kolorów: z bieżącego zapisu.')
        self.chk_kl_z = pole_wyboru(row, 'także oś Z', 'Dobierz również oś Z: składową, z którą klastry w 3D są '
                                    'najlepsze. Bez zaznaczenia Z zostaje bez zmian.', zaznaczone=True)
        self.chk_kl_kolor = pole_wyboru(row, 'koloruj klastry', 'Po doborze osi przełącza kolor na „Klaster Koopmana”: '
                                        'każdy klaster ma swoją barwę, outliery są szare. Przydział liczy shader dla '
                                        'wszystkich cząsteczek w każdej klatce (najbliższy środek klastra).',
                                        zaznaczone=True)
        row = wiersz_gora(2)
        self.spin_kl_k = self._int(row, 'k max', 2, 10, 6, 'Największa sprawdzana liczba klastrów (k = 2..k max).')
        self.spin_kl_prog = self._float(row, 'outlier od × σ', 2, 6, 3, 1,
                                        'Punkt jest outlierem, gdy leży dalej od środka swojego klastra niż ta '
                                        'wielokrotność σ klastra (σ z mediany odległości, skalibrowanej na 2D albo 3D), '
                                        'albo należy do klastra mniejszego niż 2 % próbki. Dla gaussowskiego klastra 3σ '
                                        'odcina około 1 % punktów w 2D i 3 % w 3D.')
        self.spin_kl_waga = self._float(row, 'kara za outliery', 0, 20, 4, 1,
                                        'Wynik = sylwetka − kara × udział outlierów. Większa kara = mniej outlierów '
                                        'kosztem słabiej rozdzielonych klastrów.')
        self.kl_label = rola(QLabel('Osie pod klastry: kliknij przycisk w układzie „Koopman”.'), 'opis')
        self.kl_label.setWordWrap(True)
        self.kl_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay_k.insertWidget(3, self.kl_label)

        self.opcje.dodaj_strone('Atencja')
        row = self.opcje.wiersz('Atencja')
        self.chk_att = pole_wyboru(row, 'Heatmapa atencji',
                                   'Nakłada mapę atencji ViT z wag zapisanego checkpointu bieżącego kroku (pomiary naturalne). '
                                   'Widok jednego obrazu: liczona od razu przy każdym kroku. Cały zbiór: przyciskiem „Policz '
                                   'dla całego zbioru” (1500 obrazów na karcie, kilka sekund). Rodzaj mapy, blok, głowę i '
                                   'skalę wybiera się niżej, jak w Uczenie ViT.')
        self.combo_att_gdzie = QComboBox()
        for tekst, v in (('na cząsteczkach', 1), ('na zdjęciu w przestrzeni', 2), ('na cząsteczkach i zdjęciu', 3)):
            self.combo_att_gdzie.addItem(tekst, v)
        dodaj(row, 'nakładanie', self.combo_att_gdzie,
              'Gdzie pokazać mapę: kolor (inferno) i rozmiar cząsteczek według wartości mapy patcha; plama ciepła na '
              'zdjęciu rozpiętym na cząsteczkach (strona „Wygląd”, tylko widok jednego obrazu); albo oba naraz.')
        self.spin_att_wzm = self._float(row, 'powiększ wg mapy', 0, 10, 2.5, 1,
                                        'Rozmiar cząsteczki = rozmiar · (0,4 + ta wartość · mapa). 0 = sam kolor.')
        self.spin_att_krycie = self._float(row, 'krycie na zdjęciu', 0, 1, .6, 2,
                                           'Jak mocno heatmapa przykrywa zdjęcie w przestrzeni (0 = wcale, 1 = sama mapa).')
        self.btn_att_zbior = przycisk(row, 'Policz dla całego zbioru',
                                      'Liczy na karcie mapy atencji wszystkich obrazów zbioru z wag bieżącego zapisanego '
                                      'kroku i nakłada je na cząsteczki. Przy zmianie kroku mapa zostaje z kroku, dla którego '
                                      'ją policzono (napis obok); kliknij ponownie, aby przeliczyć.')
        self.att_label = rola(QLabel('Atencja: wyłączona.'), 'opis'); self.att_label.setWordWrap(True)
        from opcje_atencji import OpcjeAtencji
        self.opcje_att = OpcjeAtencji(z_kryciem=False)
        for w in (self.opcje_att.combo_strz, self.opcje_att.spin_k, self.opcje_att.spin_grubosc):
            w.setEnabled(False)
            w.setToolTip('Strzałki atencji są w przestrzeni makrowektorów (Uczenie ViT); tutaj tylko heatmapy.')
        self.opcje._strony['Atencja'].addWidget(self.opcje_att)
        self.opcje._strony['Atencja'].addWidget(self.att_label)
        self._att_klucz = self._att_zbior = None
        self._det = self._det_sciezka = self._att_obrazy = None

        self._strona_fizyki()
        self.opcje.dodaj_strone('Wygląd')
        row = self.opcje.wiersz('Wygląd')
        self.combo_co = QComboBox()
        for tekst, n in (('wszystkie', 1), ('co 2. patch', 2), ('co 3. patch', 3), ('co 5. patch', 5)):
            self.combo_co.addItem(tekst, n)
        dodaj(row, 'cząstek', self.combo_co,
              'Ile cząsteczek rysować: co n-ty patch w wierszach i kolumnach siatki 20 × 30 każdego obrazu (równo po '
              'zdjęciu, jak w przestrzeni makrowektorów). Granice osi i statystyki liczą się dalej ze wszystkich.')
        self.chk_tylko_narz = pole_wyboru(row, 'Tylko cząstki narzędzia',
                                          'Rysuje tylko patche, których środek leży w ramce GT narzędzia danego obrazu '
                                          '(z manifestu pomiaru). Bez ramek w zapisie opcja nic nie ukrywa.')
        self.chk_tylko_zazn = pole_wyboru(row, 'Tylko zaznaczone patche',
                                          'Rysuje tylko zaznaczone patche (Ctrl + klik na cząsteczce dodaje albo usuwa jej '
                                          'patch; ten sam numer patcha we wszystkich obrazach). Bez zaznaczenia: wszystkie.')
        self.btn_clear_sel = przycisk(row, 'Wyczyść zaznaczenie', 'Usuwa wszystkie zaznaczone patche.')
        self.sel_label = rola(QLabel('Zaznaczone patche: brak (Ctrl + klik na cząsteczce).'), 'opis')
        row.addWidget(self.sel_label)
        row = self.opcje.wiersz('Wygląd')
        self.combo_zdj = QComboBox(); self.combo_zdj.addItems(['brak', 'powierzchnia', 'siatka linii', 'powierzchnia i siatka'])
        dodaj(row, 'zdjęcie w przestrzeni', self.combo_zdj,
              'Tylko w widoku „Jeden obraz”: zdjęcie tego obrazu rozpięte na jego 600 cząsteczkach (siatka 20 × 30, '
              'w każdym węźle położenie cząsteczki tego patcha, między węzłami interpolacja liczona w shaderze). '
              'Gładka siatka = podobne sąsiednie patche; rozdarcia i zawinięcia = skok latentu. W układzie „siatka '
              'zdjęcia” zdjęcie leży na terenie wybranej cechy.')
        self.combo_zdj_kolor = QComboBox(); self.combo_zdj_kolor.addItems(['kolory zdjęcia', 'kolory cząstek',
                                                                            'zdjęcie, narzędzie na czerwono'])
        dodaj(row, 'kolor zdjęcia', self.combo_zdj_kolor, 'Barwy zdjęcia, barwy cząsteczek (jak „kolor”) albo zdjęcie '
              'z patchami w ramce GT na czerwono.')
        self.spin_zdj_krycie = self._float(row, 'krycie zdjęcia', .02, 1, .45, 2, 'Krycie zdjęcia w przestrzeni (0–1).')
        self.spin_zdj_gest = self._int(row, 'gęstość', 1, 8, 4, 'Ile punktów siatki zdjęcia na bok patcha (1 = tylko '
                                       'środki patchy, 4 = gładsze zdjęcie).')
        self.chk_brzeg = pole_wyboru(row, 'Brzeg obrazu', 'Ramka wokół brzegu obrazu na cząsteczkach brzegowych '
                                     '(niebieska: góra, żółta: lewa strona, szara: dół i prawa); tylko widok jednego obrazu.')
        note = QLabel('Z przestrzeni makrowektorów nie da się tu przenieść opcji, które potrzebują pełnych latentów 768D '
                      'każdej cząsteczki: kierunków i sił kanałów, PCA wektorów, mikrowektorów, atencji, Grand Tour, atlasu '
                      'kanałów ani krajobrazu. Zapis ruchu zbioru trzyma tylko rzut PCA i normę zmiany 768D (pełne latenty '
                      '900 000 cząsteczek w 101 krokach to około 280 GB).')
        note.setWordWrap(True); self.opcje._strony['Wygląd'].addWidget(rola(note, 'opis'))
        self.zaznaczone = set()
        self._zdj_cache = {}

        controls = QHBoxLayout()
        self.btn_play = rola(QPushButton('Odtwórz'), 'glowny'); self.btn_play.setEnabled(False)
        self.btn_play.setToolTip('Odtwarza albo zatrzymuje animację. Pauza zachowuje pozycję między zapisami.')
        self.btn_begin = QPushButton('Od początku'); self.btn_begin.setToolTip('Przechodzi do pierwszego zapisanego stanu.')
        self.btn_prev = QPushButton('Poprzedni zapis'); self.btn_prev.setToolTip('Przechodzi do poprzedniego zapisanego stanu.')
        self.btn_next = QPushButton('Następny zapis'); self.btn_next.setToolTip('Przechodzi do następnego zapisanego stanu.')
        self.step_label = QLabel('Krok —'); self.step_label.setMinimumWidth(165)
        self.suwak = QSlider(Qt.Orientation.Horizontal); self.suwak.setEnabled(False)
        self.suwak.setToolTip('Wybór zapisanego stanu osi czasu (tylko rzeczywiste zapisy, bez stanów pośrednich).')
        self.chk_petla = QCheckBox('Pętla')
        self.chk_petla.setToolTip('Po dojściu do ostatniego zapisu odtwarzanie wraca do pierwszego i gra dalej, aż do pauzy.')
        self.combo_zak_od = QComboBox(); self.combo_zak_do = QComboBox()
        for c, opis in ((self.combo_zak_od, 'Pierwszy zapisany checkpoint zakresu odtwarzania: animacja, pętla, suwak i „Od '
                         'początku” zaczynają się tutaj.'),
                        (self.combo_zak_do, 'Ostatni zapisany checkpoint zakresu odtwarzania (maks. krok w liczniku): animacja '
                         'zatrzymuje się albo, z pętlą, wraca do „od”.')):
            c.setMinimumWidth(110); c.setMaxVisibleItems(25); c.setEnabled(False); c.setToolTip(opis)
        self.lbl_zak_od = QLabel('od'); self.lbl_zak_do = QLabel('do')
        self.lbl_zak_od.setToolTip(self.combo_zak_od.toolTip()); self.lbl_zak_do.setToolTip(self.combo_zak_do.toolTip())
        for widget in (self.btn_play, self.btn_begin, self.btn_prev, self.btn_next, self.chk_petla, self.lbl_zak_od,
                       self.combo_zak_od, self.lbl_zak_do, self.combo_zak_do, self.step_label, self.suwak):
            controls.addWidget(widget)
        lay.addLayout(controls)
        self.frame_label = staly_napis(rola(QLabel('Wczytaj gotowy wynik lub oblicz nowe trajektorie.'), 'opis'), 1)
        lay.addWidget(self.frame_label)
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
        dodaj_nawigacje(self, self.opcje, self.view)       # obrót 360°, kroki 15°, ujęcia, lustro (jak w Uczenie ViT)
        self.btn_odepnij = rola(QPushButton('Odepnij'), 'narzedzie'); self.opcje.pasek.addWidget(self.btn_odepnij)
        self.odpinanie = Odpinanie(self.view, 'Ruch zbioru: cząsteczki', self.btn_odepnij)
        b = rola(QPushButton('Kamera od nowa'), 'narzedzie'); b.setToolTip('Przywraca domyślną odległość i kąty kamery.')
        b.clicked.connect(lambda: self.view.setCameraPosition(distance=3.8, elevation=18, azimuth=35))
        self.opcje.pasek.addWidget(b)
        self.view.klik_px.connect(self._pick)
        split.addWidget(self.view)
        right = QWidget(); rl = QVBoxLayout(right)
        from PyQt6.QtWidgets import QSizePolicy
        from PyQt6.QtWidgets import QScrollArea, QFrame
        right.setMinimumWidth(300)
        self.panel_boczny = QScrollArea(); self.panel_boczny.setWidget(right); self.panel_boczny.setWidgetResizable(True)
        self.panel_boczny.setFrameShape(QFrame.Shape.NoFrame)        # wysoki panel przewija się, okno się nie rozciąga
        self.panel_boczny.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.panel_boczny.setMinimumWidth(320)
        self._split = split
        self.metrics = staly_napis(QLabel('Pomiary pojawią się po wczytaniu.'), 8); rl.addWidget(self.metrics)
        self.tab_koop.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.tab_koop.setFixedHeight(220)
        self.tab_koop.setVisible(False); rl.addWidget(self.tab_koop)
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
        self.log_label = staly_napis(QLabel(), 2); ll.addWidget(self.log_label)
        self.log_box.hide(); rl.addWidget(self.log_box)
        self.heat = pg.ImageView(); self.heat.ui.roiBtn.hide(); self.heat.ui.menuBtn.hide()
        self.heat.setMinimumHeight(150)
        self.heat_label = staly_napis(QLabel('Ruch 768D per patch (mapa 20 × 30, liczona na karcie)'), 2)
        rl.addWidget(self.heat_label); rl.addWidget(self.heat, 1)
        self._panel_fizyki(rl)
        split.addWidget(self.panel_boczny); split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)
        split.setCollapsible(0, False)
        self.btn_panel = rola(QPushButton('Ukryj panel boczny'), 'narzedzie')
        self.btn_panel.setCheckable(True)
        self.btn_panel.setToolTip('Chowa albo pokazuje prawy panel (pomiary, tabela modów Koopmana, wykresy metryk, '
                                  'mapa ruchu). Ukryty panel nie odświeża mapy ruchu, a widok cząsteczek zajmuje całą '
                                  'szerokość. Szerokość panelu zmienia się przeciągnięciem granicy.')
        self.btn_panel.toggled.connect(self._panel_boczny)
        self.opcje.pasek.addWidget(self.btn_panel)
        lay.addWidget(split, 1)
        self.gpu_status = staly_napis(rola(QLabel('GPU: oczekuje na dane.'), 'status'), 1); lay.addWidget(self.gpu_status)
        self.pasek = QProgressBar(); self.pasek.setRange(0, 100); lay.addWidget(self.pasek)
        self.status = staly_napis(QLabel('Nic nie wczytuje się ani nie liczy przy otwieraniu zakładki.'), 2)
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
        self.btn_szklo.clicked.connect(self._os_szkla)
        self.btn_szklo_odswiez.clicked.connect(self._wypelnij_osie_szkla)
        self.btn_checkpoint_dir.clicked.connect(lambda: self._choose_dir(self.edit_checkpoint_dir))
        self.btn_scan_natural.clicked.connect(self._scan_natural)
        self.btn_default_natural.clicked.connect(self._default_natural)
        self.btn_natural_35k.clicked.connect(self._natural_35k)
        self.btn_logs_file.clicked.connect(lambda: self._choose_file(self.edit_training_logs, 'Logi (*.json *.jsonl);;Wszystkie (*)'))
        self.btn_logs_dir.clicked.connect(lambda: self._choose_dir(self.edit_training_logs))
        self.btn_load_logs.clicked.connect(self._load_logs)
        self.btn_compute_natural.clicked.connect(self._compute_natural)
        self.btn_latest_natural.clicked.connect(self._latest_natural)
        self.btn_resume_natural.clicked.connect(self._resume_natural)
        self.combo_log_metric.currentIndexChanged.connect(self._plot_logs)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_begin.clicked.connect(lambda: self._seek_index(self._zakres_idx()[0]))
        self.btn_film.clicked.connect(self._nagraj_film_dialog)
        self.combo_zak_od.currentIndexChanged.connect(lambda _: self._zmiana_zakresu())
        self.combo_zak_do.currentIndexChanged.connect(lambda _: self._zmiana_zakresu())
        self.btn_prev.clicked.connect(lambda: self._seek_index(self.suwak.value() - 1))
        self.btn_next.clicked.connect(lambda: self._seek_index(self.suwak.value() + 1))
        self.suwak.valueChanged.connect(self._seek_index)
        for widget in (self.chk_movement, self.chk_smooth):
            widget.toggled.connect(self._display_options)
        for widget in (self.combo_scope, self.combo_color, self.combo_kadr, self.combo_uklad, self.combo_cecha,
                       self.combo_mozaika):
            widget.currentIndexChanged.connect(self._display_options)
        for widget in (self.spin_image, self.spin_opacity, self.spin_size, self.spin_trail, self.spin_zdj_krycie,
                       self.spin_zdj_gest):
            widget.valueChanged.connect(self._display_options)
        self.combo_wyg.currentIndexChanged.connect(self._display_options)
        self.chk_att.toggled.connect(lambda _: self._atencja_odswiez())
        self.combo_att_gdzie.currentIndexChanged.connect(lambda _: self._atencja_odswiez())
        for widget in (self.spin_att_wzm, self.spin_att_krycie):
            widget.valueChanged.connect(lambda _: self._atencja_odswiez())
        self.opcje_att.zmiana.connect(lambda: self._atencja_odswiez(sila=True))
        self.btn_att_zbior.clicked.connect(self._atencja_zbioru)
        for widget in (self.spin_wyg_okno, self.spin_wyg_alfa, self.spin_moc):
            widget.valueChanged.connect(self._display_options)
        for widget in (self.combo_kxy, self.combo_kz, self.combo_kczesc):
            widget.currentIndexChanged.connect(self._display_options)
        self.spin_kd.valueChanged.connect(self._display_options)
        self.btn_koop.clicked.connect(lambda: (self.cloud.przelicz_koopmana(), self._budz()))
        self.btn_koop_auto.clicked.connect(lambda: self._osie_koopmana(None))
        for widget in (self.combo_ksort, self.combo_kre, self.combo_kim):
            widget.currentIndexChanged.connect(lambda _: self._tabela_koopmana())
        for widget in self.combo_kkomb.values():
            widget.currentIndexChanged.connect(lambda _: self._z_tabeli_koopmana())
        self.btn_kustaw.clicked.connect(lambda: self._przypisz_koopmana('ustaw'))
        self.btn_kdodaj.clicked.connect(lambda: self._przypisz_koopmana('dodaj'))
        self.btn_kczysc.clicked.connect(lambda: self._przypisz_koopmana('czysc'))
        self.tab_koop.itemChanged.connect(lambda _: self._z_tabeli_koopmana())
        self.spin_kd.valueChanged.connect(lambda _: setattr(self, '_osie_reczne', None))   # nowe dopasowanie: nowe mody
        self.cloud.koopman_gotowy.connect(self._koopman_opis, Qt.ConnectionType.QueuedConnection)
        self.cloud.fizyka_gotowa.connect(self._fizyka_gotowa, Qt.ConnectionType.QueuedConnection)
        self.btn_kklastry.clicked.connect(self._klastry_koopmana)
        self.cloud.probka_koopmana.connect(self._klastry_probka, Qt.ConnectionType.QueuedConnection)
        for widget in (self.combo_co, self.combo_zdj, self.combo_zdj_kolor):
            widget.currentIndexChanged.connect(self._display_options)
        for widget in (self.chk_tylko_narz, self.chk_tylko_zazn, self.chk_brzeg, self.chk_bez_poz):
            widget.toggled.connect(self._display_options)
        self.btn_clear_sel.clicked.connect(lambda: self._zaznacz(None))
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

    def _panel_boczny(self, ukryj):
        """Chowanie prawego panelu; po pokazaniu wraca poprzednia szerokość."""
        if ukryj:
            self._szer_panelu = self._split.sizes()
            self.panel_boczny.hide()
        else:
            self.panel_boczny.show()
            if getattr(self, '_szer_panelu', None):
                self._split.setSizes(self._szer_panelu)
        self.btn_panel.setText('Pokaż panel boczny' if ukryj else 'Ukryj panel boczny')
        if not ukryj and self.timeline:
            self._last_stats = None; self._render(); self._apply_heat(force=True)

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
        return (self.btn_scan_natural, self.btn_default_natural, self.btn_natural_35k, self.btn_traj, self.btn_load_logs,
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

    def _wybor_modelu(self):
        """Model z assets albo historia: ścieżka do pola „ViT”, lista kroków historii."""
        import historie_checkpointow as HC
        rodzaj, klucz = self.combo_model.currentData()
        self.combo_model_krok.blockSignals(True); self.combo_model_krok.clear()
        self.combo_model_krok.setEnabled(rodzaj == 'historia')
        if rodzaj == 'model':
            _, nazwa, plik, krok, opis = next(m for m in HC.MODELE if m[0] == klucz)
            self.edit_checkpoint.setText(str(plik))
            self._set_text(self.model_opis, opis if plik.is_file() else f'Brak pliku {plik}.')
        elif rodzaj == 'historia':
            h = HC.historia(klucz)
            kroki = h.kroki()
            if not kroki:
                self._set_text(self.model_opis, f'Brak lokalnych wag w {h.wagi}.')
            else:
                koniec = [k for k in kroki if h.na_epoke and k % h.na_epoke == 0] if h.na_epoke else []
                for k in kroki:
                    self.combo_model_krok.addItem(h.opis_kroku(k), k)
                ostatni = koniec[-1] if koniec else kroki[-1]
                self.combo_model_krok.setCurrentIndex(kroki.index(ostatni))
                self._set_text(self.model_opis, f'{len(kroki)} checkpointów; {h.opis}')
        else:
            self._set_text(self.model_opis, '')
        self.combo_model_krok.blockSignals(False)
        if rodzaj == 'historia':
            self._wybor_kroku_modelu()

    def _wybor_kroku_modelu(self):
        import historie_checkpointow as HC
        rodzaj, klucz = self.combo_model.currentData()
        krok = self.combo_model_krok.currentData()
        if rodzaj == 'historia' and krok is not None:
            self.edit_checkpoint.setText(str(HC.historia(klucz).sciezka(krok)))

    def _krok_modelu(self, path, krok):
        """Krok z pliku, a gdy go brak (sam state_dict), z listy modeli assets."""
        if krok is not None:
            return krok
        import historie_checkpointow as HC
        for _, _, plik, k, _ in HC.MODELE:
            if Path(path).resolve() == Path(plik).resolve():
                return k
        return None

    def _load_model(self):
        path, device = self.edit_checkpoint.text().strip(), self.combo_device.currentText()
        def load(progress, cancel):
            import hashlib
            from detekcja import Detektor
            progress({'message': 'Sprawdzanie checkpointu i zgodności architektury…'})
            model = Detektor(path, device)
            with open(path, 'rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            result = {'path': str(Path(path).resolve()), 'step': self._krok_modelu(path, model.krok), 'sha256': digest,
                      'device': str(model.dev)}
            del model
            return result
        def loaded(result):
            self._active_checkpoint = result
            krok = f'k{result["step"]}' if result['step'] is not None else 'krok nieznany'
            self.model_status.setText(f'ViT {krok} · {Path(result["path"]).name} · {result["device"]} · SHA {result["sha256"][:12]}')
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

    # ------------------------------------------------------------------ historie checkpointów i wybrane zdjęcia
    def _wiersze_trajektorii(self):
        """Na górze strony „Uczenie naturalne”: historia checkpointów z zakresem epok i wybór zdjęć do trajektorii."""
        from uklad import Przeplyw
        import historie_checkpointow as HC
        import zbiory_zdjec as Z
        lay = self.opcje._strony['Uczenie naturalne']

        def wiersz(n):
            w = Przeplyw(); lay.insertLayout(n, w)
            return w
        row = wiersz(0)
        self.combo_hist = QComboBox()
        for h in HC.HISTORIE:
            self.combo_hist.addItem(h.nazwa, h.klucz)
        self.combo_hist.addItem('własny folder (pola niżej)', '')
        dodaj(row, 'historia', self.combo_hist,
              'Która historia uczenia: wypełnia folder checkpointów, logi, checkpoint referencyjny i zakres kroków. '
              + ' '.join(f'{h.nazwa}: {h.opis}' for h in HC.HISTORIE))
        self.spin_ep_od = self._int(row, 'od epoki', 0, 1000, 0, 'Pierwsza epoka (0 = stan przed uczeniem). Tylko dla '
                                    'historii z epokami; ustawia „od kroku”.')
        self.spin_ep_do = self._int(row, 'do epoki', 0, 1000, 21, 'Ostatnia epoka w pomiarze (włącznie). Tylko dla '
                                    'historii z epokami; ustawia „do kroku” i checkpoint co epokę.')
        self.hist_label = rola(QLabel(), 'opis')
        row.addWidget(self.hist_label)
        row = wiersz(1)
        self.combo_zrodlo = QComboBox()
        for k, opis in Z.ZRODLA.items():
            self.combo_zrodlo.addItem(opis, k)
        dodaj(row, 'zdjęcia z', self.combo_zrodlo,
              'Skąd brać zdjęcia: manifest 1500 obrazów pomiaru (ten sam co w dotychczasowych pomiarach) albo pula '
              '35 000 klatek, na których uczył się bieg 35 000 (to dane uczące tego modelu). Klatki puli są '
              'wypakowywane z lokalnych zipów Cataract-LMM z kontrolą SHA-256.')
        self.edit_wybor = QLineEdit(); self.edit_wybor.setMinimumWidth(220)
        self.edit_wybor.setPlaceholderText('np. 0-9, 25, 100-120')
        dodaj(row, 'wybrane', self.edit_wybor, 'Numery zdjęć źródła (od 0): pojedyncze i zakresy po przecinku. Wypełnia '
              'je też okno „Wybierz zdjęcia…” i „Losuj”.')
        self.btn_wybierz_zdj = przycisk(row, 'Wybierz zdjęcia…', 'Okno z listą zdjęć źródła: pola wyboru, filtr nazwy '
                                        'i filmu, losowanie i podgląd zdjęcia.')
        self.spin_los = self._int(row, 'losuj', 1, 35000, 100, 'Ile losowych zdjęć wybrać przyciskiem „Losuj” (ziarno 2026).')
        self.btn_losuj = przycisk(row, 'Losuj', 'Wybiera losowe zdjęcia ze źródła (ziarno 2026, zawsze to samo dla tej liczby).')
        self.btn_traj = przycisk(row, 'Policz trajektorie wybranych',
                                 'Buduje manifest wybranych zdjęć, mierzy naturalnym forwardem każdy checkpoint z '
                                 'zakresu historii (bez uczenia, zapis 16-bitowy) i od razu wczytuje oś czasu: '
                                 'liczba cząsteczek = 600 × liczba zdjęć.', glowny=True)
        self.wybor_label = rola(QLabel('Wybrane: brak.'), 'opis')
        row.addWidget(self.wybor_label)
        self.combo_hist.currentIndexChanged.connect(lambda _: self._ustaw_historie())
        for w in (self.spin_ep_od, self.spin_ep_do):
            w.valueChanged.connect(lambda _: self._ustaw_historie(tylko_zakres=True))
        self.edit_wybor.textChanged.connect(lambda _: self._opis_wyboru())
        self.combo_zrodlo.currentIndexChanged.connect(lambda _: self._opis_wyboru())
        self.btn_wybierz_zdj.clicked.connect(self._okno_zdjec)
        self.btn_losuj.clicked.connect(lambda: self.edit_wybor.setText(
            Z.zwiez(Z.losowe(self.combo_zrodlo.currentData(), self.spin_los.value()))))
        self.btn_traj.clicked.connect(self._trajektorie_wybranych)
        self._opis_wyboru()

    def _ustaw_historie(self, tylko_zakres=False):
        import historie_checkpointow as HC
        klucz = self.combo_hist.currentData()
        if not klucz:
            self.hist_label.setText('Własny folder: ustaw pola checkpointów i zakresu niżej.'); return
        h = HC.historia(klucz)
        kroki = h.kroki()
        for w in (self.spin_ep_od, self.spin_ep_do):
            w.setEnabled(bool(h.na_epoke))
        if not tylko_zakres:
            self.edit_checkpoint_dir.setText(str(h.wagi)); self.edit_training_logs.setText(str(h.dane))
        if not kroki:
            self.hist_label.setText(f'Brak lokalnych wag w {h.wagi}.'); return
        if h.na_epoke:
            epoki = h.epoki()
            od, do = min(self.spin_ep_od.value(), epoki[-1]), min(self.spin_ep_do.value(), epoki[-1])
            start, koniec, co = od * h.na_epoke, max(od, do) * h.na_epoke, h.na_epoke
            self.hist_label.setText(f'epoki {od}–{max(od, do)} z {epoki[-1]} pobranych ({len(kroki)} plików)')
        else:
            start, koniec, co = kroki[0], kroki[-1], (kroki[1] - kroki[0]) if len(kroki) > 1 else 1
            self.hist_label.setText(f'kroki {start}–{koniec} co {co} ({len(kroki)} plików)')
        self.spin_natural_start.setValue(start); self.spin_natural_end.setValue(koniec)
        self.spin_natural_stride.setValue(co)
        ref = R.DEFAULT_CHECKPOINT if klucz == 'k1000' and koniec == 1000 else h.sciezka(koniec)
        self.edit_checkpoint.setText(str(ref))

    def _opis_wyboru(self):
        import zbiory_zdjec as Z
        zr = self.combo_zrodlo.currentData()
        try:
            idx = Z.parsuj(self.edit_wybor.text(), len(Z.wiersze(zr)))
        except Exception as e:
            self.wybor_label.setText(f'Zły wybór: {e}'); return []
        self.wybor_label.setText(f'Wybrane: {len(idx)} zdjęć = {len(idx) * 600:,} cząsteczek.' if idx else
                                 'Wybrane: brak (wpisz numery, „Losuj” albo „Wybierz zdjęcia…”).')
        return idx

    def _okno_zdjec(self):
        import zbiory_zdjec as Z
        zr = self.combo_zrodlo.currentData()
        try:
            teraz = Z.parsuj(self.edit_wybor.text(), len(Z.wiersze(zr)))
        except ValueError:
            teraz = []
        wynik = okno_wyboru_zdjec(self, zr, teraz)
        if wynik is not None:
            self.edit_wybor.setText(Z.zwiez(wynik))

    def _po_zadaniu(self, funkcja, proby=100):
        """Wywołaj funkcję, gdy wątek tła się zwolni (zadanie zgłasza wynik przed swoim końcem)."""
        if self.watek is None:
            funkcja()
        elif proby > 0:
            QTimer.singleShot(100, lambda: self._po_zadaniu(funkcja, proby - 1))

    def _trajektorie_wybranych(self):
        """Manifest wybranych zdjęć + pomiar naturalny checkpointów historii, potem wczytanie osi czasu."""
        import dataclasses
        import zbiory_zdjec as Z
        idx = self._opis_wyboru()
        if not idx:
            self._error('Nie wybrano zdjęć: wpisz numery, użyj „Losuj” albo „Wybierz zdjęcia…”.'); return
        if self.combo_hist.currentData():
            self._ustaw_historie(tylko_zakres=True)
        zr = self.combo_zrodlo.currentData()
        try:
            baza = self._natural_config()
        except Exception as e:
            self._error(str(e)); return

        def licz(progress, cancel):
            manifest, katalog = Z.zbuduj(zr, idx, postep=lambda a, n: progress(
                {'phase': 'validate', 'message': f'Przygotowanie zdjęć {a}/{n}', 'fraction': a / n}))
            config = dataclasses.replace(baza, manifest=str(manifest), image_root=str(katalog),
                                         max_images=len(idx), frame_format='q16').validate()
            return manifest, katalog, N.run(config, progress=progress, cancel=cancel)

        def gotowe(wynik):
            manifest, katalog, path = wynik
            self.edit_manifest.setText(str(manifest)); self.edit_image_root.setText(str(katalog))
            self.spin_natural_images.setValue(len(idx))
            self.edit_timeline.setText(str(path))
            self.status.setText(f'Trajektorie {len(idx)} zdjęć ({len(idx) * 600:,} cząsteczek) zapisane w {path}. '
                                'Wczytuję oś czasu.')
            self.pasek.setValue(100)
            self._po_zadaniu(self._load_timeline)
        self.status.setText(f'Trajektorie {len(idx)} wybranych zdjęć: przygotowanie i pomiar w tle…')
        self._job(licz, gotowe)

    def _natural_35k(self):
        family = R.ROOT / 'assets/MalyViT_35000_od_zera_s47'
        wagi = family / 'wagi'
        self.edit_checkpoint_dir.setText(str(wagi))
        self.edit_training_logs.setText(str(family / 'dane'))
        self.edit_manifest.setText(str(R.DEFAULT_MANIFEST))
        kroki = sorted(int(p.stem[1:]) for p in wagi.glob('k*.pt') if p.stem[1:].isdigit()) if wagi.is_dir() else []
        epoki = [k for k in kroki if k and k % 1094 == 0]
        koniec = epoki[-1] if epoki else 21 * 1094
        self.edit_checkpoint.setText(str(wagi / f'k{koniec:05d}.pt') if epoki else '')
        self.spin_natural_start.setValue(0); self.spin_natural_end.setValue(koniec); self.spin_natural_stride.setValue(1094)
        if not kroki:
            self.status.setText(f'Brak lokalnych wag w {wagi}. Pobierz je skryptem pobierz_z_edena.sh z folderu '
                                'assets/MalyViT_35000_od_zera_s47 (806 plików, 8,8 GB), potem kliknij „Sprawdź historię”.')
            return
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

    def _wypelnij_osie_szkla(self):
        """Lista osi układu szkła z dysku (bez wczytywania czegokolwiek); pusta lista daje jeden wpis z podpowiedzią."""
        biezacy = self.combo_szklo.currentData()
        self.combo_szklo.blockSignals(True)
        self.combo_szklo.clear()
        for opis, p in R.osie_szkla():
            self.combo_szklo.addItem(opis, str(p))
        if self.combo_szklo.count() == 0:
            self.combo_szklo.addItem('brak osi szkła (eksport_osi_szkla.py / model_szkla.py)', '')
        if biezacy:
            i = self.combo_szklo.findData(biezacy)
            if i >= 0:
                self.combo_szklo.setCurrentIndex(i)
        self.combo_szklo.blockSignals(False)

    def _os_szkla(self):
        """Wybrana oś układu szkła do pola „zapis ruchu” i wczytanie."""
        p = self.combo_szklo.currentData()
        if not p:
            self._error('Brak osi szkła: uruchom outputs/porownanie_szklo_20261010/eksport_osi_szkla.py (układ względem klatki) '
                        'albo model_szkla.py (symulacja szkła koloidalnego).'); return
        self.edit_timeline.setText(p); self._load_timeline()

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
        self._zdj_cache = {}
        self._dane_pomocnicze(timeline)
        self._obszary_oka(timeline)
        self.cloud.wyczysc_atencje(); self._att_klucz = self._att_zbior = None; self._att_obrazy = None
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
        self._wypelnij_zakres(timeline)
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
        self.cloud.ustaw_uklad(self.combo_uklad.currentData(), self.combo_cecha.currentData(),
                               bool(self.combo_mozaika.currentData()))
        self.tab_koop.setVisible(self.combo_uklad.currentData() == 3)
        self.cloud.ustaw_bez_polozenia(self.chk_bez_poz.isChecked())
        self.cloud.ustaw_wygladzanie(self.combo_wyg.currentData(), self.spin_wyg_okno.value(), self.spin_wyg_alfa.value())
        self.cloud.ustaw_moc_koloru(self.spin_moc.value())
        self._ustaw_fizyke()
        self.cloud.ustaw_koopmana(self.spin_kd.value(), self.combo_kxy.currentData(), self.combo_kz.currentData(),
                                  self.combo_kczesc.currentData(), osie=self._osie_reczne)
        self.cloud.ustaw_wyglad(self.combo_co.currentData(), self.chk_tylko_narz.isChecked(),
                                self.chk_tylko_zazn.isChecked())
        jeden = self.combo_scope.currentIndex() == 1
        tryb = self.combo_zdj.currentIndex() if jeden else 0
        self.cloud.ustaw_zdjecie(self._zdjecie_obrazu(self.spin_image.value()) if tryb else None, tryb,
                                 self.combo_zdj_kolor.currentIndex(), self.spin_zdj_krycie.value(),
                                 self.spin_zdj_gest.value(), self.chk_brzeg.isChecked() and jeden)
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
            i0, i1 = self._zakres_idx()
            pelny = i0 == 0 and i1 == len(steps) - 1
            self._set_text(self.step_label, f'Krok {self._visual_step:.2f} / {steps[i1]}'
                           + ('' if pelny else f'  (zakres {steps[i0]}–{steps[i1]} z {steps[-1]})'))
        if 0 < s < 1:
            if etykiety:
                self._set_text(self.frame_label, f'STAN NIEZAPISANY · liniowe przejście na GPU między zapisami '
                               f'{a} → {b} (s = {s:.2f}). Zapisane są tylko stany {a} i {b}; metryki i mapa ze stanu {a}.')
            self._mark_unsaved(True)
        else:
            shown = b if s >= 1 else a
            extra = ' (płynne przejścia wyłączone)' if not self.chk_smooth.isChecked() and fraction > 0 else ''
            wyg = self.combo_wyg.currentData()
            if wyg:
                opis = (f'średnia ruchoma z {self.spin_wyg_okno.value() | 1} zapisów' if wyg == 1
                        else f'EMA α = {self.spin_wyg_alfa.value():.2f}')
                self._set_text(self.frame_label, f'STAN WYGŁADZONY w zapisie {shown}: {opis} z zapisanych stanów '
                               f'(nie jest to zapis z pliku){extra}.')
                self._mark_unsaved(True)
            else:
                self._set_text(self.frame_label, f'Zapisany stan {shown}: położenia i normy bitowo z pliku{extra}.')
                self._mark_unsaved(False)
        if self.chk_att.isChecked():
            self._atencja_odswiez()
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
        for kursor in getattr(self, '_fz_kursory', ()):
            kursor.setValue(a)
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
        if self._pending_heat is None or self.panel_boczny.isHidden():
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

    # ------------------------------------------------------------------ heatmapy atencji (liczy karta)
    def _krok_atencji(self):
        """Zapisany krok, z którego wag liczona jest atencja (lewy zapis bieżącej pary)."""
        i, _ = self._indices()
        return self.timeline.steps[i]

    def _sciezka_wag(self, krok):
        """Plik wag kroku z metadanych pomiaru; gdy przeniesiono go (np. do assets), ten sam plik z historii
        checkpointów, rozpoznany po nazwie i sumie SHA-256 zapisanej w metadanych."""
        for r in (self.timeline.metadata.get('checkpoints') or []):
            if r.get('step') == krok:
                p = r.get('path')
                if p and Path(p).is_file():
                    return p
                import hashlib
                import historie_checkpointow as HC
                for h in HC.HISTORIE:
                    q = h.wagi / Path(str(p)).name
                    if q.is_file() and (not r.get('sha256') or hashlib.sha256(q.read_bytes()).hexdigest() == r['sha256']):
                        return str(q)
                return p
        return None

    def _detektor(self, sciezka):
        """Detektor ViT z wagami checkpointu (moduł detekcji ładowany dopiero przy pierwszym użyciu atencji)."""
        import torch
        import detekcja as D
        if self._det is None:
            self._det = D.Detektor(sciezka)
        elif self._det_sciezka != sciezka:
            ck = torch.load(sciezka, map_location='cpu', weights_only=True)
            self._det.net.load_state_dict(ck.get('net', ck), strict=True)
        self._det_sciezka = sciezka
        return self._det

    def _token_atencji(self):
        if self.zaznaczone:
            return np.array(sorted(self.zaznaczone))
        w = self.cloud.wybrany
        return None if w is None else int(w % (self.cloud.tpi or 600))

    def _atencja_odswiez(self, sila=False):
        tryb = self.combo_att_gdzie.currentData() if self.chk_att.isChecked() else 0
        self.cloud.ustaw_atencje(tryb=tryb, wzm=self.spin_att_wzm.value(), krycie=self.spin_att_krycie.value())
        if not tryb or not self.timeline:
            self._set_text(self.att_label, 'Atencja: wyłączona.' if not tryb else 'Atencja: wczytaj oś czasu.')
            self._budz(); return
        krok = self._krok_atencji()
        sciezka = self._sciezka_wag(krok)
        if sciezka is None:
            self._set_text(self.att_label, 'Atencja potrzebuje checkpointu każdego kroku: dostępna dla pomiarów '
                           'naturalnych (zapisane modele kolejnych kroków treningu).')
            return
        if self.combo_scope.currentIndex() != 1:
            z = self._att_zbior
            self._set_text(self.att_label, 'Cały zbiór: kliknij „Policz dla całego zbioru”.' if z is None else
                           f'Atencja całego zbioru z wag kroku {z[0]}' + ('' if z[0] == krok else
                           f' (bieżący krok {krok}: kliknij ponownie, aby przeliczyć)') + f'. {z[1]}')
            self._budz(); return
        if self.watek is not None:
            return                                         # model zajęty liczeniem w tle
        obraz = min(self.spin_image.value(), max(self.timeline.image_count - 1, 0))
        u = None
        try:
            import atencja_zbioru as AZ
            u = AZ.ustawienia(self.opcje_att)
            token = self._token_atencji()
            klucz = (str(self.timeline.root), krok, obraz, tuple(sorted(u.items())), None if token is None else str(token))
            if klucz == self._att_klucz and not sila:
                return
            rgb = self._zdjecie_obrazu(obraz)
            if rgb is None:
                self._set_text(self.att_label, f'Brak zdjęcia obrazu {obraz}: atencji nie da się policzyć.'); return
            det = self._detektor(sciezka)
            import torch
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().to(det.dev) / 255
            v = AZ.mapy(det.net, x, u, token)
            if v is None:
                self._set_text(self.att_label, 'Ta mapa potrzebuje wybranego patcha: kliknij cząsteczkę albo zaznacz '
                               'patche (Ctrl + klik).'); return
            self.cloud.ustaw_atencje(v[0], first=obraz * (self.cloud.tpi or 600))
            self._att_klucz = klucz
            self._set_text(self.att_label, f'Atencja obrazu {obraz} z wag kroku {krok}: {self.opcje_att.opis(token)} (znormowana w obrazie do 0..1).')
        except Exception as e:                             # brak CUDA, pliku wag albo zdjęcia: komunikat, bez przerwy
            self._set_text(self.att_label, f'Atencja niedostępna: {type(e).__name__}: {e}')
        self._budz()

    def _atencja_zbioru(self):
        """W tle: mapy atencji wszystkich obrazów z wag bieżącego kroku (obrazy dekodowane raz, na karcie)."""
        if not self.timeline:
            return
        krok = self._krok_atencji()
        sciezka = self._sciezka_wag(krok)
        m = self.timeline.metadata
        if sciezka is None or not m.get('resolved_image_paths') or not m.get('images'):
            self._set_text(self.att_label, 'Atencja zbioru potrzebuje checkpointów kroków i ścieżek zdjęć (pomiar '
                           'naturalny).'); return
        import atencja_zbioru as AZ
        u = AZ.ustawienia(self.opcje_att)
        token = self._token_atencji()
        if not self.chk_att.isChecked():
            self.chk_att.setChecked(True)

        def licz(progress, cancel):
            import torch
            if self._att_obrazy is None:
                progress({'phase': 'attention', 'message': 'Dekodowanie zdjęć zbioru (raz, do pamięci karty)…', 'fraction': 0})
                obrazy = [dict(r, index=i) for i, r in enumerate(m['images'])]
                sciezki = [Path(p) for p in m['resolved_image_paths']]
                self._att_obrazy = N._Obrazy(obrazy, sciezki, torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                             torch.cuda.is_available(), cancel).tensor
            det = self._detektor(sciezka)
            wynik = AZ.mapy_zbioru(det.net, self._att_obrazy, u, token,
                                   postep=lambda a, n: progress({'phase': 'attention', 'fraction': a / n,
                                                                  'message': f'Atencja zbioru: {a}/{n} obrazów'}),
                                   przerwij=cancel)
            return wynik

        def gotowe(wynik):
            if wynik is None:
                self._set_text(self.att_label, 'Ta mapa potrzebuje wybranego patcha: kliknij cząsteczkę albo zaznacz '
                               'patche (Ctrl + klik).'); return
            self.cloud.ustaw_atencje(wynik.reshape(-1), first=0)
            self._att_zbior = (krok, self.opcje_att.opis(token))
            self._att_klucz = None
            self._atencja_odswiez()
        self._job(licz, gotowe)

    def _koopman_opis(self, info):
        """Po dopasowaniu albo zmianie osi: tabela modów i opis osi."""
        mody = info.get('mody') or []
        if not mody:
            self.koop_label.setText(info.get('opis', '')); return
        self._koop_mody = mody
        osie = info.get('osie') or {}
        lam = {m['j']: m for m in mody}

        def opis(os_):
            o = osie.get(os_, {'skladowe': [], 'kombinacja': 0})
            if not o['skladowe']:
                return '0'
            cz = [f'{c.capitalize()} ψ(λ = {lam[j]["re"]:+.3f}{lam[j]["im"]:+.3f}i)' for j, c in o['skladowe'] if j in lam]
            return ('√(' + '² + '.join(cz) + '²)') if o['kombinacja'] == 1 and len(cz) > 1 else \
                   ('|' + cz[0] + '|' if o['kombinacja'] == 1 else ' + '.join(cz))
        tryb = 'automatyczne' if info.get('automatyczne') else 'ręczne'
        self.koop_label.setText(f'Operator Koopmana (Hankel-DMD, {info["d"]} opóźnienia, {info["pary"]:,} par stanów, '
                                f'zapis co {info["krok"]:.0f} kroków treningu)'
                                + (f'; pominięto {info["zdegenerowane"]} modów o zerowej wariancji (artefakty zanurzenia)'
                                   if info.get('zdegenerowane') else '') + f'. Osie {tryb}:\n'
                                f'X = {opis("x")}\nY = {opis("y")}\nZ = {opis("z")}')
        self._koop_osie = osie
        for os_, c in self.combo_kkomb.items():
            c.blockSignals(True); c.setCurrentIndex(int(osie.get(os_, {}).get('kombinacja', 0))); c.blockSignals(False)
        self._tabela_koopmana()

    def _mody_widoczne(self):
        """Mody w kolejności sortowania i po filtrach znaków (lista słowników z koop_info)."""
        tol = 1e-7
        mody = [m for m in self._koop_mody
                if {0: True, 1: m['re'] > 0, 2: m['re'] < 0}[self.combo_kre.currentData()]
                and {0: True, 1: m['im'] > tol, 2: m['im'] < -tol, 3: abs(m['im']) <= tol}[self.combo_kim.currentData()]]
        klucze = {0: lambda m: abs(complex(m['re'], m['im']) - 1), 1: lambda m: -m['modul'], 2: lambda m: abs(m['kat']),
                  3: lambda m: -abs(m['kat']), 4: lambda m: -m['re'], 5: lambda m: -m['im'],
                  6: lambda m: -abs(complex(m['re'], m['im']) - 1)}
        return sorted(mody, key=klucze[self.combo_ksort.currentData()])

    def _tabela_koopmana(self):
        from PyQt6.QtWidgets import QTableWidgetItem
        mody = self._mody_widoczne()
        osie = getattr(self, '_koop_osie', {}) or {}
        wybrane = {(os_, j, c) for os_, o in osie.items() for j, c in o['skladowe']}
        t = self.tab_koop
        t.blockSignals(True)
        t.setRowCount(len(mody))
        for r, m in enumerate(mody):
            okres = f'{m["okres"]:.0f}' if m['okres'] else '—'
            for k, tekst in zip((0, 7, 8, 9, 10, 11, 12),
                                (str(r + 1), f'{m["re"]:+.3f}', f'{m["im"]:+.3f}', f'{m["modul"]:.3f}',
                                 f'{abs(complex(m["re"], m["im"]) - 1):.3f}', f'{m["kat"]:+.0f}', okres)):
                it = QTableWidgetItem(tekst); it.setData(Qt.ItemDataRole.UserRole, m['j']); t.setItem(r, k, it)
            for k, (os_, c) in enumerate((('x', 're'), ('x', 'im'), ('y', 're'), ('y', 'im'), ('z', 're'), ('z', 'im'))):
                it = QTableWidgetItem()
                it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                it.setCheckState(Qt.CheckState.Checked if (os_, m['j'], c) in wybrane else Qt.CheckState.Unchecked)
                it.setData(Qt.ItemDataRole.UserRole, (os_, m['j'], c))
                t.setItem(r, 1 + k, it)
        t.resizeColumnsToContents()
        t.blockSignals(False)
        self.spin_kod.setMaximum(max(len(mody), 1)); self.spin_kdo.setMaximum(max(len(mody), 1))

    def _osie_koopmana(self, osie):
        """Nowe przypisanie osi (None = automatyczne) -> karta przelicza wagi i zakres osi."""
        self._osie_reczne = osie
        self._display_options()

    def _z_tabeli_koopmana(self):
        """Przypisanie osi ze stanu tabeli (pola zaznaczone także w modach ukrytych przez filtry zostają)."""
        if not self._koop_mody:
            return
        osie = {os_: dict(skladowe=list(o['skladowe']), kombinacja=self.combo_kkomb[os_].currentData())
                for os_, o in (getattr(self, '_koop_osie', {}) or {}).items()}
        for os_ in ('x', 'y', 'z'):
            osie.setdefault(os_, dict(skladowe=[], kombinacja=self.combo_kkomb[os_].currentData()))
        t = self.tab_koop
        for r in range(t.rowCount()):
            for k in range(1, 7):
                it = t.item(r, k)
                if it is None:
                    continue
                os_, j, c = it.data(Qt.ItemDataRole.UserRole)
                sk = osie[os_]['skladowe']
                if it.checkState() == Qt.CheckState.Checked and (j, c) not in sk:
                    sk.append((j, c))
                elif it.checkState() != Qt.CheckState.Checked and (j, c) in sk:
                    sk.remove((j, c))
        self._osie_koopmana(osie)

    def _przypisz_koopmana(self, akcja):
        """Szybkie przypisanie: mody od..do (w kolejności tabeli) i wybrane części na wybraną oś."""
        if not self._koop_mody:
            self._set_text(self.koop_label, 'Najpierw wybierz układ „Koopman”, aby dopasować operator.'); return
        self._z_tabeli_koopmana()                       # punkt wyjścia: bieżące przypisanie
        osie = {k: dict(skladowe=list(v['skladowe']), kombinacja=v['kombinacja']) for k, v in self._osie_reczne.items()}
        os_ = self.combo_kos.currentData()
        mody = self._mody_widoczne()[self.spin_kod.value() - 1:self.spin_kdo.value()]
        nowe = [(m['j'], c) for m in mody for c in self.combo_kczesci.currentData()]
        if akcja == 'czysc':
            osie[os_]['skladowe'] = []
        elif akcja == 'ustaw':
            osie[os_]['skladowe'] = nowe
        else:
            osie[os_]['skladowe'] += [n for n in nowe if n not in osie[os_]['skladowe']]
        self._osie_koopmana(osie)

    # ------------------------------------------------------------------ osie Koopmana pod klastry
    def _klastry_koopmana(self):
        """Krok 1: próbka stanów Hankela cząsteczek widoku z karty (wynik przez sygnał probka_koopmana)."""
        if not self.timeline:
            self._set_text(self.kl_label, 'Osie pod klastry: najpierw wczytaj oś czasu.'); return
        if self.watek is not None:
            self._set_text(self.kl_label, 'Osie pod klastry: poczekaj, aż skończy się bieżące zadanie w tle.'); return
        if self.combo_uklad.currentData() != 3:
            self.combo_uklad.setCurrentIndex(self.combo_uklad.findData(3))      # dopasowanie Koopmana, potem próbka
        self._set_text(self.kl_label, 'Osie pod klastry: pobieram próbkę cząsteczek z karty…')
        self.cloud.zadaj_probke_koopmana(self.spin_kl_m.value(), self.combo_kl_kiedy.currentData())
        self._budz()

    def _klastry_probka(self, p):
        """Krok 2: ocena par i trójek osi w tle (k-means na karcie przez torch), potem nowe osie."""
        if 'blad' in p:
            self._set_text(self.kl_label, p['blad']); return
        if len(p['H']) < 50:
            self._set_text(self.kl_label, f'Osie pod klastry: w widoku tylko {len(p["H"])} widocznych cząsteczek '
                           '(potrzeba co najmniej 50). Zmień filtry albo zakres.'); return
        if self.watek is not None:
            self._set_text(self.kl_label, 'Osie pod klastry: poczekaj, aż skończy się bieżące zadanie w tle.'); return
        import klastry_koopmana as KK
        ustaw = dict(z_tez=self.chk_kl_z.isChecked(), k_max=self.spin_kl_k.value(), prog=self.spin_kl_prog.value(),
                     waga_out=self.spin_kl_waga.value())
        d = p['d']

        def licz(progress, cancel):
            grupy = p['grupy'] if len(p['kroki']) > 1 else None
            return KK.dobierz(p['H'], p['W'], p['war'], p['prog'], p['vals'], p['zdeg'], osie_obecne=p['osie'],
                              grupy=grupy, grupa_modelu=p['biezacy'],
                              postep=lambda t, f: progress({'phase': 'clusters', 'message': t, 'fraction': f}), **ustaw)

        def gotowe(w):
            if w is None:
                return
            if self.spin_kd.value() != d:                  # w międzyczasie zmieniono opóźnienia: inne mody
                self._set_text(self.kl_label, 'Osie pod klastry: zmieniono liczbę opóźnień, kliknij ponownie.'); return
            kroki = p['kroki']
            gdzie = f'zapis {kroki[0]}' if len(kroki) == 1 else f'{len(kroki)} zapisów ({kroki[0]}..{kroki[-1]})'
            self._set_text(self.kl_label, KK.opis(w, p['vals']) + f'\nOceniano: {gdzie}, k = 2..{ustaw["k_max"]}, '
                           f'outlier od {ustaw["prog"]:.1f} σ, kara {ustaw["waga_out"]:.1f}.')
            if w['model'] is None:
                return
            self.cloud.ustaw_klastry(w['model'])
            self._osie_koopmana(w['osie'])
            if self.chk_kl_kolor.isChecked():
                self.combo_color.setCurrentIndex(self.combo_color.findData(7))
            self._set_text(self.status, 'Osie pod klastry ustawione (opis na stronie „Koopman”).')
        self._set_text(self.kl_label, f'Osie pod klastry: oceniam {len(p["H"]):,} cząsteczek…')
        self._job(licz, gotowe)

    # ------------------------------------------------------------------ fizyka ruchu i obszary oka
    WLASCIWOSCI = {0: 'temperatura (energia drgań własnych, log)', 1: 'wykładnik MSD α',
                   2: 'VACF: cos kolejnych przesunięć', 3: 'prostoliniowość: netto / droga',
                   4: 'tempo powrotu κ (Ornstein-Uhlenbeck)', 5: 'ruch w rozgrzewce (log)',
                   6: 'zmiana 768D na końcu okna (log)', 8: 'okres dominujący ruchu (zapisy)',
                   9: 'udział energii ruchu w paśmie okresów', 10: 'energia kinetyczna ½⟨|Δx|²⟩ (log)',
                   11: 'energia potencjalna ½κ⟨|x − x̄|²⟩ (log)', 12: 'energia całkowita (log)'}
    WL_LOG = (0, 5, 6, 10, 11, 12)
    OBSZARY = ('tło', 'rogówka', 'źrenica', 'narzędzie')
    KOLORY_OBSZAROW = ('#8a8f99', '#4cd964', '#4a8cff', '#ff4433')

    def _strona_fizyki(self):
        self.opcje.dodaj_strone('Fizyka ruchu')
        row = self.opcje.wiersz('Fizyka ruchu')
        self.spin_fz_okno = self._int(row, 'okno (zapisy)', 0, 2000, 5,
                                      'Ile zapisów w każdą stronę bieżącego kroku liczyć do właściwości ruchu. Okno ruchome '
                                      'pokazuje, jak reżimy zmieniają się w czasie uczenia: karta liczy okna wokół obu '
                                      'sąsiednich zapisów i układ „Fizyka ruchu” płynnie przechodzi między nimi. 0 = cała oś '
                                      'czasu: właściwości stałe, więc w tym układzie cząsteczki stoją podczas odtwarzania.')
        self.spin_fz_rozg = self._int(row, 'koniec rozgrzewki (krok)', 0, 10000000, 60,
                                      'Krok treningu, do którego liczony jest „ruch w rozgrzewce” (od kroku 0). Dla '
                                      'historii K1000 kolektywny przepływ trwa do około 30–60 kroków; dla historii po '
                                      'epokach pierwszy zapis to koniec epoki 1.')
        self.spin_fz_tau = self._int(row, 'τ dla α (zapisy)', 2, 100, 4,
                                     'Opóźnienie do wykładnika MSD: α = log(MSD(τ) / MSD(1)) / log τ. 1 = dyfuzja, '
                                     'mniej = uwięzienie (subdyfuzja), 2 = ruch balistyczny.')
        row = self.opcje.wiersz('Fizyka ruchu')
        self.combo_fz_osie = []
        for nazwa, dom in (('X', 0), ('Y', 6), ('Z', 1)):
            c = QComboBox()
            for i, t in self.WLASCIWOSCI.items():
                c.addItem(t, i)
            c.setCurrentIndex(c.findData(dom))
            dodaj(row, f'oś {nazwa}', c, f'Która właściwość ruchu leży na osi {nazwa} układu „Fizyka ruchu”. Zakres osi: '
                  'percentyle 1–99 widocznych cząsteczek.')
            self.combo_fz_osie.append(c)
        row = self.opcje.wiersz('Fizyka ruchu')
        self.spin_fz_pa = self._float(row, 'próg α dyfuzji', 0, 2, .7, 2,
                                      'Reżim: dyfuzyjny, gdy α ≥ ten próg; uwięziony, gdy mniejsze (i nie balistyczny).')
        self.spin_fz_pb = self._float(row, 'próg balistyczny', 0, 1, .3, 2,
                                      'Reżim balistyczny: VACF i prostoliniowość większe od tego progu (cząsteczka jedzie '
                                      'w jednym kierunku, jak w rozgrzewce).')
        self.combo_fz_obszar = QComboBox()
        for i, t in enumerate(('wszystkie obszary',) + tuple(f'tylko {o}' for o in self.OBSZARY)):
            self.combo_fz_obszar.addItem(t, i)
        dodaj(row, 'pokaż', self.combo_fz_obszar,
              'Filtr widocznych cząsteczek po obszarze oka z segmentacji Cataract-LMM (annotation.csv: wielokąty '
              'narzędzia, źrenicy i rogówki). Bez segmentacji: narzędzie wg ramki GT, reszta jako tło.')
        self.combo_fz_hist = QComboBox()
        for i, t in self.WLASCIWOSCI.items():
            self.combo_fz_hist.addItem(t, i)
        dodaj(row, 'histogram', self.combo_fz_hist, 'Właściwość na histogramie w panelu bocznym (osobno dla obszarów oka).')
        self.btn_fz = przycisk(row, 'Odśwież statystyki', 'Odczytuje z karty właściwości wszystkich widocznych '
                               'cząsteczek i przelicza histogramy, AUC i mapę rozgrzewki (dzieje się też samo po '
                               'zmianie kroku w pauzie).')
        # --- częstotliwość ruchu i energia
        row = self.opcje.wiersz('Fizyka ruchu')
        self.spin_pas_lo = self._float(row, 'pasmo: okres od (zapisy)', 2, 10000, 2, 1,
                                       'Dolna granica pasma okresów ruchu dla koloru „Udział energii w paśmie” (16). Okres w '
                                       'zapisach: 2 = zawracanie co zapis (najszybsze drgania), duże wartości = wolny dryf. Widmo '
                                       'liczy karta z przyrostów cząsteczki w oknie (najwyżej 128 kolejnych przyrostów ze środka '
                                       'okna), z pominięciem średniego dryfu (f = 0).')
        self.spin_pas_hi = self._float(row, 'do', 2, 10000, 4, 1, 'Górna granica pasma okresów (zapisy, włącznie).')
        self.pas_label = rola(QLabel(''), 'opis'); row.addWidget(self.pas_label)
        self.combo_fz_en = QComboBox()
        for t, v in (('kinetyczna ½⟨|Δx|²⟩', 0), ('potencjalna ½κ⟨|x − x̄|²⟩', 1), ('całkowita', 2)):
            self.combo_fz_en.addItem(t, v)
        dodaj(row, 'energia (kolor 15)', self.combo_fz_en,
              'Którą energię pokazuje kolor „Energia cząsteczki”: kinetyczna = połowa średniego kwadratu przesunięcia na zapis '
              '(z ruchem kolektywnym), potencjalna = energia sprężyny Ornsteina-Uhlenbecka ½κ⟨|x − x̄|²⟩ wokół średniego '
              'położenia w oknie, całkowita = suma. Skala log, zakres z percentyli 2–98.')
        # --- zakresy z kolorami (kolor 17)
        row = self.opcje.wiersz('Fizyka ruchu')
        self.combo_zak_c = QComboBox()
        for i, t in self.WLASCIWOSCI.items():
            self.combo_zak_c.addItem(t, i)
        self.combo_zak_c.setCurrentIndex(self.combo_zak_c.findData(8))
        dodaj(row, 'zakresy (kolor 17)', self.combo_zak_c,
              'Właściwość kolorowana zakresami: cząsteczka dostaje kolor pierwszego zakresu, do którego należy jej wartość '
              '(w jednostkach właściwości, bez logarytmu), reszta jest szara. Np. okres dominujący 2–3 zapisy = szybkie '
              'drgania, 3–6 = średnie, powyżej 6 = wolne; albo temperatura czy energia.')
        self.zak_wiersze = []
        for q, (od, do, kol) in enumerate(((2.0, 3.0, '#ff4026'), (3.0, 6.0, '#ffd933'), (6.0, 1e6, '#4d99ff'))):
            chk = pole_wyboru(row, f'#{q + 1}', f'Włącza zakres {q + 1}.', zaznaczone=True)
            lo = QDoubleSpinBox(); lo.setRange(-1e9, 1e9); lo.setDecimals(4); lo.setValue(od); lo.setMaximumWidth(95)
            hi = QDoubleSpinBox(); hi.setRange(-1e9, 1e9); hi.setDecimals(4); hi.setValue(do); hi.setMaximumWidth(95)
            lo.setToolTip(f'Początek zakresu {q + 1} (włącznie).'); hi.setToolTip(f'Koniec zakresu {q + 1} (włącznie).')
            btn = QPushButton(' '); btn.setMaximumWidth(36); btn.setToolTip(f'Kolor zakresu {q + 1}: kliknij, aby zmienić.')
            btn.setProperty('kolor', kol); btn.setStyleSheet(f'background:{kol};')
            btn.clicked.connect(lambda _, b=btn: self._wybierz_kolor_zakresu(b))
            for w in (lo, hi, btn):
                row.addWidget(w)
            self.zak_wiersze.append((chk, lo, hi, btn))
        self.btn_zak_tercyle = przycisk(row, 'Zakresy z tercyli', 'Ustawia trzy zakresy na tercyle wybranej właściwości '
                                        'widocznych cząsteczek (po jednej trzeciej w każdym kolorze).')
        self.fz_label = rola(QLabel('Fizyka ruchu: wybierz układ „Fizyka ruchu” albo kolor temperatury, reżimu lub α.'), 'opis')
        self.fz_label.setWordWrap(True)
        self.opcje._strony['Fizyka ruchu'].addWidget(self.fz_label)
        for w in (self.spin_fz_okno, self.spin_fz_rozg, self.spin_fz_tau, self.spin_fz_pa, self.spin_fz_pb):
            w.valueChanged.connect(lambda _: self._display_options())
        for c in self.combo_fz_osie + [self.combo_fz_obszar]:
            c.currentIndexChanged.connect(lambda _: self._display_options())
        self.combo_fz_hist.currentIndexChanged.connect(lambda _: self._fizyka_wykresy())
        self.btn_fz.clicked.connect(lambda: (self.cloud.zadaj_odczyt_fizyki(), self._budz()))
        for w in (self.spin_pas_lo, self.spin_pas_hi):
            w.valueChanged.connect(lambda _: self._display_options())
        for c in (self.combo_fz_en, self.combo_zak_c):
            c.currentIndexChanged.connect(lambda _: self._display_options())
        for chk, lo, hi, _ in self.zak_wiersze:
            chk.toggled.connect(lambda _: self._display_options())
            lo.valueChanged.connect(lambda _: self._display_options()); hi.valueChanged.connect(lambda _: self._display_options())
        self.btn_zak_tercyle.clicked.connect(self._zakresy_z_tercyli)
        self._fz_wynik = None
        self._fz_ost = 0.0
        self._fz_klucz = None
        self._obszary_klasy = None

    def _panel_fizyki(self, rl):
        self.fz_box = QWidget(); fl = QVBoxLayout(self.fz_box); fl.setContentsMargins(0, 0, 0, 0)
        self.fz_opis = staly_napis(rola(QLabel('Fizyka ruchu'), 'opis'), 7)
        fl.addWidget(self.fz_opis)
        self.fz_hist = pg.PlotWidget(title='Rozkład właściwości: obszary oka'); self.fz_hist.setMinimumHeight(170)
        self.fz_hist.addLegend(offset=(5, 5))
        self.fz_temp = pg.PlotWidget(title='Temperatura w czasie (energia drgań własnych na zapis)')
        self.fz_temp.setMinimumHeight(170); self.fz_temp.setLogMode(y=True); self.fz_temp.addLegend(offset=(5, 5))
        self.fz_cos = pg.PlotWidget(title='Reżim w czasie: cos kolejnych przesunięć (> próg = balistyczny)')
        self.fz_cos.setMinimumHeight(150); self.fz_cos.setYRange(-1, 1)
        self.fz_mapa = pg.ImageView(); self.fz_mapa.ui.roiBtn.hide(); self.fz_mapa.ui.menuBtn.hide()
        self.fz_mapa.setMinimumHeight(150)
        self.fz_mapa_label = staly_napis(QLabel('Ruch w rozgrzewce na siatce patchy (mediana po obrazach)'), 2)
        self._fz_kursory = []
        for w in (self.fz_temp, self.fz_cos):
            k = pg.InfiniteLine(angle=90, pen='#ffffff'); w.addItem(k); self._fz_kursory.append(k)
            w.setLabel('bottom', 'krok treningu')
        for w in (self.fz_hist, self.fz_temp, self.fz_cos, self.fz_mapa_label, self.fz_mapa):
            fl.addWidget(w)
        self.fz_box.setVisible(False)
        rl.addWidget(self.fz_box)

    def _obszary_oka(self, timeline):
        """W tle: klasy patchy z segmentacji Cataract-LMM (raz na oś czasu, potem z pliku obok osi)."""
        from concurrent.futures import ThreadPoolExecutor
        import obszary_oka as O
        self._obszary_klasy = None
        self.cloud.ustaw_obszary(None)
        item = self.combo_color.model().item(self.combo_color.findData(11))
        if item is not None:
            item.setEnabled(False)
        if not hasattr(self, '_obszary_pula'):
            self._obszary_pula = ThreadPoolExecutor(1, thread_name_prefix='obszary-oka')
        fut = self._obszary_pula.submit(O.dla_osi, timeline)

        def sprawdz():
            if timeline is not self.timeline:
                return
            if not fut.done():
                QTimer.singleShot(200, sprawdz); return
            try:
                klasy, _, brak = fut.result()
            except Exception as e:                                   # brak frame_id, zipów albo csv: bez obszarów
                self._set_text(self.fz_label, f'Obszary oka niedostępne ({e}); narzędzie wg ramki GT, reszta jako tło.')
                return
            self._obszary_klasy = klasy.reshape(-1)
            self.cloud.ustaw_obszary(self._obszary_klasy)
            if item is not None:
                item.setEnabled(True)
            udzialy = ', '.join(f'{o} {100 * (klasy == c).mean():.1f} %' for c, o in enumerate(self.OBSZARY))
            self._set_text(self.fz_label, f'Obszary oka z segmentacji: {udzialy}' +
                           (f'; {int(brak.sum())} zdjęć bez wpisu w annotation.csv (tło)' if brak.any() else '') + '.')
            self._budz()
        QTimer.singleShot(200, sprawdz)

    def _ustaw_fizyke(self):
        """Parametry fizyki do karty; odczyt statystyk w pauzie po zmianie, w ruchu najwyżej co 1 s."""
        self.cloud.ustaw_fizyke(okno=self.spin_fz_okno.value(), rozgrzewka=self.spin_fz_rozg.value(),
                                tau=self.spin_fz_tau.value(), prog_a=self.spin_fz_pa.value(), prog_b=self.spin_fz_pb.value(),
                                osie=[c.currentData() for c in self.combo_fz_osie])
        self.cloud.ustaw_filtr_obszaru(self.combo_fz_obszar.currentData())
        from PyQt6.QtGui import QColor
        zak = [(lo.value(), hi.value(), 1.0 if chk.isChecked() else 0.0, 0.0) for chk, lo, hi, _ in self.zak_wiersze]
        zkol = [QColor(b.property('kolor')).getRgbF()[:3] for *_, b in self.zak_wiersze]
        self.cloud.ustaw_kolory_fizyki(energia=self.combo_fz_en.currentData(), zak_c=self.combo_zak_c.currentData(),
                                       zak=zak, zkol=zkol, pas=(self.spin_pas_lo.value(), max(self.spin_pas_hi.value(),
                                                                                              self.spin_pas_lo.value())))
        if self.timeline is not None and len(self.timeline.steps) > 1:
            dk = float(np.median(np.diff(self.timeline.steps)))
            self._set_text(self.pas_label, f'≈ {self.spin_pas_lo.value() * dk:.0f}–{self.spin_pas_hi.value() * dk:.0f} kroków '
                           f'treningu (mediana odstępu zapisów {dk:.0f})')
        aktywna = self.combo_uklad.currentData() == 4 or self.combo_color.currentData() in (8, 9, 10, 11, 15, 16, 17)
        if self.combo_uklad.currentData() == 4 and self.spin_fz_okno.value() == 0 and self._playing:
            self._set_text(self.status, 'Układ „Fizyka ruchu” z oknem 0: właściwości z całej osi czasu są stałe, więc '
                           'cząsteczki stoją. Ustaw „okno (zapisy)” > 0 na stronie „Fizyka ruchu”, aby ruszały się w czasie.')
        self.fz_box.setVisible(aktywna)
        if not aktywna or not self.timeline:
            return
        first, count = self._scope()
        klucz = (str(self.timeline.root), first, count, self.spin_fz_okno.value(), self.spin_fz_rozg.value(),
                 self.spin_fz_tau.value(), self.spin_fz_pa.value(), self.spin_fz_pb.value(),
                 self._indices()[0] if self.spin_fz_okno.value() else None, self._obszary_klasy is not None,
                 self.spin_pas_lo.value(), self.spin_pas_hi.value(), self.combo_fz_en.currentData())
        teraz = time.perf_counter()
        if klucz != self._fz_klucz and (not self._playing or teraz - self._fz_ost > 1.0):
            self._fz_klucz, self._fz_ost = klucz, teraz
            self.cloud.zadaj_odczyt_fizyki()

    @staticmethod
    def _auc(a, b):
        """P(a > b) z rang (Mann-Whitney); None, gdy któraś grupa pusta."""
        if len(a) == 0 or len(b) == 0:
            return None
        x = np.concatenate([a, b])
        r = np.empty(len(x)); r[np.argsort(x, kind='mergesort')] = np.arange(1, len(x) + 1)
        return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))

    @classmethod
    def _wartosc(cls, wl, c):
        return (wl[:, 10] + wl[:, 11] if c == 12 else wl[:, c]).astype(np.float64)

    @classmethod
    def _przeksztalc(cls, wl, c):
        v = cls._wartosc(wl, c)
        return np.log(np.maximum(v, 1e-12)) if c in cls.WL_LOG else v

    def _wybierz_kolor_zakresu(self, btn):
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        k = QColorDialog.getColor(QColor(btn.property('kolor')), self, 'Kolor zakresu')
        if k.isValid():
            btn.setProperty('kolor', k.name()); btn.setStyleSheet(f'background:{k.name()};')
            self._display_options()

    def _zakresy_z_tercyli(self):
        if self._fz_wynik is None:
            self._set_text(self.fz_label, 'Najpierw odczytaj właściwości: układ „Fizyka ruchu” albo kolor 8–17, potem '
                           '„Odśwież statystyki”.'); return
        v = self._wartosc(self._fz_wynik['wl'], self.combo_zak_c.currentData())
        q = np.percentile(v, [0, 100 / 3, 200 / 3, 100])
        for (chk, lo, hi, _), a, b in zip(self.zak_wiersze, q[:-1], q[1:]):
            lo.blockSignals(True); hi.blockSignals(True)
            lo.setValue(float(a)); hi.setValue(float(b)); chk.setChecked(True)
            lo.blockSignals(False); hi.blockSignals(False)
        self._display_options()

    def _fizyka_gotowa(self, w):
        """Odczyt z karty: zakresy osi i kolorów, klasy obszarów, potem wykresy panelu."""
        wl = w['wl']
        if not len(wl):
            return
        osie = [c.currentData() for c in self.combo_fz_osie]
        tr = [self._przeksztalc(wl, c) for c in osie]
        lo = [float(np.percentile(t, 1)) for t in tr]; hi = [float(np.percentile(t, 99)) for t in tr]
        hi = [h if h > l else l + 1e-6 for l, h in zip(lo, hi)]
        lt = np.log(np.maximum(wl[:, 0], 1e-12))
        kol = (float(np.percentile(lt, 2)), float(np.percentile(lt, 98)),
               float(np.percentile(wl[:, 1], 2)), float(np.percentile(wl[:, 1], 98)))
        if (tuple(lo), tuple(hi), kol) != (self.cloud.fiz['lo'], self.cloud.fiz['hi'], self.cloud.fiz['kol']):
            self.cloud.ustaw_zakresy_fizyki(lo, hi, kol)
        le = self._przeksztalc(wl, {0: 10, 1: 11, 2: 12}[self.combo_fz_en.currentData()])
        self.cloud.ustaw_kolory_fizyki(en_kol=(float(np.percentile(le, 2)), float(np.percentile(le, 98)), 0.0, 0.0))
        klasy = w.get('obszary')
        if klasy is None:
            m = getattr(self.cloud, '_maska', None)
            klasy = (np.asarray(m[w['first']:w['first'] + w['count']]) & 1).astype(np.uint8) * 3 if m is not None \
                else np.zeros(len(wl), np.uint8)
        self._fz_wynik = dict(w, klasy=np.asarray(klasy))
        self._fizyka_wykresy()
        self._budz()

    def _fizyka_wykresy(self):
        w = self._fz_wynik
        if w is None:
            return
        wl, klasy = w['wl'], w['klasy']
        c = self.combo_fz_hist.currentData()
        v = self._przeksztalc(wl, c)
        lo, hi = np.percentile(v, [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1e-6, 61)
        self.fz_hist.clear(); self.fz_hist.addLegend(offset=(5, 5))
        for k, (nazwa, kol) in enumerate(zip(self.OBSZARY, self.KOLORY_OBSZAROW)):
            m = klasy == k
            if m.sum() < 20:
                continue
            h, e = np.histogram(v[m], bins=bins, density=True)
            self.fz_hist.plot(e, h, stepMode='center', pen=pg.mkPen(kol, width=2), name=f'{nazwa} ({m.sum():,})')
        self.fz_hist.setLabel('bottom', self.WLASCIWOSCI.get(c, ''))
        # AUC względem tła i udziały reżimów
        tlo = klasy == 0
        linie = []
        for k in (3, 2, 1):
            m = klasy == k
            a_t = self._auc(np.log(np.maximum(wl[m, 0], 1e-12)), np.log(np.maximum(wl[tlo, 0], 1e-12)))
            a_z = self._auc(np.log(np.maximum(wl[m, 6], 1e-12)), np.log(np.maximum(wl[tlo, 6], 1e-12)))
            if a_t is None:
                continue
            rez = wl[m, 7]
            linie.append(f'{self.OBSZARY[k]} vs tło: AUC temperatury {a_t:.2f}, zmiany 768D {a_z:.2f}; reżimy bal./dyf./uw. '
                         f'{100 * (rez == 0).mean():.0f}/{100 * (rez == 1).mean():.0f}/{100 * (rez == 2).mean():.0f} %')
        rez = wl[:, 7]
        med_t = ' · '.join(f'{self.OBSZARY[k]} {np.median(wl[klasy == k, 0]):.3g}' for k in range(4) if (klasy == k).sum() >= 20)
        med_t += '.\nMediana okresu dominującego (zapisy) / udział pasma: ' + ' · '.join(
            f'{self.OBSZARY[k]} {np.median(wl[klasy == k, 8]):.1f} / {np.mean(wl[klasy == k, 9]):.2f}'
            for k in range(4) if (klasy == k).sum() >= 20)
        self._set_text(self.fz_opis, f'Okno {w["okno"][0]}–{w["okno"][1]}, rozgrzewka do kroku {w["rozgrzewka"]}, τ = {w["tau"]}. '
                       f'{len(wl):,} cząsteczek; reżimy bal./dyf./uw.: {100 * (rez == 0).mean():.0f}/'
                       f'{100 * (rez == 1).mean():.0f}/{100 * (rez == 2).mean():.0f} %.\nMediana temperatury: {med_t}.\n'
                       + '\n'.join(linie))
        # serie populacji w czasie
        s = w.get('seria')
        if s is not None:
            kr = np.asarray(s['kroki'], np.float64)
            self.fz_temp.clear(); self.fz_temp.addLegend(offset=(5, 5))
            for k, (nazwa, kol) in enumerate(zip(self.OBSZARY, self.KOLORY_OBSZAROW)):
                t = s['temperatura'][:, k]
                ok = np.isfinite(t) & (t > 0)
                if ok.sum() > 1:
                    self.fz_temp.plot(kr[ok], t[ok], pen=pg.mkPen(kol, width=2), name=nazwa)
            ok = np.isfinite(s['pole']) & (s['pole'] > 0)
            if ok.sum() > 1:
                self.fz_temp.plot(kr[ok], s['pole'][ok], pen=pg.mkPen('#ffd166', width=1, style=Qt.PenStyle.DashLine),
                                  name='pole kolektywne |średni ruch|²')
            self.fz_cos.clear()
            prog = self.spin_fz_pb.value()
            self.fz_cos.addItem(pg.InfiniteLine(pos=prog, angle=0, pen=pg.mkPen('#ffd166', style=Qt.PenStyle.DashLine)))
            self.fz_cos.addItem(pg.InfiniteLine(pos=0, angle=0, pen='#555555'))
            for k, kol in enumerate(self.KOLORY_OBSZAROW):
                cs = s['cos'][:, k]
                ok = np.isfinite(cs)
                if ok.sum() > 1:
                    self.fz_cos.plot(kr[ok], cs[ok], pen=pg.mkPen(kol, width=2))
            for k, wdg in zip(self._fz_kursory, (self.fz_temp, self.fz_cos)):
                wdg.addItem(k)
        # mapa rozgrzewki na siatce patchy i udział wariancji tłumaczony przez położenie patcha
        patch = (w['first'] + np.arange(len(wl))) % 600
        r = np.log(np.maximum(wl[:, 5], 1e-12))
        sr = np.bincount(patch, r, minlength=600) / np.maximum(np.bincount(patch, minlength=600), 1)
        expl = 1 - ((r - sr[patch]) ** 2).mean() / max(r.var(), 1e-30)
        med = np.array([np.median(r[patch == t]) if (patch == t).any() else np.nan for t in range(600)])
        self.fz_mapa.setImage(np.nan_to_num(med.reshape(20, 30).T, nan=float(np.nanmin(med)) if np.isfinite(med).any() else 0),
                              autoLevels=True)
        self._set_text(self.fz_mapa_label, f'Ruch w rozgrzewce (log, mediana po obrazach) na siatce patchy. Położenie '
                       f'patcha tłumaczy {100 * expl:.0f} % jego wariancji.')

    def _zaznacz(self, patche):
        self.zaznaczone = set() if not patche else set(int(p) for p in patche)
        self.cloud.ustaw_zaznaczenie(self.zaznaczone)
        lista = ', '.join(str(p) for p in sorted(self.zaznaczone)[:20]) + (' …' if len(self.zaznaczone) > 20 else '')
        self.sel_label.setText(f'Zaznaczone patche ({len(self.zaznaczone)}): {lista}' if self.zaznaczone
                               else 'Zaznaczone patche: brak (Ctrl + klik na cząsteczce).')
        self._display_options()

    def _zdjecie_obrazu(self, i):
        """Zdjęcie obrazu i (320 × 480, jak wejście ViT) z manifestu pomiaru; SHA256 sprawdzane; None, gdy brak."""
        if not self.timeline:
            return None
        if i in self._zdj_cache:
            return self._zdj_cache[i]
        rgb = None
        try:
            m = self.timeline.metadata
            sciezki, obrazy = m.get('resolved_image_paths') or [], m.get('images') or []
            if i < len(sciezki):
                import hashlib
                raw = Path(sciezki[i]).read_bytes()
                if i >= len(obrazy) or hashlib.sha256(raw).hexdigest() == obrazy[i].get('image_sha256'):
                    rgb = R._private_module('dane').decode_resize(raw)
        except Exception as e:                             # brak pliku albo dekodera: bez zdjęcia, z komunikatem
            self._set_text(self.status, f'Zdjęcie obrazu {i} niedostępne: {e}')
        if len(self._zdj_cache) > 32:
            self._zdj_cache.clear()
        self._zdj_cache[i] = rgb
        return rgb

    def _dane_pomocnicze(self, timeline):
        """Maska patchy w ramce GT (z manifestu) i pewność ViT obrazów w zapisanych krokach (metrics.npy)."""
        maska = pewnosci = None
        m = getattr(timeline, 'metadata', None) or {}
        obrazy = m.get('images') or []
        tpi = timeline.count // max(timeline.image_count, 1) if timeline.image_count else 0
        if tpi == 600 and len(obrazy) == timeline.image_count and all('bbox_xyxy_480' in r for r in obrazy):
            b = np.asarray([r['bbox_xyxy_480'] for r in obrazy], np.float32)               # [obrazy, 4]
            t = np.arange(600); x, y = (t % 30) * 16 + 8, (t // 30) * 16 + 8
            maska = ((x[None] >= b[:, :1]) & (x[None] <= b[:, 2:3]) & (y[None] >= b[:, 1:2]) & (y[None] <= b[:, 3:4]))
            maska = maska.reshape(-1).astype(np.uint32)
        root = getattr(timeline, 'root', None)                  # atrapy osi czasu w testach nie mają folderu
        sciezka = Path(root) / 'metrics.npy' if root is not None else None
        kroki = m.get('metric_steps')
        if sciezka is not None and sciezka.is_file() and kroki:
            try:
                metr = np.load(sciezka, mmap_mode='r')
                idx = [kroki.index(s) for s in timeline.steps]
                pewnosci = np.nan_to_num(np.asarray(metr[:, idx, 0], np.float32).T, nan=0.0)   # [T, obrazy]
            except (ValueError, IndexError, OSError):
                pewnosci = None
        self.cloud.ustaw_dane_pomocnicze(maska, pewnosci)
        self.chk_tylko_narz.setEnabled(maska is not None)
        for indeks, jest in ((4, maska is not None), (5, pewnosci is not None)):
            item = self.combo_color.model().item(indeks)
            if item is not None:
                item.setEnabled(jest)

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
        i0, i1 = self._zakres_idx()
        if self._visual_step >= self.timeline.steps[i1] or self._visual_step < self.timeline.steps[i0]:
            self._seek_index(i0)
        self._playing = True; self._last_tick = time.perf_counter(); self.btn_play.setText('Pauza')
        self._budz()

    # ------------------------------------------------------------------ nagrywanie filmu z odtwarzania
    def _nagraj_film_dialog(self):
        if not self.timeline:
            self._error('Najpierw wczytaj oś czasu.'); return
        domyslna = Path(self.timeline.root) / f'film_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        sciezka, _ = QFileDialog.getSaveFileName(self, 'Zapisz film', str(domyslna), 'Film MP4 (*.mp4)')
        if not sciezka:
            return
        i0, i1 = self._zakres_idx()
        od, do = self.timeline.steps[i0], self.timeline.steps[i1]
        sek = (do - od) / max(self.spin_tempo.value(), 1e-6)
        self.status.setText(f'Nagrywam {sciezka}: kroki {od}–{do}, {sek:.1f} s przy {self.spin_fps.value()} kl./s…')
        try:
            n = self.nagraj_film(sciezka, fps=self.spin_fps.value(), postep=lambda f, nn: self.pasek.setValue(int(100 * f / nn)))
            self.status.setText(f'Film zapisany: {sciezka} ({n} klatek, {sek:.1f} s).')
        except Exception as e:                                   # brak ffmpeg/kodeka albo przerwanie: komunikat
            self._error(f'nagrywanie filmu: {e}')

    def nagraj_film(self, sciezka, fps=30, sekundy=None, tempo=None, od=None, do=None, napis=None, klatka=None,
                    postep=None, limit_czekania=60.0):
        """Nagrywa odtwarzanie do MP4 z bieżącymi ustawieniami widoku. tempo: kroki treningu na sekundę filmu
        (domyślnie „tempo (kroki/s)”), sekundy: długość filmu zamiast tempa; od/do: kroki (domyślnie zakres
        „od–do”); napis(krok) -> tekst nad widokiem; klatka(obraz uint8 [H, W, 3], krok) -> obraz (dowolne złożenie).
        Klatki są renderowane deterministycznie (nie z zegara) i każda czeka na stany na karcie. Zwraca liczbę klatek."""
        import subprocess
        import shutil
        from PyQt6.QtWidgets import QApplication
        if not self.timeline:
            raise ValueError('brak osi czasu')
        self._pause()
        steps = self.timeline.steps
        i0, i1 = self._zakres_idx()
        od = steps[i0] if od is None else float(od); do = steps[i1] if do is None else float(do)
        if sekundy is None:
            sekundy = (do - od) / max(float(tempo if tempo is not None else self.spin_tempo.value()), 1e-6)
        n = max(1, int(round(sekundy * fps)))
        self.cloud.wczytaj_teraz([s for s in steps if od <= s <= do] or steps[:1])
        pisarz = None
        proc = None
        try:
            for f in range(n + 1):
                krok = od + (do - od) * f / n
                self._visual_step = float(krok); self._last_index = None
                t0 = time.perf_counter()
                while True:                                      # czekaj na stany pary na karcie
                    self.cloud.pompuj()
                    gotowe = self._render()
                    QApplication.processEvents()
                    if gotowe or time.perf_counter() - t0 > limit_czekania:
                        break
                    time.sleep(.01)
                img = self.view.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
                w, h = img.width(), img.height()
                buf = img.constBits(); buf.setsize(img.sizeInBytes())
                a = np.frombuffer(buf, np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3].reshape(h, w, 3).copy()
                if klatka is not None:
                    a = np.ascontiguousarray(klatka(a, krok))
                if napis is not None:
                    a = _napis_na_klatce(a, napis(krok))
                a = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2]        # parzyste wymiary (yuv420p)
                if pisarz is None and proc is None:
                    H, W = a.shape[:2]
                    ffmpeg = shutil.which('ffmpeg')
                    if ffmpeg:
                        proc = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                                 '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
                                                 '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(sciezka)],
                                                stdin=subprocess.PIPE)
                    else:
                        import cv2
                        pisarz = cv2.VideoWriter(str(sciezka), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
                        if not pisarz.isOpened():
                            raise RuntimeError('nie można otworzyć kodeka mp4v')
                if proc is not None:
                    proc.stdin.write(a.tobytes())
                else:
                    import cv2
                    pisarz.write(np.ascontiguousarray(a[:, :, ::-1]))
                if postep is not None:
                    postep(f, n)
        finally:
            if proc is not None:
                proc.stdin.close(); proc.wait()
                if proc.returncode:
                    raise RuntimeError(f'ffmpeg zakończył się kodem {proc.returncode}')
            if pisarz is not None:
                pisarz.release()
        return n + 1

    def _zakres_idx(self):
        """Indeksy (pierwszy, ostatni) zapisów zakresu odtwarzania; cała oś, gdy zakres nie jest ustawiony."""
        if not self.timeline:
            return 0, 0
        n = len(self.timeline.steps)
        i0, i1 = self.combo_zak_od.currentIndex(), self.combo_zak_do.currentIndex()
        if i0 < 0 or i1 < 0 or self.combo_zak_od.count() != n:
            return 0, n - 1
        return min(i0, i1), max(i0, i1)

    def _wypelnij_zakres(self, timeline):
        """Listy „od” i „do”: kroki zapisanych checkpointów osi (z numerem epoki dla historii 35 000)."""
        import historie_checkpointow as HC
        na_epoke = None
        sciezka = str(((getattr(timeline, 'metadata', None) or {}).get('config') or {}).get('checkpoint_dir', ''))
        if 'MalyViT_35000_od_zera' in sciezka or 'od_zera_35000' in sciezka:
            na_epoke = HC.historia('35k').na_epoke
        for c in (self.combo_zak_od, self.combo_zak_do):
            c.blockSignals(True); c.clear()
            for k in timeline.steps:
                c.addItem(f'{k} (epoka {k // na_epoke})' if na_epoke and k and k % na_epoke == 0 else str(k), k)
            c.setEnabled(len(timeline.steps) > 1)
        self.combo_zak_od.setCurrentIndex(0); self.combo_zak_do.setCurrentIndex(len(timeline.steps) - 1)
        for c in (self.combo_zak_od, self.combo_zak_do):
            c.blockSignals(False)

    def _zmiana_zakresu(self):
        if not self.timeline:
            return
        i0, i1 = self._zakres_idx()
        steps = self.timeline.steps
        self.suwak.blockSignals(True); self.suwak.setRange(i0, i1); self.suwak.blockSignals(False)
        if not steps[i0] <= self._visual_step <= steps[i1]:
            self._seek_index(i0)
        else:
            self._last_index = None; self._render(); self._budz()

    def _seek_index(self, index):
        self._pause()
        if self.timeline:
            i0, i1 = self._zakres_idx()
            index = max(i0, min(int(index), i1))
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
        i0, i1 = self._zakres_idx()
        if self._visual_step >= steps[i1]:
            if self.chk_petla.isChecked() and i1 > i0:
                self._visual_step = float(steps[i0])               # pętla: od początku zakresu, bez pauzy
                self._last_stats = None
            else:
                self._visual_step = float(steps[i1])
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
        from wspolne import ctrl_wcisniety
        if ctrl_wcisniety():                               # Ctrl + klik: patch do zaznaczenia albo z niego
            self._zaznacz(self.zaznaczone ^ {int(token)})
            return
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
