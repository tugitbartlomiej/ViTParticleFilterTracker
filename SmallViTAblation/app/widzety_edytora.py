"""Widżety i wątki zakładki „Edytor SD”: obraz z siatką patchy, lista suwaków kanałów, wątki modelu SD,
generacji DDIM i latentu ViT ze zdjęcia. Logika edycji wektora zostaje w app.Okno."""
import traceback
from pathlib import Path

import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

import silnik
from wspolne import KOLUMNY as KOLUMNY_SIATKI, WIERSZE as WIERSZE_SIATKI   # siatka patchy 20x30

SUWAK_MAX = 10000


class ObrazSiatka(QLabel):
    """Obraz z nałożoną siatką patchy 20x30; kliknięcie wybiera token (mikrowektor).

    Numeracja wierszami jak w adapterze modelu: token = wiersz*30 + kolumna,
    token 0 = lewy górny patch obrazu. Obraz rysowany w proporcji 3:2 (576x384),
    siatka i kliknięcia liczone w tym samym prostokącie.
    """

    tokenWybrany = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(576, 384)
        self.pix = None
        self.tekst = '(brak obrazu — ustaw wektor i kliknij GENERUJ)'
        self.token = 0
        self.pokaz_siatke = True
        self.podpis = ''
        self.setToolTip('Siatka patchy 20x30 (600 tokenów = mikrowektorów).\n'
                        'Kliknij patch, aby wybrać token (tryb przełączy się na „jeden token”).\n'
                        'Token 0 = lewy górny róg obrazu, numeracja wierszami.')

    def ustaw_pixmap(self, pix, podpis=''):
        self.pix = pix
        self.podpis = podpis
        self.update()

    def ustaw_token(self, token):
        self.token = int(token)
        self.update()

    def ustaw_siatke(self, pokaz):
        self.pokaz_siatke = bool(pokaz)
        self.update()

    def _obszar(self):
        w, h = self.width(), self.height()
        if w * 2 > h * 3:
            sz = h * 3 / 2
            return QRectF((w - sz) / 2, 0, sz, h)
        wy = w * 2 / 3
        return QRectF(0, (h - wy) / 2, w, wy)

    def paintEvent(self, zdarzenie):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#111'))
        o = self._obszar()
        if self.pix is not None:
            p.drawPixmap(o.toRect(), self.pix)
        else:
            p.setPen(QColor('#888'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.tekst)
        if self.pokaz_siatke:
            w = o.width() / KOLUMNY_SIATKI
            h = o.height() / WIERSZE_SIATKI
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            for i in range(1, KOLUMNY_SIATKI):
                x = int(o.left() + i * w)
                p.drawLine(x, int(o.top()), x, int(o.bottom()))
            for j in range(1, WIERSZE_SIATKI):
                y = int(o.top() + j * h)
                p.drawLine(int(o.left()), y, int(o.right()), y)
            wiersz, kolumna = divmod(self.token, KOLUMNY_SIATKI)
            cel = QRectF(o.left() + kolumna * w, o.top() + wiersz * h, w, h)
            p.fillRect(cel, QColor(255, 60, 60, 70))
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.drawRect(cel)
        if self.podpis:
            p.setPen(QColor(255, 255, 255))
            p.drawText(o.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, self.podpis)
        p.end()

    def mousePressEvent(self, zdarzenie):
        o = self._obszar()
        x, y = zdarzenie.position().x(), zdarzenie.position().y()
        if zdarzenie.button() == Qt.MouseButton.LeftButton and o.contains(x, y):
            kolumna = min(max(int((x - o.left()) / (o.width() / KOLUMNY_SIATKI)), 0), KOLUMNY_SIATKI - 1)
            wiersz = min(max(int((y - o.top()) / (o.height() / WIERSZE_SIATKI)), 0), WIERSZE_SIATKI - 1)
            self.tokenWybrany.emit(wiersz * KOLUMNY_SIATKI + kolumna)
        super().mousePressEvent(zdarzenie)


class ListaKanalow(QWidget):
    """Lista wierszy etykieta + suwak + pole liczbowe; `kanaly[r]` mówi, który kanał pokazuje wiersz r.

    Używana dwa razy: strona 64 z 768 kanałów i stała lista 128 kanałów narzędzia (J128).
    Obie edytują ten sam wektor 768 wartości w oknie.
    """

    def __init__(self, okno, wierszy):
        super().__init__()
        self.okno = okno
        self.kanaly = list(range(wierszy))
        self.wiersze = []
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        for r in range(wierszy):
            wiersz_k = QHBoxLayout()
            wiersz_k.setSpacing(6)
            etykieta = QLabel(f'K{r:03d}')
            etykieta.setFixedWidth(96)
            suwak = QSlider(Qt.Orientation.Horizontal)
            suwak.setRange(0, SUWAK_MAX)
            suwak.setToolTip('Przeciągnij, aby zmienić wartość kanału (makrowektora).\n'
                             'Zakres suwaka wybierasz w „Zakres suwaków”; przy jednostce\n'
                             '„Zmiana o % limitu” suwak to −100%..+100% szerokości limitu kanału.\n'
                             'Dokładną wartość wpisz w polu po prawej.')
            suwak.valueChanged.connect(lambda v, r=r: okno._suwak_zmieniony(self, r, v))
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(4)
            spin.setSingleStep(0.05)
            spin.setFixedWidth(120)
            spin.setToolTip('Dokładna wartość kanału — przyjmuje też wartości spoza zakresu suwaka.\n'
                            'Absolutne: dowolna liczba. „% zakresu”: pozycja w zakresie suwaka.\n'
                            '„Zmiana o % limitu”: przesunięcie = % × (max − min) limitu kanału.')
            spin.valueChanged.connect(lambda v, r=r: okno._spin_zmieniony(self, r, v))
            wiersz_k.addWidget(etykieta)
            wiersz_k.addWidget(suwak, 1)
            wiersz_k.addWidget(spin)
            lay.addLayout(wiersz_k)
            self.wiersze.append((etykieta, suwak, spin))


class WatekModelu(QThread):
    gotowy = pyqtSignal(object)
    blad = pyqtSignal(str)

    def run(self):
        try:
            urzadzenie = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.gotowy.emit(silnik.zaladuj_model(urzadzenie))
        except Exception:
            self.blad.emit(traceback.format_exc())


class WatekGeneracji(QThread):
    postep = pyqtSignal(int, int)
    podglad = pyqtSignal(int, int, object)
    gotowy = pyqtSignal(object)
    blad = pyqtSignal(str)

    def __init__(self, trained, latent, kroki, ziarno, parent=None, **opcje):
        super().__init__(parent)
        self.trained, self.latent, self.kroki, self.ziarno = trained, latent, kroki, ziarno
        self.opcje = opcje

    def run(self):
        try:
            rgb = silnik.generuj(self.trained, self.latent, self.kroki, self.ziarno,
                                 lambda i, n: self.postep.emit(i, n),
                                 podglad=lambda i, n, x: self.podglad.emit(i, n, x), **self.opcje)
            self.gotowy.emit(rgb)
        except Exception:
            self.blad.emit(traceback.format_exc())


class WatekViT(QThread):
    gotowy = pyqtSignal(object, str)
    blad = pyqtSignal(str)

    def __init__(self, sciezka, parent=None):
        super().__init__(parent)
        self.sciezka = sciezka

    def run(self):
        try:
            urzadzenie = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.gotowy.emit(silnik.latent_z_obrazu(self.sciezka, urzadzenie), Path(self.sciezka).name)
        except Exception:
            self.blad.emit(traceback.format_exc())
