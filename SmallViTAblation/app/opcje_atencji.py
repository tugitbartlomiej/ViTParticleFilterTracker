"""Wspólny panel opcji atencji ViT dla widoków cząsteczek (2D i 3D).

Atencja małego ViT nie ma tokenu CLS: w każdym bloku jest macierz [głowy, 600, 600], gdzie wiersz i
(zapytanie) mówi, na które patche j (klucze) patrzy patch i, a wiersz sumuje się do 1. „Patch i ciągnie
z patcha j” znaczy więc: wynik atencji patcha i to średnia wartości patchy j z wagami A[i, j].

Co można pokazać: mapę na siatce 20x30 (średnia otrzymana, wiersz, kolumna, skupienie, zasięg,
atencja na siebie) oraz strzałki źródło -> cel (z wybranego patcha, do wybranego patcha, najsilniejszy
cel każdego patcha, najsilniejsze pary). Głowa: średnia, maksimum albo jedna. Bloki: jeden albo rollout
(Abnar i Zuidema 2020: iloczyn (A + I) / 2 przez kolejne bloki, przepływ przez warstwy).
"""
import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QVBoxLayout, QWidget
from uklad import Przeplyw, WaskiCombo, dodaj
from wspolne import KOLUMNY, WIERSZE
_t = np.arange(600)
ODLEGLOSC = np.hypot((_t[:, None] % KOLUMNY) - (_t[None] % KOLUMNY), (_t[:, None] // KOLUMNY) - (_t[None] // KOLUMNY))


class OpcjeAtencji(QWidget):
    zmiana = pyqtSignal()
    MAPY = ['średnia otrzymana: ile uwagi patch dostaje od wszystkich',
            'z wybranego patcha: na które patche patrzy (skąd ciągnie)',
            'do wybranego patcha: które patche na niego patrzą (komu oddaje)',
            'skupienie patcha: 1 − entropia jego atencji',
            'zasięg patcha: średnia odległość, na którą patrzy',
            'atencja patcha na samego siebie']
    STRZALKI = ['bez strzałek', 'z wybranego patcha: k najsilniejszych celów',
                'do wybranego patcha: k najsilniejszych źródeł', 'każdy patch: jego najsilniejszy cel',
                'k najsilniejszych par w całym obrazie']
    POTRZEBUJE_TOKENU_MAPY = (1, 2)
    POTRZEBUJE_TOKENU_STRZALEK = (1, 2)

    def __init__(self, z_kryciem=True, na_poczatku=()):
        """na_poczatku: kontrolki widoku wstawiane na początek pierwszego wiersza (np. „Atencja na obrazie”).
        Wiersze: włącznik i wygląd mapy | co pokazać (mapa, strzałki) | z której macierzy (blok, głowa)."""
        super().__init__()
        self.fn, self._bloki, self._cache = None, {}, {}
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        w1, w2, w3 = Przeplyw(), Przeplyw(), Przeplyw()
        for w in (w1, w2, w3):
            lay.addLayout(w)
        for k in na_poczatku:
            w1.addWidget(k)

        self.combo_mapa = dodaj(w2, 'Mapa:', WaskiCombo(18),
                                'Co pokazuje mapa cieplna na siatce 20x30 (A[i, j] = ile patch i patrzy na patch j):\n'
                                '- średnia otrzymana: średnia kolumny, czyli ile uwagi patch dostaje od wszystkich,\n'
                                '- z wybranego patcha: wiersz A[wybrany, :], skąd wybrany patch ciągnie informację,\n'
                                '- do wybranego patcha: kolumna A[:, wybrany], które patche ciągną z wybranego,\n'
                                '- skupienie: 1 − entropia / log 600; 1 = patrzy w jedno miejsce, 0 = równo wszędzie,\n'
                                '- zasięg: średnia odległość (w patchach) do miejsc, na które patrzy,\n'
                                '- na siebie: A[i, i], ile patch patrzy na samego siebie.\n'
                                'Wybrany patch ustawia się kliknięciem.')
        self.combo_mapa.addItems(self.MAPY)
        self.spin_blok = dodaj(w3, 'Blok:', QSpinBox(), 'Blok transformera ViT (1–6). 6 = ostatni, tuż przed latentem.')
        self.spin_blok.setRange(1, 6); self.spin_blok.setValue(6)
        self.combo_bloki = QComboBox()
        self.combo_bloki.setToolTip('Tylko ten blok: wagi atencji jednego bloku.\n'
                                    'Rollout: iloczyn (A + I) / 2 przez bloki 1..blok (Abnar i Zuidema 2020).\n'
                                    'Pokazuje, z których patchy wejścia pochodzi informacja tokenu po przejściu przez\n'
                                    'kolejne warstwy, z uwzględnieniem połączeń rezydualnych (I).')
        self.combo_bloki.addItems(['tylko ten blok', 'rollout bloków 1..blok'])
        w3.addWidget(self.combo_bloki)
        self.combo_glowa = dodaj(w3, 'Głowa:', QComboBox(), 'Średnia albo maksimum po głowach atencji, albo jedna głowa.\n'
                                 'Każda głowa uczy się innego wzorca, więc warto przejrzeć je osobno.')
        self.combo_glowa.addItems(['średnia', 'maksimum'])
        self.chk_bez_siebie = QCheckBox('Pomiń siebie')
        self.chk_bez_siebie.setToolTip('Zeruje atencję patcha na samego siebie (przekątna) i normuje wiersze od nowa.\n'
                                       'Przekątna często dominuje i zasłania, do których INNYCH patchy patch ciągnie.')
        w3.addWidget(self.chk_bez_siebie)
        self.combo_skala = dodaj(w1, 'Skala:', QComboBox(), 'Liniowa: wprost wagi.\nLogarytmiczna: widać też słabe wagi.\n'
                                 'Rangi: kolor = percentyl wartości (równy rozkład kolorów).')
        self.combo_skala.addItems(['liniowa', 'logarytmiczna', 'rangi'])
        self.combo_strz = dodaj(w2, 'Strzałki:', WaskiCombo(15),
                                'Strzałka źródło → cel: patch u nasady patrzy na patch na grocie (ciągnie z niego).\n'
                                'Grubość i jasność = waga atencji.\n'
                                '- z wybranego patcha: k celów, na które wybrany patrzy najmocniej,\n'
                                '- do wybranego patcha: k patchy, które najmocniej patrzą na wybrany,\n'
                                '- każdy patch: 600 strzałek do najsilniejszego celu każdego patcha (pole przepływu),\n'
                                '- najsilniejsze pary: k największych wag A[i, j] w całym obrazie (bez przekątnej).')
        self.combo_strz.addItems(self.STRZALKI)
        self.spin_k = dodaj(w2, 'k:', QSpinBox(), 'Liczba strzałek dla opcji „k najsilniejszych”.')
        self.spin_k.setRange(1, 600); self.spin_k.setValue(10)
        self.spin_grubosc = dodaj(w2, 'Grubość strzałek:', QDoubleSpinBox(),
                                  'Grubość najsilniejszej strzałki w pikselach; słabsze cieńsze, w 5 klasach według siły\n'
                                  'atencji (waga strzałki względem najsilniejszej). 1 = wszystkie cienkie.')
        self.spin_grubosc.setRange(1, 20); self.spin_grubosc.setSingleStep(1); self.spin_grubosc.setValue(6)
        self.spin_alfa = None
        if z_kryciem:
            self.spin_alfa = dodaj(w1, 'Krycie mapy:', QDoubleSpinBox(), 'Krycie mapy atencji: 0 = niewidoczna, 1 = pełna.')
            self.spin_alfa.setRange(0, 1); self.spin_alfa.setSingleStep(.1); self.spin_alfa.setValue(.5)
        for s in (self.combo_mapa.currentIndexChanged, self.combo_glowa.currentIndexChanged, self.spin_blok.valueChanged,
                  self.combo_bloki.currentIndexChanged, self.chk_bez_siebie.toggled, self.combo_skala.currentIndexChanged,
                  self.combo_strz.currentIndexChanged, self.spin_k.valueChanged, self.spin_grubosc.valueChanged) + (
                     (self.spin_alfa.valueChanged,) if self.spin_alfa else ()):
            s.connect(lambda *_: self.zmiana.emit())

    def kopiuj_z(self, inne):
        for a, b in ((self.combo_mapa, inne.combo_mapa), (self.combo_bloki, inne.combo_bloki),
                     (self.combo_skala, inne.combo_skala), (self.combo_strz, inne.combo_strz)):
            a.setCurrentIndex(b.currentIndex())
        self.spin_blok.setValue(inne.spin_blok.value()); self.spin_k.setValue(inne.spin_k.value())
        self.spin_grubosc.setValue(inne.spin_grubosc.value())
        self.chk_bez_siebie.setChecked(inne.chk_bez_siebie.isChecked())
        if self.spin_alfa and inne.spin_alfa:
            self.spin_alfa.setValue(inne.spin_alfa.value())
        self._glowa_do_ustawienia = inne.combo_glowa.currentIndex()

    def krycie(self):
        return self.spin_alfa.value() if self.spin_alfa else .5

    # ------------------------------------------------------------ dane
    def ustaw_zrodlo(self, atencja_fn):
        """atencja_fn(blok_od_0) -> [głowy, 600, 600]."""
        self.fn, self._bloki, self._cache = atencja_fn, {}, {}

    def _blok(self, b):
        if b not in self._bloki:
            self._bloki[b] = np.asarray(self.fn(b), np.float32)
            H = self._bloki[b].shape[0]
            if self.combo_glowa.count() != 2 + H:
                i = getattr(self, '_glowa_do_ustawienia', self.combo_glowa.currentIndex())
                self.combo_glowa.blockSignals(True)
                while self.combo_glowa.count() > 2:
                    self.combo_glowa.removeItem(2)
                self.combo_glowa.addItems([f'głowa {h + 1}' for h in range(H)])
                self.combo_glowa.setCurrentIndex(min(i, 1 + H))
                self.combo_glowa.blockSignals(False)
        return self._bloki[b]

    def _jeden(self, b):
        A = self._blok(b)
        g = self.combo_glowa.currentIndex()
        A = A.mean(0) if g == 0 else A.max(0) if g == 1 else A[min(g - 2, len(A) - 1)]
        if g == 1 or self.chk_bez_siebie.isChecked():
            A = A.copy()
            if self.chk_bez_siebie.isChecked():
                np.fill_diagonal(A, 0)
            A /= np.maximum(A.sum(1, keepdims=True), 1e-12)
        return A

    def macierz(self):
        """Macierz atencji [600, 600] według opcji (wiersz = kto patrzy, kolumna = na kogo)."""
        if self.fn is None:
            return None
        klucz = (self.spin_blok.value(), self.combo_glowa.currentIndex(), self.combo_bloki.currentIndex(),
                 self.chk_bez_siebie.isChecked())
        if klucz not in self._cache:
            b = self.spin_blok.value() - 1
            if self.combo_bloki.currentIndex() == 0:
                M = self._jeden(b)
            else:
                M = np.eye(600, dtype=np.float32)
                for i in range(b + 1):
                    A = .5 * self._jeden(i) + .5 * np.eye(600, dtype=np.float32)
                    M = (A / A.sum(1, keepdims=True)) @ M
                if self.chk_bez_siebie.isChecked():
                    M = M.copy(); np.fill_diagonal(M, 0); M /= np.maximum(M.sum(1, keepdims=True), 1e-12)
            self._cache[klucz] = M
        return self._cache[klucz]

    def potrzebuje_tokenu(self):
        return self.combo_mapa.currentIndex() in self.POTRZEBUJE_TOKENU_MAPY or \
            self.combo_strz.currentIndex() in self.POTRZEBUJE_TOKENU_STRZALEK

    def mapa(self, token):
        """(mapa [20, 30] znormowana do 0..1 według skali albo None, opis). token: jeden patch albo zbiór patchy
        (tablica): mapy „z wybranego” i „do wybranego” są wtedy średnią po patchach zbioru."""
        M = self.macierz()
        if M is None:
            return None, ''
        i = self.combo_mapa.currentIndex()
        if i in (1, 2) and token is None:
            return None, 'kliknij patch, aby pokazać jego atencję'
        if i == 0:
            v = M.mean(0)
        elif i == 1:
            v = M[token].mean(0) if np.ndim(token) else M[token]
        elif i == 2:
            v = M[:, token].mean(1) if np.ndim(token) else M[:, token]
        elif i == 3:
            v = 1 + (M * np.log(np.maximum(M, 1e-12))).sum(1) / np.log(600)
        elif i == 4:
            v = (M * ODLEGLOSC).sum(1)
        else:
            v = np.diag(M).copy()
        return self.normuj(v).reshape(WIERSZE, KOLUMNY), self.opis(token, v)

    def normuj(self, v):
        s = self.combo_skala.currentIndex()
        if s == 1:
            v = np.log10(np.maximum(v, max(float(v.max()), 1e-12) * 1e-4))
        elif s == 2:
            v = np.argsort(np.argsort(v)).astype(np.float32)
        return (v - v.min()) / max(float(v.max() - v.min()), 1e-12)

    def opis(self, token, v=None):
        m = self.MAPY[self.combo_mapa.currentIndex()].split(':')[0]
        g = self.combo_glowa.currentText()
        bl = f'blok {self.spin_blok.value()}' if self.combo_bloki.currentIndex() == 0 else f'rollout 1..{self.spin_blok.value()}'
        kto = f'{len(token)} patchy' if np.ndim(token) else f'patch {token}'
        t = f'atencja {bl}, {g}: {m}' + (f' ({kto})' if self.combo_mapa.currentIndex() in (1, 2) else '')
        if v is not None:
            t += f', zakres {float(v.min()):.3g}..{float(v.max()):.3g}'
        return t

    def strzalki(self, token):
        """(źródła, cele, wagi 0..1) albo None. Źródło patrzy na cel (ciągnie z niego)."""
        M = self.macierz()
        s = self.combo_strz.currentIndex()
        if M is None or s == 0 or (s in (1, 2) and token is None):
            return None
        k = self.spin_k.value()
        if s == 1:
            row = M[token].copy(); row[token] = -1
            dst = np.argsort(row)[::-1][:k]; src = np.full(len(dst), token); w = M[token, dst]
        elif s == 2:
            col = M[:, token].copy(); col[token] = -1
            src = np.argsort(col)[::-1][:k]; dst = np.full(len(src), token); w = M[src, token]
        elif s == 3:
            B = M.copy(); np.fill_diagonal(B, -1)
            src = np.arange(600); dst = B.argmax(1); w = M[src, dst]
        else:
            B = M.copy(); np.fill_diagonal(B, -1)
            idx = np.argpartition(B.ravel(), -k)[-k:]
            idx = idx[np.argsort(B.ravel()[idx])[::-1]]
            src, dst = np.divmod(idx, 600); w = M[src, dst]
        w = np.asarray(w, np.float32)
        return src, dst, w / max(float(w.max()), 1e-12)
