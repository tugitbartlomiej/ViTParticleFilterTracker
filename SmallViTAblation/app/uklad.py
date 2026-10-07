"""Układ „przepływowy”: kontrolki w wierszu, a gdy brakuje miejsca, zawijają się do następnej linii
(jak tekst). Dzięki temu panele z wieloma opcjami nie rozpychają okna na szerokość."""
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLayout, QPushButton, QSplitter, QStackedWidget, QStyle,
                             QStyleOptionComboBox, QStylePainter, QTabWidget, QVBoxLayout, QWidget, QWidgetItem)


class Przeplyw(QLayout):
    def __init__(self, rodzic=None, odstep=6):
        super().__init__(rodzic)
        self._el = []
        self._odstep = odstep
        if rodzic is not None:
            self.setContentsMargins(0, 0, 0, 0)

    # zgodność z QHBoxLayout: addWidget(w, rozciąganie) i addStretch() bez skutku
    def addWidget(self, w, *_):
        super().addWidget(w)

    def addStretch(self, *_):
        pass

    def addItem(self, el):
        self._el.append(el)

    def count(self):
        return len(self._el)

    def itemAt(self, i):
        return self._el[i] if 0 <= i < len(self._el) else None

    def takeAt(self, i):
        return self._el.pop(i) if 0 <= i < len(self._el) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, szer):
        return self._uloz(QRect(0, 0, szer, 0), tylko_licz=True)

    def setGeometry(self, r):
        super().setGeometry(r)
        self._uloz(r, tylko_licz=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for el in self._el:
            s = s.expandedTo(el.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _uloz(self, r, tylko_licz):
        m = self.contentsMargins()
        x0, y = r.x() + m.left(), r.y() + m.top()
        prawa = r.right() - m.right()
        x, wys_linii = x0, 0
        for el in self._el:
            if isinstance(el, QWidgetItem) and el.widget() is not None and el.widget().isHidden():
                continue
            h = el.sizeHint()
            if x + h.width() > prawa and wys_linii > 0:
                x, y = x0, y + wys_linii + self._odstep // 2
                wys_linii = 0
            if not tylko_licz:
                el.setGeometry(QRect(QPoint(x, y), h))
            x += h.width() + self._odstep
            wys_linii = max(wys_linii, h.height())
        return y + wys_linii - r.y() + m.bottom()


def dodaj(uklad, nazwa, widget, opis):
    """Etykieta i kontrolka obok siebie w układzie, ten sam opis jako podpowiedź obu; zwraca kontrolkę."""
    e = QLabel(nazwa); e.setToolTip(opis); widget.setToolTip(opis)
    if isinstance(uklad, Przeplyw):                 # etykieta i pole w jednym kawałku: zawijają się razem
        para = QWidget(); l = QHBoxLayout(para); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(4)
        l.addWidget(e); l.addWidget(widget)
        uklad.addWidget(para)
    else:
        uklad.addWidget(e); uklad.addWidget(widget)
    return widget


class PanelOpcji:
    """Opcje widoku: pasek przycisków zawsze widoczny (z przyciskiem „Opcje”, który chowa resztę) i zakładki
    z grupami opcji. Dzięki temu widok 3D dostaje prawie całą wysokość, a opcje są pogrupowane tematycznie."""

    def __init__(self, uklad, strony):
        from styl import rola
        self.pasek = Przeplyw(); uklad.addLayout(self.pasek)
        self.btn_opcje = rola(QPushButton('Opcje ▾'), 'narzedzie'); self.btn_opcje.setCheckable(True)
        self.btn_opcje.setChecked(True)
        self.btn_opcje.setToolTip('Pokaż albo schowaj opcje widoku (więcej miejsca na widok).')
        self.pasek.addWidget(self.btn_opcje)
        self.zakladki = QTabWidget(); self.zakladki.setObjectName('opcje_widoku'); self.zakladki.setDocumentMode(True)
        self._strony = {}
        for n in strony:
            s = QWidget(); lay = QVBoxLayout(s); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(2)
            self.zakladki.addTab(s, n); self._strony[n] = lay
        uklad.addWidget(self.zakladki)
        self.btn_opcje.toggled.connect(lambda v: (self.zakladki.setVisible(v),
                                                  self.btn_opcje.setText('Opcje ▾' if v else 'Opcje ▸')))

    def dodaj_strone(self, nazwa):
        s = QWidget(); lay = QVBoxLayout(s); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(2)
        self.zakladki.addTab(s, nazwa); self._strony[nazwa] = lay

    def wiersz(self, strona):
        w = Przeplyw(); self._strony[strona].addLayout(w)
        return w

    def strona(self, strona):
        return self._strony[strona]

    def zakoncz(self):
        for lay in self._strony.values():
            lay.addStretch(1)


class WaskiCombo(QComboBox):
    """Pole wyboru o stałej szerokości około `znaki` znaków: za długi wybrany tekst kończy się „…”, a rozwinięta
    lista ma pełną szerokość. Długie opisy pozycji nie rozpychają wąskich kolumn (pełny opis w podpowiedzi)."""

    def __init__(self, znaki=14):
        super().__init__()
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(znaki)

    def showPopup(self):
        fm = self.view().fontMetrics()
        self.view().setMinimumWidth(max([fm.horizontalAdvance(self.itemText(i)) for i in range(self.count())] + [0]) + 36)
        super().showPopup()

    def paintEvent(self, _):
        p = QStylePainter(self)
        opt = QStyleOptionComboBox(); self.initStyleOption(opt)
        p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        r = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt, QStyle.SubControl.SC_ComboBoxEditField, self)
        opt.currentText = self.fontMetrics().elidedText(opt.currentText, Qt.TextElideMode.ElideRight, max(r.width() - 4, 10))
        p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


def etykieta_statusu(etykieta):
    """Tekst stanu widoku (liczby cząsteczek, skala, PCA): mała szara czcionka (rola „status”), z zawijaniem."""
    from styl import rola
    etykieta.setWordWrap(True)
    return rola(etykieta, 'status')


def etykieta_opisu(etykieta):
    """Tekst objaśnień (zakres, źródło, opis filmu): szary (rola „opis”), żeby nie konkurował z kontrolkami."""
    from styl import rola
    return rola(etykieta, 'opis')


class _OknoOdpiete(QWidget):
    def __init__(self, odpinanie, tytul):
        super().__init__(None, Qt.WindowType.Window)
        self._odp = odpinanie
        self.setWindowTitle(tytul)
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)

    def closeEvent(self, e):
        self._odp.przycisk.setChecked(False)          # zamknięcie okna = przypięcie widoku z powrotem
        super().closeEvent(e)


class Odpinanie:
    """Przycisk „Odepnij”: przenosi żywy widok (ten sam widżet, dalej aktualizowany przez zakładkę, np. przy
    odtwarzaniu filmu) do osobnego okna, a w jego miejscu zostawia napis. „Przypnij” albo zamknięcie okna
    wstawia widok z powrotem w to samo miejsce (splitter, zakładka albo zwykły układ)."""

    def __init__(self, widok, tytul, przycisk):
        self.widok, self.tytul, self.przycisk = widok, tytul, przycisk
        self.okno = self._oddaj = None
        przycisk.setCheckable(True)
        przycisk.setToolTip('Przenosi ten widok do osobnego okna (np. na drugi monitor); dalej działa i odtwarza\n'
                            'sekwencję razem z zakładką. Przypnij albo zamknij okno, żeby wrócił na miejsce.')
        przycisk.toggled.connect(lambda v: self.odepnij() if v else self.przypnij())

    def _wyjmij(self, zastepca):
        """Wstawia zastępcę w miejsce widoku; zwraca funkcję, która odda widok na to miejsce."""
        w = self.widok
        rodzic = w.parentWidget()
        if isinstance(rodzic, QSplitter):
            i = rodzic.indexOf(w); rozm = rodzic.sizes()
            rodzic.replaceWidget(i, zastepca)
            return lambda: (rodzic.replaceWidget(i, w), rodzic.setSizes(rozm))
        if isinstance(rodzic, QStackedWidget) and isinstance(rodzic.parentWidget(), QTabWidget):
            tab = rodzic.parentWidget(); i = tab.indexOf(w); tekst = tab.tabText(i); biezaca = tab.currentIndex()
            tab.removeTab(i); tab.insertTab(i, zastepca, tekst); tab.setCurrentIndex(biezaca)

            def oddaj():
                j = tab.indexOf(zastepca); tab.removeTab(j); tab.insertTab(j, w, tekst); tab.setCurrentIndex(j)
            return oddaj
        lay = rodzic.layout()
        lay.replaceWidget(w, zastepca)
        return lambda: lay.replaceWidget(zastepca, w)

    def odepnij(self):
        if self.okno is not None:
            return
        zastepca = QLabel(f'„{self.tytul}” jest w osobnym oknie.\nZamknij to okno albo kliknij „Przypnij”, żeby widok wrócił tutaj.')
        zastepca.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zastepca = zastepca
        rozmiar = self.widok.size()
        self._oddaj = self._wyjmij(zastepca)
        self.okno = _OknoOdpiete(self, self.tytul)
        self.okno.layout().addWidget(self.widok)
        self.widok.show()
        self.okno.resize(max(900, rozmiar.width()), max(700, rozmiar.height()))
        self.okno.show()
        self.przycisk.setText('Przypnij')

    def przypnij(self):
        if self.okno is None:
            return
        okno, self.okno = self.okno, None
        self.widok.setParent(None)
        self._oddaj()
        self.widok.show()
        self._zastepca.deleteLater()
        okno.hide(); okno.deleteLater()
        self.przycisk.setText('Odepnij')


# ujęcia kamery: (azymut, elewacja); „przód” jak domyślny widok nad obrazem (azymut -90: x w prawo, y w głąb)
UJECIA = (('Przód', -90, 0), ('Tył', 90, 0), ('Lewo', 180, 0), ('Prawo', 0, 0), ('Góra', -90, 90), ('Dół', -90, -90))


def dodaj_nawigacje(wlasciciel, panel, widok, strona='Nawigacja'):
    """Zakładka „Nawigacja” w PanelOpcji widoku 3D: obrót 360 st. (włącz/wyłącz, prędkość, kierunek), kroki 15 st.,
    przybliżanie, ujęcia (przód, tył, lewo, prawo, góra, dół) i lustrzane odbicie w osi X, Y, Z. Kontrolki zostają
    atrybutami właściciela (zapis w ustawieniach): btn_obrot, spin_obrot, combo_obrot, chk_lustro {x, y, z}."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox
    from styl import rola
    panel.dodaj_strone(strona)

    w = panel.wiersz(strona)
    wlasciciel.btn_obrot = b = rola(QPushButton('Obrót 360°'), 'narzedzie'); b.setCheckable(True)
    b.setToolTip('Kamera krąży dookoła sceny bez końca (pełne 360 st. i dalej); kliknij ponownie, żeby zatrzymać.\n'
                 'Myszą można w tym czasie dalej przybliżać i przechylać.')
    w.addWidget(b)
    wlasciciel.spin_obrot = dodaj(w, 'Prędkość:', QDoubleSpinBox(), 'Prędkość obrotu w stopniach na sekundę (360 st. / prędkość = czas okrążenia).')
    wlasciciel.spin_obrot.setRange(2, 360); wlasciciel.spin_obrot.setValue(30); wlasciciel.spin_obrot.setSuffix(' °/s')
    wlasciciel.combo_obrot = dodaj(w, 'Kierunek:', QComboBox(), 'W którą stronę krąży kamera.')
    wlasciciel.combo_obrot.addItems(['w lewo', 'w prawo'])
    timer = QTimer(widok); timer.setInterval(30)
    timer.timeout.connect(lambda: widok.orbit((1 if wlasciciel.combo_obrot.currentIndex() == 0 else -1)
                                              * wlasciciel.spin_obrot.value() * 0.03, 0))
    b.toggled.connect(lambda v: timer.start() if v else timer.stop())

    w = panel.wiersz(strona)
    for tekst, opis, fn in (('⟲ 15°', 'Obrót kamery o 15 st. w lewo.', lambda: widok.orbit(15, 0)),
                            ('⟳ 15°', 'Obrót kamery o 15 st. w prawo.', lambda: widok.orbit(-15, 0)),
                            ('▲ 15°', 'Kamera 15 st. wyżej.', lambda: widok.orbit(0, 15)),
                            ('▼ 15°', 'Kamera 15 st. niżej.', lambda: widok.orbit(0, -15)),
                            ('Bliżej', 'Przybliżenie (odległość kamery x 0,8).',
                             lambda: widok.setCameraPosition(distance=widok.opts['distance'] * .8)),
                            ('Dalej', 'Oddalenie (odległość kamery x 1,25).',
                             lambda: widok.setCameraPosition(distance=widok.opts['distance'] * 1.25))):
        p = rola(QPushButton(tekst), 'narzedzie'); p.setToolTip(opis); p.clicked.connect(fn); w.addWidget(p)

    w = panel.wiersz(strona)
    w.addWidget(QLabel('Ujęcie:'))
    for nazwa, az, el in UJECIA:
        p = rola(QPushButton(nazwa), 'narzedzie')
        p.setToolTip(f'Kamera: azymut {az} st., elewacja {el} st. (odległość i środek bez zmian).')
        p.clicked.connect(lambda _=False, az=az, el=el: widok.setCameraPosition(azimuth=az, elevation=el))
        w.addWidget(p)

    w = panel.wiersz(strona)
    w.addWidget(QLabel('Lustrzane odbicie:'))
    wlasciciel.chk_lustro = {}
    for os_ in ('x', 'y', 'z'):
        c = QCheckBox(f'oś {os_.upper()}')
        c.setToolTip(f'Odbija całą scenę w osi {os_.upper()} względem środka widoku (współrzędna {os_} -> -{os_}).\n'
                     'Jedno odbicie zmienia układ prawoskrętny w lewoskrętny; dwa odbicia to obrót o 180 st.')
        wlasciciel.chk_lustro[os_] = c; w.addWidget(c)

    def lustro(*_):
        widok.lustro = tuple(-1 if wlasciciel.chk_lustro[o].isChecked() else 1 for o in 'xyz')
        widok.update()
    for c in wlasciciel.chk_lustro.values():
        c.toggled.connect(lustro)
    panel.strona(strona).addStretch(1)



class RamaPanelu(QWidget):
    """Panel z cienkim paskiem: tytuł, „Odepnij” (osobne okno, np. na drugi monitor) i „✕” (chowa panel).
    Zamknięcie odznacza pole wyboru `chk` (lista „Widoki”), które pokazuje panel z powrotem; schowanie
    odpiętego panelu najpierw go przypina."""

    def __init__(self, tytul, widzet, chk):
        super().__init__()
        self.widzet, self.chk = widzet, chk
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(1)
        pasek = QHBoxLayout(); pasek.setContentsMargins(4, 0, 0, 0); pasek.setSpacing(3)
        from styl import rola
        et = rola(QLabel(tytul), 'opis'); pasek.addWidget(et, 1)
        self.btn_odepnij = QPushButton('Odepnij'); self.btn_odepnij.setFixedHeight(20)
        self.btn_zamknij = QPushButton('✕'); self.btn_zamknij.setFixedSize(24, 20)
        self.btn_zamknij.setToolTip(f'Chowa panel „{tytul}”. Pokaż go z powrotem polem „{chk.text()}” w „Widoki”.')
        pasek.addWidget(self.btn_odepnij); pasek.addWidget(self.btn_zamknij)
        lay.addLayout(pasek); lay.addWidget(widzet, 1)
        self.odpinanie = Odpinanie(self, tytul, self.btn_odepnij)
        self.btn_zamknij.clicked.connect(lambda: chk.setChecked(False))
        chk.toggled.connect(self.pokaz)

    def pokaz(self, wl):
        if not wl and self.odpinanie.okno is not None:
            self.btn_odepnij.setChecked(False)
        self.setVisible(wl)


def zwijana(grupa, rozwinieta=True):
    """QGroupBox zwijany polem w tytule: odznaczone chowa zawartość (zostaje sam tytuł), zaznaczone pokazuje."""
    grupa.setCheckable(True); grupa.setChecked(rozwinieta)

    def przelacz(v):
        for c in grupa.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            c.setVisible(v)

    grupa.toggled.connect(przelacz)
    przelacz(rozwinieta)
    grupa.setToolTip((grupa.toolTip() + '\n' if grupa.toolTip() else '') + 'Pole w tytule zwija i rozwija tę grupę.')
    return grupa
