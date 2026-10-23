"""Nagłówki grup w długich sekcjach panelu parametrów.

`zastosuj(z)` (z = zbudowana `LasersTab`) wstawia do formularzy sekcji (`z._sek[klucz].layout()`, QFormLayout) wiersze z krótkim
nagłówkiem (pogrubiony tytuł i cienka linia), tuż przed pierwszą kontrolką każdej grupy. Grupy opisuje `GRUPY`: klucz sekcji →
lista (tytuł, nazwa atrybutu pierwszej kontrolki grupy), w kolejności wierszy. Brakujący atrybut pomija nagłówek bez błędu;
nagłówki mają objectName „naglowek_grupy”, więc powtórne wywołanie niczego nie dubluje.

Klik w nagłówek zwija (▸) i rozwija (▾) wiersze jego grupy (do następnego nagłówka); `zwin_wszystko(korzen, zwin)` robi to
naraz dla wszystkich grup pod widżetem `korzen` (okienko ustawień z prawego kliku, strona panelu), a sekcje bez grup zwija
w całości (nagłówek sekcji Zwijana)."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget

NAZWA_NAGLOWKA = 'naglowek_grupy'
STYL_NAGLOWKA = 'color:#8fb8ff; font-weight:bold; padding:6px 0 2px 0; border-bottom:1px solid #2f3a4a;'

GRUPY = {
    'dane': [('Historia checkpointów', 'cb_historia'), ('Zdjęcie', 'cb_film'), ('Liczenie', 'sp_okno')],
    'sfera': [('Wiązki laserów', 'ch_swiatlo'), ('Strzałki obrotu igieł', 'cb_obrot'), ('Koloidy i średnie skoków', 'ch_sfera'),
              ('Wektory i ścieżki', 'cb_wek_kol'), ('Wygładzanie i zanik', 'cb_wyg')],
    'pulapka': [('Liczenie pułapki', 'ch_og'), ('Pomiar w podkrokach dla całej historii', 'bt_pom'),
                ('Wiązki zbiegające się w cząsteczce', 'ch_og_wiazki'), ('Światło ogniska', 'ch_og_sw'),
                ('Obrys ogniska 3D', 'ch_og_obrys'), ('Wkłady do skoku', 'cb_og_wk'), ('Lokalne pole: mgła', 'ch_og_mgla'),
                ('Strzałki pola', 'cb_og_strz'),
                ('Oznaczenia w kadrze: osie, skok, ruch, tło', 'ch_og_osie'), ('Ślad, kadr i odtwarzanie', 'cb_og_slad')],
    'mapa_pul': [('Liczenie mapy', 'ch_mp'), ('Strzałki pola w komórkach wszystkich cząsteczek', 'cb_mp_strz'),
                 ('Wygląd mapy', 'cb_mp_kolor')],
    'dolina': [('Liczenie doliny', 'ch_dol'), ('Progi pewności i IoU', 'sp_dol_p'), ('Wygląd', 'ch_dol_mgla')],
    'przejscie': [('Widoczność: wszystkie wiązki naraz', 'ch_p_wiazki'), ('Uwaga: siły między cząsteczkami', 'cb_sily'),
                  ('Lasery MLP', 'ch_lasery'), ('Tory, przesunięcia i kolory', 'cb_tory')],
    'zapis': [('Zapis policzonych przejść', 'ch_prz_zapis'), ('Zapis wszystkich checkpointów', 'sp_prz_co')],
    'mikro': [('Naświetlenie: zdjęcie i płyty łodygi', 'ch_mk_obraz'), ('Formowanie: promienie, linki i sfera', 'cb_mk_prom'),
              ('Lampa, matryca soczewek i ekran', 'ch_mk_socz')],
}


def _zawiera(el, w):
    """Czy element formularza (widżet, jego potomek albo układ z takim widżetem) zawiera widżet w."""
    if el is None:
        return False
    x = el.widget()
    if x is not None:
        return x is w or x.isAncestorOf(w)
    uklad = el.layout()
    return uklad is not None and any(_zawiera(uklad.itemAt(i), w) for i in range(uklad.count()))


def wiersz(formularz, w):
    """Numer wiersza QFormLayout, w którym jest widżet w (jako pole, etykieta albo wewnątrz pola-układu); -1, gdy go nie ma."""
    for r in range(formularz.rowCount()):
        for rola in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole, QFormLayout.ItemRole.SpanningRole):
            if _zawiera(formularz.itemAt(r, rola), w):
                return r
    return -1


def _formularz_widzeta(sek, w):
    """QFormLayout sekcji, w którym leży widżet w (sekcja może mieć zagnieżdżone formularze)."""
    for f in [sek.layout()] + sek.findChildren(QFormLayout):
        if isinstance(f, QFormLayout) and wiersz(f, w) >= 0:
            return f
    return None


class GroupHeader(QLabel):
    """Nagłówek grupy: pogrubiony tytuł z cienką linią; klik zwija i rozwija wiersze grupy w formularzu `formularz`.
    Tytuł bez strzałki: `tytul_grupy`. Wiersze ukryte przez program przed zwinięciem (np. zależne od trybu) zostają ukryte."""

    def __init__(self, tytul):
        super().__init__()
        self.tytul_grupy = tytul
        self.formularz = None
        self.zwiniety = False
        self._ukryte = set()
        self.setObjectName(NAZWA_NAGLOWKA); self.setStyleSheet(STYL_NAGLOWKA)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip('Kliknij, żeby zwinąć (▸) albo rozwinąć (▾) tę grupę ustawień.')
        self._napis()

    def _napis(self):
        self.setText(('▸ ' if self.zwiniety else '▾ ') + self.tytul_grupy)

    def _wiersze(self):
        """Wiersze grupy: od wiersza pod nagłówkiem do następnego nagłówka (albo końca formularza)."""
        f = self.formularz
        r = wiersz(f, self) if f is not None else -1
        if r < 0:
            return []
        out = []
        for rr in range(r + 1, f.rowCount()):
            el = f.itemAt(rr, QFormLayout.ItemRole.SpanningRole)
            if el is not None and isinstance(el.widget(), GroupHeader):
                break
            out.append(rr)
        return out

    def zwin(self, zwin=True):
        if zwin == self.zwiniety:
            return
        f = self.formularz
        if zwin:
            rows = self._wiersze()
            self._ukryte = {rr for rr in rows if not f.isRowVisible(rr)}
            for rr in rows:
                f.setRowVisible(rr, False)
        else:
            for rr in self._wiersze():
                if rr not in self._ukryte:
                    f.setRowVisible(rr, True)
        self.zwiniety = zwin
        self._napis()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.zwin(not self.zwiniety)
            e.accept()
            return
        super().mousePressEvent(e)


def naglowek(tekst):
    """Widżet nagłówka grupy: pogrubiony tytuł z cienką linią pod spodem (klik zwija grupę)."""
    return GroupHeader(tekst)


def zwin_wszystko(korzen, zwin=True):
    """Zwija (rozwija) wszystkie grupy pod `korzen`; sekcje Zwijana bez grup zwija (rozwija) w całości. Zwraca liczbę zmienionych."""
    from laser_app.ui.widgets import CollapsibleSection
    n = 0
    for g in korzen.findChildren(GroupHeader):
        if g.zwiniety != zwin:
            g.zwin(zwin); n += 1
    sekcje = korzen.findChildren(CollapsibleSection) + ([korzen] if isinstance(korzen, CollapsibleSection) else [])
    for s in sekcje:
        if zwin and s.findChildren(GroupHeader):
            continue                                                 # sekcja z grupami: zwinięte grupy, nagłówki zostają widoczne
        if s.zwinieta() != zwin:
            s.naglowek.setChecked(not zwin); n += 1
    return n


def zastosuj(z, grupy=None):
    """Wstawia nagłówki grup według `grupy` (domyślnie GRUPY); zwraca słownik {klucz sekcji: liczba wstawionych nagłówków}.
    Idempotentne: sekcja, w której nagłówek o tym tekście już jest, nie dostaje drugiego."""
    wstawione = {}
    for klucz, lista in (grupy or GRUPY).items():
        sek = getattr(z, '_sek', {}).get(klucz)
        if sek is None:
            continue
        cele = []                                                    # (formularz, wiersz, tytuł) dla grup, których kontrolka istnieje
        for tytul, atr in lista:
            w = getattr(z, atr, None)
            if not isinstance(w, QWidget):
                continue
            f = _formularz_widzeta(sek, w)
            if f is None:
                continue
            istnieje = any(isinstance(l, QLabel) and l.objectName() == NAZWA_NAGLOWKA and getattr(l, 'tytul_grupy', l.text()) == tytul
                           for l in sek.findChildren(QLabel))
            if not istnieje:
                cele.append((f, wiersz(f, w), tytul))
        n = 0
        for f, r, tytul in sorted(cele, key=lambda c: c[1], reverse=True):     # od końca, żeby numery wierszy zostały ważne
            lb = naglowek(tytul); lb.formularz = f
            f.insertRow(r, lb); n += 1
        wstawione[klucz] = n
    return wstawione
