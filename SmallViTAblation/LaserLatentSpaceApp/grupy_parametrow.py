"""Nagłówki grup w długich sekcjach panelu parametrów.

`zastosuj(z)` (z = zbudowana `ZakladkaLaserow`) wstawia do formularzy sekcji (`z._sek[klucz].layout()`, QFormLayout) wiersze z krótkim
nagłówkiem (pogrubiony tytuł i cienka linia), tuż przed pierwszą kontrolką każdej grupy. Grupy opisuje `GRUPY`: klucz sekcji →
lista (tytuł, nazwa atrybutu pierwszej kontrolki grupy), w kolejności wierszy. Brakujący atrybut pomija nagłówek bez błędu;
nagłówki mają objectName „naglowek_grupy”, więc powtórne wywołanie niczego nie dubluje."""
from PyQt6.QtWidgets import QFormLayout, QLabel, QLayout, QWidget

NAZWA_NAGLOWKA = 'naglowek_grupy'
STYL_NAGLOWKA = 'color:#8fb8ff; font-weight:bold; padding:6px 0 2px 0; border-bottom:1px solid #2f3a4a;'

GRUPY = {
    'dane': [('Historia checkpointów', 'cb_historia'), ('Zdjęcie', 'cb_film'), ('Liczenie', 'sp_okno')],
    'sfera': [('Wiązki laserów', 'ch_swiatlo'), ('Strzałki obrotu igieł', 'cb_obrot'), ('Koloidy i średnie skoków', 'ch_sfera'),
              ('Wektory i ścieżki', 'cb_wek_kol'), ('Wygładzanie i zanik', 'cb_wyg')],
    'pulapka': [('Liczenie pułapki', 'ch_og'), ('Pomiar w podkrokach dla całej historii', 'bt_pom'),
                ('Wiązki zbiegające się w cząsteczce', 'ch_og_wiazki'), ('Światło ogniska', 'ch_og_sw'),
                ('Obrys ogniska 3D', 'ch_og_obrys'), ('Wkłady do skoku', 'cb_og_wk'), ('Lokalne pole: mgła i strzałki', 'ch_og_mgla'),
                ('Oznaczenia w kadrze: osie, skok, ruch, tło', 'ch_og_osie'), ('Ślad, kadr i odtwarzanie', 'cb_og_slad')],
    'dolina': [('Liczenie doliny', 'ch_dol'), ('Progi pewności i IoU', 'sp_dol_p'), ('Wygląd', 'ch_dol_mgla')],
    'przejscie': [('Uwaga: siły między cząsteczkami', 'cb_sily'), ('Lasery MLP', 'ch_lasery'), ('Tory i kolory', 'cb_tory')],
    'zapis': [('Zapis policzonych przejść', 'ch_prz_zapis'), ('Zapis wszystkich checkpointów', 'sp_prz_co')],
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


def naglowek(tekst):
    """Widżet nagłówka grupy: pogrubiony tytuł z cienką linią pod spodem."""
    lb = QLabel(tekst); lb.setObjectName(NAZWA_NAGLOWKA); lb.setStyleSheet(STYL_NAGLOWKA)
    return lb


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
            istnieje = any(isinstance(l, QLabel) and l.objectName() == NAZWA_NAGLOWKA and l.text() == tytul
                           for l in sek.findChildren(QLabel))
            if not istnieje:
                cele.append((f, wiersz(f, w), tytul))
        n = 0
        for f, r, tytul in sorted(cele, key=lambda c: c[1], reverse=True):     # od końca, żeby numery wierszy zostały ważne
            f.insertRow(r, naglowek(tytul)); n += 1
        wstawione[klucz] = n
    return wstawione
