"""Sprawdza, które kontrolki edytora latentu nie mają dymka (toolTip), i wypisuje ich listę.

Buduje okno aplikacji (bez wczytywania ustawień i bez automatycznego wczytywania danych), buduje wszystkie
zakładki leniwe i okna pomocnicze, po czym przechodzi po przyciskach, polach wyboru, listach, polach liczbowych,
suwakach i polach tekstowych. Dla każdej kontrolki bez dymka podaje plik i linię, w której ją utworzono
(klasy Qt są na czas sprawdzania opakowane i zapamiętują miejsce utworzenia), klasę, tekst i zakładkę.

Użycie (z katalogu app):
    F:\\Python\\3.13\\python.exe narzedzia/sprawdz_dymki.py                  # pełna lista pogrupowana według plików
    F:\\Python\\3.13\\python.exe narzedzia/sprawdz_dymki.py --podsumowanie    # tylko liczby na plik
    F:\\Python\\3.13\\python.exe narzedzia/sprawdz_dymki.py --plik zakladka_uczenia   # tylko wybrany plik (fragment nazwy)
Kod wyjścia: 0 gdy wszystkie kontrolki mają dymek, 1 gdy brakuje (po zastosowaniu --plik tylko dla tego pliku).
Wewnętrzne elementy Qt (pole tekstowe w polu liczbowym, strzałki pasków kart, przyciski menu) są pomijane.
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ['EDYTOR_BEZ_USTAWIEN'] = '1'            # start na ustawieniach domyślnych, bez zapisu pliku
os.environ['EDYTOR_BEZ_AUTOSTARTU'] = '1'          # nic nie wczytuje się samo
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, Qt

TYPY = ('QPushButton', 'QToolButton', 'QCheckBox', 'QRadioButton', 'QComboBox', 'QSpinBox', 'QDoubleSpinBox',
        'QSlider', 'QLineEdit')
# pliki pomocnicze, przez które tworzy się kontrolki: miejsce utworzenia to pierwszy wołający spoza nich
POMOCNICZE = {'wygoda.py', 'styl.py'}


def _opakuj_klasy():
    """Podmienia klasy Qt na podklasy zapamiętujące plik i linię utworzenia (przed importem modułów aplikacji)."""
    for nazwa in TYPY:
        baza = getattr(QtWidgets, nazwa)

        def init(self, *a, __baza=baza, **k):
            __baza.__init__(self, *a, **k)
            f = sys._getframe(1)
            # pomiń konstruktory podklas tej samej kontrolki (np. WaskiCombo.__init__ wołające QComboBox.__init__)
            while f is not None and f.f_code.co_name == '__init__' and f.f_locals.get('self') is self:
                f = f.f_back
            if f is not None:
                try:
                    plik = Path(f.f_code.co_filename)
                    # kontrolki tworzone przez biblioteki (pyqtgraph itp.) nie są do poprawienia w tym projekcie
                    self.setProperty('pochodzenie', f'{plik.name}:{f.f_lineno}' if plik.parent == APP else 'biblioteka:0')
                except RuntimeError:
                    pass

        setattr(QtWidgets, nazwa, type(nazwa, (baza,), {'__init__': init, '__module__': baza.__module__}))


def ma_dymek(w):
    if w.toolTip().strip():
        return True
    if isinstance(w, QtWidgets.QToolButton):                # przycisk paska narzędzi: dymek może być w akcji
        a = w.defaultAction()
        return a is not None and bool(a.toolTip().strip()) and a.toolTip().replace('&', '') != a.text().replace('&', '')
    return False


def wewnetrzny(w):
    """Elementy tworzone przez Qt wewnątrz innych kontrolek (nie są kontrolkami aplikacji)."""
    r = w.parentWidget()
    if isinstance(w, QtWidgets.QLineEdit) and isinstance(r, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
        return True
    if isinstance(w, QtWidgets.QToolButton) and isinstance(r, (QtWidgets.QTabBar, QtWidgets.QMenu)):
        return True
    return w.property('pochodzenie') is None and not w.objectName()    # utworzone poza kodem aplikacji (np. przez Qt)


def tekst(w):
    for n in ('text', 'currentText'):
        f = getattr(w, n, None)
        if callable(f):
            try:
                t = f()
            except TypeError:
                continue
            if t:
                return str(t).replace('\n', ' ')[:60]
    if isinstance(w, QtWidgets.QAbstractSpinBox):
        return f'wartość {w.text()}'
    if isinstance(w, QtWidgets.QSlider):
        return f'zakres {w.minimum()}..{w.maximum()}'
    return ''


def zakladka_widgetu(w, okno):
    """Nazwa zakładki głównej (atrybut okna) zawierającej kontrolkę albo nazwa okna pomocniczego."""
    cel = w.window()
    if cel is not okno:
        return type(cel).__name__
    for nazwa in list(getattr(okno, '_leniwe', {})) + ['zakladka_edytora']:
        z = getattr(okno, '__dict__', {}).get(nazwa)
        p = w
        while p is not None:
            if p is z:
                return nazwa
            p = p.parentWidget()
    return 'Edytor SD'


def zbierz(okno, dodatkowe):
    wynik, wszystkich, wewn = [], 0, 0
    korzenie = [okno] + list(dodatkowe)
    widziane = set()
    for korzen in korzenie:
        for nazwa in TYPY:
            for w in korzen.findChildren(getattr(QtWidgets, nazwa)):
                if id(w) in widziane:
                    continue
                widziane.add(id(w))
                if wewnetrzny(w):
                    wewn += 1
                    continue
                wszystkich += 1
                if not ma_dymek(w):
                    wynik.append(dict(plik=(w.property('pochodzenie') or '?:0'), klasa=type(w).__name__,
                                      tekst=tekst(w), zakladka=zakladka_widgetu(w, okno)))
    return wynik, wszystkich, wewn


def main():
    arg = argparse.ArgumentParser(description='Lista kontrolek bez dymka (toolTip).')
    arg.add_argument('--podsumowanie', action='store_true', help='tylko liczby na plik')
    arg.add_argument('--plik', default='', help='tylko pliki, których nazwa zawiera ten fragment')
    a = arg.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')       # strzałki i polskie znaki także w konsoli cp1250
    _opakuj_klasy()
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    aplikacja = QtWidgets.QApplication(sys.argv)
    import torch  # noqa: F401  (przed oknem, jak w start.py)
    import app
    from styl import STYL
    aplikacja.setStyleSheet(STYL)
    okno = app.Okno()
    for nazwa in list(getattr(okno, '_leniwe', {})):
        okno.zbuduj_zakladke(nazwa)
    # okna pomocnicze tworzone na żądanie
    dodatkowe = []
    try:
        from okno_zrodel_sily import OknoZrodelSily
        dodatkowe.append(OknoZrodelSily(lambda: None, okno.zakladka_uczenia))
    except Exception as e:
        print(f'(pominięto okno „Źródła siły”: {e})')
    aplikacja.processEvents()
    brakujace, wszystkich, wewn = zbierz(okno, dodatkowe)
    bibl = [b for b in brakujace if b['plik'].startswith('biblioteka')]
    brakujace = [b for b in brakujace if not b['plik'].startswith('biblioteka')]
    po_plikach = defaultdict(list)
    for b in brakujace:
        po_plikach[b['plik'].split(':')[0]].append(b)
    if a.plik:
        po_plikach = {p: v for p, v in po_plikach.items() if a.plik in p}
    razem = sum(len(v) for v in po_plikach.values())
    if not a.podsumowanie:
        for plik in sorted(po_plikach):
            print(f'\n== {plik} ({len(po_plikach[plik])})')
            for b in sorted(po_plikach[plik], key=lambda x: int(x['plik'].split(':')[1])):
                print(f'  {b["plik"]:<28} {b["klasa"]:<14} {b["zakladka"]:<20} „{b["tekst"]}”')
    print('\nPodsumowanie (kontrolek bez dymka według pliku, w którym je utworzono):')
    for plik in sorted(po_plikach, key=lambda p: -len(po_plikach[p])):
        print(f'  {plik:<28} {len(po_plikach[plik])}')
    print(f'Razem: {razem} bez dymka, kontrolek sprawdzonych: {wszystkich} '
          f'(pominięto {len(bibl)} kontrolek bibliotecznych, np. pyqtgraph, i {wewn} elementów wewnętrznych Qt).')
    sys.stdout.flush()
    os._exit(1 if razem else 0)                      # twarde wyjście: wątki i konteksty GL nie blokują zamknięcia


if __name__ == '__main__':
    main()
