"""Zapis i odczyt ustawień całej aplikacji do pliku konfiguracyjnego JSON.

Ustawienia to stan kontrolek: listy wyboru (indeks i tekst), pola liczbowe i pola wyboru (checkboxy), zebrane z atrybutów okna i jego zakładek, np. `zakladka_filmu.w3d.combo_pca`. Nie zapisuje danych
(obrazów, latentu, sekwencji) ani suwaków przebiegu (klatka, przejście PRZED/PO).

- Plik domyślny `ustawienia.json` obok aplikacji: wczytywany przy starcie, zapisywany przy zamknięciu okna.
- Menu „Ustawienia”: zapis i wczytanie dowolnego pliku, przywrócenie domyślnych (stan zaraz po uruchomieniu).
Listy z pozycjami zależnymi od danych (np. film, głowa atencji) wracają po tekście; gdy pozycji nie ma, zostają.

Zakładki budowane leniwie (`app.ZAKLADKI_LENIWE`, powstają przy pierwszym wejściu na kartę): ustawienia takiej
zakładki, której jeszcze nie ma, czekają w `okno._ustawienia_oczekujace`, stosują się zaraz po jej zbudowaniu
(`Okno.zbuduj_zakladke`) i są dopisywane do zapisu, żeby plik nie tracił ustawień zakładek, których nie otwarto.
"""
import json
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QWidget

PLIK = Path(__file__).resolve().parent / 'ustawienia.json'
# EDYTOR_BEZ_USTAWIEN=1: start na ustawieniach domyślnych, bez wczytywania i zapisu pliku (testy)
WERSJA = 1
MODULY = ('app', 'widzety_edytora', 'zakladka_detekcji', 'zakladka_filmu', 'czasteczki', 'czasteczki3d',
          'przestrzen_makro', 'opcje_atencji', 'zakladka_ruchu')
POMIN = {'okno', 'suwak', 'pasek'}            # odnośnik wstecz do okna, suwaki przebiegu, paski postępu


def _kontener(w):
    """Widżet aplikacji, w którym szukamy dalszych kontrolek (zakładki, widoki, panele opcji)."""
    return isinstance(w, QWidget) and type(w).__module__ in MODULY


def kontrolki(korzen, sciezka='', widziane=None):
    """Słownik ścieżka -> kontrolka (lista, pole liczbowe, pole wyboru, przycisk przełączany)."""
    widziane = set() if widziane is None else widziane
    if id(korzen) in widziane:
        return {}
    widziane.add(id(korzen))
    wynik = {}
    for nazwa, w in sorted(vars(korzen).items()):
        if nazwa.startswith('_') or nazwa in POMIN:
            continue
        el = w.items() if isinstance(w, dict) else [(None, w)]
        for k, x in el:
            s = f'{sciezka}{nazwa}' + (f'.{k}' if k is not None else '')
            if isinstance(x, (QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox)):   # bez przycisków przełączanych
                                                                                  # (odtwarzanie, animacja, rysowanie)
                wynik[s] = x
            elif isinstance(x, QLineEdit) and type(korzen).__module__ == 'zakladka_ruchu':
                wynik[s] = x
            elif _kontener(x):
                wynik.update(kontrolki(x, s + '.', widziane))
    return wynik


def stan(korzen):
    wynik = {}
    for s, w in kontrolki(korzen).items():
        if isinstance(w, QComboBox):
            wynik[s] = dict(indeks=w.currentIndex(), tekst=w.currentText())
        elif isinstance(w, QAbstractSpinBox):
            wynik[s] = w.value()
        elif isinstance(w, QLineEdit):
            wynik[s] = w.text()
        else:
            wynik[s] = w.isChecked()
    for s, v in vars(korzen).get('_ustawienia_oczekujace', {}).items():      # zakładki jeszcze niezbudowane
        wynik.setdefault(s, v)
    return wynik


def ustaw(korzen, ustawienia):
    """Ustawia kontrolki według słownika; zwraca (ustawione, pominięte)."""
    k = kontrolki(korzen)
    ok, pom = 0, 0
    leniwe = vars(korzen).get('_leniwe', {})
    for s, v in ustawienia.items():
        w = k.get(s)
        if w is None:
            zakladka = s.split('.', 1)[0]
            if '.' in s and zakladka in leniwe and zakladka not in vars(korzen):   # zakładka jeszcze nie istnieje:
                vars(korzen).setdefault('_ustawienia_oczekujace', {})[s] = v          # ustawimy po jej zbudowaniu
                ok += 1
            else:
                pom += 1
            continue
        try:
            if isinstance(w, QComboBox):
                i = w.findText(v.get('tekst', ''))
                i = i if i >= 0 else (v.get('indeks', -1) if v.get('indeks', -1) < w.count() else -1)
                if i < 0:
                    pom += 1
                    continue
                if i != w.currentIndex():
                    w.setCurrentIndex(i)
            elif isinstance(w, QAbstractSpinBox):
                if w.value() != v:
                    w.setValue(v)
            elif isinstance(w, QLineEdit):
                if w.text() != str(v):
                    w.setText(str(v))
            elif w.isChecked() != bool(v):
                w.setChecked(bool(v))
            ok += 1
        except Exception:
            pom += 1
    return ok, pom


def zapisz(korzen, plik=PLIK):
    plik = Path(plik)
    dane = dict(wersja=WERSJA, zapisano=datetime.now().isoformat(timespec='seconds'), ustawienia=stan(korzen))
    plik.write_text(json.dumps(dane, ensure_ascii=False, indent=1), encoding='utf-8')
    return len(dane['ustawienia'])


def wczytaj(korzen, plik=PLIK):
    dane = json.loads(Path(plik).read_text(encoding='utf-8'))
    return ustaw(korzen, dane.get('ustawienia', {}))
