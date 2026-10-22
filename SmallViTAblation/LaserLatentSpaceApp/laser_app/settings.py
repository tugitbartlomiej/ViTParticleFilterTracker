"""Zapis i odczyt ustawień aplikacji do pliku JSON (`settings.json` obok aplikacji, paths.SETTINGS_FILE).

Ustawienia to stan kontrolek: listy wyboru (indeks i tekst), pola liczbowe i pola wyboru (checkboxy), zebrane z atrybutów
okna i obiektów z pakietu laser_app (np. `zakladka_laserow.cb_og_G`, `zakladka_laserow.rzuty.cb_R12_baza`), oraz układ
okienek (`uklad_okienek` / `ustaw_uklad_okienek` zakładki). Nie zapisuje danych ani suwaków przebiegu.

- Plik domyślny: wczytywany przy starcie, zapisywany przy zamknięciu okna; plik sprzed refaktoru (`ustawienia.json`)
  jest przy pierwszym starcie przenoszony pod nową nazwę (`przenies_stary_plik`). Klucze są nazwami atrybutów, więc
  zmiana nazw modułów i klas ich nie zmienia.
- Menu „Plik”: zapis i wczytanie dowolnego pliku, przywrócenie domyślnych (stan zaraz po uruchomieniu).
Listy z pozycjami zależnymi od danych (np. film) wracają po tekście; gdy pozycji nie ma, zostają.
EDYTOR_BEZ_USTAWIEN=1: start na ustawieniach domyślnych, bez wczytywania i zapisu pliku (testy).
"""
import json
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QWidget

from laser_app.paths import OLD_SETTINGS_FILE, SETTINGS_FILE

PLIK = SETTINGS_FILE
WERSJA = 1
PAKIET = 'laser_app'                          # kontrolki szukamy w obiektach klas z tego pakietu (okno, zakładka, rzuty)
POMIN = {'okno', 'suwak', 'pasek', 'cb_og_gotowe'}   # odnośnik wstecz do okna, suwaki przebiegu, paski postępu, lista plików


def _kontener(w):
    """Widżet aplikacji, w którym szukamy dalszych kontrolek (zakładki, widoki, panele opcji)."""
    return isinstance(w, QWidget) and type(w).__module__.split('.')[0] == PAKIET


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
    for nazwa, z in vars(korzen).items():                              # układ okienek zakładek (np. doki „Lasery i koloidy”)
        if nazwa.startswith('zakladka_') and hasattr(z, 'uklad_okienek'):
            try:
                wynik[f'{nazwa}.uklad_okienek'] = z.uklad_okienek()
            except Exception:                                          # noqa: BLE001
                pass
    for s, v in vars(korzen).get('_ustawienia_oczekujace', {}).items():      # zakładki jeszcze niezbudowane
        wynik.setdefault(s, v)
    return wynik


def ustaw(korzen, ustawienia):
    """Ustawia kontrolki według słownika; zwraca (ustawione, pominięte)."""
    k = kontrolki(korzen)
    ok, pom = 0, 0
    leniwe = vars(korzen).get('_leniwe', {})
    edytor_czeka = vars(korzen).get('_edytor_gotowy') is False          # edytor SD (kontrolki okna) jeszcze niezbudowany
    for s, v in ustawienia.items():
        if s.endswith('.uklad_okienek') and hasattr(vars(korzen).get(s.split('.', 1)[0]), 'ustaw_uklad_okienek'):
            vars(korzen)[s.split('.', 1)[0]].ustaw_uklad_okienek(v); ok += 1
            continue
        w = k.get(s)
        if w is None:
            zakladka = s.split('.', 1)[0]
            if (edytor_czeka and zakladka not in leniwe) or \
                    ('.' in s and zakladka in leniwe and zakladka not in vars(korzen)):   # zakładka jeszcze nie istnieje:
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
                    if w.count() == 0 and v.get('indeks', -1) < 0 and not v.get('tekst', ''):   # pusta lista zapisana i pusta teraz (zestaw bez zdjęć)
                        ok += 1
                        continue
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


def przenies_stary_plik():
    """`ustawienia.json` sprzed refaktoru → `settings.json` (raz; gdy nowy plik już jest, stary zostaje nietknięty)."""
    if not SETTINGS_FILE.exists() and OLD_SETTINGS_FILE.exists():
        try:
            OLD_SETTINGS_FILE.replace(SETTINGS_FILE)
        except OSError:
            pass


def zapisz(korzen, plik=None):
    plik = Path(PLIK if plik is None else plik)
    dane = dict(wersja=WERSJA, zapisano=datetime.now().isoformat(timespec='seconds'), ustawienia=stan(korzen))
    plik.write_text(json.dumps(dane, ensure_ascii=False, indent=1), encoding='utf-8')
    return len(dane['ustawienia'])


def wczytaj(korzen, plik=None):
    dane = json.loads(Path(PLIK if plik is None else plik).read_text(encoding='utf-8'))
    return ustaw(korzen, dane.get('ustawienia', {}))
