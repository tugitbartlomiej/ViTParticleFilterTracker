# -*- coding: utf-8 -*-
"""Deklaracja `WYNIKI.json`: umowa miedzy katalogiem eksperymentu a rodzina wynikow.

PO CO. Zwiazek `experiments/<x>` z `outputs/<rodzina>` istnial dotad wylacznie
jako twardo wpisany string w kodzie (`OUT = KAT / "outputs" / "jepa_kaskada"`).
Skaner rejestru czyta tylko `outputs/`, wiec o tym zwiazku nie wiedzial nic.
Skutkiem byly rozjazdy nazw (`dataset_studnie_syntetyczny` pisze do
`studnie_z_niezmiennikow`) i rodziny bez wlasciciela (`reports`, `figures`).

CO ROBI TEN PLIK. Definiuje schemat pliku `WYNIKI.json`, ktory lezy w katalogu
eksperymentu i mowi, do ktorej rodziny ten eksperyment pisze i jak nazywa swoje
artefakty. Daje tez `zapisz_wynik`, ktora ODMAWIA zapisu wyniku bez metadanych
pochodzenia - regula w dokumentacji nikogo nie powstrzymala, regula w kodzie
powstrzymuje.

Schemat:

    {
     "rodzina": "jepa_kaskada",          # katalog w outputs/, wymagane
     "klasa": "badawcza",                 # badawcza | robocza, wymagane
     "pytanie": "co ten eksperyment ma rozstrzygnac",
     "role": ["BIEG", "model", "LATENT"], # prefiksy nazw plikow
     "wymagane": ["tag", "ziarno"],       # klucze, bez ktorych JSON nie wejdzie
     "kontrole": ["_bezkary"],            # sufiksy tagu ramion kontrolnych
     "uwagi": ""
    }

`git_commit`, `czas_utc` i `eksperyment` dokleja `zapisz_wynik` sama - nie
trzeba ich wymieniac w `wymagane`.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

KORZEN = Path(__file__).resolve().parents[2]
EKSPERYMENTY = KORZEN / "experiments"
WYNIKI = KORZEN / "outputs"
NAZWA_DEKLARACJI = "WYNIKI.json"

# Katalogi w `experiments`, ktore nie sa eksperymentem i deklaracji nie potrzebuja.
POMIJANE = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}

POLA_WYMAGANE = ("rodzina", "klasa")
KLASY = ("badawcza", "robocza")

SZABLON = {
    "rodzina": "",
    "klasa": "badawcza",
    "pytanie": "",
    "role": ["BIEG", "model"],
    "wymagane": ["tag", "ziarno"],
    "kontrole": [],
    "uwagi": "",
}


class BladDeklaracji(RuntimeError):
    """Deklaracja nie istnieje albo jest niespojna. Zapis wyniku ma sie NIE udac."""


def katalogi_eksperymentow() -> list[Path]:
    if not EKSPERYMENTY.exists():
        return []
    return sorted(p for p in EKSPERYMENTY.iterdir()
                  if p.is_dir() and p.name not in POMIJANE
                  and not p.name.startswith("."))


def sciezka_deklaracji(katalog: Path) -> Path:
    return katalog / NAZWA_DEKLARACJI


def wczytaj(katalog: Path) -> dict:
    """Deklaracja katalogu eksperymentu. Rzuca, gdy jej nie ma albo jest zla."""
    p = sciezka_deklaracji(katalog)
    if not p.exists():
        raise BladDeklaracji(
            f"brak {p.relative_to(KORZEN)} - uruchom "
            f"`python -m src.narzedzia.sprawdz_eksperymenty "
            f"--zaloz {katalog.name} --rodzina <nazwa> --klasa badawcza`")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise BladDeklaracji(f"{p.relative_to(KORZEN)} nie jest poprawnym "
                             f"JSON-em: {type(e).__name__}") from e
    braki = [k for k in POLA_WYMAGANE if not d.get(k)]
    if braki:
        raise BladDeklaracji(f"{p.relative_to(KORZEN)}: brak pol {braki}")
    if d["klasa"] not in KLASY:
        raise BladDeklaracji(f"{p.relative_to(KORZEN)}: klasa musi byc jedna z "
                             f"{KLASY}, jest {d['klasa']!r}")
    return d


def wszystkie_deklaracje() -> tuple[dict, list[tuple[Path, str]]]:
    """({katalog: deklaracja}, [(katalog, powod bledu)]). Nie rzuca."""
    ok, zle = {}, []
    for k in katalogi_eksperymentow():
        try:
            ok[k] = wczytaj(k)
        except BladDeklaracji as e:
            zle.append((k, str(e)))
    return ok, zle


def zaloz(katalog: Path, rodzina: str, klasa: str = "badawcza",
          pytanie: str = "") -> Path:
    """Zaklada deklaracje. NIE nadpisuje istniejacej."""
    p = sciezka_deklaracji(katalog)
    if p.exists():
        raise BladDeklaracji(f"{p.relative_to(KORZEN)} juz istnieje")
    if klasa not in KLASY:
        raise BladDeklaracji(f"klasa musi byc jedna z {KLASY}")
    d = dict(SZABLON, rodzina=rodzina, klasa=klasa, pytanie=pytanie)
    katalog.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=KORZEN,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def _brudne_drzewo() -> bool:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=KORZEN,
                           capture_output=True, text=True, timeout=20)
        return bool(r.stdout.strip())
    except Exception:
        return False


def katalog_z_pliku(plik: str | Path) -> Path:
    """Katalog eksperymentu, w ktorym lezy dany plik zrodlowy."""
    p = Path(plik).resolve()
    try:
        wzgledna = p.relative_to(EKSPERYMENTY)
    except ValueError:
        raise BladDeklaracji(f"{p} nie lezy pod experiments/")
    return EKSPERYMENTY / wzgledna.parts[0]


def zapisz_wynik(plik_zrodlowy: str | Path, tag: str, dane: dict,
                 rola: str = "BIEG", podkatalog: str = "") -> Path:
    """Zapisuje wynik do `outputs/<rodzina>/<ROLA>_<tag>.json`.

    `plik_zrodlowy` to zwykle `__file__` skryptu eksperymentu - z niego bierze
    sie katalog, a z katalogu deklaracja. Funkcja ODMAWIA zapisu, gdy brakuje
    ktoregos z kluczy `wymagane`. To jedyne miejsce w projekcie, gdzie regula
    pochodzenia jest wymuszona, a nie tylko opisana.
    """
    katalog = katalog_z_pliku(plik_zrodlowy)
    d = wczytaj(katalog)

    dane = dict(dane)
    dane.setdefault("tag", tag)
    braki = [k for k in d.get("wymagane", []) if k not in dane]
    if braki:
        raise BladDeklaracji(
            f"{katalog.name}: wynik nie ma wymaganych kluczy {braki}; "
            f"lista pochodzi z {NAZWA_DEKLARACJI}")

    dane["eksperyment"] = katalog.name
    dane["git_commit"] = _commit()
    dane["drzewo_brudne"] = _brudne_drzewo()
    dane["czas_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cel = WYNIKI / d["rodzina"]
    if podkatalog:
        cel = cel / podkatalog
    cel.mkdir(parents=True, exist_ok=True)
    p = cel / f"{rola}_{tag}.json"
    p.write_text(json.dumps(dane, indent=1, ensure_ascii=False),
                 encoding="utf-8")
    return p


def katalog_wynikow(plik_zrodlowy: str | Path) -> Path:
    """Katalog `outputs/<rodzina>` dla wag, latentow i innych plikow binarnych."""
    d = wczytaj(katalog_z_pliku(plik_zrodlowy))
    cel = WYNIKI / d["rodzina"]
    cel.mkdir(parents=True, exist_ok=True)
    return cel


def _utf8():
    try:
        os.system("")
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
