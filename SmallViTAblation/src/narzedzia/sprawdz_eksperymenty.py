# -*- coding: utf-8 -*-
"""Sprawdza, czy kazdy eksperyment deklaruje, do ktorej rodziny wynikow pisze.

PO CO. Skaner `skan_wynikow.py` widzi tylko `outputs/`. Nie wie, ktory
eksperyment wyprodukowal dana rodzine, wiec nowy katalog eksperymentu przechodzi
niezauwazony, a nowa rodzina wynikow laduje w indeksie jako `nieznana`. Ten
skrypt domyka druga strone: czyta deklaracje `WYNIKI.json` z katalogow
eksperymentow i zestawia je z rodzinami w `outputs/`.

Rejestracji NIE robi sam z siebie. Podzial na rodzine badawcza i robocza to
decyzja o tym, czy dany bieg jest pomiarem, czy proba dymna - skrypt tego nie
wie. Wykrywa i pyta; wpis zaklada czlowiek albo agent.

Uruchomienie:
    python -m src.narzedzia.sprawdz_eksperymenty              # pelny raport
    python -m src.narzedzia.sprawdz_eksperymenty --nowe       # same braki, po linii
    python -m src.narzedzia.sprawdz_eksperymenty --zaloz rytm_syntetyczny \
        --rodzina rytm_syntetyczny --klasa badawcza --pytanie "..."
    python -m src.narzedzia.sprawdz_eksperymenty --przed-commitem
    python -m src.narzedzia.sprawdz_eksperymenty --hook       # z hooka Claude Code
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.narzedzia.deklaracja_wynikow import (  # noqa: E402
    EKSPERYMENTY, KORZEN, NAZWA_DEKLARACJI, WYNIKI, BladDeklaracji,
    sciezka_deklaracji, wszystkie_deklaracje, zaloz, _utf8,
)

PLIK_RODZIN = WYNIKI / "rejestr" / "rodziny.json"
# Rodziny zbiorcze: pisze do nich kilka eksperymentow naraz przez domyslny
# `out_dir`. Nie maja i nie beda mialy jednego wlasciciela, wiec nie zglaszamy
# ich jako bledu - tylko jako fakt do posprzatania osobna decyzja.
ZBIORCZE = {"reports", "figures", "ckpt", "logs", "smoke", "telemetry",
            "rejestr"}


def rodziny_z_rejestru() -> dict:
    if not PLIK_RODZIN.exists():
        return {}
    try:
        return json.loads(PLIK_RODZIN.read_text(encoding="utf-8"))
    except Exception:
        return {}


def rodziny_na_dysku() -> set[str]:
    if not WYNIKI.exists():
        return set()
    return {p.name for p in WYNIKI.iterdir()
            if p.is_dir() and p.name != "rejestr"}


def zbierz() -> dict:
    """Caly obraz zaleznosci experiments <-> outputs, bez wypisywania."""
    dekl, zle = wszystkie_deklaracje()
    kl = rodziny_z_rejestru()
    na_dysku = rodziny_na_dysku()

    wlasciciele: dict[str, list[str]] = {}
    for kat, d in dekl.items():
        wlasciciele.setdefault(d["rodzina"], []).append(kat.name)

    return {
        "deklaracje": {k.name: d for k, d in dekl.items()},
        "bez_deklaracji": [(k.name, powod) for k, powod in zle],
        "kolizje": {r: w for r, w in wlasciciele.items() if len(w) > 1},
        "rodzina_bez_katalogu": sorted(
            r for r in wlasciciele if r not in na_dysku),
        "rodzina_bez_wlasciciela": sorted(
            r for r in na_dysku
            if r not in wlasciciele and r not in ZBIORCZE),
        "niezgodna_klasa": sorted(
            f"{d['rodzina']}: deklaracja {d['klasa']}, rejestr {kl[d['rodzina']]}"
            for d in dekl.values()
            if d["rodzina"] in kl and kl[d["rodzina"]] != d["klasa"]),
        "brak_w_rejestrze": sorted(
            d["rodzina"] for d in dekl.values() if d["rodzina"] not in kl),
    }


def dopisz_do_rejestru(rodzina: str, klasa: str) -> bool:
    """Dopisuje rodzine do `rodziny.json`. Istniejacego wpisu NIE zmienia."""
    kl = rodziny_z_rejestru()
    if kl.get(rodzina) == klasa:
        return False
    kl[rodzina] = klasa
    PLIK_RODZIN.parent.mkdir(parents=True, exist_ok=True)
    PLIK_RODZIN.write_text(
        json.dumps(dict(sorted(kl.items())), indent=1, ensure_ascii=False),
        encoding="utf-8")
    return True


def raport(o: dict) -> int:
    """Wypisuje raport. Zwraca liczbe spraw wymagajacych decyzji czlowieka."""
    do_decyzji = 0

    if o["bez_deklaracji"]:
        do_decyzji += len(o["bez_deklaracji"])
        print(f"\nEKSPERYMENTY BEZ DEKLARACJI ({len(o['bez_deklaracji'])}):")
        for nazwa, powod in o["bez_deklaracji"]:
            krotki = powod.split(" - ")[0]
            print(f"  {nazwa:<32} {krotki}")
        print("  decyzja: do ktorej rodziny pisza i czy sa badawcze, "
              "czy robocze")

    if o["kolizje"]:
        do_decyzji += len(o["kolizje"])
        print(f"\nJEDNA RODZINA, KILKU WLASCICIELI ({len(o['kolizje'])}):")
        for r, w in sorted(o["kolizje"].items()):
            print(f"  {r:<32} {', '.join(w)}")

    if o["rodzina_bez_wlasciciela"]:
        print(f"\nRODZINY W outputs/ BEZ EKSPERYMENTU "
              f"({len(o['rodzina_bez_wlasciciela'])}):")
        for r in o["rodzina_bez_wlasciciela"]:
            print(f"  {r}")
        print("  to biegi starsze niz mechanizm deklaracji albo katalogi "
              "po przenosinach")

    if o["rodzina_bez_katalogu"]:
        print(f"\nZADEKLAROWANE, ALE JESZCZE NIC NIE ZAPISALY "
              f"({len(o['rodzina_bez_katalogu'])}):")
        for r in o["rodzina_bez_katalogu"]:
            print(f"  {r}")

    if o["brak_w_rejestrze"]:
        do_decyzji += len(o["brak_w_rejestrze"])
        print(f"\nRODZINY SPOZA rodziny.json ({len(o['brak_w_rejestrze'])}):")
        for r in o["brak_w_rejestrze"]:
            print(f"  {r}")

    if o["niezgodna_klasa"]:
        do_decyzji += len(o["niezgodna_klasa"])
        print(f"\nKLASA W DEKLARACJI KLOCI SIE Z REJESTREM "
              f"({len(o['niezgodna_klasa'])}):")
        for w in o["niezgodna_klasa"]:
            print(f"  {w}")

    ile = len(o["deklaracje"])
    print(f"\nzadeklarowanych eksperymentow: {ile}, "
          f"bez deklaracji: {len(o['bez_deklaracji'])}, "
          f"do decyzji: {do_decyzji}")
    return do_decyzji


# --- tryb hooka Claude Code ------------------------------------------------

def _katalog_z_hooka(sciezka: str) -> Path | None:
    """Katalog eksperymentu, w ktorym lezy zapisany plik, albo None.

    Plik lezacy WPROST w `experiments/` (README.md, __init__.py) nie nalezy do
    zadnego eksperymentu - stad warunek na co najmniej dwie czesci sciezki
    i na to, ze pierwsza z nich jest katalogiem.
    """
    try:
        wzgl = Path(sciezka).resolve().relative_to(EKSPERYMENTY)
    except Exception:
        return None
    if len(wzgl.parts) < 2:
        return None
    kat = EKSPERYMENTY / wzgl.parts[0]
    return kat if kat.is_dir() else None


def tryb_hooka() -> int:
    """Wolane przez PostToolUse na Write/Edit. Zawsze konczy sie kodem 0.

    Hook nie moze przerywac pracy ani sypac wyjatkiem, wiec kazdy blad konczy
    sie cisza. Gdy katalog nie ma deklaracji, hook wstrzykuje polecenie
    zapytania uzytkownika - decyzje o klasie rodziny podejmuje czlowiek.
    """
    try:
        wej = json.load(sys.stdin)
    except Exception:
        return 0
    sciezka = ((wej.get("tool_response") or {}).get("filePath")
               or (wej.get("tool_input") or {}).get("file_path") or "")
    if not sciezka:
        return 0
    katalog = _katalog_z_hooka(str(sciezka))
    if katalog is None or katalog.name in {"__pycache__"}:
        return 0
    if sciezka_deklaracji(katalog).exists():
        return 0
    if not katalog.exists():
        return 0

    nazwa = katalog.name
    komunikat = (f"Eksperyment '{nazwa}' nie ma pliku {NAZWA_DEKLARACJI} - "
                 f"nie wiadomo, do ktorej rodziny w outputs/ pisze.")
    kontekst = (
        f"HOOK REJESTRU EKSPERYMENTOW: katalog experiments/{nazwa} nie ma "
        f"deklaracji {NAZWA_DEKLARACJI}, wiec jego wyniki trafia do outputs/ "
        f"bez wlasciciela.\n"
        f"ZANIM skonczysz ta ture: zapytaj uzytkownika narzedziem "
        f"AskUserQuestion o dwie rzeczy - (1) nazwe rodziny w outputs/, "
        f"do ktorej ten eksperyment ma pisac (domyslnie '{nazwa}'), "
        f"(2) czy rodzina jest 'badawcza' (pomiar, na ktorym moga stanac tezy) "
        f"czy 'robocza' (biegi dymne, logi, checkpointy przejsciowe).\n"
        f"Potem zaloz deklaracje poleceniem:\n"
        f"  python -m src.narzedzia.sprawdz_eksperymenty --zaloz {nazwa} "
        f"--rodzina <rodzina> --klasa <badawcza|robocza> --pytanie \"...\"\n"
        f"Nie zgaduj klasy samodzielnie - to decyzja uzytkownika.")
    print(json.dumps({
        "systemMessage": komunikat,
        "suppressOutput": True,
        "hookSpecificOutput": {"hookEventName": "PostToolUse",
                               "additionalContext": kontekst},
    }, ensure_ascii=False))
    return 0


# --- tryb pre-commit -------------------------------------------------------

def tryb_przed_commitem() -> int:
    """Sprawdza katalogi eksperymentow dotkniete przez commit. 1 = blokuj."""
    try:
        r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                           cwd=KORZEN, capture_output=True, text=True,
                           timeout=30)
        pliki = [x for x in r.stdout.splitlines() if x.strip()]
    except Exception:
        return 0

    dotkniete = set()
    for f in pliki:
        czesci = Path(f).parts
        if len(czesci) >= 2 and czesci[0] == "experiments":
            dotkniete.add(czesci[1])

    braki = [n for n in sorted(dotkniete)
             if (EKSPERYMENTY / n).is_dir()
             and n not in {"__pycache__"}
             and not sciezka_deklaracji(EKSPERYMENTY / n).exists()]
    if not braki:
        return 0

    print("Eksperymenty bez deklaracji " + NAZWA_DEKLARACJI + ":")
    for n in braki:
        print(f"  experiments/{n}")
    print("\nBez niej wyniki tego eksperymentu trafia do outputs/ bez "
          "wlasciciela i wypadna z rejestru jako sieroty.")
    print("Zaloz ja tak:")
    for n in braki:
        print(f"  python -m src.narzedzia.sprawdz_eksperymenty --zaloz {n} "
              f"--rodzina {n} --klasa badawcza --pytanie \"...\"")
    return 1


def main() -> int:
    _utf8()
    ap = argparse.ArgumentParser(description="Deklaracje eksperymentow")
    ap.add_argument("--nowe", action="store_true",
                    help="wypisz same katalogi bez deklaracji, po linii")
    ap.add_argument("--json", action="store_true", help="caly obraz jako JSON")
    ap.add_argument("--zaloz", metavar="KATALOG",
                    help="zaloz deklaracje dla katalogu eksperymentu")
    ap.add_argument("--rodzina", help="rodzina w outputs/ (do --zaloz)")
    ap.add_argument("--klasa", choices=("badawcza", "robocza"),
                    help="klasa rodziny (do --zaloz)")
    ap.add_argument("--pytanie", default="",
                    help="co ten eksperyment ma rozstrzygnac")
    ap.add_argument("--hook", action="store_true",
                    help="tryb hooka Claude Code, wejscie JSON na stdin")
    ap.add_argument("--przed-commitem", action="store_true",
                    dest="przed_commitem",
                    help="sprawdz tylko katalogi dotkniete przez commit")
    a = ap.parse_args()

    if a.hook:
        return tryb_hooka()
    if a.przed_commitem:
        return tryb_przed_commitem()

    if a.zaloz:
        if not a.rodzina or not a.klasa:
            print("--zaloz wymaga --rodzina i --klasa", file=sys.stderr)
            return 2
        katalog = EKSPERYMENTY / a.zaloz
        try:
            p = zaloz(katalog, a.rodzina, a.klasa, a.pytanie)
        except BladDeklaracji as e:
            print(f"nie zalozono: {e}", file=sys.stderr)
            return 2
        zmiana = dopisz_do_rejestru(a.rodzina, a.klasa)
        print(f"zalozono {p.relative_to(KORZEN)}")
        print(f"rodziny.json: {'dopisano ' + a.rodzina if zmiana else 'bez zmian'}")
        return 0

    o = zbierz()
    if a.json:
        print(json.dumps(o, indent=1, ensure_ascii=False))
        return 0
    if a.nowe:
        for nazwa, _ in o["bez_deklaracji"]:
            print(nazwa)
        return 0
    return 1 if raport(o) else 0


if __name__ == "__main__":
    raise SystemExit(main())
