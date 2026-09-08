#!/usr/bin/env python3
"""
Audyt wyciekow miedzy splitami dla zbiorow DETR.

Sprawdza trzy poziomy naraz, bo kazdy lapie inny rodzaj bledu:

  1. PACJENT  - czy ten sam pacjent (stem nazwy pliku) wystepuje w dwoch splitach.
                To jest wyciek, ktory zarzucil recenzent 1 w punkcie 1.
  2. KLATKA ZRODLOWA - czy ta sama klatka zrodlowa wystepuje w dwoch splitach
                pod roznymi wariantami augmentacji (np. oryginal w train,
                _aug_2 w test). Tego nie widac po nazwach plikow.
  3. PLIK     - czy dokladnie ten sam plik jest w dwoch splitach.

Konwencja nazw: <pacjent>_frame_<numer>[_aug_<n>].jpg

Uzycie:
    py -3.11 tools/audit_leakage.py <katalog_z_annotations_{train,val,test}.json> ...
    py -3.11 tools/audit_leakage.py --all       # wszystkie znane zbiory projektu
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

AUG = re.compile(r"_aug_\d+(?=\.\w+$)")


def patient(name: str) -> str:
    return name.split("_frame_")[0] if "_frame_" in name else name.rsplit(".", 1)[0]


def source_frame(name: str) -> str:
    return AUG.sub("", name)


def load_split(d: Path, split: str) -> list[str] | None:
    f = d / f"annotations_{split}.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text(encoding="utf-8"))
    return [im["file_name"] for im in data["images"]]


def audit(d: Path) -> dict:
    splits = {}
    for s in ("train", "val", "test"):
        names = load_split(d, s)
        if names is not None:
            splits[s] = names
    if len(splits) < 2:
        return {"dir": str(d), "error": "mniej niz dwa splity - nie ma czego porownac"}

    res = {"dir": str(d), "counts": {}, "overlaps": {}, "verdict": "CZYSTY"}
    sets = {}
    for s, names in splits.items():
        pats = {patient(n) for n in names}
        srcs = {source_frame(n) for n in names}
        files = set(names)
        sets[s] = (pats, srcs, files)
        res["counts"][s] = {"obrazy": len(names), "klatki_zrodlowe": len(srcs),
                            "pacjenci": len(pats)}

    for a, b in combinations(sets, 2):
        pa, sa, fa = sets[a]
        pb, sb, fb = sets[b]
        ov = {"pacjenci": sorted(pa & pb),
              "klatki_zrodlowe": len(sa & sb),
              "pliki": len(fa & fb)}
        res["overlaps"][f"{a}-{b}"] = ov
        if ov["pacjenci"] or ov["klatki_zrodlowe"] or ov["pliki"]:
            res["verdict"] = "WYCIEK"
    return res


def report(r: dict) -> None:
    print(f"\n=== {r['dir']}")
    if "error" in r:
        print(f"    {r['error']}")
        return
    for s, c in r["counts"].items():
        print(f"    {s:5} obrazy={c['obrazy']:6}  klatki_zrodlowe={c['klatki_zrodlowe']:6}"
              f"  pacjenci={c['pacjenci']:3}")
    for pair, ov in r["overlaps"].items():
        flag = "OK " if not (ov["pacjenci"] or ov["klatki_zrodlowe"] or ov["pliki"]) else "!! "
        print(f"    {flag}{pair:12} wspolni pacjenci={len(ov['pacjenci'])}"
              f"  wspolne klatki zrodlowe={ov['klatki_zrodlowe']}"
              f"  wspolne pliki={ov['pliki']}")
        if ov["pacjenci"]:
            print(f"        pacjenci po obu stronach: {', '.join(ov['pacjenci'][:12])}")
    print(f"    WERDYKT: {r['verdict']}")


DEFAULT_DIRS = [
    REPO / "Eden/EdenBackup/DETR/fullpool_videosplit",
    *[REPO / f"Eden/EdenBackup/DETR/ablation/v{i}" for i in range(1, 8)],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    dirs = args.dirs or (DEFAULT_DIRS if args.all else [])
    if not dirs:
        sys.exit("Podaj katalogi albo --all")

    verdicts = []
    for d in dirs:
        r = audit(Path(d))
        report(r)
        verdicts.append((str(d), r.get("verdict", "BLAD")))

    print("\n" + "=" * 60)
    bad = [d for d, v in verdicts if v != "CZYSTY"]
    print(f"Sprawdzono: {len(verdicts)} | czystych: {len(verdicts) - len(bad)} | z wyciekiem: {len(bad)}")
    for d in bad:
        print(f"  WYCIEK: {d}")


if __name__ == "__main__":
    main()
