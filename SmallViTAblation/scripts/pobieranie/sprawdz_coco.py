#!/usr/bin/env python3
"""Podglad pobranych adnotacji Cataract-LMM - co tam jest i czy pasuje do nas.

Wolany przez 5_sprawdz_pobrane.bat. Nie potrzebuje kart graficznej ani obrazow -
czyta sam plik JSON.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DEST = Path(sys.argv[1] if len(sys.argv) > 1 else r"F:\Cataract-LMM")
COCO = DEST / "2_Instance_Segmentation" / "annotations" / "coco_json" / "_annotations.coco.json"


def main() -> None:
    if not COCO.exists():
        print(f"BRAK PLIKU: {COCO}")
        print("Uruchom najpierw 1_pobierz_ramki_coco.bat")
        return

    d = json.loads(COCO.read_text(encoding="utf-8"))
    imgs, anns, cats = d["images"], d["annotations"], d["categories"]

    print(f"obrazow:    {len(imgs):,}")
    print(f"adnotacji:  {len(anns):,}")
    print(f"klas:       {len(cats)}")
    print()
    print("KLASY:")
    per_cat = Counter(a["category_id"] for a in anns)
    for c in cats:
        print(f"  {c['id']:>3}  {c['name']:<28} {per_cat.get(c['id'], 0):>6} instancji")

    print()
    res = Counter((im.get("width"), im.get("height")) for im in imgs)
    print("ROZDZIELCZOSCI:")
    for (w, h), n in res.most_common():
        zrodlo = "Noor, pelne HD" if (w, h) == (1920, 1080) else "Farabi" if (w, h) == (720, 480) else ""
        print(f"  {w}x{h}: {n:>6} obrazow   {zrodlo}")

    print()
    print("RAMKI (bbox w formacie COCO: x, y, szerokosc, wysokosc):")
    ws = [a["bbox"][2] for a in anns if "bbox" in a]
    hs = [a["bbox"][3] for a in anns if "bbox" in a]
    if ws:
        print(f"  mediana szerokosci: {statistics.median(ws):.0f} px")
        print(f"  mediana wysokosci:  {statistics.median(hs):.0f} px")
        print("  przyklad:", [round(v, 1) for v in anns[0]["bbox"]])

    print()
    print("CZY MAJA MASKI:", "tak" if any("segmentation" in a and a["segmentation"] for a in anns[:50]) else "nie")

    print()
    print("--- Dla porownania NASZE dane ---")
    print("  1920x1080, mediana ramki 94x128 px, czyli 0,56% powierzchni obrazu")
    print("  jedna klasa: koncowka narzedzia")
    print()
    print("Do ustalenia: ktore z ich klas odpowiadaja naszej jednej klasie")
    print("i czy ich ramka obejmuje cale narzedzie, czy sama koncowke.")


if __name__ == "__main__":
    main()
