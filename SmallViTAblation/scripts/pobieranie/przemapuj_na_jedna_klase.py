#!/usr/bin/env python3
"""Przemapowanie Cataract-LMM z dziesieciu narzedzi na JEDNA klase.

Po co: artykul bada dobor danych, a nie rozpoznawanie typu narzedzia. Dziesiec
klas dolozyloby druga zmienna do eksperymentu, a najrzadsze narzedzie ma tylko
111 wystapien - po podziale na czesci zostaloby okolo trzydziestu w tescie,
wiec srednia po klasach skakalaby losowo.

Co robi:
  - scala dziesiec klas narzedzi w jedna: "instrument",
  - anatomie (rogowka, zrenica) zapisuje OSOBNO, do wyboru,
  - NIE scala ramek: gdy na klatce sa dwa narzedzia, zostaja dwie ramki,
  - zachowuje maski, rozmiary obrazow i identyfikatory,
  - zapisuje mapowanie stara klasa -> nowa do pliku, zeby dalo sie odtworzyc.

Uzycie:
    py -3.11 przemapuj_na_jedna_klase.py [katalog_zbioru]
    py -3.11 przemapuj_na_jedna_klase.py --with-anatomy   # anatomia w tym samym pliku
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ANATOMIA = {"Cornea", "Pupil"}
PUSTE = {"my-dataset-main"}          # artefakt eksportu z Roboflow, zero instancji


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", type=Path,
                    default=Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                                 r"\Dataset\Cataract-LMM"))
    ap.add_argument("--with-anatomy", action="store_true",
                    help="zostaw rogowke i zrenice w tym samym pliku jako osobne klasy")
    args = ap.parse_args()

    src = args.dataset / "2_Instance_Segmentation/annotations/coco_json/_annotations.coco.json"
    if not src.exists():
        raise SystemExit(f"Brak pliku: {src}")

    d = json.loads(src.read_text(encoding="utf-8"))
    stare = {c["id"]: c["name"] for c in d["categories"]}

    narzedzia = sorted(n for n in stare.values() if n not in ANATOMIA | PUSTE)

    # nowe kategorie: 1 = instrument, potem opcjonalnie anatomia
    nowe_cats = [{"id": 1, "name": "instrument", "supercategory": "tool"}]
    mapa = {cid: 1 for cid, n in stare.items() if n in narzedzia}
    if args.with_anatomy:
        for i, n in enumerate(sorted(ANATOMIA), start=2):
            nowe_cats.append({"id": i, "name": n.lower(), "supercategory": "anatomy"})
            for cid, nazwa in stare.items():
                if nazwa == n:
                    mapa[cid] = i

    anns_out, licz = [], collections.Counter()
    for a in d["annotations"]:
        nazwa = stare[a["category_id"]]
        if a["category_id"] not in mapa:
            licz[f"POMINIETE: {nazwa}"] += 1
            continue
        b = dict(a)
        b["category_id"] = mapa[a["category_id"]]
        anns_out.append(b)
        licz[nazwa] += 1

    out = {
        "info": {**d.get("info", {}),
                 "description": "Cataract-LMM, dziesiec klas narzedzi scalone w jedna",
                 "remap": "10 tool classes -> 'instrument'",
                 "anatomy_included": args.with_anatomy},
        "licenses": d.get("licenses", []),
        "images": d["images"],
        "categories": nowe_cats,
        "annotations": anns_out,
    }
    nazwa_pliku = ("_annotations.single_class_with_anatomy.coco.json"
                   if args.with_anatomy else "_annotations.single_class.coco.json")
    dst = src.with_name(nazwa_pliku)
    dst.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # mapowanie osobno, zeby dalo sie odtworzyc co z czego powstalo
    mapa_txt = {stare[cid]: next(c["name"] for c in nowe_cats if c["id"] == nid)
                for cid, nid in mapa.items()}
    dst.with_suffix(".mapping.json").write_text(
        json.dumps({"mapping": mapa_txt,
                    "dropped": sorted(set(stare.values()) - set(mapa_txt))},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # raport
    n_tool = sum(1 for a in anns_out if a["category_id"] == 1)
    per_img = collections.Counter()
    for a in anns_out:
        if a["category_id"] == 1:
            per_img[a["image_id"]] += 1
    rozklad = collections.Counter(per_img.get(i["id"], 0) for i in d["images"])

    print(f"Zapisano: {dst.name}")
    print(f"          {dst.with_suffix('.mapping.json').name}\n")
    print(f"obrazow:            {len(d['images']):,}")
    print(f"ramek narzedzia:    {n_tool:,}   (klasa 'instrument')")
    if args.with_anatomy:
        print(f"ramek anatomii:     {len(anns_out) - n_tool:,}")
    print("\nSCALONE KLASY:")
    for n in narzedzia:
        print(f"  {n:<22} {licz[n]:>5}")
    print("\nNARZEDZI NA KLATCE PO SCALENIU:")
    for k in sorted(rozklad):
        print(f"  {k}: {rozklad[k]:>5} klatek ({rozklad[k]/len(d['images'])*100:>5.1f}%)")


if __name__ == "__main__":
    main()
