#!/usr/bin/env python3
"""Rysuje przykladowe klatki Cataract-LMM z naniesionymi adnotacjami.

Dwa zrodla, bo maja zupelnie inny format:
  2_Instance_Segmentation - maski i ramki dziesieciu narzedzi plus rogowka i zrenica
  3_Object_Tracking       - jedno narzedzie (cystotom), ale z punktem KONCOWKI

Uzycie:
    py -3.11 podglad_adnotacji.py [katalog_zbioru] [katalog_wyjsciowy]
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else
         r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else D / "_podglad")
OUT.mkdir(parents=True, exist_ok=True)

# kolory BGR: anatomia chlodna, narzedzia ciepłe
KOLOR = {
    "Cornea": (255, 200, 0), "Pupil": (255, 120, 60),
    "_narzedzie": (0, 220, 60), "_koncowka": (0, 0, 255),
}


def kolor(nazwa: str) -> tuple[int, int, int]:
    return KOLOR.get(nazwa, KOLOR["_narzedzie"])


def podpis(img, tekst, x, y, kol):
    (w, h), _ = cv2.getTextSize(tekst, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, h + 6)
    cv2.rectangle(img, (x, y - h - 5), (x + w + 6, y + 3), kol, -1)
    cv2.putText(img, tekst, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def segmentacja(n: int = 4) -> None:
    coco = D / "2_Instance_Segmentation/annotations/coco_json/_annotations.coco.json"
    root = next((p.parent for p in
                 (D / "2_Instance_Segmentation/annotations/coco_json/images").rglob("*.jpg")), None)
    if root is None:
        print("Brak rozpakowanych klatek segmentacji - pomijam.")
        return

    d = json.loads(coco.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    per_img = defaultdict(list)
    for a in d["annotations"]:
        per_img[a["image_id"]].append(a)

    # wybierz klatki z NAJWIEKSZA liczba roznych narzedzi, po dwie z kazdej rozdzielczosci
    def liczba_narzedzi(iid):
        return len({cats[a["category_id"]] for a in per_img[iid]} - {"Cornea", "Pupil"})

    wybrane = []
    for res in [(1920, 1080), (720, 480)]:
        kand = [i for i in per_img if (imgs[i]["width"], imgs[i]["height"]) == res]
        kand.sort(key=liczba_narzedzi, reverse=True)
        wybrane += kand[: n // 2]

    for k, iid in enumerate(wybrane, 1):
        im = imgs[iid]
        p = root / im["file_name"]
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        nakl = img.copy()
        for a in sorted(per_img[iid], key=lambda a: cats[a["category_id"]] not in ("Cornea", "Pupil")):
            nazwa = cats[a["category_id"]]
            kol = kolor(nazwa)
            for seg in a.get("segmentation") or []:
                if len(seg) >= 6:
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(nakl, [pts], kol)
        img = cv2.addWeighted(nakl, 0.35, img, 0.65, 0)
        for a in per_img[iid]:
            nazwa = cats[a["category_id"]]
            kol = kolor(nazwa)
            x, y, w, h = [int(v) for v in a["bbox"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), kol, 2)
            podpis(img, f"{nazwa} {w}x{h}px", x, y, kol)
        opis = f'SEGMENTACJA  {im["width"]}x{im["height"]}  |  {im["file_name"][:38]}'
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, opis, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        f = OUT / f"segmentacja_{k}_{im['width']}x{im['height']}.jpg"
        cv2.imwrite(str(f), img)
        print("zapisano:", f.name)


def sledzenie(klip: str = "TR_0129_S1_P03", ile: int = 3) -> None:
    z = D / "3_Object_Tracking/annotations" / f"{klip}.zip"
    if not z.exists():
        print(f"Brak {z.name} - pomijam.")
        return
    with zipfile.ZipFile(z) as f:
        nazwa_csv = [n for n in f.namelist() if n.endswith("annotation.csv")][0]
        rows = list(csv.DictReader(io.StringIO(
            f.read(nazwa_csv).decode("utf-8", "replace"))))
        # klatki, na ktorych koncowka JEST podana - to jest sedno tego zbioru
        z_tipem = [r for r in rows if r.get("cystotome_tip", "").strip()]
        wybrane = z_tipem[:: max(1, len(z_tipem) // ile)][:ile]

        for k, r in enumerate(wybrane, 1):
            sciezka = f"{klip}/{r['filename']}"
            if sciezka not in f.namelist():
                continue
            img = cv2.imdecode(np.frombuffer(f.read(sciezka), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            nakl = img.copy()
            for pole, nazwa in (("cornea", "Cornea"), ("pupil", "Pupil"),
                                ("cystotome", "Cap-Cystotome")):
                seg = r.get(f"{pole}_seg", "").strip()
                if seg:
                    try:
                        for wielokat in literal_eval(seg):
                            pts = np.array(wielokat, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(nakl, [pts], kolor(nazwa))
                    except Exception:
                        pass
            img = cv2.addWeighted(nakl, 0.35, img, 0.65, 0)

            for pole, nazwa in (("cornea", "Cornea"), ("pupil", "Pupil"),
                                ("cystotome", "Cap-Cystotome")):
                bb = r.get(f"{pole}_bbox", "").strip()
                if bb:
                    try:
                        x, y, w, h = [int(round(v)) for v in literal_eval(bb)]
                        kol = kolor(nazwa)
                        cv2.rectangle(img, (x, y), (x + w, y + h), kol, 2)
                        podpis(img, f"{nazwa} {w}x{h}px", x, y, kol)
                    except Exception:
                        pass

            tip = r.get("cystotome_tip", "").strip()
            if tip:
                try:
                    tx, ty = [int(round(v)) for v in literal_eval(tip)]
                    cv2.drawMarker(img, (tx, ty), KOLOR["_koncowka"],
                                   cv2.MARKER_CROSS, 26, 3)
                    cv2.circle(img, (tx, ty), 13, KOLOR["_koncowka"], 2)
                    podpis(img, f"KONCOWKA ({tx},{ty})", tx + 16, ty, KOLOR["_koncowka"])
                except Exception:
                    pass

            opis = f'SLEDZENIE  {klip}  klatka {r["frame_number"]}  {img.shape[1]}x{img.shape[0]}'
            cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(img, opis, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            fpath = OUT / f"sledzenie_{k}_klatka{r['frame_number']}.jpg"
            cv2.imwrite(str(fpath), img)
            print("zapisano:", fpath.name)


if __name__ == "__main__":
    print(f"Wyjscie: {OUT}\n")
    segmentacja()
    sledzenie()
