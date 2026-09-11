#!/usr/bin/env python3
"""Losuje probke koncowek narzedzia i sklada je w arkusze do oceny wzrokowej.

Po co: zbior podaje koncowke na 88,5% klatek, ale podglad pokazal, ze czesc
znacznikow stoi w pustym miejscu. Zanim uznamy te adnotacje za wzorzec do
sprawdzania wlasnej propagacji ramek, trzeba zmierzyc, jaka czesc jest trafna.

Losowanie warstwowe: N klipow, po M klatek z kazdego. Ziarno stale, wiec
probka jest odtwarzalna.

Uzycie:
    py -3.11 probka_koncowek_do_oceny.py [katalog_zbioru] [--klipow 40] [--na-klip 5]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from ast import literal_eval
from pathlib import Path

import cv2
import numpy as np

CZERWONY = (0, 0, 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", type=Path,
                    default=Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                                 r"\Dataset\Cataract-LMM"))
    ap.add_argument("--klipow", type=int, default=40)
    ap.add_argument("--na-klip", type=int, default=5)
    ap.add_argument("--bok", type=int, default=160, help="bok wycinka w px oryginalu")
    ap.add_argument("--skala", type=int, default=2)
    ap.add_argument("--kol", type=int, default=5, help="kafli w wierszu arkusza")
    ap.add_argument("--na-arkusz", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1606)
    args = ap.parse_args()

    ann = args.dataset / "3_Object_Tracking/annotations"
    out = args.dataset / "_probka_koncowek"
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    zs = sorted(ann.glob("*.zip"))
    wybor = rng.sample(zs, min(args.klipow, len(zs)))

    probki, idx = [], 0
    for z in sorted(wybor):
        try:
            with zipfile.ZipFile(z) as f:
                nz = [n for n in f.namelist() if n.endswith("annotation.csv")][0]
                rows = [r for r in csv.DictReader(io.StringIO(
                    f.read(nz).decode("utf-8", "replace")))
                    if r.get("cystotome_tip", "").strip()]
                if not rows:
                    continue
                for r in rng.sample(rows, min(args.na_klip, len(rows))):
                    p = f"{z.stem}/{r['filename']}"
                    if p not in f.namelist():
                        continue
                    img = cv2.imdecode(np.frombuffer(f.read(p), np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    tx, ty = [int(round(v)) for v in literal_eval(r["cystotome_tip"])]
                    h, w = img.shape[:2]
                    b = args.bok
                    x0 = min(max(0, tx - b // 2), max(0, w - b))
                    y0 = min(max(0, ty - b // 2), max(0, h - b))
                    wyc = img[y0:y0 + b, x0:x0 + b].copy()
                    if wyc.shape[:2] != (b, b):
                        wyc = cv2.copyMakeBorder(wyc, 0, b - wyc.shape[0], 0,
                                                 b - wyc.shape[1], cv2.BORDER_CONSTANT)
                    wyc = cv2.resize(wyc, (b * args.skala, b * args.skala),
                                     interpolation=cv2.INTER_CUBIC)
                    cx, cy = (tx - x0) * args.skala, (ty - y0) * args.skala
                    cv2.drawMarker(wyc, (cx, cy), CZERWONY, cv2.MARKER_CROSS, 26, 2)
                    cv2.circle(wyc, (cx, cy), 16, CZERWONY, 2)
                    idx += 1
                    cv2.rectangle(wyc, (0, 0), (wyc.shape[1], 22), (0, 0, 0), -1)
                    cv2.putText(wyc, f'#{idx:03d}  {z.stem[:12]} kl.{r["frame_number"]}',
                                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    probki.append({"id": idx, "clip": z.stem,
                                   "frame": int(r["frame_number"]),
                                   "tip": [tx, ty], "tile": wyc})
        except Exception as e:
            print("  pominieto", z.name, e)

    print(f"wylosowano probek: {len(probki)}")

    # arkusze
    ark = 0
    for i in range(0, len(probki), args.na_arkusz):
        grupa = probki[i:i + args.na_arkusz]
        kafle = [p["tile"] for p in grupa]
        while len(kafle) % args.kol:
            kafle.append(np.zeros_like(kafle[0]))
        wiersze = [np.hstack(kafle[j:j + args.kol])
                   for j in range(0, len(kafle), args.kol)]
        siatka = np.vstack(wiersze)
        pas = np.zeros((32, siatka.shape[1], 3), np.uint8)
        ark += 1
        cv2.putText(pas, f"ARKUSZ {ark}: probki #{grupa[0]['id']:03d}-#{grupa[-1]['id']:03d}"
                         f"  |  czy krzyzyk jest na CZUBKU narzedzia?",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        f = out / f"arkusz_{ark:02d}.jpg"
        cv2.imwrite(str(f), np.vstack([pas, siatka]))
        print("zapisano:", f.name)

    (out / "probka.json").write_text(json.dumps(
        [{k: v for k, v in p.items() if k != "tile"} for p in probki],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("zapisano: probka.json  (identyfikatory probek, do zapisu werdyktow)")


if __name__ == "__main__":
    main()
