#!/usr/bin/env python3
"""Losuje probke RAMEK narzedzia i sklada je w arkusze do oceny wzrokowej.

Po co: koncowki sprawdzilismy (91,8% trafien). Ramki to druga polowa adnotacji
i drugi zrodlo bledu - moga byc za luzne, przesuniete albo obejmowac cien.

Dwa zrodla, bo maja inny format i inna jakosc:
  seg  - 2_Instance_Segmentation, COCO, 10 klas narzedzi, klatki rozpakowane
  trk  - 3_Object_Tracking, cystotome_bbox z CSV w archiwach zip

Wycinek to bbox powiekszony o margines, zeby bylo widac, czy ramka nie ucina
narzedzia i czy nie ma wokol pustego zapasu.

Uzycie:
    py -3.11 probka_ramek_do_oceny.py --zrodlo seg
    py -3.11 probka_ramek_do_oceny.py --zrodlo trk
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
import zipfile
from ast import literal_eval
from pathlib import Path

import cv2
import numpy as np

ZIELONY = (0, 220, 60)
BOK = 260          # bok kafla w arkuszu


def kafel(img, x, y, w, h, podpis_txt):
    """Wycinek wokol ramki z marginesem, ramka narysowana, przeskalowany do BOK."""
    H, W = img.shape[:2]
    m = int(max(w, h) * 0.45) + 8
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    wyc = img[y0:y1, x0:x1].copy()
    if wyc.size == 0:
        return None
    cv2.rectangle(wyc, (x - x0, y - y0), (x + w - x0, y + h - y0), ZIELONY, 2)
    s = BOK / max(wyc.shape[0], wyc.shape[1])
    wyc = cv2.resize(wyc, (max(1, int(wyc.shape[1] * s)), max(1, int(wyc.shape[0] * s))),
                     interpolation=cv2.INTER_CUBIC)
    plo = np.zeros((BOK, BOK, 3), np.uint8)
    oy, ox = (BOK - wyc.shape[0]) // 2, (BOK - wyc.shape[1]) // 2
    plo[oy:oy + wyc.shape[0], ox:ox + wyc.shape[1]] = wyc
    cv2.rectangle(plo, (0, 0), (BOK, 20), (0, 0, 0), -1)
    cv2.putText(plo, podpis_txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return plo


def z_segmentacji(D: Path, rng, ile: int):
    coco = D / "2_Instance_Segmentation/annotations/coco_json/_annotations.coco.json"
    root = D / "2_Instance_Segmentation/annotations/coco_json/images/images/images"
    d = json.loads(coco.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    narz = [a for a in d["annotations"]
            if cats[a["category_id"]] not in ("Cornea", "Pupil", "my-dataset-main")]
    probki, idx = [], 0
    for a in rng.sample(narz, min(ile, len(narz))):
        im = imgs[a["image_id"]]
        p = root / im["file_name"]
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        x, y, w, h = [int(round(v)) for v in a["bbox"]]
        if w < 2 or h < 2:
            continue
        idx += 1
        k = kafel(img, x, y, w, h,
                  f'#{idx:03d} {cats[a["category_id"]][:14]} {w}x{h}')
        if k is None:
            continue
        probki.append({"id": idx, "zrodlo": "seg", "obraz": im["file_name"],
                       "klasa": cats[a["category_id"]], "bbox": [x, y, w, h],
                       "tile": k})
    return probki


def ze_sledzenia(D: Path, rng, klipow: int, na_klip: int):
    ann = D / "3_Object_Tracking/annotations"
    zs = sorted(ann.glob("*.zip"))
    wybor = sorted(rng.sample(zs, min(klipow, len(zs))))
    probki, idx = [], 0
    for z in wybor:
        try:
            with zipfile.ZipFile(z) as f:
                nz = [n for n in f.namelist() if n.endswith("annotation.csv")][0]
                rows = [r for r in csv.DictReader(io.StringIO(
                    f.read(nz).decode("utf-8", "replace")))
                    if r.get("cystotome_bbox", "").strip()]
                if not rows:
                    continue
                for r in rng.sample(rows, min(na_klip, len(rows))):
                    p = f"{z.stem}/{r['filename']}"
                    if p not in f.namelist():
                        continue
                    img = cv2.imdecode(np.frombuffer(f.read(p), np.uint8),
                                       cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    x, y, w, h = [int(round(v)) for v in
                                  literal_eval(r["cystotome_bbox"])]
                    if w < 2 or h < 2:
                        continue
                    idx += 1
                    k = kafel(img, x, y, w, h,
                              f'#{idx:03d} {z.stem[:12]} kl.{r["frame_number"]}')
                    if k is None:
                        continue
                    probki.append({"id": idx, "zrodlo": "trk", "clip": z.stem,
                                   "frame": int(r["frame_number"]),
                                   "bbox": [x, y, w, h], "tile": k})
        except Exception as e:
            print("  pominieto", z.name, e)
    return probki


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zrodlo", choices=["seg", "trk"], required=True)
    ap.add_argument("dataset", nargs="?", type=Path,
                    default=Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker"
                                 r"\Dataset\Cataract-LMM"))
    ap.add_argument("--ile", type=int, default=100, help="dla seg: ile ramek")
    ap.add_argument("--klipow", type=int, default=20, help="dla trk")
    ap.add_argument("--na-klip", type=int, default=5, help="dla trk")
    ap.add_argument("--kol", type=int, default=5)
    ap.add_argument("--na-arkusz", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1606)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = args.dataset / f"_probka_ramek_{args.zrodlo}"
    out.mkdir(parents=True, exist_ok=True)

    if args.zrodlo == "seg":
        probki = z_segmentacji(args.dataset, rng, args.ile)
        naglowek = "czy RAMKA ciasno obejmuje narzedzie?"
    else:
        probki = ze_sledzenia(args.dataset, rng, args.klipow, args.na_klip)
        naglowek = "czy RAMKA ciasno obejmuje cystotom?"

    print(f"wylosowano probek: {len(probki)}")
    if not probki:
        return

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
        cv2.putText(pas, f"ARKUSZ {ark} [{args.zrodlo}]: probki "
                         f"#{grupa[0]['id']:03d}-#{grupa[-1]['id']:03d}  |  {naglowek}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        f = out / f"arkusz_{ark:02d}.jpg"
        cv2.imwrite(str(f), np.vstack([pas, siatka]))
        print("zapisano:", f.name)

    (out / "probka.json").write_text(json.dumps(
        [{k: v for k, v in p.items() if k != "tile"} for p in probki],
        ensure_ascii=False, indent=1), encoding="utf-8")

    if args.zrodlo == "seg":
        c = collections.Counter(p["klasa"] for p in probki)
        print("\nklasy w probce:")
        for k, v in c.most_common():
            print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
