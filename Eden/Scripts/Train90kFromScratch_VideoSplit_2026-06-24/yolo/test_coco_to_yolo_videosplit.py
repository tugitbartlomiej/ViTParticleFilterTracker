#!/usr/bin/env python3
"""
Test konwersji COCO -> YOLO + video-split (coco_to_yolo_videosplit.py).

Buduje syntetyczny COCO (train=wideo vidA, val=wideo vidB) z PRAWDZIWYM przykladem
bbox z datasetu, uruchamia konwersje i sprawdza:
  - dokladnosc matematyki bbox (COCO px -> YOLO znormalizowane),
  - mapowanie klasy, w tym category_id jako FLOAT (0.0),
  - obraz bez anotacji -> pusty plik .txt (tlo),
  - obraz z 2 anotacjami -> 2 linie,
  - clamping bbox poza obraz -> [0,1],
  - rozdzial train/val po wideo (zero nakladania),
  - fizyczne umieszczenie obrazow (copy) + dataset.yaml (nc=1, names=['tool']).

Uruchom:  py -3.11 test_coco_to_yolo_videosplit.py   (lub python3 ...)
Brak zaleznosci zewnetrznych (nie wymaga pyyaml ani obrazow z poolu).
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import coco_to_yolo_videosplit as conv

TOL = 1e-4
_fail = 0


def check(cond, msg):
    global _fail
    status = "PASS" if cond else "FAIL"
    if not cond:
        _fail += 1
    print(f"  [{status}] {msg}")


def parse_label(path):
    """Zwraca liste krotek (cls, xc, yc, w, h)."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def approx(a, b, tol=TOL):
    return abs(a - b) <= tol


def main():
    work = Path(tempfile.mkdtemp(prefix="yolo_conv_test_"))
    pool = work / "pool"
    pool.mkdir()
    out = work / "yolo_dataset"

    categories = [{"id": 0, "name": "tool", "supercategory": "none"}]

    # --- TRAIN: wideo vidA, 4 obrazy ---
    train_images = [
        {"id": 1, "file_name": "vidA_frame_0000001.jpg", "width": 1920, "height": 1080},
        {"id": 2, "file_name": "vidA_frame_0000002.jpg", "width": 100, "height": 100},
        {"id": 3, "file_name": "vidA_frame_0000003.jpg", "width": 100, "height": 100},
        {"id": 4, "file_name": "vidA_frame_0000004.jpg", "width": 100, "height": 100},
    ]
    train_anns = [
        # obraz 1: prawdziwy przyklad z datasetu [1130,775,179,245] na 1920x1080
        {"id": 101, "image_id": 1, "category_id": 0, "bbox": [1130.0, 775.0, 179.0, 245.0]},
        # obraz 2: DWIE anotacje, category_id jako FLOAT 0.0
        {"id": 102, "image_id": 2, "category_id": 0.0, "bbox": [0, 0, 50, 50]},
        {"id": 103, "image_id": 2, "category_id": 0.0, "bbox": [50, 50, 50, 50]},
        # obraz 3: BRAK anotacji (nie ma wpisu) -> pusty label
        # obraz 4: bbox czesciowo poza obrazem -> clamping
        {"id": 104, "image_id": 4, "category_id": 0, "bbox": [90, 90, 30, 30]},
    ]
    # --- VAL: wideo vidB, 1 obraz ---
    val_images = [
        {"id": 10, "file_name": "vidB_frame_0000001.jpg", "width": 200, "height": 100},
    ]
    val_anns = [
        {"id": 110, "image_id": 10, "category_id": 0, "bbox": [50, 25, 100, 50]},
    ]

    train_coco = work / "annotations_train.json"
    val_coco = work / "annotations_val.json"
    json.dump({"categories": categories, "images": train_images, "annotations": train_anns},
              open(train_coco, "w"))
    json.dump({"categories": categories, "images": val_images, "annotations": val_anns},
              open(val_coco, "w"))

    # fake obrazy w poolu (puste pliki) -> test trybu copy
    for im in train_images + val_images:
        (pool / im["file_name"]).write_bytes(b"")

    print("== Konwersja train ==")
    st = conv.convert_split(train_coco, pool, out, "train", link_mode="copy")
    print("   stats:", st)
    print("== Konwersja val ==")
    sv = conv.convert_split(val_coco, pool, out, "val", link_mode="copy")
    print("   stats:", sv)
    conv.write_dataset_yaml(out, st["names"])

    print("\n== Asercje ==")
    # 1. liczby
    check(st["images"] == 4, "train: 4 obrazy")
    check(st["label_lines"] == 4, f"train: 4 linie etykiet (jest {st['label_lines']})")
    check(st["empty_labels"] == 1, "train: 1 pusty label (obraz bez anotacji)")
    check(st["missing_images"] == 0, "train: 0 brakujacych obrazow")
    check(sv["images"] == 1 and sv["label_lines"] == 1, "val: 1 obraz, 1 linia")

    # 2. matematyka bbox - prawdziwy przyklad
    lab1 = parse_label(out / "labels" / "train" / "vidA_frame_0000001.txt")
    check(len(lab1) == 1, "obraz1: 1 linia")
    cls, xc, yc, w, h = lab1[0]
    check(cls == 0, "obraz1: klasa 0")
    check(approx(xc, (1130 + 179 / 2) / 1920), f"obraz1: xc~=0.635156 (jest {xc:.6f})")
    check(approx(yc, (775 + 245 / 2) / 1080), f"obraz1: yc~=0.831019 (jest {yc:.6f})")
    check(approx(w, 179 / 1920), f"obraz1: w~=0.093229 (jest {w:.6f})")
    check(approx(h, 245 / 1080), f"obraz1: h~=0.226852 (jest {h:.6f})")

    # 3. multi-anotacja + float category_id
    lab2 = parse_label(out / "labels" / "train" / "vidA_frame_0000002.txt")
    check(len(lab2) == 2, "obraz2: 2 linie (multi-ann)")
    check(all(c == 0 for c, *_ in lab2), "obraz2: float category_id 0.0 -> klasa 0")
    check(approx(lab2[0][1], 0.25) and approx(lab2[0][2], 0.25)
          and approx(lab2[0][3], 0.5) and approx(lab2[0][4], 0.5),
          "obraz2: bbox [0,0,50,50]/100 -> (0.25,0.25,0.5,0.5)")

    # 4. pusty label
    e = (out / "labels" / "train" / "vidA_frame_0000003.txt")
    check(e.exists() and e.read_text().strip() == "", "obraz3: pusty label (tlo)")

    # 5. clamping
    lab4 = parse_label(out / "labels" / "train" / "vidA_frame_0000004.txt")
    cls, xc, yc, w, h = lab4[0]
    check(approx(xc, 1.0) and approx(yc, 1.0), "obraz4: xc,yc poza obrazem -> clamp 1.0")
    check(0.0 <= w <= 1.0 and 0.0 <= h <= 1.0, "obraz4: w,h w [0,1]")

    # 6. val bbox
    labv = parse_label(out / "labels" / "val" / "vidB_frame_0000001.txt")
    cls, xc, yc, w, h = labv[0]
    check(approx(xc, 0.5) and approx(yc, 0.5) and approx(w, 0.5) and approx(h, 0.5),
          "val: bbox [50,25,100,50] na 200x100 -> (0.5,0.5,0.5,0.5)")

    # 7. rozdzial train/val po wideo (zero nakladania)
    train_imgs = {p.stem for p in (out / "images" / "train").glob("*.jpg")}
    val_imgs = {p.stem for p in (out / "images" / "val").glob("*.jpg")}
    check(all(s.startswith("vidA") for s in train_imgs), "train images: tylko vidA")
    check(all(s.startswith("vidB") for s in val_imgs), "val images: tylko vidB")
    check(len(train_imgs & val_imgs) == 0, "zero nakladania train/val")
    check(len(train_imgs) == 4 and len(val_imgs) == 1, "obrazy fizycznie umieszczone (copy)")

    # 8. wszystkie wartosci w [0,1] dla kazdej etykiety
    allvals_ok = True
    for txt in (out / "labels").rglob("*.txt"):
        for _, *coords in parse_label(txt):
            if not all(0.0 <= v <= 1.0 for v in coords):
                allvals_ok = False
    check(allvals_ok, "wszystkie xc,yc,w,h w [0,1]")

    # 9. dataset.yaml
    ytext = (out / "dataset.yaml").read_text()
    check("images/train" in ytext and "images/val" in ytext, "yaml: train/val sciezki")
    check("nc: 1" in ytext, "yaml: nc=1")
    check("tool" in ytext, "yaml: nazwa klasy 'tool'")

    shutil.rmtree(work, ignore_errors=True)

    print("\n" + ("=" * 50))
    if _fail == 0:
        print("WYNIK: WSZYSTKIE TESTY PRZESZLY (0 bledow)")
        return 0
    print(f"WYNIK: {_fail} BLEDOW")
    return 1


if __name__ == "__main__":
    sys.exit(main())
