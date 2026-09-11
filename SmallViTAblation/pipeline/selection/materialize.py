"""Materializacja podzbioru: variants/<run_id>.json -> annotations_train.json (COCO).

AGENT_1 zadanie 6. OBRAZY NIGDY NIE SA KOPIOWANE - pole file_name wskazuje na
wspolny katalog obrazow (sciezka relatywna z pool_manifest.jsonl, albo
absolutna po --absolute-paths). Klatki bez narzedzia wchodza do images bez
anotacji (poprawne COCO - tla negatywne).

Uzycie:
    py -3.11 pipeline/selection/materialize.py --variant-file variants/X.json \
        [--manifest pool_manifest.jsonl] [--out DIR/annotations_train.json]
        [--absolute-paths] [--validate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.features.common import (  # noqa: E402
    load_protocol, read_manifest, resolve_under_root, utc_now)

CATEGORY = {"id": 1, "name": "instrument", "supercategory": "tool"}


def build_coco(variant: dict, manifest_path: Path, absolute: bool) -> dict:
    records = read_manifest(manifest_path)
    by_id = {r["frame_id"]: r for r in records}
    frame_ids = list(variant.get("frame_ids", variant.get("selected_frame_ids", [])))
    if not frame_ids:
        raise ValueError(f"{variant.get('run_id')}: pusty wariant")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"{variant.get('run_id')}: zduplikowane frame_id")
    budget = variant.get("budget")
    if isinstance(budget, int) and len(frame_ids) != budget:
        raise ValueError(f"{variant.get('run_id')}: len(frame_ids) != budget")
    missing = [f for f in frame_ids if f not in by_id]
    if missing:
        raise ValueError(f"{variant['run_id']}: {len(missing)} frame_id spoza "
                         f"manifestu, np. {missing[:3]}")
    bad_split = [f for f in frame_ids if by_id[f]["split"] != "train"]
    if bad_split:
        raise ValueError(f"{variant['run_id']}: {len(bad_split)} klatek "
                         f"spoza split=train, np. {bad_split[:3]}")

    images, annotations = [], []
    ann_id = 1
    for img_id, fid in enumerate(frame_ids, start=1):
        r = by_id[fid]
        if r.get("label_status") == "unknown":
            raise ValueError(f"{fid}: label_status=unknown; najpierw oznacz lub potwierdź negatyw")
        if not r.get("path"):
            raise ValueError(f"{fid}: obraz w archiwum wymaga wcześniejszego eksportu do path")
        file_name = r["path"]
        if absolute:
            source = Path(r["path"])
            file_name = str((source if source.is_absolute()
                             else manifest_path.parent / source).resolve())
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": int(r["native_width"]),
            "height": int(r["native_height"]),
        })
        row_annotations = r.get("annotations")
        if row_annotations is None:
            row_annotations = ([{"bbox": r["bbox"], "iscrowd": 0}]
                               if r.get("has_instrument") and r.get("bbox") else [])
        for source_ann in row_annotations:
            x, y, w, h = [float(v) for v in source_ann["bbox"]]
            if min(x, y) < 0 or min(w, h) <= 0:
                raise ValueError(f"{fid}: niepoprawny bbox {source_ann['bbox']}")
            if x + w > float(r["native_width"]) + 1e-3 or y + h > float(r["native_height"]) + 1e-3:
                raise ValueError(f"{fid}: bbox poza obrazem {source_ann['bbox']}")
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": CATEGORY["id"],
                "bbox": [x, y, w, h], "area": w * h,
                "iscrowd": int(source_ann.get("iscrowd", 0)),
            })
            ann_id += 1
    return {
        "info": {
            "description": f"Subzet treningowy {variant['run_id']} "
                           f"(SmallViTAblation, Agent 1 materialize)",
            "version": 1, "date_created": utc_now(),
            "run_id": variant["run_id"], "seed": variant.get("seed"),
            "budget": variant.get("budget"),
        },
        "licenses": [],
        "categories": [CATEGORY],
        "images": images,
        "annotations": annotations,
    }


def validate(coco_path: Path) -> None:
    """Walidacja strukturalna: pycocotools jesli zainstalowany, inaczej pola."""
    try:
        from pycocotools.coco import COCO
        coco = COCO(str(coco_path))
        assert len(coco.getImgIds()) == len(coco.dataset["images"])
        assert len(coco.getCatIds()) == 1
        print(f"[validate] pycocotools OK: {len(coco.getImgIds())} obrazow, "
              f"{len(coco.getAnnIds())} anotacji")
    except ImportError:
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        for key in ("images", "annotations", "categories"):
            assert key in data and isinstance(data[key], list), f"brak {key}"
        img_ids = {im["id"] for im in data["images"]}
        for im in data["images"]:
            for f in ("id", "file_name", "width", "height"):
                assert f in im, f"image bez {f}"
        for a in data["annotations"]:
            for f in ("id", "image_id", "category_id", "bbox", "area"):
                assert f in a, f"anotacja bez {f}"
            assert a["image_id"] in img_ids, "anotacja wskazuje nieistniejacy obraz"
        print(f"[validate] struktura OK (bez pycocotools): "
              f"{len(data['images'])} obrazow, {len(data['annotations'])} anotacji")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant-file", type=Path, required=True,
                    help="variants/<run_id>.json")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="domyslnie variants/<run_id>/annotations_train.json")
    ap.add_argument("--absolute-paths", action="store_true")
    ap.add_argument("--validate", action="store_true", default=True)
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    args = ap.parse_args()

    protocol = load_protocol()
    paths_cfg = protocol.get("paths", {})
    manifest = args.manifest or resolve_under_root(
        paths_cfg.get("pool_manifest", "pool_manifest.jsonl"))

    variant = json.loads(args.variant_file.read_text(encoding="utf-8"))
    run_id = variant["run_id"]
    out = args.out or (args.variant_file.parent / run_id
                       / "annotations_train.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    coco = build_coco(variant, manifest, args.absolute_paths)
    out.write_text(json.dumps(coco), encoding="utf-8")
    print(f"{run_id}: {len(coco['images'])} obrazow, "
          f"{len(coco['annotations'])} anotacji -> {out}")
    if args.validate:
        validate(out)


if __name__ == "__main__":
    main()
