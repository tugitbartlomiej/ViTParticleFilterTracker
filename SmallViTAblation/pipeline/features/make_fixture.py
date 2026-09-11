"""FIXTURE - generator syntetycznych danych WYLACZNIE do testow dymnych.

Nie jest to czesc pipeline'u produkcyjnego. Produkuje maly odpowiednik
kontraktu `pool_manifest.jsonl` (README agents sekcja 4.1) plus syntetyczne
obrazy JPEG 128x128, tak aby build_cache / select_subset / baselines /
materialize daly sie przetestowac bez prawdziwych danych i bez GPU.

Tworzy tez fixture queue.jsonl z dwoma biegami macierzowymi (M-DESF, M-D)
i czterema baseline'ami - do testow dymnych Agent 1.

Uzycie:
    py -3.11 pipeline/features/make_fixture.py [--out-dir DIR] [--n 300]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

FIXTURE_TAG = "FIXTURE - dane syntetyczne, tylko testy dymne Agent 1"

# pacjenci i ich splity (split PRZYPISANY PO PACJENCIE, jak w protokole)
PATIENT_SPLITS = {
    "p01": "train", "p02": "train", "p03": "train",
    "p04": "train", "p05": "train", "p06": "train",
    "p07": "val",
    "p08": "test",
}


def make_image(rng: np.random.RandomState, size: int = 128) -> Image.Image:
    """Deterministyczny syntetyczny obraz: tlo + 'narzedzie' + szum."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    base = rng.randint(20, 90, size=3)
    img[:, :] = base
    img += rng.randint(0, 25, size=(size, size, 3), dtype=np.uint8) // 2
    # "koncowka narzedzia": jasny prostokat pod losowym katem
    x0, y0 = rng.randint(10, size - 40, size=2)
    w, h = rng.randint(8, 30), rng.randint(4, 12)
    img[y0:y0 + h, x0:x0 + w] = np.clip(
        img[y0:y0 + h, x0:x0 + w].astype(int) + rng.randint(80, 170), 0, 255
    ).astype(np.uint8)
    # troche struktur wysokoczestotliwosciowych (rozna ostrosc)
    if rng.rand() < 0.5:
        img = np.asarray(Image.fromarray(img).filter(
            ImageFilter.FIND_EDGES)).astype(np.uint8) // 2 + img // 2
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def build(out_dir: Path, n_frames: int, image_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / "images"

    patients = list(PATIENT_SPLITS)
    # po rowno klatek bazowych na pacjenta
    per_patient = max(1, n_frames // len(patients))
    records: list[dict] = []
    idx = 0
    for pat in patients:
        split = PATIENT_SPLITS[pat]
        for k in range(per_patient):
            frame_index = 100 + k * 7 + (idx % 5)
            base_id = f"cataracts/{pat}/{frame_index:07d}"
            # augmentacje TYLKO w puli train (wzorzec ..._aug_N)
            n_aug = 0
            if split == "train" and (k % 3 == 0):
                n_aug = 2  # _aug_1, _aug_2
            variants = [None] + [f"_aug_{a}" for a in range(1, n_aug + 1)]
            for suf in variants:
                fid = base_id + (suf or "")
                rel = f"images/{fid}.jpg"
                rng = np.random.RandomState(zlib.crc32(fid.encode()) % (2 ** 31))
                img_path = out_dir / rel
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img = make_image(np.random.RandomState(idx * 1000 + k),
                                 image_size)
                img.save(img_path, quality=90)
                has_instr = bool(rng.rand() < 0.75)
                rec = {
                    "frame_id": fid,
                    "path": rel,
                    "source_dataset": "cataracts",
                    "patient_id": pat,
                    "video_id": f"{pat}_video1",
                    "frame_index": frame_index,
                    "native_width": image_size,
                    "native_height": image_size,
                    "native_fps": 29.17,
                    "split": split,
                    "label_origin": "human" if split in ("val", "test")
                    else ("human" if rng.rand() < 0.5 else "propagated"),
                    "seed_distance_s": 0.0,
                    "has_instrument": has_instr,
                }
                if has_instr:
                    x = int(rng.randint(5, image_size - 40))
                    y = int(rng.randint(5, image_size - 25))
                    w = int(rng.randint(8, min(35, image_size - x - 1)))
                    h = int(rng.randint(6, min(20, image_size - y - 1)))
                    rec["bbox"] = [x, y, w, h]
                records.append(rec)
                idx += 1

    manifest_path = out_dir / "pool_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # fixture queue.jsonl do testow dymnych (budzet 50)
    base_weights = {"dino": 0.35, "el2n": 0.30, "sam": 0.20, "fourier": 0.15}

    def renorm(groups: list[str]) -> dict[str, float]:
        key = {"D": "dino", "E": "el2n", "S": "sam", "F": "fourier"}
        active = {key[g]: base_weights[key[g]] for g in groups}
        total = sum(active.values())
        out = {name: 0.0 for name in base_weights}
        for name, w in active.items():
            out[name] = round(w / total, 6)
        return out

    runs = [
        {"run_id": "M-DESF__s42__b50", "tier": "FIXTURE", "kind": "matrix",
         "variant": "M-DESF", "groups": ["D", "E", "S", "F"],
         "weights": renorm(["D", "E", "S", "F"]), "seed": 42, "budget": 50,
         "arch": "rf-detr-small", "note": FIXTURE_TAG},
        {"run_id": "M-D__s42__b50", "tier": "FIXTURE", "kind": "matrix",
         "variant": "M-D", "groups": ["D"], "weights": renorm(["D"]),
         "seed": 42, "budget": 50, "arch": "rf-detr-small", "note": FIXTURE_TAG},
    ]
    for bid, budget in (("B-RANDAUG", 50), ("B-RANDSRC", 50),
                        ("B-UNIFORM", 48), ("B-FULLPOOL", "full_train_pool")):
        runs.append({
            "run_id": f"{bid}__s42__b{budget}", "tier": "FIXTURE",
            "kind": "baseline", "variant": bid, "groups": [], "weights": {},
            "seed": 42, "budget": budget, "arch": "rf-detr-small",
            "note": FIXTURE_TAG})

    queue_path = out_dir / "queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_train = sum(1 for r in records if r["split"] == "train")
    print(f"[FIXTURE] {manifest_path}: {len(records)} klatek "
          f"(train={n_train}, val/test={len(records) - n_train})")
    print(f"[FIXTURE] obrazy: {images_root}")
    print(f"[FIXTURE] kolejka: {queue_path} ({len(runs)} biegow)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "fixture_data")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()
    print(f"[FIXTURE] {FIXTURE_TAG}")
    build(args.out_dir, args.n, args.image_size)


if __name__ == "__main__":
    main()
