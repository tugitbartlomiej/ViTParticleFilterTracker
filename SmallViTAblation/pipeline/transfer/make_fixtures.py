#!/usr/bin/env python3
"""Generator syntetycznych fixture dla testow Agent 4 (bez GPU, bez ultralytics).

Tworzy:
  <root>/pool_manifest.jsonl         — train/val/test, jedna klasa, czesc tla
  <root>/images/cataracts/.../*.jpg  — male syntetyczne obrazy (PIL)
  <root>/variants/<run_id>.json      — plik wariantu (kontrakt 4.2)

Uzycie:
    py -3.11 pipeline/transfer/make_fixtures.py --root /tmp/fix --run-id FIXTURE__s42__b20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.transfer import transfer_common as tc  # noqa: E402


def _make_image(path: Path, w: int, h: int, seed: int) -> None:
    from PIL import Image  # lazy
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=95)


def build(root: Path, run_id: str, n_train=20, n_val=6, n_test=4,
          seed=42, img_w=64, img_h=48):
    root = Path(root)
    (root / "variants").mkdir(parents=True, exist_ok=True)

    rows = []
    frame_ids_train = []

    def add_frame(idx, split, has_instr, patient, video):
        fid = f"cataracts/{video}/{idx:07d}"
        rel = f"images/cataracts/{video}/{idx:07d}.jpg"
        _make_image(root / rel, img_w, img_h, seed + idx)
        row = {
            "frame_id": fid, "path": rel, "source_dataset": "cataracts",
            "patient_id": patient, "video_id": video, "frame_index": idx,
            "native_width": img_w, "native_height": img_h, "native_fps": 29.17,
            "split": split,
            "label_origin": "human" if split in ("val", "test") else "propagated",
            "seed_distance_s": 0.0, "has_instrument": has_instr,
        }
        if has_instr:
            # bbox [x,y,w,h] w pikselach, miesci sie w obrazie
            row["bbox"] = [int(img_w * 0.3), int(img_h * 0.3),
                           int(img_w * 0.25), int(img_h * 0.3)]
        rows.append(row)
        return fid

    idx = 0
    for i in range(n_train):
        fid = add_frame(idx, "train", has_instr=(i % 4 != 0),
                        patient="test01", video="test01")
        frame_ids_train.append(fid)
        idx += 1
    for i in range(n_val):
        add_frame(idx, "val", has_instr=(i % 3 != 0), patient="val01", video="val01")
        idx += 1
    for i in range(n_test):
        add_frame(idx, "test", has_instr=True, patient="test09", video="test09")
        idx += 1

    pool_path = root / "pool_manifest.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    variant = {
        "run_id": run_id, "variant": "FIXTURE", "seed": seed,
        "budget": len(frame_ids_train), "frame_ids": frame_ids_train,
        "selection_meta": {"weights": {"dino": 1.0}, "pca_dim": 32,
                           "algorithm": "kmeans", "representative": "centroid"},
    }
    var_path = root / "variants" / f"{run_id}.json"
    tc.write_json(var_path, variant)
    return {"pool_manifest": str(pool_path), "variant_file": str(var_path),
            "data_root": str(root), "n_train": len(frame_ids_train),
            "n_val": n_val, "n_test": n_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run-id", default="FIXTURE__s42__b20")
    args = ap.parse_args()
    info = build(args.root, args.run_id)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
