from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation")
INDEX = ROOT / "outputs" / "vit_dino_faiss_35000"
RETRIEVAL = ROOT / "outputs" / "latent_cel_do_obrazow" / "sd_k8000_m0011_dino_train_retrieval" / "WYNIK.json"
REFERENCE = ROOT / "outputs" / "latent_cel_do_obrazow" / "sd_k8000_m0011_interpolacja" / "referencja_m0011.png"
OUT = ROOT / "outputs" / "vit_k8000_podmiana_1z256_dino_m0011_eden_20261008" / "input_pack"
SEED = 20261008


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    images = OUT / "images"
    images.mkdir(parents=True)

    manifest = [json.loads(x) for x in (INDEX / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {r["id"]: r for r in manifest}
    retrieval = json.loads(RETRIEVAL.read_text(encoding="utf-8"))

    step_to_id = []
    unique_ids = []
    for rec in retrieval["records"]:
        if rec["control"]:
            continue
        image_id = rec["matches"][0]["id"]
        step_to_id.append({"query": rec["query"], "image_id": image_id, "cosine": rec["matches"][0]["cosine"]})
        if image_id not in unique_ids:
            unique_ids.append(image_id)
    if len(step_to_id) != 11 or len(unique_ids) != 8:
        raise RuntimeError((len(step_to_id), len(unique_ids)))

    candidate_films = {by_id[i]["film"] for i in unique_ids}
    films = sorted({r["film"] for r in manifest if r["film"] not in candidate_films})
    rng = np.random.default_rng(SEED)
    rng.shuffle(films)
    cut = int(round(0.8 * len(films)))
    train_films, guard_films = set(films[:cut]), set(films[cut:])
    train_rows = [r for r in manifest if r["film"] in train_films and r["id"] not in unique_ids]
    guard_rows = [r for r in manifest if r["film"] in guard_films]
    if len(train_rows) < 256 or len(guard_rows) < 512:
        raise RuntimeError("Za mało danych do kontrolowanego podziału")
    common_idx = rng.choice(len(train_rows), 256, replace=False)
    common_plus_control = [train_rows[int(i)] for i in common_idx]
    common, control = common_plus_control[:255], common_plus_control[255]
    guard_idx = rng.choice(len(guard_rows), 512, replace=False)
    guard = [guard_rows[int(i)] for i in guard_idx]

    selected = []
    seen = set()
    for row in common + [control] + [by_id[i] for i in unique_ids] + guard:
        if row["id"] not in seen:
            selected.append(row)
            seen.add(row["id"])

    zip_cache: dict[str, zipfile.ZipFile] = {}
    packed_rows = []
    try:
        for n, row in enumerate(selected):
            archive = row["archive"]
            zf = zip_cache.setdefault(archive, zipfile.ZipFile(archive, "r"))
            raw = zf.read(row["member"])
            suffix = Path(row["member"]).suffix.lower() or ".jpg"
            filename = f"img_{n:04d}{suffix}"
            (images / filename).write_bytes(raw)
            with Image.open(images / filename) as im:
                width, height = im.size
            if (width, height) != (int(row["width"]), int(row["height"])):
                raise RuntimeError(f"Rozmiar obrazu nie zgadza się dla {row['id']}")
            packed_rows.append(
                {
                    "id": row["id"],
                    "film": row["film"],
                    "frame": int(row["frame"]),
                    "file": f"images/{filename}",
                    "width": width,
                    "height": height,
                    "gt_xywh": [g["bbox_xywh"] for g in row["gt"]],
                    "categories": [g["category"] for g in row["gt"]],
                    "source_archive": archive,
                    "source_member": row["member"],
                    "source_image_sha256": row["image_sha256"],
                    "packed_sha256": sha(images / filename),
                }
            )
    finally:
        for zf in zip_cache.values():
            zf.close()

    row_by_id = {r["id"]: r for r in packed_rows}
    pool = {
        "seed": SEED,
        "common_255": [r["id"] for r in common],
        "control_slot": control["id"],
        "interventions": unique_ids,
        "step_mapping": step_to_id,
        "guard_512": [r["id"] for r in guard],
        "train_films": sorted(train_films),
        "guard_films": sorted(guard_films),
        "film_disjoint_common_guard": not bool(train_films & guard_films),
        "candidate_films_excluded_from_guard": not bool(candidate_films & guard_films),
    }
    (OUT / "records.json").write_text(json.dumps(packed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copy2(REFERENCE, OUT / "probe_m0011.png")
    probe = {
        "id": "m0011",
        "file": "probe_m0011.png",
        "gt_xyxy_480x320": [[274.6666564941406, 183.11111450195312, 480.0, 213.3333282470703]],
        "source_size": list(Image.open(REFERENCE).size),
        "packed_sha256": sha(OUT / "probe_m0011.png"),
    }
    (OUT / "probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "records": len(packed_rows),
        "common": len(common),
        "control": control["id"],
        "interventions_unique": len(unique_ids),
        "steps": len(step_to_id),
        "guard": len(guard),
        "all_gt_present": all(bool(r["gt_xywh"]) for r in packed_rows),
        "pool_sha256": sha(OUT / "pool.json"),
        "records_sha256": sha(OUT / "records.json"),
    }
    (OUT / "PREPARED.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
