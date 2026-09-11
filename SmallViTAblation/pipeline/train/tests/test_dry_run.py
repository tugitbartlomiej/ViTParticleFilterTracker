#!/usr/bin/env python3
"""
Test 5 (bez GPU): train_rfdetr.py --dry-run na fixture.

Fixture: maly variants/<run_id>.json + syntetyczne obrazy + annotations_val.json
+ pool_manifest.jsonl. Dry-run przechodzi cala inicjalizacje, waliduje pliki,
liczy sha256, buduje i ZAPISUJE run_manifest.json (bez gpu/torch jesli brak,
reszta kompletna wg agents/README.md 4.4).

Drugi przypadek: variant-file z polem annotations_train.

Uruchomienie:  py -3.11 pipeline/train/tests/test_dry_run.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "train_rfdetr.py"
PROTOCOL = HERE.parents[2] / "configs" / "protocol.yaml"

REQUIRED_FIELDS = ["run_id", "git_commit", "protocol_sha256", "variant_file_sha256",
                   "started_at", "finished_at", "hostname", "python",
                   "epochs_completed", "resolution", "seed"]
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def build_fixture(root: Path) -> Path:
    """Syntetyczny data-root: 4 obrazy train + 2 val, pool_manifest, annotations."""
    data = root / "data"
    (data / "images" / "cataracts" / "test01").mkdir(parents=True)
    (data / "images" / "val").mkdir(parents=True)
    frame_ids = []
    pool_lines = []
    for i in range(1, 5):
        fid = f"cataracts/test01/{i:07d}"
        frame_ids.append(fid)
        img = data / "images" / f"{fid}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0SYNTHETIC_JPEG")  # dry-run nie dekoduje
        pool_lines.append({
            "frame_id": fid, "path": f"images/{fid}.jpg",
            "source_dataset": "cataracts", "patient_id": "test01",
            "video_id": "test01", "frame_index": i,
            "native_width": 64, "native_height": 48, "native_fps": 29.17,
            "split": "train", "label_origin": "human", "seed_distance_s": 0.0,
            "bbox": [10.0, 10.0, 20.0, 15.0], "has_instrument": True,
        })
    with open(data / "pool_manifest.jsonl", "w", encoding="utf-8") as f:
        for rec in pool_lines:
            f.write(json.dumps(rec) + "\n")

    val_images = []
    for j in range(2):
        p = data / "images" / "val" / f"val{j}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0SYNTHETIC_JPEG")
        val_images.append({"id": j + 1, "file_name": f"images/val/val{j}.jpg",
                           "width": 64, "height": 48})
    val = {"images": val_images,
           "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                            "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0}],
           "categories": [{"id": 1, "name": "instrument_tip"}]}
    (data / "annotations_val.json").write_text(json.dumps(val), encoding="utf-8")

    run_id = "TEST-DESF__s42__b4"
    variants = data / "variants"
    variants.mkdir(parents=True)
    variant = {"run_id": run_id, "variant": "TEST-DESF", "seed": 42, "budget": 4,
               "frame_ids": frame_ids,
               "selection_meta": {"weights": {"dino": 1.0}, "pca_dim": 32,
                                  "algorithm": "kmeans", "representative": "centroid"}}
    (variants / f"{run_id}.json").write_text(json.dumps(variant), encoding="utf-8")
    return data


def run_dry(data: Path, root: Path, run_id: str, role: str = "smoke",
            variant_file: Path | None = None) -> dict:
    out = root / f"out_{run_id}"
    vf = variant_file or (data / "variants" / f"{run_id}.json")
    proc = subprocess.run(
        [sys.executable, str(TRAIN), "--run-id", run_id, "--variant-file", str(vf),
         "--protocol", str(PROTOCOL), "--data-root", str(data), "--out", str(out),
         "--role", role, "--dry-run"],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"exit={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    mp = out / run_id / "run_manifest.json"
    assert mp.is_file(), f"brak run_manifest.json w {out / run_id}"
    return json.loads(mp.read_text(encoding="utf-8"))


def check_manifest(m: dict, run_id: str, seed: int, resolution: int) -> None:
    for field in REQUIRED_FIELDS:
        assert field in m, f"manifest bez pola {field}: {sorted(m)}"
    assert m["run_id"] == run_id
    assert SHA_RE.match(m["protocol_sha256"]), m["protocol_sha256"]
    assert SHA_RE.match(m["variant_file_sha256"])
    assert m["seed"] == seed
    assert m["resolution"] == resolution
    assert m["epochs_completed"] == 0  # dry-run nie trenuje
    assert m.get("dry_run") is True
    assert m["hostname"] and m["python"]
    if "library_versions" in m:
        assert isinstance(m["library_versions"], dict)
    # gpu/torch moga byc nieobecne bez GPU — ale jesli sa, maja sens
    assert m["git_commit"] is None or GIT_SHA_RE.fullmatch(m["git_commit"])


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="dryrun_"))
    failures = 0
    try:
        data = build_fixture(root)

        # przypadek 1: frame_ids -> generowane annotations_train
        run_id = "TEST-DESF__s42__b4"
        m = run_dry(data, root, run_id)
        check_manifest(m, run_id, seed=42, resolution=384)  # smoke: nano @384
        ds = m["dataset"]
        assert ds["source"] == "generated_from_frame_ids", ds
        assert ds["n_images"] == 4 and ds["n_annotations"] == 4, ds
        assert ds["n_val_images"] == 2, ds
        gen = json.loads(Path(ds["train_annotations"]).read_text(encoding="utf-8"))
        assert Path(gen["images"][0]["file_name"]).is_absolute()
        print("przypadek 1 (frame_ids): PASS")

        # przypadek 2: variant z polem annotations_train
        run_id2 = "TEST-ANN__s7__b2"
        ann = {"images": gen["images"][:2], "annotations": gen["annotations"][:2],
               "categories": gen["categories"]}
        (data / "annotations_train_custom.json").write_text(json.dumps(ann),
                                                            encoding="utf-8")
        v2 = {"run_id": run_id2, "variant": "TEST-ANN", "seed": 7,
              "annotations_train": "annotations_train_custom.json"}
        (data / "variants" / f"{run_id2}.json").write_text(json.dumps(v2),
                                                           encoding="utf-8")
        m2 = run_dry(data, root, run_id2, variant_file=data / "variants" / f"{run_id2}.json")
        check_manifest(m2, run_id2, seed=7, resolution=384)
        assert m2["dataset"]["source"].endswith("annotations_train_custom.json"), m2["dataset"]
        assert m2["dataset"]["n_images"] == 2, m2["dataset"]
        print("przypadek 2 (annotations_train): PASS")
    except Exception as e:
        failures += 1
        import traceback
        traceback.print_exc()
        print(f"FAIL: {e}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("WYNIK:", "PASS" if failures == 0 else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
