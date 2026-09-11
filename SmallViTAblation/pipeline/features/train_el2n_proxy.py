"""Trening modelu wstepnego (proxy) RF-DETR-Small do liczenia EL2N.

Zasady (AGENT_1 zadanie 2, protocol.yaml el2n.*):
  - trenujemy przez el2n.proxy_epochs epok (BEZ early stopping),
  - WYLACZNIE na losowym podzbiorze klatek split=='train'
    (el2n.proxy_subset_size), nigdy na val/test,
  - jeden proxy na ziarno -> feature_cache/el2n_proxy_<seed>.pth.

Produkcyjnie wymaga pakietu `rfdetr` (leniwy import; nie jest instalowany
przez Agent 1). --fake zapisuje deterministyczny atrapy-checkpoint, zeby
reszta pipeline'u (el2n_from_proxy.py) dala sie przetestowac dymnie.

Uzycie:
    py -3.11 pipeline/features/train_el2n_proxy.py --seed 42 [--fake]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.features.common import (  # noqa: E402
    load_protocol, read_manifest, resolve_under_root, train_pool, utc_now,
    write_cache_manifest)


def build_proxy_coco_dir(records: list[dict], images_root: Path,
                         work_dir: Path) -> Path:
    """COCO dataset for RF-DETR with a train-only plumbing validation split.

    file_name to ABSOLUTNE sciezki obrazow - rfdetr/pycocotools lacza
    os.path.join(root, file_name), a sciezka absolutna wygrywa, wiec
    obrazy NIGDY nie sa kopiowane.

    ``valid`` intentionally mirrors ``train``. It is required by RF-DETR, but
    never selects the proxy: EL2N always uses the fixed final epoch. Therefore
    no val/test operation enters feature construction and no frames are held
    out asymmetrically from the proxy fit.
    """
    images, annotations = [], []
    img_id, ann_id = 1, 1
    for r in records:
        if r.get("split") != "train":
            raise ValueError(f"{r['frame_id']}: proxy accepts only split=train")
        if r.get("label_status") == "unknown":
            raise ValueError(f"{r['frame_id']}: proxy cannot train on an unknown label")
        if not r.get("path"):
            raise ValueError(f"{r['frame_id']}: materialize archive-backed image before proxy training")
        image_path = (images_root / r["path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{r['frame_id']}: missing image {image_path}")
        width, height = int(r["native_width"]), int(r["native_height"])
        images.append({
            "id": img_id,
            "file_name": str(image_path),
            "width": width,
            "height": height,
        })
        row_annotations = r.get("annotations")
        if row_annotations is None:
            row_annotations = ([{"bbox": r["bbox"], "iscrowd": 0}]
                               if r.get("bbox") and r.get("has_instrument") else [])
        for source_ann in row_annotations:
            x, y, w, h = map(float, source_ann["bbox"])
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > width + 1e-3 \
                    or y + h > height + 1e-3:
                raise ValueError(f"{r['frame_id']}: invalid bbox {source_ann['bbox']}")
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [x, y, w, h], "area": w * h,
                "iscrowd": int(source_ann.get("iscrowd", 0)),
            })
            ann_id += 1
        img_id += 1
    coco = {
        "info": {"description": "EL2N proxy subset (train pool only)",
                 "date_created": utc_now()},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "instrument"}],
    }
    for split in ("train", "valid"):
        split_dir = work_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8")
    return work_dir


def train_real(protocol: dict, records: list[dict], images_root: Path,
               out_path: Path, seed: int) -> None:
    from rfdetr import RFDETRSmall  # leniwie - pakiet wymagany produkcyjnie

    el2n = protocol["el2n"]
    model_cfg = protocol["model"]
    epochs = int(el2n["proxy_epochs"])
    np.random.seed(seed)

    work = Path(tempfile.mkdtemp(prefix=f"el2n_proxy_s{seed}_"))
    dataset_dir = build_proxy_coco_dir(records, images_root, work)
    output_dir = work / "rfdetr-checkpoints"
    det = RFDETRSmall(
        resolution=int(el2n.get("proxy_resolution", model_cfg["resolution"])),
    )
    det.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        resolution=int(el2n.get("proxy_resolution", model_cfg["resolution"])),
        epochs=epochs,                       # stalow liczba epok, bez early stop
        batch_size=int(el2n.get("proxy_batch_size", 8)),
        lr=float(protocol["training"]["lr"]),
        lr_encoder=float(protocol["training"]["lr_backbone"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
        warmup_epochs=int(protocol["training"]["warmup_epochs"]),
        grad_accum_steps=int(protocol["training"]["grad_accum"]),
        amp=protocol["training"].get("precision") == "bf16",
        amp_dtype="bf16" if protocol["training"].get("precision") == "bf16" else "fp16",
        seed=seed,
        resume=None,
        early_stopping=False,
        eval_interval=epochs,
        best_model_metric="map",
        run_test=False,
        aug_config={},
        scale_jitter=False,
        multi_scale=False,
    )
    # Fixed final epoch, never a validation-selected checkpoint.
    ckpt = output_dir / "last.pth"
    if not ckpt.is_file():
        raise RuntimeError(f"rfdetr nie zapisal wymaganego checkpointu {ckpt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ckpt, out_path)
    shutil.rmtree(work, ignore_errors=True)
    print(f"Proxy (real) zapisany: {out_path} (epok: {epochs}, seed: {seed})")


def train_fake(out_path: Path, seed: int, epochs: int, n_subset: int) -> None:
    """FIXTURE: deterministyczna atrapa checkpointu - tylko testy dymne."""
    import torch  # leniwie
    rng = np.random.RandomState(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fake_proxy": True,
        "seed": seed,
        "proxy_epochs": epochs,
        "subset_size": n_subset,
        # deterministyczne "wagi" zastepcze, zeby plik mial strukture .pth
        "state": torch.from_numpy(rng.rand(64).astype(np.float32)),
        "created_at": utc_now(),
    }, out_path)
    print(f"[FIXTURE] Proxy (fake) zapisany: {out_path} (seed: {seed})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--images-root", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--fake", action="store_true",
                    help="FIXTURE/test dymny: atrapa proxy zamiast rfdetr")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    paths_cfg = protocol.get("paths", {})
    manifest = args.manifest or resolve_under_root(
        paths_cfg.get("pool_manifest", "pool_manifest.jsonl"))
    images_root = args.images_root or resolve_under_root(
        paths_cfg.get("images_root", "."))
    cache_dir = args.cache_dir or resolve_under_root(
        paths_cfg.get("feature_cache_dir", "feature_cache"))

    pool = train_pool(read_manifest(manifest))
    if not pool:
        sys.exit("Pusta pula train.")
    subset_size = int(protocol["el2n"].get("proxy_subset_size", 2500))
    rng = np.random.RandomState(args.seed)
    n = min(subset_size, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    subset = [pool[i] for i in sorted(idx)]
    print(f"Podzbior proxy: {n}/{len(pool)} klatek train (seed={args.seed}, "
          f"el2n.proxy_subset_size={subset_size})")

    out_path = Path(cache_dir) / f"el2n_proxy_{args.seed}.pth"
    epochs = int(protocol["el2n"]["proxy_epochs"])
    if args.fake:
        train_fake(out_path, args.seed, epochs, n)
    else:
        train_real(protocol, subset, images_root, out_path, args.seed)

    write_cache_manifest(cache_dir, {
        f"el2n_proxy_{args.seed}": {
            "file": out_path.name, "seed": args.seed, "proxy_epochs": epochs,
            "subset_size": n, "fake": bool(args.fake),
            "model": "fake-proxy" if args.fake else protocol["model"]["name"],
        }
    })


if __name__ == "__main__":
    main()
