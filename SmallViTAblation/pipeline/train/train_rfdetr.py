#!/usr/bin/env python3
"""
Trening JEDNEGO biegu RF-DETR zgodnie z configs/protocol.yaml (Agent 2).

Wymagania twarde (agents/AGENT_2_TRENING_I_RUNNER.md):
  1. stala liczba epok z protocol.yaml -> training.epochs, ZERO early stopping,
  2. start zawsze z wag COCO (pretrained) - nigdy z checkpointu innego biegu,
  3. zero augmentacji w treningu (training.train_time_augmentation: none),
  4. dwa checkpointi: last.pth i best.pth (kryterium:
     checkpoint_selection.best_val_criterion = map_50_95_val), walidacja
     mAP@0.5:0.95 na zbiorze val z --data-root,
  5. seed steruje torch / numpy / random / cuda,
  6. run_manifest.json wg agents/README.md sekcja 4.4,
  7. po biegu zostaja tylko last.pth, best.pth, run_manifest.json i logi.

Skrypt NIE liczy metryk koncowych i NIE dotyka zbioru zewnetrznego.

Konwencja wejscia:
  --variant-file  variants/<run_id>.json (agents/README.md 4.2); jesli ma pole
                  "annotations_train" (sciezka wzgl. --data-root lub absolutna),
                  uzywamy go; inaczej generujemy COCO z "frame_ids" na podstawie
                  <data-root>/pool_manifest.jsonl.
  --data-root     zawiera images/, annotations_val.json, pool_manifest.jsonl,
                  (opcjonalnie) variants/.

Uzycie:
  py -3.11 train_rfdetr.py --run-id M-DESF__s42__b5000 \
      --variant-file variants/M-DESF__s42__b5000.json \
      --protocol ../configs/protocol.yaml --data-root /raid/data --out out
  py -3.11 train_rfdetr.py ... --dry-run     # bez treningu, zapisuje manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("Brak pyyaml. Zainstaluj: py -3.11 -m pip install pyyaml")

MODEL_CLASS_MAP = {
    "rf-detr-nano": "RFDETRNano",
    "rf-detr-small": "RFDETRSmall",
    "rf-detr-medium": "RFDETRMedium",
    "rf-detr-large": "RFDETRLarge",
    "rf-detr-base": "RFDETRBase",
    "rf-detr": "RFDETRBase",
}


# --------------------------------------------------------------------------- #
# narzedzia
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def library_versions() -> dict[str, str]:
    """freeze: wszystkie zainstalowane dystrybucje (importlib.metadata)."""
    versions: dict[str, str] = {}
    try:
        from importlib import metadata
        for dist in metadata.distributions():
            name = dist.metadata.get("Name")
            ver = dist.version
            if name:
                versions[name] = ver
    except Exception:
        pass
    return dict(sorted(versions.items()))


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        line = out.stdout.strip().splitlines()
        return line[0].strip() if line else None
    except Exception:
        return None


def torch_version() -> str | None:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def set_seed(seed: int) -> None:
    """Wymaganie 5: seed steruje wszystkim."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# dane: zbior treningowy z wariantu
# --------------------------------------------------------------------------- #
def load_pool_manifest(data_root: Path) -> dict[str, dict]:
    pool_path = data_root / "pool_manifest.jsonl"
    if not pool_path.is_file():
        raise FileNotFoundError(
            f"Brak {pool_path} - nie da sie wygenerowac annotations_train "
            f"z frame_ids (variant-file nie ma pola annotations_train)")
    index: dict[str, dict] = {}
    with pool_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            index[rec["frame_id"]] = rec
    return index


def build_train_annotations(variant: dict, data_root: Path, dest: Path) -> tuple[Path, dict]:
    """Zwraca (sciezka do COCO train, info).

    Jesli variant ma "annotations_train" - wskazujemy ten plik (konwersja na
    absolutne sciezki obrazow przez kopie z przepisaniem file_name, bez
    kopiowania obrazow). Inaczej generujemy COCO z frame_ids + pool_manifest.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    info: dict = {"source": None, "n_images": 0, "n_annotations": 0}

    if variant.get("annotations_train"):
        src = Path(variant["annotations_train"])
        if not src.is_absolute():
            src = data_root / src
        if not src.is_file():
            raise FileNotFoundError(f"annotations_train nie istnieje: {src}")
        coco = json.loads(src.read_text(encoding="utf-8"))
        _rewrite_file_names_absolute(coco, src.parent, data_root)
        info.update(source=str(src), n_images=len(coco.get("images", [])),
                    n_annotations=len(coco.get("annotations", [])))
    else:
        frame_ids = variant.get("frame_ids")
        if not frame_ids:
            raise ValueError("variant-file nie ma ani annotations_train, ani frame_ids")
        pool = load_pool_manifest(data_root)
        missing = [fid for fid in frame_ids if fid not in pool]
        if missing:
            raise KeyError(f"{len(missing)} frame_ids spoza pool_manifest, np. {missing[:3]}")
        coco = _coco_from_frames([pool[fid] for fid in frame_ids], data_root)
        info.update(source="generated_from_frame_ids",
                    n_images=len(coco["images"]), n_annotations=len(coco["annotations"]))

    dest.write_text(json.dumps(coco), encoding="utf-8")
    return dest, info


def _rewrite_file_names_absolute(coco: dict, ann_dir: Path, data_root: Path) -> None:
    """file_name w COCO -> sciezka absolutna (obrazy leza raz, nie kopiujemy)."""
    for im in coco.get("images", []):
        fn = Path(im["file_name"])
        if not fn.is_absolute():
            cand = ann_dir / fn
            if not cand.is_file():
                cand = data_root / fn
            im["file_name"] = str(cand.resolve())


def _coco_from_frames(frames: list[dict], data_root: Path) -> dict:
    categories = [{"id": 1, "name": "instrument_tip", "supercategory": "instrument"}]
    images, annotations = [], []
    img_id = 1
    ann_id = 1
    for fr in frames:
        img_path = data_root / fr["path"]
        images.append({
            "id": img_id,
            "file_name": str(img_path.resolve()),
            "width": fr.get("native_width"),
            "height": fr.get("native_height"),
        })
        if fr.get("bbox") and fr.get("has_instrument"):
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [float(v) for v in fr["bbox"]],
                "area": float(fr["bbox"][2]) * float(fr["bbox"][3]),
                "iscrowd": 0,
            })
            ann_id += 1
        img_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def prepare_dataset_dir(variant: dict, data_root: Path, run_dir: Path) -> tuple[Path, dict]:
    """Katalog dataset_dir w ukladzie, jakiego oczekuje rfdetr:
    dataset/train/_annotations.coco.json + dataset/valid/_annotations.coco.json.
    Obrazy NIE sa kopiowane - sciezki absolutne w plikach COCO."""
    dataset_dir = run_dir / "dataset"
    train_ann, info = build_train_annotations(
        variant, data_root, dataset_dir / "train" / "_annotations.coco.json")

    val_src = data_root / "annotations_val.json"
    if not val_src.is_file():
        raise FileNotFoundError(f"Brak {val_src} (walidacja po kazdej epoce jest wymagana)")
    val_coco = json.loads(val_src.read_text(encoding="utf-8"))
    _rewrite_file_names_absolute(val_coco, val_src.parent, data_root)
    val_dst = dataset_dir / "valid" / "_annotations.coco.json"
    val_dst.parent.mkdir(parents=True, exist_ok=True)
    val_dst.write_text(json.dumps(val_coco), encoding="utf-8")
    info["n_val_images"] = len(val_coco.get("images", []))
    info["train_annotations"] = str(train_ann)
    info["val_annotations"] = str(val_dst)
    return dataset_dir, info


# --------------------------------------------------------------------------- #
# manifest (README 4.4)
# --------------------------------------------------------------------------- #
def build_manifest(args, protocol: dict, protocol_path: Path, variant_path: Path,
                   variant: dict, started_at: str, finished_at: str,
                   epochs_completed: int, resolution: int, seed: int,
                   model_name: str, extra: dict | None = None) -> dict:
    manifest: dict = {
        "run_id": args.run_id,
        "git_commit": git_commit(),
        "protocol_sha256": sha256_file(protocol_path),
        "variant_file_sha256": sha256_file(variant_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "epochs_completed": epochs_completed,
        "resolution": resolution,
        "seed": seed,
        "model": model_name,
    }
    gpu = gpu_name()
    if gpu:
        manifest["gpu"] = gpu
    tv = torch_version()
    if tv:
        manifest["torch"] = tv
    versions = library_versions()
    if versions:
        manifest["library_versions"] = versions
    if extra:
        manifest.update(extra)
    return manifest


# --------------------------------------------------------------------------- #
# walidacja mAP@0.5:0.95 (kryterium best.pth)
# --------------------------------------------------------------------------- #
def evaluate_map_50_95(det, val_ann_path: Path, conf_threshold: float) -> float:
    """mAP@0.5:0.95 na zbiorze val. Predykcje z progiem conf_threshold
    (protocol.yaml -> evaluation.map_conf_threshold)."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(val_ann_path))
    results = []
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        preds = det.predict(img_info["file_name"], threshold=conf_threshold)
        for box in preds:
            x1, y1, x2, y2, score, cls = (float(box[0]), float(box[1]), float(box[2]),
                                          float(box[3]), float(box[4]), int(box[5]))
            results.append({"image_id": img_id, "category_id": cls + 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    if not results:
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0])


# --------------------------------------------------------------------------- #
# trening
# --------------------------------------------------------------------------- #
def _filter_kwargs(fn, kwargs: dict, run_dir: Path) -> dict:
    """Przekaz tylko te parametry, ktore obsluguje dana wersja rfdetr.
    Odrzucone zapisujemy w unsupported_params.json (bez zaszytych wartosci -
    wszystko pochodzi z protocol.yaml)."""
    import inspect
    sig = inspect.signature(fn)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
    if accepts_var_kw:
        return kwargs
    supported, dropped = {}, {}
    for k, v in kwargs.items():
        (supported if k in sig.parameters else dropped)[k] = v
    if dropped:
        (run_dir / "unsupported_params.json").write_text(
            json.dumps(dropped, indent=2, default=str), encoding="utf-8")
        print(f"[train] UWAGA: rfdetr nie przyjmuje parametrow: {sorted(dropped)}",
              file=sys.stderr)
    return supported


def run_training(args, protocol: dict, run_dir: Path, dataset_dir: Path,
                 model_name: str, resolution: int, seed: int) -> dict:
    """Pelny trening: stala liczba epok, walidacja po kazdej epoce
    (ewaluacja kazdego checkpointu epokowego), wybor best/last, porzadki."""
    import rfdetr  # lazy import - skrypt testowalny bez GPU

    tr = protocol["training"]
    epochs = int(tr["epochs"])
    assert tr.get("early_stopping") is False, "protokol wymaga early_stopping: false"
    aug = tr.get("train_time_augmentation", "none")
    assert aug == "none", f"protokol wymaga train_time_augmentation=none, jest {aug}"

    model_cls = getattr(rfdetr, MODEL_CLASS_MAP[model_name])
    ckpt_dir = run_dir / "checkpoints_raw"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Start ZAWSZE z wag COCO (pretrain_weights domyslne = COCO pretrained).
    ctor_kwargs = _filter_kwargs(model_cls.__init__, {"resolution": resolution}, run_dir)
    det = model_cls(**ctor_kwargs)

    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": int(tr["batch_size"]),
        "gradient_accumulation_steps": int(tr["grad_accum"]),
        "lr": float(tr["lr"]),
        "lr_backbone": float(tr["lr_backbone"]),
        "weight_decay": float(tr["weight_decay"]),
        "warmup_epochs": int(tr["warmup_epochs"]),
        "amp": tr.get("precision") == "bf16",
        "seed": seed,
        "exp_name": args.run_id,
        "output_dir": str(ckpt_dir),
        "resume": False,
    }
    det.train(**_filter_kwargs(det.train, train_kwargs, run_dir))

    # Walidacja mAP@0.5:0.95 - po kazdej epoce (checkpointy epokowe rfdetr).
    criterion = protocol["checkpoint_selection"]["best_val_criterion"]
    assert criterion == "map_50_95_val", f"nieobslugiwane kryterium: {criterion}"
    conf_threshold = float(protocol["evaluation"]["map_conf_threshold"])
    val_ann = dataset_dir / "valid" / "_annotations.coco.json"

    epoch_ckpts = sorted(ckpt_dir.rglob("checkpoint*.pth"))
    if not epoch_ckpts:
        raise FileNotFoundError(f"rfdetr nie zapisal checkpointow w {ckpt_dir}")

    best_map, best_ckpt, per_epoch = -1.0, None, {}
    for i, ckpt in enumerate(epoch_ckpts, start=1):
        m = evaluate_map_50_95(
            model_cls(**{**ctor_kwargs,
                         **_filter_kwargs(model_cls.__init__,
                                          {"pretrain_weights": str(ckpt)}, run_dir)}),
            val_ann, conf_threshold)
        per_epoch[i] = m
        print(f"[train] epoka {i}/{len(epoch_ckpts)}: map_50_95_val={m:.4f}")
        if m > best_map:
            best_map, best_ckpt = m, ckpt

    last_ckpt = epoch_ckpts[-1]
    shutil.copyfile(last_ckpt, run_dir / "last.pth")
    shutil.copyfile(best_ckpt, run_dir / "best.pth")
    (run_dir / "epoch_val_map.json").write_text(
        json.dumps({"criterion": criterion, "per_epoch_map_50_95_val": per_epoch,
                    "best_epoch": max(per_epoch, key=per_epoch.get),
                    "best_map_50_95_val": best_map}, indent=2), encoding="utf-8")

    # Wymaganie 7: kasujemy checkpointy posrednie.
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    return {"epochs_completed": len(epoch_ckpts), "best_map_50_95_val": best_map}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant-file", required=True, type=Path)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--role", choices=["main", "smoke"], default="main",
                    help="main: protocol->model.name; smoke: model.smoke_model "
                         "@ model.smoke_resolution")
    ap.add_argument("--dry-run", action="store_true",
                    help="pelna inicjalizacja i walidacja plikow, manifest "
                         "in-memory + zapis, BEZ treningu")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started_at = now_iso()

    protocol_path = args.protocol.resolve()
    variant_path = args.variant_file.resolve()
    data_root = args.data_root.resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    if not variant_path.is_file():
        sys.exit(f"variant-file nie istnieje: {variant_path}")
    if not data_root.is_dir():
        sys.exit(f"data-root nie istnieje: {data_root}")

    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    if "seed" not in variant:
        sys.exit("variant-file bez pola seed")
    seed = int(variant["seed"])
    set_seed(seed)

    model_name = (protocol["model"]["smoke_model"] if args.role == "smoke"
                  else protocol["model"]["name"])
    resolution = int(protocol["model"]["smoke_resolution"] if args.role == "smoke"
                     else protocol["model"]["resolution"])
    if model_name not in MODEL_CLASS_MAP:
        sys.exit(f"nieznany model w protocol.yaml: {model_name}")

    run_id = args.run_id
    if variant.get("run_id") and variant["run_id"] != run_id:
        print(f"[train] OSTRZEZENIE: run_id w variant-file ({variant['run_id']}) "
              f"!= --run-id ({run_id})", file=sys.stderr)

    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # kontrakt 4.2: liczba frame_ids == budget
    budget = variant.get("budget")
    frame_ids = variant.get("frame_ids")
    if frame_ids is not None and isinstance(budget, int) and len(frame_ids) != budget:
        sys.exit(f"variant {run_id}: len(frame_ids)={len(frame_ids)} != budget={budget}")

    dataset_dir, data_info = prepare_dataset_dir(variant, data_root, run_dir)
    print(f"[train] dataset: {json.dumps(data_info, ensure_ascii=False)}")

    epochs_planned = 0 if args.dry_run else int(protocol["training"]["epochs"])
    extra = {"dry_run": True, "role": args.role, "dataset": data_info} if args.dry_run \
        else {"role": args.role, "dataset": data_info}

    if args.dry_run:
        print("[train] DRY-RUN: walidacja plikow + manifest, bez treningu")
    else:
        result = run_training(args, protocol, run_dir, dataset_dir,
                              model_name, resolution, seed)
        epochs_planned = result["epochs_completed"]
        extra["best_map_50_95_val"] = result["best_map_50_95_val"]

    manifest = build_manifest(args, protocol, protocol_path, variant_path, variant,
                              started_at, now_iso(), epochs_planned, resolution,
                              seed, model_name, extra)
    manifest_path = run_dir / "run_manifest.json"
    tmp_fd, tmp_name = tempfile.mkstemp(dir=run_dir, suffix=".json")
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_name, manifest_path)
    print(f"[train] zapisano {manifest_path}")

    # porzadki: poza last.pth/best.pth/run_manifest.json/logami nie zostaje nic
    if not args.dry_run:
        for junk in ("dataset", "unsupported_params.json"):
            p = run_dir / junk
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
