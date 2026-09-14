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
import csv
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
    # Archive deployments carry the exact originating commit in the job environment.
    if os.environ.get("SMALLVIT_SOURCE_COMMIT"):
        return os.environ["SMALLVIT_SOURCE_COMMIT"]
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
        _assert_coco_images_exist(coco, "train annotations")
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


def _assert_coco_images_exist(coco: dict, label: str) -> None:
    missing = [im.get("file_name") for im in coco.get("images", [])
               if not im.get("file_name") or not Path(im["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label}: {len(missing)} images are missing, for example {missing[:3]}")


def _coco_from_frames(frames: list[dict], data_root: Path) -> dict:
    categories = [{"id": 1, "name": "instrument", "supercategory": "tool"}]
    images, annotations = [], []
    img_id = 1
    ann_id = 1
    for fr in frames:
        if fr.get("split") != "train":
            raise ValueError(f"frame {fr.get('frame_id')} is outside split=train")
        if fr.get("label_status") == "unknown":
            raise ValueError(
                f"frame {fr.get('frame_id')} has an unverified label; annotate or verify negative")
        if not fr.get("path"):
            raise ValueError(
                f"frame {fr.get('frame_id')} is archive-backed; materialize it before training")
        img_path = data_root / fr["path"]
        if not img_path.is_file():
            raise FileNotFoundError(f"missing training image for {fr.get('frame_id')}: {img_path}")
        images.append({
            "id": img_id,
            "file_name": str(img_path.resolve()),
            "width": fr.get("native_width"),
            "height": fr.get("native_height"),
        })
        source_annotations = fr.get("annotations")
        if source_annotations is None:
            source_annotations = ([{"bbox": fr["bbox"], "iscrowd": 0}]
                                  if fr.get("bbox") and fr.get("has_instrument") else [])
        for source_ann in source_annotations:
            bbox = [float(v) for v in source_ann["bbox"]]
            if len(bbox) != 4 or min(bbox[0], bbox[1]) < 0 or min(bbox[2], bbox[3]) <= 0:
                raise ValueError(f"invalid bbox for {fr.get('frame_id')}: {bbox}")
            width, height = float(fr.get("native_width") or 0), float(fr.get("native_height") or 0)
            if width <= 0 or height <= 0 or bbox[0] + bbox[2] > width + 1e-3 or bbox[1] + bbox[3] > height + 1e-3:
                raise ValueError(f"bbox outside image for {fr.get('frame_id')}: {bbox}")
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": int(source_ann.get("iscrowd", 0)),
            })
            ann_id += 1
        img_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def prepare_dataset_dir(variant: dict, data_root: Path, run_dir: Path,
                        val_annotations: Path | None = None) -> tuple[Path, dict]:
    """Katalog dataset_dir w ukladzie, jakiego oczekuje rfdetr:
    dataset/train/_annotations.coco.json + dataset/valid/_annotations.coco.json.
    Obrazy NIE sa kopiowane - sciezki absolutne w plikach COCO."""
    dataset_dir = run_dir / "dataset"
    train_ann, info = build_train_annotations(
        variant, data_root, dataset_dir / "train" / "_annotations.coco.json")

    val_src = val_annotations or (data_root / "annotations_val.json")
    if not val_src.is_absolute():
        val_src = data_root / val_src
    if not val_src.is_file():
        raise FileNotFoundError(f"Brak {val_src} (walidacja po kazdej epoce jest wymagana)")
    val_coco = json.loads(val_src.read_text(encoding="utf-8"))
    _rewrite_file_names_absolute(val_coco, val_src.parent, data_root)
    _assert_coco_images_exist(val_coco, "validation annotations")
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
        "training_recipe": {
            "epochs": int(protocol["training"]["epochs"]),
            "batch_size": int(protocol["training"]["batch_size"]),
            "grad_accum": int(protocol["training"]["grad_accum"]),
            "precision": str(protocol["training"]["precision"]),
            "train_time_augmentation": str(
                protocol["training"].get("train_time_augmentation", "none")),
            "early_stopping": bool(protocol["training"].get("early_stopping", False)),
        },
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
def iter_predictions(predictions):
    """Yield ``(x1, y1, x2, y2, score, class_id)`` across RF-DETR APIs."""
    import numpy as np

    if isinstance(predictions, (list, tuple)) and len(predictions) == 1 \
            and hasattr(predictions[0], "xyxy"):
        predictions = predictions[0]
    if hasattr(predictions, "xyxy"):
        boxes = np.asarray(predictions.xyxy, dtype=float).reshape(-1, 4)
        confidence = getattr(predictions, "confidence", None)
        class_id = getattr(predictions, "class_id", None)
        confidence = (np.ones(len(boxes), dtype=float) if confidence is None
                      else np.asarray(confidence, dtype=float).reshape(-1))
        class_id = (np.zeros(len(boxes), dtype=int) if class_id is None
                    else np.asarray(class_id, dtype=int).reshape(-1))
        if len(confidence) != len(boxes) or len(class_id) != len(boxes):
            raise ValueError("RF-DETR prediction fields have inconsistent lengths")
        for box, score, cls in zip(boxes, confidence, class_id):
            yield (*map(float, box), float(score), int(cls))
        return
    for item in predictions:
        if isinstance(item, dict):
            box = item.get("xyxy")
            if box is None:
                box = item.get("bbox")
            if box is None:
                raise ValueError("prediction dictionary has no xyxy/bbox")
            score = item.get("confidence", item.get("score", 1.0))
            cls = item.get("class_id", item.get("category_id", 0))
            values = list(map(float, box))
            if "bbox" in item and "xyxy" not in item:
                values = [values[0], values[1], values[0] + values[2], values[1] + values[3]]
            yield (*values, float(score), int(cls))
        else:
            values = list(map(float, item))
            if len(values) < 4:
                raise ValueError("prediction row has fewer than four coordinates")
            yield (*values[:4], values[4] if len(values) > 4 else 1.0,
                   int(values[5]) if len(values) > 5 else 0)


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
        for x1, y1, x2, y2, score, cls in iter_predictions(preds):
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
        raise TypeError(
            "Installed rfdetr API cannot honor the locked training recipe; "
            f"unsupported parameters: {sorted(dropped)}")
    return supported


def run_training(args, protocol: dict, run_dir: Path, dataset_dir: Path,
                 model_name: str, resolution: int, seed: int) -> dict:
    """Run the locked recipe and collect RF-DETR's validation-selected outputs."""
    import rfdetr  # lazy import - skrypt testowalny bez GPU

    tr = protocol["training"]
    epochs = int(tr["epochs"])
    assert tr.get("early_stopping") is False, "protokol wymaga early_stopping: false"
    aug = tr.get("train_time_augmentation", "none")
    assert aug == "none", f"protokol wymaga train_time_augmentation=none, jest {aug}"

    model_cls = getattr(rfdetr, MODEL_CLASS_MAP[model_name])
    ckpt_dir = run_dir / "checkpoints_raw"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Start ZAWSZE z wag COCO (pretrain_weights domyslne = COCO pretrained).
    ctor_kwargs = _filter_kwargs(model_cls.__init__, {"resolution": resolution}, run_dir)
    det = model_cls(**ctor_kwargs)

    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": int(tr["batch_size"]),
        "grad_accum_steps": int(tr["grad_accum"]),
        "lr": float(tr["lr"]),
        "lr_encoder": float(tr["lr_backbone"]),
        "weight_decay": float(tr["weight_decay"]),
        "warmup_epochs": int(tr["warmup_epochs"]),
        "amp": str(tr.get("precision", "")).lower() in {"bf16", "fp16"},
        "amp_dtype": "bf16" if tr.get("precision") == "bf16" else "fp16",
        "seed": seed,
        "output_dir": str(ckpt_dir),
        "resume": None,
        "early_stopping": False,
        "eval_interval": 1,
        "best_model_metric": "map",
        "run_test": False,
        "aug_config": {},
        "scale_jitter": False,
        "multi_scale": False,
    }
    det.train(**_filter_kwargs(det.train, train_kwargs, run_dir))

    criterion = protocol["checkpoint_selection"]["best_val_criterion"]
    assert criterion == "map_50_95_val", f"nieobslugiwane kryterium: {criterion}"

    # Current RF-DETR emits these exact portable weight files. Its callback
    # selects checkpoint_best_total.pth by validation box mAP@0.5:0.95 and
    # writes last.pth at fit end. Re-evaluating ad-hoc checkpoint globs here
    # would mix .ckpt and best/EMA files and can select the wrong epoch.
    best_ckpt = ckpt_dir / "checkpoint_best_total.pth"
    last_ckpt = ckpt_dir / "last.pth"
    missing = [path.name for path in (best_ckpt, last_ckpt) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"RF-DETR did not emit required checkpoint(s) {missing} in {ckpt_dir}")
    shutil.copyfile(last_ckpt, run_dir / "last.pth")
    shutil.copyfile(best_ckpt, run_dir / "best.pth")

    history = _read_validation_history(ckpt_dir)
    summary = {"criterion": criterion, "source": "RF-DETR BestModelCallback",
               "per_epoch_map_50_95_val": history}
    if history:
        best_epoch = max(history, key=history.get)
        summary.update(best_epoch=best_epoch,
                       best_map_50_95_val=history[best_epoch])
    (run_dir / "epoch_val_map.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # Keep metrics/configuration logs before removing intermediate checkpoints.
    for path in ckpt_dir.rglob("*"):
        if path.is_file() and path.suffix not in {".pth", ".ckpt", ".pt"}:
            target = run_dir / "training_logs" / path.relative_to(ckpt_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    import torch
    performance = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    (run_dir / "training_performance.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8")

    # Wymaganie 7: kasujemy checkpointy posrednie.
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    return {"epochs_completed": epochs,
            "best_map_50_95_val": summary.get("best_map_50_95_val")}


def _read_validation_history(output_dir: Path) -> dict[int, float]:
    """Read RF-DETR's CSV logger without assuming a version-specific folder."""
    result: dict[int, float] = {}
    for path in sorted(output_dir.rglob("metrics.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                metric_key = next((key for key in row
                                   if key.startswith("val/")
                                   and "mAP_50_95" in key
                                   and row.get(key) not in (None, "")), None)
                if metric_key is None:
                    continue
                epoch_raw = row.get("epoch", "")
                if epoch_raw in (None, ""):
                    continue
                result[int(float(epoch_raw))] = float(row[metric_key])
    return result


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
    ap.add_argument("--train-seed", type=int, default=None,
                    help="paired training seed; independent of selection seed")
    ap.add_argument("--profile", default=None,
                    help="execution profile from configs/profiles.yaml")
    ap.add_argument("--profiles-config", type=Path, default=None)
    ap.add_argument("--val-annotations", type=Path, default=None,
                    help="fixed validation COCO; defaults to <data-root>/annotations_val.json")
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
    if args.profile:
        import copy
        profile_path = args.profiles_config or (protocol_path.parent / "profiles.yaml")
        profiles = yaml.safe_load(profile_path.read_text(encoding="utf-8")).get("profiles", {})
        if args.profile not in profiles:
            sys.exit(f"unknown experiment profile: {args.profile}")
        model_for_profile = ("rf-detr-nano" if args.role == "smoke"
                             else protocol["model"]["name"])
        allowed = profiles[args.profile].get("models", [])
        if allowed and model_for_profile not in allowed:
            sys.exit(f"profile {args.profile} does not allow model {model_for_profile}")
        protocol = copy.deepcopy(protocol)
        profile = profiles[args.profile]
        protocol["training"]["epochs"] = int(profile["epochs"])
        resolutions = profile.get("resolution_by_model", {})
        if model_for_profile in resolutions:
            key = "smoke_resolution" if args.role == "smoke" else "resolution"
            protocol["model"][key] = int(resolutions[model_for_profile])
        precisions = profile.get("precision_by_model", {})
        if model_for_profile in precisions:
            protocol["training"]["precision"] = str(precisions[model_for_profile])

    if not variant_path.is_file():
        sys.exit(f"variant-file nie istnieje: {variant_path}")
    if not data_root.is_dir():
        sys.exit(f"data-root nie istnieje: {data_root}")

    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    if "seed" not in variant:
        sys.exit("variant-file bez pola seed")
    seed = int(args.train_seed if args.train_seed is not None else variant["seed"])
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
    if frame_ids is not None and len(frame_ids) != len(set(frame_ids)):
        sys.exit(f"variant {run_id}: frame_ids contains duplicates")
    if frame_ids is not None and isinstance(budget, int) and len(frame_ids) != budget:
        sys.exit(f"variant {run_id}: len(frame_ids)={len(frame_ids)} != budget={budget}")

    dataset_dir, data_info = prepare_dataset_dir(
        variant, data_root, run_dir, args.val_annotations)
    print(f"[train] dataset: {json.dumps(data_info, ensure_ascii=False)}")

    epochs_planned = 0 if args.dry_run else int(protocol["training"]["epochs"])
    experiment = {"strategy": variant.get(
                      "strategy", variant.get("variant", variant.get("variant_id"))),
                  "budget": variant.get("budget"),
                  "selection_seed": variant.get("seed"),
                  "train_seed": seed, "round": variant.get("round")}
    extra = {"dry_run": True, "role": args.role, "dataset": data_info,
             "experiment_profile": args.profile, "experiment": experiment} if args.dry_run \
        else {"role": args.role, "dataset": data_info,
              "experiment_profile": args.profile, "experiment": experiment}
    if args.profile:
        extra["profiles_config"] = str(Path(profile_path).resolve())
        extra["profiles_sha256"] = sha256_file(Path(profile_path))

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
        for junk in ("dataset",):
            p = run_dir / junk
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
