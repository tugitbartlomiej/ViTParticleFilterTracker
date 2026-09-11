#!/usr/bin/env python3
"""Wspolna logika dla Agent 4 (przeniesienie na inne architektury).

Zawiera: czytanie konfiguracji, walidacje pliku wariantu, rozwiazywanie
ustawien iso-resource, budowe adnotacji COCO z pool_manifest.jsonl,
skroty sha256, metadane sprzetu/bibliotek i budowe run_manifest.json.

Nic tu nie importuje ultralytics ani d-fine — te sa LAZY, w train_transfer.py.
Modul celowo dzieli logike miedzy train_transfer.py, nms_sensitivity.py,
metrics.py i check_iso_resource.py, zeby iso-resource bylo liczone RAZ.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("Brak pyyaml. Zainstaluj: py -3.11 -m pip install pyyaml")

# Cataract-LMM udostępnia bbox instrumentu i osobny punkt końcówki. Pipeline
# detekcji bbox nie może przedstawiać ramki instrumentu jako ramki końcówki.
CATEGORY_NAME = "instrument"
CATEGORY_ID = 1

# Pola manifestu wymagane przez agents/README.md 4.4 + agents/ops/validate_contracts.py
MANIFEST_REQUIRED_FIELDS = {
    "run_id", "git_commit", "protocol_sha256", "variant_file_sha256",
    "started_at", "finished_at", "hostname", "gpu",
    "python", "torch", "library_versions", "epochs_completed",
    "resolution", "seed",
}


# --------------------------------------------------------------------------- #
# skroty i narzedzia plikowe
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    """sha256 pliku w kawalkach 1 MiB (te same co validate_contracts.py)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_protocol(path: Path) -> dict:
    proto = load_yaml(path)
    for key in ("model", "training", "checkpoint_selection", "evaluation"):
        if key not in proto:
            raise ValueError(f"protocol.yaml bez sekcji '{key}': {path}")
    return proto


def load_variant(path: Path) -> dict:
    var = json.loads(Path(path).read_text(encoding="utf-8"))
    if "frame_ids" not in var and "selected_frame_ids" in var:
        var["frame_ids"] = list(var["selected_frame_ids"])
    if "run_id" not in var and "variant_id" in var:
        var["run_id"] = var["variant_id"]
    for key in ("run_id", "frame_ids", "seed"):
        if key not in var:
            raise ValueError(f"variant-file bez pola '{key}': {path}")
    return var


# --------------------------------------------------------------------------- #
# iso-resource — wszystko czytane z protocol.yaml, zero zaszytych wartosci
# --------------------------------------------------------------------------- #
def resolve_iso_resource(protocol: dict) -> dict:
    """Return the locked recipe used by one architecture/profile comparison.

    Rozdzielczosc moze byc natywna dla architektury. W obrebie modelu budzet
    epok, ziarno, podzial, batch, checkpoint i ewaluacja pozostaja identyczne.
    """
    model = protocol["model"]
    training = protocol["training"]
    ckpt = protocol["checkpoint_selection"]
    ev = protocol["evaluation"]
    transfer = protocol.get("transfer", {})
    nms = transfer.get("nms", {})
    return {
        "resolution": int(model["resolution"]),
        "epochs": int(training["epochs"]),
        "early_stopping": bool(training.get("early_stopping", False)),
        "batch_size": int(training.get("batch_size", 16)),
        "grad_accum": int(training.get("grad_accum", 1)),
        "precision": training.get("precision", "bf16"),
        "best_val_criterion": ckpt["best_val_criterion"],
        "report_checkpoints": list(ckpt.get("report", ["last_epoch", "best_val"])),
        "map_conf_threshold": float(ev["map_conf_threshold"]),
        "eval_iou_threshold": float(ev["iou_threshold"]),
        "operating_threshold_rule": ev.get("operating_threshold", "select_on_val"),
        "metrics": list(ev["metrics"]),
        "split_seed": int(protocol.get("split", {}).get("seed", 42)),
        "nms_conf_threshold": float(nms.get("conf_threshold", ev["map_conf_threshold"])),
        "nms_iou_threshold": float(nms.get("iou_threshold", 0.7)),
        "nms_sweep_iou": list(nms.get("sweep_iou", [0.5, 0.7])),
    }


def assert_no_early_stopping(iso: dict) -> None:
    if iso["early_stopping"]:
        raise ValueError(
            "protocol.yaml: training.early_stopping=true — zakazane (README sekcja 2).")


# --------------------------------------------------------------------------- #
# walidacja wariantu (kontrakt agents/README.md 4.2)
# --------------------------------------------------------------------------- #
def validate_variant(variant: dict, pool_by_id: dict[str, dict] | None = None) -> list[str]:
    """Sprawdza plik wariantu. Zwraca liste komunikatow ostrzegawczych.

    Rzuca ValueError na bledy twarde (budget != len(frame_ids), duplikaty).
    """
    warnings: list[str] = []
    frame_ids = variant["frame_ids"]
    budget = variant.get("budget")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("variant: frame_ids zawiera duplikaty")
    if budget == "full_train_pool":
        warnings.append("variant: budget=full_train_pool (INNY budzet, raportowany osobno)")
    elif budget is not None and len(frame_ids) != int(budget):
        raise ValueError(
            f"variant: len(frame_ids)={len(frame_ids)} != budget={budget}")

    if pool_by_id is not None:
        outside_split = []
        missing = []
        for fid in frame_ids:
            row = pool_by_id.get(fid)
            if row is None:
                missing.append(fid)
            elif row.get("split") != "train":
                outside_split.append(fid)
        if missing:
            raise ValueError(
                f"variant: {len(missing)} frame_ids nieobecnych w pool_manifest "
                f"(przyklad: {missing[:3]})")
        if outside_split:
            raise ValueError(
                f"variant: {len(outside_split)} frame_ids poza split=train "
                f"(przyklad: {outside_split[:3]})")
    return warnings


def load_pool_manifest(path: Path) -> dict[str, dict]:
    """Caly pool_manifest.jsonl jako {frame_id: row}."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"pool_manifest.jsonl nie istnieje: {path}\n"
            "Oczekiwany layout: <data-root>/pool_manifest.jsonl (kontrakt 4.1).")
    return {row["frame_id"]: row for row in read_jsonl(path)}


# --------------------------------------------------------------------------- #
# budowa adnotacji COCO z pool_manifest (jedna klasa)
# --------------------------------------------------------------------------- #
def build_coco(
    rows: Iterable[dict],
    data_root: Path,
    image_size_reader=None,
) -> dict:
    """Buduje slownik COCO (images/annotations/categories) z wierszy pool_manifest.

    bbox w pool_manifest to [x, y, w, h] w pikselach obrazu po normalizacji
    (konwencja COCO). Klatki bez narzedzia wchodza jako obrazy tla (bez adnotacji).
    image_size_reader(path)->(w,h); jesli None, uzywa native_width/native_height.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    for img_id, row in enumerate(rows, start=1):
        rel = row["path"]
        abs_path = Path(data_root) / rel
        if image_size_reader is not None:
            w, h = image_size_reader(abs_path)
        else:
            w = int(row.get("native_width", 0)) or 1
            h = int(row.get("native_height", 0)) or 1
        images.append({
            "id": img_id,
            "file_name": rel,
            "width": w,
            "height": h,
            # metadane wymagane przez docs/13, zeby znac zrodlo ramki
            "source_dataset": row.get("source_dataset"),
            "patient_id": row.get("patient_id"),
            "video_id": row.get("video_id"),
            "frame_index": row.get("frame_index"),
            "label_origin": row.get("label_origin"),
            "seed_distance_s": row.get("seed_distance_s"),
        })
        row_annotations = row.get("annotations")
        if row_annotations is None:
            row_annotations = ([{"bbox": row["bbox"], "iscrowd": 0}]
                               if row.get("has_instrument") and row.get("bbox") else [])
        for source_ann in row_annotations:
            x, y, bw, bh = source_ann["bbox"]
            if min(float(x), float(y)) < 0 or min(float(bw), float(bh)) <= 0:
                raise ValueError(f"invalid bbox for {row.get('frame_id')}: {source_ann['bbox']}")
            if float(x) + float(bw) > w + 1e-3 or float(y) + float(bh) > h + 1e-3:
                raise ValueError(f"bbox outside image for {row.get('frame_id')}: {source_ann['bbox']}")
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORY_ID,
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": float(bw) * float(bh),
                "iscrowd": int(source_ann.get("iscrowd", 0)),
            })
            ann_id += 1
    return {
        "info": {"description": "SmallViTAblation instrument bbox (transfer)"},
        "licenses": [],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": "instrument"}],
        "images": images,
        "annotations": annotations,
    }


def image_size_from_file(path: Path) -> tuple[int, int]:
    """Czyta (w, h) z pliku przez PIL — LAZY import."""
    from PIL import Image  # lazy
    with Image.open(path) as im:
        return int(im.width), int(im.height)


# --------------------------------------------------------------------------- #
# metadane srodowiska do run_manifest
# --------------------------------------------------------------------------- #
def git_commit(repo_root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def gpu_name() -> str:
    try:
        import torch  # lazy
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def torch_version() -> str:
    try:
        import torch  # lazy
        return torch.__version__
    except Exception:
        return "absent"


def library_versions(names: Iterable[str]) -> dict[str, str]:
    """Wersje bibliotek (importlib.metadata), bez importowania ciezarnych modulow."""
    from importlib import metadata
    out: dict[str, str] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except Exception:
            out[n] = "absent"
    return out


# --------------------------------------------------------------------------- #
# run_manifest.json — format 4.4 + pole arch + ustawienia NMS
# --------------------------------------------------------------------------- #
def nms_manifest_block(arch: str, iso: dict, protocol: dict) -> dict:
    """Ustawienia NMS do manifestu. Dla D-FINE: NMS nie dotyczy."""
    transfer = protocol.get("transfer", {})
    archs = transfer.get("architectures", {})
    arch_cfg = archs.get(arch, {})
    if arch_cfg.get("nms_applies", arch == "yolo-current") is False or arch == "d-fine-s":
        reasons = {
            "d-fine-s": "D-FINE predicts an end-to-end set of detections without NMS.",
            "rtdetrv2-s": "RT-DETRv2 predicts an end-to-end set of detections without NMS.",
            "rf-detr-nano": "RF-DETR predicts an end-to-end set of detections without NMS.",
        }
        return {
            "applies": False,
            "reason": reasons.get(arch, "This architecture does not apply NMS."),
            "iou_threshold": None,
            "conf_threshold": None,
        }
    return {
        "applies": True,
        "iou_threshold": iso["nms_iou_threshold"],
        "conf_threshold": iso["nms_conf_threshold"],
        "sweep_iou": iso["nms_sweep_iou"],
        "end_to_end": bool(arch_cfg.get("end_to_end", False)),
        "note": ("YOLO26 jest natywnie end-to-end (bez NMS); progi dotycza "
                 "sciezki ewaluacyjnej/fallback oraz analizy wrazliwosci."),
    }


def build_run_manifest(
    *,
    run_id: str,
    arch: str,
    protocol_path: Path,
    variant_path: Path,
    iso: dict,
    protocol: dict,
    seed: int,
    started_at: str,
    finished_at: str | None,
    epochs_completed: int,
    repo_root: Path | None,
    dry_run: bool,
) -> dict:
    """Buduje run_manifest.json (DOKLADNIE 4.4 + arch + nms)."""
    transfer = protocol.get("transfer", {}).get("architectures", {}).get(arch, {})
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_commit": git_commit(repo_root),
        "protocol_sha256": sha256_file(protocol_path),
        "variant_file_sha256": sha256_file(variant_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "hostname": socket.gethostname(),
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "torch": torch_version(),
        "library_versions": library_versions(
            ["ultralytics", "numpy", "pycocotools", "pillow", "torch"]),
        "epochs_completed": int(epochs_completed),
        "resolution": int(iso["resolution"]),
        "seed": int(seed),
        # --- pola Agent 4 ---
        "arch": arch,
        "model_id": transfer.get("model_id"),
        "library": transfer.get("library"),
        "iso_resource": {
            "resolution": int(iso["resolution"]),
            "epochs": int(iso["epochs"]),
            "seed": int(seed),
            "batch_size": int(iso["batch_size"]),
            "grad_accum": int(iso["grad_accum"]),
            "precision": str(iso["precision"]),
            "train_time_augmentation": str(
                protocol["training"].get("train_time_augmentation", "none")),
            "early_stopping": bool(iso["early_stopping"]),
            "variant_file": str(Path(variant_path).name),
            "variant_file_sha256": sha256_file(variant_path),
            "best_val_criterion": iso["best_val_criterion"],
            "map_conf_threshold": iso["map_conf_threshold"],
            "eval_iou_threshold": iso["eval_iou_threshold"],
            "operating_threshold_rule": iso["operating_threshold_rule"],
        },
        "nms": nms_manifest_block(arch, iso, protocol),
        "checkpoint_selection": {
            "best_val_criterion": iso["best_val_criterion"],
            "report": iso["report_checkpoints"],
        },
        "dry_run": bool(dry_run),
    }
    missing = MANIFEST_REQUIRED_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"run_manifest bez wymaganych pol: {sorted(missing)}")
    return manifest


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
