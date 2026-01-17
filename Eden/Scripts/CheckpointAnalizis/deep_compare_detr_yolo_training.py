"""
Deep DETR vs YOLO training analysis (two-phase schedule).

Phase protocol (per your training setup):
  - Phase 1: epoch <= 170, training on the augmented pool (~91k images)
  - Phase 2: epoch > 170, fine-tuning on the curated subset (20k images)

Outputs (written to `Eden\\Scripts\\CheckpointAnalizis\\output\\compare_detr_yolo\\` by default):
  - `deep_yolo_metrics.png/.pdf`
  - `deep_detr_finetune_log.png/.pdf`
  - `deep_weight_drift.png/.pdf`
  - `deep_group_drift.png/.pdf`
  - `deep_training_analysis.md`

Run:
  py -3.11 Eden\\Scripts\\CheckpointAnalizis\\deep_compare_detr_yolo_training.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] torch is required. Run with Python 3.11:\n"
        "  py -3.11 Eden\\Scripts\\CheckpointAnalizis\\deep_compare_detr_yolo_training.py"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

PHASE_BOUNDARY_EPOCH = 170
PHASE1_DATASET_SIZE = 91260
PHASE2_DATASET_SIZE = 20000
DEFAULT_MAX_EPOCH = 200

DTYPES_FLOAT = (torch.float16, torch.float32, torch.bfloat16)

DEFAULT_DETR_PHASE1_DIR = (
    REPO_ROOT
    / "Eden"
    / "Checkpoints"
    / "DETR"
    / "DETR_Training_Sessions"
    / "2025-10-31_tooltip_single_class"
    / "DETR_Checkpoints"
)
DEFAULT_DETR_PHASE2_DIR = (
    REPO_ROOT
    / "Eden"
    / "Checkpoints"
    / "DETR"
    / "DETR_Training_Sessions"
    / "2025-12-13_20kDataset_small_LR"
    / "ckpt_20k_finetune"
)
DEFAULT_DETR_PHASE2_LOG = (
    REPO_ROOT
    / "Eden"
    / "Checkpoints"
    / "DETR"
    / "DETR_Training_Sessions"
    / "2025-12-13_20kDataset_small_LR"
    / "detr_20k_finetune_pascal_1454332.log"
)

DEFAULT_YOLO_PHASE1_DIR = REPO_ROOT / "Eden" / "Checkpoints" / "YOLO_EDEN_TRAIN" / "exp" / "weights"
DEFAULT_YOLO_PHASE2_DIR = (
    REPO_ROOT
    / "Eden"
    / "Checkpoints"
    / "YOLO_EDEN_TRAIN"
    / "YoloTreningSesnions"
    / "20KTreningFrom170epochStart"
    / "exp"
    / "weights"
)
DEFAULT_YOLO_PHASE1_CSV = REPO_ROOT / "Eden" / "Checkpoints" / "YOLO_EDEN_TRAIN" / "exp" / "results.csv"
DEFAULT_YOLO_PHASE2_CSV = (
    REPO_ROOT
    / "Eden"
    / "Checkpoints"
    / "YOLO_EDEN_TRAIN"
    / "YoloTreningSesnions"
    / "20KTreningFrom170epochStart"
    / "exp"
    / "results.csv"
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "compare_detr_yolo"


@dataclass(frozen=True)
class CheckpointRef:
    epoch: int
    path: Path


@dataclass
class PairwiseStats:
    l2_a: float
    l2_b: float
    l2_diff: float
    rel_diff: float
    cosine: Optional[float]
    num_tensors: int


@dataclass
class DriftPoint:
    epoch: int
    weight_l2: float
    update_l2: Optional[float]
    update_rel: Optional[float]
    dist170_l2: Optional[float]
    dist170_rel: Optional[float]
    cos170: Optional[float]


@dataclass
class DetrLogEpoch:
    epoch: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    grad_norm: Optional[float]
    epoch_time_s: Optional[float]


def _add_phase_markers(ax, x_min: float, x_max: float, boundary: int = PHASE_BOUNDARY_EPOCH) -> None:
    phase1_end = min(boundary, x_max)
    phase2_start = max(boundary, x_min)

    if x_min < phase1_end:
        ax.axvspan(x_min, phase1_end, color="tab:gray", alpha=0.08)
    if phase2_start < x_max:
        ax.axvspan(phase2_start, x_max, color="tab:orange", alpha=0.08)

    ax.axvline(boundary, color="black", linestyle=":", linewidth=1)


def _parse_epoch_from_name(path: Path, prefix: str, suffix: str) -> Optional[int]:
    m = re.search(re.escape(prefix) + r"(?P<epoch>\d+)" + re.escape(suffix), path.name)
    if not m:
        return None
    try:
        return int(m.group("epoch"))
    except Exception:
        return None


def discover_checkpoints(
    checkpoint_dir: Path,
    prefix: str,
    suffix: str,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    extra_named: Optional[Sequence[Tuple[str, int]]] = None,
) -> List[CheckpointRef]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    refs: List[CheckpointRef] = []
    for p in sorted(checkpoint_dir.glob(f"{prefix}*{suffix}")):
        epoch = _parse_epoch_from_name(p, prefix, suffix)
        if epoch is None:
            continue
        if min_epoch is not None and epoch < min_epoch:
            continue
        if max_epoch is not None and epoch > max_epoch:
            continue
        refs.append(CheckpointRef(epoch=epoch, path=p))

    if extra_named:
        for name, epoch in extra_named:
            p = checkpoint_dir / name
            if not p.exists():
                continue
            if min_epoch is not None and epoch < min_epoch:
                continue
            if max_epoch is not None and epoch > max_epoch:
                continue
            refs.append(CheckpointRef(epoch=epoch, path=p))

    refs.sort(key=lambda r: r.epoch)
    dedup: Dict[int, CheckpointRef] = {}
    for r in refs:
        dedup[r.epoch] = r
    return [dedup[e] for e in sorted(dedup.keys())]


def load_detr_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    # Prefer safe loading when available (avoids pickle of arbitrary objects).
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected DETR checkpoint format (missing model_state_dict): {path}")
    return sd


def _get_state_dict_from_ultralytics_model(obj: Any) -> Optional[Dict[str, torch.Tensor]]:
    if obj is None:
        return None
    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict()
        except Exception:
            return None
    inner = getattr(obj, "model", None)
    if inner is not None and hasattr(inner, "state_dict"):
        try:
            return inner.state_dict()
        except Exception:
            return None
    return None


def load_yolo_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_obj = ckpt.get("model")
    ema_obj = ckpt.get("ema")

    sd = _get_state_dict_from_ultralytics_model(model_obj)
    if sd is None:
        sd = _get_state_dict_from_ultralytics_model(ema_obj)
    if sd is None and isinstance(ckpt.get("state_dict"), dict):
        sd = ckpt["state_dict"]

    if sd is None:
        raise ValueError(f"Could not extract YOLO state_dict from: {path}")
    return sd


def _iter_common_float_tensors(
    sd_a: Dict[str, Any],
    sd_b: Dict[str, Any],
) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor]]:
    keys = set(sd_a.keys()) & set(sd_b.keys())
    for k in keys:
        ta = sd_a.get(k)
        tb = sd_b.get(k)
        if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
            continue
        if ta.dtype not in DTYPES_FLOAT or tb.dtype not in DTYPES_FLOAT:
            continue
        yield k, ta, tb


def compute_pairwise_stats(sd_a: Dict[str, Any], sd_b: Dict[str, Any], with_cosine: bool = True) -> PairwiseStats:
    sum_sq_a = 0.0
    sum_sq_b = 0.0
    sum_sq_diff = 0.0
    dot = 0.0
    n = 0

    for _, ta, tb in _iter_common_float_tensors(sd_a, sd_b):
        a = ta.float()
        b = tb.float()
        sum_sq_a += float(a.pow(2).sum().item())
        sum_sq_b += float(b.pow(2).sum().item())
        sum_sq_diff += float((b - a).pow(2).sum().item())
        if with_cosine:
            dot += float((a * b).sum().item())
        n += 1

    l2_a = math.sqrt(sum_sq_a) if sum_sq_a > 0 else 0.0
    l2_b = math.sqrt(sum_sq_b) if sum_sq_b > 0 else 0.0
    l2_diff = math.sqrt(sum_sq_diff) if sum_sq_diff > 0 else 0.0
    rel_diff = (l2_diff / l2_a) if l2_a > 0 else 0.0

    cosine: Optional[float] = None
    if with_cosine and l2_a > 0 and l2_b > 0:
        cosine = dot / (l2_a * l2_b)

    return PairwiseStats(
        l2_a=l2_a,
        l2_b=l2_b,
        l2_diff=l2_diff,
        rel_diff=rel_diff,
        cosine=cosine,
        num_tensors=n,
    )


def compute_drift_series(
    refs: Sequence[CheckpointRef],
    load_state_dict: Callable[[Path], Dict[str, torch.Tensor]],
    baseline_epoch: int = PHASE_BOUNDARY_EPOCH,
    max_epoch: Optional[int] = DEFAULT_MAX_EPOCH,
    label: str = "",
) -> Tuple[List[DriftPoint], Optional[CheckpointRef]]:
    if not refs:
        return [], None

    refs_sorted = [r for r in refs if max_epoch is None or r.epoch <= max_epoch]
    refs_sorted.sort(key=lambda r: r.epoch)

    baseline_ref = next((r for r in refs_sorted if r.epoch == baseline_epoch), None)
    baseline_sd: Optional[Dict[str, torch.Tensor]] = None

    points: List[DriftPoint] = []
    prev_sd: Optional[Dict[str, torch.Tensor]] = None

    for idx, ref in enumerate(refs_sorted):
        print(f"[{label}] Loading {ref.epoch}: {ref.path.name} ({idx+1}/{len(refs_sorted)})")
        sd = load_state_dict(ref.path)

        if baseline_ref and ref.epoch == baseline_ref.epoch:
            baseline_sd = sd

        update_l2: Optional[float] = None
        update_rel: Optional[float] = None
        weight_l2: float

        if prev_sd is None:
            stats_self = compute_pairwise_stats(sd, sd, with_cosine=False)
            weight_l2 = stats_self.l2_a
        else:
            stats_prev = compute_pairwise_stats(prev_sd, sd, with_cosine=False)
            weight_l2 = stats_prev.l2_b
            update_l2 = stats_prev.l2_diff
            update_rel = stats_prev.rel_diff

        dist170_l2: Optional[float] = None
        dist170_rel: Optional[float] = None
        cos170: Optional[float] = None

        if baseline_sd is not None and ref.epoch >= baseline_epoch:
            stats_base = compute_pairwise_stats(baseline_sd, sd, with_cosine=True)
            dist170_l2 = stats_base.l2_diff
            dist170_rel = stats_base.rel_diff
            cos170 = stats_base.cosine

        points.append(
            DriftPoint(
                epoch=ref.epoch,
                weight_l2=weight_l2,
                update_l2=update_l2,
                update_rel=update_rel,
                dist170_l2=dist170_l2,
                dist170_rel=dist170_rel,
                cos170=cos170,
            )
        )

        prev_sd = sd

    return points, baseline_ref


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_yolo_results_csv(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {k: [] for k in ("epoch",)}

    out: Dict[str, List[float]] = {
        "epoch": [],
        "lr_pg0": [],
        "lr_pg1": [],
        "lr_pg2": [],
        "precision": [],
        "recall": [],
        "map50": [],
        "map50_95": [],
        "train_box_loss": [],
        "train_cls_loss": [],
        "train_dfl_loss": [],
        "val_box_loss": [],
        "val_cls_loss": [],
        "val_dfl_loss": [],
    }

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = _safe_float(row.get("epoch"))
            if epoch is None:
                continue
            out["epoch"].append(int(epoch))

            def g(key: str, fallback: Optional[str] = None) -> float:
                val = _safe_float(row.get(key))
                if val is None and fallback:
                    val = _safe_float(row.get(fallback))
                return float(val) if val is not None else float("nan")

            out["lr_pg0"].append(g("lr/pg0"))
            out["lr_pg1"].append(g("lr/pg1", "lr/pg0"))
            out["lr_pg2"].append(g("lr/pg2", "lr/pg0"))
            out["precision"].append(g("metrics/precision(B)"))
            out["recall"].append(g("metrics/recall(B)"))
            out["map50"].append(g("metrics/mAP50(B)"))
            out["map50_95"].append(g("metrics/mAP50-95(B)"))
            out["train_box_loss"].append(g("train/box_loss"))
            out["train_cls_loss"].append(g("train/cls_loss"))
            out["train_dfl_loss"].append(g("train/dfl_loss"))
            out["val_box_loss"].append(g("val/box_loss"))
            out["val_cls_loss"].append(g("val/cls_loss"))
            out["val_dfl_loss"].append(g("val/dfl_loss"))

    return out


def slice_series(data: Dict[str, List[float]], min_epoch: Optional[int], max_epoch: Optional[int]) -> Dict[str, List[float]]:
    epochs = data.get("epoch", [])
    if not epochs:
        return data
    keep_idx = [i for i, e in enumerate(epochs) if (min_epoch is None or e >= min_epoch) and (max_epoch is None or e <= max_epoch)]
    return {k: [v[i] for i in keep_idx] for k, v in data.items()}


def parse_detr_finetune_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"header": {}, "epochs": []}

    header: Dict[str, Any] = {"log_path": str(path)}
    epochs: Dict[int, DetrLogEpoch] = {}

    re_image_count = re.compile(r"^Image count:\s*(?P<count>\d+)")
    re_gpus = re.compile(r"^\s*GPUs:\s*(?P<num>\d+)x\s*(?P<name>.+)$")
    re_batch = re.compile(r"^\s*Batch per GPU:\s*(?P<per>\d+)\s*\(effective batch\s*(?P<eff>\d+)\)")
    re_resume_epoch = re.compile(r"^\[Checkpoint\]\s+Successfully loaded state from epoch\s+(?P<epoch>\d+)")
    re_lr_old = re.compile(r"^\s*Old LR \(from checkpoint\):\s*(?P<lr>[0-9.eE+-]+)")
    re_lr_new = re.compile(r"^\s*New LR:\s*(?P<lr>[0-9.eE+-]+)\s*\(backbone:\s*(?P<bb>[0-9.eE+-]+)\)")
    re_sched = re.compile(r"^\[Scheduler\]\s+Re-initialized cosine scheduler for\s+(?P<rem>\d+)\s+remaining epochs")

    re_train_avg = re.compile(
        r"^Epoch\s+(?P<epoch>\d+)\s+Average Training Loss:\s*(?P<loss>[0-9.]+)\s*\(time:\s*(?P<sec>[0-9.]+)s\)"
    )
    re_grad = re.compile(r"^\s*Avg Gradient Norm:\s*(?P<gn>[0-9.]+)")
    re_val_avg = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s+Average Validation Loss:\s*(?P<loss>[0-9.]+)")

    pending_grad_epoch: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if (m := re_image_count.match(line)):
                header["image_count"] = int(m.group("count"))
                continue
            if (m := re_gpus.match(line)):
                header["gpus"] = f"{m.group('num')}x {m.group('name').strip()}"
                continue
            if (m := re_batch.match(line)):
                header["batch_per_gpu"] = int(m.group("per"))
                header["effective_batch"] = int(m.group("eff"))
                continue
            if (m := re_resume_epoch.match(line)):
                header["resume_epoch"] = int(m.group("epoch"))
                continue
            if (m := re_lr_old.match(line)):
                header["lr_old"] = float(m.group("lr"))
                continue
            if (m := re_lr_new.match(line)):
                header["lr_new"] = float(m.group("lr"))
                header["lr_backbone_new"] = float(m.group("bb"))
                continue
            if (m := re_sched.match(line)):
                header["scheduler_remaining_epochs"] = int(m.group("rem"))
                continue

            if (m := re_train_avg.match(line)):
                e = int(m.group("epoch"))
                loss = float(m.group("loss"))
                sec = float(m.group("sec"))
                epochs[e] = DetrLogEpoch(
                    epoch=e,
                    train_loss=loss,
                    val_loss=epochs.get(e).val_loss if e in epochs else None,
                    grad_norm=epochs.get(e).grad_norm if e in epochs else None,
                    epoch_time_s=sec,
                )
                pending_grad_epoch = e
                continue

            if (m := re_grad.match(line)) and pending_grad_epoch is not None:
                e = pending_grad_epoch
                gn = float(m.group("gn"))
                cur = epochs.get(e) or DetrLogEpoch(e, None, None, None, None)
                epochs[e] = DetrLogEpoch(
                    epoch=e,
                    train_loss=cur.train_loss,
                    val_loss=cur.val_loss,
                    grad_norm=gn,
                    epoch_time_s=cur.epoch_time_s,
                )
                pending_grad_epoch = None
                continue

            if (m := re_val_avg.match(line)):
                e = int(m.group("epoch"))
                loss = float(m.group("loss"))
                cur = epochs.get(e) or DetrLogEpoch(e, None, None, None, None)
                epochs[e] = DetrLogEpoch(
                    epoch=e,
                    train_loss=cur.train_loss,
                    val_loss=loss,
                    grad_norm=cur.grad_norm,
                    epoch_time_s=cur.epoch_time_s,
                )
                continue

    rows = [epochs[e] for e in sorted(epochs.keys())]
    return {"header": header, "epochs": [r.__dict__ for r in rows]}


def plot_yolo_metrics(phase1: Dict[str, List[float]], phase2: Dict[str, List[float]], out_dir: Path) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _phase_xlim() -> Optional[Tuple[int, int]]:
        all_x = (phase1.get("epoch") or []) + (phase2.get("epoch") or [])
        return (min(all_x), max(all_x)) if all_x else None

    # mAP
    ax = axes[0, 0]
    if phase1.get("epoch"):
        ax.plot(phase1["epoch"], phase1["map50_95"], color="tab:blue", label="mAP50-95 (phase1)")
        ax.plot(phase1["epoch"], phase1["map50"], color="tab:blue", linestyle=":", label="mAP50 (phase1)")
    if phase2.get("epoch"):
        ax.plot(phase2["epoch"], phase2["map50_95"], color="tab:blue", linestyle="--", label="mAP50-95 (phase2)")
        ax.plot(phase2["epoch"], phase2["map50"], color="tab:blue", linestyle="-.", label="mAP50 (phase2)")
    if (xlim := _phase_xlim()) is not None:
        _add_phase_markers(ax, xlim[0], xlim[1])
    ax.set_title("YOLO validation: mAP (higher is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Precision/Recall
    ax = axes[0, 1]
    if phase1.get("epoch"):
        ax.plot(phase1["epoch"], phase1["precision"], color="tab:green", label="Precision (phase1)")
        ax.plot(phase1["epoch"], phase1["recall"], color="tab:orange", label="Recall (phase1)")
    if phase2.get("epoch"):
        ax.plot(phase2["epoch"], phase2["precision"], color="tab:green", linestyle="--", label="Precision (phase2)")
        ax.plot(phase2["epoch"], phase2["recall"], color="tab:orange", linestyle="--", label="Recall (phase2)")
    if (xlim := _phase_xlim()) is not None:
        _add_phase_markers(ax, xlim[0], xlim[1])
    ax.set_title("YOLO validation: precision/recall (higher is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Train losses
    ax = axes[1, 0]
    if phase1.get("epoch"):
        ax.plot(phase1["epoch"], phase1["train_box_loss"], label="train box (phase1)", color="tab:red")
        ax.plot(phase1["epoch"], phase1["train_cls_loss"], label="train cls (phase1)", color="tab:purple")
        ax.plot(phase1["epoch"], phase1["train_dfl_loss"], label="train dfl (phase1)", color="tab:brown")
    if phase2.get("epoch"):
        ax.plot(phase2["epoch"], phase2["train_box_loss"], label="train box (phase2)", color="tab:red", linestyle="--")
        ax.plot(phase2["epoch"], phase2["train_cls_loss"], label="train cls (phase2)", color="tab:purple", linestyle="--")
        ax.plot(phase2["epoch"], phase2["train_dfl_loss"], label="train dfl (phase2)", color="tab:brown", linestyle="--")
    if (xlim := _phase_xlim()) is not None:
        _add_phase_markers(ax, xlim[0], xlim[1])
    ax.set_title("YOLO training losses (lower is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Val losses
    ax = axes[1, 1]
    if phase1.get("epoch"):
        ax.plot(phase1["epoch"], phase1["val_box_loss"], label="val box (phase1)", color="tab:red")
        ax.plot(phase1["epoch"], phase1["val_cls_loss"], label="val cls (phase1)", color="tab:purple")
        ax.plot(phase1["epoch"], phase1["val_dfl_loss"], label="val dfl (phase1)", color="tab:brown")
    if phase2.get("epoch"):
        ax.plot(phase2["epoch"], phase2["val_box_loss"], label="val box (phase2)", color="tab:red", linestyle="--")
        ax.plot(phase2["epoch"], phase2["val_cls_loss"], label="val cls (phase2)", color="tab:purple", linestyle="--")
        ax.plot(phase2["epoch"], phase2["val_dfl_loss"], label="val dfl (phase2)", color="tab:brown", linestyle="--")
    if (xlim := _phase_xlim()) is not None:
        _add_phase_markers(ax, xlim[0], xlim[1])
    ax.set_title("YOLO validation losses (lower is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    phase_handles = [
        mpatches.Patch(color="tab:gray", alpha=0.08, label=f"Phase 1 (~{PHASE1_DATASET_SIZE:,} images)"),
        mpatches.Patch(color="tab:orange", alpha=0.08, label=f"Phase 2 ({PHASE2_DATASET_SIZE:,} images fine-tune)"),
    ]
    fig.legend(handles=phase_handles, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(top=0.9)

    out_path = out_dir / "deep_yolo_metrics.png"
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / "deep_yolo_metrics.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_detr_finetune_log(log_data: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    epochs = log_data.get("epochs", [])
    if not epochs:
        return None

    xs = [int(r["epoch"]) for r in epochs if r.get("epoch") is not None]
    train = [r.get("train_loss") for r in epochs]
    val = [r.get("val_loss") for r in epochs]
    grad = [r.get("grad_norm") for r in epochs]
    tsec = [r.get("epoch_time_s") for r in epochs]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Losses
    ax = axes[0]
    ax.plot(xs, train, label="train loss", color="tab:red")
    ax.plot(xs, val, label="val loss", color="tab:blue")
    _add_phase_markers(ax, min(xs), max(xs))
    ax.set_title("DETR fine-tune losses (lower is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Grad norm
    ax = axes[1]
    ax.plot(xs, grad, label="avg grad norm", color="tab:purple")
    _add_phase_markers(ax, min(xs), max(xs))
    ax.set_yscale("log")
    ax.set_title("DETR fine-tune gradient norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Grad norm (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Time per epoch
    ax = axes[2]
    minutes = [(v / 60.0) if v is not None else float("nan") for v in tsec]
    ax.plot(xs, minutes, label="epoch time (min)", color="tab:green")
    _add_phase_markers(ax, min(xs), max(xs))
    ax.set_title("DETR fine-tune speed (per epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Minutes")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    phase_handles = [
        mpatches.Patch(color="tab:gray", alpha=0.08, label=f"Phase 1 (~{PHASE1_DATASET_SIZE:,} images)"),
        mpatches.Patch(color="tab:orange", alpha=0.08, label=f"Phase 2 ({PHASE2_DATASET_SIZE:,} images fine-tune)"),
    ]
    fig.legend(handles=phase_handles, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(top=0.83)

    out_path = out_dir / "deep_detr_finetune_log.png"
    plt.tight_layout(rect=(0, 0, 1, 0.8))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / "deep_detr_finetune_log.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_weight_drift(
    detr_points: List[DriftPoint],
    yolo_points: List[DriftPoint],
    out_dir: Path,
) -> Optional[Path]:
    if not detr_points and not yolo_points:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Update magnitude (relative) vs epoch
    ax = axes[0]
    if detr_points:
        xs = [p.epoch for p in detr_points if p.update_rel is not None]
        ys = [p.update_rel for p in detr_points if p.update_rel is not None]
        ax.plot(xs, ys, marker="o", label="DETR dW/||w|| (checkpoint-to-checkpoint)", color="tab:red")
    if yolo_points:
        xs = [p.epoch for p in yolo_points if p.update_rel is not None]
        ys = [p.update_rel for p in yolo_points if p.update_rel is not None]
        ax.plot(xs, ys, marker="s", label="YOLO dW/||w|| (checkpoint-to-checkpoint)", color="tab:blue")
    all_x = ([p.epoch for p in detr_points] if detr_points else []) + ([p.epoch for p in yolo_points] if yolo_points else [])
    if all_x:
        _add_phase_markers(ax, min(all_x), max(all_x))
    ax.set_yscale("log")
    ax.set_title("Relative update size (log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("dW / ||w||")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # Distance from epoch170 (relative)
    ax = axes[1]
    if detr_points:
        xs = [p.epoch for p in detr_points if p.dist170_rel is not None]
        ys = [p.dist170_rel for p in detr_points if p.dist170_rel is not None]
        ax.plot(xs, ys, marker="o", label="DETR ||w - w170||/||w170||", color="tab:red")
    if yolo_points:
        xs = [p.epoch for p in yolo_points if p.dist170_rel is not None]
        ys = [p.dist170_rel for p in yolo_points if p.dist170_rel is not None]
        ax.plot(xs, ys, marker="s", label="YOLO ||w - w170||/||w170||", color="tab:blue")
    all_x = ([p.epoch for p in detr_points] if detr_points else []) + ([p.epoch for p in yolo_points] if yolo_points else [])
    if all_x:
        _add_phase_markers(ax, min(all_x), max(all_x))
    ax.set_yscale("log")
    ax.set_title("Drift from epoch170 (log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||w - w170|| / ||w170||")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    phase_handles = [
        mpatches.Patch(color="tab:gray", alpha=0.08, label=f"Phase 1 (~{PHASE1_DATASET_SIZE:,} images)"),
        mpatches.Patch(color="tab:orange", alpha=0.08, label=f"Phase 2 ({PHASE2_DATASET_SIZE:,} images fine-tune)"),
    ]
    fig.legend(handles=phase_handles, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(top=0.82)

    out_path = out_dir / "deep_weight_drift.png"
    plt.tight_layout(rect=(0, 0, 1, 0.78))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / "deep_weight_drift.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def _detr_group_key(param_name: str) -> str:
    if param_name.startswith("model.backbone"):
        return "backbone"
    if param_name.startswith("model.encoder"):
        return "encoder"
    if param_name.startswith("model.decoder"):
        return "decoder"
    if param_name.startswith("class_labels_classifier"):
        return "cls_head"
    if param_name.startswith("bbox_predictor"):
        return "bbox_head"
    return "other"


_re_yolo_layer = re.compile(r"^model\.(?P<idx>\d+)\.")


def _yolo_group_key(param_name: str) -> str:
    m = _re_yolo_layer.match(param_name)
    if m:
        return f"layer_{m.group('idx')}"
    return "other"


def compute_group_drift(
    sd_base: Dict[str, Any],
    sd_final: Dict[str, Any],
    group_fn: Callable[[str], str],
    top_k: int = 15,
) -> List[Tuple[str, float]]:
    sum_sq_base: Dict[str, float] = {}
    sum_sq_diff: Dict[str, float] = {}

    for k, ta, tb in _iter_common_float_tensors(sd_base, sd_final):
        g = group_fn(k)
        a = ta.float()
        b = tb.float()
        sum_sq_base[g] = sum_sq_base.get(g, 0.0) + float(a.pow(2).sum().item())
        sum_sq_diff[g] = sum_sq_diff.get(g, 0.0) + float((b - a).pow(2).sum().item())

    rows: List[Tuple[str, float]] = []
    for g in sorted(sum_sq_base.keys()):
        base = math.sqrt(sum_sq_base[g]) if sum_sq_base[g] > 0 else 0.0
        diff = math.sqrt(sum_sq_diff.get(g, 0.0)) if sum_sq_diff.get(g, 0.0) > 0 else 0.0
        rel = diff / base if base > 0 else 0.0
        rows.append((g, rel))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def plot_group_drift(
    detr_groups: List[Tuple[str, float]],
    yolo_groups: List[Tuple[str, float]],
    out_dir: Path,
) -> Optional[Path]:
    if not detr_groups and not yolo_groups:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if detr_groups:
        labels = [g for g, _ in detr_groups]
        vals = [v for _, v in detr_groups]
        ax.barh(labels[::-1], vals[::-1], color="tab:red", alpha=0.8)
    ax.set_title("DETR: top group drift (170-200)")
    ax.set_xlabel("||dW|| / ||w170||")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    if yolo_groups:
        labels = [g for g, _ in yolo_groups]
        vals = [v for _, v in yolo_groups]
        ax.barh(labels[::-1], vals[::-1], color="tab:blue", alpha=0.8)
    ax.set_title("YOLO: top layer drift (170-200)")
    ax.set_xlabel("||dW|| / ||w170||")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)

    out_path = out_dir / "deep_group_drift.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / "deep_group_drift.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_report(
    out_dir: Path,
    max_epoch: int,
    detr_drift: List[DriftPoint],
    yolo_drift: List[DriftPoint],
    detr_log: Dict[str, Any],
    yolo1_csv: Dict[str, List[float]],
    yolo2_csv: Dict[str, List[float]],
    plot_paths: Dict[str, Path],
    group_drift_detr: List[Tuple[str, float]],
    group_drift_yolo: List[Tuple[str, float]],
) -> Path:
    def _fmt(x: Optional[float], nd: int = 4) -> str:
        if x is None or not math.isfinite(x):
            return "N/A"
        return f"{x:.{nd}g}" if abs(x) >= 1000 else f"{x:.{nd}f}"

    def _last(points: List[DriftPoint]) -> Optional[DriftPoint]:
        pts = [p for p in points if p.epoch <= max_epoch]
        return pts[-1] if pts else None

    detr_last = _last(detr_drift)
    yolo_last = _last(yolo_drift)

    lines: List[str] = []
    lines.append("# Deep DETR vs YOLO training analysis (two-phase)")
    lines.append("")
    lines.append("## Phase protocol (critical for interpretation)")
    lines.append(f"- Phase 1: epoch <= {PHASE_BOUNDARY_EPOCH}, dataset ~{PHASE1_DATASET_SIZE:,} images (augmented pool)")
    lines.append(f"- Phase 2: epoch > {PHASE_BOUNDARY_EPOCH}, dataset {PHASE2_DATASET_SIZE:,} images (fine-tune subset)")
    lines.append(f"- Focus: epoch <= {max_epoch} for direct DETR vs YOLO comparability (YOLO ended at 200).")
    lines.append("")

    lines.append("## Plots")
    for k, p in plot_paths.items():
        lines.append(f"- {k}: `{p.resolve()}`")
    lines.append("")

    lines.append("## Plot reading guide")
    lines.append("- `deep_yolo_metrics.png`: YOLO mAP/precision/recall + train/val losses; watch the phase boundary at epoch 170.")
    lines.append("- `deep_detr_finetune_log.png`: DETR fine-tune train/val loss + grad norm + epoch time (parsed from the training log).")
    lines.append("- `deep_weight_drift.png`: weight-space dynamics; left=relative checkpoint update size, right=drift from epoch170 baseline.")
    lines.append("- `deep_group_drift.png`: where the weights change most during fine-tune (170->200); DETR groups vs YOLO layer indices.")
    lines.append("")

    lines.append("## Weight-space drift summary (epoch170 -> end)")
    lines.append("| Model | End epoch | ||w_end - w170|| / ||w170|| | cos(w_end, w170) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| DETR | {detr_last.epoch if detr_last else 'N/A'} | {_fmt(detr_last.dist170_rel) if detr_last else 'N/A'} | {_fmt(detr_last.cos170) if detr_last else 'N/A'} |"
    )
    lines.append(
        f"| YOLO | {yolo_last.epoch if yolo_last else 'N/A'} | {_fmt(yolo_last.dist170_rel) if yolo_last else 'N/A'} | {_fmt(yolo_last.cos170) if yolo_last else 'N/A'} |"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `||w_end - w170|| / ||w170||` measures how much parameters moved during fine-tune (phase2).")
    lines.append("- `cos(w_end, w170)` close to 1.0 means the final weights are directionally very similar to epoch170.")
    lines.append("")

    def _median(vals: List[float]) -> Optional[float]:
        xs = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if not xs:
            return None
        xs.sort()
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])

    def _phase_updates(points: List[DriftPoint], phase: int) -> List[float]:
        out: List[float] = []
        for p in points:
            if p.update_rel is None:
                continue
            if phase == 1 and p.epoch <= PHASE_BOUNDARY_EPOCH:
                out.append(p.update_rel)
            if phase == 2 and p.epoch > PHASE_BOUNDARY_EPOCH:
                out.append(p.update_rel)
        return out

    detr_u1 = _median(_phase_updates(detr_drift, 1))
    detr_u2 = _median(_phase_updates(detr_drift, 2))
    yolo_u1 = _median(_phase_updates(yolo_drift, 1))
    yolo_u2 = _median(_phase_updates(yolo_drift, 2))

    lines.append("## Checkpoint-to-checkpoint update size (median dW/||w||)")
    lines.append("| Model | Phase 1 (<=170) | Phase 2 (>170) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| DETR | {_fmt(detr_u1)} | {_fmt(detr_u2)} |")
    lines.append(f"| YOLO | {_fmt(yolo_u1)} | {_fmt(yolo_u2)} |")
    lines.append("")

    def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)) or b == 0:
            return None
        return a / b

    lines.append("## Key takeaways (training dynamics)")
    r_detr = _ratio(detr_u2, detr_u1)
    r_yolo = _ratio(yolo_u2, yolo_u1)
    inv_detr = (1.0 / r_detr) if (r_detr is not None and r_detr > 0) else None
    inv_yolo = (1.0 / r_yolo) if (r_yolo is not None and r_yolo > 0) else None
    lines.append(
        f"- Phase switch shrinks median dW/||w|| to ~{_fmt(r_detr)}x of phase1 (~{_fmt(inv_detr)}x smaller) for DETR, "
        f"and to ~{_fmt(r_yolo)}x of phase1 (~{_fmt(inv_yolo)}x smaller) for YOLO."
    )
    if detr_last and yolo_last:
        lines.append(
            f"- Fine-tune drift (170->200): DETR `{_fmt(detr_last.dist170_rel)}` vs YOLO `{_fmt(yolo_last.dist170_rel)}` in ||w_end-w170||/||w170|| (both cosine ~1.0)."
        )
    if group_drift_detr:
        lines.append(f"- DETR fine-tune concentrates updates in: `{group_drift_detr[0][0]}` (top drift `{_fmt(group_drift_detr[0][1])}`).")
    if group_drift_yolo:
        lines.append(f"- YOLO fine-tune shows strongest drift in: `{group_drift_yolo[0][0]}` (top drift `{_fmt(group_drift_yolo[0][1])}`).")
    lines.append("")

    if group_drift_detr or group_drift_yolo:
        lines.append("## Where the model changes during fine-tune (170 -> 200)")
        lines.append("")
        if group_drift_detr:
            lines.append("**DETR top groups (relative drift):**")
            for g, v in group_drift_detr:
                lines.append(f"- {g}: {_fmt(v)}")
            lines.append("")
        if group_drift_yolo:
            lines.append("**YOLO top layers (relative drift):**")
            for g, v in group_drift_yolo:
                lines.append(f"- {g}: {_fmt(v)}")
            lines.append("")

    header = detr_log.get("header", {}) if isinstance(detr_log, dict) else {}
    if header:
        lines.append("## DETR fine-tune run details (from log)")
        lines.append(f"- Image count: `{header.get('image_count', 'N/A')}`")
        lines.append(f"- GPUs: `{header.get('gpus', 'N/A')}`")
        lines.append(
            f"- Batch per GPU: `{header.get('batch_per_gpu', 'N/A')}` (effective `{header.get('effective_batch', 'N/A')}`)"
        )
        if header.get("resume_epoch") is not None:
            lines.append(f"- Resume epoch detected: `{header.get('resume_epoch')}`")
        if header.get("lr_old") is not None or header.get("lr_new") is not None:
            lines.append(
                f"- LR reset: old `{header.get('lr_old', 'N/A')}` -> new `{header.get('lr_new', 'N/A')}` (backbone `{header.get('lr_backbone_new', 'N/A')}`)"
            )
        if header.get("scheduler_remaining_epochs") is not None:
            lines.append(f"- Cosine scheduler re-init: `{header.get('scheduler_remaining_epochs')}` remaining epochs")
        lines.append("")

    log_epochs = detr_log.get("epochs", []) if isinstance(detr_log, dict) else []
    if log_epochs:
        def _finite(x: Any) -> Optional[float]:
            if not isinstance(x, (int, float)) or not math.isfinite(x):
                return None
            return float(x)

        start = log_epochs[0]
        end = log_epochs[-1]
        start_e = int(start.get("epoch"))
        end_e = int(end.get("epoch"))
        start_tr = _finite(start.get("train_loss"))
        start_vl = _finite(start.get("val_loss"))
        end_tr = _finite(end.get("train_loss"))
        end_vl = _finite(end.get("val_loss"))

        best_e: Optional[int] = None
        best_v: Optional[float] = None
        best_tr: Optional[float] = None
        for row in log_epochs:
            e = int(row.get("epoch"))
            v = _finite(row.get("val_loss"))
            if v is None:
                continue
            if best_v is None or v < best_v:
                best_v = v
                best_e = e
                best_tr = _finite(row.get("train_loss"))

        grad_vals = [_finite(r.get("grad_norm")) for r in log_epochs]
        grad_vals = [v for v in grad_vals if v is not None]
        time_vals = [_finite(r.get("epoch_time_s")) for r in log_epochs]
        time_vals = [v for v in time_vals if v is not None]

        lines.append("## DETR fine-tune loss dynamics (from log)")
        lines.append(f"- Log coverage: epoch `{start_e}`..`{end_e}` (n={len(log_epochs)})")
        lines.append(f"- Start: epoch `{start_e}` train `{_fmt(start_tr)}` / val `{_fmt(start_vl)}`")
        lines.append(f"- Best val loss: `{_fmt(best_v)}` @ epoch `{best_e}` (train `{_fmt(best_tr)}`)")
        lines.append(f"- Last parsed: epoch `{end_e}` train `{_fmt(end_tr)}` / val `{_fmt(end_vl)}`")
        if grad_vals:
            lines.append(f"- Grad norm: median `{_fmt(_median(grad_vals))}`, min `{_fmt(min(grad_vals))}`, max `{_fmt(max(grad_vals))}`")
        if time_vals:
            mean_min = sum(time_vals) / len(time_vals) / 60.0
            lines.append(f"- Epoch time: mean `{_fmt(mean_min)}` minutes (n={len(time_vals)})")
        lines.append("")

    def _best_epoch(key: str, data: Dict[str, List[float]]) -> Tuple[Optional[int], Optional[float]]:
        xs = data.get("epoch") or []
        ys = data.get(key) or []
        best_e: Optional[int] = None
        best_v: Optional[float] = None
        for e, v in zip(xs, ys):
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                continue
            if best_v is None or float(v) > best_v:
                best_v = float(v)
                best_e = int(e)
        return best_e, best_v

    def _value_at_epoch(key: str, epoch: int, data: Dict[str, List[float]]) -> Optional[float]:
        xs = data.get("epoch") or []
        ys = data.get(key) or []
        for e, v in zip(xs, ys):
            if int(e) != int(epoch):
                continue
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                return None
            return float(v)
        return None

    if yolo1_csv.get("epoch"):
        best_e, best_v = _best_epoch("map50_95", yolo1_csv)
        lines.append("## YOLO phase1 headline (epoch <= 170)")
        lines.append(f"- Best mAP50-95: `{_fmt(best_v)}` @ epoch `{best_e}`")
        lines.append("")

    if yolo2_csv.get("epoch"):
        best_e2, best_v2 = _best_epoch("map50_95", yolo2_csv)
        base170 = _value_at_epoch("map50_95", PHASE_BOUNDARY_EPOCH, yolo1_csv)
        gain = (best_v2 - base170) if (best_v2 is not None and base170 is not None) else None

        best_e50, best_v50 = _best_epoch("map50", yolo2_csv)
        base170_50 = _value_at_epoch("map50", PHASE_BOUNDARY_EPOCH, yolo1_csv)
        gain50 = (best_v50 - base170_50) if (best_v50 is not None and base170_50 is not None) else None

        _, best_p = _best_epoch("precision", yolo2_csv)
        _, best_r = _best_epoch("recall", yolo2_csv)

        lines.append("## YOLO fine-tune headline (from results.csv)")
        lines.append(f"- Epoch range: `{yolo2_csv['epoch'][0]}`..`{yolo2_csv['epoch'][-1]}`")
        lines.append(f"- Peak mAP50-95: `{_fmt(best_v2)}` @ epoch `{best_e2}` (gain vs ep170: `{_fmt(gain)}`)")
        lines.append(f"- Peak mAP50: `{_fmt(best_v50)}` @ epoch `{best_e50}` (gain vs ep170: `{_fmt(gain50)}`)")
        lines.append(f"- Peak precision: `{_fmt(best_p)}`")
        lines.append(f"- Peak recall: `{_fmt(best_r)}`")
        lines.append("")

    report_path = out_dir / "deep_training_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep DETR vs YOLO training analysis (two-phase schedule).")
    parser.add_argument("--detr-phase1-dir", type=str, default=str(DEFAULT_DETR_PHASE1_DIR))
    parser.add_argument("--detr-phase2-dir", type=str, default=str(DEFAULT_DETR_PHASE2_DIR))
    parser.add_argument("--detr-phase2-log", type=str, default=str(DEFAULT_DETR_PHASE2_LOG))
    parser.add_argument("--yolo-phase1-dir", type=str, default=str(DEFAULT_YOLO_PHASE1_DIR))
    parser.add_argument("--yolo-phase2-dir", type=str, default=str(DEFAULT_YOLO_PHASE2_DIR))
    parser.add_argument("--yolo-phase1-csv", type=str, default=str(DEFAULT_YOLO_PHASE1_CSV))
    parser.add_argument("--yolo-phase2-csv", type=str, default=str(DEFAULT_YOLO_PHASE2_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epoch = int(args.max_epoch)

    # YOLO metrics
    yolo1 = slice_series(load_yolo_results_csv(Path(args.yolo_phase1_csv)), 1, PHASE_BOUNDARY_EPOCH)
    yolo2 = slice_series(load_yolo_results_csv(Path(args.yolo_phase2_csv)), PHASE_BOUNDARY_EPOCH, max_epoch)
    yolo_metrics_plot = plot_yolo_metrics(yolo1, yolo2, out_dir)

    # DETR log metrics (fine-tune)
    detr_log = parse_detr_finetune_log(Path(args.detr_phase2_log))
    detr_log_plot = plot_detr_finetune_log(detr_log, out_dir)

    # Weight drift: DETR (phase1 + phase2)
    detr_refs: List[CheckpointRef] = []
    detr_refs += discover_checkpoints(Path(args.detr_phase1_dir), "checkpoint_epoch_", ".pth", min_epoch=0, max_epoch=PHASE_BOUNDARY_EPOCH)
    detr_refs += discover_checkpoints(Path(args.detr_phase2_dir), "checkpoint_epoch_", ".pth", min_epoch=PHASE_BOUNDARY_EPOCH, max_epoch=max_epoch)
    # De-duplicate epochs (phase2 overrides phase1 on duplicates, e.g., epoch170).
    detr_by_epoch: Dict[int, CheckpointRef] = {}
    for r in detr_refs:
        detr_by_epoch[r.epoch] = r
    detr_refs = [detr_by_epoch[e] for e in sorted(detr_by_epoch.keys())]
    detr_points, detr_base_ref = compute_drift_series(
        detr_refs,
        load_state_dict=load_detr_state_dict,
        baseline_epoch=PHASE_BOUNDARY_EPOCH,
        max_epoch=max_epoch,
        label="DETR",
    )

    # Weight drift: YOLO (phase1 <=170, phase2 from separate dir; include last.pt as epoch=max_epoch)
    yolo_phase1_refs = discover_checkpoints(Path(args.yolo_phase1_dir), "epoch", ".pt", min_epoch=0, max_epoch=PHASE_BOUNDARY_EPOCH)
    yolo_phase2_refs = discover_checkpoints(
        Path(args.yolo_phase2_dir),
        "epoch",
        ".pt",
        min_epoch=PHASE_BOUNDARY_EPOCH,
        max_epoch=max_epoch,
        extra_named=[("last.pt", max_epoch)],
    )
    yolo_refs = list(yolo_phase1_refs) + [r for r in yolo_phase2_refs if r.epoch > PHASE_BOUNDARY_EPOCH]
    yolo_refs.sort(key=lambda r: r.epoch)
    yolo_points, yolo_base_ref = compute_drift_series(
        yolo_refs,
        load_state_dict=load_yolo_state_dict,
        baseline_epoch=PHASE_BOUNDARY_EPOCH,
        max_epoch=max_epoch,
        label="YOLO",
    )

    weight_drift_plot = plot_weight_drift(detr_points, yolo_points, out_dir)

    # Group drift (170 -> max_epoch)
    group_drift_detr: List[Tuple[str, float]] = []
    group_drift_yolo: List[Tuple[str, float]] = []
    group_drift_plot: Optional[Path] = None

    if detr_base_ref and any(r.epoch == max_epoch for r in detr_refs):
        detr_final_ref = next(r for r in detr_refs if r.epoch == max_epoch)
        print("[DETR] Loading baseline/final for group drift...")
        sd_base = load_detr_state_dict(detr_base_ref.path)
        sd_final = load_detr_state_dict(detr_final_ref.path)
        group_drift_detr = compute_group_drift(sd_base, sd_final, _detr_group_key, top_k=10)

    if yolo_base_ref and any(r.epoch == max_epoch for r in yolo_refs):
        yolo_final_ref = next(r for r in yolo_refs if r.epoch == max_epoch)
        print("[YOLO] Loading baseline/final for group drift...")
        sd_base = load_yolo_state_dict(yolo_base_ref.path)
        sd_final = load_yolo_state_dict(yolo_final_ref.path)
        group_drift_yolo = compute_group_drift(sd_base, sd_final, _yolo_group_key, top_k=15)

    group_drift_plot = plot_group_drift(group_drift_detr, group_drift_yolo, out_dir)

    plot_paths: Dict[str, Path] = {"YOLO metrics (mAP/PR + losses)": yolo_metrics_plot}
    if detr_log_plot:
        plot_paths["DETR fine-tune (loss/grad/time)"] = detr_log_plot
    if weight_drift_plot:
        plot_paths["Weight drift (dW and distance to epoch170)"] = weight_drift_plot
    if group_drift_plot:
        plot_paths["Group drift (where weights change)"] = group_drift_plot
    overview_plot = out_dir / "detr_vs_yolo_comparison.png"
    if overview_plot.exists():
        plot_paths["Overview (LR/weights/loss/mAP)"] = overview_plot
    layer_plot = out_dir / "detr_layer_evolution_compare.png"
    if layer_plot.exists():
        plot_paths["DETR layer evolution (selected layers)"] = layer_plot

    report_path = build_report(
        out_dir=out_dir,
        max_epoch=max_epoch,
        detr_drift=detr_points,
        yolo_drift=yolo_points,
        detr_log=detr_log,
        yolo1_csv=yolo1,
        yolo2_csv=yolo2,
        plot_paths=plot_paths,
        group_drift_detr=group_drift_detr,
        group_drift_yolo=group_drift_yolo,
    )

    print(f"[OK] Report: {report_path}")
    for k, p in plot_paths.items():
        print(f"[OK] Plot: {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
