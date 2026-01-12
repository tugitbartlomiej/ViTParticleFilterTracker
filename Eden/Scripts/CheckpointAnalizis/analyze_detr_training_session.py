"""
DETR Training Session Analysis Script

Analyzes a single DETR training session folder that contains:
- checkpoints (*.pth)
- TensorBoard event logs (events.out.tfevents.*)
- a plain-text training log (*.log)

Outputs:
- JSON summary
- Markdown report
- plots (PNG/PDF)

Run (on this machine):
  py -3.11 Eden\\Scripts\\CheckpointAnalizis\\analyze_detr_training_session.py --session-dir \"<PATH>\"
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"[ERROR] matplotlib is required: {exc}") from exc

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] torch is required. Try running with Python 3.11:\n"
        "  py -3.11 Eden\\Scripts\\CheckpointAnalizis\\analyze_detr_training_session.py ..."
    ) from exc

try:
    from tensorboard.compat.proto import event_pb2
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] tensorboard protobufs are required. Install tensorboard or use the configured env."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_SESSION_DIR = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "detr_sessions"


@dataclass
class ScalarPoint:
    step: int
    wall_time: float
    value: float
    file: str


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "N/A"
    seconds_int = int(round(seconds))
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def discover_session(session_dir: Path) -> Dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    if not session_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {session_dir}")

    log_files = sorted(session_dir.glob("*.log"))
    ckpt_dirs = sorted([p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt_")])
    train_out_dirs = sorted([p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("train_out_")])

    tb_log_dirs: List[Path] = []
    final_model_dirs: List[Path] = []
    for out_dir in train_out_dirs:
        tb_dir = out_dir / "logs"
        if tb_dir.exists():
            tb_log_dirs.append(tb_dir)
        fm_dir = out_dir / "final_model"
        if fm_dir.exists():
            final_model_dirs.append(fm_dir)

    checkpoint_files: List[Path] = []
    for d in ckpt_dirs:
        checkpoint_files.extend(sorted(d.glob("*.pth")))

    return {
        "session_dir": session_dir,
        "log_files": log_files,
        "ckpt_dirs": ckpt_dirs,
        "train_out_dirs": train_out_dirs,
        "tb_log_dirs": tb_log_dirs,
        "final_model_dirs": final_model_dirs,
        "checkpoint_files": checkpoint_files,
    }


def parse_namespace_from_log_line(line: str) -> Tuple[Optional[str], Dict[str, Any]]:
    m = re.search(r"Namespace\(.*\)$", line.strip())
    if not m:
        return None, {}
    ns_text = m.group(0)
    try:
        node = ast.parse(ns_text, mode="eval").body
        if not isinstance(node, ast.Call):
            return ns_text, {}
        kwargs: Dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                kwargs[kw.arg] = ast.unparse(kw.value) if hasattr(ast, "unparse") else str(kw.value)
        return ns_text, kwargs
    except Exception:
        return ns_text, {}


def parse_training_log(log_path: Path) -> Dict[str, Any]:
    log_path = Path(log_path)

    header: Dict[str, Any] = {"log_path": str(log_path)}
    epochs: Dict[int, Dict[str, Any]] = {}
    progress: Dict[int, Dict[str, Any]] = {}

    # Header patterns
    re_job = re.compile(r"^Job ID:\s*(?P<job>\S+)")
    re_start = re.compile(r"^Start Time:\s*(?P<start>.+)$")
    re_node = re.compile(r"^Node:\s*(?P<node>\S+)")
    re_images = re.compile(r"^Image count:\s*(?P<count>\d+)")
    re_gpus = re.compile(r"^\s*GPUs:\s*(?P<num>\d+)x\s*(?P<name>.+)$")
    re_batch = re.compile(r"^\s*Batch per GPU:\s*(?P<per>\d+)\s*\(effective batch\s*(?P<eff>\d+)\)")
    re_mp = re.compile(r"^\s*Mixed Precision:\s*(?P<mp>.+)$")
    re_args = re.compile(r"^Arguments passed or defaults used:\s*(?P<ns>Namespace\(.*\))\s*$")
    re_running_gpus = re.compile(r"^Running on\s+(?P<num>\d+)\s+GPUs")

    # Events around resume / LR reset
    re_resume_found = re.compile(r"^\[Checkpoint\]\s+Found\s+(?P<file>\S+)\s+-\s+using it for resume")
    re_resume_load = re.compile(r"^\[Checkpoint\]\s+Loading checkpoint from:\s*(?P<path>.+)$")
    re_resume_epoch = re.compile(r"^\[Checkpoint\]\s+Successfully loaded state from epoch\s+(?P<epoch>\d+)")
    re_lr_reset = re.compile(r"^\[LR Reset\]\s+Resetting learning rates")
    re_lr_old = re.compile(r"^\s*Old LR \(from checkpoint\):\s*(?P<lr>[0-9.eE+-]+)")
    re_lr_new = re.compile(r"^\s*New LR:\s*(?P<lr>[0-9.eE+-]+)\s*\(backbone:\s*(?P<bb>[0-9.eE+-]+)\)")
    re_sched_reinit = re.compile(
        r"^\[Scheduler\]\s+Re-initialized cosine scheduler for\s+(?P<rem>\d+)\s+remaining epochs"
    )
    re_start_loop = re.compile(r"^Starting training loop from epoch\s+(?P<epoch>\d+)")

    # Per-epoch patterns
    re_epoch_hdr = re.compile(r"^--- Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+---")
    re_gpu_mem = re.compile(
        r"^\s*GPU Memory:\s*(?P<alloc>[0-9.]+)GB allocated,\s*(?P<res>[0-9.]+)GB reserved"
    )
    re_train_avg = re.compile(
        r"^Epoch\s+(?P<epoch>\d+)\s+Average Training Loss:\s*(?P<loss>[0-9.]+)\s*\(time:\s*(?P<sec>[0-9.]+)s\)"
    )
    re_grad = re.compile(r"^\s*Avg Gradient Norm:\s*(?P<gn>[0-9.]+)")
    re_val_avg = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s+Average Validation Loss:\s*(?P<loss>[0-9.]+)")
    re_val_improve = re.compile(
        r"^Validation loss improved from\s+(?P<from>[^\s]+)\s+to\s+(?P<to>[0-9.]+)\."
    )
    re_ckpt_saved = re.compile(r"^Checkpoint saved to\s+(?P<path>.+)$")

    # Progress-bar tail (detect interrupted training)
    re_train_prog = re.compile(
        r"^Training\s+E(?P<epoch>\d+):\s*(?P<pct>\d+)%.*\|\s*(?P<it>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[0-9:]+)<(?P<remain>[0-9:]+),\s*(?P<s_it>[0-9.]+)s/it,\s*loss=(?P<loss>[0-9.]+)\]"
    )

    current_epoch: Optional[int] = None
    last_seen_epoch_for_summary: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Header
            if (m := re_job.match(line)):
                header["job_id"] = m.group("job")
                continue
            if (m := re_start.match(line)):
                header["start_time_raw"] = m.group("start").strip()
                continue
            if (m := re_node.match(line)):
                header["node"] = m.group("node")
                continue
            if (m := re_images.match(line)):
                header["image_count"] = int(m.group("count"))
                continue
            if (m := re_gpus.match(line)):
                header["gpus"] = {"count": int(m.group("num")), "name": m.group("name").strip()}
                continue
            if (m := re_batch.match(line)):
                header["batch_per_gpu"] = int(m.group("per"))
                header["effective_batch"] = int(m.group("eff"))
                continue
            if (m := re_mp.match(line)):
                header["mixed_precision"] = m.group("mp").strip()
                continue
            if (m := re_running_gpus.match(line)):
                header["running_gpus"] = int(m.group("num"))
                continue
            if (m := re_args.match(line)):
                ns_text = m.group("ns")
                _, ns_dict = parse_namespace_from_log_line(ns_text)
                header["args_raw"] = ns_text
                header["args"] = ns_dict
                continue

            # Resume / LR reset
            if (m := re_resume_found.match(line)):
                header.setdefault("resume", {})["found_file"] = m.group("file")
                continue
            if (m := re_resume_load.match(line)):
                header.setdefault("resume", {})["loaded_path"] = m.group("path").strip()
                continue
            if (m := re_resume_epoch.match(line)):
                header.setdefault("resume", {})["loaded_epoch"] = int(m.group("epoch"))
                continue
            if re_lr_reset.match(line):
                header.setdefault("lr_reset", {})["present"] = True
                continue
            if (m := re_lr_old.match(line)):
                header.setdefault("lr_reset", {})["old_lr"] = _safe_float(m.group("lr"))
                continue
            if (m := re_lr_new.match(line)):
                header.setdefault("lr_reset", {})["new_lr"] = _safe_float(m.group("lr"))
                header.setdefault("lr_reset", {})["new_lr_backbone"] = _safe_float(m.group("bb"))
                continue
            if (m := re_sched_reinit.match(line)):
                header.setdefault("lr_reset", {})["cosine_remaining_epochs"] = int(m.group("rem"))
                continue
            if (m := re_start_loop.match(line)):
                header["start_loop_from_epoch"] = int(m.group("epoch"))
                continue

            # Epoch header
            if (m := re_epoch_hdr.match(line)):
                current_epoch = int(m.group("epoch"))
                header["total_epochs"] = int(m.group("total"))
                epochs.setdefault(current_epoch, {"epoch": current_epoch})
                continue

            # GPU memory line near epoch header
            if current_epoch is not None and (m := re_gpu_mem.match(line)):
                epochs.setdefault(current_epoch, {"epoch": current_epoch})
                epochs[current_epoch]["gpu_alloc_gb"] = _safe_float(m.group("alloc"))
                epochs[current_epoch]["gpu_reserved_gb"] = _safe_float(m.group("res"))
                continue

            # Epoch summaries
            if (m := re_train_avg.match(line)):
                ep = int(m.group("epoch"))
                epochs.setdefault(ep, {"epoch": ep})
                epochs[ep]["train_loss"] = _safe_float(m.group("loss"))
                epochs[ep]["train_time_s"] = _safe_float(m.group("sec"))
                last_seen_epoch_for_summary = ep
                continue

            if (m := re_grad.match(line)):
                if last_seen_epoch_for_summary is not None:
                    epochs.setdefault(last_seen_epoch_for_summary, {"epoch": last_seen_epoch_for_summary})
                    epochs[last_seen_epoch_for_summary]["grad_norm"] = _safe_float(m.group("gn"))
                continue

            if (m := re_val_avg.match(line)):
                ep = int(m.group("epoch"))
                epochs.setdefault(ep, {"epoch": ep})
                epochs[ep]["val_loss"] = _safe_float(m.group("loss"))
                continue

            if (m := re_val_improve.match(line)):
                if last_seen_epoch_for_summary is not None:
                    ep = last_seen_epoch_for_summary
                    epochs.setdefault(ep, {"epoch": ep})
                    from_raw = m.group("from")
                    from_val = None if from_raw.lower() == "inf" else _safe_float(from_raw)
                    epochs[ep]["val_improved_from"] = from_val
                    epochs[ep]["val_improved_to"] = _safe_float(m.group("to"))
                continue

            if (m := re_ckpt_saved.match(line)):
                if last_seen_epoch_for_summary is not None:
                    ep = last_seen_epoch_for_summary
                    epochs.setdefault(ep, {"epoch": ep})
                    epochs[ep].setdefault("saved_checkpoints", []).append(m.group("path").strip())
                continue

            if (m := re_train_prog.match(line)):
                ep = int(m.group("epoch"))
                progress[ep] = {
                    "epoch": ep,
                    "pct": int(m.group("pct")),
                    "iter": int(m.group("it")),
                    "total": int(m.group("total")),
                    "elapsed": m.group("elapsed"),
                    "remain": m.group("remain"),
                    "sec_per_it": _safe_float(m.group("s_it")),
                    "loss": _safe_float(m.group("loss")),
                }
                continue

    epoch_list = [epochs[k] for k in sorted(epochs)]
    header["epochs_parsed"] = len(epoch_list)
    header["progress_epochs"] = len(progress)

    if progress:
        last_prog_epoch = max(progress.keys())
        if last_prog_epoch not in epochs or epochs[last_prog_epoch].get("train_loss") is None:
            header["incomplete_epoch"] = progress[last_prog_epoch]

    return {"header": header, "epochs": epoch_list, "progress": progress}


def iter_tfrecord_records(path: Path) -> Iterable[bytes]:
    # TFRecord format:
    #   uint64 length
    #   uint32 masked_crc(length)
    #   byte[length] data
    #   uint32 masked_crc(data)
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            _ = f.read(4)
            data = f.read(length)
            _ = f.read(4)
            if len(data) != length:
                return
            yield data


def tensorproto_to_scalar(tensor) -> Optional[float]:
    # Most scalar summaries use float32/float64.
    if getattr(tensor, "float_val", None):
        if len(tensor.float_val):
            return float(tensor.float_val[0])
    if getattr(tensor, "double_val", None):
        if len(tensor.double_val):
            return float(tensor.double_val[0])
    if getattr(tensor, "int_val", None):
        if len(tensor.int_val):
            return float(tensor.int_val[0])
    if getattr(tensor, "int64_val", None):
        if len(tensor.int64_val):
            return float(tensor.int64_val[0])

    raw = getattr(tensor, "tensor_content", b"")
    if not raw:
        return None

    dtype = int(getattr(tensor, "dtype", 0))
    try:
        if dtype == 1:  # DT_FLOAT
            return float(struct.unpack("<f", raw[:4])[0])
        if dtype == 2:  # DT_DOUBLE
            return float(struct.unpack("<d", raw[:8])[0])
        if dtype == 3:  # DT_INT32
            return float(struct.unpack("<i", raw[:4])[0])
        if dtype == 9:  # DT_INT64
            return float(struct.unpack("<q", raw[:8])[0])
        if dtype == 19:  # DT_HALF (float16)
            return float(np.frombuffer(raw[:2], dtype=np.float16)[0])
    except Exception:
        return None
    return None


def read_tensorboard_scalars(tb_dirs: List[Path]) -> Dict[str, List[ScalarPoint]]:
    # Merge across all event files, dedupe by step (keep last wall_time).
    by_tag: Dict[str, Dict[int, ScalarPoint]] = {}

    event_files: List[Path] = []
    for d in tb_dirs:
        if not d.exists():
            continue
        event_files.extend(sorted(d.glob("events.out.tfevents.*")))

    for ev_path in event_files:
        try:
            for rec in iter_tfrecord_records(ev_path):
                ev = event_pb2.Event()
                ev.ParseFromString(rec)
                if not ev.HasField("summary"):
                    continue
                step = int(getattr(ev, "step", 0))
                wall_time = float(getattr(ev, "wall_time", 0.0))
                for v in ev.summary.value:
                    tag = v.tag
                    val: Optional[float] = None
                    if v.HasField("simple_value"):
                        val = float(v.simple_value)
                    elif v.HasField("tensor"):
                        val = tensorproto_to_scalar(v.tensor)
                    if val is None or not math.isfinite(val):
                        continue

                    by_tag.setdefault(tag, {})
                    prev = by_tag[tag].get(step)
                    point = ScalarPoint(step=step, wall_time=wall_time, value=val, file=ev_path.name)
                    if prev is None or point.wall_time >= prev.wall_time:
                        by_tag[tag][step] = point
        except Exception as exc:
            print(f"[WARN] Failed to parse TB file {ev_path}: {exc}")

    out: Dict[str, List[ScalarPoint]] = {}
    for tag, step_map in by_tag.items():
        out[tag] = [step_map[s] for s in sorted(step_map)]
    return out


def _series_to_epoch_map(series: List[ScalarPoint]) -> Dict[int, float]:
    # In this training codebase, epoch-level scalars use step = epoch-1.
    out: Dict[int, float] = {}
    for p in series:
        out[p.step + 1] = p.value
    return out


def load_checkpoint(path: Path) -> Dict[str, Any]:
    # Torch 2.6+ defaults to weights_only=True in some setups; force full load.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def analyze_checkpoint(path: Path, key_layers: List[str]) -> Dict[str, Any]:
    ckpt = load_checkpoint(path)

    stat = path.stat()
    info: Dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "epoch": ckpt.get("epoch"),
        "loss": ckpt.get("loss"),
        "has_model": "model_state_dict" in ckpt,
        "has_optimizer": "optimizer_state_dict" in ckpt,
        "has_scheduler": "scheduler_state_dict" in ckpt,
        "has_scaler": "scaler_state_dict" in ckpt,
    }

    # Optimizer info (LRs)
    opt = ckpt.get("optimizer_state_dict")
    if isinstance(opt, dict) and "param_groups" in opt:
        lrs = [g.get("lr") for g in opt.get("param_groups", [])]
        info["learning_rates"] = lrs
        if len(lrs) >= 2:
            info["lr_main"] = lrs[0]
            info["lr_backbone"] = lrs[1]

    # Scheduler info
    sch = ckpt.get("scheduler_state_dict")
    if isinstance(sch, dict):
        for k in ["T_max", "eta_min", "last_epoch", "_last_lr"]:
            if k in sch:
                info.setdefault("scheduler", {})[k] = sch.get(k)

    # Model weight stats (lightweight)
    msd = ckpt.get("model_state_dict")
    if isinstance(msd, dict):
        norms: List[float] = []
        nan_layers: List[str] = []
        inf_layers: List[str] = []
        total_params = 0
        key_stats: Dict[str, Dict[str, Any]] = {}

        for name, tensor in msd.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.dtype.is_floating_point:
                continue

            total_params += int(tensor.numel())
            t = tensor.detach()

            if name in key_layers:
                t32 = t.float()
                key_stats[name] = {
                    "shape": list(t.shape),
                    "norm": float(torch.norm(t32).item()),
                    "mean": float(t32.mean().item()),
                    "std": float(t32.std(unbiased=False).item()),
                }

            if bool(torch.isnan(t).any().item()):
                nan_layers.append(name)
            if bool(torch.isinf(t).any().item()):
                inf_layers.append(name)

            norms.append(float(torch.norm(t.float()).item()))

        if norms:
            info["total_params"] = total_params
            info["weight_norm_mean"] = float(np.mean(norms))
            info["weight_norm_max"] = float(np.max(norms))
            info["nan_layers"] = nan_layers
            info["inf_layers"] = inf_layers
            info["key_layers"] = key_stats

    return info


def analyze_checkpoints(checkpoint_files: List[Path]) -> Dict[str, Any]:
    key_layers = [
        "class_labels_classifier.weight",
        "bbox_predictor.layers.2.weight",
        "model.encoder.layers.0.self_attn.out_proj.weight",
        "model.decoder.layers.0.self_attn.out_proj.weight",
        "model.backbone.conv_encoder.model.layer4.2.conv3.weight",
    ]

    results: List[Dict[str, Any]] = []
    for p in sorted(checkpoint_files, key=lambda x: x.name):
        if p.suffix.lower() != ".pth":
            continue
        print(f"[CKPT] Loading {p.name} ...")
        try:
            results.append(analyze_checkpoint(p, key_layers=key_layers))
        except Exception as exc:
            print(f"[WARN] Failed to analyze {p}: {exc}")
            results.append({"file": p.name, "path": str(p), "error": str(exc)})

    def _epoch_key(d: Dict[str, Any]) -> Tuple[int, str]:
        ep = d.get("epoch")
        return (int(ep) if isinstance(ep, int) else 10**9, d.get("file", ""))

    return {"key_layers": key_layers, "checkpoints": sorted(results, key=_epoch_key)}


def build_epoch_table(log_epochs: List[Dict[str, Any]], tb_scalars: Dict[str, List[ScalarPoint]]) -> List[Dict[str, Any]]:
    log_by_epoch = {e["epoch"]: e for e in log_epochs if isinstance(e.get("epoch"), int)}

    tb_epoch_maps = {
        "tb_train_loss": _series_to_epoch_map(tb_scalars.get("Loss/train_epoch", [])),
        "tb_val_loss": _series_to_epoch_map(tb_scalars.get("Loss/validation_epoch", [])),
        "tb_grad_norm": _series_to_epoch_map(tb_scalars.get("Gradients/norm_avg", [])),
        "tb_lr_main": _series_to_epoch_map(tb_scalars.get("LR/main", [])),
        "tb_lr_backbone": _series_to_epoch_map(tb_scalars.get("LR/backbone", [])),
        "tb_ce": _series_to_epoch_map(tb_scalars.get("Loss_Components/ce_epoch", [])),
        "tb_bbox": _series_to_epoch_map(tb_scalars.get("Loss_Components/bbox_epoch", [])),
        "tb_giou": _series_to_epoch_map(tb_scalars.get("Loss_Components/giou_epoch", [])),
        "tb_time_s": _series_to_epoch_map(tb_scalars.get("Time/epoch_seconds", [])),
        "tb_gpu_alloc_gb": _series_to_epoch_map(tb_scalars.get("GPU/memory_allocated_GB", [])),
        "tb_gpu_res_gb": _series_to_epoch_map(tb_scalars.get("GPU/memory_reserved_GB", [])),
    }

    epochs_all = set(log_by_epoch.keys())
    for m in tb_epoch_maps.values():
        epochs_all.update(m.keys())

    rows: List[Dict[str, Any]] = []
    for ep in sorted(epochs_all):
        row: Dict[str, Any] = {"epoch": ep}
        le = log_by_epoch.get(ep, {})
        row["train_loss_log"] = le.get("train_loss")
        row["val_loss_log"] = le.get("val_loss")
        row["grad_norm_log"] = le.get("grad_norm")
        row["train_time_s_log"] = le.get("train_time_s")
        row["gpu_alloc_gb_log"] = le.get("gpu_alloc_gb")
        row["gpu_reserved_gb_log"] = le.get("gpu_reserved_gb")
        row["saved_checkpoints_log"] = le.get("saved_checkpoints", [])
        row["val_improved_to_log"] = le.get("val_improved_to")

        for k, m in tb_epoch_maps.items():
            row[k] = m.get(ep)
        rows.append(row)
    return rows


def create_plots(epoch_rows: List[Dict[str, Any]], ckpt_analysis: Dict[str, Any], out_dir: Path) -> List[str]:
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    def _normalize_epochs(vals: List[int]) -> List[float]:
        if not vals:
            return []
        vmin = min(vals)
        vmax = max(vals)
        if vmax == vmin:
            return [0.0 for _ in vals]
        return [(v - vmin) / (vmax - vmin) for v in vals]

    def _min_max(series_list: List[List[Optional[float]]]) -> Tuple[Optional[float], Optional[float]]:
        vals = [v for series in series_list for v in series if v is not None]
        if not vals:
            return None, None
        return min(vals), max(vals)

    def _normalize_series(vals: List[Optional[float]], vmin: Optional[float], vmax: Optional[float]) -> List[Optional[float]]:
        if vmin is None or vmax is None:
            return [None for _ in vals]
        if vmax == vmin:
            return [0.0 if v is not None else None for v in vals]
        return [(v - vmin) / (vmax - vmin) if v is not None else None for v in vals]

    def _plot_series(ax, xs: List[float], ys: List[Optional[float]], *args, **kwargs) -> None:
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pairs:
            return
        x_p, y_p = zip(*pairs)
        ax.plot(x_p, y_p, *args, **kwargs)

    epochs = [r["epoch"] for r in epoch_rows]
    epochs_norm = _normalize_epochs(epochs)
    created: List[str] = []

    def _save(fig, name: str):
        png = plot_dir / f"{name}.png"
        pdf = plot_dir / f"{name}.pdf"
        fig.tight_layout()
        fig.savefig(png, dpi=150, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        created.append(str(png.relative_to(out_dir)))

    # Loss curves
    fig, ax = plt.subplots(figsize=(11, 6))
    train_loss_log = [r.get("train_loss_log") for r in epoch_rows]
    val_loss_log = [r.get("val_loss_log") for r in epoch_rows]
    train_loss_tb = [r.get("tb_train_loss") for r in epoch_rows]
    val_loss_tb = [r.get("tb_val_loss") for r in epoch_rows]
    y_min, y_max = _min_max([train_loss_log, val_loss_log, train_loss_tb, val_loss_tb])
    _plot_series(ax, epochs_norm, _normalize_series(train_loss_log, y_min, y_max), "o-", label="Train loss (log)")
    _plot_series(ax, epochs_norm, _normalize_series(val_loss_log, y_min, y_max), "o-", label="Val loss (log)")
    _plot_series(ax, epochs_norm, _normalize_series(train_loss_tb, y_min, y_max), "s--", label="Train loss (TB)")
    _plot_series(ax, epochs_norm, _normalize_series(val_loss_tb, y_min, y_max), "s--", label="Val loss (TB)")
    ax.set_xlabel("Normalized Epoch")
    ax.set_ylabel("Normalized Loss")
    ax.set_title("Loss curves")
    ax.legend()
    _save(fig, "loss_curves")

    # LR schedule
    fig, ax = plt.subplots(figsize=(11, 5))
    lr_main = [r.get("tb_lr_main") for r in epoch_rows]
    lr_bb = [r.get("tb_lr_backbone") for r in epoch_rows]
    y_min, y_max = _min_max([lr_main, lr_bb])
    _plot_series(ax, epochs_norm, _normalize_series(lr_main, y_min, y_max), "o-", label="LR main (TB)")
    _plot_series(ax, epochs_norm, _normalize_series(lr_bb, y_min, y_max), "o-", label="LR backbone (TB)")
    ax.set_xlabel("Normalized Epoch")
    ax.set_ylabel("Normalized Learning rate")
    ax.set_title("Learning rate schedule")
    ax.legend()
    _save(fig, "lr_schedule")

    # Gradient norm
    fig, ax = plt.subplots(figsize=(11, 5))
    grad_log = [r.get("grad_norm_log") for r in epoch_rows]
    grad_tb = [r.get("tb_grad_norm") for r in epoch_rows]
    y_min, y_max = _min_max([grad_log, grad_tb])
    _plot_series(ax, epochs_norm, _normalize_series(grad_log, y_min, y_max), "o-", label="Avg grad norm (log)")
    _plot_series(ax, epochs_norm, _normalize_series(grad_tb, y_min, y_max), "s--", label="Avg grad norm (TB)")
    ax.set_xlabel("Normalized Epoch")
    ax.set_ylabel("Normalized Gradient norm")
    ax.set_title("Gradient norm")
    ax.legend()
    _save(fig, "grad_norm")

    # Loss components
    fig, ax = plt.subplots(figsize=(11, 5))
    comp_ce = [r.get("tb_ce") for r in epoch_rows]
    comp_bbox = [r.get("tb_bbox") for r in epoch_rows]
    comp_giou = [r.get("tb_giou") for r in epoch_rows]
    y_min, y_max = _min_max([comp_ce, comp_bbox, comp_giou])
    _plot_series(ax, epochs_norm, _normalize_series(comp_ce, y_min, y_max), "o-", label="CE (TB)")
    _plot_series(ax, epochs_norm, _normalize_series(comp_bbox, y_min, y_max), "o-", label="BBox (TB)")
    _plot_series(ax, epochs_norm, _normalize_series(comp_giou, y_min, y_max), "o-", label="GIoU (TB)")
    ax.set_xlabel("Normalized Epoch")
    ax.set_ylabel("Normalized Component loss")
    ax.set_title("Loss components (epoch)")
    ax.legend()
    _save(fig, "loss_components")

    # GPU memory
    fig, ax = plt.subplots(figsize=(11, 5))
    gpu_alloc = [r.get("tb_gpu_alloc_gb") for r in epoch_rows]
    gpu_res = [r.get("tb_gpu_res_gb") for r in epoch_rows]
    y_min, y_max = _min_max([gpu_alloc, gpu_res])
    _plot_series(ax, epochs_norm, _normalize_series(gpu_alloc, y_min, y_max), "o-", label="GPU allocated GB (TB)")
    _plot_series(ax, epochs_norm, _normalize_series(gpu_res, y_min, y_max), "o-", label="GPU reserved GB (TB)")
    ax.set_xlabel("Normalized Epoch")
    ax.set_ylabel("Normalized GB")
    ax.set_title("GPU memory (epoch)")
    ax.legend()
    _save(fig, "gpu_memory")

    # Epoch time
    fig, ax = plt.subplots(figsize=(11, 5))
    time_log = [r.get("train_time_s_log") for r in epoch_rows]
    time_tb = [r.get("tb_time_s") for r in epoch_rows]
    y_min, y_max = _min_max([time_log, time_tb])
    _plot_series(ax, epochs_norm, _normalize_series(time_log, y_min, y_max), "o-", label="Train epoch time (log)")
    _plot_series(ax, epochs_norm, _normalize_series(time_tb, y_min, y_max), "s--", label="Epoch seconds (TB)")
    ax.set_xlabel("Normalized Epoch")
    ax.set_ylabel("Normalized Seconds")
    ax.set_title("Epoch duration")
    ax.legend()
    _save(fig, "epoch_time")

    # Checkpoint weight norm evolution
    ckpts = ckpt_analysis.get("checkpoints", [])
    ckpt_epochs = [
        c.get("epoch")
        for c in ckpts
        if isinstance(c.get("epoch"), int) and c.get("weight_norm_mean") is not None
    ]
    ckpt_norms = [
        c.get("weight_norm_mean")
        for c in ckpts
        if isinstance(c.get("epoch"), int) and c.get("weight_norm_mean") is not None
    ]
    if ckpt_epochs:
        fig, ax = plt.subplots(figsize=(11, 5))
        ckpt_x_norm = _normalize_epochs(ckpt_epochs)
        y_min, y_max = _min_max([ckpt_norms])
        ckpt_y_norm = _normalize_series(ckpt_norms, y_min, y_max)
        _plot_series(ax, ckpt_x_norm, ckpt_y_norm, "o-", label="Mean tensor weight-norm")
        ax.set_xlabel("Normalized Checkpoint epoch")
        ax.set_ylabel("Normalized Mean norm")
        ax.set_title("Checkpoint weight norm evolution")
        ax.legend()
        _save(fig, "checkpoint_weight_norm")

    return created


def write_markdown_report(
    out_dir: Path,
    session: Dict[str, Any],
    log_analysis: Dict[str, Any],
    tb_scalars: Dict[str, List[ScalarPoint]],
    epoch_rows: List[Dict[str, Any]],
    ckpt_analysis: Dict[str, Any],
    plot_paths: List[str],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "detr_training_session_report.md"

    header = log_analysis.get("header", {})
    session_dir = Path(session["session_dir"])

    # Best epoch by validation loss (prefer log, fallback to TB)
    best_val = None
    best_epoch = None
    for r in epoch_rows:
        v = r.get("val_loss_log")
        if v is None:
            v = r.get("tb_val_loss")
        if v is None:
            continue
        if best_val is None or v < best_val:
            best_val = v
            best_epoch = r["epoch"]

    # Final model config (if present)
    fm_info: Dict[str, Any] = {}
    fm_dirs = session.get("final_model_dirs") or []
    if fm_dirs:
        fm_dir = Path(fm_dirs[0])
        fm_info["final_model_dir"] = str(fm_dir)
        cfg = _read_json(fm_dir / "config.json")
        pre = _read_json(fm_dir / "preprocessor_config.json")
        if cfg:
            fm_info["config"] = cfg
        if pre:
            fm_info["preprocessor"] = pre

    lines: List[str] = []
    lines.append("# DETR training session analysis")
    lines.append("")
    lines.append(f"- Session dir: `{session_dir}`")
    if header.get("job_id"):
        lines.append(f"- Job ID: `{header.get('job_id')}`")
    if header.get("start_time_raw"):
        lines.append(f"- Start time: `{header.get('start_time_raw')}`")
    if header.get("node"):
        lines.append(f"- Node: `{header.get('node')}`")
    if header.get("image_count") is not None:
        lines.append(f"- Image count: `{header.get('image_count')}`")
    if header.get("gpus"):
        g = header["gpus"]
        lines.append(f"- GPUs: `{g.get('count')}x {g.get('name')}`")
    if header.get("effective_batch"):
        lines.append(f"- Effective batch: `{header.get('effective_batch')}`")
    if header.get("mixed_precision"):
        lines.append(f"- Mixed precision: `{header.get('mixed_precision')}`")
    if best_epoch is not None:
        lines.append(f"- Best validation loss: `{best_val:.6f}` at epoch `{best_epoch}`")
    if header.get("incomplete_epoch"):
        ie = header["incomplete_epoch"]
        lines.append(
            f"- Training ended mid-epoch: `E{ie.get('epoch')}` at `{ie.get('iter')}/{ie.get('total')}` (~{ie.get('pct')}%)"
        )
    lines.append("")

    lines.append("## Session contents")
    lines.append("")
    lines.append(f"- Log files: `{len(session.get('log_files', []))}`")
    lines.append(f"- Checkpoint dirs: `{len(session.get('ckpt_dirs', []))}`")
    lines.append(f"- Checkpoints: `{len(session.get('checkpoint_files', []))}`")
    lines.append(f"- TensorBoard dirs: `{len(session.get('tb_log_dirs', []))}`")
    if session.get("tb_log_dirs"):
        tb_count = 0
        for d in session.get("tb_log_dirs", []):
            tb_count += len(list(Path(d).glob('events.out.tfevents.*')))
        lines.append(f"- TensorBoard event files: `{tb_count}`")
    lines.append("")

    lines.append("## Key training events (from log)")
    lines.append("")
    resume = header.get("resume") or {}
    if resume:
        lines.append(
            "- Resume: found `{found}`, loaded `{loaded}`, epoch `{epoch}`".format(
                found=resume.get("found_file"),
                loaded=resume.get("loaded_path"),
                epoch=resume.get("loaded_epoch"),
            )
        )
    lr_reset = header.get("lr_reset") or {}
    if lr_reset.get("present"):
        lines.append(
            "- LR reset: old `{old}`, new `{new}` (backbone `{bb}`), cosine remaining epochs `{rem}`".format(
                old=lr_reset.get("old_lr"),
                new=lr_reset.get("new_lr"),
                bb=lr_reset.get("new_lr_backbone"),
                rem=lr_reset.get("cosine_remaining_epochs"),
            )
        )
    lines.append("")

    args_cfg = header.get("args") or {}
    if args_cfg:
        lines.append("## Configuration (from Namespace)")
        lines.append("")
        show_keys = [
            "model_checkpoint",
            "num_queries",
            "train_val_split",
            "augment",
            "batch_size",
            "epochs",
            "lr",
            "lr_backbone",
            "weight_decay",
            "max_grad_norm",
            "use_amp",
            "lr_scheduler",
            "warmup_epochs",
            "lr_min",
            "save_interval",
            "patience",
            "resume_training",
            "checkpoint_dir",
            "best_model_dir",
            "output_dir",
        ]
        for k in show_keys:
            if k in args_cfg:
                lines.append(f"- `{k}`: `{args_cfg[k]}`")
        lines.append("")

    # Derived stats / quick findings
    full_epochs = [r for r in epoch_rows if r.get("train_loss_log") is not None]
    if full_epochs:
        first_ep = min(r["epoch"] for r in full_epochs)
        last_ep = max(r["epoch"] for r in full_epochs)
        epoch_times = [r.get("train_time_s_log") for r in full_epochs if isinstance(r.get("train_time_s_log"), (int, float))]
        avg_time = float(np.mean(epoch_times)) if epoch_times else None

        val_losses = [r.get("val_loss_log") for r in full_epochs if isinstance(r.get("val_loss_log"), (int, float))]
        worst_val = max(val_losses) if val_losses else None

        lines.append("## Findings (derived)")
        lines.append("")
        lines.append(f"- Full epochs in log: `{len(full_epochs)}` (E{first_ep}..E{last_ep})")
        if avg_time is not None:
            lines.append(f"- Avg epoch time (log): `{avg_time:.1f}s` (~{_format_seconds(avg_time)})")
        if worst_val is not None:
            lines.append(f"- Worst validation loss (log): `{worst_val:.6f}`")

        image_count = header.get("image_count")
        split = args_cfg.get("train_val_split")
        eff_batch = header.get("effective_batch")
        if isinstance(image_count, int) and isinstance(split, (int, float)) and isinstance(eff_batch, int) and avg_time:
            train_images = int(round(image_count * float(split)))
            approx_ips = train_images / avg_time if avg_time > 0 else None
            if approx_ips:
                lines.append(f"- Approx throughput: `{approx_ips:.2f}` train images/s (all GPUs)")

        # Suggest resume checkpoint
        ckpts = [c for c in ckpt_analysis.get("checkpoints", []) if isinstance(c.get("epoch"), int)]
        if ckpts:
            best_ckpt = max(ckpts, key=lambda c: c["epoch"])
            lines.append(f"- Suggested resume checkpoint: `{best_ckpt.get('file')}` (epoch `{best_ckpt.get('epoch')}`)")
        lines.append("")

    lines.append("## TensorBoard scalar tags found")
    lines.append("")
    for tag in sorted(tb_scalars.keys()):
        lines.append(f"- `{tag}` ({len(tb_scalars[tag])} points)")
    lines.append("")

    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for rel in plot_paths:
            rel_md = rel.replace("\\", "/")
            lines.append(f"![]({rel_md})")
            lines.append("")

    lines.append("## Epoch summary table")
    lines.append("")
    lines.append(
        "| Epoch | Train loss (log) | Val loss (log) | Train loss (TB) | Val loss (TB) | Grad norm (log) | LR main (TB) | LR bb (TB) | Epoch time |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in epoch_rows:
        ep = r["epoch"]
        lines.append(
            "| {ep} | {tr_l} | {va_l} | {tr_tb} | {va_tb} | {gn} | {lr} | {bb} | {t} |".format(
                ep=ep,
                tr_l=f"{r.get('train_loss_log'):.4f}" if r.get("train_loss_log") is not None else "N/A",
                va_l=f"{r.get('val_loss_log'):.4f}" if r.get("val_loss_log") is not None else "N/A",
                tr_tb=f"{r.get('tb_train_loss'):.4f}" if r.get("tb_train_loss") is not None else "N/A",
                va_tb=f"{r.get('tb_val_loss'):.4f}" if r.get("tb_val_loss") is not None else "N/A",
                gn=f"{r.get('grad_norm_log'):.2f}" if r.get("grad_norm_log") is not None else "N/A",
                lr=f"{r.get('tb_lr_main'):.2e}" if r.get("tb_lr_main") is not None else "N/A",
                bb=f"{r.get('tb_lr_backbone'):.2e}" if r.get("tb_lr_backbone") is not None else "N/A",
                t=_format_seconds(r.get("train_time_s_log") or r.get("tb_time_s")),
            )
        )
    lines.append("")

    lines.append("## Checkpoint analysis")
    lines.append("")
    lines.append("| File | Epoch | Loss | LR main | LR bb | Scheduler last_epoch | Mean weight norm | NaN/Inf tensors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in ckpt_analysis.get("checkpoints", []):
        if c.get("error"):
            lines.append(f"| `{c.get('file')}` | N/A | N/A | N/A | N/A | N/A | N/A | error: `{c.get('error')}` |")
            continue
        lr_main = c.get("lr_main")
        lr_bb = c.get("lr_backbone")
        sch_last = (c.get("scheduler") or {}).get("last_epoch")
        nan_inf = (
            f"{len(c.get('nan_layers', []))}/{len(c.get('inf_layers', []))}"
            if c.get("nan_layers") is not None
            else "N/A"
        )
        lines.append(
            "| `{file}` | {ep} | {loss} | {lr} | {bb} | {sch} | {wn} | {ni} |".format(
                file=c.get("file"),
                ep=c.get("epoch") if c.get("epoch") is not None else "N/A",
                loss=f"{c.get('loss'):.6f}" if isinstance(c.get("loss"), (int, float)) else "N/A",
                lr=f"{lr_main:.2e}" if isinstance(lr_main, (int, float)) else "N/A",
                bb=f"{lr_bb:.2e}" if isinstance(lr_bb, (int, float)) else "N/A",
                sch=sch_last if sch_last is not None else "N/A",
                wn=f"{c.get('weight_norm_mean'):.4f}"
                if isinstance(c.get("weight_norm_mean"), (int, float))
                else "N/A",
                ni=nan_inf,
            )
        )
    lines.append("")

    if fm_info:
        lines.append("## Final model artifacts (HuggingFace)")
        lines.append("")
        lines.append(f"- Final model dir: `{fm_info.get('final_model_dir')}`")
        cfg = fm_info.get("config") or {}
        pre = fm_info.get("preprocessor") or {}
        if cfg.get("transformers_version"):
            lines.append(f"- `transformers_version`: `{cfg.get('transformers_version')}`")
        if cfg.get("num_queries") is not None:
            lines.append(f"- `num_queries`: `{cfg.get('num_queries')}`")
        if cfg.get("id2label"):
            lines.append(f"- `id2label`: `{cfg.get('id2label')}`")
        if pre.get("size"):
            lines.append(f"- Preprocessor `size`: `{pre.get('size')}`")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a DETR training session folder (log + TensorBoard + checkpoints)"
    )
    parser.add_argument("--session-dir", type=str, default=str(DEFAULT_SESSION_DIR), help="Session directory to analyze")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (default: Eden/Scripts/CheckpointAnalizis/output/detr_sessions/<session_name>)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise SystemExit(f"[ERROR] Session dir not found: {session_dir}")

    session = discover_session(session_dir)

    out_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_OUTPUT_ROOT / session_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log analysis (pick the largest *.log if multiple)
    log_files: List[Path] = session.get("log_files", [])
    if not log_files:
        raise SystemExit(f"[ERROR] No *.log found in {session_dir}")
    log_path = max(log_files, key=lambda p: p.stat().st_size)
    print(f"[LOG] Parsing {log_path.name} ...")
    log_analysis = parse_training_log(log_path)

    # TensorBoard
    print("[TB] Parsing TensorBoard event files ...")
    tb_scalars = read_tensorboard_scalars(session.get("tb_log_dirs", []))

    # Epoch table
    epoch_rows = build_epoch_table(log_analysis.get("epochs", []), tb_scalars)

    # Checkpoints
    ckpt_analysis = analyze_checkpoints(session.get("checkpoint_files", []))

    # Save JSON
    json_path = out_dir / "detr_training_session_analysis.json"
    serializable_tb = {
        tag: [{"step": p.step, "wall_time": p.wall_time, "value": p.value, "file": p.file} for p in series]
        for tag, series in tb_scalars.items()
    }
    json_path.write_text(
        json.dumps(
            {
                "session_dir": str(session["session_dir"]),
                "log_analysis": log_analysis,
                "tensorboard": {"tags": list(tb_scalars.keys()), "scalars": serializable_tb},
                "epoch_rows": epoch_rows,
                "checkpoints": ckpt_analysis,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "python": sys.version,
                "torch": getattr(torch, "__version__", "unknown"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Wrote JSON: {json_path}")

    plot_paths: List[str] = []
    if not args.no_plots:
        print("[PLOT] Generating plots ...")
        plot_paths = create_plots(epoch_rows, ckpt_analysis, out_dir)

    report_path = write_markdown_report(
        out_dir=out_dir,
        session=session,
        log_analysis=log_analysis,
        tb_scalars=tb_scalars,
        epoch_rows=epoch_rows,
        ckpt_analysis=ckpt_analysis,
        plot_paths=plot_paths,
    )
    print(f"[OK] Wrote report: {report_path}")
    print(f"[DONE] Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
