"""
Compare DETR vs YOLO training parameters across two phases.

Usage:
  py -3.11 compare_detr_yolo_training.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DEFAULT_DETR_PHASE1_JSON = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output"
    r"\detr_custom2\detr_checkpoint_analysis.json"
)
DEFAULT_DETR_PHASE2_JSON = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output"
    r"\detr_custom_finetune\detr_checkpoint_analysis.json"
)
DEFAULT_YOLO_PHASE1_JSON = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output"
    r"\yolo_custom2\yolo_checkpoint_analysis.json"
)
DEFAULT_YOLO_PHASE2_JSON = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output"
    r"\yolo_custom_finetune\yolo_checkpoint_analysis.json"
)
DEFAULT_YOLO_PHASE1_CSV = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\exp\results.csv"
)
DEFAULT_YOLO_PHASE2_CSV = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN"
    r"\YoloTreningSesnions\20KTreningFrom170epochStart\exp\results.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output"
    r"\compare_detr_yolo"
)

PHASE1_MAX_EPOCH = 170
PHASE2_MIN_EPOCH = 170
PHASE2_MAX_EPOCH = 200
PHASE1_DATASET = "~100k images"
PHASE2_DATASET = "20k images (fine-tune)"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_detr(detr_json: Path) -> Dict:
    obj = _load_json(detr_json)
    layer_evolution = obj.get("layer_evolution", {})

    if "log_analysis" in obj:
        header = obj.get("log_analysis", {}).get("header", {})
        args = header.get("args", {})
        epoch_rows = obj.get("epoch_rows", [])
        ckpts = obj.get("checkpoints", {}).get("checkpoints", [])

        lr_rows = [(r.get("epoch"), r.get("tb_lr_main"), r.get("tb_lr_backbone"))
                   for r in epoch_rows if r.get("tb_lr_main") is not None]
        val_loss_rows = [(r.get("epoch"), r.get("tb_val_loss"))
                         for r in epoch_rows if r.get("tb_val_loss") is not None]
        weight_rows = [(c.get("epoch"), c.get("weight_norm_mean"))
                       for c in ckpts if c.get("epoch") is not None and c.get("weight_norm_mean") is not None]
        weight_max_rows = [(c.get("epoch"), c.get("weight_norm_max"))
                           for c in ckpts if c.get("epoch") is not None and c.get("weight_norm_max") is not None]
        loss_rows = [(c.get("epoch"), c.get("loss"))
                     for c in ckpts if c.get("epoch") is not None and c.get("loss") is not None]

        return {
            "header": header,
            "args": args,
            "lr_rows": lr_rows,
            "val_loss_rows": val_loss_rows,
            "weight_rows": weight_rows,
            "weight_max_rows": weight_max_rows,
            "loss_rows": loss_rows,
            "layer_evolution": layer_evolution,
        }

    summary = obj.get("summary", [])
    summary = [r for r in summary if r.get("epoch") is not None]
    summary.sort(key=lambda r: r["epoch"])

    lr_rows = [(r.get("epoch"), (r.get("learning_rates") or [None])[0],
               (r.get("learning_rates") or [None, None])[1] if r.get("learning_rates") else None)
               for r in summary if r.get("learning_rates")]
    weight_rows = [(r.get("epoch"), r.get("weight_norm_mean"))
                   for r in summary if r.get("weight_norm_mean") is not None]
    weight_max_rows = [(r.get("epoch"), r.get("weight_norm_max"))
                       for r in summary if r.get("weight_norm_max") is not None]
    loss_rows = [(r.get("epoch"), r.get("loss"))
                 for r in summary if r.get("loss") is not None]

    return {
        "header": {},
        "args": {},
        "lr_rows": lr_rows,
        "val_loss_rows": [],
        "weight_rows": weight_rows,
        "weight_max_rows": weight_max_rows,
        "loss_rows": loss_rows,
        "layer_evolution": layer_evolution,
    }


def load_yolo(yolo_json: Path) -> Dict:
    obj = _load_json(yolo_json)
    rows = obj.get("summary", [])
    rows = [r for r in rows if r.get("epoch") is not None]
    rows.sort(key=lambda r: r["epoch"])

    train_args = {}
    for r in rows:
        if r.get("train_args"):
            train_args = r["train_args"]
            break

    weight_rows = [(r.get("epoch"), r.get("weight_norm_mean"))
                   for r in rows if r.get("weight_norm_mean") is not None]
    weight_max_rows = [(r.get("epoch"), r.get("weight_norm_max"))
                       for r in rows if r.get("weight_norm_max") is not None]
    weight_std_rows = [(r.get("epoch"), r.get("weight_norm_std"))
                       for r in rows if r.get("weight_norm_std") is not None]
    lr_rows = [(r.get("epoch"), (r.get("learning_rates") or [None])[0])
               for r in rows if r.get("learning_rates")]
    fitness_rows = [(r.get("epoch"), r.get("best_fitness"))
                    for r in rows if r.get("best_fitness") is not None]

    return {
        "rows": rows,
        "train_args": train_args,
        "weight_rows": weight_rows,
        "weight_max_rows": weight_max_rows,
        "weight_std_rows": weight_std_rows,
        "lr_rows": lr_rows,
        "fitness_rows": fitness_rows,
    }


def load_yolo_csv(results_csv: Path) -> Dict:
    if not results_csv.exists():
        return {
            "epochs": [],
            "lr_pg0": [],
            "lr_pg1": [],
            "lr_pg2": [],
            "map50_95": [],
            "map50": [],
            "precision": [],
            "recall": [],
            "train_box_loss": [],
            "val_box_loss": [],
        }

    epochs = []
    lr_pg0 = []
    lr_pg1 = []
    lr_pg2 = []
    map50_95 = []
    map50 = []
    precision = []
    recall = []
    train_box_loss = []
    val_box_loss = []

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(float(row["epoch"])))
                lr_pg0.append(float(row["lr/pg0"]))
                lr_pg1.append(float(row.get("lr/pg1", row["lr/pg0"])))
                lr_pg2.append(float(row.get("lr/pg2", row["lr/pg0"])))
                map50_95.append(float(row["metrics/mAP50-95(B)"]))
                map50.append(float(row["metrics/mAP50(B)"]))
                precision.append(float(row["metrics/precision(B)"]))
                recall.append(float(row["metrics/recall(B)"]))
                train_box_loss.append(float(row["train/box_loss"]))
                val_box_loss.append(float(row["val/box_loss"]))
            except Exception:
                continue

    return {
        "epochs": epochs,
        "lr_pg0": lr_pg0,
        "lr_pg1": lr_pg1,
        "lr_pg2": lr_pg2,
        "map50_95": map50_95,
        "map50": map50,
        "precision": precision,
        "recall": recall,
        "train_box_loss": train_box_loss,
        "val_box_loss": val_box_loss,
    }

def filter_epoch_rows(rows: List[Tuple[int, float]], min_epoch: Optional[int], max_epoch: Optional[int]) -> List[Tuple[int, float]]:
    out = []
    for e, v in rows:
        if min_epoch is not None and e < min_epoch:
            continue
        if max_epoch is not None and e > max_epoch:
            continue
        out.append((e, v))
    return out


def filter_epoch_rows_three(
    rows: List[Tuple[int, float, float]],
    min_epoch: Optional[int],
    max_epoch: Optional[int],
) -> List[Tuple[int, float, float]]:
    out = []
    for e, a, b in rows:
        if min_epoch is not None and e < min_epoch:
            continue
        if max_epoch is not None and e > max_epoch:
            continue
        out.append((e, a, b))
    return out


def filter_yolo_csv(data: Dict, min_epoch: Optional[int], max_epoch: Optional[int]) -> Dict:
    if not data["epochs"]:
        return data
    keep = [i for i, e in enumerate(data["epochs"])
            if (min_epoch is None or e >= min_epoch) and (max_epoch is None or e <= max_epoch)]
    return {
        "epochs": [data["epochs"][i] for i in keep],
        "lr_pg0": [data["lr_pg0"][i] for i in keep],
        "lr_pg1": [data["lr_pg1"][i] for i in keep],
        "lr_pg2": [data["lr_pg2"][i] for i in keep],
        "map50_95": [data["map50_95"][i] for i in keep],
        "map50": [data["map50"][i] for i in keep],
        "precision": [data["precision"][i] for i in keep],
        "recall": [data["recall"][i] for i in keep],
        "train_box_loss": [data["train_box_loss"][i] for i in keep],
        "val_box_loss": [data["val_box_loss"][i] for i in keep],
    }


def _first_last(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    return vals[0], vals[-1]


def _min_max(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    return min(vals), max(vals)


def _best_epoch(rows: List[Tuple[int, float]], mode: str) -> Tuple[Optional[int], Optional[float]]:
    if not rows:
        return None, None
    if mode == "min":
        epoch, val = min(rows, key=lambda x: x[1])
    else:
        epoch, val = max(rows, key=lambda x: x[1])
    return epoch, val


def _phase_summary(
    detr: Dict,
    yolo: Dict,
    yolo_csv: Dict,
) -> Dict:
    detr_lr_vals = [lr for _, lr, _ in detr["lr_rows"] if lr is not None]
    detr_lr_bb_vals = [lr for _, _, lr in detr["lr_rows"] if lr is not None]
    yolo_lr_vals = yolo_csv["lr_pg0"] or [lr for _, lr in yolo["lr_rows"] if lr is not None]

    detr_val_rows = [(e, v) for e, v in detr["val_loss_rows"] if v is not None]
    detr_loss_rows = [(e, v) for e, v in detr["loss_rows"] if v is not None]

    yolo_map_rows = list(zip(yolo_csv["epochs"], yolo_csv["map50_95"])) if yolo_csv["epochs"] else []
    yolo_fit_rows = [(e, v) for e, v in yolo["fitness_rows"] if v is not None]

    detr_weight_vals = [v for _, v in detr["weight_rows"] if v is not None]
    detr_weight_max_vals = [v for _, v in detr["weight_max_rows"] if v is not None]
    yolo_weight_vals = [v for _, v in yolo["weight_rows"] if v is not None]
    yolo_weight_max_vals = [v for _, v in yolo["weight_max_rows"] if v is not None]
    yolo_weight_std_vals = [v for _, v in yolo["weight_std_rows"] if v is not None]

    detr_lr_start, detr_lr_end = _first_last(detr_lr_vals)
    detr_lr_bb_start, detr_lr_bb_end = _first_last(detr_lr_bb_vals)
    yolo_lr_start, yolo_lr_end = _first_last(yolo_lr_vals)

    detr_best_epoch, detr_best_val = _best_epoch(detr_val_rows, "min")
    detr_best_label = "val loss"
    if detr_best_epoch is None:
        detr_best_epoch, detr_best_val = _best_epoch(detr_loss_rows, "min")
        detr_best_label = "train loss"

    yolo_best_epoch, yolo_best_val = _best_epoch(yolo_map_rows, "max")
    yolo_best_label = "mAP50-95"
    if yolo_best_epoch is None:
        yolo_best_epoch, yolo_best_val = _best_epoch(yolo_fit_rows, "max")
        yolo_best_label = "fitness"

    detr_weight_min, detr_weight_max = _min_max(detr_weight_vals)
    detr_weight_max_min, detr_weight_max_max = _min_max(detr_weight_max_vals)
    yolo_weight_min, yolo_weight_max = _min_max(yolo_weight_vals)
    yolo_weight_max_min, yolo_weight_max_max = _min_max(yolo_weight_max_vals)
    yolo_weight_std_min, yolo_weight_std_max = _min_max(yolo_weight_std_vals)

    return {
        "detr_lr_start": detr_lr_start,
        "detr_lr_end": detr_lr_end,
        "detr_lr_bb_start": detr_lr_bb_start,
        "detr_lr_bb_end": detr_lr_bb_end,
        "yolo_lr_start": yolo_lr_start,
        "yolo_lr_end": yolo_lr_end,
        "detr_best_epoch": detr_best_epoch,
        "detr_best_val": detr_best_val,
        "detr_best_label": detr_best_label,
        "yolo_best_epoch": yolo_best_epoch,
        "yolo_best_val": yolo_best_val,
        "yolo_best_label": yolo_best_label,
        "detr_weight_min": detr_weight_min,
        "detr_weight_max": detr_weight_max,
        "detr_weight_max_min": detr_weight_max_min,
        "detr_weight_max_max": detr_weight_max_max,
        "yolo_weight_min": yolo_weight_min,
        "yolo_weight_max": yolo_weight_max,
        "yolo_weight_max_min": yolo_weight_max_min,
        "yolo_weight_max_max": yolo_weight_max_max,
        "yolo_weight_std_min": yolo_weight_std_min,
        "yolo_weight_std_max": yolo_weight_std_max,
    }


def build_report(
    phase1: Dict,
    phase2: Dict,
    paths: Dict[str, Path],
    output_dir: Path,
    output_plots: Dict[str, Path],
    detr_layer_changes: List[Tuple[str, float]],
) -> Path:
    lines = []
    lines.append("# DETR vs YOLO training comparison (two phases)")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- DETR phase1: `{paths['detr1']}`")
    lines.append(f"- DETR phase2: `{paths['detr2']}`")
    lines.append(f"- YOLO phase1: `{paths['yolo1']}`")
    lines.append(f"- YOLO phase2: `{paths['yolo2']}`")
    lines.append(f"- YOLO phase1 CSV: `{paths['yolo1_csv']}`")
    lines.append(f"- YOLO phase2 CSV: `{paths['yolo2_csv']}`")
    lines.append("")

    lines.append("## Dataset phases")
    lines.append(f"- Phase 1 (epoch <= {PHASE1_MAX_EPOCH}): {PHASE1_DATASET}")
    lines.append(f"- Phase 2 (epoch {PHASE2_MIN_EPOCH}-{PHASE2_MAX_EPOCH}): {PHASE2_DATASET}")
    lines.append("")

    lines.append("## Plots")
    for label, path in output_plots.items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")

    lines.append("## Phase 1 (epoch <= 170)")
    lines.append("| Metric | DETR | YOLO |")
    lines.append("|---|---|---|")
    lines.append(
        f"| LR start -> end | {phase1['detr_lr_start']} -> {phase1['detr_lr_end']} | "
        f"{phase1['yolo_lr_start']} -> {phase1['yolo_lr_end']} |"
    )
    lines.append(
        f"| LR backbone start -> end | {phase1['detr_lr_bb_start']} -> {phase1['detr_lr_bb_end']} | N/A |"
    )
    lines.append(
        f"| Best metric | {phase1['detr_best_label']} {phase1['detr_best_val']} @ {phase1['detr_best_epoch']} | "
        f"{phase1['yolo_best_label']} {phase1['yolo_best_val']} @ {phase1['yolo_best_epoch']} |"
    )
    lines.append(
        f"| Weight norm range | {phase1['detr_weight_min']} -> {phase1['detr_weight_max']} | "
        f"{phase1['yolo_weight_min']} -> {phase1['yolo_weight_max']} |"
    )
    lines.append(
        f"| Weight norm max range | {phase1['detr_weight_max_min']} -> {phase1['detr_weight_max_max']} | "
        f"{phase1['yolo_weight_max_min']} -> {phase1['yolo_weight_max_max']} |"
    )
    if phase1["yolo_weight_std_min"] is not None:
        lines.append(
            f"| YOLO weight std range | N/A | {phase1['yolo_weight_std_min']} -> {phase1['yolo_weight_std_max']} |"
        )
    lines.append("")

    lines.append("## Phase 2 (epoch 170-200)")
    lines.append("| Metric | DETR | YOLO |")
    lines.append("|---|---|---|")
    lines.append(
        f"| LR start -> end | {phase2['detr_lr_start']} -> {phase2['detr_lr_end']} | "
        f"{phase2['yolo_lr_start']} -> {phase2['yolo_lr_end']} |"
    )
    lines.append(
        f"| LR backbone start -> end | {phase2['detr_lr_bb_start']} -> {phase2['detr_lr_bb_end']} | N/A |"
    )
    lines.append(
        f"| Best metric | {phase2['detr_best_label']} {phase2['detr_best_val']} @ {phase2['detr_best_epoch']} | "
        f"{phase2['yolo_best_label']} {phase2['yolo_best_val']} @ {phase2['yolo_best_epoch']} |"
    )
    lines.append(
        f"| Weight norm range | {phase2['detr_weight_min']} -> {phase2['detr_weight_max']} | "
        f"{phase2['yolo_weight_min']} -> {phase2['yolo_weight_max']} |"
    )
    lines.append(
        f"| Weight norm max range | {phase2['detr_weight_max_min']} -> {phase2['detr_weight_max_max']} | "
        f"{phase2['yolo_weight_max_min']} -> {phase2['yolo_weight_max_max']} |"
    )
    if phase2["yolo_weight_std_min"] is not None:
        lines.append(
            f"| YOLO weight std range | N/A | {phase2['yolo_weight_std_min']} -> {phase2['yolo_weight_std_max']} |"
        )
    lines.append("")

    if detr_layer_changes:
        lines.append("## DETR layer norm drift (earliest -> latest)")
        lines.append("| Layer | Norm change (%) |")
        lines.append("|---|---|")
        for name, change in detr_layer_changes:
            lines.append(f"| {name} | {change:+.2f}% |")
        lines.append("")

    lines.append("## Notes")
    lines.append("- DETR metrics are losses (lower is better). YOLO metrics are mAP (higher is better).")
    lines.append("- Weight statistics come from checkpoint analysis; YOLO std is only available for YOLO checkpoints.")
    lines.append("")

    report_path = output_dir / "detr_vs_yolo_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _add_phase_markers(ax, x_min: float, x_max: float) -> None:
    phase1_end = min(PHASE1_MAX_EPOCH, x_max)
    phase2_start = max(PHASE2_MIN_EPOCH, x_min)

    if x_min < phase1_end:
        ax.axvspan(x_min, phase1_end, color="tab:gray", alpha=0.08)
    if phase2_start < x_max:
        ax.axvspan(phase2_start, x_max, color="tab:orange", alpha=0.08)

    ax.axvline(PHASE2_MIN_EPOCH, color="black", linestyle=":", linewidth=1)


def _merge_layer_evolution(layer_a: Dict, layer_b: Dict) -> Dict[str, List[Dict]]:
    merged = {}
    keys = set(layer_a.keys()) | set(layer_b.keys())
    for key in keys:
        combined = list(layer_a.get(key, [])) + list(layer_b.get(key, []))
        combined.sort(key=lambda e: e.get("epoch", 0))
        merged[key] = combined
    return merged


def _summarize_layer_changes(layer_evolution: Dict[str, List[Dict]]) -> List[Tuple[str, float]]:
    rows = []
    for name, entries in layer_evolution.items():
        if not entries:
            continue
        first = entries[0].get("norm")
        last = entries[-1].get("norm")
        if first is None or last is None or first == 0:
            continue
        change = (last - first) / first * 100.0
        rows.append((name, change))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows


def plot_comparison(
    detr1: Dict,
    detr2: Dict,
    yolo1: Dict,
    yolo2: Dict,
    yolo1_csv: Dict,
    yolo2_csv: Dict,
    output_dir: Path,
) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) LR schedule
    ax = axes[0, 0]
    lr_series = []
    if detr1["lr_rows"]:
        lr_series.append((
            "DETR lr/main (phase1)",
            [e for e, _, _ in detr1["lr_rows"]],
            [lr for _, lr, _ in detr1["lr_rows"]],
            {},
        ))
        lr_series.append((
            "DETR lr/backbone (phase1)",
            [e for e, _, _ in detr1["lr_rows"]],
            [lr for _, _, lr in detr1["lr_rows"]],
            {"linestyle": ":"},
        ))
    if detr2["lr_rows"]:
        lr_series.append((
            "DETR lr/main (phase2)",
            [e for e, _, _ in detr2["lr_rows"]],
            [lr for _, lr, _ in detr2["lr_rows"]],
            {"linestyle": "--"},
        ))
        lr_series.append((
            "DETR lr/backbone (phase2)",
            [e for e, _, _ in detr2["lr_rows"]],
            [lr for _, _, lr in detr2["lr_rows"]],
            {"linestyle": "-."},
        ))

    if yolo1_csv["epochs"]:
        lr_series.append(("YOLO lr/pg0 (phase1)", yolo1_csv["epochs"], yolo1_csv["lr_pg0"], {"alpha": 0.8}))
    elif yolo1["lr_rows"]:
        lr_series.append((
            "YOLO lr/pg0 (phase1)",
            [e for e, _ in yolo1["lr_rows"]],
            [v for _, v in yolo1["lr_rows"]],
            {"alpha": 0.8},
        ))

    if yolo2_csv["epochs"]:
        lr_series.append((
            "YOLO lr/pg0 (phase2)",
            yolo2_csv["epochs"],
            yolo2_csv["lr_pg0"],
            {"linestyle": "--", "alpha": 0.8},
        ))
    elif yolo2["lr_rows"]:
        lr_series.append((
            "YOLO lr/pg0 (phase2)",
            [e for e, _ in yolo2["lr_rows"]],
            [v for _, v in yolo2["lr_rows"]],
            {"linestyle": "--", "alpha": 0.8},
        ))

    lr_x = [x for _, xs, _, _ in lr_series for x in xs]
    lr_x_min, lr_x_max = _min_max(lr_x)
    for label, xs, ys, style in lr_series:
        ax.plot(xs, ys, label=label, **style)

    if lr_x_min is not None and lr_x_max is not None:
        _add_phase_markers(ax, lr_x_min, lr_x_max)
    ax.set_title("Learning rate schedule (log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2) Weight norms
    ax = axes[0, 1]
    weight_series = []
    if detr1["weight_rows"]:
        weight_series.append((
            "DETR weight mean (phase1)",
            [e for e, _ in detr1["weight_rows"]],
            [v for _, v in detr1["weight_rows"]],
            {},
        ))
    if detr2["weight_rows"]:
        weight_series.append((
            "DETR weight mean (phase2)",
            [e for e, _ in detr2["weight_rows"]],
            [v for _, v in detr2["weight_rows"]],
            {"linestyle": "--"},
        ))
    if detr1["weight_max_rows"]:
        weight_series.append((
            "DETR weight max (phase1)",
            [e for e, _ in detr1["weight_max_rows"]],
            [v for _, v in detr1["weight_max_rows"]],
            {"linestyle": ":", "color": "tab:red"},
        ))
    if detr2["weight_max_rows"]:
        weight_series.append((
            "DETR weight max (phase2)",
            [e for e, _ in detr2["weight_max_rows"]],
            [v for _, v in detr2["weight_max_rows"]],
            {"linestyle": "-.", "color": "tab:red"},
        ))

    if yolo1["weight_rows"]:
        weight_series.append((
            "YOLO weight mean (phase1)",
            [e for e, _ in yolo1["weight_rows"]],
            [v for _, v in yolo1["weight_rows"]],
            {"color": "tab:green"},
        ))
    if yolo2["weight_rows"]:
        weight_series.append((
            "YOLO weight mean (phase2)",
            [e for e, _ in yolo2["weight_rows"]],
            [v for _, v in yolo2["weight_rows"]],
            {"linestyle": "--", "color": "tab:green"},
        ))
    if yolo1["weight_max_rows"]:
        weight_series.append((
            "YOLO weight max (phase1)",
            [e for e, _ in yolo1["weight_max_rows"]],
            [v for _, v in yolo1["weight_max_rows"]],
            {"linestyle": ":", "color": "tab:purple"},
        ))
    if yolo2["weight_max_rows"]:
        weight_series.append((
            "YOLO weight max (phase2)",
            [e for e, _ in yolo2["weight_max_rows"]],
            [v for _, v in yolo2["weight_max_rows"]],
            {"linestyle": "-.", "color": "tab:purple"},
        ))

    weight_x = [x for _, xs, _, _ in weight_series for x in xs]
    weight_x_min, weight_x_max = _min_max(weight_x)
    for label, xs, ys, style in weight_series:
        ax.plot(xs, ys, label=label, **style)

    if weight_x_min is not None and weight_x_max is not None:
        _add_phase_markers(ax, weight_x_min, weight_x_max)
    ax.set_title("Weight norm evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight norm")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3) DETR loss
    ax = axes[1, 0]
    detr_series = []
    if detr1["val_loss_rows"]:
        detr_series.append((
            "DETR val loss (phase1)",
            [e for e, _ in detr1["val_loss_rows"]],
            [v for _, v in detr1["val_loss_rows"]],
            {"color": "tab:red"},
        ))
    elif detr1["loss_rows"]:
        detr_series.append((
            "DETR train loss (phase1)",
            [e for e, _ in detr1["loss_rows"]],
            [v for _, v in detr1["loss_rows"]],
            {"color": "tab:red"},
        ))

    if detr2["val_loss_rows"]:
        detr_series.append((
            "DETR val loss (phase2)",
            [e for e, _ in detr2["val_loss_rows"]],
            [v for _, v in detr2["val_loss_rows"]],
            {"color": "tab:red", "linestyle": "--"},
        ))
    elif detr2["loss_rows"]:
        detr_series.append((
            "DETR train loss (phase2)",
            [e for e, _ in detr2["loss_rows"]],
            [v for _, v in detr2["loss_rows"]],
            {"color": "tab:red", "linestyle": "--"},
        ))

    detr_x = [x for _, xs, _, _ in detr_series for x in xs]
    detr_x_min, detr_x_max = _min_max(detr_x)
    for label, xs, ys, style in detr_series:
        ax.plot(xs, ys, label=label, **style)
    if detr_x_min is not None and detr_x_max is not None:
        _add_phase_markers(ax, detr_x_min, detr_x_max)
    ax.set_title("DETR loss (lower is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 4) YOLO metrics
    ax = axes[1, 1]
    yolo_series = []
    if yolo1_csv["epochs"]:
        yolo_series.append((
            "YOLO mAP50-95 (phase1)",
            yolo1_csv["epochs"],
            yolo1_csv["map50_95"],
            {"color": "tab:blue"},
        ))
        yolo_series.append((
            "YOLO mAP50 (phase1)",
            yolo1_csv["epochs"],
            yolo1_csv["map50"],
            {"color": "tab:blue", "linestyle": ":"},
        ))
    if yolo2_csv["epochs"]:
        yolo_series.append((
            "YOLO mAP50-95 (phase2)",
            yolo2_csv["epochs"],
            yolo2_csv["map50_95"],
            {"color": "tab:blue", "linestyle": "--"},
        ))
        yolo_series.append((
            "YOLO mAP50 (phase2)",
            yolo2_csv["epochs"],
            yolo2_csv["map50"],
            {"color": "tab:blue", "linestyle": "-."},
        ))

    yolo_x = [x for _, xs, _, _ in yolo_series for x in xs]
    yolo_x_min, yolo_x_max = _min_max(yolo_x)
    for label, xs, ys, style in yolo_series:
        ax.plot(xs, ys, label=label, **style)
    if yolo_x_min is not None and yolo_x_max is not None:
        _add_phase_markers(ax, yolo_x_min, yolo_x_max)
    ax.set_title("YOLO validation metrics (higher is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    phase_handles = [
        mpatches.Patch(color="tab:gray", alpha=0.08, label=f"Phase 1 ({PHASE1_DATASET})"),
        mpatches.Patch(color="tab:orange", alpha=0.08, label=f"Phase 2 ({PHASE2_DATASET})"),
    ]
    fig.legend(handles=phase_handles, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.88)
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    out_path = output_dir / "detr_vs_yolo_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "detr_vs_yolo_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_detr_layer_evolution(detr1: Dict, detr2: Dict, output_dir: Path) -> Optional[Path]:
    layer_evolution = _merge_layer_evolution(detr1.get("layer_evolution", {}), detr2.get("layer_evolution", {}))
    if not layer_evolution:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    layer_items = list(layer_evolution.items())
    n_layers = len(layer_items)
    cols = 2
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (layer_name, entries) in enumerate(layer_items):
        ax = axes[idx]
        epochs = [e["epoch"] for e in entries]
        norms = [e["norm"] for e in entries]
        ax.plot(epochs, norms, marker="o", linewidth=2)
        if epochs:
            _add_phase_markers(ax, min(epochs), max(epochs))
        ax.set_title(layer_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight norm")
        ax.grid(True, alpha=0.3)

    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    phase_handles = [
        mpatches.Patch(color="tab:gray", alpha=0.08, label=f"Phase 1 ({PHASE1_DATASET})"),
        mpatches.Patch(color="tab:orange", alpha=0.08, label=f"Phase 2 ({PHASE2_DATASET})"),
    ]
    fig.legend(handles=phase_handles, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.9)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    out_path = output_dir / "detr_layer_evolution_compare.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "detr_layer_evolution_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DETR vs YOLO training parameters (two phases).")
    parser.add_argument("--detr-phase1-json", type=str, default=str(DEFAULT_DETR_PHASE1_JSON))
    parser.add_argument("--detr-phase2-json", type=str, default=str(DEFAULT_DETR_PHASE2_JSON))
    parser.add_argument("--yolo-phase1-json", type=str, default=str(DEFAULT_YOLO_PHASE1_JSON))
    parser.add_argument("--yolo-phase2-json", type=str, default=str(DEFAULT_YOLO_PHASE2_JSON))
    parser.add_argument("--yolo-phase1-csv", type=str, default=str(DEFAULT_YOLO_PHASE1_CSV))
    parser.add_argument("--yolo-phase2-csv", type=str, default=str(DEFAULT_YOLO_PHASE2_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detr1 = load_detr(Path(args.detr_phase1_json))
    detr2 = load_detr(Path(args.detr_phase2_json))
    yolo1 = load_yolo(Path(args.yolo_phase1_json))
    yolo2 = load_yolo(Path(args.yolo_phase2_json))
    yolo1_csv = load_yolo_csv(Path(args.yolo_phase1_csv))
    yolo2_csv = load_yolo_csv(Path(args.yolo_phase2_csv))

    # Apply phase epoch ranges
    detr1["lr_rows"] = filter_epoch_rows_three(detr1["lr_rows"], None, PHASE1_MAX_EPOCH)
    detr1["weight_rows"] = filter_epoch_rows(detr1["weight_rows"], None, PHASE1_MAX_EPOCH)
    detr1["weight_max_rows"] = filter_epoch_rows(detr1["weight_max_rows"], None, PHASE1_MAX_EPOCH)
    detr1["val_loss_rows"] = filter_epoch_rows(detr1["val_loss_rows"], None, PHASE1_MAX_EPOCH)
    detr1["loss_rows"] = filter_epoch_rows(detr1["loss_rows"], None, PHASE1_MAX_EPOCH)

    detr2["lr_rows"] = filter_epoch_rows_three(detr2["lr_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    detr2["weight_rows"] = filter_epoch_rows(detr2["weight_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    detr2["weight_max_rows"] = filter_epoch_rows(detr2["weight_max_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    detr2["val_loss_rows"] = filter_epoch_rows(detr2["val_loss_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    detr2["loss_rows"] = filter_epoch_rows(detr2["loss_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)

    yolo1["lr_rows"] = filter_epoch_rows(yolo1["lr_rows"], None, PHASE1_MAX_EPOCH)
    yolo1["weight_rows"] = filter_epoch_rows(yolo1["weight_rows"], None, PHASE1_MAX_EPOCH)
    yolo1["weight_max_rows"] = filter_epoch_rows(yolo1["weight_max_rows"], None, PHASE1_MAX_EPOCH)
    yolo1["weight_std_rows"] = filter_epoch_rows(yolo1["weight_std_rows"], None, PHASE1_MAX_EPOCH)
    yolo1["fitness_rows"] = filter_epoch_rows(yolo1["fitness_rows"], None, PHASE1_MAX_EPOCH)
    yolo1_csv = filter_yolo_csv(yolo1_csv, None, PHASE1_MAX_EPOCH)

    yolo2["lr_rows"] = filter_epoch_rows(yolo2["lr_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    yolo2["weight_rows"] = filter_epoch_rows(yolo2["weight_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    yolo2["weight_max_rows"] = filter_epoch_rows(yolo2["weight_max_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    yolo2["weight_std_rows"] = filter_epoch_rows(yolo2["weight_std_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    yolo2["fitness_rows"] = filter_epoch_rows(yolo2["fitness_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    yolo2_csv = filter_yolo_csv(yolo2_csv, PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)

    phase1 = _phase_summary(detr1, yolo1, yolo1_csv)
    phase2 = _phase_summary(detr2, yolo2, yolo2_csv)

    paths = {
        "detr1": Path(args.detr_phase1_json),
        "detr2": Path(args.detr_phase2_json),
        "yolo1": Path(args.yolo_phase1_json),
        "yolo2": Path(args.yolo_phase2_json),
        "yolo1_csv": Path(args.yolo_phase1_csv),
        "yolo2_csv": Path(args.yolo_phase2_csv),
    }

    plot_path = plot_comparison(detr1, detr2, yolo1, yolo2, yolo1_csv, yolo2_csv, output_dir)
    layer_plot_path = plot_detr_layer_evolution(detr1, detr2, output_dir)

    layer_changes = _summarize_layer_changes(
        _merge_layer_evolution(detr1.get("layer_evolution", {}), detr2.get("layer_evolution", {}))
    )

    output_plots = {
        "Training overview": plot_path,
    }
    if layer_plot_path:
        output_plots["DETR layer evolution"] = layer_plot_path

    report_path = build_report(phase1, phase2, paths, output_dir, output_plots, layer_changes)

    print(f"[OK] Report: {report_path}")
    print(f"[OK] Plot: {plot_path}")
    if layer_plot_path:
        print(f"[OK] Plot: {layer_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
