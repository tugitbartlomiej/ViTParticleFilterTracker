"""
Compare DETR vs YOLO training parameters across two phases.

Usage (recommended):
  set PYTHONNOUSERSITE=1
  py -3.11 compare_detr_yolo_training.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


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


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_detr(detr_json: Path) -> Dict:
    obj = _load_json(detr_json)

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
        loss_rows = [(c.get("epoch"), c.get("loss"))
                     for c in ckpts if c.get("epoch") is not None and c.get("loss") is not None]

        return {
            "header": header,
            "args": args,
            "lr_rows": lr_rows,
            "val_loss_rows": val_loss_rows,
            "weight_rows": weight_rows,
            "loss_rows": loss_rows,
        }

    summary = obj.get("summary", [])
    summary = [r for r in summary if r.get("epoch") is not None]
    summary.sort(key=lambda r: r["epoch"])

    lr_rows = [(r.get("epoch"), (r.get("learning_rates") or [None])[0],
               (r.get("learning_rates") or [None, None])[1] if r.get("learning_rates") else None)
               for r in summary if r.get("learning_rates")]
    weight_rows = [(r.get("epoch"), r.get("weight_norm_mean"))
                   for r in summary if r.get("weight_norm_mean") is not None]
    loss_rows = [(r.get("epoch"), r.get("loss"))
                 for r in summary if r.get("loss") is not None]

    return {
        "header": {},
        "args": {},
        "lr_rows": lr_rows,
        "val_loss_rows": [],
        "weight_rows": weight_rows,
        "loss_rows": loss_rows,
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
    lr_rows = [(r.get("epoch"), (r.get("learning_rates") or [None])[0])
               for r in rows if r.get("learning_rates")]
    fitness_rows = [(r.get("epoch"), r.get("best_fitness"))
                    for r in rows if r.get("best_fitness") is not None]

    return {
        "rows": rows,
        "train_args": train_args,
        "weight_rows": weight_rows,
        "lr_rows": lr_rows,
        "fitness_rows": fitness_rows,
    }


def load_yolo_csv(results_csv: Path) -> Dict:
    if not results_csv.exists():
        return {"epochs": [], "lr_pg0": [], "map50_95": [], "map50": []}

    epochs = []
    lr_pg0 = []
    map50_95 = []
    map50 = []

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(float(row["epoch"])))
                lr_pg0.append(float(row["lr/pg0"]))
                map50_95.append(float(row["metrics/mAP50-95(B)"]))
                map50.append(float(row["metrics/mAP50(B)"]))
            except Exception:
                continue

    return {"epochs": epochs, "lr_pg0": lr_pg0, "map50_95": map50_95, "map50": map50}

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
        "map50_95": [data["map50_95"][i] for i in keep],
        "map50": [data["map50"][i] for i in keep],
    }


def _first_last(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    return vals[0], vals[-1]


def _min_max(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    return min(vals), max(vals)


def _normalize_vals(vals: List[float], vmin: Optional[float], vmax: Optional[float]) -> List[float]:
    if not vals or vmin is None or vmax is None:
        return vals
    if vmax == vmin:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


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
    yolo_weight_vals = [v for _, v in yolo["weight_rows"] if v is not None]

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
    yolo_weight_min, yolo_weight_max = _min_max(yolo_weight_vals)

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
        "yolo_weight_min": yolo_weight_min,
        "yolo_weight_max": yolo_weight_max,
    }


def build_report(
    phase1: Dict,
    phase2: Dict,
    paths: Dict[str, Path],
    output_dir: Path,
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

    lines.append("## Phase 1 (100k, epoch <= 170)")
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
    lines.append("")

    lines.append("## Phase 2 (fine-tuning, 170-200)")
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
    lines.append("")

    report_path = output_dir / "detr_vs_yolo_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def plot_comparison(
    detr1: Dict,
    detr2: Dict,
    yolo1: Dict,
    yolo2: Dict,
    yolo1_csv: Dict,
    yolo2_csv: Dict,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    # 1) LR schedule
    ax = axes[0]
    lr_series = []
    if detr1["lr_rows"]:
        lr_series.append((
            "DETR lr/main (phase1)",
            [e for e, _, _ in detr1["lr_rows"]],
            [lr for _, lr, _ in detr1["lr_rows"]],
            {},
        ))
    if detr2["lr_rows"]:
        lr_series.append((
            "DETR lr/main (phase2)",
            [e for e, _, _ in detr2["lr_rows"]],
            [lr for _, lr, _ in detr2["lr_rows"]],
            {"linestyle": "--"},
        ))

    if yolo1_csv["epochs"]:
        lr_series.append(("YOLO lr/pg0 (phase1)", yolo1_csv["epochs"], yolo1_csv["lr_pg0"], {}))
    elif yolo1["lr_rows"]:
        lr_series.append((
            "YOLO lr/pg0 (phase1)",
            [e for e, _ in yolo1["lr_rows"]],
            [v for _, v in yolo1["lr_rows"]],
            {},
        ))

    if yolo2_csv["epochs"]:
        lr_series.append((
            "YOLO lr/pg0 (phase2)",
            yolo2_csv["epochs"],
            yolo2_csv["lr_pg0"],
            {"linestyle": "--"},
        ))
    elif yolo2["lr_rows"]:
        lr_series.append((
            "YOLO lr/pg0 (phase2)",
            [e for e, _ in yolo2["lr_rows"]],
            [v for _, v in yolo2["lr_rows"]],
            {"linestyle": "--"},
        ))

    lr_x = [x for _, xs, _, _ in lr_series for x in xs]
    lr_y = [y for _, _, ys, _ in lr_series for y in ys]
    lr_x_min, lr_x_max = _min_max(lr_x)
    lr_y_min, lr_y_max = _min_max(lr_y)

    for label, xs, ys, style in lr_series:
        ax.plot(
            _normalize_vals(xs, lr_x_min, lr_x_max),
            _normalize_vals(ys, lr_y_min, lr_y_max),
            label=label,
            **style,
        )

    ax.set_title("Learning rate schedule")
    ax.set_xlabel("Normalized epoch")
    ax.set_ylabel("Normalized LR")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) Weight norm mean
    ax = axes[1]
    weight_series = []
    if detr1["weight_rows"]:
        weight_series.append((
            "DETR weight norm (phase1)",
            [e for e, _ in detr1["weight_rows"]],
            [v for _, v in detr1["weight_rows"]],
            {},
        ))
    if detr2["weight_rows"]:
        weight_series.append((
            "DETR weight norm (phase2)",
            [e for e, _ in detr2["weight_rows"]],
            [v for _, v in detr2["weight_rows"]],
            {"linestyle": "--"},
        ))

    if yolo1["weight_rows"]:
        weight_series.append((
            "YOLO weight norm (phase1)",
            [e for e, _ in yolo1["weight_rows"]],
            [v for _, v in yolo1["weight_rows"]],
            {},
        ))
    if yolo2["weight_rows"]:
        weight_series.append((
            "YOLO weight norm (phase2)",
            [e for e, _ in yolo2["weight_rows"]],
            [v for _, v in yolo2["weight_rows"]],
            {"linestyle": "--"},
        ))

    weight_x = [x for _, xs, _, _ in weight_series for x in xs]
    weight_y = [y for _, _, ys, _ in weight_series for y in ys]
    weight_x_min, weight_x_max = _min_max(weight_x)
    weight_y_min, weight_y_max = _min_max(weight_y)

    for label, xs, ys, style in weight_series:
        ax.plot(
            _normalize_vals(xs, weight_x_min, weight_x_max),
            _normalize_vals(ys, weight_y_min, weight_y_max),
            label=label,
            **style,
        )

    ax.set_title("Weight norm evolution")
    ax.set_xlabel("Normalized epoch")
    ax.set_ylabel("Normalized mean weight norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Performance proxy
    ax = axes[2]
    detr_perf_series = []
    if detr1["val_loss_rows"]:
        detr_perf_series.append((
            "DETR val loss (phase1)",
            [e for e, _ in detr1["val_loss_rows"]],
            [v for _, v in detr1["val_loss_rows"]],
            {"color": "tab:red"},
        ))
    elif detr1["loss_rows"]:
        detr_perf_series.append((
            "DETR train loss (phase1)",
            [e for e, _ in detr1["loss_rows"]],
            [v for _, v in detr1["loss_rows"]],
            {"color": "tab:red"},
        ))

    if detr2["val_loss_rows"]:
        detr_perf_series.append((
            "DETR val loss (phase2)",
            [e for e, _ in detr2["val_loss_rows"]],
            [v for _, v in detr2["val_loss_rows"]],
            {"color": "tab:red", "linestyle": "--"},
        ))
    elif detr2["loss_rows"]:
        detr_perf_series.append((
            "DETR train loss (phase2)",
            [e for e, _ in detr2["loss_rows"]],
            [v for _, v in detr2["loss_rows"]],
            {"color": "tab:red", "linestyle": "--"},
        ))

    yolo_perf_series = []
    if yolo1_csv["epochs"] and yolo1_csv["map50_95"]:
        yolo_perf_series.append((
            "YOLO mAP50-95 (phase1)",
            yolo1_csv["epochs"],
            yolo1_csv["map50_95"],
            {"color": "tab:blue"},
        ))
    elif yolo1["fitness_rows"]:
        yolo_perf_series.append((
            "YOLO fitness (phase1)",
            [e for e, _ in yolo1["fitness_rows"]],
            [v for _, v in yolo1["fitness_rows"]],
            {"color": "tab:blue"},
        ))

    if yolo2_csv["epochs"] and yolo2_csv["map50_95"]:
        yolo_perf_series.append((
            "YOLO mAP50-95 (phase2)",
            yolo2_csv["epochs"],
            yolo2_csv["map50_95"],
            {"color": "tab:blue", "linestyle": "--"},
        ))
    elif yolo2["fitness_rows"]:
        yolo_perf_series.append((
            "YOLO fitness (phase2)",
            [e for e, _ in yolo2["fitness_rows"]],
            [v for _, v in yolo2["fitness_rows"]],
            {"color": "tab:blue", "linestyle": "--"},
        ))

    perf_x = [x for _, xs, _, _ in detr_perf_series for x in xs]
    perf_x += [x for _, xs, _, _ in yolo_perf_series for x in xs]
    perf_x_min, perf_x_max = _min_max(perf_x)

    detr_y = [y for _, _, ys, _ in detr_perf_series for y in ys]
    detr_y_min, detr_y_max = _min_max(detr_y)

    yolo_y = [y for _, _, ys, _ in yolo_perf_series for y in ys]
    yolo_y_min, yolo_y_max = _min_max(yolo_y)

    for label, xs, ys, style in detr_perf_series:
        ax.plot(
            _normalize_vals(xs, perf_x_min, perf_x_max),
            _normalize_vals(ys, detr_y_min, detr_y_max),
            label=label,
            **style,
        )

    ax.set_title("Performance over training")
    ax.set_xlabel("Normalized epoch")
    ax.set_ylabel("Normalized DETR loss", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    for label, xs, ys, style in yolo_perf_series:
        ax2.plot(
            _normalize_vals(xs, perf_x_min, perf_x_max),
            _normalize_vals(ys, yolo_y_min, yolo_y_max),
            label=label,
            **style,
        )

    ax2.set_ylabel("Normalized YOLO metric", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="lower right")

    plt.tight_layout()
    out_path = output_dir / "detr_vs_yolo_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "detr_vs_yolo_comparison.pdf", bbox_inches="tight")
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
    detr1["val_loss_rows"] = filter_epoch_rows(detr1["val_loss_rows"], None, PHASE1_MAX_EPOCH)
    detr1["loss_rows"] = filter_epoch_rows(detr1["loss_rows"], None, PHASE1_MAX_EPOCH)

    detr2["lr_rows"] = filter_epoch_rows_three(detr2["lr_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    detr2["weight_rows"] = filter_epoch_rows(detr2["weight_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    detr2["val_loss_rows"] = filter_epoch_rows(detr2["val_loss_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    detr2["loss_rows"] = filter_epoch_rows(detr2["loss_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)

    yolo1["lr_rows"] = filter_epoch_rows(yolo1["lr_rows"], None, PHASE1_MAX_EPOCH)
    yolo1["weight_rows"] = filter_epoch_rows(yolo1["weight_rows"], None, PHASE1_MAX_EPOCH)
    yolo1["fitness_rows"] = filter_epoch_rows(yolo1["fitness_rows"], None, PHASE1_MAX_EPOCH)
    yolo1_csv = filter_yolo_csv(yolo1_csv, None, PHASE1_MAX_EPOCH)

    yolo2["lr_rows"] = filter_epoch_rows(yolo2["lr_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
    yolo2["weight_rows"] = filter_epoch_rows(yolo2["weight_rows"], PHASE2_MIN_EPOCH, PHASE2_MAX_EPOCH)
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

    report_path = build_report(phase1, phase2, paths, output_dir)
    plot_path = plot_comparison(detr1, detr2, yolo1, yolo2, yolo1_csv, yolo2_csv, output_dir)

    print(f"[OK] Report: {report_path}")
    print(f"[OK] Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
