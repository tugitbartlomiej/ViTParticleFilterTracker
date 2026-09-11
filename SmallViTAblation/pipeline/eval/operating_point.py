#!/usr/bin/env python3
"""operating_point.py — próg operacyjny max-F1 + krzywe F1 + test przeniesienia.

Wszystko liczone Z ZAPISANYCH predykcji (predictions_<set>.json) — bez inferencji.

Zasady protokołu:
  1. Próg operacyjny (max F1) wybierany WYŁĄCZNIE na zbiorze walidacyjnym.
  2. Krzywa F1-vs-próg per zbiór — dane do rysunku (JSON + CSV).
  3. Test przeniesienia progu: próg z walidacji zastosowany do predykcji na
     zbiorze zewnętrznym — metryki, porównanie z F1 na walidacji i diagnostyka
     kalibracji (jaki ułamek trafień leży poniżej progu — zarzut recenzenta).

Wyjście (w --output-dir):
  operating_point.json  — {operating_threshold, threshold_selected_on, f1_curves, transfer}
  f1_curve_<set>.csv    — kolumny: threshold,precision,recall,f1,TP,FP,FN

Przykład:
  py -3.11 pipeline/eval/operating_point.py --protocol configs/protocol.yaml \
      --output-dir results/M-DESF__s42__b5000 --val-set val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from SmallViTAblation.pipeline.eval import common  # noqa: E402


def load_sets(output_dir: Path, protocol: dict | None = None) -> dict[str, dict]:
    """Znajdź zbiory z gotowymi predykcjami + sidecarami w katalogu wyników."""
    sets: dict[str, dict] = {}
    for sidecar_path in sorted(output_dir.glob("evaluated_images_*.json")):
        set_name = sidecar_path.stem[len("evaluated_images_"):]
        sidecar = common.load_json(sidecar_path)
        pred_path = Path(sidecar["predictions_file"])
        if not pred_path.exists():
            print(f"OSTRZEŻENIE: brak {pred_path} — zbiór '{set_name}' pominięty",
                  file=sys.stderr)
            continue
        gt = common.load_gt_dataset(sidecar["annotations_file"])
        gt_trim = common.filter_gt_to_images(gt, {int(i) for i in sidecar["image_ids"]})
        dt = common.load_predictions(pred_path)
        dt = [d for d in dt if int(d["image_id"]) in {int(i) for i in sidecar["image_ids"]}]
        if common.protocol_field(protocol or {}, "eval_agent3", "evaluation",
                                 "single_class_remap", default=True):
            dt, _ = common.remap_dt_categories_single_class(gt_trim, dt)
        sets[set_name] = {
            "gt_index": common.build_gt_index(gt_trim),
            "dt": dt,
            "sidecar": sidecar,
        }
    return sets


def write_curve_csv(path: Path, curve: dict) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "precision", "recall", "f1", "TP", "FP", "FN"])
        for i, thr in enumerate(curve["threshold"]):
            writer.writerow([
                f"{thr:.6f}", f"{curve['precision'][i]:.6f}", f"{curve['recall'][i]:.6f}",
                f"{curve['f1'][i]:.6f}", curve["TP"][i], curve["FP"][i], curve["FN"][i],
            ])


def main(argv: list[str] | None = None) -> int:
    common.ensure_utf8_io()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path,
                   help="katalog wyników biegu z predictions_<set>.json")
    p.add_argument("--val-set", default="val", choices=["val"],
                   help="nazwa zbioru walidacyjnego (na nim WYŁĄCZNIE wybierany próg)")
    p.add_argument("--sets", default=None,
                   help="lista zbiorów po przecinku (domyślnie: wszystkie znalezione)")
    args = p.parse_args(argv)

    protocol = common.load_protocol(args.protocol)
    iou_thr = float(common.protocol_field(protocol, "evaluation", "iou_threshold"))
    max_points = int(common.protocol_field(
        protocol, "eval_agent3", "operating_point", "f1_curve_max_points"))
    probe_thr = float(common.protocol_field(
        protocol, "eval_agent3", "operating_point", "low_confidence_probe_threshold"))

    found = load_sets(args.output_dir, protocol)
    if not found:
        print(f"BŁĄD: {args.output_dir} nie zawiera predictions_<set>.json + sidecarów "
              f"(najpierw uruchom evaluate.py)", file=sys.stderr)
        return 2
    if args.sets:
        wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
        missing = [s for s in wanted if s not in found]
        if missing:
            print(f"BŁĄD: brak danych dla zbiorów: {missing}", file=sys.stderr)
            return 2
        found = {s: found[s] for s in wanted}
    if args.val_set not in found:
        print(f"BŁĄD: zbiór walidacyjny '{args.val_set}' nie ma predykcji w {args.output_dir}. "
              f"Próg operacyjny wolno wybierać TYLKO na walidacji.", file=sys.stderr)
        return 2

    # 1) próg max-F1 WYŁĄCZNIE na walidacji
    val = found[args.val_set]
    val_scores = np.asarray([float(d["score"]) for d in val["dt"]], dtype=float)
    val_thresholds = common.threshold_candidates(val_scores, max_points)
    val_curve = common.f1_curve(val["gt_index"], val["dt"], val_thresholds, iou_thr)
    operating_threshold, val_best_f1 = common.best_f1_threshold(val_curve)

    # 2) krzywe F1 per zbiór (zawsze z punktem progu operacyjnego)
    curves: dict[str, dict] = {args.val_set: val_curve}
    for set_name, data in found.items():
        if set_name == args.val_set:
            continue
        scores = np.asarray([float(d["score"]) for d in data["dt"]], dtype=float)
        thresholds = common.threshold_candidates(
            scores, max_points, extra=[operating_threshold])
        curves[set_name] = common.f1_curve(data["gt_index"], data["dt"], thresholds, iou_thr)

    for set_name, curve in curves.items():
        write_curve_csv(args.output_dir / f"f1_curve_{set_name}.csv", curve)

    # 3) test przeniesienia progu: próg z walidacji -> pozostałe zbiory
    transfer: dict[str, dict] = {}
    for set_name, data in found.items():
        at_val = common.counting_metrics(
            data["gt_index"], data["dt"], operating_threshold, iou_thr)
        own_best_thr, own_best_f1 = common.best_f1_threshold(curves[set_name])
        # diagnostyka kalibracji: dopasowanie IGNORUJĄC ufność (wszystkie predykcje)
        all_matched = common.match_detections(data["gt_index"], data["dt"], 0.0, iou_thr)
        tp_scores = np.asarray(all_matched["tp_scores"], dtype=float)
        frac_below_probe = (
            float(np.mean(tp_scores < probe_thr)) if tp_scores.size else None)
        frac_below_val_thr = (
            float(np.mean(tp_scores < operating_threshold)) if tp_scores.size else None)
        transfer[set_name] = {
            "f1_at_val_threshold": at_val["f1"],
            "precision_at_val_threshold": at_val["precision"],
            "recall_at_val_threshold": at_val["recall"],
            "TP": at_val["TP"], "FP": at_val["FP"], "FN": at_val["FN"],
            "val_f1_at_same_threshold": (
                val_best_f1 if set_name == args.val_set
                else next((curves[args.val_set]["f1"][i]
                           for i, t in enumerate(curves[args.val_set]["threshold"])
                           if abs(t - operating_threshold) < 1e-12), val_best_f1)),
            "best_f1_on_set": own_best_f1,
            "best_threshold_on_set": own_best_thr,
            "f1_loss_vs_best_on_set": own_best_f1 - at_val["f1"],
            "f1_gap_vs_val": at_val["f1"] - (
                val_best_f1 if set_name == args.val_set
                else next((curves[args.val_set]["f1"][i]
                           for i, t in enumerate(curves[args.val_set]["threshold"])
                           if abs(t - operating_threshold) < 1e-12), val_best_f1)),
            "tp_matched_ignoring_confidence": int(all_matched["tp"]),
            "frac_tp_below_probe_threshold": frac_below_probe,
            "frac_tp_below_val_threshold": frac_below_val_thr,
            "probe_threshold": probe_thr,
        }

    out = {
        "operating_threshold": float(operating_threshold),
        "threshold_selected_on": args.val_set,
        "selection_rule": "max_f1_on_val_only",
        "iou_threshold": iou_thr,
        "val_best_f1": val_best_f1,
        "f1_curves": curves,
        "transfer": transfer,
    }
    path = common.save_json(out, args.output_dir / "operating_point.json")

    print(f"Próg operacyjny (max F1 na '{args.val_set}'): {operating_threshold:.4f} "
          f"(F1={val_best_f1:.4f})")
    for set_name, tr in transfer.items():
        print(f"  [{set_name}] F1@próg_val={tr['f1_at_val_threshold']:.4f}  "
              f"best_F1_na_zbiorze={tr['best_f1_on_set']:.4f} (próg {tr['best_threshold_on_set']:.4f})  "
              f"Δ={tr['f1_loss_vs_best_on_set']:+.4f}  "
              f"TP<próg_val={tr['frac_tp_below_val_threshold']}")
    print(f"Zapisano {path} + krzywe f1_curve_<set>.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
