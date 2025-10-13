import argparse
import os
import sys
import json
import time

from pathlib import Path

# Ensure local imports work when running from this folder
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import yaml
from pycocotools.coco import COCO

from run_inference import load_config, run_yolo_inference, run_detr_inference, resolve_path
from evaluate_metrics import calculate_standard_metrics, calculate_temporal_stability


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run full YOLO vs DETR benchmark (inference + metrics).")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO inference stage")
    parser.add_argument("--skip-detr", action="store_true", help="Skip DETR inference stage")
    parser.add_argument("--only", choices=["yolo", "detr"], help="Run only one model end-to-end", default=None)
    parser.add_argument("--config", default=str(HERE / "config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["directory"]
    ensure_output_dir(output_dir)

    # Inference
    if args.only == "yolo":
        run_yolo_inference(config)
    elif args.only == "detr":
        run_detr_inference(config)
    else:
        if not args.skip_yolo:
            run_yolo_inference(config)
        if not args.skip_detr:
            run_detr_inference(config)

    # Evaluation
    gt_path = resolve_path(
        config['dataset']['annotations_path'],
        expect_dir=False,
    )
    gt_coco = COCO(gt_path)

    report = {}
    for model_type in ([args.only] if args.only else ["yolo", "detr"]):
        pred_path = os.path.join(output_dir, f"{model_type}_predictions.json")
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing predictions for {model_type}: {pred_path}")
            continue

        with open(pred_path, 'r') as f:
            preds = json.load(f)
        img_ids = [p['image_id'] for p in preds]

        pred_coco = gt_coco.loadRes(pred_path)
        std_stats = calculate_standard_metrics(gt_coco, pred_coco, img_ids=img_ids)
        temp_stats = calculate_temporal_stability(preds)

        # Load measured performance if present
        perf = None
        perf_path = os.path.join(output_dir, f"{model_type}_performance.json")
        if os.path.exists(perf_path):
            try:
                with open(perf_path, 'r') as f:
                    perf = json.load(f)
            except Exception:
                perf = None

        report[model_type] = {
            "standard_metrics": std_stats,
            "temporal_metrics": temp_stats,
            "performance_metrics": perf,
        }

    # Save consolidated report
    consolidated = os.path.join(output_dir, "benchmark_report.json")
    with open(consolidated, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n[OK] Full benchmark report -> {consolidated}")


if __name__ == "__main__":
    main()

