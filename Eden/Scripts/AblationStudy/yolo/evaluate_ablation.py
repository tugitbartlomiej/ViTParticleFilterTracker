#!/usr/bin/env python
"""
YOLOv8 ablation evaluation script for Eden HPC.

Analogiczne do detr/evaluate_ablation.py ale dla YOLOv8. Ewaluuje best.pt
kazdego variantu na:
  1. Same-distribution: val split
  2. Cross-dataset: Roboflow Cataract Surgery Instruments (external benchmark)

Wynik: tabela mAP@0.5 / mAP@0.5:0.95 / TP/FP/FN per variant.

Usage (TODO):
  python evaluate_ablation.py \
    --ablation-dir /mnt/evafs/.../Yolo/ablation \
    --val-data /path/to/val_dataset.yaml \
    --external-data /path/to/external_dataset.yaml \
    --output ablation_yolo_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics required. pip install ultralytics", file=sys.stderr)
    sys.exit(1)


VARIANTS = ['ye_a', 'ye_b', 'ye_c']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO ablation variants")
    parser.add_argument('--ablation-dir', type=Path, required=True)
    parser.add_argument('--val-data', type=Path, required=True)
    parser.add_argument('--external-data', type=Path, required=True)
    parser.add_argument('--leaked-images', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path('./ablation_yolo_results.json'))
    return parser.parse_args()


def evaluate_variant(variant_id: str, weights_path: Path,
                     data_yaml: Path) -> Dict[str, float]:
    """TODO: Wywolaj YOLO(weights).val(data=data_yaml) i wyciagnij metryki."""
    raise NotImplementedError("Implement YOLO .val() pipeline")


def main() -> None:
    args = parse_args()
    results: Dict[str, Dict] = {}

    for variant_id in VARIANTS:
        weights = args.ablation_dir / variant_id / 'weights' / 'best.pt'
        if not weights.exists():
            print(f"[SKIP] {variant_id}: {weights} not found")
            continue
        print(f"[EVAL] {variant_id} weights={weights}")
        # results[variant_id] = {
        #     'same_dist': evaluate_variant(variant_id, weights, args.val_data),
        #     'external': evaluate_variant(variant_id, weights, args.external_data),
        # }  # TODO

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
