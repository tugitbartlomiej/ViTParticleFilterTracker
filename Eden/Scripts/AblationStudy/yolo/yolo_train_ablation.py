#!/usr/bin/env python
"""
YOLOv8 ablation training script for Eden HPC.

Analogiczne do detr/detr_train_ablation.py ale uzywa Ultralytics YOLO API.
Match z produkcyjnym YOLO 20k fine-tune (cos_lr=false, batch=32, lr0=1e-4).

Variants:
  YE-A: YOLO-EL2N -> 20k subset -> YOLOv8 fine-tune
  YE-B: YOLO-EL2N -> 20k subset -> DETR fine-tune (cross-arch comparison)
  YE-C: DETR-EL2N -> 20k subset -> YOLOv8 fine-tune (cross-arch comparison)

Reference produkcyjnego YOLO config:
  Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune.py
  args.yaml: cos_lr=false, optimizer=SGD, lr0=0.0001, batch=32, epochs=160

Usage (TODO):
  python yolo_train_ablation.py \
    --variant ye_a \
    --weights /path/to/yolov8m_ep170.pt \
    --data /path/to/variant_dataset.yaml \
    --epochs 160 \
    --batch 32 \
    --output ./runs/ye_a
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics required. pip install ultralytics", file=sys.stderr)
    sys.exit(1)


VARIANT_CONFIG = {
    'ye_a': {'el2n_source': 'yolo', 'target_arch': 'yolo'},
    'ye_b': {'el2n_source': 'yolo', 'target_arch': 'detr'},
    'ye_c': {'el2n_source': 'detr', 'target_arch': 'yolo'},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO ablation training")
    parser.add_argument('--variant', required=True, choices=list(VARIANT_CONFIG.keys()))
    parser.add_argument('--weights', type=Path, required=True,
                        help='Pretrained YOLOv8 weights (typowo ep170 base)')
    parser.add_argument('--data', type=Path, required=True,
                        help='dataset.yaml file (Ultralytics format)')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr0', type=float, default=0.0001)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0,1,2,3')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VARIANT_CONFIG[args.variant]
    if cfg['target_arch'] != 'yolo':
        print(f"[ERROR] variant {args.variant} target_arch={cfg['target_arch']} — uzyj detr_train_ablation.py", file=sys.stderr)
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[YOLO ABLATION] variant={args.variant} weights={args.weights} data={args.data}")

    model = YOLO(str(args.weights))
    # TODO: wywolaj model.train() z parametrami match do produkcyjnego YOLO 20k:
    #   cos_lr=False, optimizer='SGD', lr0=args.lr0, batch=args.batch,
    #   epochs=args.epochs, imgsz=args.imgsz, device=args.device,
    #   project=str(args.output), name=args.variant
    raise NotImplementedError("Implement model.train(...) call")


if __name__ == '__main__':
    main()
