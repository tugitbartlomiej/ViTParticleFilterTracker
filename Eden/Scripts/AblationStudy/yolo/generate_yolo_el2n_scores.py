#!/usr/bin/env python
"""
Extract per-frame loss (EL2N proxy) z YOLOv8 ep170 dla pelnej puli 91,336 frames.

Output: yolo_el2n_scores.json
  {
    "image_id_or_filename": float_loss,
    ...
  }

EL2N (Error L2 Norm) heurystyka: per-sample loss (lub mAP miss-rate) na ep170 przed
fine-tuningiem. Wysokie loss = "trudny" przyklad. Wybor 20k z najwyzszymi loss
to YOLO-EL2N selection (analogia do DETR-EL2N w glownym pipeline DINOv3+SAM+EL2N).

Implementacja (TODO):
  1. Zaladuj YOLO ep170 weights w trybie eval.
  2. Iteruj po wszystkich 91,336 obrazach z augmented_coco_*.json.
  3. Dla kazdego obrazu: forward pass, oblicz total loss (box + cls + dfl).
  4. Zapisz {image_id: loss_value} do JSON.

Usage (TODO):
  python generate_yolo_el2n_scores.py \
    --weights /path/to/yolov8m_ep170.pt \
    --images-dir /path/to/augmented_pool \
    --annotations augmented_coco_20250417_030014.json \
    --output yolo_el2n_scores.json \
    --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("ERROR: torch + ultralytics required.", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-EL2N per-frame scores")
    parser.add_argument('--weights', type=Path, required=True,
                        help='YOLOv8 ep170 base checkpoint')
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True,
                        help='Augmented COCO annotations (91,336 frames)')
    parser.add_argument('--output', type=Path, default=Path('./yolo_el2n_scores.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=16)
    return parser.parse_args()


def compute_per_image_loss(model: 'YOLO', image_paths, device: str) -> Dict[str, float]:
    """TODO: Forward pass z labels -> sum (box + cls + dfl) per sample."""
    raise NotImplementedError("Implement per-image loss extraction from YOLO")


def main() -> None:
    args = parse_args()
    print(f"[YOLO-EL2N] weights={args.weights} images_dir={args.images_dir}")

    with open(args.annotations) as f:
        coco = json.load(f)

    # model = YOLO(str(args.weights))  # zaladuj
    # scores = compute_per_image_loss(model, ...)  # TODO

    scores: Dict[str, float] = {}
    args.output.write_text(json.dumps(scores, indent=2))
    print(f"Saved scores for {len(scores)} images -> {args.output}")


if __name__ == '__main__':
    main()
