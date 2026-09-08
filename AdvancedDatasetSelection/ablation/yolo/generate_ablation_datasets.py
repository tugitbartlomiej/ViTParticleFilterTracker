"""
YOLO-EL2N based dataset selection for ablation variants.

Analogiczne do detr/generate_ablation_datasets.py, ale uzywa YOLOv8 ep170
do EL2N scoring zamiast DETR. Generuje 3 dataset variants z tej samej puli
91,336 frames z augmented_coco.

Variants:
  YE-A: YOLO-EL2N (YOLOv8 ep170) -> 20k subset -> YOLOv8 fine-tune
  YE-B: YOLO-EL2N (YOLOv8 ep170) -> 20k subset -> DETR fine-tune
  YE-C: DETR-EL2N (DETR ep170)   -> 20k subset -> YOLOv8 fine-tune

Wszystkie variants:
  - Uzywaja tego samego video-level splita (../generate_video_split.py)
  - Wybieraja 20,000 obrazow z TRAIN videos
  - Wynikowe COCO annotations dla train/val/test
  - Pakuja obrazy do .tar archives

Usage (TODO):
  python generate_ablation_datasets.py \
    --images-dir ../../Eden/Datasets/.../images \
    --annotations ../../Eden/Datasets/.../augmented_coco_20250417_030014.json \
    --split ../generate_video_split.py output \
    --yolo-el2n yolo_el2n_scores.json \
    --detr-el2n ../../detr_el2n_scores.json \
    --output ./output/ablation_yolo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


VARIANTS = {
    'ye_a': {'el2n_source': 'yolo', 'target_arch': 'yolo'},
    'ye_b': {'el2n_source': 'yolo', 'target_arch': 'detr'},
    'ye_c': {'el2n_source': 'detr', 'target_arch': 'yolo'},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO ablation datasets")
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True,
                        help='Augmented COCO annotations (91,336 frames)')
    parser.add_argument('--split', type=Path, required=True,
                        help='video_split.json with train/val/test video IDs')
    parser.add_argument('--yolo-el2n', type=Path, required=True,
                        help='Per-frame loss scores from YOLOv8 ep170')
    parser.add_argument('--detr-el2n', type=Path, required=True,
                        help='Per-frame loss scores from DETR ep170')
    parser.add_argument('--target-size', type=int, default=20000)
    parser.add_argument('--output', type=Path, default=Path('./output/ablation_yolo'))
    return parser.parse_args()


def select_by_el2n(scores_path: Path, train_image_ids: Set[int],
                   target_size: int) -> List[int]:
    """TODO: Top-K (high EL2N = hard examples) lub stratified sampling."""
    raise NotImplementedError("Implement EL2N-based selection")


def write_variant(variant_id: str, selected_ids: List[int],
                  annotations: Dict, output_dir: Path) -> None:
    """TODO: Write COCO annotation subset + tar images for variant."""
    raise NotImplementedError("Implement variant packaging")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as f:
        annotations = json.load(f)
    with open(args.split) as f:
        split = json.load(f)
    train_image_ids: Set[int] = set(split.get('train', []))

    for variant_id, cfg in VARIANTS.items():
        scores_path = args.yolo_el2n if cfg['el2n_source'] == 'yolo' else args.detr_el2n
        print(f"[VARIANT {variant_id}] el2n_source={cfg['el2n_source']} target={cfg['target_arch']}")
        # selected = select_by_el2n(scores_path, train_image_ids, args.target_size)
        # write_variant(variant_id, selected, annotations, args.output / variant_id)


if __name__ == '__main__':
    main()
