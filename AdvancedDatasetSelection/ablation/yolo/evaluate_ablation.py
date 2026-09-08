"""
YOLOv8 ablation evaluation.

Analogiczne do detr/evaluate_ablation.py, ale uzywa Ultralytics YOLO API zamiast
HuggingFace transformers DETR. Ewaluuje kazdy variant ablacji na:
  1. Same-distribution: val split (z video_split.json)
  2. Cross-dataset: Roboflow Cataract Surgery Instruments (external benchmark)

Variants (do dopracowania w trakcie implementacji):
  - YE-A: YOLO-EL2N -> 20k subset -> YOLOv8 fine-tune
  - YE-B: YOLO-EL2N -> 20k subset -> DETR fine-tune (cross-arch porownanie)
  - YE-C: DETR-EL2N -> 20k subset -> YOLOv8 fine-tune (cross-arch porownanie)

Usage (TODO):
  python evaluate_ablation.py \
    --ablation-dir /path/to/ablation \
    --images-dir /path/to/all/images \
    --external-dir E:/cataract_surgery_Instruments_detection.v1i.coco \
    --leaked-images ../../YOLO_DETR_Benchmarks/scripts/leaked_valid_images.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics required. pip install ultralytics")
    sys.exit(1)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("ERROR: pycocotools required. pip install pycocotools")
    sys.exit(1)


VARIANTS = {
    'ye_a': 'YOLO-EL2N -> YOLOv8',
    'ye_b': 'YOLO-EL2N -> DETR',
    'ye_c': 'DETR-EL2N -> YOLOv8',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO ablation variants")
    parser.add_argument('--ablation-dir', type=Path, required=True,
                        help='Root directory with variant subfolders')
    parser.add_argument('--images-dir', type=Path, required=True,
                        help='Directory with all augmented pool images')
    parser.add_argument('--external-dir', type=Path, required=True,
                        help='Roboflow external benchmark root')
    parser.add_argument('--leaked-images', type=Path, default=None,
                        help='JSON list of leaked filenames to exclude')
    parser.add_argument('--output', type=Path, default=Path('./ablation_results.json'))
    return parser.parse_args()


def evaluate_variant(variant_id: str, weights_path: Path,
                     val_data: Path, external_data: Path,
                     leaked: Optional[List[str]] = None) -> Dict:
    """TODO: Run Ultralytics .val() on same-distribution + external benchmark."""
    raise NotImplementedError("Implement YOLO variant evaluation pipeline")


def main() -> None:
    args = parse_args()
    results: Dict[str, Dict] = {}
    for variant_id, variant_name in VARIANTS.items():
        weights_path = args.ablation_dir / variant_id / 'weights' / 'best.pt'
        if not weights_path.exists():
            print(f"[SKIP] {variant_id}: weights not found at {weights_path}")
            continue
        print(f"[EVAL] {variant_id} ({variant_name})")
        # results[variant_id] = evaluate_variant(...)  # TODO

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
