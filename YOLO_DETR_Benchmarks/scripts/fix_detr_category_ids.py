#!/usr/bin/env python3
"""
Fix DETR predictions by mapping category_id 0 → 1 for COCO evaluation
DETR trained with category_id=0 for tooltip, but COCO requires category_id=1
"""

import json
from pathlib import Path

def fix_category_ids(input_file, output_file):
    """Map category_id 0 → 1 in DETR predictions"""

    print(f"Loading predictions from: {input_file}")
    with open(input_file, 'r') as f:
        predictions = json.load(f)

    print(f"Total predictions: {len(predictions)}")

    # Count category IDs
    category_counts = {}
    for pred in predictions:
        cat_id = pred['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

    print(f"Category distribution before fix: {category_counts}")

    # Remap category_id 0 → 1
    fixed_count = 0
    for pred in predictions:
        if pred['category_id'] == 0:
            pred['category_id'] = 1
            fixed_count += 1

    print(f"Fixed {fixed_count} predictions (0 -> 1)")

    # Save fixed predictions
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"✅ Saved fixed predictions to: {output_file}")

    return predictions


if __name__ == "__main__":
    benchmark_dir = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Benchmarks/VALIDATION_YOLO_DETR_20251113_031106")

    input_file = benchmark_dir / "detr_validation_predictions.json"
    output_file = benchmark_dir / "detr_validation_predictions_fixed.json"

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        exit(1)

    fix_category_ids(input_file, output_file)
