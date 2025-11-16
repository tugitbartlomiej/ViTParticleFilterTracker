#!/usr/bin/env python3
"""
Re-evaluate DETR with fixed category IDs (0 -> 1)
"""

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path

# Paths
ANNOTATIONS = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Datasets/TooltipMining/annotations/val_annotations.json"
PREDICTIONS_FIXED = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Benchmarks/VALIDATION_YOLO_DETR_20251113_031106/detr_validation_predictions_fixed.json"

print("="*60)
print("DETR Re-Evaluation with Fixed Category IDs")
print("="*60)

# Load annotations
print("\nLoading validation annotations...")
coco = COCO(ANNOTATIONS)

# Load fixed predictions
print(f"Loading fixed predictions from: {PREDICTIONS_FIXED}")
coco_dt = coco.loadRes(str(PREDICTIONS_FIXED))

# Run evaluation
print("\nRunning COCO evaluation...")
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract metrics
print("\n" + "="*60)
print("DETR Epoch 100 - CORRECTED RESULTS")
print("="*60)

metrics = {
    "mAP@0.5:0.95": coco_eval.stats[0] * 100,
    "mAP@0.5": coco_eval.stats[1] * 100,
    "mAP@0.75": coco_eval.stats[2] * 100,
    "AR@100": coco_eval.stats[8] * 100,
    "AR_medium": coco_eval.stats[10] * 100,
}

print(f"\nmAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.2f}%")
print(f"mAP@0.5:     {metrics['mAP@0.5']:.2f}%")
print(f"mAP@0.75:    {metrics['mAP@0.75']:.2f}%")
print(f"AR@100:      {metrics['AR@100']:.2f}%")
print(f"AR Medium:   {metrics['AR_medium']:.2f}%")

print("\n" + "="*60)
print("Comparison: YOLO vs DETR (Corrected)")
print("="*60)

# YOLO results from previous benchmark
yolo_map_50 = 83.79
yolo_map_50_95 = 75.87
yolo_ar_100 = 92.61

print(f"\n{'Metric':<20} {'YOLO':<12} {'DETR':<12} {'Gap'}")
print(f"{'-'*60}")
print(f"{'mAP@0.5':<20} {yolo_map_50:>6.2f}%     {metrics['mAP@0.5']:>6.2f}%     {yolo_map_50 - metrics['mAP@0.5']:+6.2f}%")
print(f"{'mAP@0.5:0.95':<20} {yolo_map_50_95:>6.2f}%     {metrics['mAP@0.5:0.95']:>6.2f}%     {yolo_map_50_95 - metrics['mAP@0.5:0.95']:+6.2f}%")
print(f"{'AR@100':<20} {yolo_ar_100:>6.2f}%     {metrics['AR@100']:>6.2f}%     {yolo_ar_100 - metrics['AR@100']:+6.2f}%")

if metrics['mAP@0.5:0.95'] > yolo_map_50_95:
    print("\nWinner: DETR")
else:
    print("\nWinner: YOLO")
