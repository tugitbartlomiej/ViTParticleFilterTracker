#!/usr/bin/env python3
"""
YOLO vs DETR Validation Benchmark
Tests on PROPER VALIDATION SET (not training data!)
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection

# Fix for legacy YOLO checkpoints with PyTorch 2.6+
import ultralytics.utils.loss as loss_module
if not hasattr(loss_module, 'DFLoss'):
    class DFLoss:
        """Compatibility stub for legacy DFLoss class"""
        pass
    loss_module.DFLoss = DFLoss
    sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss

# Monkey-patch torch.load to use weights_only=False for trusted YOLO checkpoints
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "Advanced_Analysis"))
from run_inference import load_detr_from_checkpoint

# Configuration - VALIDATION SET
YOLO_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Models/YOLO/epoch100.pt"
DETR_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR/checkpoint_epoch_100.pth"

# PROPER VALIDATION SET - NOT TRAINING DATA!
COCO_ANNOTATIONS = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Datasets/TooltipMining/annotations/val_annotations.json"
IMAGES_DIR = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Datasets/TooltipMining/val"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Benchmarks") / f"VALIDATION_YOLO_DETR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_yolo_benchmark(coco, image_id_map, output_dir):
    """Run YOLO Epoch 100 benchmark on validation set"""
    print(f"\n{'='*60}")
    print("Running YOLO Epoch 100 Validation Benchmark")
    print(f"{'='*60}\n")

    # Load YOLO model (compatibility fixes applied at module level)
    print(f"Loading YOLO from: {YOLO_CHECKPOINT}")
    try:
        model = YOLO(YOLO_CHECKPOINT)
        model.to(DEVICE)
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        import traceback
        traceback.print_exc()
        return None

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())
    print(f"Processing {len(image_ids)} validation images...")

    for img_id in tqdm(image_ids, desc="YOLO inference"):
        img_path = image_id_map[img_id]

        if not Path(img_path).exists():
            print(f"⚠️  Warning: Image not found: {img_path}")
            continue

        # Run inference
        start_time = time.time()
        results = model.predict(img_path, conf=0.5, iou=0.5, verbose=False)
        total_time += time.time() - start_time

        # Process results
        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                # Convert to COCO format [x, y, w, h]
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Map to COCO category (YOLO class 0 -> COCO category 1)
                category_id = 1

                predictions.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf)
                })

        image_count += 1

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "yolo_validation_predictions.json"

    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nYOLO Predictions: {len(predictions)}")
    print(f"Processing time: {total_time:.2f}s ({image_count/total_time:.2f} FPS)")

    # Evaluate
    if predictions:
        coco_dt = coco.loadRes(str(pred_file))
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "model": "YOLO Epoch 100",
            "checkpoint": str(YOLO_CHECKPOINT),
            "dataset": "Validation Set (56 images)",
            "mAP_0.5": float(coco_eval.stats[1]) * 100,
            "mAP_0.5:0.95": float(coco_eval.stats[0]) * 100,
            "mAP_0.75": float(coco_eval.stats[2]) * 100,
            "mAP_small": float(coco_eval.stats[3]) * 100,
            "mAP_medium": float(coco_eval.stats[4]) * 100,
            "mAP_large": float(coco_eval.stats[5]) * 100,
            "AR_max_1": float(coco_eval.stats[6]) * 100,
            "AR_max_10": float(coco_eval.stats[7]) * 100,
            "AR_max_100": float(coco_eval.stats[8]) * 100,
            "AR_small": float(coco_eval.stats[9]) * 100,
            "AR_medium": float(coco_eval.stats[10]) * 100,
            "AR_large": float(coco_eval.stats[11]) * 100,
            "total_predictions": len(predictions),
            "fps": image_count / total_time,
            "total_time": total_time
        }

        # Save metrics
        with open(output_dir / "yolo_validation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
    else:
        print("❌ No predictions generated")
        return None


def run_detr_benchmark(coco, image_id_map, output_dir):
    """Run DETR Epoch 100 benchmark on validation set"""
    print(f"\n{'='*60}")
    print("Running DETR Epoch 100 Validation Benchmark")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading DETR from: {DETR_CHECKPOINT}")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = load_detr_from_checkpoint(DETR_CHECKPOINT, device=DEVICE)
    model.eval()
    print("✅ DETR model loaded successfully")

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())
    print(f"Processing {len(image_ids)} validation images...")

    with torch.no_grad():
        for img_id in tqdm(image_ids, desc="DETR inference"):
            img_path = image_id_map[img_id]

            if not Path(img_path).exists():
                print(f"⚠️  Warning: Image not found: {img_path}")
                continue

            # Load and process image
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)

            # Run inference
            start_time = time.time()
            outputs = model(**inputs)
            total_time += time.time() - start_time

            # Process predictions (conf threshold = 0.5, iou = 0.5)
            target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]

            # Extract predictions
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score = float(score.cpu())
                label = int(label.cpu())
                box = box.cpu().numpy()

                # Convert to COCO format
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                # DETR already uses category_id = 1 for tooltip
                category_id = label

                predictions.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })

            image_count += 1

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "detr_validation_predictions.json"

    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nDETR Predictions: {len(predictions)}")
    print(f"Processing time: {total_time:.2f}s ({image_count/total_time:.2f} FPS)")

    # Evaluate
    if predictions:
        coco_dt = coco.loadRes(str(pred_file))
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "model": "DETR Epoch 100",
            "checkpoint": str(DETR_CHECKPOINT),
            "dataset": "Validation Set (56 images)",
            "mAP_0.5": float(coco_eval.stats[1]) * 100,
            "mAP_0.5:0.95": float(coco_eval.stats[0]) * 100,
            "mAP_0.75": float(coco_eval.stats[2]) * 100,
            "mAP_small": float(coco_eval.stats[3]) * 100,
            "mAP_medium": float(coco_eval.stats[4]) * 100,
            "mAP_large": float(coco_eval.stats[5]) * 100,
            "AR_max_1": float(coco_eval.stats[6]) * 100,
            "AR_max_10": float(coco_eval.stats[7]) * 100,
            "AR_max_100": float(coco_eval.stats[8]) * 100,
            "AR_small": float(coco_eval.stats[9]) * 100,
            "AR_medium": float(coco_eval.stats[10]) * 100,
            "AR_large": float(coco_eval.stats[11]) * 100,
            "total_predictions": len(predictions),
            "fps": image_count / total_time,
            "total_time": total_time
        }

        # Save metrics
        with open(output_dir / "detr_validation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
    else:
        print("❌ No predictions generated")
        return None


def generate_comparison_report(yolo_metrics, detr_metrics, output_dir):
    """Generate detailed comparison report"""

    print(f"\n{'='*60}")
    print("GENERATING VALIDATION BENCHMARK REPORT")
    print(f"{'='*60}\n")

    # Calculate gaps
    gaps = {
        "mAP_0.5": yolo_metrics["mAP_0.5"] - detr_metrics["mAP_0.5"],
        "mAP_0.5:0.95": yolo_metrics["mAP_0.5:0.95"] - detr_metrics["mAP_0.5:0.95"],
        "mAP_0.75": yolo_metrics["mAP_0.75"] - detr_metrics["mAP_0.75"],
        "AR_max_100": yolo_metrics["AR_max_100"] - detr_metrics["AR_max_100"],
        "AR_medium": yolo_metrics["AR_medium"] - detr_metrics["AR_medium"],
        "fps": yolo_metrics["fps"] - detr_metrics["fps"]
    }

    # Determine winner
    winner = "YOLO" if yolo_metrics["mAP_0.5:0.95"] > detr_metrics["mAP_0.5:0.95"] else "DETR"

    # Create markdown report
    report = f"""# YOLO vs DETR - Proper Validation Benchmark Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** Validation Set (56 images) - UNSEEN DATA
**Winner:** {winner}

---

## ⚠️ CRITICAL: This Uses PROPER Validation Data

**Previous benchmarks were INVALID** - they tested on training data (218 images).
This benchmark uses the **proper validation set** (56 unseen images).

---

## Executive Summary

### Performance Comparison

| Metric | YOLO Epoch 100 | DETR Epoch 100 | Gap | Winner |
|--------|----------------|----------------|-----|--------|
| **mAP@0.5** | {yolo_metrics['mAP_0.5']:.2f}% | {detr_metrics['mAP_0.5']:.2f}% | {gaps['mAP_0.5']:+.2f}% | {'🥇 YOLO' if gaps['mAP_0.5'] > 0 else '🥇 DETR'} |
| **mAP@0.5:0.95** | {yolo_metrics['mAP_0.5:0.95']:.2f}% | {detr_metrics['mAP_0.5:0.95']:.2f}% | {gaps['mAP_0.5:0.95']:+.2f}% | {'🥇 YOLO' if gaps['mAP_0.5:0.95'] > 0 else '🥇 DETR'} |
| **mAP@0.75** | {yolo_metrics['mAP_0.75']:.2f}% | {detr_metrics['mAP_0.75']:.2f}% | {gaps['mAP_0.75']:+.2f}% | {'🥇 YOLO' if gaps['mAP_0.75'] > 0 else '🥇 DETR'} |
| **AR@100** | {yolo_metrics['AR_max_100']:.2f}% | {detr_metrics['AR_max_100']:.2f}% | {gaps['AR_max_100']:+.2f}% | {'🥇 YOLO' if gaps['AR_max_100'] > 0 else '🥇 DETR'} |

### Speed Comparison

| Model | FPS | Total Time | Detections |
|-------|-----|------------|------------|
| **YOLO Epoch 100** | {yolo_metrics['fps']:.2f} | {yolo_metrics['total_time']:.2f}s | {yolo_metrics['total_predictions']} |
| **DETR Epoch 100** | {detr_metrics['fps']:.2f} | {detr_metrics['total_time']:.2f}s | {detr_metrics['total_predictions']} |
| **Gap** | {gaps['fps']:+.2f} FPS | {detr_metrics['total_time'] - yolo_metrics['total_time']:+.2f}s | {yolo_metrics['total_predictions'] - detr_metrics['total_predictions']:+d} |

---

## Detailed Metrics

### Detection Quality (Medium Objects - Surgical Tools)

| Metric | YOLO | DETR | Gap |
|--------|------|------|-----|
| **mAP Medium** | {yolo_metrics['mAP_medium']:.2f}% | {detr_metrics['mAP_medium']:.2f}% | {(yolo_metrics['mAP_medium'] - detr_metrics['mAP_medium']):+.2f}% |
| **AR Medium** | {yolo_metrics['AR_medium']:.2f}% | {detr_metrics['AR_medium']:.2f}% | {gaps['AR_medium']:+.2f}% |

### All Size Categories

| Model | Small | Medium | Large |
|-------|-------|--------|-------|
| **YOLO mAP** | {yolo_metrics['mAP_small']:.2f}% | {yolo_metrics['mAP_medium']:.2f}% | {yolo_metrics['mAP_large']:.2f}% |
| **DETR mAP** | {detr_metrics['mAP_small']:.2f}% | {detr_metrics['mAP_medium']:.2f}% | {detr_metrics['mAP_large']:.2f}% |
| **YOLO AR** | {yolo_metrics['AR_small']:.2f}% | {yolo_metrics['AR_medium']:.2f}% | {yolo_metrics['AR_large']:.2f}% |
| **DETR AR** | {detr_metrics['AR_small']:.2f}% | {detr_metrics['AR_medium']:.2f}% | {detr_metrics['AR_large']:.2f}% |

---

## Key Findings

### 1. Validation Performance vs Training Performance

**Previous (INVALID) Training Set Results:**
- YOLO: 84.82% mAP@0.5, 78.46% mAP@0.5:0.95
- DETR: 83.10% mAP@0.5, 63.76% mAP@0.5:0.95

**Current (VALID) Validation Set Results:**
- YOLO: {yolo_metrics['mAP_0.5']:.2f}% mAP@0.5, {yolo_metrics['mAP_0.5:0.95']:.2f}% mAP@0.5:0.95
- DETR: {detr_metrics['mAP_0.5']:.2f}% mAP@0.5, {detr_metrics['mAP_0.5:0.95']:.2f}% mAP@0.5:0.95

**Generalization Gap:**
- YOLO drop: {84.82 - yolo_metrics['mAP_0.5']:.2f}% (mAP@0.5)
- DETR drop: {83.10 - detr_metrics['mAP_0.5']:.2f}% (mAP@0.5)

### 2. Detection Accuracy

- mAP@0.5 gap: **{gaps['mAP_0.5']:.2f}%**
- mAP@0.5:0.95 gap: **{gaps['mAP_0.5:0.95']:.2f}%**
- {'YOLO maintains lead on unseen data' if gaps['mAP_0.5:0.95'] > 0 else 'DETR performs better on unseen data'}

### 3. Recall Performance

- AR@100 gap: **{gaps['AR_max_100']:.2f}%**
- Medium object AR gap: **{gaps['AR_medium']:.2f}%**
- {'YOLO finds more objects' if gaps['AR_max_100'] > 5 else 'Similar recall performance'}

### 4. Speed & Efficiency

- YOLO is **{yolo_metrics['fps']/detr_metrics['fps']:.2f}x faster** than DETR
- FPS gap: **{gaps['fps']:.2f} FPS**
- Better for real-time surgical applications

---

## Conclusions

### Winner: {winner}

**Reasons:**
1. {'✅ Better accuracy on unseen data' if winner == 'YOLO' else '✅ Better late-stage generalization'}
2. {'✅ Higher recall - finds more surgical tools' if winner == 'YOLO' and gaps['AR_max_100'] > 5 else '✅ Competitive recall performance'}
3. {'✅ Faster inference for real-time use' if winner == 'YOLO' else '✅ Transformer architecture advantages'}

### Overfitting Analysis

**YOLO:**
- Training mAP@0.5: 84.82%
- Validation mAP@0.5: {yolo_metrics['mAP_0.5']:.2f}%
- **Overfitting:** {84.82 - yolo_metrics['mAP_0.5']:.2f}% drop

**DETR:**
- Training mAP@0.5: 83.10%
- Validation mAP@0.5: {detr_metrics['mAP_0.5']:.2f}%
- **Overfitting:** {83.10 - detr_metrics['mAP_0.5']:.2f}% drop

---

## Recommendations

### For Production Use:
**Recommended:** {winner}
- {'Better generalization to unseen data' if winner == 'YOLO' else 'DETR shows better feature learning'}
- {'Faster inference for real-time applications' if winner == 'YOLO' else 'Consider RT-DETR for speed improvements'}

### For Research:
- Investigate regularization techniques to reduce overfitting
- Test with larger validation set for more robust metrics
- Consider data augmentation to improve generalization

---

## Files Generated

- `yolo_validation_predictions.json` - YOLO detection results
- `yolo_validation_metrics.json` - YOLO metrics
- `detr_validation_predictions.json` - DETR detection results
- `detr_validation_metrics.json` - DETR metrics
- `comparison_summary.json` - Complete comparison data
- `VALIDATION_BENCHMARK_REPORT.md` - This report

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** Validation Set (56 unseen images)
**Models:** YOLO Epoch 100 vs DETR Epoch 100
"""

    # Save markdown report
    report_file = output_dir / "VALIDATION_BENCHMARK_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # Save JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "validation_set_56_images",
        "winner": winner,
        "yolo_metrics": yolo_metrics,
        "detr_metrics": detr_metrics,
        "gaps": gaps
    }

    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Report saved to: {report_file}")
    print(f"✅ Summary saved to: {summary_file}")

    return summary


def main():
    print("="*60)
    print("YOLO vs DETR - PROPER Validation Benchmark")
    print("="*60)
    print("\n⚠️  Testing on VALIDATION SET (56 unseen images)")
    print("Previous benchmarks tested on TRAINING data (invalid!)\n")

    # Verify files exist
    if not Path(YOLO_CHECKPOINT).exists():
        print(f"❌ ERROR: YOLO checkpoint not found: {YOLO_CHECKPOINT}")
        return

    if not Path(DETR_CHECKPOINT).exists():
        print(f"❌ ERROR: DETR checkpoint not found: {DETR_CHECKPOINT}")
        return

    if not Path(COCO_ANNOTATIONS).exists():
        print(f"❌ ERROR: Validation annotations not found: {COCO_ANNOTATIONS}")
        return

    # Load COCO annotations
    print("\nLoading validation annotations...")
    coco = COCO(COCO_ANNOTATIONS)

    # Build image ID map
    print("Building image ID map...")
    image_id_map = {}
    for img in coco.dataset.get('images', []):
        img_id = img['id']
        img_filename = img['file_name']

        # Build full path
        img_path = Path(IMAGES_DIR) / img_filename

        if not img_path.exists():
            print(f"⚠️  Warning: Image not found: {img_path}")
            continue

        image_id_map[img_id] = str(img_path)

    print(f"✅ Found {len(image_id_map)} validation images")

    # Run benchmarks
    yolo_metrics = run_yolo_benchmark(coco, image_id_map, OUTPUT_DIR)
    detr_metrics = run_detr_benchmark(coco, image_id_map, OUTPUT_DIR)

    # Generate comparison
    if yolo_metrics and detr_metrics:
        summary = generate_comparison_report(yolo_metrics, detr_metrics, OUTPUT_DIR)

        # Print final summary
        print("\n" + "="*60)
        print("VALIDATION BENCHMARK RESULTS (UNSEEN DATA)")
        print("="*60)
        print(f"\n🏆 Winner: {summary['winner']}\n")

        print(f"YOLO Epoch 100:")
        print(f"  mAP@0.5:0.95: {yolo_metrics['mAP_0.5:0.95']:.2f}%")
        print(f"  mAP@0.5:     {yolo_metrics['mAP_0.5']:.2f}%")
        print(f"  AR@100:      {yolo_metrics['AR_max_100']:.2f}%")
        print(f"  FPS:         {yolo_metrics['fps']:.2f}")
        print(f"  Detections:  {yolo_metrics['total_predictions']}")

        print(f"\nDETR Epoch 100:")
        print(f"  mAP@0.5:0.95: {detr_metrics['mAP_0.5:0.95']:.2f}%")
        print(f"  mAP@0.5:     {detr_metrics['mAP_0.5']:.2f}%")
        print(f"  AR@100:      {detr_metrics['AR_max_100']:.2f}%")
        print(f"  FPS:         {detr_metrics['fps']:.2f}")
        print(f"  Detections:  {detr_metrics['total_predictions']}")

        print(f"\nPerformance Gaps:")
        print(f"  mAP@0.5:0.95: {summary['gaps']['mAP_0.5:0.95']:+.2f}%")
        print(f"  mAP@0.5:     {summary['gaps']['mAP_0.5']:+.2f}%")
        print(f"  AR@100:      {summary['gaps']['AR_max_100']:+.2f}%")
        print(f"  FPS:         {summary['gaps']['fps']:+.2f}")

        print(f"\n📊 Full report: {OUTPUT_DIR / 'VALIDATION_BENCHMARK_REPORT.md'}")
    else:
        print("\n❌ Benchmark failed - could not generate comparison")
        if not yolo_metrics:
            print("   - YOLO benchmark failed")
        if not detr_metrics:
            print("   - DETR benchmark failed")


if __name__ == "__main__":
    main()
