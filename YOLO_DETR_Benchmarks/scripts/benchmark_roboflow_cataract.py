#!/usr/bin/env python3
"""
Complete YOLO vs DETR Benchmark on Roboflow Cataract Surgery Instruments Dataset
Tests domain generalization: models trained on tooltips, tested on surgical instruments

Dataset: E:/cataract_surgery_Instruments_detection.v1i.coco
- Train: 2083 images, 4 instrument classes
- Test: 50 images
- Valid: 248 images

Models:
- YOLO Epoch 70 (trained on tooltips)
- DETR Epoch 100 (trained on tooltips)

All instrument categories mapped to class 1 for single-class evaluation.
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import time

# ==================== YOLO LEGACY LOADING FIX ====================
print("Setting up YOLO legacy checkpoint compatibility...")
import ultralytics.utils.loss as loss_module

if not hasattr(loss_module, 'DFLoss'):
    class DFLoss:
        """Compatibility stub for legacy DFLoss class removed in newer ultralytics"""
        pass
    loss_module.DFLoss = DFLoss
    sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss
    print("✓ DFLoss stub installed")

# Monkey-patch torch.load to use weights_only=False for legacy checkpoints
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
print("✓ torch.load patched for legacy checkpoint support")

# ==================== CONFIGURATION ====================
YOLO_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/YOLO_EDEN_TRAIN/epoch70.pt"
DETR_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR/checkpoint_epoch_100.pth"

# Roboflow Cataract Dataset
DATASET_ROOT = Path("E:/cataract_surgery_Instruments_detection.v1i.coco")
DATASETS = {
    "train": {
        "annotations": DATASET_ROOT / "train" / "_annotations.coco.json",
        "images_dir": DATASET_ROOT / "train"
    },
    "test": {
        "annotations": DATASET_ROOT / "test" / "_annotations.coco.json",
        "images_dir": DATASET_ROOT / "test"
    },
    "valid": {
        "annotations": DATASET_ROOT / "valid" / "_annotations.coco.json",
        "images_dir": DATASET_ROOT / "valid"
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Benchmarks")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BENCHMARK_DIR = OUTPUT_DIR / f"ROBOFLOW_CATARACT_YOLO70_DETR100_{TIMESTAMP}"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*80}")
print("ROBOFLOW CATARACT INSTRUMENTS BENCHMARK")
print(f"{'='*80}")
print(f"Models: YOLO Epoch 70 vs DETR Epoch 100")
print(f"Task: Domain Generalization (Tooltip -> Surgical Instruments)")
print(f"Device: {DEVICE}")
print(f"Output: {BENCHMARK_DIR}")
print(f"{'='*80}\n")

# ==================== LOAD MODELS ====================
print("Loading YOLO Epoch 70...")
from ultralytics import YOLO
yolo_model = YOLO(YOLO_CHECKPOINT)
yolo_model.to(DEVICE)
print(f"✓ YOLO loaded from {YOLO_CHECKPOINT}")

print("\nLoading DETR Epoch 100...")
from transformers import DetrImageProcessor, DetrForObjectDetection

detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=1,
    ignore_mismatched_sizes=True
)
checkpoint = torch.load(DETR_CHECKPOINT, map_location=DEVICE)
detr_model.load_state_dict(checkpoint['model_state_dict'])
detr_model.to(DEVICE)
detr_model.eval()
print(f"✓ DETR loaded from {DETR_CHECKPOINT}")

# ==================== HELPER FUNCTION ====================
def map_categories_to_single_class(coco, target_category_id=1):
    """
    Map all tool categories (knife, sideport_knife, phaco_probe) to single category_id

    Original categories:
    - 0: cataract-surgery-instruments (parent)
    - 1: knife
    - 2: sideport_knife
    - 3: phaco_probe

    All → category_id=1 (tool/tooltip)
    """
    print("\n🔧 Category Mapping:")
    print("Original categories:")
    for cat in coco.dataset['categories']:
        print(f"  - ID {cat['id']}: {cat['name']}")

    print(f"\nMapping all tool categories → category_id={target_category_id} (tool/tooltip)")

    # Update annotations to use single category
    for ann in coco.dataset['annotations']:
        if ann['category_id'] > 0:  # Skip parent category 0
            ann['category_id'] = target_category_id

    # Update categories list to single class
    coco.dataset['categories'] = [
        {'id': target_category_id, 'name': 'tool', 'supercategory': 'surgical-instrument'}
    ]

    # Reload COCO with updated dataset
    coco.createIndex()

    print(f"✅ All tool annotations mapped to category_id={target_category_id}")
    return coco


def build_image_id_map(coco, images_dir):
    """Build mapping from image_id to full image path"""
    image_id_map = {}
    for img in coco.dataset.get('images', []):
        img_id = img['id']
        img_filename = img['file_name']
        img_path = Path(images_dir) / img_filename
        image_id_map[img_id] = str(img_path)
    return image_id_map


def run_yolo_benchmark(coco, image_id_map, output_dir):
    """Run YOLO Epoch 100 benchmark"""
    print(f"\n{'='*60}")
    print("Running YOLO Epoch 100 Benchmark (Roboflow Dataset)")
    print(f"{'='*60}\n")

    # Allow pickle for YOLO
    _allowlist_ultralytics_pickle_classes()

    # Load YOLO model
    print(f"Loading YOLO from: {YOLO_CHECKPOINT}")
    model = YOLO(YOLO_CHECKPOINT)
    model.to(DEVICE)

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())

    for img_id in tqdm(image_ids, desc="YOLO inference"):
        img_path = image_id_map[img_id]

        if not Path(img_path).exists():
            print(f"⚠️  Image not found: {img_path}")
            continue

        # Run inference
        start_time = time.time()
        results = model.predict(img_path, conf=0.25, verbose=False)
        total_time += time.time() - start_time

        # Process results - map all detections to category_id=1
        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                # Convert to COCO format [x, y, w, h]
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # All detections → category_id=1 (tool/tooltip)
                predictions.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf)
                })

        image_count += 1

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "yolo_predictions.json"

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
            "dataset": "Roboflow Cataract Surgery (test)",
            "images": image_count,
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_100": float(coco_eval.stats[8]),
            "total_predictions": len(predictions),
            "fps": image_count / total_time,
            "total_time": total_time
        }

        # Save metrics
        with open(output_dir / "yolo_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
    else:
        return None


def run_detr_benchmark(coco, image_id_map, output_dir):
    """Run DETR Epoch 100 benchmark"""
    print(f"\n{'='*60}")
    print("Running DETR Epoch 100 Benchmark (Roboflow Dataset)")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading DETR from: {DETR_CHECKPOINT}")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = load_detr_from_checkpoint(DETR_CHECKPOINT, device=DEVICE)
    model.eval()

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())

    with torch.no_grad():
        for img_id in tqdm(image_ids, desc="DETR inference"):
            img_path = image_id_map[img_id]

            if not Path(img_path).exists():
                print(f"⚠️  Image not found: {img_path}")
                continue

            # Load and process image
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)

            # Run inference
            start_time = time.time()
            outputs = model(**inputs)
            total_time += time.time() - start_time

            # Process predictions (conf threshold = 0.25 to match YOLO)
            target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, threshold=0.25, target_sizes=target_sizes
            )[0]

            # Extract predictions - map all to category_id=1
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score = float(score.cpu())
                box = box.cpu().numpy()

                # Convert to COCO format
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                # All detections → category_id=1 (tool/tooltip)
                predictions.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })

            image_count += 1

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "detr_predictions.json"

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
            "dataset": "Roboflow Cataract Surgery (test)",
            "images": image_count,
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_100": float(coco_eval.stats[8]),
            "total_predictions": len(predictions),
            "fps": image_count / total_time,
            "total_time": total_time
        }

        # Save metrics
        with open(output_dir / "detr_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
    else:
        return None


def generate_comparison_report(yolo_metrics, detr_metrics, output_dir):
    """Generate comprehensive comparison report"""

    gaps = {
        "mAP_0.5": (yolo_metrics["mAP_0.5"] - detr_metrics["mAP_0.5"]) * 100,
        "mAP_0.5:0.95": (yolo_metrics["mAP_0.5:0.95"] - detr_metrics["mAP_0.5:0.95"]) * 100,
        "mAP_0.75": (yolo_metrics["mAP_0.75"] - detr_metrics["mAP_0.75"]) * 100,
        "AR_max_100": (yolo_metrics["AR_max_100"] - detr_metrics["AR_max_100"]) * 100,
        "fps": yolo_metrics["fps"] - detr_metrics["fps"]
    }

    winner = "YOLO" if yolo_metrics["mAP_0.5:0.95"] > detr_metrics["mAP_0.5:0.95"] else "DETR"

    report = f"""# YOLO vs DETR Benchmark - Roboflow Cataract Surgery Dataset

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** External Validation (Roboflow)
**Test Images:** 50 (unseen surgical tool images)
**Winner:** {winner}

---

## Executive Summary

### Performance Comparison

| Metric | YOLO Epoch 100 | DETR Epoch 100 | Gap | Winner |
|--------|----------------|----------------|-----|--------|
| **mAP@0.5** | {yolo_metrics['mAP_0.5']*100:.2f}% | {detr_metrics['mAP_0.5']*100:.2f}% | {gaps['mAP_0.5']:+.2f}% | {'🥇 YOLO' if gaps['mAP_0.5'] > 0 else '🥇 DETR'} |
| **mAP@0.5:0.95** | {yolo_metrics['mAP_0.5:0.95']*100:.2f}% | {detr_metrics['mAP_0.5:0.95']*100:.2f}% | {gaps['mAP_0.5:0.95']:+.2f}% | {'🥇 YOLO' if gaps['mAP_0.5:0.95'] > 0 else '🥇 DETR'} |
| **mAP@0.75** | {yolo_metrics['mAP_0.75']*100:.2f}% | {detr_metrics['mAP_0.75']*100:.2f}% | {gaps['mAP_0.75']:+.2f}% | {'🥇 YOLO' if gaps['mAP_0.75'] > 0 else '🥇 DETR'} |
| **AR@100** | {yolo_metrics['AR_max_100']*100:.2f}% | {detr_metrics['AR_max_100']*100:.2f}% | {gaps['AR_max_100']:+.2f}% | {'🥇 YOLO' if gaps['AR_max_100'] > 0 else '🥇 DETR'} |

### Speed Comparison

| Model | FPS | Total Time |
|-------|-----|------------|
| **YOLO** | {yolo_metrics['fps']:.2f} | {yolo_metrics['total_time']:.2f}s |
| **DETR** | {detr_metrics['fps']:.2f} | {detr_metrics['total_time']:.2f}s |
| **Gap** | {gaps['fps']:+.2f} FPS | {detr_metrics['total_time'] - yolo_metrics['total_time']:+.2f}s |

---

## Dataset Details

**Source:** Roboflow Cataract Surgery Instruments Detection v1
**Test Set:** 50 images (external validation)
**Original Classes:** 4 (knife, sideport_knife, phaco_probe, parent)
**Mapped to:** Single class "tool/tooltip"

**Why This Dataset is Better:**
- ✅ **External validation** - not used in training
- ✅ **Professional annotations** - Roboflow quality
- ✅ **Multiple tool types** - diverse surgical instruments
- ✅ **Real generalization test** - unseen data

---

## Key Findings

### 1. Generalization Performance
- **YOLO:** {yolo_metrics['mAP_0.5:0.95']*100:.1f}% mAP@0.5:0.95 on unseen data
- **DETR:** {detr_metrics['mAP_0.5:0.95']*100:.1f}% mAP@0.5:0.95 on unseen data
- **Gap:** {gaps['mAP_0.5:0.95']:.1f}% {'(YOLO better)' if gaps['mAP_0.5:0.95'] > 0 else '(DETR better)'}

### 2. Comparison to Training Set Results
**Previous (TooltipMining train set - 218 images):**
- YOLO: 84.82% mAP@0.5, 78.46% mAP@0.5:0.95
- DETR: 83.10% mAP@0.5, 63.76% mAP@0.5:0.95

**Current (Roboflow test set - 50 images):**
- YOLO: {yolo_metrics['mAP_0.5']*100:.2f}% mAP@0.5, {yolo_metrics['mAP_0.5:0.95']*100:.2f}% mAP@0.5:0.95
- DETR: {detr_metrics['mAP_0.5']*100:.2f}% mAP@0.5, {detr_metrics['mAP_0.5:0.95']*100:.2f}% mAP@0.5:0.95

**Generalization Drop:**
- YOLO mAP@0.5:0.95: {78.46 - yolo_metrics['mAP_0.5:0.95']*100:.2f}% drop
- DETR mAP@0.5:0.95: {63.76 - detr_metrics['mAP_0.5:0.95']*100:.2f}% drop

---

## Conclusions

### Winner: {winner}
- mAP@0.5:0.95 gap: **{gaps['mAP_0.5:0.95']:.2f}%**
- Recall gap: **{gaps['AR_max_100']:.2f}%**
- Speed advantage: **{gaps['fps']:.2f} FPS**

### Training Set Overfit Analysis
{'Both models show significant performance drop on external data, indicating overfitting to TooltipMining training set.' if yolo_metrics['mAP_0.5:0.95']*100 < 70 else 'Models generalize reasonably well to external validation data.'}

---

## Files Generated
- `yolo_predictions.json` - YOLO detection results
- `yolo_metrics.json` - YOLO metrics
- `detr_predictions.json` - DETR detection results
- `detr_metrics.json` - DETR metrics
- `comparison_summary.json` - Complete comparison data
- `BENCHMARK_REPORT.md` - This report

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Save report
    report_file = output_dir / "BENCHMARK_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # Save JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "Roboflow Cataract Surgery (test)",
        "test_images": 50,
        "winner": winner,
        "yolo_metrics": yolo_metrics,
        "detr_metrics": detr_metrics,
        "gaps": gaps
    }

    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Report saved to: {report_file}")
    print(f"✅ Summary saved to: {summary_file}")

    return summary


def main():
    print("="*60)
    print("YOLO vs DETR - Roboflow Cataract Surgery Benchmark")
    print("External Validation on Unseen Data")
    print("="*60)

    # Verify checkpoints
    if not Path(YOLO_CHECKPOINT).exists():
        print(f"❌ ERROR: YOLO checkpoint not found: {YOLO_CHECKPOINT}")
        return

    if not Path(DETR_CHECKPOINT).exists():
        print(f"❌ ERROR: DETR checkpoint not found: {DETR_CHECKPOINT}")
        return

    # Load COCO annotations
    print(f"\nLoading test annotations from: {TEST_ANNOTATIONS}")
    coco = COCO(TEST_ANNOTATIONS)

    # Map all categories to single class
    coco = map_categories_to_single_class(coco, target_category_id=1)

    # Build image ID map
    print(f"\nBuilding image paths from: {TEST_IMAGES_DIR}")
    image_id_map = build_image_id_map(coco, TEST_IMAGES_DIR)
    print(f"Found {len(image_id_map)} test images")

    # Run benchmarks (YOLO disabled due to PyTorch 2.6+ pickle compatibility)
    print("\n⚠️  YOLO benchmark skipped (PyTorch 2.6+ pickle compatibility issue)")
    print("Running DETR benchmark only for external validation...\n")

    yolo_metrics = None
    detr_metrics = run_detr_benchmark(coco, image_id_map, OUTPUT_DIR)

    # Generate comparison or single model report
    if detr_metrics:
        print("\n" + "="*60)
        print("DETR EXTERNAL VALIDATION RESULTS")
        print("="*60)
        print(f"\nDETR Epoch 100 on Roboflow Cataract Dataset:")
        print(f"  mAP@0.5:0.95: {detr_metrics['mAP_0.5:0.95']*100:.2f}%")
        print(f"  mAP@0.5:     {detr_metrics['mAP_0.5']*100:.2f}%")
        print(f"  mAP@0.75:    {detr_metrics['mAP_0.75']*100:.2f}%")
        print(f"  AR@100:      {detr_metrics['AR_max_100']*100:.2f}%")
        print(f"  FPS:         {detr_metrics['fps']:.2f}")
        print(f"  Total detections: {detr_metrics['total_predictions']}")

        # Save results
        with open(OUTPUT_DIR / "detr_external_validation_summary.json", 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "dataset": "Roboflow Cataract Surgery (test - 50 images)",
                "model": "DETR Epoch 100",
                "metrics": detr_metrics,
                "note": "External validation on unseen surgical tool data"
            }, f, indent=2)

        print(f"\n✅ Results saved to: {OUTPUT_DIR}")
    else:
        print("\n❌ Benchmark failed - could not complete evaluation")


if __name__ == "__main__":
    main()
