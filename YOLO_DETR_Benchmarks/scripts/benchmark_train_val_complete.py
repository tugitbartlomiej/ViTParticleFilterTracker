#!/usr/bin/env python3
"""
Complete Benchmark: YOLO Epoch 70 vs DETR Epoch 100
Tests on BOTH train and validation sets
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

# Configuration
YOLO_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/YOLO_EDEN_TRAIN/epoch70.pt"
DETR_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR/checkpoint_epoch_100.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset configurations
DATASETS = {
    "train": {
        "annotations": "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Datasets/TooltipMining/annotations/tool_train_annotations.json",
        "images_dir": "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Datasets/TooltipMining/train"
    },
    "val": {
        "annotations": "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Datasets/TooltipMining/annotations/val_annotations.json",
        "images_dir": "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/BackgroundFinetuned/Datasets/TooltipMining/val"
    }
}

OUTPUT_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Benchmarks") / f"COMPLETE_YOLO70_DETR100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_yolo_benchmark(dataset_name, coco, image_id_map, output_dir):
    """Run YOLO Epoch 70 benchmark"""
    print(f"\n{'='*60}")
    print(f"Running YOLO Epoch 70 on {dataset_name.upper()} Set")
    print(f"{'='*60}\n")

    print(f"Loading YOLO from: {YOLO_CHECKPOINT}")
    try:
        model = YOLO(YOLO_CHECKPOINT)
        model.to(DEVICE)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())
    print(f"Processing {len(image_ids)} images...")

    for img_id in tqdm(image_ids, desc=f"YOLO {dataset_name}"):
        img_path = image_id_map[img_id]

        if not Path(img_path).exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        start_time = time.time()
        results = model.predict(img_path, conf=0.5, iou=0.5, verbose=False)
        total_time += time.time() - start_time

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Map to COCO category_id=1
                predictions.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf)
                })

        image_count += 1

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / f"yolo_{dataset_name}_predictions.json"

    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"YOLO Predictions: {len(predictions)}")
    print(f"Processing time: {total_time:.2f}s ({image_count/total_time:.2f} FPS)")

    # Evaluate
    if predictions:
        coco_dt = coco.loadRes(str(pred_file))
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "model": "YOLO Epoch 70",
            "dataset": dataset_name,
            "mAP_0.5": float(coco_eval.stats[1]) * 100,
            "mAP_0.5:0.95": float(coco_eval.stats[0]) * 100,
            "mAP_0.75": float(coco_eval.stats[2]) * 100,
            "AR_max_100": float(coco_eval.stats[8]) * 100,
            "AR_medium": float(coco_eval.stats[10]) * 100,
            "total_predictions": len(predictions),
            "fps": image_count / total_time
        }

        with open(output_dir / f"yolo_{dataset_name}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
    else:
        print("No predictions generated")
        return None


def run_detr_benchmark(dataset_name, coco, image_id_map, output_dir):
    """Run DETR Epoch 100 benchmark"""
    print(f"\n{'='*60}")
    print(f"Running DETR Epoch 100 on {dataset_name.upper()} Set")
    print(f"{'='*60}\n")

    print(f"Loading DETR from: {DETR_CHECKPOINT}")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = load_detr_from_checkpoint(DETR_CHECKPOINT, device=DEVICE)
    model.eval()
    print("DETR model loaded successfully")

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())
    print(f"Processing {len(image_ids)} images...")

    with torch.no_grad():
        for img_id in tqdm(image_ids, desc=f"DETR {dataset_name}"):
            img_path = image_id_map[img_id]

            if not Path(img_path).exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)

            start_time = time.time()
            outputs = model(**inputs)
            total_time += time.time() - start_time

            target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score = float(score.cpu())
                label = int(label.cpu())
                box = box.cpu().numpy()

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                # Map category_id 0 -> 1 for COCO compatibility
                category_id = 1 if label == 0 else label

                predictions.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })

            image_count += 1

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / f"detr_{dataset_name}_predictions.json"

    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"DETR Predictions: {len(predictions)}")
    print(f"Processing time: {total_time:.2f}s ({image_count/total_time:.2f} FPS)")

    # Evaluate
    if predictions:
        coco_dt = coco.loadRes(str(pred_file))
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "model": "DETR Epoch 100",
            "dataset": dataset_name,
            "mAP_0.5": float(coco_eval.stats[1]) * 100,
            "mAP_0.5:0.95": float(coco_eval.stats[0]) * 100,
            "mAP_0.75": float(coco_eval.stats[2]) * 100,
            "AR_max_100": float(coco_eval.stats[8]) * 100,
            "AR_medium": float(coco_eval.stats[10]) * 100,
            "total_predictions": len(predictions),
            "fps": image_count / total_time
        }

        with open(output_dir / f"detr_{dataset_name}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
    else:
        print("No predictions generated")
        return None


def generate_comparison_report(results, output_dir):
    """Generate comprehensive comparison report"""

    report = f"""# Complete Benchmark: YOLO Epoch 70 vs DETR Epoch 100

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**YOLO Checkpoint:** epoch70.pt (Eden/Checkpoints/YOLO_EDEN_TRAIN)
**DETR Checkpoint:** checkpoint_epoch_100.pth

---

## Results Summary

"""

    # Create comparison table for each dataset
    for dataset_name in ["train", "val"]:
        yolo_metrics = results[f"yolo_{dataset_name}"]
        detr_metrics = results[f"detr_{dataset_name}"]

        if not yolo_metrics or not detr_metrics:
            continue

        report += f"""
### {dataset_name.upper()} Set Results

| Metric | YOLO Epoch 70 | DETR Epoch 100 | Gap | Winner |
|--------|---------------|----------------|-----|--------|
| **mAP@0.5** | {yolo_metrics['mAP_0.5']:.2f}% | {detr_metrics['mAP_0.5']:.2f}% | {yolo_metrics['mAP_0.5'] - detr_metrics['mAP_0.5']:+.2f}% | {'YOLO' if yolo_metrics['mAP_0.5'] > detr_metrics['mAP_0.5'] else 'DETR'} |
| **mAP@0.5:0.95** | {yolo_metrics['mAP_0.5:0.95']:.2f}% | {detr_metrics['mAP_0.5:0.95']:.2f}% | {yolo_metrics['mAP_0.5:0.95'] - detr_metrics['mAP_0.5:0.95']:+.2f}% | {'YOLO' if yolo_metrics['mAP_0.5:0.95'] > detr_metrics['mAP_0.5:0.95'] else 'DETR'} |
| **mAP@0.75** | {yolo_metrics['mAP_0.75']:.2f}% | {detr_metrics['mAP_0.75']:.2f}% | {yolo_metrics['mAP_0.75'] - detr_metrics['mAP_0.75']:+.2f}% | {'YOLO' if yolo_metrics['mAP_0.75'] > detr_metrics['mAP_0.75'] else 'DETR'} |
| **AR@100** | {yolo_metrics['AR_max_100']:.2f}% | {detr_metrics['AR_max_100']:.2f}% | {yolo_metrics['AR_max_100'] - detr_metrics['AR_max_100']:+.2f}% | {'YOLO' if yolo_metrics['AR_max_100'] > detr_metrics['AR_max_100'] else 'DETR'} |

**Detections:** YOLO: {yolo_metrics['total_predictions']}, DETR: {detr_metrics['total_predictions']}
**Speed:** YOLO: {yolo_metrics['fps']:.2f} FPS, DETR: {detr_metrics['fps']:.2f} FPS

"""

    # Generalization analysis
    if "yolo_train" in results and "yolo_val" in results:
        report += f"""
---

## Generalization Analysis

### YOLO Epoch 70

| Dataset | mAP@0.5 | mAP@0.5:0.95 | AR@100 |
|---------|---------|--------------|--------|
| Train | {results['yolo_train']['mAP_0.5']:.2f}% | {results['yolo_train']['mAP_0.5:0.95']:.2f}% | {results['yolo_train']['AR_max_100']:.2f}% |
| Val | {results['yolo_val']['mAP_0.5']:.2f}% | {results['yolo_val']['mAP_0.5:0.95']:.2f}% | {results['yolo_val']['AR_max_100']:.2f}% |
| **Drop** | {results['yolo_train']['mAP_0.5'] - results['yolo_val']['mAP_0.5']:.2f}% | {results['yolo_train']['mAP_0.5:0.95'] - results['yolo_val']['mAP_0.5:0.95']:.2f}% | {results['yolo_train']['AR_max_100'] - results['yolo_val']['AR_max_100']:.2f}% |

### DETR Epoch 100

| Dataset | mAP@0.5 | mAP@0.5:0.95 | AR@100 |
|---------|---------|--------------|--------|
| Train | {results['detr_train']['mAP_0.5']:.2f}% | {results['detr_train']['mAP_0.5:0.95']:.2f}% | {results['detr_train']['AR_max_100']:.2f}% |
| Val | {results['detr_val']['mAP_0.5']:.2f}% | {results['detr_val']['mAP_0.5:0.95']:.2f}% | {results['detr_val']['AR_max_100']:.2f}% |
| **Drop** | {results['detr_train']['mAP_0.5'] - results['detr_val']['mAP_0.5']:.2f}% | {results['detr_train']['mAP_0.5:0.95'] - results['detr_val']['mAP_0.5:0.95']:.2f}% | {results['detr_train']['AR_max_100'] - results['detr_val']['AR_max_100']:.2f}% |

---

## Conclusions

### Overall Winner

Based on validation set performance: **{'YOLO' if results['yolo_val']['mAP_0.5:0.95'] > results['detr_val']['mAP_0.5:0.95'] else 'DETR'}**

### Key Findings

1. **Training Performance**: {'YOLO leads' if results['yolo_train']['mAP_0.5'] > results['detr_train']['mAP_0.5'] else 'DETR leads'} on training set
2. **Validation Performance**: {'YOLO leads' if results['yolo_val']['mAP_0.5'] > results['detr_val']['mAP_0.5'] else 'DETR leads'} on validation set
3. **Generalization**: {'YOLO generalizes better' if (results['yolo_train']['mAP_0.5'] - results['yolo_val']['mAP_0.5']) < (results['detr_train']['mAP_0.5'] - results['detr_val']['mAP_0.5']) else 'DETR generalizes better'} (smaller train-val gap)
4. **Speed**: {'YOLO is faster' if results['yolo_val']['fps'] > results['detr_val']['fps'] else 'DETR is faster'}

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    report_file = output_dir / "COMPLETE_BENCHMARK_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

    return report


def main():
    print("="*60)
    print("Complete Benchmark: YOLO Epoch 70 vs DETR Epoch 100")
    print("Testing on TRAIN and VAL sets")
    print("="*60)

    results = {}

    # Run benchmarks on each dataset
    for dataset_name, config in DATASETS.items():
        print(f"\n{'#'*60}")
        print(f"# Processing {dataset_name.upper()} Dataset")
        print(f"{'#'*60}")

        # Load COCO annotations
        print(f"\nLoading {dataset_name} annotations...")
        coco = COCO(config["annotations"])

        # Build image ID map
        print("Building image ID map...")
        image_id_map = {}
        for img in coco.dataset.get('images', []):
            img_id = img['id']
            img_filename = img['file_name']
            img_path = Path(config["images_dir"]) / img_filename

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            image_id_map[img_id] = str(img_path)

        print(f"Found {len(image_id_map)} images")

        # Run YOLO benchmark
        yolo_metrics = run_yolo_benchmark(dataset_name, coco, image_id_map, OUTPUT_DIR)
        if yolo_metrics:
            results[f"yolo_{dataset_name}"] = yolo_metrics

        # Run DETR benchmark
        detr_metrics = run_detr_benchmark(dataset_name, coco, image_id_map, OUTPUT_DIR)
        if detr_metrics:
            results[f"detr_{dataset_name}"] = detr_metrics

    # Generate comprehensive report
    if results:
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)

        generate_comparison_report(results, OUTPUT_DIR)

        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)

        for dataset_name in ["train", "val"]:
            if f"yolo_{dataset_name}" in results and f"detr_{dataset_name}" in results:
                yolo = results[f"yolo_{dataset_name}"]
                detr = results[f"detr_{dataset_name}"]

                print(f"\n{dataset_name.upper()} Set:")
                print(f"  YOLO: mAP@0.5={yolo['mAP_0.5']:.2f}%, mAP@0.5:0.95={yolo['mAP_0.5:0.95']:.2f}%, AR@100={yolo['AR_max_100']:.2f}%")
                print(f"  DETR: mAP@0.5={detr['mAP_0.5']:.2f}%, mAP@0.5:0.95={detr['mAP_0.5:0.95']:.2f}%, AR@100={detr['AR_max_100']:.2f}%")
                print(f"  Gap:  mAP@0.5={yolo['mAP_0.5'] - detr['mAP_0.5']:+.2f}%, mAP@0.5:0.95={yolo['mAP_0.5:0.95'] - detr['mAP_0.5:0.95']:+.2f}%")

        print(f"\nFull report: {OUTPUT_DIR / 'COMPLETE_BENCHMARK_REPORT.md'}")
    else:
        print("\nBenchmark failed - no results generated")


if __name__ == "__main__":
    main()
