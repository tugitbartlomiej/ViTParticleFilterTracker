#!/usr/bin/env python3
"""
Simple Multi-Epoch Benchmark for DETR vs YOLO
Runs benchmark for epochs 40, 60, 80, 100, 160
"""

import sys
import os
import json
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "Advanced_Analysis"))
from run_inference import (
    load_detr_from_checkpoint,
    build_coco_image_id_map,
    get_category_id_mapper,
    _allowlist_ultralytics_pickle_classes,
)

# Configuration
YOLO_MODEL = "E:/Cataract/yolo11/train5/weights/best.pt"
COCO_ANNOTATIONS = "E:/Cataract/CADTD/COCO/annotations/instances_test.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR")
OUTPUT_BASE_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Benchmarks")

EPOCHS_TO_TEST = [40, 60, 80, 100, 160]
CONF_THRESHOLDS = [0.15, 0.2, 0.25, 0.3]


def run_yolo_benchmark(coco, image_id_map, category_mapper, output_dir):
    """Run YOLO benchmark"""
    print(f"\n{'='*60}")
    print("Running YOLO Benchmark")
    print(f"{'='*60}\n")

    # Allow pickle for YOLO
    _allowlist_ultralytics_pickle_classes()

    # Load YOLO model
    model = YOLO(YOLO_MODEL)
    model.to(DEVICE)

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())

    for img_id in tqdm(image_ids, desc="YOLO inference"):
        img_path = image_id_map[img_id]

        # Run inference
        start_time = time.time()
        results = model.predict(img_path, conf=0.25, verbose=False)
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

                # Map category
                category_id = category_mapper[cls]

                predictions.append({
                    "image_id": img_id,
                    "category_id": category_id,
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
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.5": float(coco_eval.stats[1]),
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


def run_detr_benchmark(epoch, conf_threshold, coco, image_id_map, category_mapper, output_dir):
    """Run DETR benchmark for specific epoch and confidence threshold"""
    print(f"\n{'='*60}")
    print(f"Running DETR Epoch {epoch} (conf={conf_threshold})")
    print(f"{'='*60}\n")

    # Load checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None

    # Load model
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = load_detr_from_checkpoint(checkpoint_path, device=DEVICE)
    model.eval()

    predictions = []
    total_time = 0
    image_count = 0

    image_ids = list(image_id_map.keys())

    with torch.no_grad():
        for img_id in tqdm(image_ids, desc=f"DETR epoch {epoch}"):
            img_path = image_id_map[img_id]

            # Load and process image
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)

            # Run inference
            start_time = time.time()
            outputs = model(**inputs)
            total_time += time.time() - start_time

            # Process predictions
            target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
            results = processor.post_process_object_detection(
                outputs, threshold=conf_threshold, target_sizes=target_sizes
            )[0]

            # Extract predictions
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score = float(score.cpu())
                label = int(label.cpu())
                box = box.cpu().numpy()

                # Convert to COCO format
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                # Map category (DETR uses 0-91, we need to map to COCO)
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
    pred_file = output_dir / f"detr_epoch{epoch}_conf{conf_threshold:.2f}_predictions.json"

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
            "epoch": epoch,
            "conf_threshold": conf_threshold,
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_100": float(coco_eval.stats[8]),
            "total_predictions": len(predictions),
            "fps": image_count / total_time,
            "total_time": total_time
        }

        # Save metrics
        metrics_file = output_dir / f"detr_epoch{epoch}_conf{conf_threshold:.2f}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics
    else:
        return None


def main():
    print("="*60)
    print("DETR Multi-Epoch Benchmark Suite")
    print("="*60)

    # Load COCO annotations
    print("\nLoading COCO annotations...")
    coco = COCO(COCO_ANNOTATIONS)

    # Build image ID map
    print("Building image ID map...")
    image_id_map = build_coco_image_id_map(coco)
    print(f"Found {len(image_id_map)} images")

    # Get category mapper
    category_mapper = get_category_id_mapper()

    # Results collection
    all_results = {
        "yolo": None,
        "detr_epochs": {}
    }

    # Run YOLO benchmark (once)
    yolo_output = OUTPUT_BASE_DIR / "YOLO_Baseline"
    yolo_metrics = run_yolo_benchmark(coco, image_id_map, category_mapper, yolo_output)
    all_results["yolo"] = yolo_metrics

    # Run DETR benchmarks for each epoch and confidence threshold
    for epoch in EPOCHS_TO_TEST:
        print(f"\n{'='*60}")
        print(f"Testing DETR Epoch {epoch}")
        print(f"{'='*60}")

        epoch_results = []

        for conf in CONF_THRESHOLDS:
            detr_output = OUTPUT_BASE_DIR / f"DETR_Epoch_{epoch}"
            metrics = run_detr_benchmark(epoch, conf, coco, image_id_map, category_mapper, detr_output)

            if metrics:
                epoch_results.append(metrics)

        all_results["detr_epochs"][epoch] = epoch_results

    # Generate summary
    summary_file = OUTPUT_BASE_DIR / "multi_epoch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_BASE_DIR}")
    print(f"Summary: {summary_file}")

    # Print comparison table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    if all_results["yolo"]:
        print(f"\nYOLO Baseline:")
        print(f"  mAP@0.5:0.95: {all_results['yolo']['mAP_0.5:0.95']:.4f}")
        print(f"  mAP@0.5:     {all_results['yolo']['mAP_0.5']:.4f}")
        print(f"  AR@100:      {all_results['yolo']['AR_max_100']:.4f}")
        print(f"  FPS:         {all_results['yolo']['fps']:.2f}")

    for epoch, results in all_results["detr_epochs"].items():
        if results:
            # Find best mAP
            best = max(results, key=lambda x: x["mAP_0.5:0.95"])
            print(f"\nDETR Epoch {epoch} (best conf={best['conf_threshold']}):")
            print(f"  mAP@0.5:0.95: {best['mAP_0.5:0.95']:.4f}")
            print(f"  mAP@0.5:     {best['mAP_0.5']:.4f}")
            print(f"  AR@100:      {best['AR_max_100']:.4f}")
            print(f"  FPS:         {best['fps']:.2f}")


if __name__ == "__main__":
    main()
