#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETR Training Degradation Analysis
===================================

Comprehensive analysis of DETR model degradation during training.
Tests epochs: 40, 60, 80, 100, 120, 140, 160

Key analyses:
1. Query specialization evolution (especially Query 81)
2. Model performance metrics (mAP, AR, etc.)
3. Confidence distribution changes
4. Overfitting detection
5. Comparison with YOLO baseline

Author: Bartłomiej Łówko
Date: 2025-10-31
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import matplotlib.pyplot as plt

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# PyTorch compatibility patches
try:
    from ultralytics.utils.loss import DFLoss
except (ImportError, AttributeError):
    class DFLoss:
        pass
    if 'ultralytics.utils.loss' in sys.modules:
        sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss

_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

# Imports
sys.path.append(str(Path(__file__).parent.parent / "Advanced_Analysis"))
from run_inference import load_detr_from_checkpoint
from transformers import DetrImageProcessor
from ultralytics import YOLO

# Paths configuration
DETR_CHECKPOINTS_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR")
YOLO_MODEL = "E:/Cataract/yolo11/train5/weights/best.pt"
VOC_LABELS_DIR = "E:/CaDTD-main/Setup 2/VOC Labels"
IMAGES_DIR = "E:/CaDTD-main/Setup 2/Labels"
OUTPUT_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Degradation_Analysis")

EPOCHS_TO_TEST = [40, 60, 80, 100, 120, 140, 160]
CONF_THRESHOLDS = [0.15, 0.2, 0.25]  # Test multiple thresholds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QueryAnalyzer:
    """Analyzes DETR query specialization and weights"""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def analyze_query_weights(self, checkpoint_path):
        """Extract and analyze query embedding weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

        # Extract query embeddings
        query_embed = state_dict.get('query_embed.weight', None)

        if query_embed is None:
            print("Warning: query_embed.weight not found in checkpoint")
            return None

        # Analyze
        query_norms = torch.norm(query_embed, dim=1).numpy()
        query_mean = query_embed.mean(dim=1).numpy()
        query_std = query_embed.std(dim=1).numpy()

        return {
            "norms": query_norms.tolist(),
            "means": query_mean.tolist(),
            "stds": query_std.tolist(),
            "shape": list(query_embed.shape),
            "query_81_norm": float(query_norms[81]) if len(query_norms) > 81 else None,
            "top_5_queries_by_norm": np.argsort(query_norms)[-5:].tolist()[::-1]
        }

    def analyze_query_usage(self, image_paths, conf_threshold=0.2):
        """Analyze which queries are used during inference"""
        query_usage = defaultdict(int)
        query_confidences = defaultdict(list)
        total_detections = 0

        with torch.no_grad():
            for img_path in tqdm(image_paths[:100], desc="Analyzing query usage"):  # Sample 100 images
                image = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

                outputs = self.model(**inputs)
                target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)

                # Get raw logits and boxes
                logits = outputs.logits[0]  # [100, num_classes]
                boxes = outputs.pred_boxes[0]  # [100, 4]

                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                scores, labels = probs[:, :-1].max(dim=-1)  # Exclude background class

                # Filter by confidence
                keep = scores > conf_threshold

                for query_idx in range(100):
                    if keep[query_idx]:
                        query_usage[query_idx] += 1
                        query_confidences[query_idx].append(float(scores[query_idx]))
                        total_detections += 1

        # Compute statistics
        usage_stats = {}
        for query_idx in range(100):
            usage_count = query_usage[query_idx]
            confidences = query_confidences[query_idx]

            usage_stats[query_idx] = {
                "count": usage_count,
                "percentage": (usage_count / total_detections * 100) if total_detections > 0 else 0,
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "max_confidence": np.max(confidences) if confidences else 0.0,
            }

        # Find dominant queries
        dominant = sorted(usage_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]

        return {
            "total_detections": total_detections,
            "usage_stats": usage_stats,
            "dominant_queries": [(idx, stats) for idx, stats in dominant],
            "query_81_usage": usage_stats[81] if 81 in usage_stats else {"count": 0, "percentage": 0}
        }


class DETRBenchmark:
    """Run DETR benchmark on CaDTD dataset"""

    def __init__(self, checkpoint_path, epoch, coco_gt, image_id_map, conf_threshold=0.2):
        self.checkpoint_path = Path(checkpoint_path)
        self.epoch = epoch
        self.coco_gt = coco_gt
        self.image_id_map = image_id_map
        self.conf_threshold = conf_threshold

        # Load model
        print(f"\nLoading DETR epoch {epoch}...")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = load_detr_from_checkpoint(checkpoint_path, device=DEVICE)
        self.model.eval()

    def run_benchmark(self):
        """Run full benchmark"""
        predictions = []

        with torch.no_grad():
            for img_id, img_path in tqdm(self.image_id_map.items(), desc=f"DETR epoch {self.epoch}"):
                image = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

                outputs = self.model(**inputs)
                target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)

                results = self.processor.post_process_object_detection(
                    outputs, threshold=self.conf_threshold, target_sizes=target_sizes
                )[0]

                # Convert to COCO format
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    w, h = x2 - x1, y2 - y1

                    predictions.append({
                        "image_id": img_id,
                        "category_id": 1,  # All tools mapped to single class
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score.cpu())
                    })

        # Evaluate
        if not predictions:
            return None

        # Save predictions
        pred_file = OUTPUT_DIR / f"epoch{self.epoch}_predictions.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)

        # COCO evaluation
        coco_dt = self.coco_gt.loadRes(str(pred_file))
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "epoch": self.epoch,
            "conf_threshold": self.conf_threshold,
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_1": float(coco_eval.stats[6]),
            "AR_max_10": float(coco_eval.stats[7]),
            "AR_max_100": float(coco_eval.stats[8]),
            "total_predictions": len(predictions)
        }

        return metrics


def convert_voc_to_coco():
    """Convert VOC annotations to COCO format"""
    print("\n" + "="*60)
    print("Converting VOC → COCO")
    print("="*60)

    from voc_to_coco_converter import VOCToCOCOConverter

    output_json = OUTPUT_DIR / "cadtd_coco_annotations.json"

    if output_json.exists():
        print(f"COCO annotations already exist: {output_json}")
        return output_json

    converter = VOCToCOCOConverter(
        voc_labels_dir=VOC_LABELS_DIR,
        images_dir=IMAGES_DIR,
        output_json_path=output_json,
        merge_classes=True  # Single "surgical_tool" class
    )

    converter.convert()
    print(f"COCO annotations saved: {output_json}")

    return output_json


def main():
    print("="*60)
    print("DETR Training Degradation Analysis")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs to test: {EPOCHS_TO_TEST}")
    print(f"Confidence thresholds: {CONF_THRESHOLDS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert VOC to COCO
    coco_json = convert_voc_to_coco()
    coco_gt = COCO(str(coco_json))

    # Build image ID map
    image_id_map = {}
    for img_info in coco_gt.loadImgs(coco_gt.getImgIds()):
        img_path = Path(IMAGES_DIR) / img_info['file_name']
        if img_path.exists():
            image_id_map[img_info['id']] = str(img_path)

    print(f"\nTotal images: {len(image_id_map)}")

    # Collect all results
    all_results = {
        "epochs": {},
        "query_analysis": {},
        "degradation_metrics": {}
    }

    # Step 2: Analyze each epoch
    for epoch in EPOCHS_TO_TEST:
        checkpoint_path = DETR_CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch}.pth"

        if not checkpoint_path.exists():
            print(f"\n[WARNING] Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing Epoch {epoch}")
        print(f"{'='*60}")

        # Load model for query analysis
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = load_detr_from_checkpoint(checkpoint_path, device=DEVICE)
        model.eval()

        # Query weight analysis
        print(f"\nAnalyzing query weights...")
        analyzer = QueryAnalyzer(model, processor)
        query_weights = analyzer.analyze_query_weights(checkpoint_path)

        # Query usage analysis
        print(f"\nAnalyzing query usage...")
        image_paths = list(image_id_map.values())
        query_usage = analyzer.analyze_query_usage(image_paths, conf_threshold=0.2)

        all_results["query_analysis"][epoch] = {
            "weights": query_weights,
            "usage": query_usage
        }

        # Benchmark for each confidence threshold
        epoch_results = []
        for conf in CONF_THRESHOLDS:
            benchmark = DETRBenchmark(checkpoint_path, epoch, coco_gt, image_id_map, conf)
            metrics = benchmark.run_benchmark()

            if metrics:
                epoch_results.append(metrics)
                print(f"\nEpoch {epoch} (conf={conf}):")
                print(f"  mAP@0.5:0.95: {metrics['mAP_0.5:0.95']:.4f}")
                print(f"  mAP@0.5:     {metrics['mAP_0.5']:.4f}")
                print(f"  AR@100:      {metrics['AR_max_100']:.4f}")

        all_results["epochs"][epoch] = epoch_results

    # Step 3: Degradation analysis
    print("\n" + "="*60)
    print("Degradation Analysis")
    print("="*60)

    # Extract best mAP for each epoch
    epoch_map = {}
    for epoch, results in all_results["epochs"].items():
        if results:
            best = max(results, key=lambda x: x["mAP_0.5:0.95"])
            epoch_map[epoch] = best

    # Detect degradation
    epochs_sorted = sorted(epoch_map.keys())
    degradation_detected = False

    for i in range(1, len(epochs_sorted)):
        prev_epoch = epochs_sorted[i-1]
        curr_epoch = epochs_sorted[i]

        prev_map = epoch_map[prev_epoch]["mAP_0.5:0.95"]
        curr_map = epoch_map[curr_epoch]["mAP_0.5:0.95"]

        delta = curr_map - prev_map

        print(f"\nEpoch {prev_epoch} → {curr_epoch}:")
        print(f"  mAP change: {delta:+.4f} ({delta/prev_map*100:+.2f}%)")

        if delta < -0.02:  # >2% degradation
            print(f"  [WARNING] Degradation detected!")
            degradation_detected = True
        elif delta > 0.05:  # >5% improvement
            print(f"  [GOOD] Significant improvement!")

    # Query 81 analysis
    print("\n" + "="*60)
    print("Query 81 Specialization Analysis")
    print("="*60)

    for epoch in epochs_sorted:
        if epoch in all_results["query_analysis"]:
            usage = all_results["query_analysis"][epoch]["usage"]
            q81 = usage.get("query_81_usage", {})

            print(f"\nEpoch {epoch}:")
            print(f"  Query 81 usage: {q81.get('count', 0)} detections ({q81.get('percentage', 0):.2f}%)")
            print(f"  Avg confidence: {q81.get('avg_confidence', 0):.4f}")

            # Show top 3 dominant queries
            dominant = usage.get("dominant_queries", [])[:3]
            print(f"  Top 3 dominant queries:")
            for idx, (query_idx, stats) in enumerate(dominant, 1):
                print(f"    {idx}. Query {query_idx}: {stats['count']} ({stats['percentage']:.2f}%)")

    # Save final report
    report_file = OUTPUT_DIR / "degradation_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"Report saved: {report_file}")

    return all_results


if __name__ == "__main__":
    main()
