#!/usr/bin/env python3
"""
=============================================================================
BENCHMARK: YOLO vs DETR (Query 81 Only) - ALL EPOCHS COMPARISON
=============================================================================
Comprehensive benchmark comparing ALL available checkpoints:

TRAINING STRATEGY:
  - Phase 1: Both models trained on ~100k images to epoch 170
  - Phase 2: Fine-tuning on 20k selected images from epoch 170

CHECKPOINTS TESTED:
  YOLO Original (100k): 20, 40, 60, 80, 100, 120, 140, 160, 170, 180, 190
  YOLO 20k Finetune:    180, 190
  DETR Original (100k): 20, 40, 60, 80, 100, 120, 140, 160, 170
  DETR 20k Finetune:    170, 180, 190, 200, 210, ..., 320, 330

Dataset: Cataract Surgery Instruments (Roboflow - cross-dataset test)
Metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1

Author: PhD Research - Cataract Surgery Instrument Detection
Date: 2026-01-12
=============================================================================
"""

import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*pass `assign=True`.*")

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection

# =============================================================================
# YOLO LEGACY COMPATIBILITY FIX (PyTorch 2.6+)
# =============================================================================
print("Setting up YOLO legacy checkpoint compatibility...")
import ultralytics.utils.loss as loss_module
if not hasattr(loss_module, 'DFLoss'):
    class DFLoss:
        pass
    loss_module.DFLoss = DFLoss
    sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss
    print("  - Added DFLoss stub")

original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
print("  - Patched torch.load for legacy support")

from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker")

# =============================================================================
# CHECKPOINT PATHS - ALL AVAILABLE EPOCHS
# =============================================================================

# YOLO Original Training (~100k dataset, epochs 0-200)
YOLO_ORIGINAL_PATH = BASE_PATH / "Eden/Checkpoints/YOLO_EDEN_TRAIN/exp/weights"
YOLO_ORIGINAL_CHECKPOINTS = {
    20: YOLO_ORIGINAL_PATH / "epoch20.pt",
    40: YOLO_ORIGINAL_PATH / "epoch40.pt",
    60: YOLO_ORIGINAL_PATH / "epoch60.pt",
    80: YOLO_ORIGINAL_PATH / "epoch80.pt",
    100: YOLO_ORIGINAL_PATH / "epoch100.pt",
    120: YOLO_ORIGINAL_PATH / "epoch120.pt",
    140: YOLO_ORIGINAL_PATH / "epoch140.pt",
    160: YOLO_ORIGINAL_PATH / "epoch160.pt",
    170: YOLO_ORIGINAL_PATH / "epoch170.pt",
    180: YOLO_ORIGINAL_PATH / "epoch180.pt",
    190: YOLO_ORIGINAL_PATH / "epoch190.pt",
}

# YOLO 20k Finetune (from epoch 170, trained on 20k selected images)
YOLO_20K_PATH = BASE_PATH / "Eden/Checkpoints/YOLO_EDEN_TRAIN/YoloTreningSesnions/20KTreningFrom170epochStart/exp/weights"
YOLO_20K_CHECKPOINTS = {
    180: YOLO_20K_PATH / "epoch180.pt",
    190: YOLO_20K_PATH / "epoch190.pt",
    200: YOLO_20K_PATH / "last.pt",  # last.pt is epoch 200
}

# DETR Original Training (~100k dataset, epochs 0-170)
DETR_ORIGINAL_PATH = BASE_PATH / "Eden/Checkpoints/DETR/DETR_Training_Sessions/2025-10-31_tooltip_single_class/DETR_Checkpoints"
DETR_ORIGINAL_CHECKPOINTS = {
    20: DETR_ORIGINAL_PATH / "checkpoint_epoch_20.pth",
    40: DETR_ORIGINAL_PATH / "checkpoint_epoch_40.pth",
    60: DETR_ORIGINAL_PATH / "checkpoint_epoch_60.pth",
    80: DETR_ORIGINAL_PATH / "checkpoint_epoch_80.pth",
    100: DETR_ORIGINAL_PATH / "checkpoint_epoch_100.pth",
    120: DETR_ORIGINAL_PATH / "checkpoint_epoch_120.pth",
    140: DETR_ORIGINAL_PATH / "checkpoint_epoch_140.pth",
    160: DETR_ORIGINAL_PATH / "checkpoint_epoch_160.pth",
    170: DETR_ORIGINAL_PATH / "checkpoint_epoch_170.pth",
}

# DETR 20k Finetune (from epoch 170, trained on 20k selected images)
DETR_20K_PATH = BASE_PATH / "Eden/Checkpoints/DETR/DETR_Training_Sessions/2025-12-13_20kDataset_small_LR/ckpt_20k_finetune"
DETR_20K_CHECKPOINTS = {
    170: DETR_20K_PATH / "checkpoint_epoch_170.pth",
    180: DETR_20K_PATH / "checkpoint_epoch_180.pth",
    190: DETR_20K_PATH / "checkpoint_epoch_190.pth",
    200: DETR_20K_PATH / "checkpoint_epoch_200.pth",
    210: DETR_20K_PATH / "checkpoint_epoch_210.pth",
    220: DETR_20K_PATH / "checkpoint_epoch_220.pth",
    230: DETR_20K_PATH / "checkpoint_epoch_230.pth",
    240: DETR_20K_PATH / "checkpoint_epoch_240.pth",
    250: DETR_20K_PATH / "checkpoint_epoch_250.pth",
    260: DETR_20K_PATH / "checkpoint_epoch_260.pth",
    270: DETR_20K_PATH / "checkpoint_epoch_270.pth",
    280: DETR_20K_PATH / "checkpoint_epoch_280.pth",
    290: DETR_20K_PATH / "checkpoint_epoch_290.pth",
    300: DETR_20K_PATH / "checkpoint_epoch_300.pth",
    310: DETR_20K_PATH / "checkpoint_epoch_310.pth",
    320: DETR_20K_PATH / "checkpoint_epoch_320.pth",
    330: DETR_20K_PATH / "checkpoint_epoch_330.pth",
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
DATASET_ROOT = Path("E:/cataract_surgery_Instruments_detection.v1i.coco")
ANNOTATION_FILE = None
IMAGES_DIR = None
ALL_SPLITS = ["train", "valid", "test"]

# =============================================================================
# DATA LEAKAGE PREVENTION
# =============================================================================
# Load list of images from Roboflow valid that were included in 20k training
LEAKED_IMAGES_FILE = Path(__file__).parent / "leaked_valid_images.json"
LEAKED_VALID_IMAGES = set()
if LEAKED_IMAGES_FILE.exists():
    with open(LEAKED_IMAGES_FILE) as f:
        leaked_data = json.load(f)
        LEAKED_VALID_IMAGES = set(leaked_data.get("images", []))
    print(f"[DATA LEAKAGE PREVENTION] Loaded {len(LEAKED_VALID_IMAGES)} leaked images to exclude from valid split for 20k models")

# Output
OUTPUT_DIR = BASE_PATH / "YOLO_DETR_Benchmarks/Benchmarks"
BENCHMARK_DIR = OUTPUT_DIR / f"BENCHMARK_ALL_EPOCHS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds
DETR_QUERY_ID = 81
CONF_THRESHOLD_STANDARD = 0.3
IOU_THRESHOLD = 0.3
YOLO_MAX_DETECTIONS = 300

print(f"""
{'='*80}
BENCHMARK: YOLO vs DETR (Query 81) - ALL EPOCHS COMPARISON
{'='*80}
Device: {DEVICE}
Dataset: {DATASET_ROOT}
Output: {BENCHMARK_DIR}

YOLO Original Epochs: {list(YOLO_ORIGINAL_CHECKPOINTS.keys())}
YOLO 20k Finetune Epochs: {list(YOLO_20K_CHECKPOINTS.keys())}
DETR Original Epochs: {list(DETR_ORIGINAL_CHECKPOINTS.keys())}
DETR 20k Finetune Epochs: {list(DETR_20K_CHECKPOINTS.keys())}

Threshold: {CONF_THRESHOLD_STANDARD*100:.0f}%
DETR Query: Q{DETR_QUERY_ID}
{'='*80}
""")

# Create output directories
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL LOADERS
# =============================================================================
DETR_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def load_yolo_model(checkpoint_path):
    print(f"  Loading YOLO from {checkpoint_path.name}...")
    model = YOLO(str(checkpoint_path))
    model.to(DEVICE)
    return model

def load_detr_model(checkpoint_path):
    print(f"  Loading DETR from {checkpoint_path.name}...")
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)

    model.to(DEVICE)
    model.eval()
    return model

# =============================================================================
# DATASET PREPARATION
# =============================================================================
def prepare_coco_annotations(split_name, exclude_images=None):
    """
    Prepare COCO annotations for a split.

    Args:
        split_name: Name of the split (train, valid, test)
        exclude_images: Set of image filenames to exclude (for data leakage prevention)

    Returns:
        coco: COCO object
        images_dir: Path to images directory
    """
    ann_path = DATASET_ROOT / split_name / "_annotations.coco.json"
    images_dir = DATASET_ROOT / split_name

    if not ann_path.exists():
        return None, None

    with open(ann_path) as f:
        coco_data = json.load(f)

    # Filter out excluded images (data leakage prevention)
    excluded_count = 0
    if exclude_images:
        excluded_img_ids = set()
        filtered_images = []
        for img in coco_data['images']:
            if img['file_name'] in exclude_images:
                excluded_img_ids.add(img['id'])
                excluded_count += 1
            else:
                filtered_images.append(img)
        coco_data['images'] = filtered_images

        # Also filter annotations for excluded images
        coco_data['annotations'] = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] not in excluded_img_ids
        ]

        if excluded_count > 0:
            print(f"    [LEAKAGE PREVENTION] Excluded {excluded_count} images from {split_name}")

    coco_data['categories'] = [{"id": 1, "name": "instrument", "supercategory": "surgical"}]
    for ann in coco_data['annotations']:
        ann['category_id'] = 1

    suffix = "_clean" if exclude_images else ""
    temp_ann_path = BENCHMARK_DIR / f"temp_{split_name}{suffix}_annotations.json"
    with open(temp_ann_path, 'w') as f:
        json.dump(coco_data, f)

    coco = COCO(str(temp_ann_path))
    return coco, images_dir

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================
def run_yolo_inference(model, image_path, conf_threshold):
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=IOU_THRESHOLD,
        max_det=YOLO_MAX_DETECTIONS,
        verbose=False,
    )[0]

    predictions = []
    boxes = getattr(results, "boxes", None)
    if boxes is not None and boxes.shape[0] > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(xyxy, confs):
            predictions.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })
    return predictions

def run_detr_q81_inference(model, image_path, conf_threshold):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    inputs = DETR_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    boxes = outputs.pred_boxes[0]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    q81_score = float(probs[DETR_QUERY_ID, 0])

    predictions = []
    if q81_score >= conf_threshold:
        cx, cy, w, h = boxes[DETR_QUERY_ID].cpu().numpy()
        cx *= img_width
        cy *= img_height
        w *= img_width
        h *= img_height
        x = cx - w / 2
        y = cy - h / 2

        predictions.append({
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": q81_score,
            "query_id": DETR_QUERY_ID
        })
    return predictions

def run_inference_on_split(model, coco, images_dir, model_name, split_name, conf_threshold, is_yolo=True):
    image_ids = coco.getImgIds()
    predictions = []
    inference_times = []

    for img_id in tqdm(image_ids, desc=f"{model_name} on {split_name}", leave=False):
        img_info = coco.loadImgs(img_id)[0]
        img_path = images_dir / img_info['file_name']

        if not img_path.exists():
            continue

        start_time = time.time()
        if is_yolo:
            preds = run_yolo_inference(model, img_path, conf_threshold)
        else:
            preds = run_detr_q81_inference(model, img_path, conf_threshold)
        inference_times.append(time.time() - start_time)

        for pred in preds:
            predictions.append({
                "image_id": int(img_id),
                "category_id": 1,
                "bbox": pred["bbox"],
                "score": pred["score"]
            })

    fps = len(image_ids) / sum(inference_times) if inference_times else 0
    return predictions, fps

# =============================================================================
# EVALUATION
# =============================================================================
def bbox_iou_xywh(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = aw * ah + bw * bh - inter_area
    return inter_area / union if union > 0 else 0.0

def compute_detection_stats(coco, predictions, iou_threshold=0.5):
    preds_by_image = defaultdict(list)
    for pred in predictions:
        preds_by_image[int(pred["image_id"])].append(pred)

    tp = fp = fn = 0
    for img_id in coco.getImgIds():
        gt_annotations = coco.imgToAnns.get(img_id, []) or []
        pred_list = preds_by_image.get(img_id, [])
        pred_used = [False] * len(pred_list)

        for gt in gt_annotations:
            gt_box = gt["bbox"]
            best_idx = None
            best_iou = 0.0
            for idx, pred in enumerate(pred_list):
                if pred_used[idx]:
                    continue
                iou = bbox_iou_xywh(gt_box, pred["bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None:
                pred_used[best_idx] = True
                tp += 1
            else:
                fn += 1

        for used in pred_used:
            if not used:
                fp += 1

    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def evaluate_predictions(coco, predictions, model_name, split_name):
    if not predictions:
        return {
            "mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "mAP@0.75": 0.0, "AR@100": 0.0,
            "true_positives": 0, "false_positives": 0, "false_negatives": 0,
            "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
        }

    pred_file = BENCHMARK_DIR / f"{model_name}_{split_name}_predictions.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)

    coco_dt = coco.loadRes(str(pred_file))
    coco_eval = COCOeval(coco, coco_dt, 'bbox')

    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    metrics = {
        "mAP@0.5": float(coco_eval.stats[1]) * 100,
        "mAP@0.5:0.95": float(coco_eval.stats[0]) * 100,
        "mAP@0.75": float(coco_eval.stats[2]) * 100,
        "AR@100": float(coco_eval.stats[8]) * 100
    }

    detection_stats = compute_detection_stats(coco, predictions)
    metrics.update(detection_stats)
    return metrics

# =============================================================================
# AGGREGATION
# =============================================================================
def aggregate_model_results(model_data):
    total_tp = total_fp = total_fn = 0

    for split_name, split_data in model_data.get("splits", {}).items():
        metrics = split_data.get("metrics", {})
        total_tp += metrics.get("true_positives", 0)
        total_fp += metrics.get("false_positives", 0)
        total_fn += metrics.get("false_negatives", 0)

    precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0
    recall = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# =============================================================================
# REPORT GENERATION
# =============================================================================
def generate_report(results):
    leaked_count = len(LEAKED_VALID_IMAGES)
    report = f"""# Benchmark Report: YOLO vs DETR - ALL EPOCHS COMPARISON

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Device:** {DEVICE}
**Dataset:** Roboflow Cataract Surgery (cross-dataset test)

## Training Strategy

Both models were trained in two phases:
1. **Phase 1 (Original):** Training on ~100,000 images to epoch 170
2. **Phase 2 (20k Finetune):** Fine-tuning on 20,000 intelligently selected images from epoch 170

## Data Leakage Prevention

⚠️ **IMPORTANT:** During 20k dataset selection, {leaked_count} images from Roboflow validation set
were inadvertently included in the 20k training dataset (data leakage).

**Evaluation Strategy:**
- **Original models (100k):** Evaluated on FULL validation set (no leakage issue)
- **20k finetune models:** Evaluated on CLEAN validation set ({leaked_count} leaked images excluded)

---

## Summary: YOLO Original Training (100k dataset)

| Epoch | mAP@0.5 | Precision | Recall | F1 | TP | FP | FN |
|-------|---------|-----------|--------|----|----|----|----|
"""

    # YOLO Original
    for model_name, model_data in sorted(results.items()):
        if model_data.get('type') == 'YOLO_Original':
            agg = model_data.get('aggregated', {})
            valid = model_data.get('splits', {}).get('valid', {}).get('metrics', {})
            report += f"| {model_data.get('epoch')} | {valid.get('mAP@0.5', 0):.2f}% | "
            report += f"{agg.get('precision', 0):.1f}% | {agg.get('recall', 0):.1f}% | {agg.get('f1_score', 0):.1f}% | "
            report += f"{agg.get('total_tp', 0)} | {agg.get('total_fp', 0)} | {agg.get('total_fn', 0)} |\n"

    report += """
---

## Summary: YOLO 20k Finetune (⚠️ CLEAN dataset - leaked images excluded)

| Epoch | mAP@0.5 | Precision | Recall | F1 | TP | FP | FN |
|-------|---------|-----------|--------|----|----|----|----|
"""

    for model_name, model_data in sorted(results.items()):
        if model_data.get('type') == 'YOLO_20k':
            agg = model_data.get('aggregated', {})
            valid = model_data.get('splits', {}).get('valid', {}).get('metrics', {})
            report += f"| {model_data.get('epoch')} | {valid.get('mAP@0.5', 0):.2f}% | "
            report += f"{agg.get('precision', 0):.1f}% | {agg.get('recall', 0):.1f}% | {agg.get('f1_score', 0):.1f}% | "
            report += f"{agg.get('total_tp', 0)} | {agg.get('total_fp', 0)} | {agg.get('total_fn', 0)} |\n"

    report += """
---

## Summary: DETR Original Training (100k dataset, Query 81)

| Epoch | mAP@0.5 | Precision | Recall | F1 | TP | FP | FN |
|-------|---------|-----------|--------|----|----|----|----|
"""

    for model_name, model_data in sorted(results.items()):
        if model_data.get('type') == 'DETR_Original':
            agg = model_data.get('aggregated', {})
            valid = model_data.get('splits', {}).get('valid', {}).get('metrics', {})
            report += f"| {model_data.get('epoch')} | {valid.get('mAP@0.5', 0):.2f}% | "
            report += f"{agg.get('precision', 0):.1f}% | {agg.get('recall', 0):.1f}% | {agg.get('f1_score', 0):.1f}% | "
            report += f"{agg.get('total_tp', 0)} | {agg.get('total_fp', 0)} | {agg.get('total_fn', 0)} |\n"

    report += """
---

## Summary: DETR 20k Finetune (Query 81) (⚠️ CLEAN dataset - leaked images excluded)

| Epoch | mAP@0.5 | Precision | Recall | F1 | TP | FP | FN |
|-------|---------|-----------|--------|----|----|----|----|
"""

    for model_name, model_data in sorted(results.items()):
        if model_data.get('type') == 'DETR_20k':
            agg = model_data.get('aggregated', {})
            valid = model_data.get('splits', {}).get('valid', {}).get('metrics', {})
            report += f"| {model_data.get('epoch')} | {valid.get('mAP@0.5', 0):.2f}% | "
            report += f"{agg.get('precision', 0):.1f}% | {agg.get('recall', 0):.1f}% | {agg.get('f1_score', 0):.1f}% | "
            report += f"{agg.get('total_tp', 0)} | {agg.get('total_fp', 0)} | {agg.get('total_fn', 0)} |\n"

    report += f"""
---

## Best Models Comparison

| Model | Best Epoch | mAP@0.5 | F1 |
|-------|------------|---------|----|
"""

    # Find best for each category
    categories = ['YOLO_Original', 'YOLO_20k', 'DETR_Original', 'DETR_20k']
    for cat in categories:
        best_f1 = 0
        best_model = None
        for model_name, model_data in results.items():
            if model_data.get('type') == cat:
                agg = model_data.get('aggregated', {})
                f1 = agg.get('f1_score', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_data
        if best_model:
            valid = best_model.get('splits', {}).get('valid', {}).get('metrics', {})
            report += f"| {cat} | {best_model.get('epoch')} | {valid.get('mAP@0.5', 0):.2f}% | {best_f1:.1f}% |\n"

    report += f"""
---

**Output Directory:** `{BENCHMARK_DIR}`
"""
    return report

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    start_time = time.time()
    results = {}

    # Prepare datasets - FULL (for Original 100k models)
    print("\nPreparing FULL datasets (for Original 100k models)...")
    datasets_full = {}
    for split in ALL_SPLITS:
        coco, images_dir = prepare_coco_annotations(split)
        if coco:
            datasets_full[split] = (coco, images_dir)
            print(f"  {split}: {len(coco.getImgIds())} images")

    # Prepare datasets - CLEAN (for 20k finetune models, excluding leaked images from valid)
    print("\nPreparing CLEAN datasets (for 20k finetune models, no data leakage)...")
    datasets_clean = {}
    for split in ALL_SPLITS:
        # Only exclude leaked images from valid split
        exclude = LEAKED_VALID_IMAGES if split == "valid" else None
        coco, images_dir = prepare_coco_annotations(split, exclude_images=exclude)
        if coco:
            datasets_clean[split] = (coco, images_dir)
            print(f"  {split}: {len(coco.getImgIds())} images {'(CLEAN)' if exclude else ''}")

    if not datasets_full:
        print("ERROR: No valid datasets found!")
        sys.exit(1)

    # =================================================================
    # BENCHMARK YOLO ORIGINAL (uses FULL datasets - no leakage issue)
    # =================================================================
    print(f"\n{'='*60}")
    print("BENCHMARKING YOLO ORIGINAL (100k dataset) - FULL datasets")
    print(f"{'='*60}")

    for epoch, checkpoint_path in sorted(YOLO_ORIGINAL_CHECKPOINTS.items()):
        model_name = f"YOLO_Orig_ep{epoch}"

        if not checkpoint_path.exists():
            print(f"\n  SKIP: {model_name} (not found)")
            continue

        print(f"\n  Processing {model_name}...")
        model = load_yolo_model(checkpoint_path)

        results[model_name] = {"type": "YOLO_Original", "epoch": epoch, "splits": {}, "dataset_type": "full"}

        for split_name, (coco, images_dir) in datasets_full.items():
            predictions, fps = run_inference_on_split(
                model, coco, images_dir, model_name, split_name, CONF_THRESHOLD_STANDARD, is_yolo=True
            )
            metrics = evaluate_predictions(coco, predictions, model_name, split_name)
            results[model_name]["splits"][split_name] = {"metrics": metrics, "fps": fps}
            print(f"    {split_name}: mAP@0.5={metrics['mAP@0.5']:.2f}% | TP={metrics['true_positives']} | F1={metrics['f1_score']:.1f}%")

        results[model_name]["aggregated"] = aggregate_model_results(results[model_name])
        del model
        torch.cuda.empty_cache()

    # =================================================================
    # BENCHMARK YOLO 20K FINETUNE
    # =================================================================
    print(f"\n{'='*60}")
    print("BENCHMARKING YOLO 20k FINETUNE")
    print(f"{'='*60}")

    for epoch, checkpoint_path in sorted(YOLO_20K_CHECKPOINTS.items()):
        model_name = f"YOLO_20k_ep{epoch}"

        if not checkpoint_path.exists():
            print(f"\n  SKIP: {model_name} (not found)")
            continue

        print(f"\n  Processing {model_name}...")
        model = load_yolo_model(checkpoint_path)

        results[model_name] = {"type": "YOLO_20k", "epoch": epoch, "splits": {}, "dataset_type": "clean"}

        for split_name, (coco, images_dir) in datasets_clean.items():
            predictions, fps = run_inference_on_split(
                model, coco, images_dir, model_name, split_name, CONF_THRESHOLD_STANDARD, is_yolo=True
            )
            metrics = evaluate_predictions(coco, predictions, model_name, split_name)
            results[model_name]["splits"][split_name] = {"metrics": metrics, "fps": fps}
            print(f"    {split_name}: mAP@0.5={metrics['mAP@0.5']:.2f}% | TP={metrics['true_positives']} | F1={metrics['f1_score']:.1f}%")

        results[model_name]["aggregated"] = aggregate_model_results(results[model_name])
        del model
        torch.cuda.empty_cache()

    # =================================================================
    # BENCHMARK DETR ORIGINAL
    # =================================================================
    print(f"\n{'='*60}")
    print(f"BENCHMARKING DETR ORIGINAL (100k dataset, Q{DETR_QUERY_ID})")
    print(f"{'='*60}")

    for epoch, checkpoint_path in sorted(DETR_ORIGINAL_CHECKPOINTS.items()):
        model_name = f"DETR_Orig_ep{epoch}"

        if not checkpoint_path.exists():
            print(f"\n  SKIP: {model_name} (not found)")
            continue

        print(f"\n  Processing {model_name}...")
        model = load_detr_model(checkpoint_path)

        results[model_name] = {"type": "DETR_Original", "epoch": epoch, "splits": {}, "dataset_type": "full"}

        for split_name, (coco, images_dir) in datasets_full.items():
            predictions, fps = run_inference_on_split(
                model, coco, images_dir, model_name, split_name, CONF_THRESHOLD_STANDARD, is_yolo=False
            )
            metrics = evaluate_predictions(coco, predictions, model_name, split_name)
            results[model_name]["splits"][split_name] = {"metrics": metrics, "fps": fps}
            print(f"    {split_name}: mAP@0.5={metrics['mAP@0.5']:.2f}% | TP={metrics['true_positives']} | F1={metrics['f1_score']:.1f}%")

        results[model_name]["aggregated"] = aggregate_model_results(results[model_name])
        del model
        torch.cuda.empty_cache()

    # =================================================================
    # BENCHMARK DETR 20K FINETUNE
    # =================================================================
    print(f"\n{'='*60}")
    print(f"BENCHMARKING DETR 20k FINETUNE (Q{DETR_QUERY_ID})")
    print(f"{'='*60}")

    for epoch, checkpoint_path in sorted(DETR_20K_CHECKPOINTS.items()):
        model_name = f"DETR_20k_ep{epoch}"

        if not checkpoint_path.exists():
            print(f"\n  SKIP: {model_name} (not found)")
            continue

        print(f"\n  Processing {model_name}...")
        model = load_detr_model(checkpoint_path)

        results[model_name] = {"type": "DETR_20k", "epoch": epoch, "splits": {}, "dataset_type": "clean"}

        for split_name, (coco, images_dir) in datasets_clean.items():
            predictions, fps = run_inference_on_split(
                model, coco, images_dir, model_name, split_name, CONF_THRESHOLD_STANDARD, is_yolo=False
            )
            metrics = evaluate_predictions(coco, predictions, model_name, split_name)
            results[model_name]["splits"][split_name] = {"metrics": metrics, "fps": fps}
            print(f"    {split_name}: mAP@0.5={metrics['mAP@0.5']:.2f}% | TP={metrics['true_positives']} | F1={metrics['f1_score']:.1f}%")

        results[model_name]["aggregated"] = aggregate_model_results(results[model_name])
        del model
        torch.cuda.empty_cache()

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    results_file = BENCHMARK_DIR / "results_all_epochs.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {results_file}")

    report = generate_report(results)
    report_file = BENCHMARK_DIR / "BENCHMARK_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report: {report_file}")

    # Cleanup temp files
    for temp_file in BENCHMARK_DIR.glob("temp_*_annotations.json"):
        temp_file.unlink()

    elapsed_time = time.time() - start_time

    print(f"""
{'='*80}
BENCHMARK COMPLETE
{'='*80}
Total time: {elapsed_time/60:.2f} minutes
Output: {BENCHMARK_DIR}
Report: {report_file}
{'='*80}
""")

if __name__ == "__main__":
    main()
