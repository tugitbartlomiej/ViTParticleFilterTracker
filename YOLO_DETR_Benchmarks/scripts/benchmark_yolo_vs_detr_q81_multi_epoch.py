#!/usr/bin/env python3
"""
=============================================================================
BENCHMARK: YOLO vs DETR (Query 81 Only) - MULTI-EPOCH
=============================================================================
Compares:
  - YOLO Epoch 70 and 100
  - DETR Epochs 100, 120, 140, 160, 170 (using ONLY Query 81)

Key differences from standard benchmark:
  - DETR uses ONLY Query 81 for detections (most specialized for tooltips)
  - Additional visualizations with 85% confidence threshold

Dataset: Cataract Surgery Instruments (all splits combined)
Metrics: mAP@0.5, mAP@0.5:0.95, mAP@0.75, AR@100, FPS

Author: PhD Research - Cataract Surgery Instrument Detection
Date: 2025-12-08
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

# Suppress PyTorch meta-parameter warnings (harmless during checkpoint loading)
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
        """Compatibility stub for legacy DFLoss class"""
        pass
    loss_module.DFLoss = DFLoss
    sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss
    print("  - Added DFLoss stub")

# Monkey-patch torch.load for legacy YOLO checkpoints
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
# >>> TUTAJ MOŻNA ZMIENIĆ KTÓRE EPOKI TESTOWAĆ <<<
# =============================================================================

# YOLO Checkpoints - ZMIEŃ TUTAJ KTÓRE EPOKI YOLO CHCESZ TESTOWAĆ
# Dostępne: 70, 100, 120, 140, 160, 170
YOLO_CHECKPOINTS = {
    70: BASE_PATH / "Eden/Checkpoints/YOLO_EDEN_TRAIN/epoch70.pt",
    100: BASE_PATH / "Eden/Checkpoints/YOLO_EDEN_TRAIN/epoch100.pt",
    120: BASE_PATH / "Eden/Checkpoints/YOLO_EDEN_TRAIN/exp/weights/epoch120.pt",
    170: BASE_PATH / "Eden/Checkpoints/YOLO_EDEN_TRAIN/exp/weights/epoch170.pt",
}

# DETR Checkpoints - ZMIEŃ TUTAJ KTÓRE EPOKI DETR CHCESZ TESTOWAĆ
# Dostępne: 40, 60, 80, 100, 120, 140, 160, 170
DETR_CHECKPOINTS = {
    100: BASE_PATH / "Eden/Checkpoints/DETR/DETR_Checkpoints/checkpoint_epoch_100.pth",
    140: BASE_PATH / "Eden/Checkpoints/DETR/DETR_Checkpoints/checkpoint_epoch_140.pth",
    160: BASE_PATH / "Eden/Checkpoints/DETR/DETR_Checkpoints/checkpoint_epoch_160.pth",
    170: BASE_PATH / "Eden/Checkpoints/DETR/DETR_Checkpoints/checkpoint_epoch_170.pth",
}

# =============================================================================
# >>> KONIEC SEKCJI KONFIGURACJI EPOK <<<
# =============================================================================

# =============================================================================
# >>> TUTAJ MOŻNA ZMIENIĆ DATASET <<<
# =============================================================================

# OPCJA 1: Original Training Dataset (same-distribution test)
# - Ten sam dataset co trening, testuje accuracy na znanej dystrybucji
DATASET_ROOT = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/TestDatasetGenerator/output")
ANNOTATION_FILE = DATASET_ROOT / "annotations_reviewed_1040_coco.json"  # Single annotation file
IMAGES_DIR = DATASET_ROOT / "test_frames"  # Images directory
ALL_SPLITS = ["original_test"]  # Single split name for reporting

# OPCJA 2: External Roboflow Dataset (cross-dataset test) - zakomentowane
# DATASET_ROOT = Path("E:/cataract_surgery_Instruments_detection.v1i.coco")
# ANNOTATION_FILE = None  # Use per-split annotation files
# IMAGES_DIR = None  # Use split directories
# ALL_SPLITS = ["train", "valid", "test"]

# =============================================================================
# >>> KONIEC SEKCJI KONFIGURACJI DATASETU <<<
# =============================================================================

# Output
OUTPUT_DIR = BASE_PATH / "YOLO_DETR_Benchmarks/Benchmarks"
BENCHMARK_DIR = OUTPUT_DIR / f"BENCHMARK_Q81_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# >>> TUTAJ MOŻNA ZMIENIĆ THRESHOLDY I QUERY <<<
# =============================================================================

# DETR Query - który query używać (Q81 jest najlepszy dla tooltip)
DETR_QUERY_ID = 81

# Threshold standardowy - używany do głównych metryk i porównań
CONF_THRESHOLD_STANDARD = 0.3   # 30%

# Threshold wysoki - do testów z wysoką pewnością
CONF_THRESHOLD_HIGH = 0.85      # 85%

# IoU threshold dla matching (nie zmieniaj bez powodu)
IOU_THRESHOLD = 0.3

# =============================================================================
# >>> KONIEC SEKCJI THRESHOLDÓW <<<
# =============================================================================
YOLO_MAX_DETECTIONS = 300

# Visualization settings
VIS_SAMPLE_COUNT = 50  # Number of images to visualize per threshold
VISUALIZATION_SEED = 42

MODEL_COLORS = {
    "Ground Truth": (0, 0, 255),      # Blue
    "YOLO": (0, 255, 0),              # Green
    "DETR_Q81": (255, 165, 0),        # Orange
}

print(f"""
{'='*80}
BENCHMARK: YOLO vs DETR (Query 81 Only) - MULTI-EPOCH
{'='*80}
Device: {DEVICE}
Dataset: {DATASET_ROOT}
Splits: {ALL_SPLITS} (combined)
Output: {BENCHMARK_DIR}

YOLO Checkpoints: {list(YOLO_CHECKPOINTS.keys())}
DETR Checkpoints: {list(DETR_CHECKPOINTS.keys())}

DETR Query: Q{DETR_QUERY_ID} ONLY
Standard Threshold: {CONF_THRESHOLD_STANDARD*100:.0f}%
High Threshold: {CONF_THRESHOLD_HIGH*100:.0f}%
{'='*80}
""")

# Create output directories
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = BENCHMARK_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

# Visualization subdirectories
VIS_COMPARISONS_DIR = VIS_DIR / "comparisons"
VIS_YOLO_85_DIR = VIS_DIR / "YOLO_threshold_85"
VIS_DETR_Q81_85_DIR = VIS_DIR / "DETR_Q81_threshold_85"

VIS_COMPARISONS_DIR.mkdir(exist_ok=True)
VIS_YOLO_85_DIR.mkdir(exist_ok=True)
VIS_DETR_Q81_85_DIR.mkdir(exist_ok=True)

# =============================================================================
# MODEL LOADERS
# =============================================================================

DETR_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def load_yolo_model(checkpoint_path):
    """Load YOLO model from checkpoint"""
    print(f"  Loading YOLO from {checkpoint_path.name}...")
    model = YOLO(str(checkpoint_path))
    model.to(DEVICE)
    return model

def load_detr_model(checkpoint_path):
    """Load DETR model from checkpoint"""
    print(f"  Loading DETR from {checkpoint_path.name}...")

    # Suppress HuggingFace loading messages temporarily
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)

    # Use strict=False to ignore extra keys like num_batches_tracked (harmless BatchNorm stats)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Restore logging level
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)

    model.to(DEVICE)
    model.eval()
    return model

# =============================================================================
# DATASET PREPARATION
# =============================================================================

def load_all_images():
    """Load images from configured directory

    Supports two modes:
    1. Single images directory (IMAGES_DIR is set) - for original training dataset
    2. Per-split directories - for Roboflow dataset
    """
    print(f"\nLoading images...")
    image_files = []

    if IMAGES_DIR is not None and IMAGES_DIR.exists():
        # Mode 1: Single images directory
        split_images = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
        image_files.extend(split_images)
        print(f"  {IMAGES_DIR.name}: {len(split_images)} images")
    else:
        # Mode 2: Per-split directories
        for split in ALL_SPLITS:
            split_dir = DATASET_ROOT / split
            if split_dir.exists():
                split_images = sorted(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))
                image_files.extend(split_images)
                print(f"  {split}: {len(split_images)} images")
            else:
                print(f"  {split}: NOT FOUND")

    print(f"  TOTAL: {len(image_files)} images")
    return image_files

def prepare_coco_annotations(split_name):
    """Load and normalize COCO annotations (map all categories to id=1)

    Supports two modes:
    1. Single annotation file (ANNOTATION_FILE is set) - for original training dataset
    2. Per-split annotation files (_annotations.coco.json in each split dir) - for Roboflow
    """
    # Determine annotation path and images directory
    if ANNOTATION_FILE is not None and ANNOTATION_FILE.exists():
        # Mode 1: Single annotation file (original training dataset)
        ann_path = ANNOTATION_FILE
        images_dir = IMAGES_DIR
    else:
        # Mode 2: Per-split annotation files (Roboflow dataset)
        ann_path = DATASET_ROOT / split_name / "_annotations.coco.json"
        images_dir = DATASET_ROOT / split_name

    if not ann_path.exists():
        print(f"  WARNING: Annotation file not found: {ann_path}")
        return None, None

    if not images_dir.exists():
        print(f"  WARNING: Images directory not found: {images_dir}")
        return None, None

    with open(ann_path) as f:
        coco_data = json.load(f)

    # Normalize to single category
    coco_data['categories'] = [{"id": 1, "name": "instrument", "supercategory": "surgical"}]
    for ann in coco_data['annotations']:
        ann['category_id'] = 1

    temp_ann_path = BENCHMARK_DIR / f"temp_{split_name}_annotations.json"
    with open(temp_ann_path, 'w') as f:
        json.dump(coco_data, f)

    coco = COCO(str(temp_ann_path))

    return coco, images_dir

# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def run_yolo_inference(model, image_path, conf_threshold):
    """Run YOLO inference on single image"""
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
    """
    Run DETR inference using ONLY Query 81.

    This function extracts predictions specifically from Query 81,
    which has been identified as the most specialized for tooltip detection.
    """
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    inputs = DETR_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get raw outputs
    logits = outputs.logits[0]  # [100, num_classes+1]
    boxes = outputs.pred_boxes[0]  # [100, 4] normalized cxcywh

    # Apply softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get ONLY Query 81's prediction
    q81_score = float(probs[DETR_QUERY_ID, 0])  # Class 0 probability for Q81

    predictions = []

    if q81_score >= conf_threshold:
        # Convert normalized cxcywh to pixel xyxy then to xywh
        cx, cy, w, h = boxes[DETR_QUERY_ID].cpu().numpy()

        # Denormalize
        cx *= img_width
        cy *= img_height
        w *= img_width
        h *= img_height

        # Convert to x, y, w, h format
        x = cx - w / 2
        y = cy - h / 2

        predictions.append({
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": q81_score,
            "query_id": DETR_QUERY_ID
        })

    return predictions

def run_yolo_inference_on_split(model, coco, images_dir, model_name, split_name, conf_threshold):
    """Run YOLO inference on entire split"""
    image_ids = coco.getImgIds()
    predictions = []
    inference_times = []

    for img_id in tqdm(image_ids, desc=f"{model_name} on {split_name}", leave=False):
        img_info = coco.loadImgs(img_id)[0]
        img_path = images_dir / img_info['file_name']

        if not img_path.exists():
            continue

        start_time = time.time()
        preds = run_yolo_inference(model, img_path, conf_threshold)
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

def run_detr_q81_inference_on_split(model, coco, images_dir, model_name, split_name, conf_threshold):
    """Run DETR Q81 inference on entire split"""
    image_ids = coco.getImgIds()
    predictions = []
    inference_times = []

    for img_id in tqdm(image_ids, desc=f"{model_name} on {split_name}", leave=False):
        img_info = coco.loadImgs(img_id)[0]
        img_path = images_dir / img_info['file_name']

        if not img_path.exists():
            continue

        start_time = time.time()
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
# VISUALIZATION
# =============================================================================

def draw_boxes_on_image(image, boxes, color, label, conf_threshold=0.0):
    """Draw bounding boxes on image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Draw label
    label_bbox = draw.textbbox((0, 0), label, font=font)
    label_h = label_bbox[3] - label_bbox[1]
    draw.rectangle([0, 0, 200, label_h + 10], fill=color)
    draw.text((5, 5), label, fill="white", font=font)

    # Draw boxes
    num_drawn = 0
    for box_data in boxes:
        if box_data.get('score', 1.0) < conf_threshold:
            continue

        bbox = box_data.get('bbox', box_data)
        if isinstance(bbox, dict):
            bbox = bbox['bbox']

        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        score = box_data.get('score', None)
        if score is not None:
            score_text = f"{score:.2f}"
            draw.text((x1 + 2, y1 + 2), score_text, fill=color, font=font_small)

        num_drawn += 1

    # Detection count
    count_text = f"Det: {num_drawn}"
    draw.text((5, label_h + 15), count_text, fill="yellow", font=font_small)

    return img

def create_comparison_visualization(image_path, gt_boxes, yolo_preds, detr_preds, output_path, threshold):
    """Create side-by-side comparison: GT | YOLO | DETR_Q81"""
    original = Image.open(image_path).convert("RGB")
    img_w, img_h = original.size

    # Create three images
    gt_img = draw_boxes_on_image(
        original,
        [{"bbox": b, "score": 1.0} for b in gt_boxes],
        MODEL_COLORS["Ground Truth"],
        f"Ground Truth",
        conf_threshold=0.0
    )

    yolo_img = draw_boxes_on_image(
        original,
        yolo_preds,
        MODEL_COLORS["YOLO"],
        f"YOLO (thr={threshold*100:.0f}%)",
        conf_threshold=threshold
    )

    detr_img = draw_boxes_on_image(
        original,
        detr_preds,
        MODEL_COLORS["DETR_Q81"],
        f"DETR Q81 (thr={threshold*100:.0f}%)",
        conf_threshold=threshold
    )

    # Combine horizontally
    padding = 5
    combined_w = 3 * img_w + 4 * padding
    combined_h = img_h + 2 * padding

    combined = Image.new("RGB", (combined_w, combined_h), (255, 255, 255))
    combined.paste(gt_img, (padding, padding))
    combined.paste(yolo_img, (2 * padding + img_w, padding))
    combined.paste(detr_img, (3 * padding + 2 * img_w, padding))

    combined.save(output_path, quality=95)

def generate_threshold_visualizations(image_files, yolo_model, detr_model, coco_dict, threshold, output_dir, description):
    """Generate visualizations for specific threshold"""
    print(f"\n  Generating {description} visualizations (threshold={threshold*100:.0f}%)...")

    rng = random.Random(VISUALIZATION_SEED)
    sample_files = rng.sample(image_files, min(VIS_SAMPLE_COUNT, len(image_files)))

    for img_path in tqdm(sample_files, desc=description, leave=False):
        # Find which split this image belongs to
        split_name = img_path.parent.name
        coco = coco_dict.get(split_name)

        if coco is None:
            continue

        # Find image_id
        img_name = img_path.name
        img_id = None
        for img_info in coco.dataset['images']:
            if img_info['file_name'] == img_name:
                img_id = img_info['id']
                break

        if img_id is None:
            continue

        # Get ground truth
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in annotations]

        # Run inference
        yolo_preds = run_yolo_inference(yolo_model, img_path, threshold)
        detr_preds = run_detr_q81_inference(detr_model, img_path, threshold)

        # Save visualization
        output_path = output_dir / f"vis_{img_path.stem}.jpg"
        create_comparison_visualization(img_path, gt_boxes, yolo_preds, detr_preds, output_path, threshold)

# =============================================================================
# EVALUATION
# =============================================================================

def bbox_iou_xywh(box_a, box_b):
    """Calculate IoU between two boxes in xywh format"""
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
    """Compute TP, FP, FN statistics"""
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
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
    }

def evaluate_predictions(coco, predictions, model_name, split_name):
    """Evaluate predictions using COCO metrics"""
    if not predictions:
        return {
            "mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "mAP@0.75": 0.0, "AR@100": 0.0,
            "true_positives": 0, "false_positives": 0, "false_negatives": 0,
            "precision": 0.0, "recall": 0.0,
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
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_model_results(model_data):
    """Aggregate results across all splits (train+valid+test) for a model"""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_images = 0

    for split_name, split_data in model_data.get("splits", {}).items():
        metrics = split_data.get("metrics", {})
        total_tp += metrics.get("true_positives", 0)
        total_fp += metrics.get("false_positives", 0)
        total_fn += metrics.get("false_negatives", 0)
        total_images += split_data.get("num_predictions", 0) // max(1, metrics.get("true_positives", 0) + metrics.get("false_positives", 0))

    # Calculate aggregated metrics
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
    """Generate markdown report"""
    report = f"""# Benchmark Report: YOLO vs DETR (Query 81 Only)

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Device:** {DEVICE}
**Dataset:** Cataract Surgery Instruments (all splits combined)

## Configuration

- **DETR Query:** Q{DETR_QUERY_ID} ONLY (most specialized for tooltip detection)
- **Standard Threshold:** {CONF_THRESHOLD_STANDARD*100:.0f}%
- **High Threshold:** {CONF_THRESHOLD_HIGH*100:.0f}%

---

## Summary Table (Validation Set, Standard Threshold {CONF_THRESHOLD_STANDARD*100:.0f}%)

| Model | Epoch | mAP@0.5 | mAP@0.5:0.95 | AR@100 | TP | FP | FN | Precision | Recall |
|-------|-------|---------|--------------|--------|----|----|----|-----------| -------|
"""

    sorted_models = sorted(results.items(), key=lambda x: (x[1].get('type', ''), x[1].get('epoch', 0)))

    for model_name, model_data in sorted_models:
        if "valid" in model_data.get("splits", {}):
            m = model_data["splits"]["valid"]["metrics"]
            report += f"| {model_data.get('type', 'N/A')} | {model_data.get('epoch', 'N/A')} | "
            report += f"{m['mAP@0.5']:.2f}% | {m['mAP@0.5:0.95']:.2f}% | {m['AR@100']:.2f}% | "
            report += f"{m['true_positives']} | {m['false_positives']} | {m['false_negatives']} | "
            report += f"{m['precision']:.1f}% | {m['recall']:.1f}% |\n"

    # AGGREGATED RESULTS TABLE (train + valid + test combined)
    report += f"""
---

## AGGREGATED Results (train + valid + test combined)

This is the most important table - shows model performance across ALL data splits.

| Model | Epoch | Total TP | Total FP | Total FN | Precision | Recall | F1-Score |
|-------|-------|----------|----------|----------|-----------|--------|----------|
"""

    for model_name, model_data in sorted_models:
        agg = aggregate_model_results(model_data)
        report += f"| {model_data.get('type', 'N/A')} | {model_data.get('epoch', 'N/A')} | "
        report += f"{agg['total_tp']} | {agg['total_fp']} | {agg['total_fn']} | "
        report += f"{agg['precision']:.1f}% | {agg['recall']:.1f}% | {agg['f1_score']:.1f}% |\n"

    # Add per-split breakdown
    report += f"""
---

## Per-Split Breakdown

"""

    for model_name, model_data in sorted_models:
        report += f"### {model_data.get('type', 'N/A')} Epoch {model_data.get('epoch', 'N/A')}\n\n"
        report += "| Split | TP | FP | FN | Precision | Recall |\n"
        report += "|-------|----|----|----|-----------| -------|\n"
        for split_name in ["train", "valid", "test"]:
            if split_name in model_data.get("splits", {}):
                m = model_data["splits"][split_name]["metrics"]
                report += f"| {split_name} | {m['true_positives']} | {m['false_positives']} | {m['false_negatives']} | "
                report += f"{m['precision']:.1f}% | {m['recall']:.1f}% |\n"
        agg = aggregate_model_results(model_data)
        report += f"| **TOTAL** | **{agg['total_tp']}** | **{agg['total_fp']}** | **{agg['total_fn']}** | "
        report += f"**{agg['precision']:.1f}%** | **{agg['recall']:.1f}%** |\n\n"

    report += f"""
---

## Visualizations

### Standard Threshold ({CONF_THRESHOLD_STANDARD*100:.0f}%)
- Location: `visualizations/comparisons/`

### High Threshold ({CONF_THRESHOLD_HIGH*100:.0f}%)
- YOLO: `visualizations/YOLO_threshold_85/`
- DETR Q81: `visualizations/DETR_Q81_threshold_85/`

---

## Key Findings

### DETR Query 81 Analysis
Query 81 has been identified as the most specialized query for tooltip detection,
accounting for ~50% of all DETR detections across all epochs.

By using ONLY Query 81, we expect:
- More consistent detections
- Reduced false positives from other queries
- Better precision at high confidence thresholds

---

**Output Directory:** `{BENCHMARK_DIR}`
"""

    return report

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    start_time = time.time()

    results = defaultdict(dict)

    # Prepare datasets
    print("\nPreparing datasets...")
    datasets = {}
    coco_dict = {}
    for split in ALL_SPLITS:
        coco, images_dir = prepare_coco_annotations(split)
        if coco:
            datasets[split] = (coco, images_dir)
            coco_dict[split] = coco
            print(f"  {split}: {len(coco.getImgIds())} images")

    if not datasets:
        print("ERROR: No valid datasets found!")
        sys.exit(1)

    # Load all image files for visualization
    image_files = load_all_images()

    # =================================================================
    # BENCHMARK YOLO MODELS
    # =================================================================
    print(f"\n{'='*60}")
    print("BENCHMARKING YOLO MODELS")
    print(f"{'='*60}")

    yolo_model_for_vis = None  # Keep one for visualization

    for epoch, checkpoint_path in sorted(YOLO_CHECKPOINTS.items()):
        model_name = f"YOLO_epoch{epoch}"

        if not checkpoint_path.exists():
            print(f"\n  SKIP: {model_name} (not found)")
            continue

        print(f"\n  Processing {model_name}...")
        model = load_yolo_model(checkpoint_path)

        if yolo_model_for_vis is None:
            yolo_model_for_vis = model

        results[model_name] = {
            "type": "YOLO",
            "epoch": epoch,
            "splits": {}
        }

        for split_name, (coco, images_dir) in datasets.items():
            predictions, fps = run_yolo_inference_on_split(
                model, coco, images_dir, model_name, split_name, CONF_THRESHOLD_STANDARD
            )
            metrics = evaluate_predictions(coco, predictions, model_name, split_name)

            results[model_name]["splits"][split_name] = {
                "metrics": metrics,
                "fps": fps,
                "num_predictions": len(predictions)
            }

            print(f"    {split_name}: mAP@0.5={metrics['mAP@0.5']:.2f}% | "
                  f"TP={metrics['true_positives']} | FP={metrics['false_positives']} | FN={metrics['false_negatives']}")

        if model != yolo_model_for_vis:
            del model
            torch.cuda.empty_cache()

    # =================================================================
    # BENCHMARK DETR MODELS (Q81 ONLY)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"BENCHMARKING DETR MODELS (Query {DETR_QUERY_ID} ONLY)")
    print(f"{'='*60}")

    detr_model_for_vis = None  # Keep one for visualization

    for epoch, checkpoint_path in sorted(DETR_CHECKPOINTS.items()):
        model_name = f"DETR_Q81_epoch{epoch}"

        if not checkpoint_path.exists():
            print(f"\n  SKIP: {model_name} (not found)")
            continue

        print(f"\n  Processing {model_name}...")
        model = load_detr_model(checkpoint_path)

        if detr_model_for_vis is None:
            detr_model_for_vis = model

        results[model_name] = {
            "type": "DETR_Q81",
            "epoch": epoch,
            "splits": {}
        }

        for split_name, (coco, images_dir) in datasets.items():
            predictions, fps = run_detr_q81_inference_on_split(
                model, coco, images_dir, model_name, split_name, CONF_THRESHOLD_STANDARD
            )
            metrics = evaluate_predictions(coco, predictions, model_name, split_name)

            results[model_name]["splits"][split_name] = {
                "metrics": metrics,
                "fps": fps,
                "num_predictions": len(predictions)
            }

            print(f"    {split_name}: mAP@0.5={metrics['mAP@0.5']:.2f}% | "
                  f"TP={metrics['true_positives']} | FP={metrics['false_positives']} | FN={metrics['false_negatives']}")

        if model != detr_model_for_vis:
            del model
            torch.cuda.empty_cache()

    # =================================================================
    # GENERATE VISUALIZATIONS
    # =================================================================
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    if yolo_model_for_vis and detr_model_for_vis:
        # Standard threshold comparisons
        generate_threshold_visualizations(
            image_files, yolo_model_for_vis, detr_model_for_vis, coco_dict,
            CONF_THRESHOLD_STANDARD, VIS_COMPARISONS_DIR, "Standard (30%)"
        )

        # High threshold - YOLO only
        print(f"\n  Generating YOLO 85% threshold visualizations...")
        rng = random.Random(VISUALIZATION_SEED)
        sample_files = rng.sample(image_files, min(VIS_SAMPLE_COUNT, len(image_files)))

        for img_path in tqdm(sample_files, desc="YOLO 85%", leave=False):
            split_name = img_path.parent.name
            coco = coco_dict.get(split_name)
            if coco is None:
                continue

            img_name = img_path.name
            img_id = None
            for img_info in coco.dataset['images']:
                if img_info['file_name'] == img_name:
                    img_id = img_info['id']
                    break
            if img_id is None:
                continue

            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            gt_boxes = [ann['bbox'] for ann in annotations]

            yolo_preds = run_yolo_inference(yolo_model_for_vis, img_path, CONF_THRESHOLD_HIGH)

            original = Image.open(img_path).convert("RGB")
            vis_img = draw_boxes_on_image(
                original, yolo_preds, MODEL_COLORS["YOLO"],
                f"YOLO (thr=85%)", CONF_THRESHOLD_HIGH
            )
            vis_img.save(VIS_YOLO_85_DIR / f"yolo85_{img_path.stem}.jpg", quality=95)

        # High threshold - DETR Q81 only
        print(f"\n  Generating DETR Q81 85% threshold visualizations...")
        for img_path in tqdm(sample_files, desc="DETR Q81 85%", leave=False):
            split_name = img_path.parent.name
            coco = coco_dict.get(split_name)
            if coco is None:
                continue

            img_name = img_path.name
            img_id = None
            for img_info in coco.dataset['images']:
                if img_info['file_name'] == img_name:
                    img_id = img_info['id']
                    break
            if img_id is None:
                continue

            detr_preds = run_detr_q81_inference(detr_model_for_vis, img_path, CONF_THRESHOLD_HIGH)

            original = Image.open(img_path).convert("RGB")
            vis_img = draw_boxes_on_image(
                original, detr_preds, MODEL_COLORS["DETR_Q81"],
                f"DETR Q81 (thr=85%)", CONF_THRESHOLD_HIGH
            )
            vis_img.save(VIS_DETR_Q81_85_DIR / f"detr_q81_85_{img_path.stem}.jpg", quality=95)

    # Cleanup
    if yolo_model_for_vis:
        del yolo_model_for_vis
    if detr_model_for_vis:
        del detr_model_for_vis
    torch.cuda.empty_cache()

    # =================================================================
    # SAVE RESULTS AND REPORT
    # =================================================================
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    # Add aggregated results to each model
    results_with_aggregated = {}
    for model_name, model_data in results.items():
        results_with_aggregated[model_name] = dict(model_data)
        results_with_aggregated[model_name]["aggregated"] = aggregate_model_results(model_data)

    # Save raw results with aggregation
    results_file = BENCHMARK_DIR / "results_summary.json"
    with open(results_file, 'w') as f:
        json.dump(results_with_aggregated, f, indent=2, default=str)
    print(f"  Results: {results_file}")

    # Generate and save report
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

Visualizations:
  - Comparisons (30%): {VIS_COMPARISONS_DIR}
  - YOLO (85%): {VIS_YOLO_85_DIR}
  - DETR Q81 (85%): {VIS_DETR_Q81_85_DIR}

Report: {report_file}
{'='*80}
""")


if __name__ == "__main__":
    main()
