"""
Ensemble Query Selector for DETR

Implements intelligent query selection strategies to improve upon Query81's 83.5% hit rate.
Based on query_analyzer.py findings, combines multiple specialist queries.

Strategies:
1. Multi-Query Voting: Combine predictions from Query81 + Query7 + top performers
2. Confidence-Weighted Selection: Choose query with highest confidence
3. NMS Ensemble: Use NMS across all queries to find best detection
"""

import yaml
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inference import load_config, get_image_files, build_coco_image_id_map, resolve_path, load_detr_from_checkpoint, get_category_id_mapper


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def nms_across_queries(predictions, iou_threshold=0.5):
    """
    Non-Maximum Suppression across multiple queries.
    Select best prediction from overlapping detections.

    Args:
        predictions: List of dicts with 'box', 'confidence', 'query_idx'
        iou_threshold: IoU threshold for considering boxes as duplicates

    Returns:
        List of selected predictions after NMS
    """
    if not predictions:
        return []

    # Sort by confidence descending
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

    selected = []
    while predictions:
        # Take highest confidence prediction
        best = predictions.pop(0)
        selected.append(best)

        # Remove all predictions that overlap with best
        predictions = [
            p for p in predictions
            if compute_iou(best['box'], p['box']) < iou_threshold
        ]

    return selected


def ensemble_inference_multi_query(config, selected_queries=[81, 7, 89, 53],
                                   strategy='nms', conf_threshold=0.5):
    """
    Run DETR inference with ensemble of multiple queries.

    Args:
        config: Benchmark configuration
        selected_queries: List of query indices to use
        strategy: Selection strategy - 'nms', 'max_conf', or 'voting'
        conf_threshold: Confidence threshold

    Returns:
        COCO-format predictions
    """
    print(f"\n{'='*80}")
    print(f"DETR Ensemble Inference")
    print(f"{'='*80}")
    print(f"Strategy: {strategy}")
    print(f"Selected queries: {selected_queries}")
    print(f"Confidence threshold: {conf_threshold}")

    # Load model
    checkpoint_candidates = [
        'models/DETR/checkpoint_epoch_100.pth',
        '../BackgroundFinetuned/Models/DETR/checkpoint_epoch_100.pth',
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        try:
            p = resolve_path(candidate, must_exist=True, expect_dir=False)
            checkpoint_path = p
            break
        except FileNotFoundError:
            continue

    if not checkpoint_path:
        raise FileNotFoundError("Could not find DETR checkpoint!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_detr_from_checkpoint(checkpoint_path, device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load dataset
    images_dir = resolve_path(config['dataset']['images_dir'], expect_dir=True)
    ann_path = config['dataset'].get('annotations_path')
    if ann_path:
        try:
            ann_path = resolve_path(ann_path, expect_dir=False)
        except FileNotFoundError:
            ann_path = None

    image_files = get_image_files(images_dir)
    gt_map = build_coco_image_id_map(ann_path)
    image_id_map = gt_map if gt_map else {name: i for i, name in enumerate(image_files)}
    cat_map = get_category_id_mapper(config, 'detr')

    coco_results = []

    # Run inference
    for image_name in tqdm(image_files, desc=f"DETR Ensemble ({strategy})"):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get predictions from selected queries
        logits = outputs.logits[0]  # [100, num_classes]
        pred_boxes = outputs.pred_boxes[0]  # [100, 4]
        probs = logits.softmax(-1)

        img_w, img_h = image.size
        query_predictions = []

        for query_idx in selected_queries:
            tool_prob = probs[query_idx, 0].item()

            # Skip low confidence
            if tool_prob < conf_threshold:
                continue

            # Convert box to pixels
            box_normalized = pred_boxes[query_idx]
            cx, cy, w, h = box_normalized.tolist()
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h

            query_predictions.append({
                'query_idx': query_idx,
                'confidence': tool_prob,
                'box': [x1, y1, x2, y2]
            })

        # Apply selection strategy
        if strategy == 'nms':
            # Use NMS to select best among overlapping predictions
            selected_preds = nms_across_queries(query_predictions, iou_threshold=0.5)
        elif strategy == 'max_conf':
            # Simply take highest confidence
            selected_preds = [max(query_predictions, key=lambda x: x['confidence'])] if query_predictions else []
        elif strategy == 'voting':
            # If multiple queries agree (IoU > 0.5), average their boxes
            selected_preds = nms_across_queries(query_predictions, iou_threshold=0.5)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Convert to COCO format
        image_id = image_id_map.get(image_name, image_id_map.get(os.path.basename(image_name)))
        if image_id is None:
            image_id = image_files.index(image_name)

        for pred in selected_preds:
            x1, y1, x2, y2 = pred['box']
            width = x2 - x1
            height = y2 - y1

            category_id = 0
            if cat_map is not None:
                category_id = int(cat_map.get(0, 0))
            else:
                category_id = 0

            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": round(pred['confidence'], 3),
                "query_idx": pred['query_idx']  # Track which query was selected
            })

    return coco_results


def run_ensemble_benchmark(config, output_dir="./ensemble_results"):
    """
    Run benchmark with different ensemble strategies and compare results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define strategies to test
    strategies = [
        {
            'name': 'Query81_only',
            'queries': [81],
            'strategy': 'max_conf',
            'conf_threshold': 0.5
        },
        {
            'name': 'Top4_NMS',
            'queries': [81, 7, 89, 53],
            'strategy': 'nms',
            'conf_threshold': 0.5
        },
        {
            'name': 'Top4_MaxConf',
            'queries': [81, 7, 89, 53],
            'strategy': 'max_conf',
            'conf_threshold': 0.5
        },
        {
            'name': 'Top10_NMS',
            'queries': [81, 7, 89, 53, 4, 24, 99, 58, 25, 1],
            'strategy': 'nms',
            'conf_threshold': 0.5
        },
        {
            'name': 'Query81_7_NMS',
            'queries': [81, 7],
            'strategy': 'nms',
            'conf_threshold': 0.5
        },
    ]

    results_summary = {}

    for strat in strategies:
        print(f"\n{'='*80}")
        print(f"Testing strategy: {strat['name']}")
        print(f"{'='*80}")

        predictions = ensemble_inference_multi_query(
            config,
            selected_queries=strat['queries'],
            strategy=strat['strategy'],
            conf_threshold=strat['conf_threshold']
        )

        # Save predictions
        pred_file = os.path.join(output_dir, f"{strat['name']}_predictions.json")
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved predictions to: {pred_file}")
        print(f"Total detections: {len(predictions)}")

        # Quick stats on which queries were used
        query_usage = defaultdict(int)
        for pred in predictions:
            query_usage[pred['query_idx']] += 1

        print(f"Query usage distribution:")
        for q_idx in sorted(query_usage.keys()):
            print(f"  Query {q_idx:3d}: {query_usage[q_idx]:4d} detections")

        results_summary[strat['name']] = {
            'total_detections': len(predictions),
            'query_usage': dict(query_usage),
            'pred_file': pred_file
        }

    # Save summary
    summary_file = os.path.join(output_dir, 'ensemble_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Ensemble benchmark complete!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

    return results_summary


if __name__ == "__main__":
    config = load_config('config.yaml')
    results = run_ensemble_benchmark(config)
