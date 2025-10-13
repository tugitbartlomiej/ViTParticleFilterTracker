"""
DETR Query Analyzer - Find optimal query selection strategy

This script analyzes all 100 DETR queries to find which queries perform best
for different images/scenarios, potentially improving upon Query81's 96.3% hit rate.

Strategy:
1. Run inference capturing ALL query outputs (not just above threshold)
2. For each image, identify which queries produce best detections
3. Compare against ground truth to find "oracle" performance
4. Identify query specialization patterns
5. Create adaptive query selection strategy
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
from pycocotools import mask as mask_util

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inference import load_config, get_image_files, build_coco_image_id_map, resolve_path, load_detr_from_checkpoint


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


def analyze_all_queries(config, output_dir="./query_analysis"):
    """
    Analyze all 100 DETR queries for each image to find optimal query selection.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load DETR model with Query81 specialization preserved
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
    ann_path = resolve_path(config['dataset']['annotations_path'], expect_dir=False)

    image_files = get_image_files(images_dir)
    coco = COCO(ann_path)
    gt_map = build_coco_image_id_map(ann_path)

    print(f"\n{'='*80}")
    print(f"DETR Query Analysis - Analyzing all 100 queries")
    print(f"{'='*80}")
    print(f"Images: {len(image_files)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Statistics collectors
    query_stats = defaultdict(lambda: {
        'total_detections': 0,
        'high_conf_detections': 0,  # confidence > 0.7
        'matched_gt': 0,  # IoU > 0.5 with ground truth
        'best_for_image': 0,  # Times this query was best for an image
        'confidence_sum': 0.0,
        'iou_sum': 0.0,
        'image_hits': []  # Which images this query detected well
    })

    per_image_results = []

    # Analyze each image
    for image_name in tqdm(image_files, desc="Analyzing queries per image"):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_id = gt_map.get(image_name)

        # Get ground truth boxes for this image
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in anns]  # [x, y, w, h]
        # Convert to [x1, y1, x2, y2]
        gt_boxes_xyxy = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in gt_boxes]

        # Run DETR inference
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get raw predictions from all 100 queries
        # outputs.logits shape: [batch_size, num_queries, num_classes]
        # outputs.pred_boxes shape: [batch_size, num_queries, 4]

        logits = outputs.logits[0]  # [100, num_classes]
        pred_boxes = outputs.pred_boxes[0]  # [100, 4] in normalized [cx, cy, w, h]

        # Convert to probabilities
        probs = logits.softmax(-1)

        # For each query, analyze its predictions
        image_query_results = []

        for query_idx in range(logits.shape[0]):  # 100 queries
            # Get class probabilities (class 0 is our surgical_tool)
            tool_prob = probs[query_idx, 0].item()

            # Convert box from normalized [cx, cy, w, h] to [x1, y1, x2, y2] in pixel coordinates
            img_w, img_h = image.size
            box_normalized = pred_boxes[query_idx]
            cx, cy, w, h = box_normalized.tolist()

            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h

            pred_box_xyxy = [x1, y1, x2, y2]

            # Compute best IoU with ground truth
            best_iou = 0.0
            if gt_boxes_xyxy:
                ious = [compute_iou(pred_box_xyxy, gt_box) for gt_box in gt_boxes_xyxy]
                best_iou = max(ious)

            # Record query statistics
            query_stats[query_idx]['total_detections'] += 1
            query_stats[query_idx]['confidence_sum'] += tool_prob
            query_stats[query_idx]['iou_sum'] += best_iou

            if tool_prob > 0.7:
                query_stats[query_idx]['high_conf_detections'] += 1

            if best_iou > 0.5:
                query_stats[query_idx]['matched_gt'] += 1
                query_stats[query_idx]['image_hits'].append(image_id)

            # Store per-image query result
            image_query_results.append({
                'query_idx': query_idx,
                'confidence': tool_prob,
                'iou': best_iou,
                'box': [float(x) for x in pred_box_xyxy],
                'matched': best_iou > 0.5
            })

        # Find best query for this image (highest IoU)
        best_query = max(image_query_results, key=lambda x: x['iou'])
        query_stats[best_query['query_idx']]['best_for_image'] += 1

        # Also find best by confidence
        best_by_conf = max(image_query_results, key=lambda x: x['confidence'])

        per_image_results.append({
            'image_name': image_name,
            'image_id': image_id,
            'gt_boxes': gt_boxes_xyxy,
            'best_query_by_iou': best_query,
            'best_query_by_conf': best_by_conf,
            'query81_result': image_query_results[81],
            'all_queries': image_query_results
        })

    # Calculate final statistics
    num_images = len(image_files)
    query_performance = []

    for query_idx in range(100):
        stats = query_stats[query_idx]
        avg_conf = stats['confidence_sum'] / num_images
        avg_iou = stats['iou_sum'] / num_images
        hit_rate = (stats['matched_gt'] / num_images) * 100

        query_performance.append({
            'query_idx': query_idx,
            'hit_rate': hit_rate,
            'avg_confidence': avg_conf,
            'avg_iou': avg_iou,
            'high_conf_detections': stats['high_conf_detections'],
            'matched_gt': stats['matched_gt'],
            'best_for_image_count': stats['best_for_image']
        })

    # Sort by hit rate
    query_performance.sort(key=lambda x: x['hit_rate'], reverse=True)

    # Calculate oracle performance (always picking best query)
    oracle_matches = sum(1 for img in per_image_results if img['best_query_by_iou']['matched'])
    oracle_hit_rate = (oracle_matches / num_images) * 100

    # Query81 performance
    query81_matches = sum(1 for img in per_image_results if img['query81_result']['matched'])
    query81_hit_rate = (query81_matches / num_images) * 100

    # Print results
    print(f"\n{'='*80}")
    print(f"ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"\nQuery81 Performance:")
    print(f"  Hit Rate: {query81_hit_rate:.1f}% ({query81_matches}/{num_images} images)")
    print(f"  Avg Confidence: {query_stats[81]['confidence_sum']/num_images:.3f}")
    print(f"  Avg IoU: {query_stats[81]['iou_sum']/num_images:.3f}")

    print(f"\nOracle Performance (always best query):")
    print(f"  Hit Rate: {oracle_hit_rate:.1f}% ({oracle_matches}/{num_images} images)")
    print(f"  Potential Improvement: +{oracle_hit_rate - query81_hit_rate:.1f}%")

    print(f"\nTop 10 Queries by Hit Rate:")
    for i, q in enumerate(query_performance[:10], 1):
        print(f"  {i}. Query {q['query_idx']:3d}: "
              f"{q['hit_rate']:5.1f}% hit rate, "
              f"conf={q['avg_confidence']:.3f}, "
              f"IoU={q['avg_iou']:.3f}, "
              f"best_for={q['best_for_image_count']:3d} images")

    print(f"\nQuery Specialization Analysis:")
    # Find queries that are best for many images
    specialist_queries = [q for q in query_performance if q['best_for_image_count'] > 5]
    print(f"  Specialist queries (best for >5 images): {len(specialist_queries)}")
    for q in sorted(specialist_queries, key=lambda x: x['best_for_image_count'], reverse=True)[:5]:
        print(f"    Query {q['query_idx']:3d}: best for {q['best_for_image_count']:3d} images, "
              f"hit_rate={q['hit_rate']:.1f}%")

    # Save results
    results = {
        'summary': {
            'num_images': num_images,
            'query81_hit_rate': query81_hit_rate,
            'oracle_hit_rate': oracle_hit_rate,
            'potential_improvement': oracle_hit_rate - query81_hit_rate
        },
        'query_performance': query_performance,
        'per_image_results': per_image_results
    }

    output_path = os.path.join(output_dir, 'query_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    config = load_config('config.yaml')
    results = analyze_all_queries(config)
