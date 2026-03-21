#!/usr/bin/env python3
"""
Evaluate all ablation variants with COCO mAP + TP/FP/F1 metrics.

Usage:
  python evaluate_ablation.py \
    --variants-dir ~/DETR/ablation_best \
    --test-annotations ~/DETR/ablation/v2/annotations_test.json \
    --images-dir $TMPDIR/images \
    --output results.json \
    --coco-eval

  # With V1 from existing checkpoint:
  python evaluate_ablation.py \
    --variants-dir ~/DETR/ablation_best \
    --v1-checkpoint ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth \
    --test-annotations ~/DETR/ablation/v2/annotations_test.json \
    --images-dir $TMPDIR/images \
    --cross-annotations ~/DETR/external_benchmark/annotations.json \
    --cross-images-dir ~/DETR/external_benchmark/images \
    --output results.json \
    --coco-eval
"""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
)

VARIANTS = {
    "v1": "FULL",
    "v2": "NO_FOURIER",
    "v3": "NO_EL2N",
    "v4": "QUALITY_ONLY",
    "v5": "RANDOM",
    "v6": "DIVERSITY_ONLY",
    "v7": "EL2N_ONLY",
}

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


# ============================================================================
# TP/FP/FN evaluation (simple, always available)
# ============================================================================

def compute_iou(box1, box2):
    """IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def evaluate_tp_fp_fn(model, processor, annotations_path, images_dir,
                      conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD,
                      device="cuda"):
    """Run inference and compute TP/FP/FN/Precision/Recall/F1."""
    with open(annotations_path, 'r') as f:
        coco = json.load(f)

    gt_by_image = defaultdict(list)
    for ann in coco['annotations']:
        gt_by_image[ann['image_id']].append(ann['bbox'])

    images_dir = Path(images_dir)
    total_tp, total_fp, total_fn = 0, 0, 0

    for img_info in tqdm(coco['images'], desc="  TP/FP/FN eval"):
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]

        pred_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]]
                      for b in results['boxes'].cpu().tolist()]

        gt_boxes = gt_by_image.get(img_info['id'], [])
        gt_matched = [False] * len(gt_boxes)

        tp, fp = 0, 0
        for pb in pred_boxes:
            best_iou, best_j = 0, -1
            for j, gb in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0:
                tp += 1
                gt_matched[best_j] = True
            else:
                fp += 1

        fn = sum(1 for m in gt_matched if not m)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return {
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "Precision": round(prec * 100, 1),
        "Recall": round(rec * 100, 1),
        "F1": round(f1 * 100, 1),
        "images_evaluated": len(coco['images']),
    }


# ============================================================================
# COCO mAP evaluation (pycocotools)
# ============================================================================

def evaluate_coco_map(model, processor, annotations_path, images_dir,
                      conf_threshold=0.0, device="cuda"):
    """
    Compute COCO mAP@0.5 and mAP@0.5:0.95 using pycocotools.
    Generates predictions in COCO format, then runs COCOeval.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(annotations_path)
    images_dir = Path(images_dir)

    predictions = []
    for img_id in tqdm(coco_gt.getImgIds(), desc="  COCO mAP eval"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]

        for box, score, label in zip(
                results['boxes'].cpu().tolist(),
                results['scores'].cpu().tolist(),
                results['labels'].cpu().tolist()):
            x1, y1, x2, y2 = box
            predictions.append({
                'image_id': img_id,
                'category_id': label,
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': score,
            })

    if not predictions:
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}

    # Write predictions to temp file for COCOeval
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        pred_path = f.name

    try:
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP@0.5:0.95": round(coco_eval.stats[0] * 100, 1),
            "mAP@0.5": round(coco_eval.stats[1] * 100, 1),
            "mAP@0.75": round(coco_eval.stats[2] * 100, 1),
            "AR@100": round(coco_eval.stats[8] * 100, 1),
        }
    finally:
        os.unlink(pred_path)


# ============================================================================
# Model loading — unified .pth format for all variants
# ============================================================================

def _build_model(device="cuda"):
    """Create a fresh DETR model with single-class config."""
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = 1
    config.id2label = {0: "tooltip"}
    config.label2id = {"tooltip": 0}
    config.num_queries = 100

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", config=config, ignore_mismatched_sizes=True)
    return model, processor


def load_model_from_pth(pth_path, device="cuda"):
    """
    Load model from a .pth file. Handles both formats:
      - Pure state_dict (from best_model.pth saved by ablation trainer)
      - Full checkpoint dict with 'model_state_dict' key (from checkpoint_epoch_*.pth)
    Strips 'module.' prefix from DDP-saved weights automatically.
    """
    model, processor = _build_model(device)

    ckpt = torch.load(pth_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    clean_sd = {(k[7:] if k.startswith('module.') else k): v
                for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model = model.to(device)
    model.eval()
    return model, processor


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Evaluate ablation variants")
    ap.add_argument("--variants-dir", type=str, required=True,
                    help="Dir with v1/, v2/, ... subdirs (best models)")
    ap.add_argument("--v1-checkpoint", type=str, default=None,
                    help="Path to V1 .pth checkpoint (if not in variants-dir)")
    ap.add_argument("--test-annotations", type=str, required=True,
                    help="Same-distribution test annotations JSON")
    ap.add_argument("--images-dir", type=str, required=True)
    ap.add_argument("--cross-annotations", type=str, default=None,
                    help="Cross-dataset annotations JSON")
    ap.add_argument("--cross-images-dir", type=str, default=None)
    ap.add_argument("--coco-eval", action="store_true",
                    help="Compute COCO mAP using pycocotools")
    ap.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    ap.add_argument("--output", type=str, default="ablation_results.json")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--variants", type=str, default=None,
                    help="Comma-separated list of variants to evaluate (default: all)")
    args = ap.parse_args()

    variants_dir = Path(args.variants_dir)
    results = {}

    variant_list = args.variants.split(',') if args.variants else list(VARIANTS.keys())

    for vid in variant_list:
        vid = vid.strip()
        vname = VARIANTS.get(vid, vid)

        # Load model — all variants use .pth format
        if vid == "v1" and args.v1_checkpoint:
            pth_path = args.v1_checkpoint
        else:
            pth_path = str(variants_dir / vid / "best_model.pth")

        if not os.path.isfile(pth_path):
            print(f"Skipping {vid} ({vname}): {pth_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {vid} ({vname}) from: {pth_path}")
        print(f"{'='*60}")
        model, processor = load_model_from_pth(pth_path, args.device)

        # Same-distribution TP/FP/FN
        print("  Same-distribution (TP/FP/FN)...")
        same_dist = evaluate_tp_fp_fn(model, processor, args.test_annotations,
                                       args.images_dir, args.conf_threshold,
                                       args.iou_threshold, args.device)
        results[vid] = {"name": vname, "same_distribution": same_dist}

        # Same-distribution COCO mAP
        if args.coco_eval:
            print("  Same-distribution (COCO mAP)...")
            coco_metrics = evaluate_coco_map(model, processor, args.test_annotations,
                                              args.images_dir, device=args.device)
            results[vid]["same_distribution"].update(coco_metrics)

        # Cross-dataset
        if args.cross_annotations and args.cross_images_dir:
            print("  Cross-dataset (TP/FP/FN)...")
            cross = evaluate_tp_fp_fn(model, processor, args.cross_annotations,
                                       args.cross_images_dir, args.conf_threshold,
                                       args.iou_threshold, args.device)
            results[vid]["cross_dataset"] = cross

            if args.coco_eval:
                print("  Cross-dataset (COCO mAP)...")
                cross_coco = evaluate_coco_map(model, processor, args.cross_annotations,
                                                args.cross_images_dir, device=args.device)
                results[vid]["cross_dataset"].update(cross_coco)

        del model
        torch.cuda.empty_cache()

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print tables
    has_coco = any("mAP@0.5" in results.get(v, {}).get("same_distribution", {})
                   for v in variant_list)

    print(f"\n{'='*90}")
    print("ABLATION RESULTS — Same Distribution")
    print(f"{'='*90}")
    if has_coco:
        print(f"{'Variant':<18} {'mAP@.5':>7} {'mAP@.5:.95':>10} {'TP':>6} {'FP':>6} {'FN':>6} "
              f"{'Prec%':>7} {'Rec%':>7} {'F1%':>7}")
    else:
        print(f"{'Variant':<18} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec%':>7} {'Rec%':>7} {'F1%':>7}")
    print("-" * 90)

    for vid in variant_list:
        if vid not in results:
            continue
        r = results[vid]["same_distribution"]
        vn = VARIANTS.get(vid, vid)
        if has_coco:
            print(f"{vn:<18} {r.get('mAP@0.5', 0):>6.1f}% {r.get('mAP@0.5:0.95', 0):>9.1f}% "
                  f"{r['TP']:>6} {r['FP']:>6} {r['FN']:>6} "
                  f"{r['Precision']:>6.1f}% {r['Recall']:>6.1f}% {r['F1']:>6.1f}%")
        else:
            print(f"{vn:<18} {r['TP']:>6} {r['FP']:>6} {r['FN']:>6} "
                  f"{r['Precision']:>6.1f}% {r['Recall']:>6.1f}% {r['F1']:>6.1f}%")

    if any("cross_dataset" in results.get(v, {}) for v in variant_list):
        print(f"\n{'='*90}")
        print("ABLATION RESULTS — Cross-Dataset")
        print(f"{'='*90}")
        if has_coco:
            print(f"{'Variant':<18} {'mAP@.5':>7} {'mAP@.5:.95':>10} {'TP':>6} {'FP':>6} {'FN':>6} "
                  f"{'Prec%':>7} {'Rec%':>7} {'F1%':>7}")
        else:
            print(f"{'Variant':<18} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec%':>7} {'Rec%':>7} {'F1%':>7}")
        print("-" * 90)
        for vid in variant_list:
            if vid not in results or "cross_dataset" not in results[vid]:
                continue
            r = results[vid]["cross_dataset"]
            vn = VARIANTS.get(vid, vid)
            if has_coco:
                print(f"{vn:<18} {r.get('mAP@0.5', 0):>6.1f}% {r.get('mAP@0.5:0.95', 0):>9.1f}% "
                      f"{r['TP']:>6} {r['FP']:>6} {r['FN']:>6} "
                      f"{r['Precision']:>6.1f}% {r['Recall']:>6.1f}% {r['F1']:>6.1f}%")
            else:
                print(f"{vn:<18} {r['TP']:>6} {r['FP']:>6} {r['FN']:>6} "
                      f"{r['Precision']:>6.1f}% {r['Recall']:>6.1f}% {r['F1']:>6.1f}%")


if __name__ == "__main__":
    main()
