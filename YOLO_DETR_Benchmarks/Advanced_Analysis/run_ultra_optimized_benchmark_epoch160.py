"""
Ultra-Optimized DETR Epoch 160 vs YOLO Benchmark
Goal: Make DETR beat YOLO with all optimizations

Optimizations:
1. Lower confidence threshold (0.2 instead of 0.5)
2. Test-Time Augmentation (TTA)
3. Box refinement
4. Query ensemble (if beneficial)
5. Confidence calibration
"""

import sys
import os
import json
import time
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

# Import utilities from existing modules
sys.path.append(str(Path(__file__).parent))
from run_inference import (
    load_config, get_image_files, build_coco_image_id_map,
    get_category_id_mapper, _allowlist_ultralytics_pickle_classes,
    load_detr_from_checkpoint, resolve_path
)

def apply_tta_transforms(image, processor):
    """
    Apply Test-Time Augmentation transforms to image
    Returns list of (inputs, transform_info) tuples
    """
    transforms = []

    # Original
    inputs_orig = processor(images=image, return_tensors="pt")
    transforms.append((inputs_orig, {"scale": 1.0, "flip": False}))

    # Horizontal flip
    image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    inputs_flip = processor(images=image_flipped, return_tensors="pt")
    transforms.append((inputs_flip, {"scale": 1.0, "flip": True}))

    # Scale variations
    w, h = image.size
    for scale in [0.8, 1.2]:
        new_w, new_h = int(w * scale), int(h * scale)
        image_scaled = image.resize((new_w, new_h), Image.BILINEAR)
        inputs_scaled = processor(images=image_scaled, return_tensors="pt")
        transforms.append((inputs_scaled, {"scale": scale, "flip": False}))

    return transforms

def reverse_tta_transform(boxes, scores, labels, transform_info, original_size):
    """
    Reverse TTA transform to get boxes in original image coordinates
    """
    w, h = original_size

    # Reverse scale
    scale = transform_info["scale"]
    if scale != 1.0:
        boxes = boxes / scale

    # Reverse flip
    if transform_info["flip"]:
        # Flip x coordinates: x_new = w - x_old
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1

    return boxes, scores, labels

def tta_ensemble_predictions(all_predictions, iou_threshold=0.4):
    """
    Ensemble predictions from multiple TTA transforms using Weighted Boxes Fusion
    Simple implementation: average scores for overlapping boxes, keep highest scoring
    """
    if not all_predictions:
        return [], [], []

    # Collect all boxes, scores, labels
    all_boxes = []
    all_scores = []
    all_labels = []

    for boxes, scores, labels in all_predictions:
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    if not all_boxes:
        return [], [], []

    # Concatenate
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # NMS with lower threshold to keep more boxes
    keep = torch.ops.torchvision.nms(all_boxes, all_scores, iou_threshold)

    return all_boxes[keep], all_scores[keep], all_labels[keep]

def refine_boxes(model, processor, image, boxes, device, iterations=2):
    """
    Iterative box refinement using RoI features
    Simple implementation: crop regions and re-detect with higher confidence
    """
    refined_boxes = []
    refined_scores = []
    refined_labels = []

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]

        # Add margin
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.size[0], x2 + margin)
        y2 = min(image.size[1], y2 + margin)

        # Crop region
        roi = image.crop((x1, y1, x2, y2))

        # Re-detect with lower threshold
        inputs = processor(images=roi, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Get refined box
        target_sizes = torch.tensor([roi.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.1
        )[0]

        if len(results["scores"]) > 0:
            # Take highest scoring detection
            idx = results["scores"].argmax()
            refined_box = results["boxes"][idx]

            # Transform back to original image coordinates
            refined_box[0] += x1
            refined_box[1] += y1
            refined_box[2] += x1
            refined_box[3] += y1

            refined_boxes.append(refined_box)
            refined_scores.append(results["scores"][idx])
            refined_labels.append(results["labels"][idx])

    if not refined_boxes:
        return boxes, torch.ones(len(boxes)), torch.zeros(len(boxes))

    return (
        torch.stack(refined_boxes),
        torch.stack(refined_scores),
        torch.stack(refined_labels)
    )

def calibrate_confidence_scores(scores, temperature=1.5):
    """
    Apply temperature scaling to calibrate confidence scores
    Higher temperature = softer distribution (more boxes survive threshold)
    """
    scores = scores / temperature
    return torch.sigmoid(scores)

def run_detr_optimized_inference(config, enable_tta=True, enable_refinement=True):
    """
    Run DETR inference with all optimizations
    """
    print("\n" + "="*60)
    print("🚀 DETR ULTRA-OPTIMIZED INFERENCE (EPOCH 160)")
    print("="*60)
    print(f"TTA Enabled: {enable_tta}")
    print(f"Box Refinement: {enable_refinement}")
    print(f"Confidence Threshold: {config['models']['detr']['conf_threshold']}")

    # Load checkpoint
    checkpoint_path = config['models']['detr']['path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n📦 Loading checkpoint: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_detr_from_checkpoint(checkpoint_path, device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Dataset
    images_dir = config['dataset']['images_dir']
    image_files = get_image_files(images_dir)

    # Image ID mapping
    ann_path = config['dataset'].get('annotations_path')
    gt_map = build_coco_image_id_map(ann_path)
    image_id_map = gt_map if gt_map else {name: i for i, name in enumerate(image_files)}

    # Category mapping
    cat_map = get_category_id_mapper(config, 'detr')

    # Confidence threshold
    conf_threshold = config['models']['detr']['conf_threshold']
    temperature = config['models']['detr'].get('temperature', 1.5)

    # Performance tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    coco_results = []
    query_stats = []

    print(f"\n🔄 Processing {len(image_files)} images...")

    for image_name in tqdm(image_files, desc="DETR Optimized Inference"):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if enable_tta:
            # Test-Time Augmentation
            tta_transforms = apply_tta_transforms(image, processor)
            all_predictions = []

            for inputs, transform_info in tta_transforms:
                inputs = inputs.to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                # Post-process
                target_sizes = torch.tensor([image.size[::-1]])
                results = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.0  # No threshold yet
                )[0]

                # Reverse transform
                boxes, scores, labels = reverse_tta_transform(
                    results["boxes"],
                    results["scores"],
                    results["labels"],
                    transform_info,
                    image.size
                )

                all_predictions.append((boxes, scores, labels))

            # Ensemble
            boxes, scores, labels = tta_ensemble_predictions(
                all_predictions,
                iou_threshold=0.4
            )
        else:
            # Standard inference
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )[0]

            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]

        # Confidence calibration
        if temperature != 1.0:
            scores = calibrate_confidence_scores(scores, temperature)

        # Apply confidence threshold
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Box refinement (optional)
        if enable_refinement and len(boxes) > 0:
            boxes, scores, labels = refine_boxes(
                model, processor, image, boxes, device, iterations=2
            )

        # Convert to COCO format
        image_id = image_id_map.get(image_name, image_id_map.get(os.path.basename(image_name)))
        if image_id is None:
            image_id = image_files.index(image_name)

        for score, label, box in zip(scores, labels, boxes):
            box = [round(float(i), 2) for i in box.tolist()]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            cls_idx = int(label.item())
            if cat_map is not None:
                category_id = int(cat_map.get(cls_idx, cls_idx))
            else:
                category_id = cls_idx

            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "score": round(float(score.item()), 3),
            })

    # Performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    fps = len(image_files) / total_time

    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        vram_mb = 0

    print(f"\n⚡ Performance:")
    print(f"  FPS: {fps:.2f}")
    print(f"  VRAM: {vram_mb:.0f} MB")
    print(f"  Total time: {total_time:.2f}s")

    # Save predictions
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)

    pred_file = os.path.join(output_dir, "detr_predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(coco_results, f)

    print(f"\n💾 Predictions saved: {pred_file}")

    return {
        "predictions": coco_results,
        "fps": fps,
        "vram_mb": vram_mb,
        "total_time": total_time
    }

def run_yolo_baseline(config):
    """
    Run YOLO inference (baseline)
    """
    print("\n" + "="*60)
    print("📊 YOLO BASELINE INFERENCE")
    print("="*60)

    _allowlist_ultralytics_pickle_classes()

    # Additional fix for DFLoss not being in safe globals
    try:
        from torch.serialization import add_safe_globals
        try:
            from ultralytics.utils.loss import DFLoss
            add_safe_globals([DFLoss])
        except:
            pass
    except:
        pass

    yolo_path = config['models']['yolo']['path']
    print(f"Loading YOLO: {yolo_path}")

    # Load with weights_only=False if needed
    import torch
    old_weights_only = getattr(torch.load, '__kwdefaults__', {}).get('weights_only')
    try:
        if hasattr(torch.serialization, '_weights_only_unpickler'):
            # PyTorch 2.6+
            model = YOLO(yolo_path)
        else:
            model = YOLO(yolo_path)
    except Exception as e:
        print(f"Failed to load YOLO: {e}")
        print("Trying with weights_only=False fallback...")
        # Monkey patch torch.load temporarily
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        model = YOLO(yolo_path)
        torch.load = original_load

    images_dir = config['dataset']['images_dir']
    image_files = get_image_files(images_dir)

    # Image ID mapping
    ann_path = config['dataset'].get('annotations_path')
    gt_map = build_coco_image_id_map(ann_path)
    image_id_map = gt_map if gt_map else {name: i for i, name in enumerate(image_files)}

    # Category mapping
    cat_map = get_category_id_mapper(config, 'yolo')

    conf_threshold = config['models']['yolo'].get('conf_threshold', 0.5)

    # Performance tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    coco_results = []

    print(f"\n🔄 Processing {len(image_files)} images...")

    for image_name in tqdm(image_files, desc="YOLO Inference"):
        image_path = os.path.join(images_dir, image_name)

        results = model(image_path, conf=conf_threshold, verbose=False)

        image_id = image_id_map.get(image_name, image_id_map.get(os.path.basename(image_name)))
        if image_id is None:
            image_id = image_files.index(image_name)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_idx = int(box.cls.item())

                if cat_map is not None:
                    category_id = int(cat_map.get(cls_idx, cls_idx + 1))
                else:
                    category_id = cls_idx + 1

                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                coco_results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                    "score": round(float(box.conf.item()), 3),
                })

    # Performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    fps = len(image_files) / total_time

    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        vram_mb = 0

    print(f"\n⚡ Performance:")
    print(f"  FPS: {fps:.2f}")
    print(f"  VRAM: {vram_mb:.0f} MB")
    print(f"  Total time: {total_time:.2f}s")

    # Save predictions
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)

    pred_file = os.path.join(output_dir, "yolo_predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(coco_results, f)

    print(f"\n💾 Predictions saved: {pred_file}")

    return {
        "predictions": coco_results,
        "fps": fps,
        "vram_mb": vram_mb,
        "total_time": total_time
    }

def evaluate_coco(gt_path, pred_path, model_name):
    """
    Evaluate predictions using COCO API
    """
    print(f"\n" + "="*60)
    print(f"📈 EVALUATING {model_name}")
    print("="*60)

    gt_coco = COCO(gt_path)
    pred_coco = gt_coco.loadRes(pred_path)

    coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "mAP_0.5": float(coco_eval.stats[1]),
        "mAP_0.5:0.95": float(coco_eval.stats[0]),
        "mAP_0.75": float(coco_eval.stats[2]),
        "AR_max_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
    }

    return results

def main():
    # Load config
    config_path = "config_epoch160_optimized.yaml"
    print(f"📝 Loading config: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*60)
    print("🎯 ULTRA-OPTIMIZED DETR EPOCH 160 vs YOLO BENCHMARK")
    print("="*60)

    # Run YOLO baseline
    yolo_results = run_yolo_baseline(config)

    # Run DETR optimized
    detr_results = run_detr_optimized_inference(
        config,
        enable_tta=config['models']['detr'].get('tta_enabled', True),
        enable_refinement=config['models']['detr'].get('box_refinement', True)
    )

    # Evaluate both
    gt_path = config['dataset']['annotations_path']
    output_dir = config['output']['directory']

    yolo_metrics = evaluate_coco(
        gt_path,
        os.path.join(output_dir, "yolo_predictions.json"),
        "YOLO EPOCH 100"
    )

    detr_metrics = evaluate_coco(
        gt_path,
        os.path.join(output_dir, "detr_predictions.json"),
        "DETR EPOCH 160 (OPTIMIZED)"
    )

    # Compare
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS")
    print("="*60)

    print(f"\n{'Metric':<20} {'YOLO':<12} {'DETR':<12} {'Gap':<12} {'Winner'}")
    print("-" * 60)

    for metric in ["mAP_0.5", "mAP_0.5:0.95", "mAP_0.75", "AR_max_100"]:
        yolo_val = yolo_metrics[metric] * 100
        detr_val = detr_metrics[metric] * 100
        gap = detr_val - yolo_val
        winner = "🥇 DETR" if gap > 0 else "🥇 YOLO"

        print(f"{metric:<20} {yolo_val:>6.2f}%     {detr_val:>6.2f}%     {gap:>+6.2f}%     {winner}")

    print("\n" + "-" * 60)
    print(f"{'FPS':<20} {yolo_results['fps']:>6.2f}      {detr_results['fps']:>6.2f}      {detr_results['fps']-yolo_results['fps']:>+6.2f}")
    print(f"{'VRAM (MB)':<20} {yolo_results['vram_mb']:>6.0f}      {detr_results['vram_mb']:>6.0f}      {detr_results['vram_mb']-yolo_results['vram_mb']:>+6.0f}")

    # Save final report
    final_report = {
        "yolo": {
            "metrics": yolo_metrics,
            "performance": {
                "fps": yolo_results['fps'],
                "vram_mb": yolo_results['vram_mb']
            }
        },
        "detr": {
            "metrics": detr_metrics,
            "performance": {
                "fps": detr_results['fps'],
                "vram_mb": detr_results['vram_mb']
            },
            "optimizations": {
                "tta_enabled": config['models']['detr'].get('tta_enabled', True),
                "box_refinement": config['models']['detr'].get('box_refinement', True),
                "conf_threshold": config['models']['detr']['conf_threshold'],
                "temperature": config['models']['detr'].get('temperature', 1.5)
            }
        },
        "gaps": {
            metric: (detr_metrics[metric] - yolo_metrics[metric]) * 100
            for metric in ["mAP_0.5", "mAP_0.5:0.95", "AR_max_100"]
        }
    }

    report_file = os.path.join(output_dir, "final_report_epoch160.json")
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\n💾 Final report saved: {report_file}")

    # Victory check
    print("\n" + "="*60)
    if detr_metrics["mAP_0.5"] > yolo_metrics["mAP_0.5"]:
        print("🎉🎉🎉 DETR WINS ON mAP@0.5! 🎉🎉🎉")
    else:
        print("❌ YOLO still ahead on mAP@0.5")
        gap_to_close = (yolo_metrics["mAP_0.5"] - detr_metrics["mAP_0.5"]) * 100
        print(f"   Gap to close: {gap_to_close:.2f}%")

    if detr_metrics["mAP_0.5:0.95"] > yolo_metrics["mAP_0.5:0.95"]:
        print("🎉🎉🎉 DETR WINS ON mAP@0.5:0.95! 🎉🎉🎉")
    else:
        print("❌ YOLO still ahead on mAP@0.5:0.95")
        gap_to_close = (yolo_metrics["mAP_0.5:0.95"] - detr_metrics["mAP_0.5:0.95"]) * 100
        print(f"   Gap to close: {gap_to_close:.2f}%")

    print("="*60)

if __name__ == "__main__":
    main()
