"""Pure-function module for detection metrics and COCO annotation parsing.

All functions are stateless and side-effect-free. Bounding boxes use
the ``[x1, y1, x2, y2]`` format throughout.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union for two ``[x1, y1, x2, y2]`` boxes."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def compute_metrics(
    predictions: List[dict],
    ground_truth: List[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compare predictions against ground truth using greedy IoU matching.

    Parameters
    ----------
    predictions : list of dict
        Each with ``bbox`` key (``[x1, y1, x2, y2]``).
    ground_truth : list of dict
        Each with ``bbox`` key (``[x1, y1, x2, y2]``).
    iou_threshold : float
        Minimum IoU to count a detection as a true positive.

    Returns
    -------
    dict
        ``{tp, fp, fn, precision, recall, f1}``.
    """
    matched_gt: set = set()
    tp = 0
    fp = 0

    sorted_preds = sorted(
        predictions,
        key=lambda d: d.get("score", 0),
        reverse=True,
    )

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def load_coco_annotations(json_path: str) -> dict:
    """Parse a COCO annotations JSON file.

    Returns
    -------
    dict
        ``{
            "images": {filename: {id, width, height, ...}},
            "annotations": {filename: [{bbox, category_id, ...}]},
            "categories": {id: name},
        }``

        Bounding boxes are converted from COCO ``[x, y, w, h]`` to
        ``[x1, y1, x2, y2]`` format.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file: Dict[int, str] = {}
    images_info: Dict[str, dict] = {}
    for img in coco.get("images", []):
        fname = img.get("file_name", "")
        id_to_file[img["id"]] = fname
        images_info[fname] = img

    categories: Dict[int, str] = {}
    for cat in coco.get("categories", []):
        categories[cat["id"]] = cat.get("name", f"cat_{cat['id']}")

    annotations: Dict[str, List[dict]] = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        fname = id_to_file.get(img_id, "")
        if not fname:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            bbox_xyxy = [x, y, x + w, y + h]
        else:
            bbox_xyxy = bbox
        entry = {
            "bbox": bbox_xyxy,
            "category_id": ann.get("category_id", 0),
            "id": ann.get("id"),
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd", 0),
        }
        annotations.setdefault(fname, []).append(entry)

    return {
        "images": images_info,
        "annotations": annotations,
        "categories": categories,
    }


def batch_evaluate(
    inference_fn: Callable[[str], List[dict]],
    image_paths: List[str],
    annotations: Dict[str, List[dict]],
    iou_threshold: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Run inference on all images and compute aggregate metrics.

    Parameters
    ----------
    inference_fn : callable
        ``(image_path) -> list[dict]`` -- runs model prediction on one image.
    image_paths : list of str
        Paths to image files.
    annotations : dict
        Filename -> list of GT dicts (``[{bbox, ...}]``).
    iou_threshold : float
        IoU threshold for TP/FP classification.
    progress_callback : callable, optional
        Called with ``(current_index, total, filename)`` after each image.

    Returns
    -------
    dict
        ``{total_tp, total_fp, total_fn, precision, recall, f1,
          avg_confidence, images_tested, per_image: [...]}``.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_scores: List[float] = []
    per_image: List[dict] = []
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths):
        fname = Path(img_path).name

        preds = inference_fn(img_path)
        gt = annotations.get(fname, [])
        metrics = compute_metrics(preds, gt, iou_threshold)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        for p in preds:
            all_scores.append(p.get("score", 0))

        per_image.append({
            "filename": fname,
            "predictions": preds,
            "gt_count": len(gt),
            "pred_count": len(preds),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        })

        if progress_callback:
            progress_callback(idx + 1, total, fname)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_conf = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_confidence": round(avg_conf, 4),
        "images_tested": total,
        "per_image": per_image,
    }
