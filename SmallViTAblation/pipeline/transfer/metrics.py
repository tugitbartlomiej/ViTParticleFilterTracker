#!/usr/bin/env python3
"""Metryki w DOKLADNIE formacie metrics.json (agents/README.md 4.5).

Wspoldzieli format z Agentem 3 oraz implementacje COCO AP i licznikow
z pipeline/eval/common.py. Przelicza metryki z ZAPISANYCH predykcji COCO
(bez ponownej inferencji).

mAP/AR liczy pycocotools (LAZY). Prog operacyjny do TP/FP/F1 wybierany jest
na walidacji (maksymalizuje F1), nigdy na tescie — zgodne z protocol.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from ..eval import common as eval_common
else:  # direct CLI scripts import this module as `metrics`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline.eval import common as eval_common

# Metryki wymagane w kazdym bloku wynikowym (README 4.5 / validate_contracts.py)
METRIC_FIELDS = [
    "mAP_0.5", "mAP_0.5:0.95", "mAP_0.75", "AR@100",
    "TP", "FP", "FN", "precision", "recall", "f1",
    "n_images", "n_annotations",
]


# --------------------------------------------------------------------------- #
# dopasowanie predykcji do GT (greedy IoU) — czysty numpy
# --------------------------------------------------------------------------- #
def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU miedzy zestawem ramek [N,4] i [M,4] w formacie x,y,w,h."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    px1, py1 = pred[:, 0], pred[:, 1]
    px2, py2 = pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    ix1 = np.maximum(px1[:, None], gx1[None, :])
    iy1 = np.maximum(py1[:, None], gy1[None, :])
    ix2 = np.minimum(px2[:, None], gx2[None, :])
    iy2 = np.minimum(py2[:, None], gy2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_p = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    area_g = np.clip(gx2 - gx1, 0, None) * np.clip(gy2 - gy1, 0, None)
    union = area_p[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_at_threshold(
    preds_by_image: dict[int, tuple[np.ndarray, np.ndarray]],
    gts_by_image: dict[int, np.ndarray],
    conf_thr: float,
    iou_thr: float,
) -> tuple[int, int, int]:
    """Zwraca (TP, FP, FN) przy danym progu ufności i IoU dopasowania."""
    tp = fp = fn = 0
    for img_id, gt in gts_by_image.items():
        fn += len(gt)
    for img_id, (boxes, scores) in preds_by_image.items():
        if len(boxes) == 0:
            continue
        keep = scores >= conf_thr
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores, kind="stable")
        boxes, scores = boxes[order], scores[order]
        gt = gts_by_image.get(img_id, np.zeros((0, 4)))
        matched = np.zeros(len(gt), dtype=bool)
        ious = _iou_matrix(boxes, gt)
        for i in range(len(boxes)):
            if len(gt) == 0:
                fp += 1
                continue
            available = np.where(~matched, ious[i], -1.0)
            j = int(np.argmax(available)) if available.size else -1
            if j >= 0 and available[j] >= iou_thr:
                matched[j] = True
                tp += 1
                fn -= 1
            else:
                fp += 1
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def select_operating_threshold(
    preds_by_image: dict[int, tuple[np.ndarray, np.ndarray]],
    gts_by_image: dict[int, np.ndarray],
    iou_thr: float,
    candidates: list[float] | None = None,
) -> tuple[float, float]:
    """Prog maksymalizujacy F1. Zwraca (prog, f1)."""
    if candidates is None:
        candidates = sorted({0.0, 1.0} | {
            float(score) for _, scores in preds_by_image.values() for score in scores})
    best_thr, best_f1 = candidates[0], -1.0
    for thr in candidates:
        tp, fp, fn = match_at_threshold(preds_by_image, gts_by_image, thr, iou_thr)
        _, _, f1 = _prf(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# --------------------------------------------------------------------------- #
# mAP przez pycocotools (LAZY)
# --------------------------------------------------------------------------- #
def _coco_map(gt_coco: dict, preds: list[dict]) -> dict[str, float]:
    """mAP@0.5, mAP@0.5:0.95, mAP@0.75, AR@100 przez pycocotools."""
    return eval_common.compute_map_metrics(gt_coco, preds)


# --------------------------------------------------------------------------- #
# skladowe bloku wynikow
# --------------------------------------------------------------------------- #
def parse_gt(gt_coco: dict) -> tuple[dict[tuple[int, int], np.ndarray], int, int, set[int]]:
    """Group by (image_id, category_id) so matching/NMS cannot merge classes."""
    gts: dict[tuple[int, int], list] = {}
    n_ann = 0
    for a in gt_coco.get("annotations", []):
        gts.setdefault((int(a["image_id"]), int(a["category_id"])), []).append(a["bbox"])
        n_ann += 1
    gts_arr = {k: np.asarray(v, dtype=np.float64) for k, v in gts.items()}
    img_ids = {im["id"] for im in gt_coco.get("images", [])}
    # Keep background image/category pairs available for false positives.
    for iid in img_ids:
        for category in gt_coco["categories"]:
            gts_arr.setdefault((int(iid), int(category["id"])), np.zeros((0, 4)))
    return gts_arr, len(img_ids), n_ann, img_ids


def parse_preds(preds: list[dict]) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    by_img: dict[tuple[int, int], tuple[list, list]] = {}
    for p in preds:
        key = (int(p["image_id"]), int(p["category_id"]))
        by_img.setdefault(key, ([], []))[0].append(p["bbox"])
        by_img[key][1].append(p.get("score", p.get("confidence", 0.0)))
    return {k: (np.asarray(v[0], dtype=np.float64), np.asarray(v[1], dtype=np.float64))
            for k, v in by_img.items()}


def compute_block(
    gt_coco: dict,
    preds: list[dict],
    operating_threshold: float,
    eval_iou: float,
) -> dict[str, Any]:
    """Pojedynczy blok wynikow (val/test/...) w formacie 4.5."""
    maps = _coco_map(gt_coco, preds)
    counts = eval_common.counting_metrics(eval_common.build_gt_index(gt_coco), preds,
                                          operating_threshold, eval_iou)
    block = {
        **maps,
        **counts,
        "n_images": len(gt_coco["images"]),
    }
    missing = set(METRIC_FIELDS) - block.keys()
    if missing:
        raise ValueError(f"block metryk bez pol: {sorted(missing)}")
    return block


def assemble_metrics(
    *,
    run_id: str,
    checkpoint_rule: str,
    operating_threshold: float,
    results: dict[str, dict],
) -> dict:
    """Składa caly metrics.json (README 4.5)."""
    return {
        "run_id": run_id,
        "checkpoint_rule": checkpoint_rule,
        "results": results,
        "operating_threshold": float(operating_threshold),
        "threshold_selected_on": "val",
    }


# --------------------------------------------------------------------------- #
# opcjonalnie: delegacja do Agenta 3 (jesli istnieje jego evaluate.py)
# --------------------------------------------------------------------------- #
def try_import_agent3_evaluate(repo_root: Path):
    """Zwraca modul pipeline/eval/evaluate.py Agenta 3 albo None. NIE edytujemy go."""
    import importlib.util
    cand = Path(repo_root) / "pipeline" / "eval" / "evaluate.py"
    if not cand.exists():
        return None
    spec = importlib.util.spec_from_file_location("agent3_evaluate", cand)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None
