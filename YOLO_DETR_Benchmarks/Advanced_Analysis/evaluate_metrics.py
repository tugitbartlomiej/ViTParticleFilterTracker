import yaml
import os
import json
import time
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_standard_metrics(gt_coco, pred_coco, img_ids=None):
    """Oblicza standardowe metryki COCO (mAP, Precision, Recall).
    Jeśli podano img_ids, ewaluacja będzie ograniczona do tych obrazów (np. tylko walidacja).
    """
    coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
    if img_ids:
        coco_eval.params.imgIds = list(sorted(set(img_ids)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = {
        "mAP_0.5:0.95": coco_eval.stats[0],
        "mAP_0.5": coco_eval.stats[1],
        "mAP_0.75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "AR_max_1": coco_eval.stats[6],
        "AR_max_10": coco_eval.stats[7],
        "AR_max_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }
    return stats

def calculate_temporal_stability(pred_results):
    """
    Oblicza metryki stabilności czasowej.
    - Temporal IoU: Średnie IoU bounding boxa tego samego obiektu między kolejnymi klatkami.
    - Detection Flicker: Liczba klatek, w których obiekt został zgubiony, mimo że był w klatce poprzedniej i następnej.
    """
    preds_by_image = {}
    for p in pred_results:
        img_id = p['image_id']
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append(p)

    sorted_img_ids = sorted(preds_by_image.keys())
    
    ious = []
    flicker_count = 0
    
    for i in range(1, len(sorted_img_ids) - 1):
        prev_id, current_id, next_id = sorted_img_ids[i-1], sorted_img_ids[i], sorted_img_ids[i+1]
        
        prev_preds = preds_by_image.get(prev_id, [])
        current_preds = preds_by_image.get(current_id, [])
        next_preds = preds_by_image.get(next_id, [])

        # Flicker
        if prev_preds and next_preds and not current_preds:
            flicker_count += 1
            
        # Temporal IoU (zakładając jeden główny obiekt na obrazie)
        if current_preds and next_preds:
            box_current = current_preds[0]['bbox']
            box_next = next_preds[0]['bbox']
            
            xA = max(box_current[0], box_next[0])
            yA = max(box_current[1], box_next[1])
            xB = min(box_current[0] + box_current[2], box_next[0] + box_next[2])
            yB = min(box_current[1] + box_current[3], box_next[1] + box_next[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = box_current[2] * box_current[3]
            boxBArea = box_next[2] * box_next[3]
            
            iou = interArea / float(boxAArea + boxBArea - interArea)
            ious.append(iou)

    avg_temporal_iou = np.mean(ious) if ious else 0
    return {"avg_temporal_iou": avg_temporal_iou, "detection_flicker_count": flicker_count}

def measure_performance(model_type, config):
    """Mierzy FPS i zużycie VRAM."""
    # Ta funkcja jest uproszczona. Dokładny pomiar FPS i VRAM
    # powinien być zintegrowany z pętlą inferencji w `run_inference.py`.
    # Tutaj podajemy szacunkowe wartości lub logikę do zaimplementowania.
    
    # Symulacja pomiaru
    if model_type == 'yolo':
        fps = 30 
        vram_mb = 2048
    else: # detr
        fps = 10
        vram_mb = 4096
        
    print(f"INFO: Performance for {model_type} is estimated. For precise results, integrate measurement into inference loop.")
    
    return {"fps": fps, "vram_mb": vram_mb}

def load_performance_metrics(model_type, config):
    """Load measured performance if available, otherwise return estimates."""
    output_dir = config['output']['directory']
    perf_path = os.path.join(output_dir, f'{model_type}_performance.json')
    if os.path.exists(perf_path):
        try:
            with open(perf_path, 'r') as f:
                data = json.load(f)
            return {
                "fps": data.get("fps"),
                "vram_mb": data.get("vram_mb"),
                "frames": data.get("frames"),
                "total_seconds": data.get("total_seconds"),
            }
        except Exception:
            pass
    return measure_performance(model_type, config)

def main():
    config = load_config()
    output_dir = config['output']['directory']
    gt_path = config['dataset']['annotations_path']

    gt_coco = COCO(gt_path)

    full_report = {}

    for model_type in ['yolo', 'detr']:
        print(f"\n--- Evaluating {model_type.upper()} ---")
        pred_path = os.path.join(output_dir, f'{model_type}_predictions.json')
        if not os.path.exists(pred_path):
            print(f"Prediction file not found: {pred_path}. Run run_inference.py first.")
            continue

        pred_coco = gt_coco.loadRes(pred_path)
        with open(pred_path, 'r') as f:
            pred_results = json.load(f)

        # Ogranicz ewaluację do obrazów obecnych w predykcjach
        img_ids = [p['image_id'] for p in pred_results]

        # 1. Standardowe metryki
        print("\n[Standard COCO Metrics]")
        standard_stats = calculate_standard_metrics(gt_coco, pred_coco, img_ids=img_ids)

        # 2. Metryki stabilności czasowej
        print("\n[Temporal Stability Metrics]")
        temporal_stats = calculate_temporal_stability(pred_results)
        print(f"Average Temporal IoU: {temporal_stats['avg_temporal_iou']:.4f}")
        print(f"Detection Flicker Count: {temporal_stats['detection_flicker_count']}")

        # 3. Metryki wydajności
        print("\n[Performance Metrics]")
        performance_stats = load_performance_metrics(model_type, config)
        print(f"Frames Per Second (FPS): {performance_stats['fps']}")
        print(f"VRAM Usage (MB): {performance_stats['vram_mb']}")

        full_report[model_type] = {
            "standard_metrics": standard_stats,
            "temporal_metrics": temporal_stats,
            "performance_metrics": performance_stats
        }

    report_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    
    print(f"\nFull benchmark report saved to {report_path}")

if __name__ == "__main__":
    main()
