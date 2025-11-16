#!/usr/bin/env python3
"""
Complete YOLO vs DETR Benchmark on Cataract Surgery Instruments Dataset
YOLO Epoch 70 vs DETR Epoch 100
Tests on train/test/valid splits
"""

import sys, json, time
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor, DetrForObjectDetection

# YOLO Legacy Fix
import ultralytics.utils.loss as loss_module
if not hasattr(loss_module, 'DFLoss'):
    class DFLoss: pass
    loss_module.DFLoss = DFLoss
    sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss

original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

# Config
YOLO_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/YOLO_EDEN_TRAIN/epoch70.pt"
DETR_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR/checkpoint_epoch_100.pth"
DATASET_ROOT = Path("E:/cataract_surgery_Instruments_detection.v1i.coco")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Benchmarks")
BENCHMARK_DIR = OUTPUT_DIR / f"CATARACT_YOLO70_DETR100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

print(f"""
{'='*80}
CATARACT SURGERY INSTRUMENTS BENCHMARK
YOLO Epoch 70 vs DETR Epoch 100
{'='*80}
Device: {DEVICE}
Output: {BENCHMARK_DIR}
{'='*80}
""")

# Load models
print("Loading YOLO Epoch 70...")
yolo_model = YOLO(YOLO_CHECKPOINT)
yolo_model.to(DEVICE)
print("✓ YOLO loaded")

print("Loading DETR Epoch 100...")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=1, ignore_mismatched_sizes=True)
checkpoint = torch.load(DETR_CHECKPOINT, map_location=DEVICE)
detr_model.load_state_dict(checkpoint['model_state_dict'])
detr_model.to(DEVICE)
detr_model.eval()
print("✓ DETR loaded")

def benchmark_split(split_name, images_dir, annotations_path):
    print(f"\n{'='*60}\nBenchmarking {split_name.upper()}\n{'='*60}")

    # Load and map all categories to class 1
    with open(annotations_path) as f:
        coco_data = json.load(f)
    coco_data['categories'] = [{"id": 1, "name": "instrument", "supercategory": "surgical"}]
    for ann in coco_data['annotations']:
        ann['category_id'] = 1
    temp_ann = BENCHMARK_DIR / f"temp_{split_name}_ann.json"
    with open(temp_ann, 'w') as f:
        json.dump(coco_data, f)

    coco = COCO(str(temp_ann))
    image_ids = coco.getImgIds()
    print(f"Images: {len(image_ids)}")

    # YOLO
    print(f"\nYOLO Inference...")
    yolo_preds, yolo_times = [], []
    for img_id in tqdm(image_ids, desc="YOLO"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = images_dir / img_info['file_name']
        if not img_path.exists(): continue

        start = time.time()
        results = yolo_model(str(img_path), verbose=False)
        yolo_times.append(time.time() - start)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    yolo_preds.append({
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(box.conf[0])
                    })

    yolo_fps = len(image_ids) / sum(yolo_times) if yolo_times else 0
    print(f"✓ YOLO: {len(yolo_preds)} predictions, {yolo_fps:.2f} FPS")

    # DETR
    print(f"\nDETR Inference...")
    detr_preds, detr_times = [], []
    for img_id in tqdm(image_ids, desc="DETR"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = images_dir / img_info['file_name']
        if not img_path.exists(): continue

        # START TIMER - include ALL operations for fair comparison
        start = time.time()

        # Image loading
        image = Image.open(img_path).convert("RGB")

        # Preprocessing
        inputs = detr_processor(images=image, return_tensors="pt").to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = detr_model(**inputs)

        # Post-processing
        target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

        # END TIMER
        detr_times.append(time.time() - start)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            detr_preds.append({
                "image_id": int(img_id),
                "category_id": 1 if label == 0 else int(label),
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(score)
            })

    detr_fps = len(image_ids) / sum(detr_times) if detr_times else 0
    print(f"✓ DETR: {len(detr_preds)} predictions, {detr_fps:.2f} FPS")

    # Evaluate
    def evaluate(preds, name):
        if not preds: return None
        pred_file = BENCHMARK_DIR / f"{name}_{split_name}_predictions.json"
        with open(pred_file, 'w') as f:
            json.dump(preds, f)

        coco_dt = coco.loadRes(str(pred_file))
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP@0.5:0.95": coco_eval.stats[0] * 100,
            "mAP@0.5": coco_eval.stats[1] * 100,
            "mAP@0.75": coco_eval.stats[2] * 100,
            "AR@100": coco_eval.stats[8] * 100
        }

    print(f"\nEvaluating YOLO...")
    yolo_metrics = evaluate(yolo_preds, "yolo")
    print(f"\nEvaluating DETR...")
    detr_metrics = evaluate(detr_preds, "detr")

    return {
        "yolo": {"metrics": yolo_metrics, "fps": yolo_fps, "preds": len(yolo_preds)},
        "detr": {"metrics": detr_metrics, "fps": detr_fps, "preds": len(detr_preds)}
    }

# Run on all splits
results = {}
for split in ["train", "test", "valid"]:
    ann_path = DATASET_ROOT / split / "_annotations.coco.json"
    img_dir = DATASET_ROOT / split
    if ann_path.exists():
        results[split] = benchmark_split(split, img_dir, ann_path)

# Generate report
report = f"""# Cataract Surgery Instruments Benchmark

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Models
- **YOLO:** Epoch 70
- **DETR:** Epoch 100

## Dataset
- Train: 2083 images
- Test: 50 images
- Valid: 248 images

---

"""

for split in ["train", "test", "valid"]:
    if split not in results: continue
    r = results[split]
    y, d = r["yolo"], r["detr"]

    report += f"## {split.upper()} RESULTS\n\n"
    report += "| Metric | YOLO | DETR | Diff |\n"
    report += "|--------|------|------|------|\n"

    if y["metrics"] and d["metrics"]:
        for m in ["mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "AR@100"]:
            yv, dv = y["metrics"][m], d["metrics"][m]
            report += f"| {m} | {yv:.2f}% | {dv:.2f}% | {yv-dv:+.2f}% |\n"
        report += f"| FPS | {y['fps']:.2f} | {d['fps']:.2f} | - |\n"
        report += f"| Predictions | {y['preds']} | {d['preds']} | - |\n\n"

        winner = "YOLO" if y["metrics"]["mAP@0.5:0.95"] > d["metrics"]["mAP@0.5:0.95"] else "DETR"
        margin = abs(y["metrics"]["mAP@0.5:0.95"] - d["metrics"]["mAP@0.5:0.95"])
        report += f"**Winner:** {winner} (+{margin:.2f}%)\n\n---\n\n"

report += f"## Summary\n\nOutput: `{BENCHMARK_DIR}`\n"

report_path = BENCHMARK_DIR / "BENCHMARK_REPORT.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'='*80}\nCOMPLETE!\n{'='*80}")
print(f"Report: {report_path}")
