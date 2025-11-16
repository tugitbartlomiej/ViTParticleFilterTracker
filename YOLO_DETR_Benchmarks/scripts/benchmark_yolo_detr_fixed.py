"""
Robust YOLO vs DETR benchmark script with correct COCO evaluation.

Key features:
- Uses COCO GT annotations to drive the image list (no leakage, consistent file names)
- Enforces single-class mapping (raw COCO category_id is used for predictions)
- Converts bboxes to COCO format [x, y, width, height] in pixels
- Evaluates with pycocotools COCOeval and saves a compact summary + predictions

Usage (example):
  python benchmark_yolo_detr_fixed.py \
    --images-dir F:\\...\\output_cataract\\images \
    --annotations F:\\...\\output_cataract\\annotations_test.json \
    --out-dir YOLO_DETR_Benchmarks\\Benchmarks \
    --yolo-weights F:\\...\\models\\YOLO\\epoch100.pt \
    --detr-model facebook/detr-resnet-50 \
    --detr-state F:\\...\\DETR\\Checkpoints\\checkpoint_epoch_140.pth \
    --conf 0.25

Notes:
- If --detr-state is provided (a .pth with state_dict), script instantiates a 1-class DETR and loads it.
- If only --detr-model (HF id/path) is provided and it is already 1-class, it will be used directly; otherwise the script will try to enforce single class.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("pycocotools is required. On Windows: pip install pycocotools-windows")


def parse_args():
    ap = argparse.ArgumentParser(description="YOLO vs DETR benchmark (COCO evaluation)")
    ap.add_argument("--images-dir", required=True, help="Directory with images referenced by COCO annotations")
    ap.add_argument("--annotations", required=True, help="COCO annotations JSON (GT)")
    ap.add_argument("--out-dir", required=True, help="Output directory for results")

    # YOLO
    ap.add_argument("--yolo-weights", default=None, help="Path to YOLOv8 .pt weights (optional)")

    # DETR
    ap.add_argument("--detr-model", default="facebook/detr-resnet-50", help="HF model id or local dir")
    ap.add_argument("--detr-state", default=None, help="Path to .pth state dict for DETR (optional)")

    # Inference
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for inference")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if None)")

    return ap.parse_args()


def ensure_output_dir(out_root: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    out = out_root / f"benchmark_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_coco_and_images(annotations_path: Path, images_dir: Path) -> Tuple[COCO, List[Dict], int, str]:
    coco_gt = COCO(str(annotations_path))
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    if not cats:
        # Default single class
        raw_cid = 1
        cat_name = "tooltip"
    else:
        raw_cid = int(cats[0]['id'])
        cat_name = str(cats[0].get('name', 'tooltip'))

    coco_images = coco_gt.loadImgs(coco_gt.getImgIds())
    # Validate image files exist
    missing = []
    for im in coco_images:
        fp = images_dir / im['file_name']
        if not fp.is_file():
            missing.append(str(fp))
    if missing:
        print(f"[WARN] {len(missing)} images listed in COCO not found under images_dir. First 5:\n  " + "\n  ".join(missing[:5]))
    return coco_gt, coco_images, raw_cid, cat_name


def run_yolo(images: List[Dict], images_dir: Path, weights_path: Path, conf: float, batch: int, raw_cid: int) -> List[Dict]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        raise SystemExit("ultralytics is required for YOLO benchmarking: pip install ultralytics")

    model = YOLO(str(weights_path))
    preds: List[Dict] = []
    paths = [str(images_dir / im['file_name']) for im in images]
    for i in range(0, len(paths), batch):
        batch_paths = paths[i:i + batch]
        results = model.predict(batch_paths, conf=conf, verbose=False)
        for pth, r in zip(batch_paths, results):
            # Image id by file_name
            file_name = Path(pth).name
            # We need to map file_name to image_id; we built images list aligned, so pass index
            # But safer: use a lookup
            # We'll attach image_id later by a map built outside
            if r and r.boxes is not None:
                for b in r.boxes:
                    xyxy = b.xyxy.cpu().numpy().reshape(-1)
                    score = float(b.conf.cpu().numpy().reshape(-1)[0])
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    x = x1
                    y = y1
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    preds.append({
                        'file_name': file_name,
                        'category_id': int(raw_cid),
                        'bbox': [x, y, w, h],
                        'score': float(score),
                    })
            else:
                preds.append({'file_name': file_name, 'category_id': int(raw_cid), 'bbox': [0, 0, 0, 0], 'score': 0.0})
    return preds


def build_single_class_detr(detr_model: str, raw_cid: int, class_name: str):
    from transformers import DetrConfig, DetrForObjectDetection
    id2label = {0: class_name}
    label2id = {class_name: 0}
    config = DetrConfig.from_pretrained(detr_model, num_labels=1, id2label=id2label, label2id=label2id)
    model = DetrForObjectDetection.from_pretrained(detr_model, config=config, ignore_mismatched_sizes=True)
    return model


def run_detr(images: List[Dict], images_dir: Path, detr_model: str, detr_state: Path | None, conf: float, batch: int, raw_cid: int, class_name: str, device: torch.device) -> List[Dict]:
    from transformers import DetrImageProcessor

    model = build_single_class_detr(detr_model, raw_cid, class_name)
    if detr_state and Path(detr_state).is_file():
        sd = torch.load(detr_state, map_location='cpu')
        # accept both pure state_dict or dict with model_state_dict
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
            # strip potential module. prefix
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        try:
            model.load_state_dict(sd, strict=False)
            print("[INFO] Loaded DETR state dict with strict=False")
        except Exception as e:
            print(f"[WARN] Could not fully load DETR state dict: {e}")
    model.to(device)
    model.eval()
    processor = DetrImageProcessor.from_pretrained(detr_model)

    preds: List[Dict] = []
    # build a map to image_id by file_name
    # But here we'll attach file_name and convert to image_id later
    paths = [str(images_dir / im['file_name']) for im in images]
    for i in range(0, len(paths), batch):
        batch_paths = paths[i:i + batch]
        batch_imgs = []
        orig_sizes: List[Tuple[int, int]] = []
        file_names = []
        for p in batch_paths:
            im = Image.open(p).convert('RGB')
            file_names.append(Path(p).name)
            orig_sizes.append(im.size[::-1])  # (H, W)
            batch_imgs.append(im)
        enc = processor(images=batch_imgs, return_tensors='pt')
        with torch.no_grad():
            outputs = model(pixel_values=enc['pixel_values'].to(device), pixel_mask=enc['pixel_mask'].to(device))
        target_sizes = torch.tensor(orig_sizes, device=device)
        post = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)
        for p, fn in zip(post, file_names):
            boxes = p['boxes'].detach().cpu()
            scores = p['scores'].detach().cpu()
            labels = p['labels'].detach().cpu()
            for (x1, y1, x2, y2), s, _ in zip(boxes, scores, labels):
                # apply threshold here
                if float(s) < conf:
                    continue
                x = float(x1)
                y = float(y1)
                w = float(max(0.0, x2 - x1))
                h = float(max(0.0, y2 - y1))
                preds.append({
                    'file_name': fn,
                    'category_id': int(raw_cid),
                    'bbox': [x, y, w, h],
                    'score': float(s),
                })
    return preds


def preds_to_coco(preds: List[Dict], coco_images: List[Dict]) -> List[Dict]:
    # map file_name -> image_id
    name_to_id = {im['file_name']: int(im['id']) for im in coco_images}
    results: List[Dict] = []
    for pr in preds:
        fn = pr['file_name']
        if fn not in name_to_id:
            # Skip predictions for files not in GT
            continue
        image_id = name_to_id[fn]
        x, y, w, h = pr['bbox']
        results.append({
            'image_id': image_id,
            'category_id': int(pr['category_id']),
            'bbox': [float(x), float(y), float(w), float(h)],
            'score': float(pr['score']),
        })
    return results


def eval_coco(coco_gt: COCO, coco_preds: List[Dict]) -> Dict[str, float]:
    if not coco_preds:
        return {"mAP_0.5:0.95": 0.0, "mAP_0.5": 0.0, "mAP_0.75": 0.0, "AR@100": 0.0}
    coco_dt = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        'mAP_0.5:0.95': float(coco_eval.stats[0]),
        'mAP_0.5': float(coco_eval.stats[1]),
        'mAP_0.75': float(coco_eval.stats[2]),
        'AR@100': float(coco_eval.stats[8]),
    }


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    annotations = Path(args.annotations)
    out_root = Path(args.out_dir)
    out_dir = ensure_output_dir(out_root)
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] Output directory: {out_dir}")
    coco_gt, coco_images, raw_cid, class_name = load_coco_and_images(annotations, images_dir)
    print(f"[INFO] COCO categories: raw_cid={raw_cid}, name={class_name}")
    print(f"[INFO] Images in GT: {len(coco_images)}")

    all_results = {}

    # YOLO
    if args.yolo_weights:
        t0 = time.time()
        yolo_preds = run_yolo(coco_images, images_dir, Path(args.yolo_weights), args.conf, args.batch, raw_cid)
        yolo_coco = preds_to_coco(yolo_preds, coco_images)
        yolo_metrics = eval_coco(coco_gt, yolo_coco)
        with open(out_dir / 'yolo_predictions.json', 'w', encoding='utf-8') as f:
            json.dump(yolo_coco, f)
        all_results['yolo'] = yolo_metrics
        print(f"[YOLO] mAP@0.5={yolo_metrics['mAP_0.5']:.4f}  mAP@0.5:0.95={yolo_metrics['mAP_0.5:0.95']:.4f}  AR@100={yolo_metrics['AR@100']:.4f}  (t={time.time()-t0:.1f}s)")
    else:
        print("[YOLO] Skipping (no weights provided)")

    # DETR
    t0 = time.time()
    detr_preds = run_detr(coco_images, images_dir, args.detr_model, Path(args.detr_state) if args.detr_state else None, args.conf, args.batch, raw_cid, class_name, device)
    detr_coco = preds_to_coco(detr_preds, coco_images)
    detr_metrics = eval_coco(coco_gt, detr_coco)
    with open(out_dir / 'detr_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(detr_coco, f)
    all_results['detr'] = detr_metrics
    print(f"[DETR] mAP@0.5={detr_metrics['mAP_0.5']:.4f}  mAP@0.5:0.95={detr_metrics['mAP_0.5:0.95']:.4f}  AR@100={detr_metrics['AR@100']:.4f}  (t={time.time()-t0:.1f}s)")

    # Save summary
    summary = {
        'dataset': str(annotations),
        'images_dir': str(images_dir),
        'raw_category_id': raw_cid,
        'category_name': class_name,
        'conf_threshold': args.conf,
        'batch': args.batch,
        'device': str(device),
        'results': all_results,
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

