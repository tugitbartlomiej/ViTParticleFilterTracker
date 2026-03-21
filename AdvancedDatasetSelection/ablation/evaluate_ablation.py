"""
Evaluate all ablation variants and generate comparison table.

Evaluates each variant's best checkpoint on:
1. Same-distribution: val split (from video_split.json)
2. Cross-dataset: Roboflow Cataract Surgery Instruments (external)

Usage:
  python evaluate_ablation.py \
    --ablation-dir /path/to/ablation \
    --images-dir /path/to/all/images \
    --external-dir E:/cataract_surgery_Instruments_detection.v1i.coco \
    --leaked-images ../YOLO_DETR_Benchmarks/scripts/leaked_valid_images.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("ERROR: pycocotools required. pip install pycocotools")
    sys.exit(1)


VARIANTS = {
    'v1': 'Full Pipeline',
    'v2': 'No Fourier',
    'v3': 'No EL2N',
    'v4': 'No Clustering',
    'v5': 'Random',
}

CONFIDENCE_THRESHOLD = 0.3


class InferenceDataset(Dataset):
    def __init__(self, images_dir: str, coco_data: dict, processor):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.images = []
        for img in coco_data.get('images', []):
            fp = self.images_dir / img['file_name']
            if fp.exists():
                self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img = Image.open(self.images_dir / info['file_name']).convert('RGB')
        enc = self.processor(images=img, return_tensors='pt')
        return {
            'pixel_values': enc['pixel_values'].squeeze(0),
            'pixel_mask': enc['pixel_mask'].squeeze(0),
            'image_id': int(info['id']),
            'orig_size': (int(info.get('height', img.height)), int(info.get('width', img.width))),
        }


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'pixel_mask': torch.stack([b['pixel_mask'] for b in batch]),
        'image_ids': [b['image_id'] for b in batch],
        'orig_sizes': [b['orig_size'] for b in batch],
    }


def compute_tp_fp_fn(predictions: list, coco_gt, iou_threshold: float = 0.5, conf_threshold: float = 0.3):
    """Compute TP/FP/FN at given IoU and confidence thresholds."""
    filtered_preds = [p for p in predictions if p['score'] >= conf_threshold]

    gt_by_img = {}
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        gt_by_img.setdefault(ann['image_id'], []).append(ann)

    tp, fp, fn = 0, 0, 0
    for img_id in coco_gt.getImgIds():
        gts = gt_by_img.get(img_id, [])
        preds = [p for p in filtered_preds if p['image_id'] == img_id]
        preds.sort(key=lambda x: -x['score'])

        matched = set()
        for pred in preds:
            best_iou, best_gt = 0, -1
            for i, gt in enumerate(gts):
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = i
            if best_iou >= iou_threshold and best_gt not in matched:
                tp += 1
                matched.add(best_gt)
            else:
                fp += 1
        fn += len(gts) - len(matched)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}


def compute_iou(box1, box2):
    """IoU for [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(1e-8, union)


@torch.no_grad()
def run_inference(model, dataloader, processor, device, category_id: int = 1):
    model.eval()
    predictions = []
    for batch in dataloader:
        pv = batch['pixel_values'].to(device)
        pm = batch['pixel_mask'].to(device)
        outputs = model(pixel_values=pv, pixel_mask=pm)

        target_sizes = torch.tensor([[h, w] for h, w in batch['orig_sizes']], device=device)
        post = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)

        for p, img_id in zip(post, batch['image_ids']):
            for (x1, y1, x2, y2), s, _ in zip(
                p['boxes'].cpu().numpy(), p['scores'].cpu().numpy(), p['labels'].cpu().numpy()
            ):
                predictions.append({
                    'image_id': int(img_id),
                    'category_id': category_id,
                    'bbox': [float(x1), float(y1), float(max(0, x2 - x1)), float(max(0, y2 - y1))],
                    'score': float(s),
                })
    return predictions


def eval_coco_map(predictions, gt_json_path):
    """Run COCO mAP evaluation."""
    coco_gt = COCO(gt_json_path)
    if not predictions:
        return {'mAP_0.5': 0.0, 'mAP_0.5:0.95': 0.0}

    coco_dt = coco_gt.loadRes(predictions)
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


def evaluate_variant(variant_id: str, ablation_dir: Path, images_dir: Path,
                     processor, device, batch_size: int = 8,
                     external_dir: Optional[Path] = None,
                     leaked_images: Optional[set] = None) -> dict:
    """Evaluate one variant on val + optional external dataset."""
    vdir = ablation_dir / variant_id
    best_ckpt = vdir / 'best_model' / 'best_model.pth'
    if not best_ckpt.exists():
        # Fallback to latest checkpoint
        ckpt_dir = vdir / 'checkpoints'
        files = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'),
                       key=lambda p: int(p.stem.split('_')[-1])) if ckpt_dir.exists() else []
        best_ckpt = files[-1] if files else None

    if not best_ckpt or not best_ckpt.exists():
        print(f"  WARNING: No checkpoint for {variant_id}")
        return {}

    # Load model
    config = DetrConfig.from_pretrained('facebook/detr-resnet-50', num_labels=1,
                                        id2label={0: 'tooltip'}, label2id={'tooltip': 0})
    config.num_queries = 100
    model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', config=config,
                                                   ignore_mismatched_sizes=True)
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"  Loaded: {best_ckpt.name} (epoch {ckpt.get('epoch', '?')})")

    results = {'variant': variant_id, 'name': VARIANTS.get(variant_id, variant_id)}

    # === Same-distribution (val split) ===
    val_json = vdir / 'annotations_val.json'
    if val_json.exists():
        with open(val_json, 'r') as f:
            val_coco = json.load(f)
        ds = InferenceDataset(str(images_dir), val_coco, processor)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)
        preds = run_inference(model, loader, processor, device)

        # Save preds temp for COCO eval
        tmp_json = str(vdir / '_tmp_val_preds.json')
        with open(tmp_json, 'w') as f:
            json.dump(preds, f)

        mAP = eval_coco_map(preds, str(val_json))
        coco_gt = COCO(str(val_json))
        det_metrics = compute_tp_fp_fn(preds, coco_gt, conf_threshold=CONFIDENCE_THRESHOLD)

        results['val'] = {**mAP, **det_metrics}
        print(f"    Val: mAP@0.5={mAP['mAP_0.5']:.4f} | F1={det_metrics['F1']:.4f}")

    # === Cross-dataset (Roboflow) ===
    if external_dir and external_dir.exists():
        for split in ['valid', 'test']:
            split_dir = external_dir / split
            ann_file = split_dir / '_annotations.coco.json'
            if not ann_file.exists():
                continue

            with open(ann_file, 'r') as f:
                ext_coco = json.load(f)

            # Exclude leaked images for fairness
            if leaked_images and split == 'valid':
                ext_coco['images'] = [img for img in ext_coco['images']
                                      if img['file_name'].lower() not in leaked_images]
                valid_ids = {img['id'] for img in ext_coco['images']}
                ext_coco['annotations'] = [a for a in ext_coco['annotations'] if a['image_id'] in valid_ids]

            ds = InferenceDataset(str(split_dir), ext_coco, processor)
            loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)
            preds = run_inference(model, loader, processor, device)

            # Remap category_id to match external dataset
            cats = ext_coco.get('categories', [])
            ext_cat_id = int(cats[0]['id']) if cats else 1
            for p in preds:
                p['category_id'] = ext_cat_id

            tmp = str(vdir / f'_tmp_ext_{split}_preds.json')
            with open(tmp, 'w') as f:
                json.dump(ext_coco, f)  # just for ref

            # Save ext coco with filtered images for eval
            filtered_ann = str(vdir / f'_tmp_ext_{split}_gt.json')
            with open(filtered_ann, 'w') as f:
                json.dump(ext_coco, f)

            mAP = eval_coco_map(preds, filtered_ann)
            coco_gt = COCO(filtered_ann)
            det_metrics = compute_tp_fp_fn(preds, coco_gt, conf_threshold=CONFIDENCE_THRESHOLD)

            results[f'ext_{split}'] = {**mAP, **det_metrics}
            print(f"    Ext-{split}: mAP@0.5={mAP['mAP_0.5']:.4f} | F1={det_metrics['F1']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def format_table(all_results: List[dict]) -> str:
    """Generate markdown comparison table."""
    lines = []
    lines.append("## Ablation Study Results\n")

    # Same-distribution table
    lines.append("### Same-Distribution (Val Split)\n")
    lines.append("| Variant | mAP@0.5 | mAP@0.5:0.95 | F1 | Prec. | Recall | TP |")
    lines.append("|---------|---------|--------------|-----|-------|--------|-----|")
    for r in all_results:
        v = r.get('val', {})
        if not v:
            lines.append(f"| {r.get('name', '?')} | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['name']} | {v.get('mAP_0.5', 0)*100:.2f}% | {v.get('mAP_0.5:0.95', 0)*100:.2f}% "
            f"| {v.get('F1', 0)*100:.1f}% | {v.get('Precision', 0)*100:.1f}% "
            f"| {v.get('Recall', 0)*100:.1f}% | {v.get('TP', 0)} |"
        )

    # Cross-dataset table
    lines.append("\n### Cross-Dataset (External Benchmark)\n")
    lines.append("| Variant | mAP@0.5 (valid) | mAP@0.5 (test) | F1 (valid) | F1 (test) |")
    lines.append("|---------|----------------|----------------|-----------|----------|")
    for r in all_results:
        ev = r.get('ext_valid', {})
        et = r.get('ext_test', {})
        lines.append(
            f"| {r['name']} "
            f"| {ev.get('mAP_0.5', 0)*100:.2f}% "
            f"| {et.get('mAP_0.5', 0)*100:.2f}% "
            f"| {ev.get('F1', 0)*100:.1f}% "
            f"| {et.get('F1', 0)*100:.1f}% |"
        )

    # Delta table
    if all_results and 'val' in all_results[0]:
        baseline_map = all_results[0].get('val', {}).get('mAP_0.5', 0)
        baseline_cross = all_results[0].get('ext_valid', {}).get('mAP_0.5', 0)
        lines.append("\n### Delta vs Full Pipeline\n")
        lines.append("| Variant | Δ mAP@0.5 (val) | Δ mAP@0.5 (cross) |")
        lines.append("|---------|----------------|-------------------|")
        for r in all_results:
            vm = r.get('val', {}).get('mAP_0.5', 0)
            cm = r.get('ext_valid', {}).get('mAP_0.5', 0)
            dv = (vm - baseline_map) * 100
            dc = (cm - baseline_cross) * 100
            lines.append(f"| {r['name']} | {dv:+.2f}pp | {dc:+.2f}pp |")

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Evaluate ablation study variants')
    ap.add_argument('--ablation-dir', type=str, required=True,
                    help='Base directory with v1..v5 subdirs')
    ap.add_argument('--images-dir', type=str, required=True,
                    help='Directory with all images (for val/test inference)')
    ap.add_argument('--external-dir', type=str, default=None,
                    help='Roboflow cataract dataset for cross-dataset eval')
    ap.add_argument('--leaked-images', type=str, default=None,
                    help='JSON with leaked validation images to exclude')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--output', type=str, default='ablation_results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ablation_dir = Path(args.ablation_dir)
    images_dir = Path(args.images_dir)
    external_dir = Path(args.external_dir) if args.external_dir else None

    # Load leaked images
    leaked = None
    if args.leaked_images:
        with open(args.leaked_images, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'images' in data:
            leaked = {img.lower() for img in data['images']}
        elif isinstance(data, list):
            leaked = {img.lower() for img in data}

    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

    all_results = []
    for vid in ['v1', 'v2', 'v3', 'v4', 'v5']:
        print(f"\n{'='*50}")
        print(f"Evaluating {vid}: {VARIANTS[vid]}")
        print(f"{'='*50}")
        result = evaluate_variant(vid, ablation_dir, images_dir, processor, device,
                                  args.batch_size, external_dir, leaked)
        if result:
            all_results.append(result)

    # Save raw results
    out_path = Path(args.output)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {out_path}")

    # Generate markdown table
    table = format_table(all_results)
    md_path = out_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write(table)
    print(f"Comparison table saved to: {md_path}")
    print(f"\n{table}")


if __name__ == '__main__':
    main()
