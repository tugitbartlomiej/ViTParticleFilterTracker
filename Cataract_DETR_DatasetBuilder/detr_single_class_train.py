import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
except Exception:  # pragma: no cover
    COCO = None
    COCOeval = None


class CocoSingleClassDataset(Dataset):
    def __init__(self, images_dir: str, annotations_path: str, processor: DetrImageProcessor):
        self.images_dir = Path(images_dir)
        with open(annotations_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        # Images index
        self.images = {img['id']: img for img in coco.get('images', [])}

        # Annotations index
        anns_by_img: Dict[int, List[Dict]] = {}
        for ann in coco.get('annotations', []):
            img_id = int(ann['image_id'])
            anns_by_img.setdefault(img_id, []).append(ann)
        self.anns_by_img = anns_by_img

        # Category mapping → contiguous labels starting from 0
        raw_ids = sorted({int(c.get('id', 0)) for c in coco.get('categories', [])} | {
            int(a.get('category_id', 0)) for a in coco.get('annotations', [])
        })
        if not raw_ids:
            # Default to single class id=1 if empty
            raw_ids = [1]
        self.raw_to_contig = {rid: idx for idx, rid in enumerate(raw_ids)}
        self.contig_to_name = {}
        # Prefer category names from file, fallback to 'tooltip'
        cat_name_map = {int(c['id']): c.get('name', 'tooltip') for c in coco.get('categories', [])}
        for rid, idx in self.raw_to_contig.items():
            self.contig_to_name[idx] = cat_name_map.get(rid, 'tooltip')

        self.processor = processor
        self.image_ids = sorted(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.images[img_id]
        file_name = info['file_name']
        img_path = self.images_dir / file_name
        image = Image.open(img_path).convert('RGB')
        orig_w = int(info.get('width', image.width))
        orig_h = int(info.get('height', image.height))

        anns = []
        for ann in self.anns_by_img.get(img_id, []):
            bbox = [float(x) for x in ann['bbox']]  # [x, y, w, h]
            cid_raw = int(ann.get('category_id', 1))
            cid = int(self.raw_to_contig.get(cid_raw, 0))
            anns.append({'bbox': bbox, 'category_id': cid, 'iscrowd': int(ann.get('iscrowd', 0))})

        target = {'image_id': int(img_id), 'annotations': anns}
        enc = self.processor(images=image, annotations=target, return_tensors='pt')
        item = {
            'pixel_values': enc['pixel_values'].squeeze(0),
            'pixel_mask': enc['pixel_mask'].squeeze(0),
            'labels': enc['labels'][0] if enc['labels'] else {'class_labels': torch.zeros((0,), dtype=torch.long), 'boxes': torch.zeros((0, 4), dtype=torch.float32)},
            'orig_size': (orig_h, orig_w),
            'image_id': int(img_id),
        }
        return item


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    pixel_mask = torch.stack([b['pixel_mask'] for b in batch])
    labels = [b['labels'] for b in batch]
    orig_sizes = [b['orig_size'] for b in batch]
    image_ids = [b['image_id'] for b in batch]
    return {'pixel_values': pixel_values, 'pixel_mask': pixel_mask, 'labels': labels, 'orig_sizes': orig_sizes, 'image_ids': image_ids}


def train_one_epoch(model, dataloader, optimizer, device, scaler: Optional[torch.cuda.amp.GradScaler], log_interval=50):
    model.train()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(dataloader):
        if batch is None:
            continue
        # model accepts only these keys
        model_inputs = {
            'pixel_values': batch['pixel_values'].to(device),
            'pixel_mask': batch['pixel_mask'].to(device),
            'labels': batch['labels'],
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(**model_inputs)
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
        n += 1
        if (i + 1) % log_interval == 0:
            print(f"  iter {i+1}/{len(dataloader)}  loss={total_loss / max(1,n):.4f}")
    return total_loss / max(1, n)


@torch.no_grad()
def eval_loss(model, dataloader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in dataloader:
        if batch is None:
            continue
        model_inputs = {
            'pixel_values': batch['pixel_values'].to(device),
            'pixel_mask': batch['pixel_mask'].to(device),
            'labels': batch['labels'],
        }
        outputs = model(**model_inputs)
        total += float(outputs.loss.detach().item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def eval_coco_map(model, dataloader, processor: DetrImageProcessor, device, val_json_path: str, category_id_raw: Optional[int] = None):
    if COCO is None or COCOeval is None:
        print("pycocotools not installed: pip install pycocotools; skipping COCO mAP evaluation")
        return None
    coco_gt = COCO(val_json_path)
    # determine single raw category id if not provided
    if category_id_raw is None:
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        if not cats:
            category_id_raw = 1
        else:
            category_id_raw = int(cats[0]['id'])

    model.eval()
    predictions = []
    for batch in dataloader:
        if batch is None:
            continue
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # prepare target sizes from original sizes
        target_sizes = torch.tensor([[h, w] for (h, w) in batch['orig_sizes']], device=device)
        post = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)

        for p, img_id, (h, w) in zip(post, batch['image_ids'], batch['orig_sizes']):
            boxes = p['boxes'].detach().cpu().numpy().tolist()  # xyxy
            scores = p['scores'].detach().cpu().numpy().tolist()
            labels = p['labels'].detach().cpu().numpy().tolist()
            for (x1, y1, x2, y2), s, _ in zip(boxes, scores, labels):
                x = float(x1)
                y = float(y1)
                bw = float(max(0.0, x2 - x1))
                bh = float(max(0.0, y2 - y1))
                predictions.append({
                    'image_id': int(img_id),
                    'category_id': int(category_id_raw),
                    'bbox': [x, y, bw, bh],
                    'score': float(s),
                })

    if not predictions:
        print("No predictions generated; skipping COCO mAP evaluation")
        return None

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # coco_eval.stats indices: [0]=mAP .5:.95, [1]=mAP .5, [2]=mAP .75, [8]=AR@100
    return {
        'mAP_0.5:0.95': float(coco_eval.stats[0]),
        'mAP_0.5': float(coco_eval.stats[1]),
        'mAP_0.75': float(coco_eval.stats[2]),
        'AR@100': float(coco_eval.stats[8]),
    }


def main():
    ap = argparse.ArgumentParser(description='Single-class DETR trainer (tooltip detection)')
    ap.add_argument('--images_dir', required=True, help='Directory with images')
    ap.add_argument('--train_json', required=True, help='COCO annotations (train)')
    ap.add_argument('--val_json', required=False, help='COCO annotations (val)')
    ap.add_argument('--output_dir', required=True, help='Output directory')
    ap.add_argument('--model_checkpoint', default='facebook/detr-resnet-50', help='Base checkpoint')
    ap.add_argument('--num_queries', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_backbone', type=float, default=1e-5)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--use_amp', action='store_true')
    ap.add_argument('--coco_eval_every', type=int, default=1, help='Run COCO eval every N epochs if val_json provided')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    processor = DetrImageProcessor.from_pretrained(args.model_checkpoint)

    # Read categories from train_json to define single-class label mapping (contiguous)
    with open(args.train_json, 'r', encoding='utf-8') as f:
        train_coco = json.load(f)
    raw_ids = sorted({int(c.get('id', 0)) for c in train_coco.get('categories', [])} | {
        int(a.get('category_id', 0)) for a in train_coco.get('annotations', [])
    })
    if not raw_ids:
        raw_ids = [1]
    # Single-class mapping: take first unique id as the only class
    single_class_id = raw_ids[0]
    cat_name_map = {int(c['id']): c.get('name', 'tooltip') for c in train_coco.get('categories', [])}
    class_name = cat_name_map.get(single_class_id, 'tooltip')

    # Build HF config with 1 label
    id2label = {0: class_name}
    label2id = {class_name: 0}
    config = DetrConfig.from_pretrained(
        args.model_checkpoint,
        num_labels=1,
        id2label=id2label,
        label2id=label2id,
    )
    config.num_queries = args.num_queries

    # Create datasets (dataset internally remaps any category_id to 0)
    train_ds = CocoSingleClassDataset(args.images_dir, args.train_json, processor)
    val_ds = CocoSingleClassDataset(args.images_dir, args.val_json, processor) if args.val_json else None

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model
    model = DetrForObjectDetection.from_pretrained(args.model_checkpoint, config=config, ignore_mismatched_sizes=True).to(device)

    # Optimizer with lower LR on backbone
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())

    best_val = math.inf
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        print(f"  train_loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss = eval_loss(model, val_loader, device)
            print(f"  val_loss:   {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / 'best_model_state.pth')
            # COCO mAP (every N epochs)
            if args.coco_eval_every > 0 and ((epoch + 1) % args.coco_eval_every == 0):
                # choose category id from val json
                coco_metrics = eval_coco_map(model, val_loader, processor, device, args.val_json)
                if coco_metrics:
                    print(f"  COCO mAP:  mAP@0.5:0.95={coco_metrics['mAP_0.5:0.95']:.4f}  mAP@0.5={coco_metrics['mAP_0.5']:.4f}  AR@100={coco_metrics['AR@100']:.4f}")
        # Save last
        torch.save(model.state_dict(), out_dir / 'last_model_state.pth')

    # Persist label mapping
    with open(out_dir / 'label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'id2label': id2label, 'label2id': label2id}, f)
    print("Training complete. Saved last/best model states and label mapping.")


if __name__ == '__main__':
    main()
