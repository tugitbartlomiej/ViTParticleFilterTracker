import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
except Exception as e:  # pragma: no cover
    COCO = None
    COCOeval = None


class InferenceCocoImages(Dataset):
    def __init__(self, images_dir: Path, coco_images: List[Dict], processor: DetrImageProcessor):
        self.images_dir = images_dir
        self.items = []
        for im in coco_images:
            self.items.append({
                'image_id': int(im['id']),
                'file_name': im['file_name'],
                'width': int(im.get('width', 0)),
                'height': int(im.get('height', 0)),
            })
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = self.images_dir / it['file_name']
        image = Image.open(img_path).convert('RGB')
        if it['width'] <= 0 or it['height'] <= 0:
            it['width'], it['height'] = image.size
        enc = self.processor(images=image, return_tensors='pt')
        return {
            'pixel_values': enc['pixel_values'].squeeze(0),
            'pixel_mask': enc['pixel_mask'].squeeze(0),
            'orig_size': (it['height'], it['width']),  # (h,w)
            'image_id': it['image_id'],
            'file_name': it['file_name'],
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    pixel_values = torch.stack([b['pixel_values'] for b in batch])
    pixel_mask = torch.stack([b['pixel_mask'] for b in batch])
    orig_sizes = [b['orig_size'] for b in batch]
    image_ids = [b['image_id'] for b in batch]
    file_names = [b['file_name'] for b in batch]
    return {
        'pixel_values': pixel_values,
        'pixel_mask': pixel_mask,
        'orig_sizes': orig_sizes,
        'image_ids': image_ids,
        'file_names': file_names,
    }


def main():
    ap = argparse.ArgumentParser(description='Offline COCO evaluation for DETR (single-class tooltip detection)')
    ap.add_argument('--images_dir', required=True, help='Directory with images')
    ap.add_argument('--test_json', required=True, help='COCO annotations (test)')
    ap.add_argument('--model_checkpoint', default='facebook/detr-resnet-50', help='Base checkpoint')
    ap.add_argument('--weights_path', required=True, help='Path to model.state_dict() (best/last)')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--conf_threshold', type=float, default=0.0, help='Score threshold before exporting detections')
    ap.add_argument('--device', default=None, help='cuda or cpu (auto if None)')
    ap.add_argument('--category_id', type=int, default=None, help='Raw COCO category_id to emit (defaults to first in test JSON)')
    args = ap.parse_args()

    if COCO is None or COCOeval is None:
        raise SystemExit('pycocotools not available. Install: pip install pycocotools (or pycocotools-windows)')

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images_dir = Path(args.images_dir)
    with open(args.test_json, 'r', encoding='utf-8') as f:
        coco_test = json.load(f)

    # Define single-class mapping (id2label/label2id) for model
    cats = coco_test.get('categories', [])
    if args.category_id is None:
        if cats:
            raw_cid = int(cats[0]['id'])
            class_name = str(cats[0].get('name', 'tooltip'))
        else:
            raw_cid = 1
            class_name = 'tooltip'
    else:
        raw_cid = int(args.category_id)
        class_name = cats[0].get('name', 'tooltip') if cats else 'tooltip'

    id2label = {0: class_name}
    label2id = {class_name: 0}

    processor = DetrImageProcessor.from_pretrained(args.model_checkpoint)
    config = DetrConfig.from_pretrained(args.model_checkpoint, num_labels=1, id2label=id2label, label2id=label2id)
    model = DetrForObjectDetection.from_pretrained(args.model_checkpoint, config=config, ignore_mismatched_sizes=True).to(device)

    # Load weights (state dict saved from training script)
    state = torch.load(args.weights_path, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = InferenceCocoImages(images_dir, coco_test.get('images', []), processor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    predictions = []
    with torch.no_grad():
        for batch in dl:
            if batch is None:
                continue
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.tensor([[h, w] for (h, w) in batch['orig_sizes']], device=device)
            post = processor.post_process_object_detection(outputs, threshold=args.conf_threshold, target_sizes=target_sizes)
            for p, img_id in zip(post, batch['image_ids']):
                boxes = p['boxes'].detach().cpu().numpy().tolist()
                scores = p['scores'].detach().cpu().numpy().tolist()
                # labels are all 0 for single-class; map to raw category id
                for (x1, y1, x2, y2), s in zip(boxes, scores):
                    x = float(x1)
                    y = float(y1)
                    bw = float(max(0.0, x2 - x1))
                    bh = float(max(0.0, y2 - y1))
                    predictions.append({
                        'image_id': int(img_id),
                        'category_id': int(raw_cid),
                        'bbox': [x, y, bw, bh],
                        'score': float(s),
                    })

    coco_gt = COCO(args.test_json)
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print compact summary
    print("Summary:")
    print(f"mAP@0.5:0.95 = {coco_eval.stats[0]:.4f}")
    print(f"mAP@0.5      = {coco_eval.stats[1]:.4f}")
    print(f"mAP@0.75     = {coco_eval.stats[2]:.4f}")
    print(f"AR@100       = {coco_eval.stats[8]:.4f}")


if __name__ == '__main__':
    main()
