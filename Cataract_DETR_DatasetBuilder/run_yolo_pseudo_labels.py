import argparse
import json
from pathlib import Path
from typing import List, Dict

from PIL import Image

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None


def list_images(root: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    return sorted([p for p in root.rglob('*') if p.suffix in exts])


def build_coco(images: List[Path], preds: Dict[str, List[Dict]], category_id: int, category_name: str) -> Dict:
    coco = {
        'info': {
            'description': 'Pseudo-labeled cataract dataset (YOLO) for DETR',
            'version': '1.0',
        },
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': 'Unknown'}],
        'categories': [{'id': int(category_id), 'name': category_name, 'supercategory': 'none'}],
        'images': [],
        'annotations': [],
    }
    img_id_map = {}
    ann_id = 1
    img_id = 1
    for img_path in images:
        with Image.open(img_path) as im:
            w, h = im.size
        coco['images'].append({
            'id': img_id,
            'file_name': img_path.name,
            'width': w,
            'height': h,
            'license': 1,
        })
        img_id_map[img_path.name] = img_id
        for det in preds.get(img_path.name, []):
            # det bbox is xyxy
            x1, y1, x2, y2, conf = det['x1'], det['y1'], det['x2'], det['y2'], det['conf']
            x = float(x1)
            y = float(y1)
            w_box = float(max(0.0, x2 - x1))
            h_box = float(max(0.0, y2 - y1))
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(category_id),
                'bbox': [x, y, w_box, h_box],
                'area': float(w_box * h_box),
                'iscrowd': 0,
                'score': float(conf),
            })
            ann_id += 1
        img_id += 1
    return coco


def run_inference(images_dir: Path, weights_path: Path, conf: float) -> Dict[str, List[Dict]]:
    if YOLO is None:
        raise RuntimeError('ultralytics is not installed. Install with: pip install ultralytics')
    model = YOLO(str(weights_path))
    imgs = list_images(images_dir)
    results_map: Dict[str, List[Dict]] = {}
    # Process in batches to reduce memory use
    batch_size = 32
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i+batch_size]
        res = model.predict(batch, conf=conf, verbose=False)
        for img_path, r in zip(batch, res):
            dets = []
            if r and r.boxes is not None:
                for b in r.boxes:
                    xyxy = b.xyxy.cpu().numpy().reshape(-1)
                    confv = float(b.conf.cpu().numpy().reshape(-1)[0])
                    dets.append({'x1': float(xyxy[0]), 'y1': float(xyxy[1]), 'x2': float(xyxy[2]), 'y2': float(xyxy[3]), 'conf': confv})
            results_map[img_path.name] = dets
    return results_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Directory with selected frames')
    ap.add_argument('--weights', required=True, help='YOLOv8 weights (.pt)')
    ap.add_argument('--out', required=True, help='Output COCO JSON path')
    ap.add_argument('--category-id', type=int, default=1)
    ap.add_argument('--category-name', type=str, default='tooltip')
    ap.add_argument('--conf', type=float, default=0.25)
    args = ap.parse_args()

    images_dir = Path(args.images)
    weights = Path(args.weights)
    out_json = Path(args.out)

    preds = run_inference(images_dir, weights, args.conf)
    imgs = list_images(images_dir)
    coco = build_coco(imgs, preds, args.category_id, args.category_name)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f)
    print(f"Saved pseudo-annotations: {out_json}")


if __name__ == '__main__':
    main()
