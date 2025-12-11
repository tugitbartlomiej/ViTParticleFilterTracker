import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def parse_video_stem(filename: str) -> str:
    # Expect pattern: <video_stem>_frame_00001234.jpg
    if '_frame_' in filename:
        return filename.split('_frame_')[0]
    return filename.rsplit('.', 1)[0]


def index_preds_by_image(coco_preds: Dict) -> Dict[str, List[Dict]]:
    # Build map: file_name -> detections
    preds_by_name: Dict[str, List[Dict]] = defaultdict(list)
    img_id_to_name = {}
    for img in coco_preds.get('images', []):
        img_id_to_name[img['id']] = img['file_name']
    for ann in coco_preds.get('annotations', []):
        name = img_id_to_name.get(ann['image_id'])
        if name:
            preds_by_name[name].append(ann)
    return preds_by_name


def list_images(root: Path) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    return sorted([p for p in root.rglob('*') if p.suffix in exts])


def build_coco(images: List[Path], preds_by_name: Dict[str, List[Dict]], category_id: int, category_name: str) -> Dict:
    data = {
        'info': {'description': 'Cataract DETR dataset', 'version': '1.0'},
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': 'Unknown'}],
        'categories': [{'id': int(category_id), 'name': category_name, 'supercategory': 'none'}],
        'images': [],
        'annotations': [],
    }
    img_id = 1
    ann_id = 1
    for ip in images:
        with Image.open(ip) as im:
            w, h = im.size
        data['images'].append({'id': img_id, 'file_name': ip.name, 'width': w, 'height': h, 'license': 1})
        for ann in preds_by_name.get(ip.name, []):
            # ensure integer category_id
            cid = int(ann.get('category_id', category_id))
            x, y, w_box, h_box = ann['bbox']
            data['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cid,
                'bbox': [float(x), float(y), float(w_box), float(h_box)],
                'area': float(w_box) * float(h_box),
                'iscrowd': 0,
            })
            ann_id += 1
        img_id += 1
    return data


def split_grouped_by_video(images: List[Path], ratios: Tuple[float, float, float]) -> Tuple[List[Path], List[Path], List[Path]]:
    train_r, val_r, test_r = ratios
    groups: Dict[str, List[Path]] = defaultdict(list)
    for ip in images:
        groups[parse_video_stem(ip.name)].append(ip)
    video_stems = list(groups.keys())
    random.shuffle(video_stems)
    n = len(video_stems)
    n_train = int(round(train_r * n))
    n_val = int(round(val_r * n))
    n_test = max(0, n - n_train - n_val)
    train_vs = set(video_stems[:n_train])
    val_vs = set(video_stems[n_train:n_train + n_val])
    test_vs = set(video_stems[n_train + n_val: n_train + n_val + n_test])
    train_imgs, val_imgs, test_imgs = [], [], []
    for vs, lst in groups.items():
        if vs in train_vs:
            train_imgs.extend(lst)
        elif vs in val_vs:
            val_imgs.extend(lst)
        else:
            test_imgs.extend(lst)
    return train_imgs, val_imgs, test_imgs


def cap_negatives(images: List[Path], preds_by_name: Dict[str, List[Dict]], max_ratio: float) -> List[Path]:
    positives = [ip for ip in images if len(preds_by_name.get(ip.name, [])) > 0]
    negatives = [ip for ip in images if len(preds_by_name.get(ip.name, [])) == 0]
    if not negatives:
        return images
    max_negs = int(max_ratio * max(1, len(positives)))
    if len(negatives) > max_negs:
        random.shuffle(negatives)
        negatives = negatives[:max_negs]
    return positives + negatives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Directory with selected frames')
    ap.add_argument('--preds', required=True, help='COCO JSON with predictions (from YOLO pseudo-labels)')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--category-id', type=int, default=1)
    ap.add_argument('--category-name', type=str, default='tooltip')
    ap.add_argument('--train', type=float, default=0.7)
    ap.add_argument('--val', type=float, default=0.15)
    ap.add_argument('--test', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--include-negatives', action='store_true')
    ap.add_argument('--neg-ratio', type=float, default=0.3)
    args = ap.parse_args()

    random.seed(args.seed)
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.preds, 'r', encoding='utf-8') as f:
        preds_coco = json.load(f)
    preds_by_name = index_preds_by_image(preds_coco)

    all_images = list_images(images_dir)
    train_imgs, val_imgs, test_imgs = split_grouped_by_video(all_images, (args.train, args.val, args.test))

    if args.include_negatives:
        train_imgs = cap_negatives(train_imgs, preds_by_name, args.neg_ratio)
        val_imgs = cap_negatives(val_imgs, preds_by_name, args.neg_ratio)
        test_imgs = cap_negatives(test_imgs, preds_by_name, args.neg_ratio)

    train_coco = build_coco(train_imgs, preds_by_name, args.category_id, args.category_name)
    val_coco = build_coco(val_imgs, preds_by_name, args.category_id, args.category_name)
    test_coco = build_coco(test_imgs, preds_by_name, args.category_id, args.category_name)

    with open(out_dir / 'annotations_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_coco, f)
    with open(out_dir / 'annotations_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_coco, f)
    with open(out_dir / 'annotations_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_coco, f)

    print(f"Saved splits: train={len(train_coco['images'])}, val={len(val_coco['images'])}, test={len(test_coco['images'])}")


if __name__ == '__main__':
    main()
