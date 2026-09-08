#!/usr/bin/env python3
"""
COCO -> YOLO konwersja respektujaca VIDEO-LEVEL split.

Bierze GOTOWE, juz podzielone pliki COCO (annotations_train.json / annotations_val.json
z prepare_fullpool_videosplit.py) i buduje dataset YOLO:
    <out>/images/{train,val}/   (symlink|copy do obrazow z poolu)
    <out>/labels/{train,val}/   (YOLO txt: "cls xc yc w h", znormalizowane 0..1)
    <out>/dataset.yaml

Split per-wideo jest JUZ zaszyty w annotations_train/val.json -> ten skrypt TYLKO routuje
obrazy do train/val i konwertuje bbox. ZERO losowego shuffle (inaczej niz stary inline
konwerter w run_yolo_20k_from_scratch.slurm, ktory robil random 90/10!).

Konwersja bbox: COCO [x,y,w,h] (lewy-gorny rog, piksele) -> YOLO [xc,yc,w,h] / rozmiar (0..1).
category_id (int LUB float, np. 0.0) -> indeks klasy YOLO wg kolejnosci w 'categories'.

Uzycie (na Edenie, w slurm):
  python3 coco_to_yolo_videosplit.py \
    --train-coco ~/DETR/fullpool_videosplit/annotations_train.json \
    --val-coco   ~/DETR/fullpool_videosplit/annotations_val.json \
    --pool-images $TMPDIR/pool_images \
    --output $TMPDIR/yolo_dataset \
    --link-mode symlink

Test/dry-run (bez obrazow, tylko etykiety, ograniczone do N):
  python3 coco_to_yolo_videosplit.py --train-coco ... --val-coco ... \
    --pool-images /dev/null --output /tmp/test --link-mode none --limit 100
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x,y,w,h] (px, lewy-gorny rog) -> YOLO (xc,yc,w,h) znormalizowane [0,1]."""
    x, y, bw, bh = bbox
    xc = clamp01((x + bw / 2.0) / img_w)
    yc = clamp01((y + bh / 2.0) / img_h)
    wn = clamp01(bw / img_w)
    hn = clamp01(bh / img_h)
    return xc, yc, wn, hn


def build_category_map(coco):
    """category_id -> indeks YOLO (0..nc-1) wg kolejnosci w 'categories'. Zwraca (mapa, nazwy).

    Uwaga: dict {0: 0} obsluguje tez klucz float 0.0 (0 == 0.0, hash zgodny)."""
    cats = coco.get('categories') or [{'id': 0, 'name': 'tool'}]
    cat2yolo = {c['id']: i for i, c in enumerate(cats)}
    names = [str(c['name']) for c in cats]
    return cat2yolo, names


def image_label_lines(img_info, anns, cat2yolo):
    """Linie YOLO dla jednego obrazu (pusta lista = tlo / brak obiektow)."""
    w = img_info['width']
    h = img_info['height']
    lines = []
    for an in anns:
        xc, yc, wn, hn = coco_bbox_to_yolo(an['bbox'], w, h)
        cls = cat2yolo.get(an['category_id'], 0)
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return lines


def place_image(src: Path, dst: Path, link_mode: str) -> bool:
    """symlink|copy|none. Zwraca True jesli obraz fizycznie jest (lub link_mode=none)."""
    if link_mode == 'none':
        return True
    if not src.exists():
        return False
    if dst.exists():
        return True
    if link_mode == 'symlink':
        try:
            os.symlink(src, dst)
            return True
        except OSError:
            shutil.copy2(src, dst)
            return True
    if link_mode == 'copy':
        shutil.copy2(src, dst)
        return True
    return False


def convert_split(coco_path, pool_images_dir, out_dir, subset,
                  link_mode='symlink', limit=0):
    """Konwertuje jeden split (train/val). Zwraca dict ze statystykami + nazwy klas."""
    pool = Path(pool_images_dir)
    out = Path(out_dir)
    with open(coco_path, encoding='utf-8') as f:
        coco = json.load(f)

    cat2yolo, names = build_category_map(coco)

    labels_dir = out / 'labels' / subset
    images_dir = out / 'images' / subset
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    id2anns = {}
    for an in coco['annotations']:
        id2anns.setdefault(an['image_id'], []).append(an)

    images = coco['images']
    if limit and limit > 0:
        images = images[:limit]

    n_img = n_lines = n_missing = n_empty = 0
    for im in images:
        fn = im['file_name']
        stem = Path(fn).stem
        lines = image_label_lines(im, id2anns.get(im['id'], []), cat2yolo)
        with open(labels_dir / (stem + '.txt'), 'w') as f:
            if lines:
                f.write('\n'.join(lines) + '\n')
        n_lines += len(lines)
        if not lines:
            n_empty += 1
        if not place_image(pool / fn, images_dir / fn, link_mode):
            n_missing += 1
        n_img += 1

    return {
        'subset': subset, 'images': n_img, 'label_lines': n_lines,
        'empty_labels': n_empty, 'missing_images': n_missing, 'names': names,
    }


def write_dataset_yaml(out_dir, names):
    out = Path(out_dir)
    content = {
        'path': str(out),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: n for i, n in enumerate(names)},
        'nc': len(names),
    }
    yaml_path = out / 'dataset.yaml'
    try:
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    except Exception:
        # fallback bez pyyaml (np. lokalny test)
        with open(yaml_path, 'w') as f:
            f.write(f"path: {out}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(names)}\n")
            f.write("names:\n")
            for i, n in enumerate(names):
                f.write(f"  {i}: {n}\n")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description="COCO -> YOLO z video-level splitem")
    ap.add_argument('--train-coco', required=True)
    ap.add_argument('--val-coco', required=True)
    ap.add_argument('--pool-images', required=True, help='Katalog z obrazami poolu')
    ap.add_argument('--output', required=True, help='Katalog wyjsciowy dataset YOLO')
    ap.add_argument('--link-mode', choices=['symlink', 'copy', 'none'], default='symlink')
    ap.add_argument('--limit', type=int, default=0, help='Tylko pierwsze N obrazow / split (test)')
    args = ap.parse_args()

    stats = []
    for subset, coco in (('train', args.train_coco), ('val', args.val_coco)):
        s = convert_split(coco, args.pool_images, args.output, subset,
                          link_mode=args.link_mode, limit=args.limit)
        stats.append(s)
        print(f"  {subset}: images={s['images']} label_lines={s['label_lines']} "
              f"empty={s['empty_labels']} missing_images={s['missing_images']}")

    names = stats[0]['names']
    yaml_path = write_dataset_yaml(args.output, names)
    print(f"dataset.yaml -> {yaml_path}  (nc={len(names)}, names={names})")

    # sanity: brak nakladania sie obrazow train/val (gwarancja video-splitu)
    tr = {im['file_name'] for im in json.load(open(args.train_coco, encoding='utf-8'))['images']}
    va = {im['file_name'] for im in json.load(open(args.val_coco, encoding='utf-8'))['images']}
    overlap = tr & va
    print(f"overlap obrazow train/val: {len(overlap)} (musi byc 0)")
    if overlap:
        raise SystemExit(f"ERROR: nakladanie train/val: {list(overlap)[:5]}")
    print("OK: VIDEO-LEVEL split respektowany (train/val z osobnych annotations_*.json, BEZ shuffle).")


if __name__ == '__main__':
    main()
