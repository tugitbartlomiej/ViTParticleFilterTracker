"""
Generate a deterministic video-level train/val/test split.

This ensures ALL ablation variants use the same split,
and augmented copies stay with their source video.

Output: video_split.json
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Defaults
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def parse_video_stem(filename: str) -> str:
    """Extract video stem from filename: test01_frame_000123.jpg -> test01"""
    if '_frame_' in filename:
        return filename.split('_frame_')[0]
    return filename.rsplit('.', 1)[0]


def generate_split(images_dir: Path, seed: int = SEED) -> dict:
    """Group images by video, assign videos to splits."""
    exts = {'.jpg', '.jpeg', '.png'}
    groups = defaultdict(list)

    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in exts:
            stem = parse_video_stem(p.name)
            groups[stem].append(p.name)

    video_stems = sorted(groups.keys())
    random.seed(seed)
    random.shuffle(video_stems)

    n = len(video_stems)
    n_train = int(round(TRAIN_RATIO * n))
    n_val = int(round(VAL_RATIO * n))

    train_videos = set(video_stems[:n_train])
    val_videos = set(video_stems[n_train:n_train + n_val])
    test_videos = set(video_stems[n_train + n_val:])

    split_map = {}
    for vs in video_stems:
        if vs in train_videos:
            split_map[vs] = 'train'
        elif vs in val_videos:
            split_map[vs] = 'val'
        else:
            split_map[vs] = 'test'

    # Build image-level split
    train_images, val_images, test_images = [], [], []
    for vs, files in groups.items():
        target = {'train': train_images, 'val': val_images, 'test': test_images}[split_map[vs]]
        target.extend(files)

    result = {
        'seed': seed,
        'ratios': {'train': TRAIN_RATIO, 'val': VAL_RATIO, 'test': TEST_RATIO},
        'video_split': split_map,
        'counts': {
            'videos': {'train': len(train_videos), 'val': len(val_videos), 'test': len(test_videos), 'total': n},
            'images': {'train': len(train_images), 'val': len(val_images), 'test': len(test_images),
                       'total': len(train_images) + len(val_images) + len(test_images)},
        },
        'train_videos': sorted(train_videos),
        'val_videos': sorted(val_videos),
        'test_videos': sorted(test_videos),
        'train_images': sorted(train_images),
        'val_images': sorted(val_images),
        'test_images': sorted(test_images),
    }
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Generate deterministic video-level split')
    ap.add_argument('--images-dir', type=str, required=True,
                    help='Directory with all images (e.g. DETR_augmented_dataset_20250218/images)')
    ap.add_argument('--output', type=str, default='video_split.json')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    result = generate_split(images_dir, seed=args.seed)

    out_path = Path(args.output)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    c = result['counts']
    print(f"Video-level split generated (seed={args.seed}):")
    print(f"  Videos: {c['videos']['train']} train / {c['videos']['val']} val / {c['videos']['test']} test")
    print(f"  Images: {c['images']['train']} train / {c['images']['val']} val / {c['images']['test']} test")
    print(f"  Saved to: {out_path}")


if __name__ == '__main__':
    main()
