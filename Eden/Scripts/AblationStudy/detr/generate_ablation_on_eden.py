#!/usr/bin/env python3
"""
Generate 5 ablation dataset variants directly on Eden.

Works entirely with:
  - augmented_coco_20250417_030014.json (91k image metadata)
  - feature_cache2_weighted/ (DINO, Fourier, SAM, EL2N — pre-computed)
  - NO access to actual image files needed (cache-only selection)

Output: 5 × {annotations_train.json, annotations_val.json, annotations_test.json}
        + video_split.json + selected_filenames_{v1..v5}.txt

Usage on Eden:
  python generate_ablation_on_eden.py \
    --annotations ~/datasets_20250606_test/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json \
    --cache ~/DETR/ablation/feature_cache2_weighted \
    --output ~/DETR/ablation
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import cdist

SEED = 42
TARGET_SIZE = 20000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# VIDEO-LEVEL SPLIT
# ============================================================================

def parse_video_stem(filename: str) -> str:
    if '_frame_' in filename:
        return filename.split('_frame_')[0]
    return filename.rsplit('.', 1)[0]


def generate_video_split(coco: dict, seed: int = SEED) -> dict:
    """Generate deterministic video-level split from COCO annotations."""
    groups = defaultdict(list)
    for img in coco['images']:
        stem = parse_video_stem(img['file_name'])
        groups[stem].append(img['file_name'])

    video_stems = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(video_stems)

    n = len(video_stems)
    n_train = int(round(TRAIN_RATIO * n))
    n_val = int(round(VAL_RATIO * n))

    split_map = {}
    train_videos, val_videos, test_videos = [], [], []
    for i, vs in enumerate(video_stems):
        if i < n_train:
            split_map[vs] = 'train'
            train_videos.append(vs)
        elif i < n_train + n_val:
            split_map[vs] = 'val'
            val_videos.append(vs)
        else:
            split_map[vs] = 'test'
            test_videos.append(vs)

    return {
        'seed': seed,
        'video_split': split_map,
        'train_videos': sorted(train_videos),
        'val_videos': sorted(val_videos),
        'test_videos': sorted(test_videos),
        'counts': {
            'videos': {'train': len(train_videos), 'val': len(val_videos),
                       'test': len(test_videos), 'total': n},
        }
    }


# ============================================================================
# COCO HELPERS
# ============================================================================

def get_images_by_split(coco: dict, split_data: dict, target_split: str) -> List[dict]:
    vm = split_data['video_split']
    return [img for img in coco['images']
            if vm.get(parse_video_stem(img['file_name'])) == target_split]


def build_coco_subset(coco: dict, image_ids: Set[int]) -> dict:
    return {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': [img for img in coco['images'] if img['id'] in image_ids],
        'annotations': [ann for ann in coco['annotations'] if ann['image_id'] in image_ids],
    }


def filenames_to_coco(coco: dict, filenames: Set[str]) -> dict:
    fn_set = {f.lower() for f in filenames}
    ids = set()
    imgs = []
    for img in coco['images']:
        if img['file_name'].lower() in fn_set:
            imgs.append(img)
            ids.add(img['id'])
    return {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': imgs,
        'annotations': [a for a in coco['annotations'] if a['image_id'] in ids],
    }


# ============================================================================
# FEATURE CACHE LOADING
# ============================================================================

def load_cache(cache_dir: Path):
    """Load pre-computed features. Returns (features_dict, path_list)."""
    print("Loading feature cache...")

    with open(cache_dir / 'fourier_features.pkl', 'rb') as f:
        fourier_features, cached_paths = pickle.load(f)
    # Normalize paths: extract just filenames
    filenames = [os.path.basename(p.replace('\\', '/')) for p in cached_paths]

    with open(cache_dir / 'dino_features.pkl', 'rb') as f:
        dino_features = pickle.load(f)

    with open(cache_dir / 'sam_scores.pkl', 'rb') as f:
        sam_scores = pickle.load(f)

    with open(cache_dir / 'el2n_scores.pkl', 'rb') as f:
        el2n_scores = pickle.load(f)

    n = len(filenames)
    print(f"  Loaded {n} entries")
    print(f"  DINO: {dino_features.shape}, Fourier: {fourier_features.shape}")
    print(f"  SAM: {sam_scores.shape}, EL2N: {el2n_scores.shape}")

    return {
        'filenames': filenames,
        'dino': dino_features,
        'fourier': fourier_features,
        'sam': sam_scores,
        'el2n': el2n_scores,
    }


def get_train_mask(filenames: List[str], split_data: dict) -> np.ndarray:
    """Return boolean mask for train-split images."""
    vm = split_data['video_split']
    mask = np.array([vm.get(parse_video_stem(fn)) == 'train' for fn in filenames])
    return mask


# ============================================================================
# SELECTION ALGORITHMS (cache-only, no image access)
# ============================================================================

def normalize_features(X: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1
    return (X - mins) / rng


def weighted_select(features: dict, train_mask: np.ndarray,
                    weights: dict, target: int, seed: int = SEED) -> List[int]:
    """
    Select indices using weighted feature combination + K-Center Greedy.
    Operates purely on cached features — no image loading.
    """
    n = train_mask.sum()
    train_idx = np.where(train_mask)[0]

    # Build combined feature matrix
    components = []
    if weights.get('dino', 0) > 0:
        # PCA reduce DINO to 32 dims
        from sklearn.decomposition import PCA
        dino_train = features['dino'][train_idx]
        pca = PCA(n_components=min(32, dino_train.shape[1]), random_state=seed)
        dino_pca = pca.fit_transform(dino_train)
        components.append(normalize_features(dino_pca) * weights['dino'])

    if weights.get('fourier', 0) > 0:
        four_train = features['fourier'][train_idx]
        components.append(normalize_features(four_train) * weights['fourier'])

    if weights.get('sam', 0) > 0:
        sam_train = features['sam'][train_idx].reshape(-1, 1)
        components.append(normalize_features(sam_train) * weights['sam'])

    if weights.get('el2n', 0) > 0:
        el2n_train = features['el2n'][train_idx].reshape(-1, 1)
        components.append(normalize_features(el2n_train) * weights['el2n'])

    if not components:
        # Fallback: random
        rng = random.Random(seed)
        sel = rng.sample(range(n), min(target, n))
        return train_idx[sel].tolist()

    combined = np.hstack(components)

    # K-Center Greedy selection
    print(f"    K-Center Greedy on {combined.shape} matrix, target={target}...")
    selected_local = k_center_greedy(combined, target, seed)
    return train_idx[selected_local].tolist()


def k_center_greedy(X: np.ndarray, target: int, seed: int = SEED) -> List[int]:
    """K-Center Greedy: select diverse subset maximizing min distance."""
    n = X.shape[0]
    if n <= target:
        return list(range(n))

    rng = random.Random(seed)
    selected = [rng.randint(0, n - 1)]
    # Distance from each point to nearest selected
    min_dist = np.full(n, np.inf)

    for _ in range(target - 1):
        last = selected[-1]
        dist_to_last = np.linalg.norm(X - X[last], axis=1)
        min_dist = np.minimum(min_dist, dist_to_last)
        # Don't re-select
        min_dist[selected] = -1
        next_idx = np.argmax(min_dist)
        selected.append(int(next_idx))

        if len(selected) % 5000 == 0:
            print(f"      Selected {len(selected)}/{target}")

    return selected


# ============================================================================
# VARIANT GENERATORS
# ============================================================================

def variant_full(features, train_mask, target, seed=SEED):
    """V1: FULL — DINO 0.35 + Fourier 0.15 + SAM 0.20 + EL2N 0.30"""
    return weighted_select(features, train_mask,
                           {'dino': 0.35, 'fourier': 0.15, 'sam': 0.20, 'el2n': 0.30},
                           target, seed)


def variant_no_fourier(features, train_mask, target, seed=SEED):
    """V2: NO_FOURIER — DINO 0.45 + SAM 0.25 + EL2N 0.30"""
    return weighted_select(features, train_mask,
                           {'dino': 0.45, 'fourier': 0.0, 'sam': 0.25, 'el2n': 0.30},
                           target, seed)


def variant_no_el2n(features, train_mask, target, seed=SEED):
    """V3: NO_EL2N — DINO 0.45 + Fourier 0.20 + SAM 0.35"""
    return weighted_select(features, train_mask,
                           {'dino': 0.45, 'fourier': 0.20, 'sam': 0.35, 'el2n': 0.0},
                           target, seed)


def variant_quality_only(features, train_mask, target, seed=SEED):
    """V4: QUALITY_ONLY — Fourier quality filter + random sample."""
    train_idx = np.where(train_mask)[0]
    fourier_train = features['fourier'][train_idx]

    # High/Low Ratio = high_band / low_band energy
    high_energy = fourier_train[:, 2] if fourier_train.shape[1] > 2 else fourier_train[:, -1]
    low_energy = fourier_train[:, 0]
    ratio = np.where(low_energy > 0, high_energy / low_energy, 0)
    median_ratio = np.median(ratio)

    # Keep images above median quality
    quality_mask = ratio >= median_ratio
    quality_idx = np.where(quality_mask)[0]

    rng = random.Random(seed)
    if len(quality_idx) > target:
        sel = rng.sample(range(len(quality_idx)), target)
        return train_idx[quality_idx[sel]].tolist()
    return train_idx[quality_idx].tolist()


def variant_random(features, train_mask, target, seed=SEED):
    """V5: RANDOM — pure random 20k from train pool."""
    train_idx = np.where(train_mask)[0]
    rng = random.Random(seed)
    if len(train_idx) > target:
        sel = rng.sample(range(len(train_idx)), target)
        return train_idx[sel].tolist()
    return train_idx.tolist()


def variant_diversity_only(features, train_mask, target, seed=SEED):
    """V6: DIVERSITY_ONLY — DINO + K-Center Greedy, no Fourier/EL2N."""
    return weighted_select(features, train_mask,
                           {'dino': 1.0, 'fourier': 0.0, 'sam': 0.0, 'el2n': 0.0},
                           target, seed)


def variant_el2n_only(features, train_mask, target, seed=SEED):
    """V7: EL2N_ONLY — random 2x pool, top-k hardest by EL2N."""
    train_idx = np.where(train_mask)[0]
    el2n_train = features['el2n'][train_idx]

    # Random 2x pool
    rng = random.Random(seed)
    pool_size = min(target * 2, len(train_idx))
    pool_local = rng.sample(range(len(train_idx)), pool_size)

    # Sort by EL2N (hardest first) and take top-k
    pool_sorted = sorted(pool_local, key=lambda i: el2n_train[i], reverse=True)
    selected_local = pool_sorted[:target]

    return train_idx[selected_local].tolist()


# ============================================================================
# MAIN
# ============================================================================

GENERATORS = {
    'v1': ('FULL (DINO+Fourier+SAM+EL2N)', variant_full),
    'v2': ('NO_FOURIER (DINO+SAM+EL2N)', variant_no_fourier),
    'v3': ('NO_EL2N (DINO+Fourier+SAM)', variant_no_el2n),
    'v4': ('QUALITY_ONLY (Fourier+random)', variant_quality_only),
    'v5': ('RANDOM (pure random)', variant_random),
    'v6': ('DIVERSITY_ONLY (DINO+K-Center)', variant_diversity_only),
    'v7': ('EL2N_ONLY (random+EL2N rerank)', variant_el2n_only),
}


def main():
    ap = argparse.ArgumentParser(description='Generate ablation variants on Eden')
    ap.add_argument('--annotations', required=True,
                    help='augmented_coco_20250417_030014.json (91k entries)')
    ap.add_argument('--cache', required=True,
                    help='feature_cache2_weighted/ directory')
    ap.add_argument('--output', default='./ablation',
                    help='Output directory for annotation JSONs')
    ap.add_argument('--target', type=int, default=TARGET_SIZE)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--variants', nargs='+', default=list(GENERATORS.keys()))
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    print(f"Loading annotations: {args.annotations}")
    with open(args.annotations, 'r') as f:
        coco = json.load(f)
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    # Generate video-level split
    print("\nGenerating video-level split...")
    split_data = generate_video_split(coco, seed=args.seed)
    split_path = output_dir / 'video_split.json'
    with open(split_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    cv = split_data['counts']['videos']
    print(f"  Videos: {cv['train']} train / {cv['val']} val / {cv['test']} test")

    # Count images per split
    train_imgs = get_images_by_split(coco, split_data, 'train')
    val_imgs = get_images_by_split(coco, split_data, 'val')
    test_imgs = get_images_by_split(coco, split_data, 'test')
    print(f"  Images: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

    # Build shared val/test COCO JSONs
    val_ids = {img['id'] for img in val_imgs}
    test_ids = {img['id'] for img in test_imgs}
    val_coco = build_coco_subset(coco, val_ids)
    test_coco = build_coco_subset(coco, test_ids)

    # Load feature cache
    features = load_cache(Path(args.cache))
    filenames = features['filenames']
    train_mask = get_train_mask(filenames, split_data)
    print(f"  Train images in cache: {train_mask.sum()}/{len(filenames)}")

    # Generate each variant
    for vid in args.variants:
        vid = vid.lower()
        if vid not in GENERATORS:
            print(f"Unknown variant: {vid}")
            continue

        name, gen_fn = GENERATORS[vid]
        print(f"\n{'='*60}")
        print(f"  {vid.upper()}: {name}")
        print(f"{'='*60}")

        selected_idx = gen_fn(features, train_mask, args.target, args.seed)
        selected_filenames = {filenames[i] for i in selected_idx}

        # Build train COCO
        train_coco = filenames_to_coco(coco, selected_filenames)

        # Save
        vdir = output_dir / vid
        vdir.mkdir(parents=True, exist_ok=True)

        for split_name, split_coco in [('train', train_coco), ('val', val_coco), ('test', test_coco)]:
            p = vdir / f'annotations_{split_name}.json'
            with open(p, 'w') as f:
                json.dump(split_coco, f, indent=1)
            print(f"    {split_name}: {len(split_coco['images'])} imgs, "
                  f"{len(split_coco['annotations'])} anns -> {p.name}")

        # Save selected filenames for tar extraction
        fn_path = vdir / 'selected_train_filenames.txt'
        with open(fn_path, 'w') as f:
            for fn in sorted(selected_filenames):
                f.write(fn + '\n')

        # Metadata
        meta = {
            'variant': vid, 'name': name, 'seed': args.seed,
            'train_images': len(train_coco['images']),
            'val_images': len(val_coco['images']),
            'test_images': len(test_coco['images']),
            'train_annotations': len(train_coco['annotations']),
        }
        with open(vdir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"All variants saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nTotal disk usage (annotation JSONs only):")
    total = 0
    for vid in args.variants:
        vdir = output_dir / vid
        sz = sum(f.stat().st_size for f in vdir.glob('*.json'))
        total += sz
        print(f"  {vid}: {sz/1024/1024:.1f} MB")
    print(f"  TOTAL: {total/1024/1024:.1f} MB")


if __name__ == '__main__':
    main()
