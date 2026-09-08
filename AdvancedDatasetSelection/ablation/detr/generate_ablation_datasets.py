"""
Generate 5 ablation dataset variants from the same image pool.

All variants:
- Use the same video-level split (video_split.json)
- Select 20,000 images from TRAIN videos only
- Output COCO annotations for train/val/test
- Package images into .tar archives

Variants:
  V1: Full pipeline (DINO 0.35, Fourier 0.15, SAM 0.20, EL2N 0.30)
  V2: No Fourier  (DINO 0.45, Fourier 0.00, SAM 0.25, EL2N 0.30)
  V3: No EL2N     (DINO 0.45, Fourier 0.20, SAM 0.35, EL2N 0.00)
  V4: Quality filter + random (Fourier filters, then random sample)
  V5: Pure random (no selection at all)

Usage:
  python generate_ablation_datasets.py \
    --images-dir ../Eden/Datasets/.../images \
    --annotations ../Eden/Datasets/.../augmented_coco_20250417_030014.json \
    --split video_split.json \
    --cache ../output/feature_cache2_weighted \
    --output ./output/ablation
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

TARGET_SIZE = 20000
SEED = 42


def load_split(split_path: str) -> dict:
    with open(split_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_coco(ann_path: str) -> dict:
    with open(ann_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_video_stem(filename: str) -> str:
    if '_frame_' in filename:
        return filename.split('_frame_')[0]
    return filename.rsplit('.', 1)[0]


def filter_images_by_split(coco: dict, split_data: dict, target_split: str) -> Tuple[List[dict], Set[int]]:
    """Return images belonging to target_split according to video_split."""
    video_map = split_data['video_split']
    filtered_images = []
    filtered_ids = set()
    for img in coco['images']:
        vs = parse_video_stem(img['file_name'])
        if video_map.get(vs) == target_split:
            filtered_images.append(img)
            filtered_ids.add(img['id'])
    return filtered_images, filtered_ids


def build_coco_subset(coco: dict, image_ids: Set[int]) -> dict:
    """Build COCO dict with only specified image IDs."""
    return {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': [img for img in coco['images'] if img['id'] in image_ids],
        'annotations': [ann for ann in coco['annotations'] if ann['image_id'] in image_ids],
    }


def build_coco_from_filenames(coco: dict, filenames: Set[str]) -> dict:
    """Build COCO dict with only images whose file_name is in filenames."""
    fn_lower = {f.lower() for f in filenames}
    image_ids = set()
    images = []
    for img in coco['images']:
        if img['file_name'].lower() in fn_lower:
            images.append(img)
            image_ids.add(img['id'])
    return {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': images,
        'annotations': [ann for ann in coco['annotations'] if ann['image_id'] in image_ids],
    }


# ============================================================================
# VARIANT GENERATORS
# ============================================================================

def variant_full_pipeline(train_images: List[dict], images_dir: Path,
                          cache_dir: Optional[Path], target: int) -> List[str]:
    """V1: Full pipeline — run cluster_selector with all features."""
    train_paths = [str(images_dir / img['file_name']) for img in train_images]
    train_paths = [p for p in train_paths if os.path.isfile(p)]

    from AdvancedDatasetSelection.selection_methods.cluster_selector import ClusterBasedSelector
    from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer
    from AdvancedDatasetSelection.feature_extractors.dino_extractor import DINOExtractor
    from AdvancedDatasetSelection.feature_extractors.sam_extractor import SAMExtractor

    selector = ClusterBasedSelector(
        fourier_analyzer=FourierAnalyzer(similarity_threshold=0.7),
        dino_extractor=DINOExtractor(model_name='dinov2_vitl14', device='cuda'),
        sam_extractor=SAMExtractor(device='cuda'),
        weights={'dino': 0.35, 'fourier': 0.15, 'sam': 0.20, 'el2n': 0.30},
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    _, selected_paths, stats = selector.select_optimal_subset(
        image_paths=train_paths, target_size=target,
        strategy='centroid', use_cache=True, normalize=True,
        dino_pca_dim=32, apply_weights=True,
    )
    print(f"  V1 Full Pipeline: selected {len(selected_paths)} from {len(train_paths)} train images")
    return selected_paths


def variant_no_fourier(train_images: List[dict], images_dir: Path,
                       cache_dir: Optional[Path], target: int) -> List[str]:
    """V2: No Fourier — zero weight on Fourier features."""
    train_paths = [str(images_dir / img['file_name']) for img in train_images]
    train_paths = [p for p in train_paths if os.path.isfile(p)]

    from AdvancedDatasetSelection.selection_methods.cluster_selector import ClusterBasedSelector
    from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer
    from AdvancedDatasetSelection.feature_extractors.dino_extractor import DINOExtractor
    from AdvancedDatasetSelection.feature_extractors.sam_extractor import SAMExtractor

    selector = ClusterBasedSelector(
        fourier_analyzer=FourierAnalyzer(similarity_threshold=0.7),
        dino_extractor=DINOExtractor(model_name='dinov2_vitl14', device='cuda'),
        sam_extractor=SAMExtractor(device='cuda'),
        weights={'dino': 0.45, 'fourier': 0.0, 'sam': 0.25, 'el2n': 0.30},
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    _, selected_paths, stats = selector.select_optimal_subset(
        image_paths=train_paths, target_size=target,
        strategy='centroid', use_cache=True, normalize=True,
        dino_pca_dim=32, apply_weights=True,
    )
    print(f"  V2 No Fourier: selected {len(selected_paths)} from {len(train_paths)} train images")
    return selected_paths


def variant_no_el2n(train_images: List[dict], images_dir: Path,
                    cache_dir: Optional[Path], target: int) -> List[str]:
    """V3: No EL2N — zero weight on difficulty scoring."""
    train_paths = [str(images_dir / img['file_name']) for img in train_images]
    train_paths = [p for p in train_paths if os.path.isfile(p)]

    from AdvancedDatasetSelection.selection_methods.cluster_selector import ClusterBasedSelector
    from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer
    from AdvancedDatasetSelection.feature_extractors.dino_extractor import DINOExtractor
    from AdvancedDatasetSelection.feature_extractors.sam_extractor import SAMExtractor

    selector = ClusterBasedSelector(
        fourier_analyzer=FourierAnalyzer(similarity_threshold=0.7),
        dino_extractor=DINOExtractor(model_name='dinov2_vitl14', device='cuda'),
        sam_extractor=SAMExtractor(device='cuda'),
        weights={'dino': 0.45, 'fourier': 0.20, 'sam': 0.35, 'el2n': 0.0},
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    _, selected_paths, stats = selector.select_optimal_subset(
        image_paths=train_paths, target_size=target,
        strategy='centroid', use_cache=True, normalize=True,
        dino_pca_dim=32, apply_weights=True,
    )
    print(f"  V3 No EL2N: selected {len(selected_paths)} from {len(train_paths)} train images")
    return selected_paths


def variant_no_clustering(train_images: List[dict], images_dir: Path,
                          cache_dir: Optional[Path], target: int) -> List[str]:
    """V4: Fourier quality filter + random sample (no clustering/diversity)."""
    train_paths = [str(images_dir / img['file_name']) for img in train_images]
    train_paths = [p for p in train_paths if os.path.isfile(p)]

    # Use Fourier to compute quality scores and filter low-quality
    from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer
    analyzer = FourierAnalyzer(similarity_threshold=0.7)
    features, valid_paths = analyzer.compute_features_batch(train_paths, show_progress=True)

    # Filter: keep images with High/Low Ratio > median (quality gate)
    if features is not None and len(features) > 0:
        # High/Low Ratio is feature index 2 (low, mid, high energy → ratio = high/low)
        high_energy = features[:, 2]  # high_band_energy
        low_energy = features[:, 0]   # low_band_energy
        ratio = np.where(low_energy > 0, high_energy / low_energy, 0)
        median_ratio = np.median(ratio)
        quality_mask = ratio >= median_ratio
        quality_paths = [p for p, m in zip(valid_paths, quality_mask) if m]
    else:
        quality_paths = valid_paths

    # Random sample from quality-filtered pool
    random.seed(SEED)
    if len(quality_paths) > target:
        selected = random.sample(quality_paths, target)
    else:
        selected = quality_paths

    print(f"  V4 No Clustering: {len(train_paths)} → {len(quality_paths)} (quality) → {len(selected)} (random)")
    return selected


def variant_random(train_images: List[dict], images_dir: Path, target: int) -> List[str]:
    """V5: Pure random — no selection pipeline at all."""
    train_paths = [str(images_dir / img['file_name']) for img in train_images]
    train_paths = [p for p in train_paths if os.path.isfile(p)]

    random.seed(SEED)
    if len(train_paths) > target:
        selected = random.sample(train_paths, target)
    else:
        selected = train_paths

    print(f"  V5 Random: selected {len(selected)} from {len(train_paths)} train images")
    return selected


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_variant(variant_id: str, selected_paths: List[str],
                 coco_full: dict, split_data: dict,
                 images_dir: Path, output_dir: Path):
    """Save annotations (train/val/test) and create image tar for one variant."""
    vdir = output_dir / variant_id
    vdir.mkdir(parents=True, exist_ok=True)

    # Train annotations: only selected images
    selected_filenames = {Path(p).name for p in selected_paths}
    train_coco = build_coco_from_filenames(coco_full, selected_filenames)

    # Val/Test annotations: ALL images from val/test videos (not subsetted)
    val_images, val_ids = filter_images_by_split(coco_full, split_data, 'val')
    test_images, test_ids = filter_images_by_split(coco_full, split_data, 'test')
    val_coco = build_coco_subset(coco_full, val_ids)
    test_coco = build_coco_subset(coco_full, test_ids)

    # Save annotation files
    for name, data in [('train', train_coco), ('val', val_coco), ('test', test_coco)]:
        path = vdir / f'annotations_{name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print(f"    {name}: {len(data['images'])} images, {len(data['annotations'])} annotations → {path.name}")

    # Create tar with train + val + test images
    all_filenames = (
        selected_filenames |
        {img['file_name'] for img in val_images} |
        {img['file_name'] for img in test_images}
    )

    tar_path = vdir / f'{variant_id}_images.tar'
    print(f"    Creating archive: {tar_path.name} ({len(all_filenames)} images)...")
    with tarfile.open(tar_path, 'w') as tar:
        for fn in sorted(all_filenames):
            fp = images_dir / fn
            if fp.exists():
                tar.add(str(fp), arcname=fn)

    # Save metadata
    meta = {
        'variant': variant_id,
        'seed': SEED,
        'train_images': len(train_coco['images']),
        'val_images': len(val_coco['images']),
        'test_images': len(test_coco['images']),
        'train_annotations': len(train_coco['annotations']),
        'val_annotations': len(val_coco['annotations']),
        'test_annotations': len(test_coco['annotations']),
    }
    with open(vdir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"    Done: {variant_id}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Generate ablation dataset variants')
    ap.add_argument('--images-dir', type=str, required=True)
    ap.add_argument('--annotations', type=str, required=True)
    ap.add_argument('--split', type=str, default='video_split.json')
    ap.add_argument('--cache', type=str, default=None,
                    help='Feature cache dir (reuse DINO/Fourier/SAM/EL2N)')
    ap.add_argument('--output', type=str, default='./output/ablation')
    ap.add_argument('--target', type=int, default=TARGET_SIZE)
    ap.add_argument('--variants', type=str, nargs='+',
                    default=['v1', 'v2', 'v3', 'v4', 'v5'],
                    help='Which variants to generate (default: all)')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache) if args.cache else None

    # Load data
    print("Loading split and annotations...")
    split_data = load_split(args.split)
    coco_full = load_coco(args.annotations)

    # Get train-only images
    train_images, train_ids = filter_images_by_split(coco_full, split_data, 'train')
    print(f"Train pool: {len(train_images)} images from {len(split_data['train_videos'])} videos")
    print(f"Val: {len(filter_images_by_split(coco_full, split_data, 'val')[0])} images")
    print(f"Test: {len(filter_images_by_split(coco_full, split_data, 'test')[0])} images")

    # Generate each variant
    generators = {
        'v1': lambda: variant_full_pipeline(train_images, images_dir, cache_dir, args.target),
        'v2': lambda: variant_no_fourier(train_images, images_dir, cache_dir, args.target),
        'v3': lambda: variant_no_el2n(train_images, images_dir, cache_dir, args.target),
        'v4': lambda: variant_no_clustering(train_images, images_dir, cache_dir, args.target),
        'v5': lambda: variant_random(train_images, images_dir, args.target),
    }

    for vid in args.variants:
        vid = vid.lower()
        if vid not in generators:
            print(f"Unknown variant: {vid}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Generating variant {vid.upper()}")
        print(f"{'='*60}")

        selected_paths = generators[vid]()
        save_variant(vid, selected_paths, coco_full, split_data, images_dir, output_dir)

    # Copy video_split.json to output for reference
    shutil.copy2(args.split, output_dir / 'video_split.json')

    print(f"\n{'='*60}")
    print(f"All variants saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
