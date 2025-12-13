"""
Detect Duplicates Between Test Dataset and Training Dataset
============================================================
Compares extracted test frames against training dataset to find duplicates.
Uses both filename matching and perceptual hashing for robust detection.

Author: Claude Code
Date: 2025-12-13
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import yaml
from tqdm import tqdm

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("WARNING: imagehash not installed. Install with: pip install imagehash Pillow")
    print("         Perceptual hashing will be disabled.")


@dataclass
class DuplicateResult:
    """Result of duplicate detection for a single test image."""
    test_filename: str
    test_path: str
    is_duplicate: bool = False
    match_type: str = ""  # "exact_name", "similar_frame", "phash_match"
    matched_training_files: List[str] = field(default_factory=list)
    phash_distance: Optional[int] = None
    details: str = ""


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    default_config = {
        'training_json': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Datasets/datasets_20250606.tar/datasets_20250606/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json",
        'training_images_dir': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Datasets/datasets_20250606.tar/datasets_20250606/datasets/DETR_augmented_dataset_20250218/images",
        'test_frames_dir': "output/test_frames",
        'output_report_dir': "output/reports",
        'phash_threshold': 10,  # Hamming distance threshold for pHash similarity
        'frame_proximity_threshold': 5,  # Consider frames within N indices as "nearby"
        'use_phash': True,
        'check_nearby_frames': True,
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                default_config.update(user_config)

    return default_config


def parse_frame_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse filename to extract video name and frame index.
    Expected formats:
    - test01_frame_0000746.jpg
    - train03_frame_0012345.jpg
    - test01_frame_0000746_aug_1.jpg (augmented)
    """
    # Remove augmentation suffix if present
    base_name = re.sub(r'_aug_\d+', '', filename)

    # Match pattern: {video_name}_frame_{frame_number}.jpg
    match = re.match(r'(.+?)_frame_(\d+)\.jpg', base_name, re.IGNORECASE)
    if match:
        video_name = match.group(1)
        frame_idx = int(match.group(2))
        return video_name, frame_idx

    return None, None


def load_training_dataset_info(json_path: str) -> Dict:
    """
    Load training dataset JSON and extract frame information.
    Returns dict with:
    - filenames: set of all filenames
    - original_frames: dict mapping (video_name, frame_idx) -> list of filenames
    - video_frames: dict mapping video_name -> set of frame indices
    """
    print(f"Loading training dataset from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filenames = set()
    original_frames = defaultdict(list)  # (video, frame_idx) -> [filenames]
    video_frames = defaultdict(set)  # video_name -> {frame_indices}

    for img in tqdm(data['images'], desc="Parsing training dataset"):
        filename = img['file_name']
        filenames.add(filename)

        video_name, frame_idx = parse_frame_filename(filename)
        if video_name and frame_idx is not None:
            # Check if this is an augmented image
            is_augmented = '_aug_' in filename

            if not is_augmented:
                # This is an original frame
                original_frames[(video_name, frame_idx)].append(filename)
                video_frames[video_name].add(frame_idx)
            else:
                # Augmented image - still track it
                original_frames[(video_name, frame_idx)].append(filename)

    print(f"  Total images in training set: {len(filenames)}")
    print(f"  Unique original frames: {len([k for k, v in original_frames.items() if any('_aug_' not in f for f in v)])}")
    print(f"  Videos in training set: {list(video_frames.keys())}")

    return {
        'filenames': filenames,
        'original_frames': dict(original_frames),
        'video_frames': {k: sorted(v) for k, v in video_frames.items()}
    }


def compute_phash(image_path: str) -> Optional['imagehash.ImageHash']:
    """Compute perceptual hash for an image."""
    if not IMAGEHASH_AVAILABLE:
        return None

    try:
        img = Image.open(image_path)
        return imagehash.phash(img)
    except Exception as e:
        print(f"  WARNING: Could not compute pHash for {image_path}: {e}")
        return None


def find_nearby_frames(
    video_name: str,
    frame_idx: int,
    video_frames: Dict[str, List[int]],
    proximity: int = 5
) -> List[int]:
    """Find frame indices in training set that are within proximity of test frame."""
    if video_name not in video_frames:
        return []

    nearby = []
    for train_idx in video_frames[video_name]:
        if abs(train_idx - frame_idx) <= proximity and train_idx != frame_idx:
            nearby.append(train_idx)

    return sorted(nearby)


def check_single_frame(
    test_path: Path,
    training_info: Dict,
    config: dict,
    training_phashes: Dict[str, 'imagehash.ImageHash'] = None
) -> DuplicateResult:
    """Check a single test frame for duplicates."""
    filename = test_path.name
    result = DuplicateResult(
        test_filename=filename,
        test_path=str(test_path)
    )

    video_name, frame_idx = parse_frame_filename(filename)
    if not video_name or frame_idx is None:
        result.details = "Could not parse filename"
        return result

    # Check 1: Exact filename match (including augmentations)
    if filename in training_info['filenames']:
        result.is_duplicate = True
        result.match_type = "exact_name"
        result.matched_training_files = [filename]
        result.details = "Exact filename match found in training set"
        return result

    # Check 2: Same video + frame index (original or augmented versions exist)
    key = (video_name, frame_idx)
    if key in training_info['original_frames']:
        result.is_duplicate = True
        result.match_type = "same_frame"
        result.matched_training_files = training_info['original_frames'][key]
        result.details = f"Same frame exists in training (with {len(result.matched_training_files)} versions including augmentations)"
        return result

    # Check 3: Nearby frames (temporal proximity)
    if config.get('check_nearby_frames', True):
        proximity = config.get('frame_proximity_threshold', 5)
        nearby = find_nearby_frames(
            video_name, frame_idx,
            training_info['video_frames'],
            proximity
        )
        if nearby:
            result.details = f"WARNING: Nearby frames in training set: {nearby} (within {proximity} frames)"
            # Not marking as duplicate, but flagging for review

    # Check 4: Perceptual hash comparison (optional, expensive)
    if config.get('use_phash', True) and IMAGEHASH_AVAILABLE and training_phashes:
        test_phash = compute_phash(str(test_path))
        if test_phash:
            threshold = config.get('phash_threshold', 10)
            for train_file, train_hash in training_phashes.items():
                if train_hash:
                    distance = test_phash - train_hash
                    if distance <= threshold:
                        result.is_duplicate = True
                        result.match_type = "phash_match"
                        result.phash_distance = distance
                        result.matched_training_files.append(train_file)
                        result.details = f"Perceptual hash match (distance={distance}, threshold={threshold})"
                        # Continue to find all matches

    return result


def build_training_phashes(
    training_images_dir: str,
    training_filenames: Set[str],
    sample_size: int = None
) -> Dict[str, 'imagehash.ImageHash']:
    """
    Build perceptual hashes for training images.
    Note: This can be slow for large datasets. Consider sampling or caching.
    """
    if not IMAGEHASH_AVAILABLE:
        return {}

    images_dir = Path(training_images_dir)
    if not images_dir.exists():
        print(f"WARNING: Training images directory not found: {images_dir}")
        return {}

    # For 90k images, we'll only hash original (non-augmented) images
    original_files = [f for f in training_filenames if '_aug_' not in f]

    if sample_size and len(original_files) > sample_size:
        import random
        original_files = random.sample(original_files, sample_size)
        print(f"  Sampling {sample_size} images for pHash computation")

    phashes = {}
    print(f"Computing perceptual hashes for {len(original_files)} training images...")

    for filename in tqdm(original_files, desc="Computing pHashes"):
        img_path = images_dir / filename
        if img_path.exists():
            phash = compute_phash(str(img_path))
            if phash:
                phashes[filename] = phash

    print(f"  Computed {len(phashes)} hashes")
    return phashes


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicates between test and training datasets"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test-dir', '-t',
        type=str,
        default=None,
        help='Directory containing test frames (overrides config)'
    )
    parser.add_argument(
        '--no-phash',
        action='store_true',
        help='Disable perceptual hash comparison (faster but less thorough)'
    )
    parser.add_argument(
        '--phash-sample',
        type=int,
        default=5000,
        help='Number of training images to sample for pHash (default: 5000, 0=all)'
    )
    parser.add_argument(
        '--remove-duplicates',
        action='store_true',
        help='Remove detected duplicates (moves to duplicates/ subfolder)'
    )
    args = parser.parse_args()

    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(str(config_path) if config_path.exists() else None)

    # Override with command line arguments
    if args.test_dir:
        config['test_frames_dir'] = args.test_dir
    if args.no_phash:
        config['use_phash'] = False

    # Setup paths
    test_dir = Path(config['test_frames_dir'])
    if not test_dir.is_absolute():
        test_dir = script_dir / test_dir

    report_dir = Path(config['output_report_dir'])
    if not report_dir.is_absolute():
        report_dir = script_dir / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Duplicate Detection: Test Dataset vs Training Dataset")
    print("=" * 70)
    print(f"Test frames directory: {test_dir}")
    print(f"Training JSON: {config['training_json']}")
    print(f"Use perceptual hashing: {config.get('use_phash', True)}")
    print("=" * 70)

    # Load training dataset info
    training_info = load_training_dataset_info(config['training_json'])

    # Build perceptual hashes (optional)
    training_phashes = {}
    if config.get('use_phash', True) and IMAGEHASH_AVAILABLE:
        sample_size = args.phash_sample if args.phash_sample > 0 else None
        training_phashes = build_training_phashes(
            config['training_images_dir'],
            training_info['filenames'],
            sample_size=sample_size
        )

    # Find test frames
    test_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    print(f"\nFound {len(test_files)} test frames to check")

    # Check each test frame
    print("\n" + "=" * 70)
    print("Checking for duplicates...")
    print("=" * 70)

    results = []
    duplicates = []
    warnings = []

    for test_path in tqdm(test_files, desc="Checking frames"):
        result = check_single_frame(
            test_path,
            training_info,
            config,
            training_phashes
        )
        results.append(result)

        if result.is_duplicate:
            duplicates.append(result)
        elif result.details and "WARNING" in result.details:
            warnings.append(result)

    # Generate report
    report = {
        'detection_date': datetime.now().isoformat(),
        'config': {
            'training_json': config['training_json'],
            'test_frames_dir': str(test_dir),
            'use_phash': config.get('use_phash', True),
            'phash_threshold': config.get('phash_threshold', 10),
            'frame_proximity_threshold': config.get('frame_proximity_threshold', 5),
        },
        'summary': {
            'total_test_frames': len(test_files),
            'duplicates_found': len(duplicates),
            'warnings': len(warnings),
            'clean_frames': len(test_files) - len(duplicates),
        },
        'duplicates': [
            {
                'test_file': d.test_filename,
                'match_type': d.match_type,
                'matched_files': d.matched_training_files,
                'phash_distance': d.phash_distance,
                'details': d.details
            }
            for d in duplicates
        ],
        'warnings': [
            {
                'test_file': w.test_filename,
                'details': w.details
            }
            for w in warnings
        ]
    }

    # Save report
    report_path = report_dir / f"duplicates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Handle duplicates removal if requested
    if args.remove_duplicates and duplicates:
        duplicates_dir = test_dir / "duplicates_removed"
        duplicates_dir.mkdir(exist_ok=True)

        print(f"\nMoving {len(duplicates)} duplicates to: {duplicates_dir}")
        for dup in duplicates:
            src = Path(dup.test_path)
            dst = duplicates_dir / dup.test_filename
            if src.exists():
                src.rename(dst)

    # Print summary
    print("\n" + "=" * 70)
    print("DETECTION COMPLETE")
    print("=" * 70)
    print(f"Total test frames checked: {len(test_files)}")
    print(f"Duplicates found: {len(duplicates)}")
    print(f"Warnings (nearby frames): {len(warnings)}")
    print(f"Clean frames: {len(test_files) - len(duplicates)}")
    print(f"\nReport saved to: {report_path}")

    if duplicates:
        print("\n" + "-" * 70)
        print("DUPLICATE DETAILS:")
        print("-" * 70)
        for dup in duplicates[:10]:  # Show first 10
            print(f"  {dup.test_filename}")
            print(f"    Type: {dup.match_type}")
            print(f"    Match: {dup.matched_training_files[0] if dup.matched_training_files else 'N/A'}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more (see report)")

    if warnings:
        print("\n" + "-" * 70)
        print("WARNINGS (frames near training data):")
        print("-" * 70)
        for warn in warnings[:5]:  # Show first 5
            print(f"  {warn.test_filename}: {warn.details}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more (see report)")

    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
