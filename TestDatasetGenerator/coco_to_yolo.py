"""
COCO to YOLO Format Converter
==============================
Converts COCO JSON annotations to YOLOv8 format (.txt files).
Also copies images and creates dataset.yaml for training.

Usage:
    py -3.11 coco_to_yolo.py
    py -3.11 coco_to_yolo.py --input annotations.json --output yolo_dataset/

Output structure:
    output_dir/
    ├── images/          # Copied images with annotations
    ├── labels/          # YOLO .txt files
    ├── dataset.yaml     # YOLOv8 config
    └── classes.txt      # Class names

Author: TestDatasetGenerator
Date: 2025-12-13
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    'input_json': SCRIPT_DIR / "output" / "annotations_reviewed_coco.json",
    'images_dir': SCRIPT_DIR / "output" / "test_frames",
    'output_dir': SCRIPT_DIR / "output" / "yolo_dataset",
}

# =============================================================================
# CONVERTER
# =============================================================================

def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized).
    """
    x, y, w, h = bbox

    # Calculate center
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalize dimensions
    width = w / img_width
    height = h / img_height

    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, copy_images=True):
    """
    Convert COCO annotations to YOLO format.

    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing source images
        output_dir: Output directory for YOLO dataset
        copy_images: Whether to copy images (if False, creates symlinks or skips)

    Returns:
        Statistics dict
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Load COCO data
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    labels_dir = output_dir / "labels"
    images_out_dir = output_dir / "images"

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # Build category mapping (COCO id -> YOLO class index)
    categories = coco_data.get('categories', [])
    cat_id_to_yolo_class = {}
    class_names = []

    for idx, cat in enumerate(sorted(categories, key=lambda x: x['id'])):
        cat_id_to_yolo_class[cat['id']] = idx
        class_names.append(cat['name'])

    print(f"Categories: {class_names}")

    # Build image_id -> info mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Build image_id -> annotations mapping
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    # Process each image
    stats = {
        'total_images': len(coco_data['images']),
        'images_with_annotations': 0,
        'images_without_annotations': 0,
        'total_annotations': 0,
        'images_copied': 0,
        'images_skipped': 0,
    }

    print(f"\nConverting {len(coco_data['images'])} images...")

    for image_info in tqdm(coco_data['images'], desc="Converting"):
        image_id = image_info['id']
        filename = image_info['file_name']
        img_width = image_info.get('width', 0)
        img_height = image_info.get('height', 0)

        # Source image path
        src_image_path = images_dir / filename

        # Get annotations for this image
        annotations = image_id_to_annotations.get(image_id, [])

        if not annotations:
            stats['images_without_annotations'] += 1
            continue  # Skip images without annotations

        stats['images_with_annotations'] += 1

        # If dimensions not in JSON, read from image
        if img_width == 0 or img_height == 0:
            if src_image_path.exists():
                import cv2
                img = cv2.imread(str(src_image_path))
                if img is not None:
                    img_height, img_width = img.shape[:2]
                else:
                    print(f"  WARNING: Cannot read {filename}, skipping")
                    stats['images_skipped'] += 1
                    continue
            else:
                print(f"  WARNING: Image not found {filename}, skipping")
                stats['images_skipped'] += 1
                continue

        # Create YOLO label file
        label_filename = Path(filename).stem + ".txt"
        label_path = labels_dir / label_filename

        yolo_lines = []
        for ann in annotations:
            bbox = ann['bbox']
            cat_id = ann['category_id']

            # Convert to YOLO format
            yolo_class = cat_id_to_yolo_class.get(cat_id, 0)
            x_center, y_center, width, height = coco_bbox_to_yolo(bbox, img_width, img_height)

            yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            stats['total_annotations'] += 1

        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Copy image
        if copy_images and src_image_path.exists():
            dst_image_path = images_out_dir / filename
            shutil.copy2(src_image_path, dst_image_path)
            stats['images_copied'] += 1

    # Create dataset.yaml
    dataset_yaml_path = output_dir / "dataset.yaml"
    with open(dataset_yaml_path, 'w') as f:
        f.write(f"""# YOLOv8 Dataset Configuration
# Generated by coco_to_yolo.py
# Date: {Path(coco_json_path).stat().st_mtime}

path: {output_dir.absolute()}
train: images
val: images

# Classes
names:
""")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

        f.write(f"\nnc: {len(class_names)}\n")

    # Create classes.txt
    classes_txt_path = output_dir / "classes.txt"
    with open(classes_txt_path, 'w') as f:
        f.write('\n'.join(class_names))

    # Save conversion info
    info_path = output_dir / "conversion_info.json"
    with open(info_path, 'w') as f:
        json.dump({
            'source_coco': str(coco_json_path),
            'source_images': str(images_dir),
            'statistics': stats,
            'class_names': class_names,
            'category_mapping': {str(k): v for k, v in cat_id_to_yolo_class.items()}
        }, f, indent=2)

    return stats, class_names


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format")
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input COCO JSON file')
    parser.add_argument('--images', type=str, default=None,
                       help='Source images directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for YOLO dataset')
    parser.add_argument('--no-copy', action='store_true',
                       help='Do not copy images (only create labels)')
    args = parser.parse_args()

    # Setup paths
    input_json = Path(args.input) if args.input else DEFAULT_CONFIG['input_json']
    images_dir = Path(args.images) if args.images else DEFAULT_CONFIG['images_dir']
    output_dir = Path(args.output) if args.output else DEFAULT_CONFIG['output_dir']

    print("=" * 60)
    print("COCO TO YOLO CONVERTER")
    print("=" * 60)
    print(f"Input JSON:  {input_json}")
    print(f"Images dir:  {images_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Copy images: {not args.no_copy}")
    print("=" * 60)

    # Check input
    if not input_json.exists():
        print(f"ERROR: Input JSON not found: {input_json}")

        # Suggest alternatives
        alt_paths = [
            DEFAULT_CONFIG['input_json'],
            SCRIPT_DIR / "output" / "detr_detections_coco.json",
            SCRIPT_DIR / "output" / "test_annotations_coco.json",
        ]
        print("\nAvailable COCO files:")
        for p in alt_paths:
            if p.exists():
                print(f"  - {p}")

        sys.exit(1)

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    # Convert
    stats, class_names = convert_coco_to_yolo(
        input_json, images_dir, output_dir, copy_images=not args.no_copy
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total images in COCO:      {stats['total_images']}")
    print(f"Images with annotations:   {stats['images_with_annotations']}")
    print(f"Images without annotations:{stats['images_without_annotations']}")
    print(f"Total annotations:         {stats['total_annotations']}")
    print(f"Images copied:             {stats['images_copied']}")
    print(f"Images skipped:            {stats['images_skipped']}")
    print(f"Classes:                   {class_names}")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/        # {stats['images_copied']} images")
    print(f"  ├── labels/        # {stats['images_with_annotations']} label files")
    print(f"  ├── dataset.yaml   # YOLOv8 config")
    print(f"  └── classes.txt    # Class names")
    print("=" * 60)
    print(f"\nTo train YOLOv8:")
    print(f"  yolo detect train data={output_dir / 'dataset.yaml'} model=yolov8n.pt epochs=100")


if __name__ == "__main__":
    main()
