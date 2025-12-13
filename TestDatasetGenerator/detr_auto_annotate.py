"""
DETR Auto-Annotation Script
============================
Batch processes all images through DETR model and saves detections to COCO JSON.
No user interaction - fully automatic.

Usage:
    py -3.11 detr_auto_annotate.py
    py -3.11 detr_auto_annotate.py --threshold 0.2
    py -3.11 detr_auto_annotate.py --input path/to/images --output detections.json

Author: TestDatasetGenerator
Date: 2025-12-13
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from glob import glob
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*pass `assign=True`.*")

import torch
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    'input_dir': SCRIPT_DIR / "output" / "test_frames",
    'output_json': SCRIPT_DIR / "output" / "detr_detections_coco.json",
    'checkpoint': Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR/DETR_Checkpoints/checkpoint_epoch_170.pth"),
    'confidence_threshold': 0.8,  # 80% - only high confidence detections
    'query_id': 81,  # Best query for tooltip detection
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DETR MODEL
# =============================================================================

def load_detr_model(checkpoint_path):
    """Load DETR model from checkpoint."""
    from transformers import DetrImageProcessor, DetrForObjectDetection

    print(f"Loading DETR model...")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Device: {DEVICE}")

    # Load processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load model
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(DEVICE)
    model.eval()

    print("  Model loaded successfully!")
    return model, processor


def run_detr_inference(model, processor, image_path, conf_threshold=0.3, query_id=81):
    """
    Run DETR inference on a single image.

    Returns:
        List of detections: [{'bbox': [x,y,w,h], 'score': float, 'category_id': 0}, ...]
    """
    try:
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]  # [100, num_classes+1]
        boxes = outputs.pred_boxes[0]  # [100, 4] normalized cxcywh

        detections = []

        # Method 1: Check specific query (Q81)
        query_logits = logits[query_id]
        probs = torch.softmax(query_logits, dim=-1)
        class_prob = probs[0].item()  # Probability for class 0 (tool)

        if class_prob >= conf_threshold:
            cx, cy, w, h = boxes[query_id].tolist()
            x = (cx - w/2) * img_width
            y = (cy - h/2) * img_height
            w_px = w * img_width
            h_px = h * img_height

            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            w_px = min(w_px, img_width - x)
            h_px = min(h_px, img_height - y)

            detections.append({
                'bbox': [x, y, w_px, h_px],
                'score': class_prob,
                'category_id': 0,
                'query_id': query_id
            })

        # Method 2: If no detection from Q81, find best overall
        if not detections:
            probs_all = torch.softmax(logits, dim=-1)
            class_probs = probs_all[:, 0]
            best_idx = class_probs.argmax().item()
            best_prob = class_probs[best_idx].item()

            if best_prob >= conf_threshold:
                cx, cy, w, h = boxes[best_idx].tolist()
                x = (cx - w/2) * img_width
                y = (cy - h/2) * img_height
                w_px = w * img_width
                h_px = h * img_height

                x = max(0, x)
                y = max(0, y)
                w_px = min(w_px, img_width - x)
                h_px = min(h_px, img_height - y)

                detections.append({
                    'bbox': [x, y, w_px, h_px],
                    'score': best_prob,
                    'category_id': 0,
                    'query_id': best_idx
                })

        return detections, img_width, img_height

    except Exception as e:
        print(f"  ERROR processing {image_path}: {e}")
        return [], 0, 0


def process_all_images(model, processor, input_dir, conf_threshold, query_id):
    """Process all images and return COCO-format data."""

    # Find all images
    image_files = sorted(
        glob(str(input_dir / "*.jpg")) + glob(str(input_dir / "*.png"))
    )

    if not image_files:
        print(f"ERROR: No images found in {input_dir}")
        return None

    print(f"\nProcessing {len(image_files)} images...")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  Primary query ID: Q{query_id}")

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "DETR Auto-Annotations",
            "version": "1.0",
            "year": 2025,
            "date_created": datetime.now().isoformat(),
            "detr_checkpoint": str(DEFAULT_CONFIG['checkpoint'].name),
            "confidence_threshold": conf_threshold,
            "query_id": query_id
        },
        "licenses": [],
        "categories": [
            {"id": 0, "name": "tool", "supercategory": "surgical"}
        ],
        "images": [],
        "annotations": []
    }

    annotation_id = 1
    images_with_detections = 0
    total_detections = 0

    for idx, image_path in enumerate(tqdm(image_files, desc="DETR inference")):
        filename = os.path.basename(image_path)
        image_id = idx + 1

        # Run inference
        detections, img_width, img_height = run_detr_inference(
            model, processor, image_path, conf_threshold, query_id
        )

        # Add image entry
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": img_width,
            "height": img_height
        })

        # Add annotations
        if detections:
            images_with_detections += 1
            for det in detections:
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": det['category_id'],
                    "bbox": det['bbox'],
                    "area": det['bbox'][2] * det['bbox'][3],
                    "score": det['score'],
                    "query_id": det['query_id'],
                    "iscrowd": 0
                })
                annotation_id += 1
                total_detections += 1

    # Add statistics to info
    coco_data["info"]["statistics"] = {
        "total_images": len(image_files),
        "images_with_detections": images_with_detections,
        "detection_rate": f"{images_with_detections/len(image_files)*100:.1f}%",
        "total_detections": total_detections
    }

    return coco_data


def main():
    parser = argparse.ArgumentParser(description="DETR Auto-Annotation")
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input directory with images')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output COCO JSON file')
    parser.add_argument('--threshold', '-t', type=float, default=0.8,
                       help='Confidence threshold (default: 0.8 = 80%%)')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='DETR checkpoint path')
    parser.add_argument('--query', '-q', type=int, default=81,
                       help='DETR query ID (default: 81)')
    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input) if args.input else DEFAULT_CONFIG['input_dir']
    output_json = Path(args.output) if args.output else DEFAULT_CONFIG['output_json']
    checkpoint = Path(args.checkpoint) if args.checkpoint else DEFAULT_CONFIG['checkpoint']

    print("=" * 70)
    print("DETR AUTO-ANNOTATION")
    print("=" * 70)
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_json}")
    print(f"Checkpoint: {checkpoint.name}")
    print(f"Threshold:  {args.threshold}")
    print(f"Query ID:   Q{args.query}")
    print("=" * 70)

    # Check paths
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Load model
    model, processor = load_detr_model(checkpoint)

    # Process images
    coco_data = process_all_images(
        model, processor, input_dir, args.threshold, args.query
    )

    if coco_data is None:
        sys.exit(1)

    # Save results
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # Print summary
    stats = coco_data["info"]["statistics"]
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total images:      {stats['total_images']}")
    print(f"With detections:   {stats['images_with_detections']} ({stats['detection_rate']})")
    print(f"Total detections:  {stats['total_detections']}")
    print(f"Output saved to:   {output_json}")
    print("=" * 70)
    print("\nNext step: Run review_annotations.py to review and correct detections")


if __name__ == "__main__":
    main()
