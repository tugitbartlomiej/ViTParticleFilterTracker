#!/usr/bin/env python3
"""
Fix background annotations to match actual selected frames in train directory.
"""

import json
import os
from pathlib import Path
from datetime import datetime

def create_annotations_from_files():
    """Create annotations based on actual files in background_frames/train/"""
    
    # Path to actual background frames
    background_train_dir = Path("pipeline_output_single_test/background_frames/train")
    output_file = Path("pipeline_output_single_test/background_frames/annotations.json")
    
    if not background_train_dir.exists():
        print(f"Background train directory not found: {background_train_dir}")
        return False
    
    # Get all jpg files
    image_files = list(background_train_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} background frames")
    
    # Create COCO format annotations
    coco_annotations = {
        "info": {
            "description": "Background frames for DETR training",
            "version": "1.0",
            "year": 2025,
            "contributor": "DETR Background Training Pipeline",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown", 
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "background",
                "supercategory": "scene"
            }
        ]
    }
    
    # Process each image file
    for i, image_file in enumerate(sorted(image_files)):
        image_id = i + 1
        annotation_id = i + 1
        
        # Add image entry
        image_entry = {
            "id": image_id,
            "file_name": image_file.name,
            "width": 1920,  # Default surgical video resolution
            "height": 1080,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.now().isoformat()
        }
        coco_annotations["images"].append(image_entry)
        
        # Add annotation (full frame background)
        annotation_entry = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # background category
            "bbox": [0, 0, 1920, 1080],  # Full image bbox
            "area": 1920 * 1080,
            "iscrowd": 0,
            "segmentation": []
        }
        coco_annotations["annotations"].append(annotation_entry)
    
    # Save annotations
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"Created annotations for {len(image_files)} background frames")
    print(f"Saved to: {output_file}")
    return True

if __name__ == "__main__":
    success = create_annotations_from_files()
    if success:
        print("✅ Background annotations fixed successfully!")
    else:
        print("❌ Failed to fix background annotations")