#!/usr/bin/env python3
"""
Create COCO format annotations for background frames from detection results.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def create_background_annotations(detection_results_path: str, output_path: str):
    """Create COCO format annotations for background frames."""
    
    # Read detection results
    with open(detection_results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
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
    
    image_id = 1
    annotation_id = 1
    
    # Process each video's results
    for video_details in results.get("processing_stats", {}).get("video_details", []):
        video_name = video_details.get("video_name", "unknown")
        
        # Process each detection result
        for detection in video_details.get("detection_results", []):
            if detection.get("is_background", False):
                frame_path = detection.get("frame_path", "")
                
                # Extract frame filename from path
                frame_filename = Path(frame_path).name
                
                # Add image entry
                image_entry = {
                    "id": image_id,
                    "file_name": frame_filename,
                    "width": 1920,  # Default surgical video resolution
                    "height": 1080,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": datetime.now().isoformat()
                }
                coco_annotations["images"].append(image_entry)
                
                # For background frames, we create an annotation covering the entire image
                # to indicate this is a background class example
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
                
                image_id += 1
                annotation_id += 1
    
    # Save annotations
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"Created background annotations with {len(coco_annotations['images'])} images")
    print(f"Saved to: {output_path}")
    
    return len(coco_annotations['images'])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_background_annotations.py <detection_results.json> <output_annotations.json>")
        sys.exit(1)
    
    detection_results_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(detection_results_path).exists():
        print(f"Error: Detection results file not found: {detection_results_path}")
        sys.exit(1)
    
    try:
        count = create_background_annotations(detection_results_path, output_path)
        print(f"Successfully created annotations for {count} background frames")
    except Exception as e:
        print(f"Error creating annotations: {e}")
        sys.exit(1)