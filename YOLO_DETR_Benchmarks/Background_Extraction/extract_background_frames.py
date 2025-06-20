#!/usr/bin/env python3
"""
Main script to extract background frames from cataract surgery videos
for DETR training dataset improvement.

This script orchestrates the entire process:
1. Extract frames from videos in E:/Cataract/videos/micro
2. Identify background frames (frames without surgical tools)
3. Select high-quality diverse background frames for training

Usage:
    python extract_background_frames.py
    python extract_background_frames.py --video_dir "E:/Cataract/videos/micro" --max_frames 500
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil

def run_frame_extraction(video_dir, output_dir, frame_interval=30):
    """Run frame extraction script"""
    script_path = os.path.join("scripts", "extract_frames.py")
    
    cmd = [
        sys.executable, script_path,
        "--video_dir", video_dir,
        "--output_dir", output_dir,
        "--frame_interval", str(frame_interval)
    ]
    
    print("=" * 60)
    print("STEP 1: EXTRACTING FRAMES FROM VIDEOS")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in frame extraction: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def run_background_identification(input_dir, output_dir, confidence=0.3, max_frames=1000):
    """Run background frame identification script"""
    script_path = os.path.join("scripts", "identify_background_frames.py")
    
    cmd = [
        sys.executable, script_path,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--confidence", str(confidence),
        "--max_frames", str(max_frames)
    ]
    
    print("\n" + "=" * 60)
    print("STEP 2: IDENTIFYING BACKGROUND FRAMES")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in background identification: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def create_detr_dataset_structure(background_dir, output_dir):
    """Create DETR-compatible dataset structure with background frames"""
    print("\n" + "=" * 60)
    print("STEP 3: CREATING DETR DATASET STRUCTURE")
    print("=" * 60)
    
    # Create DETR dataset directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get background frames
    background_frames = list(Path(background_dir).glob("*.jpg"))
    background_frames.extend(list(Path(background_dir).glob("*.jpeg")))
    background_frames.extend(list(Path(background_dir).glob("*.png")))
    
    if not background_frames:
        print("No background frames found!")
        return False
    
    # Split frames: 80% train, 20% val
    split_idx = int(len(background_frames) * 0.8)
    train_frames = background_frames[:split_idx]
    val_frames = background_frames[split_idx:]
    
    # Copy frames to train directory
    for frame in train_frames:
        dst_path = os.path.join(train_dir, frame.name)
        shutil.copy2(frame, dst_path)
    
    # Copy frames to val directory
    for frame in val_frames:
        dst_path = os.path.join(val_dir, frame.name)
        shutil.copy2(frame, dst_path)
    
    print(f"Created DETR dataset structure:")
    print(f"  Train frames: {len(train_frames)} -> {train_dir}")
    print(f"  Val frames: {len(val_frames)} -> {val_dir}")
    
    # Create annotation files (empty annotations for background)
    create_empty_annotations(train_dir, "train_annotations.json")
    create_empty_annotations(val_dir, "val_annotations.json")
    
    return True

def create_empty_annotations(image_dir, annotation_file):
    """Create COCO-style annotation file for background frames (no objects)"""
    import json
    from datetime import datetime
    
    images = []
    annotations = []
    
    image_files = list(Path(image_dir).glob("*.jpg"))
    image_files.extend(list(Path(image_dir).glob("*.jpeg")))
    image_files.extend(list(Path(image_dir).glob("*.png")))
    
    for img_id, img_path in enumerate(image_files, 1):
        # Read image to get dimensions
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            height, width = img.shape[:2]
            
            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
    
    # COCO format annotation structure
    coco_annotation = {
        "info": {
            "description": "Background frames for DETR training",
            "version": "1.0",
            "year": 2024,
            "contributor": "Background Frame Extractor",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,  # Empty - no objects in background frames
        "categories": [
            {
                "id": 0,
                "name": "tool",
                "supercategory": "object"
            },
            {
                "id": 1,
                "name": "background",
                "supercategory": "none"
            }
        ]
    }
    
    annotation_path = os.path.join(image_dir, annotation_file)
    with open(annotation_path, 'w') as f:
        json.dump(coco_annotation, f, indent=2)
    
    print(f"Created annotation file: {annotation_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract background frames from cataract surgery videos')
    parser.add_argument('--video_dir', type=str, default='E:/Cataract/videos/micro',
                        help='Directory containing videos')
    parser.add_argument('--frame_interval', type=int, default=30,
                        help='Extract every N frames from videos')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='YOLO confidence threshold for object detection')
    parser.add_argument('--max_frames', type=int, default=1000,
                        help='Maximum number of background frames to extract')
    parser.add_argument('--clean_temp', action='store_true',
                        help='Clean temporary extracted frames after processing')
    
    args = parser.parse_args()
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    extracted_frames_dir = os.path.join(base_dir, "extracted_frames")
    background_frames_dir = os.path.join(base_dir, "background_frames")
    detr_dataset_dir = os.path.join(base_dir, "detr_background_dataset")
    
    print("BACKGROUND FRAME EXTRACTION FOR DETR TRAINING")
    print("=" * 60)
    print(f"Video directory: {args.video_dir}")
    print(f"Frame interval: {args.frame_interval}")
    print(f"Detection confidence: {args.confidence}")
    print(f"Max background frames: {args.max_frames}")
    print(f"Output directory: {detr_dataset_dir}")
    
    # Check if video directory exists
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory {args.video_dir} does not exist!")
        return 1
    
    # Step 1: Extract frames from videos
    if not run_frame_extraction(args.video_dir, extracted_frames_dir, args.frame_interval):
        print("Frame extraction failed!")
        return 1
    
    # Step 2: Identify background frames
    if not run_background_identification(extracted_frames_dir, background_frames_dir, 
                                       args.confidence, args.max_frames):
        print("Background frame identification failed!")
        return 1
    
    # Step 3: Create DETR dataset structure
    if not create_detr_dataset_structure(background_frames_dir, detr_dataset_dir):
        print("DETR dataset creation failed!")
        return 1
    
    # Clean up temporary files if requested
    if args.clean_temp and os.path.exists(extracted_frames_dir):
        print(f"\nCleaning temporary files: {extracted_frames_dir}")
        shutil.rmtree(extracted_frames_dir)
    
    print("\n" + "=" * 60)
    print("BACKGROUND FRAME EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Background frames dataset: {detr_dataset_dir}")
    print(f"Background frames only: {background_frames_dir}")
    print("\nNext steps:")
    print("1. Combine these background frames with your existing tool detection dataset")
    print("2. Retrain DETR with both tool examples and background examples")
    print("3. Use a proper background/no_object class in your training")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())