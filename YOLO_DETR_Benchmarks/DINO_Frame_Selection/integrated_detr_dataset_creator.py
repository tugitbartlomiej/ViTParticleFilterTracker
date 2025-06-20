#!/usr/bin/env python3
"""
Integrated DETR Dataset Creator

This script combines DINO-selected informative frames with background frames
to create a complete, balanced dataset for DETR training. It addresses the
original DETR training problem by providing both positive examples (with tools)
and negative examples (background frames).

Key Features:
- Combines DINO-selected frames (informative tool examples)
- Integrates background frames (negative examples)
- Creates balanced train/val splits
- Generates proper COCO annotations
- Provides dataset statistics and quality metrics

Usage:
    python integrated_detr_dataset_creator.py --dino_frames "dino_results/selected_frames" --background_frames "background_results/background_frames" --output_dir "complete_detr_dataset"
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from datetime import datetime
import cv2
from collections import defaultdict
from tqdm import tqdm

class IntegratedDETRDatasetCreator:
    """
    Creates integrated DETR dataset combining DINO-selected frames with background frames
    """
    
    def __init__(self, output_dir):
        """
        Initialize dataset creator
        
        Args:
            output_dir: Output directory for final dataset
        """
        self.output_dir = output_dir
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        
        # Create output structure
        os.makedirs(output_dir, exist_ok=True)
        
        self.dataset_dirs = {
            'train': os.path.join(output_dir, 'train'),
            'val': os.path.join(output_dir, 'val'),
            'annotations': os.path.join(output_dir, 'annotations')
        }
        
        for dir_path in self.dataset_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def analyze_frame_sources(self, dino_frames_dir, background_frames_dir):
        """
        Analyze source frames and their characteristics
        
        Args:
            dino_frames_dir: Directory with DINO-selected frames
            background_frames_dir: Directory with background frames
            
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing frame sources...")
        
        analysis = {
            'dino_frames': self._analyze_directory(dino_frames_dir, 'DINO-selected'),
            'background_frames': self._analyze_directory(background_frames_dir, 'Background'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate totals
        analysis['total_frames'] = analysis['dino_frames']['count'] + analysis['background_frames']['count']
        analysis['dino_percentage'] = analysis['dino_frames']['count'] / analysis['total_frames'] * 100
        analysis['background_percentage'] = analysis['background_frames']['count'] / analysis['total_frames'] * 100
        
        return analysis
    
    def _analyze_directory(self, directory, frame_type):
        """Analyze a directory of frames"""
        if not os.path.exists(directory):
            return {'count': 0, 'files': [], 'type': frame_type, 'exists': False}
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(directory).glob(f'*{ext}'))
            image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        
        # Analyze file sizes and dimensions
        file_info = []
        total_size = 0
        
        for img_path in image_files[:10]:  # Sample first 10 for analysis
            try:
                # Get file size
                file_size = img_path.stat().st_size
                total_size += file_size
                
                # Get image dimensions
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width = img.shape[:2]
                    file_info.append({
                        'name': img_path.name,
                        'size': file_size,
                        'width': width,
                        'height': height
                    })
            except Exception as e:
                print(f"Error analyzing {img_path}: {e}")
        
        return {
            'count': len(image_files),
            'files': [f.name for f in image_files],
            'type': frame_type,
            'exists': True,
            'sample_info': file_info,
            'average_file_size': total_size / len(file_info) if file_info else 0,
            'directory': str(directory)
        }
    
    def create_balanced_splits(self, dino_frames_dir, background_frames_dir, 
                             balance_ratio=0.7, train_ratio=0.8):
        """
        Create balanced train/val splits
        
        Args:
            dino_frames_dir: Directory with DINO-selected frames
            background_frames_dir: Directory with background frames
            balance_ratio: Ratio of informative frames to background frames
            train_ratio: Ratio for train split
            
        Returns:
            Dictionary with split information
        """
        print(f"Creating balanced splits (balance_ratio={balance_ratio}, train_ratio={train_ratio})...")
        
        # Get frame lists
        dino_frames = self._get_image_files(dino_frames_dir)
        background_frames = self._get_image_files(background_frames_dir)
        
        print(f"Found {len(dino_frames)} DINO-selected frames")
        print(f"Found {len(background_frames)} background frames")
        
        # Balance the dataset
        target_dino_count = int(len(dino_frames) * balance_ratio)
        target_background_count = int(target_dino_count * (1 - balance_ratio) / balance_ratio)
        
        # Sample frames
        np.random.seed(42)
        
        if len(dino_frames) > target_dino_count:
            selected_dino = np.random.choice(dino_frames, target_dino_count, replace=False)
        else:
            selected_dino = dino_frames
        
        if len(background_frames) > target_background_count:
            selected_background = np.random.choice(background_frames, target_background_count, replace=False)
        else:
            selected_background = background_frames
        
        # Combine and split
        all_frames = list(selected_dino) + list(selected_background)
        frame_labels = (['dino'] * len(selected_dino)) + (['background'] * len(selected_background))
        
        # Shuffle while keeping labels aligned
        combined = list(zip(all_frames, frame_labels))
        np.random.shuffle(combined)
        all_frames, frame_labels = zip(*combined)
        
        # Create train/val splits
        split_idx = int(len(all_frames) * train_ratio)
        
        train_frames = all_frames[:split_idx]
        train_labels = frame_labels[:split_idx]
        val_frames = all_frames[split_idx:]
        val_labels = frame_labels[split_idx:]
        
        split_info = {
            'total_frames': len(all_frames),
            'train_frames': len(train_frames),
            'val_frames': len(val_frames),
            'train_dino': train_labels.count('dino'),
            'train_background': train_labels.count('background'),
            'val_dino': val_labels.count('dino'),
            'val_background': val_labels.count('background'),
            'balance_ratio': balance_ratio,
            'train_ratio': train_ratio
        }
        
        return {
            'train': list(zip(train_frames, train_labels)),
            'val': list(zip(val_frames, val_labels)),
            'split_info': split_info
        }
    
    def _get_image_files(self, directory):
        """Get list of image files from directory"""
        if not os.path.exists(directory):
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(directory).glob(f'*{ext}'))
            image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        
        return [str(f) for f in image_files]
    
    def copy_frames_to_splits(self, splits_data):
        """
        Copy frames to train/val directories
        
        Args:
            splits_data: Data from create_balanced_splits
            
        Returns:
            Dictionary with copy results
        """
        print("Copying frames to train/val directories...")
        
        copy_results = {
            'train': {'copied': 0, 'failed': 0, 'dino': 0, 'background': 0},
            'val': {'copied': 0, 'failed': 0, 'dino': 0, 'background': 0}
        }
        
        for split_name in ['train', 'val']:
            split_frames = splits_data[split_name]
            target_dir = self.dataset_dirs[split_name]
            
            for frame_path, frame_type in tqdm(split_frames, desc=f"Copying {split_name} frames"):
                try:
                    src_path = frame_path
                    dst_name = f"{frame_type}_{Path(frame_path).name}"
                    dst_path = os.path.join(target_dir, dst_name)
                    
                    shutil.copy2(src_path, dst_path)
                    copy_results[split_name]['copied'] += 1
                    copy_results[split_name][frame_type] += 1
                    
                except Exception as e:
                    print(f"Error copying {frame_path}: {e}")
                    copy_results[split_name]['failed'] += 1
        
        return copy_results
    
    def create_coco_annotations(self, splits_data, copy_results):
        """
        Create COCO-style annotations for the dataset
        
        Args:
            splits_data: Data from create_balanced_splits
            copy_results: Results from copy_frames_to_splits
            
        Returns:
            Dictionary with annotation file paths
        """
        print("Creating COCO annotations...")
        
        annotation_files = {}
        
        for split_name in ['train', 'val']:
            images = []
            annotations = []
            annotation_id = 1
            
            split_frames = splits_data[split_name]
            target_dir = self.dataset_dirs[split_name]
            
            for image_id, (frame_path, frame_type) in enumerate(split_frames, 1):
                try:
                    # Get image info
                    img = cv2.imread(frame_path)
                    if img is None:
                        continue
                    
                    height, width = img.shape[:2]
                    dst_name = f"{frame_type}_{Path(frame_path).name}"
                    
                    # Add image info
                    images.append({
                        "id": image_id,
                        "file_name": dst_name,
                        "width": width,
                        "height": height,
                        "frame_type": frame_type
                    })
                    
                    # For background frames, no annotations (negative examples)
                    # For DINO frames, would need actual tool annotations
                    # Here we create placeholder structure
                    
                    if frame_type == 'background':
                        # No annotations for background frames
                        pass
                    else:
                        # Placeholder for tool annotations
                        # In real implementation, would load actual YOLO annotations
                        # and convert to COCO format
                        pass
                        
                except Exception as e:
                    print(f"Error processing {frame_path}: {e}")
            
            # Create COCO annotation structure
            coco_data = {
                "info": {
                    "description": "Integrated DETR dataset with DINO-selected and background frames",
                    "version": "1.0",
                    "year": 2024,
                    "contributor": "Integrated DETR Dataset Creator",
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": [
                    {
                        "id": 1,
                        "name": "surgical_tool",
                        "supercategory": "tool"
                    }
                ]
            }
            
            # Save annotation file
            annotation_file = os.path.join(self.dataset_dirs['annotations'], f"{split_name}_annotations.json")
            with open(annotation_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            annotation_files[split_name] = annotation_file
            print(f"Created {split_name} annotations: {len(images)} images")
        
        return annotation_files
    
    def generate_dataset_report(self, analysis, splits_data, copy_results, annotation_files):
        """
        Generate comprehensive dataset report
        
        Args:
            analysis: Frame analysis results
            splits_data: Split information
            copy_results: Copy results
            annotation_files: Annotation file paths
        """
        print("Generating dataset report...")
        
        report = {
            "dataset_creation": {
                "timestamp": datetime.now().isoformat(),
                "output_directory": self.output_dir,
                "creator": "Integrated DETR Dataset Creator"
            },
            "source_analysis": analysis,
            "split_information": splits_data['split_info'],
            "copy_results": copy_results,
            "annotation_files": annotation_files,
            "dataset_statistics": {
                "total_train_frames": copy_results['train']['copied'],
                "total_val_frames": copy_results['val']['copied'],
                "train_dino_frames": copy_results['train']['dino'],
                "train_background_frames": copy_results['train']['background'],
                "val_dino_frames": copy_results['val']['dino'],
                "val_background_frames": copy_results['val']['background'],
                "total_frames": copy_results['train']['copied'] + copy_results['val']['copied']
            },
            "quality_metrics": {
                "dino_train_ratio": copy_results['train']['dino'] / copy_results['train']['copied'] if copy_results['train']['copied'] > 0 else 0,
                "background_train_ratio": copy_results['train']['background'] / copy_results['train']['copied'] if copy_results['train']['copied'] > 0 else 0,
                "dino_val_ratio": copy_results['val']['dino'] / copy_results['val']['copied'] if copy_results['val']['copied'] > 0 else 0,
                "background_val_ratio": copy_results['val']['background'] / copy_results['val']['copied'] if copy_results['val']['copied'] > 0 else 0,
                "failed_copies": copy_results['train']['failed'] + copy_results['val']['failed']
            }
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, "dataset_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 80)
        print("INTEGRATED DETR DATASET CREATION SUMMARY")
        print("=" * 80)
        print(f"Total frames in dataset: {report['dataset_statistics']['total_frames']}")
        print(f"Training frames: {report['dataset_statistics']['total_train_frames']}")
        print(f"  - DINO-selected: {report['dataset_statistics']['train_dino_frames']}")
        print(f"  - Background: {report['dataset_statistics']['train_background_frames']}")
        print(f"Validation frames: {report['dataset_statistics']['total_val_frames']}")
        print(f"  - DINO-selected: {report['dataset_statistics']['val_dino_frames']}")
        print(f"  - Background: {report['dataset_statistics']['val_background_frames']}")
        print(f"Failed copies: {report['quality_metrics']['failed_copies']}")
        print(f"Dataset saved to: {self.output_dir}")
        print(f"Report saved to: {report_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Create integrated DETR dataset with DINO-selected and background frames')
    parser.add_argument('--dino_frames', type=str, required=True,
                        help='Directory containing DINO-selected frames')
    parser.add_argument('--background_frames', type=str, required=True,
                        help='Directory containing background frames')
    parser.add_argument('--output_dir', type=str, default='integrated_detr_dataset',
                        help='Output directory for integrated dataset')
    parser.add_argument('--balance_ratio', type=float, default=0.7,
                        help='Ratio of informative frames to background frames')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio for train/val split')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dino_frames):
        print(f"Error: DINO frames directory not found: {args.dino_frames}")
        return 1
    
    if not os.path.exists(args.background_frames):
        print(f"Error: Background frames directory not found: {args.background_frames}")
        return 1
    
    print("INTEGRATED DETR DATASET CREATOR")
    print("=" * 50)
    print(f"DINO frames: {args.dino_frames}")
    print(f"Background frames: {args.background_frames}")
    print(f"Output directory: {args.output_dir}")
    print(f"Balance ratio: {args.balance_ratio}")
    print(f"Train ratio: {args.train_ratio}")
    
    # Initialize creator
    creator = IntegratedDETRDatasetCreator(args.output_dir)
    
    # Step 1: Analyze frame sources
    analysis = creator.analyze_frame_sources(args.dino_frames, args.background_frames)
    
    # Step 2: Create balanced splits
    splits_data = creator.create_balanced_splits(
        args.dino_frames, args.background_frames,
        args.balance_ratio, args.train_ratio
    )
    
    # Step 3: Copy frames to splits
    copy_results = creator.copy_frames_to_splits(splits_data)
    
    # Step 4: Create COCO annotations
    annotation_files = creator.create_coco_annotations(splits_data, copy_results)
    
    # Step 5: Generate report
    report = creator.generate_dataset_report(analysis, splits_data, copy_results, annotation_files)
    
    print("\n" + "=" * 80)
    print("INTEGRATED DETR DATASET CREATION COMPLETED!")
    print("=" * 80)
    print("Next steps:")
    print("1. Review the generated dataset structure")
    print("2. Add actual tool annotations to DINO-selected frames")
    print("3. Validate the dataset balance and quality")
    print("4. Train DETR model with the integrated dataset")
    print("5. Compare results with original DETR training")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())