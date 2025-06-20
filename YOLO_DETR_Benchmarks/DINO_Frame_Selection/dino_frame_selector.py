#!/usr/bin/env python3
"""
DINO-based Intelligent Frame Selection for DETR Training

This script uses DINO (Self-Supervised Vision Transformer) to select the most
informative frames from surgical videos for DETR training. DINO's semantic
understanding helps identify diverse and representative training samples.

Key Features:
- Uses pre-trained DINO model for rich feature extraction
- Performs intelligent clustering based on semantic similarity
- Selects diverse representative frames for balanced training
- Integrates with existing background extraction pipeline
- Creates DETR-ready dataset structure

Usage:
    python dino_frame_selector.py --input_dir "path/to/frames" --output_dir "selected_frames"
    python dino_frame_selector.py --video_dir "E:/Cataract/videos/micro" --extract_first --n_clusters 25
"""

import os
import sys
import argparse
import subprocess
import json
import shutil
from pathlib import Path
import numpy as np
import pickle
from datetime import datetime

class DINOFrameSelector:
    """
    Main orchestrator for DINO-based frame selection
    """
    
    def __init__(self, work_dir=None):
        """
        Initialize DINO frame selector
        
        Args:
            work_dir: Working directory (default: current directory)
        """
        self.work_dir = work_dir or os.getcwd()
        self.scripts_dir = os.path.join(self.work_dir, "scripts")
        
        # Ensure scripts directory exists
        if not os.path.exists(self.scripts_dir):
            raise FileNotFoundError(f"Scripts directory not found: {self.scripts_dir}")
    
    def extract_frames_from_videos(self, video_dir, output_dir, frame_interval=60):
        """
        Extract frames from videos using existing extraction script
        
        Args:
            video_dir: Directory containing videos
            output_dir: Directory to save extracted frames
            frame_interval: Extract every N frames
            
        Returns:
            True if successful, False otherwise
        """
        # Use the existing frame extraction script from Background_Extraction
        bg_extraction_dir = os.path.join(os.path.dirname(self.work_dir), "Background_Extraction")
        extract_script = os.path.join(bg_extraction_dir, "scripts", "extract_frames.py")
        
        if not os.path.exists(extract_script):
            print(f"Frame extraction script not found: {extract_script}")
            return False
        
        cmd = [
            sys.executable, extract_script,
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
    
    def extract_dino_features(self, input_dir, output_file, model_name='dino_vits16', max_images=None):
        """
        Extract DINO features from frames
        
        Args:
            input_dir: Directory containing frames
            output_file: Output file for features
            model_name: DINO model variant
            max_images: Maximum number of images to process
            
        Returns:
            True if successful, False otherwise
        """
        script_path = os.path.join(self.scripts_dir, "dino_feature_extractor.py")
        
        cmd = [
            sys.executable, script_path,
            "--input_dir", input_dir,
            "--output_file", output_file,
            "--model_name", model_name
        ]
        
        if max_images:
            cmd.extend(["--max_images", str(max_images)])
        
        print("\n" + "=" * 60)
        print("STEP 2: EXTRACTING DINO FEATURES")
        print("=" * 60)
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error in DINO feature extraction: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False
    
    def perform_clustering_and_selection(self, features_file, output_dir, n_clusters=20, 
                                       frames_per_cluster=5, selection_method='centroid'):
        """
        Perform clustering and frame selection
        
        Args:
            features_file: Path to DINO features file
            output_dir: Output directory for results
            n_clusters: Number of clusters
            frames_per_cluster: Frames to select per cluster
            selection_method: Selection method ('centroid', 'diverse', 'quality')
            
        Returns:
            True if successful, False otherwise
        """
        script_path = os.path.join(self.scripts_dir, "dino_clustering.py")
        
        cmd = [
            sys.executable, script_path,
            "--features_file", features_file,
            "--output_dir", output_dir,
            "--n_clusters", str(n_clusters),
            "--frames_per_cluster", str(frames_per_cluster),
            "--selection_method", selection_method,
            "--copy_frames"
        ]
        
        print("\n" + "=" * 60)
        print("STEP 3: CLUSTERING AND FRAME SELECTION")
        print("=" * 60)
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error in clustering and selection: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False
    
    def create_detr_dataset(self, selected_frames_dir, output_dir, train_ratio=0.8):
        """
        Create DETR-compatible dataset structure
        
        Args:
            selected_frames_dir: Directory with selected frames
            output_dir: Output directory for DETR dataset
            train_ratio: Ratio for train/val split
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 60)
        print("STEP 4: CREATING DETR DATASET STRUCTURE")
        print("=" * 60)
        
        # Create dataset directories
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Get selected frames
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        selected_frames = []
        
        for ext in image_extensions:
            selected_frames.extend(Path(selected_frames_dir).glob(f'*{ext}'))
            selected_frames.extend(Path(selected_frames_dir).glob(f'*{ext.upper()}'))
        
        if not selected_frames:
            print("No selected frames found!")
            return False
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(selected_frames)
        
        split_idx = int(len(selected_frames) * train_ratio)
        train_frames = selected_frames[:split_idx]
        val_frames = selected_frames[split_idx:]
        
        # Copy frames
        for frame in train_frames:
            dst_path = os.path.join(train_dir, frame.name)
            shutil.copy2(frame, dst_path)
        
        for frame in val_frames:
            dst_path = os.path.join(val_dir, frame.name)
            shutil.copy2(frame, dst_path)
        
        print(f"Created DETR dataset:")
        print(f"  Train frames: {len(train_frames)} -> {train_dir}")
        print(f"  Val frames: {len(val_frames)} -> {val_dir}")
        
        # Create basic annotation files (can be extended based on needs)
        self._create_basic_annotations(train_dir, "train_annotations.json")
        self._create_basic_annotations(val_dir, "val_annotations.json")
        
        return True
    
    def _create_basic_annotations(self, image_dir, annotation_file):
        """Create basic COCO-style annotation structure"""
        images = []
        image_files = []
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        for img_id, img_path in enumerate(image_files, 1):
            # Basic image info (dimensions would need to be read from actual images)
            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": 1920,  # Default, should be read from actual image
                "height": 1080  # Default, should be read from actual image
            })
        
        # Basic COCO structure
        coco_data = {
            "info": {
                "description": "DINO-selected frames for DETR training",
                "version": "1.0",
                "year": 2024,
                "contributor": "DINO Frame Selector",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": images,
            "annotations": [],  # To be filled with actual annotations
            "categories": [
                {
                    "id": 1,
                    "name": "surgical_tool",
                    "supercategory": "tool"
                }
            ]
        }
        
        annotation_path = os.path.join(image_dir, annotation_file)
        with open(annotation_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Created basic annotation file: {annotation_path}")
    
    def create_summary_report(self, output_dir, features_file, clustering_results_file):
        """
        Create a comprehensive summary report
        
        Args:
            output_dir: Output directory for report
            features_file: Path to DINO features file
            clustering_results_file: Path to clustering results file
        """
        print("\n" + "=" * 60)
        print("STEP 5: CREATING SUMMARY REPORT")
        print("=" * 60)
        
        # Load data
        with open(features_file, 'rb') as f:
            features_data = pickle.load(f)
        
        with open(clustering_results_file, 'r') as f:
            clustering_data = json.load(f)
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "dino_model": features_data['model_info'],
            "processing_stats": features_data['processing_stats'],
            "clustering_info": clustering_data['cluster_info'],
            "cluster_statistics": {
                "total_clusters": len(clustering_data['cluster_stats']),
                "cluster_sizes": [stats['size'] for stats in clustering_data['cluster_stats'].values()],
                "selected_frames": len(clustering_data['selected_frames'])
            },
            "selection_summary": {
                "total_frames_processed": features_data['processing_stats']['successful_extractions'],
                "frames_selected": len(clustering_data['selected_frames']),
                "selection_ratio": len(clustering_data['selected_frames']) / features_data['processing_stats']['successful_extractions'],
                "average_cluster_size": np.mean([stats['size'] for stats in clustering_data['cluster_stats'].values()]),
                "largest_cluster_size": max([stats['size'] for stats in clustering_data['cluster_stats'].values()]),
                "smallest_cluster_size": min([stats['size'] for stats in clustering_data['cluster_stats'].values()])
            }
        }
        
        # Save report
        report_path = os.path.join(output_dir, "dino_selection_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Summary report saved to: {report_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("DINO FRAME SELECTION SUMMARY")
        print("=" * 60)
        print(f"Total frames processed: {report['selection_summary']['total_frames_processed']}")
        print(f"Frames selected: {report['selection_summary']['frames_selected']}")
        print(f"Selection ratio: {report['selection_summary']['selection_ratio']:.3f}")
        print(f"Number of clusters: {report['cluster_statistics']['total_clusters']}")
        print(f"Average cluster size: {report['selection_summary']['average_cluster_size']:.1f}")
        print(f"DINO model used: {report['dino_model']['model_name']}")
        print(f"Feature dimension: {report['dino_model']['feature_dim']}")

def main():
    parser = argparse.ArgumentParser(description='DINO-based intelligent frame selection for DETR training')
    
    # Input options
    parser.add_argument('--input_dir', type=str, 
                        help='Directory containing frames to process')
    parser.add_argument('--video_dir', type=str, 
                        help='Directory containing videos (will extract frames first)')
    parser.add_argument('--extract_first', action='store_true',
                        help='Extract frames from videos first')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='dino_selection_results',
                        help='Output directory for all results')
    
    # Processing parameters
    parser.add_argument('--frame_interval', type=int, default=60,
                        help='Extract every N frames from videos')
    parser.add_argument('--dino_model', type=str, default='dino_vits16',
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                        help='DINO model variant')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    
    # Clustering parameters
    parser.add_argument('--n_clusters', type=int, default=20,
                        help='Number of clusters for frame selection')
    parser.add_argument('--frames_per_cluster', type=int, default=5,
                        help='Number of frames to select per cluster')
    parser.add_argument('--selection_method', type=str, default='centroid',
                        choices=['centroid', 'diverse', 'quality'],
                        help='Frame selection method')
    
    # Dataset options
    parser.add_argument('--create_detr_dataset', action='store_true',
                        help='Create DETR-compatible dataset structure')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio for train/val split')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dir and not args.video_dir:
        print("Error: Either --input_dir or --video_dir must be provided")
        return 1
    
    # Initialize selector
    selector = DINOFrameSelector()
    
    # Setup directories
    base_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    
    extracted_frames_dir = os.path.join(base_output_dir, "extracted_frames")
    features_dir = os.path.join(base_output_dir, "features")
    clustering_dir = os.path.join(base_output_dir, "clustering_results")
    selected_frames_dir = os.path.join(clustering_dir, "selected_frames")
    
    os.makedirs(features_dir, exist_ok=True)
    
    print("DINO-BASED INTELLIGENT FRAME SELECTION FOR DETR TRAINING")
    print("=" * 80)
    print(f"Output directory: {base_output_dir}")
    print(f"DINO model: {args.dino_model}")
    print(f"Clusters: {args.n_clusters}")
    print(f"Frames per cluster: {args.frames_per_cluster}")
    print(f"Selection method: {args.selection_method}")
    
    # Step 1: Extract frames from videos if needed
    if args.video_dir or args.extract_first:
        if not args.video_dir:
            print("Error: --video_dir required when --extract_first is used")
            return 1
        
        if not selector.extract_frames_from_videos(args.video_dir, extracted_frames_dir, args.frame_interval):
            print("Frame extraction failed!")
            return 1
        
        frames_input_dir = extracted_frames_dir
    else:
        frames_input_dir = args.input_dir
    
    # Step 2: Extract DINO features
    features_file = os.path.join(features_dir, "dino_features.pkl")
    if not selector.extract_dino_features(frames_input_dir, features_file, args.dino_model, args.max_images):
        print("DINO feature extraction failed!")
        return 1
    
    # Step 3: Perform clustering and selection
    if not selector.perform_clustering_and_selection(features_file, clustering_dir, 
                                                   args.n_clusters, args.frames_per_cluster, 
                                                   args.selection_method):
        print("Clustering and selection failed!")
        return 1
    
    # Step 4: Create DETR dataset if requested
    if args.create_detr_dataset:
        detr_dataset_dir = os.path.join(base_output_dir, "detr_dataset")
        if not selector.create_detr_dataset(selected_frames_dir, detr_dataset_dir, args.train_ratio):
            print("DETR dataset creation failed!")
            return 1
    
    # Step 5: Create summary report
    clustering_results_file = os.path.join(clustering_dir, "clustering_results.json")
    selector.create_summary_report(base_output_dir, features_file, clustering_results_file)
    
    print("\n" + "=" * 80)
    print("DINO FRAME SELECTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Selected frames: {selected_frames_dir}")
    if args.create_detr_dataset:
        print(f"DETR dataset: {os.path.join(base_output_dir, 'detr_dataset')}")
    print(f"Full results: {base_output_dir}")
    
    print("\nNext steps:")
    print("1. Review selected frames and clustering results")
    print("2. Annotate selected frames for DETR training")
    print("3. Combine with background frames for complete dataset")
    print("4. Train DETR model with selected diverse examples")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())