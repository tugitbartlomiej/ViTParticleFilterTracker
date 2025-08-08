#!/usr/bin/env python3
"""
Intelligent Background Frame Selector
=====================================

This script combines DINO, DETR, and YOLO models to extract the most significant 
background frames from surgical videos for DETR training. It addresses the key 
problem identified in KOMPLETNY_RAPORT: DETR needs background frames to reduce 
false positives.

Key Features:
- Multi-model consensus: Uses both YOLO and DETR to identify background frames
- DINO-based diversity: Semantic clustering for diverse frame selection
- Content-aware selection: HOG and Optical Flow for content richness
- Batch processing: Handles all videos in the micro folder
- DETR-ready output: Creates COCO format dataset

Usage:
    python intelligent_background_frame_selector.py --videos_dir "E:/Cataract/videos/micro"
    python intelligent_background_frame_selector.py --videos_dir "E:/Cataract/videos/micro" --max_frames_per_video 100
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import shutil
from collections import defaultdict

# Add project paths to system path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "DINO_Frame_Selection" / "scripts"))
sys.path.insert(0, str(project_root.parent / "Annotators" / "Utils" / "SignificantImageSelector"))

# Import existing implementations
from dino_feature_extractor import DINOFeatureExtractor
from dino_clustering_wrapper import DINOClusterer
from image_feature_extractor import ImageFeatureExtractor

# Import model libraries
from ultralytics import YOLO
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image

class IntelligentBackgroundFrameSelector:
    """
    Main class for intelligent background frame selection using DINO, DETR, and YOLO
    """
    
    def __init__(self, 
                 yolo_model_path="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt",
                 detr_model_path="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final",
                 output_dir="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17",
                 device='auto'):
        """
        Initialize the intelligent background frame selector
        
        Args:
            yolo_model_path: Path to YOLO model
            detr_model_path: Path to DETR model
            output_dir: Output directory for results
            device: Device to run models on ('auto', 'cuda', 'cpu')
        """
        self.yolo_model_path = yolo_model_path
        self.detr_model_path = detr_model_path
        self.output_dir = Path(output_dir)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.yolo_model = None
        self.detr_model = None
        self.detr_processor = None
        self.dino_extractor = None
        self.hog_extractor = None
        
        # Configuration
        self.yolo_confidence_threshold = 0.3
        self.detr_confidence_threshold = 0.3
        self.frame_interval = 30  # Extract every 30 frames
        self.max_frames_per_video = 200
        
        # Statistics
        self.stats = {
            'total_videos': 0,
            'total_frames_extracted': 0,
            'total_background_frames': 0,
            'total_selected_frames': 0,
            'processing_time': 0,
            'video_details': []
        }
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create all required output directories"""
        directories = [
            'train',
            'val',
            'annotations'
        ]
        
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def load_models(self):
        """Load all required models"""
        print("Loading models...")
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"✓ YOLO model loaded from: {self.yolo_model_path}")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            return False
        
        # Load DETR model
        try:
            self.detr_model = DetrForObjectDetection.from_pretrained(
                str(self.detr_model_path)
            )
            self.detr_processor = DetrImageProcessor.from_pretrained(
                str(self.detr_model_path)
            )
            self.detr_model.to(self.device)
            self.detr_model.eval()
            print(f"✓ DETR model loaded from: {self.detr_model_path}")
        except Exception as e:
            print(f"✗ Error loading DETR model: {e}")
            return False
        
        # Initialize DINO extractor
        try:
            self.dino_extractor = DINOFeatureExtractor(
                model_name='dino_vits16',
                device=str(self.device)
            )
            print("✓ DINO feature extractor initialized")
        except Exception as e:
            print(f"✗ Error initializing DINO extractor: {e}")
            return False
        
        # Initialize HOG extractor
        try:
            self.hog_extractor = ImageFeatureExtractor(method='hog')
            print("✓ HOG feature extractor initialized")
        except Exception as e:
            print(f"✗ Error initializing HOG extractor: {e}")
            return False
        
        return True
    
    def extract_frames_from_video(self, video_path, output_dir):
        """
        Extract frames from a single video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            
        Returns:
            List of extracted frame paths
        """
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        extracted_frames = []
        frame_count = 0
        
        print(f"Processing video: {video_name} ({total_frames} frames, {fps:.1f} fps)")
        
        with tqdm(total=min(total_frames, self.max_frames_per_video), 
                  desc=f"Extracting frames from {video_name}") as pbar:
            
            while cap.isOpened() and len(extracted_frames) < self.max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % self.frame_interval == 0:
                    frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    # Save frame
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                    
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(extracted_frames)} frames from {video_name}")
        return extracted_frames
    
    def detect_tools_yolo(self, image_path):
        """
        Detect surgical tools using YOLO model
        
        Args:
            image_path: Path to image
            
        Returns:
            List of detections with confidence scores
        """
        try:
            results = self.yolo_model(image_path, conf=self.yolo_confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf)
                        if confidence >= self.yolo_confidence_threshold:
                            detections.append({
                                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                                'confidence': confidence,
                                'class_id': int(box.cls),
                                'model': 'yolo'
                            })
            
            return detections
        except Exception as e:
            print(f"Error in YOLO detection for {image_path}: {e}")
            return []
    
    def detect_tools_detr(self, image_path):
        """
        Detect surgical tools using DETR model
        
        Args:
            image_path: Path to image
            
        Returns:
            List of detections with confidence scores
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.detr_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.detr_confidence_threshold
            )
            
            detections = []
            for result in results:
                scores = result['scores'].cpu().numpy()
                boxes = result['boxes'].cpu().numpy()
                labels = result['labels'].cpu().numpy()
                
                for score, box, label in zip(scores, boxes, labels):
                    if score >= self.detr_confidence_threshold:
                        detections.append({
                            'bbox': box.tolist(),
                            'confidence': float(score),
                            'class_id': int(label),
                            'model': 'detr'
                        })
            
            return detections
        except Exception as e:
            print(f"Error in DETR detection for {image_path}: {e}")
            return []
    
    def is_background_frame(self, image_path):
        """
        Determine if a frame is a background frame (no tools detected)
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (is_background, yolo_detections, detr_detections)
        """
        yolo_detections = self.detect_tools_yolo(image_path)
        detr_detections = self.detect_tools_detr(image_path)
        
        # Frame is background if BOTH models detect no tools
        is_background = len(yolo_detections) == 0 and len(detr_detections) == 0
        
        return is_background, yolo_detections, detr_detections
    
    def analyze_content_richness(self, image_path):
        """
        Analyze content richness using HOG features
        
        Args:
            image_path: Path to image
            
        Returns:
            Content richness score
        """
        try:
            hog_features = self.hog_extractor.extract_features(image_path)
            if hog_features:
                # Use edge intensity as content richness measure
                return hog_features.get('mean_edge', 0.0)
            return 0.0
        except Exception as e:
            print(f"Error analyzing content richness for {image_path}: {e}")
            return 0.0
    
    def process_videos(self, videos_dir):
        """
        Process all videos in the directory
        
        Args:
            videos_dir: Directory containing videos
        """
        videos_path = Path(videos_dir)
        video_files = list(videos_path.glob('*.mp4'))
        
        if not video_files:
            print(f"No MP4 files found in {videos_dir}")
            return
        
        print(f"Found {len(video_files)} video files to process")
        
        all_extracted_frames = []
        background_frames = []
        detection_results = []
        
        self.stats['total_videos'] = len(video_files)
        
        for video_file in video_files:
            print(f"\n{'='*60}")
            print(f"Processing video: {video_file.name}")
            print(f"{'='*60}")
            
            # Extract frames (temporary directory)
            temp_frames_dir = self.output_dir / 'temp_extracted_frames'
            temp_frames_dir.mkdir(exist_ok=True)
            extracted_frames = self.extract_frames_from_video(
                video_file, 
                temp_frames_dir
            )
            
            all_extracted_frames.extend(extracted_frames)
            self.stats['total_frames_extracted'] += len(extracted_frames)
            
            # Process each frame for background detection
            video_background_frames = []
            video_detection_results = []
            
            print(f"Analyzing {len(extracted_frames)} frames for background detection...")
            
            for frame_path in tqdm(extracted_frames, desc="Background detection"):
                is_background, yolo_det, detr_det = self.is_background_frame(frame_path)
                
                # Record detection results
                result = {
                    'frame_path': frame_path,
                    'is_background': is_background,
                    'yolo_detections': yolo_det,
                    'detr_detections': detr_det,
                    'video_name': video_file.stem
                }
                
                video_detection_results.append(result)
                detection_results.append(result)
                
                if is_background:
                    # Analyze content richness
                    content_score = self.analyze_content_richness(frame_path)
                    
                    video_background_frames.append({
                        'frame_path': frame_path,
                        'content_score': content_score
                    })
            
            background_frames.extend(video_background_frames)
            self.stats['total_background_frames'] += len(video_background_frames)
            
            # Save video-specific results
            video_results = {
                'video_name': video_file.stem,
                'total_frames': len(extracted_frames),
                'background_frames': len(video_background_frames),
                'detection_results': video_detection_results
            }
            
            self.stats['video_details'].append(video_results)
            
        
        print(f"\n{'='*60}")
        print(f"FRAME EXTRACTION AND BACKGROUND DETECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos processed: {self.stats['total_videos']}")
        print(f"Total frames extracted: {self.stats['total_frames_extracted']}")
        print(f"Total background frames: {self.stats['total_background_frames']}")
        print(f"Background ratio: {self.stats['total_background_frames']/self.stats['total_frames_extracted']:.3f}")
        
        # Continue with DINO-based selection
        if background_frames:
            self.select_significant_frames(background_frames)
        else:
            print("No background frames found!")
    
    def select_significant_frames(self, background_frames):
        """
        Select most significant frames using DINO clustering
        
        Args:
            background_frames: List of background frame information
        """
        print(f"\n{'='*60}")
        print(f"DINO-BASED FRAME SELECTION")
        print(f"{'='*60}")
        
        if len(background_frames) == 0:
            print("No background frames to process!")
            return
        
        # Create temporary directory for background frames
        temp_bg_dir = self.output_dir / 'temp_background_frames'
        temp_bg_dir.mkdir(exist_ok=True)
        
        # Copy background frames to temporary directory
        print("Preparing background frames for DINO analysis...")
        for frame_info in background_frames:
            frame_path = frame_info['frame_path']
            frame_name = Path(frame_path).name
            temp_frame_path = temp_bg_dir / frame_name
            shutil.copy2(frame_path, temp_frame_path)
        
        # Extract DINO features
        print("Extracting DINO features from background frames...")
        
        temp_features_dir = self.output_dir / 'temp_features'
        temp_features_dir.mkdir(exist_ok=True)
        dino_features_file = temp_features_dir / 'dino_features.pkl'
        
        features_data = self.dino_extractor.process_image_directory(
            str(temp_bg_dir),
            str(dino_features_file)
        )
        
        if not features_data:
            print("Failed to extract DINO features!")
            return
        
        # Perform clustering
        print("Performing DINO-based clustering...")
        
        clusterer = DINOClusterer()
        
        # Load features
        with open(dino_features_file, 'rb') as f:
            dino_data = pickle.load(f)
        
        # Prepare features for clustering
        features_array = np.array([f['global_features'] for f in dino_data['features']])
        image_paths = [f['image_path'] for f in dino_data['features']]
        
        # Determine number of clusters (aim for 20-30 clusters)
        n_clusters = min(20, max(5, len(background_frames) // 10))
        frames_per_cluster = max(1, min(5, len(background_frames) // n_clusters))
        
        print(f"Using {n_clusters} clusters with {frames_per_cluster} frames per cluster")
        
        # Perform clustering
        cluster_labels = clusterer.perform_clustering(
            features_array, 
            n_clusters=n_clusters,
            method='kmeans'
        )
        
        # Select representative frames
        selected_frames = clusterer.select_representative_frames(
            image_paths,
            features_array,
            cluster_labels,
            frames_per_cluster=frames_per_cluster
        )
        
        # Save selected frames info
        self.stats['total_selected_frames'] = len(selected_frames)
        
        print(f"Selected {len(selected_frames)} most informative background frames")
        
        # Save clustering results
        clustering_results = {
            'n_clusters': n_clusters,
            'frames_per_cluster': frames_per_cluster,
            'total_background_frames': len(background_frames),
            'selected_frames': selected_frames,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_stats': clusterer.get_cluster_statistics(cluster_labels)
        }
        
        # Save clustering results to temp directory (optional)
        temp_results_path = self.output_dir / 'temp_clustering_results.json'
        with open(temp_results_path, 'w') as f:
            json.dump(clustering_results, f, indent=2)
        
        print(f"Selected {len(selected_frames)} significant background frames")
        
        # Create DETR-ready dataset
        self.create_detr_dataset(selected_frames)
        
        # Clean up temporary directories
        self._cleanup_temp_directories()
    
    def create_detr_dataset(self, selected_frames):
        """
        Create DETR-ready dataset with background annotations
        
        Args:
            selected_frames: List of selected frame paths
        """
        print(f"\n{'='*60}")
        print(f"CREATING DETR-READY DATASET")
        print(f"{'='*60}")
        
        # Split frames into train/val
        np.random.seed(42)
        shuffled_frames = selected_frames.copy()
        np.random.shuffle(shuffled_frames)
        
        train_ratio = 0.8
        split_idx = int(len(shuffled_frames) * train_ratio)
        train_frames = shuffled_frames[:split_idx]
        val_frames = shuffled_frames[split_idx:]
        
        print(f"Train frames: {len(train_frames)}")
        print(f"Val frames: {len(val_frames)}")
        
        # Copy frames to dataset directories
        for frame_path in train_frames:
            frame_name = Path(frame_path).name
            dst_path = self.output_dir / 'train' / frame_name
            shutil.copy2(frame_path, dst_path)
        
        for frame_path in val_frames:
            frame_name = Path(frame_path).name
            dst_path = self.output_dir / 'val' / frame_name
            shutil.copy2(frame_path, dst_path)
        
        # Create COCO-style annotations
        self._create_coco_annotations(train_frames, 'train')
        self._create_coco_annotations(val_frames, 'val')
        
        print("DETR-ready dataset created successfully!")
    
    def _create_coco_annotations(self, frame_paths, split):
        """Create COCO-style annotations for background frames"""
        images = []
        
        for img_id, frame_path in enumerate(frame_paths, 1):
            frame_name = Path(frame_path).name
            
            # Get image dimensions
            img = cv2.imread(frame_path)
            if img is not None:
                height, width, _ = img.shape
                
                images.append({
                    "id": img_id,
                    "file_name": frame_name,
                    "width": width,
                    "height": height
                })
        
        # Create COCO structure for background frames (no annotations)
        coco_data = {
            "info": {
                "description": "Background frames for DETR training - no surgical tools present",
                "version": "1.0",
                "year": 2025,
                "contributor": "Intelligent Background Frame Selector",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": images,
            "annotations": [],  # Empty - background frames have no objects
            "categories": [
                {
                    "id": 1,
                    "name": "surgical_tool",
                    "supercategory": "medical_instrument"
                }
            ]
        }
        
        # Save annotations
        annotations_path = self.output_dir / 'annotations' / f'{split}_annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Created {split} annotations: {len(images)} images, 0 annotations (background only)")
    
    def _cleanup_temp_directories(self):
        """Clean up temporary directories created during processing"""
        temp_dirs = [
            self.output_dir / 'temp_extracted_frames',
            self.output_dir / 'temp_background_frames', 
            self.output_dir / 'temp_features'
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_dir}: {e}")
    
    def save_final_report(self):
        """Save final processing report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "processing_stats": self.stats,
            "configuration": {
                "yolo_model_path": self.yolo_model_path,
                "detr_model_path": self.detr_model_path,
                "yolo_confidence_threshold": self.yolo_confidence_threshold,
                "detr_confidence_threshold": self.detr_confidence_threshold,
                "frame_interval": self.frame_interval,
                "max_frames_per_video": self.max_frames_per_video,
                "device": str(self.device)
            },
            "summary": {
                "background_detection_rate": self.stats['total_background_frames'] / max(self.stats['total_frames_extracted'], 1),
                "selection_rate": self.stats['total_selected_frames'] / max(self.stats['total_background_frames'], 1),
                "overall_efficiency": self.stats['total_selected_frames'] / max(self.stats['total_frames_extracted'], 1)
            }
        }
        
        report_path = self.output_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"FINAL PROCESSING REPORT")
        print(f"{'='*60}")
        print(f"Total videos processed: {self.stats['total_videos']}")
        print(f"Total frames extracted: {self.stats['total_frames_extracted']}")
        print(f"Background frames detected: {self.stats['total_background_frames']}")
        print(f"Final selected frames: {self.stats['total_selected_frames']}")
        print(f"Background detection rate: {report['summary']['background_detection_rate']:.3f}")
        print(f"Selection efficiency: {report['summary']['selection_rate']:.3f}")
        print(f"Overall efficiency: {report['summary']['overall_efficiency']:.3f}")
        print(f"\nFinal report saved to: {report_path}")
        print(f"DETR Background Dataset created in: {self.output_dir}")
        print(f"  - Train frames: {self.output_dir / 'train'}")
        print(f"  - Val frames: {self.output_dir / 'val'}")
        print(f"  - Annotations: {self.output_dir / 'annotations'}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Intelligent Background Frame Selector for DETR Training'
    )
    
    # Required arguments
    parser.add_argument('--videos_dir', type=str, default='E:/Cataract/videos/micro',
                        help='Directory containing surgical videos')
    
    # Model paths
    parser.add_argument('--yolo_model_path', type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt',
                        help='Path to YOLO model')
    parser.add_argument('--detr_model_path', type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final',
                        help='Path to DETR model')
    
    # Output directory
    parser.add_argument('--output_dir', type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Datasets/Detr/Background',
                        help='Output directory for results')
    
    # Processing parameters
    parser.add_argument('--frame_interval', type=int, default=20,
                        help='Extract every N frames from videos')
    parser.add_argument('--max_frames_per_video', type=int, default=200,
                        help='Maximum frames to extract per video')
    parser.add_argument('--yolo_threshold', type=float, default=0.2,
                        help='YOLO confidence threshold')
    parser.add_argument('--detr_threshold', type=float, default=0.2,
                        help='DETR confidence threshold')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run models on')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.videos_dir):
        print(f"Error: Videos directory not found: {args.videos_dir}")
        return 1
    
    if not os.path.exists(args.yolo_model_path):
        print(f"Error: YOLO model not found: {args.yolo_model_path}")
        return 1
    
    if not os.path.exists(args.detr_model_path):
        print(f"Error: DETR model not found: {args.detr_model_path}")
        return 1
    
    # Initialize selector
    selector = IntelligentBackgroundFrameSelector(
        yolo_model_path=args.yolo_model_path,
        detr_model_path=args.detr_model_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Configure parameters
    selector.frame_interval = args.frame_interval
    selector.max_frames_per_video = args.max_frames_per_video
    selector.yolo_confidence_threshold = args.yolo_threshold
    selector.detr_confidence_threshold = args.detr_threshold
    
    print("INTELLIGENT BACKGROUND FRAME SELECTOR FOR DETR TRAINING")
    print("=" * 80)
    print(f"Videos directory: {args.videos_dir}")
    print(f"YOLO model: {args.yolo_model_path}")
    print(f"DETR model: {args.detr_model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frame interval: {args.frame_interval}")
    print(f"Max frames per video: {args.max_frames_per_video}")
    print(f"YOLO threshold: {args.yolo_threshold}")
    print(f"DETR threshold: {args.detr_threshold}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load models
    if not selector.load_models():
        print("Failed to load models!")
        return 1
    
    # Process videos
    try:
        selector.process_videos(args.videos_dir)
        selector.save_final_report()
        
        print("\n" + "=" * 80)
        print("INTELLIGENT BACKGROUND FRAME SELECTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review selected frames in the selected_frames directory")
        print("2. Use the DETR-ready dataset for background class training")
        print("3. Train DETR model with background class to reduce false positives")
        print("4. Evaluate model performance on clean eye images")
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())