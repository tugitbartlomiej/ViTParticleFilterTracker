#!/usr/bin/env python3
"""
Enhanced Background Frame Selector with DINO Information Richness
=================================================================

Advanced background frame selector that combines:
1. YOLO/DETR consensus for background detection
2. DINO information richness analysis for quality assessment  
3. Intelligent frame selection for optimal DETR training

Key improvements over original selector:
- Information theory-based frame quality assessment
- Attention entropy metrics for content richness
- Progressive selection strategy for training data
- Medical video-optimized frame scoring

Usage:
    python enhanced_background_selector.py --videos_dir "E:/Cataract/videos/micro" --use_dino_quality
    python enhanced_background_selector.py --single_video path/to/video.mp4 --info_threshold 0.6
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

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "DINO_Frame_Selection"))
sys.path.insert(0, str(project_root / "DINO_Frame_Selection" / "scripts"))
sys.path.insert(0, str(project_root.parent / "Annotators" / "Utils" / "SignificantImageSelector"))

# Import DINO components
from dino_information_analyzer import DINOInformationAnalyzer
from dino_feature_extractor import DINOFeatureExtractor
from dino_clustering_wrapper import DINOClusterer
from image_feature_extractor import ImageFeatureExtractor

# Import model libraries
from ultralytics import YOLO
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image

class EnhancedBackgroundSelector:
    """
    Enhanced background frame selector with DINO information richness analysis
    """
    
    def __init__(self,
                 yolo_model_path="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt",
                 detr_model_path="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/detr_inference_model.pth",
                 output_dir="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Datasets/Detr/Background",
                 device='auto',
                 use_dino_quality=True):
        """
        Initialize enhanced background selector
        
        Args:
            yolo_model_path: Path to YOLO model
            detr_model_path: Path to DETR model  
            output_dir: Output directory for results
            device: Device to run models on
            use_dino_quality: Whether to use DINO information analysis
        """
        self.yolo_model_path = yolo_model_path
        self.detr_model_path = detr_model_path
        self.output_dir = Path(output_dir)
        self.use_dino_quality = use_dino_quality
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Enhanced Background Selector - Device: {self.device}")
        print(f"DINO Quality Analysis: {'Enabled' if use_dino_quality else 'Disabled'}")
        
        # Initialize models
        self.yolo_model = None
        self.detr_model = None
        self.detr_processor = None
        self.dino_extractor = None
        self.dino_analyzer = None
        self.hog_extractor = None
        
        # Enhanced configuration
        self.config = {
            'yolo_threshold': 0.3,
            'detr_threshold': 0.8,  # Stricter for background detection
            'frame_interval': 30,
            'max_frames_per_video': 200,
            'info_quality_threshold': 0.4,  # Minimum information richness
            'min_background_frames': 10,
            'max_background_frames': 50,
            'selection_strategy': 'info_weighted'  # 'random', 'clustering', 'info_weighted'
        }
        
        # Enhanced statistics
        self.stats = {
            'total_videos': 0,
            'total_frames_extracted': 0,
            'total_background_frames': 0,
            'high_quality_backgrounds': 0,
            'selected_frames': 0,
            'avg_information_score': 0.0,
            'processing_time': 0,
            'video_details': [],
            'quality_distribution': {}
        }
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create enhanced output directory structure"""
        directories = [
            'background_frames',
            'background_frames/images',
            'background_frames/analysis',
            'train',
            'val', 
            'annotations',
            'quality_reports'
        ]
        
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def load_models(self):
        """Load all required models including DINO analyzer"""
        print("Loading enhanced model suite...")
        
        # Load YOLO model
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"✓ YOLO model loaded")
        except Exception as e:
            print(f"✗ YOLO loading error: {e}")
            return False
        
        # Load DETR model
        try:
            if str(self.detr_model_path).endswith('.pth'):
                checkpoint = torch.load(self.detr_model_path, map_location='cpu')
                self.detr_model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50",
                    num_labels=1,
                    ignore_mismatched_sizes=True
                )
                
                # Load checkpoint weights
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                else:
                    model_state_dict = checkpoint
                
                self.detr_model.load_state_dict(model_state_dict, strict=False)
                self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
                
            else:
                self.detr_model = DetrForObjectDetection.from_pretrained(str(self.detr_model_path))
                self.detr_processor = DetrImageProcessor.from_pretrained(str(self.detr_model_path))
            
            self.detr_model.to(self.device)
            self.detr_model.eval()
            print(f"✓ DETR model loaded")
            
        except Exception as e:
            print(f"✗ DETR loading error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Load DINO models
        if self.use_dino_quality:
            try:
                # DINO information analyzer
                self.dino_analyzer = DINOInformationAnalyzer(
                    model_name='dino_vits16',
                    device=str(self.device)
                )
                print("✓ DINO Information Analyzer loaded")
                
                # Standard DINO extractor for clustering
                self.dino_extractor = DINOFeatureExtractor(
                    model_name='dino_vits16',
                    device=str(self.device)
                )
                print("✓ DINO Feature Extractor loaded")
                
            except Exception as e:
                print(f"✗ DINO loading error: {e}")
                print("Falling back to non-DINO mode")
                self.use_dino_quality = False
        
        # Load HOG extractor
        try:
            self.hog_extractor = ImageFeatureExtractor(method='hog')
            print("✓ HOG Feature Extractor loaded")
        except Exception as e:
            print(f"✗ HOG loading error: {e}")
            return False
        
        print("✓ All models loaded successfully!")
        return True
    
    def extract_frames_from_video(self, video_path, output_dir):
        """Extract frames from video with enhanced logging"""
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
        print(f"Frame extraction interval: every {self.config['frame_interval']} frames")
        
        with tqdm(total=min(total_frames, self.config['max_frames_per_video']), 
                  desc=f"Extracting frames") as pbar:
            
            while cap.isOpened() and len(extracted_frames) < self.config['max_frames_per_video']:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.config['frame_interval'] == 0:
                    frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                    
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(extracted_frames)} frames from {video_name}")
        return extracted_frames
    
    def detect_tools_yolo(self, image_path):
        """YOLO tool detection"""
        try:
            results = self.yolo_model(image_path, conf=self.config['yolo_threshold'])
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf)
                        if confidence >= self.config['yolo_threshold']:
                            detections.append({
                                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                                'confidence': confidence,
                                'class_id': int(box.cls),
                                'model': 'yolo'
                            })
            
            return detections
        except Exception as e:
            print(f"YOLO detection error for {image_path}: {e}")
            return []
    
    def detect_tools_detr(self, image_path):
        """DETR tool detection with enhanced error handling"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.detr_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.config['detr_threshold']
            )
            
            detections = []
            for result in results:
                scores = result['scores'].cpu().numpy()
                boxes = result['boxes'].cpu().numpy()
                labels = result['labels'].cpu().numpy()
                
                for score, box, label in zip(scores, boxes, labels):
                    if score >= self.config['detr_threshold']:
                        detections.append({
                            'bbox': box.tolist(),
                            'confidence': float(score),
                            'class_id': int(label),
                            'model': 'detr'
                        })
            
            return detections
        except Exception as e:
            print(f"DETR detection error for {image_path}: {e}")
            return []
    
    def analyze_frame_quality(self, image_path):
        """Analyze frame quality using DINO information richness"""
        if not self.use_dino_quality:
            # Fallback to HOG-based analysis
            try:
                hog_features = self.hog_extractor.extract_features(image_path)
                return {
                    'information_score': hog_features.get('mean_edge', 0.5) if hog_features else 0.5,
                    'quality_source': 'hog_fallback'
                }
            except:
                return {'information_score': 0.5, 'quality_source': 'default'}
        
        # Use DINO information analysis
        try:
            analysis = self.dino_analyzer.extract_comprehensive_features(image_path)
            if analysis is not None:
                return {
                    'information_score': analysis['information_score'],
                    'attention_entropy': analysis.get('attention_entropy', 0.0),
                    'feature_variance': analysis.get('feature_variance', 0.0),
                    'spatial_coherence': analysis.get('spatial_coherence', 0.0),
                    'quality_source': 'dino'
                }
            else:
                return {'information_score': 0.0, 'quality_source': 'dino_failed'}
        except Exception as e:
            print(f"DINO quality analysis error for {image_path}: {e}")
            return {'information_score': 0.0, 'quality_source': 'error'}
    
    def is_background_frame(self, image_path):
        """Enhanced background detection with quality analysis"""
        # Tool detection
        yolo_detections = self.detect_tools_yolo(image_path)
        detr_detections = self.detect_tools_detr(image_path)
        
        # Background if both models detect no tools
        is_background = len(yolo_detections) == 0 and len(detr_detections) == 0
        
        # Quality analysis if background
        quality_info = None
        if is_background:
            quality_info = self.analyze_frame_quality(image_path)
        
        return is_background, yolo_detections, detr_detections, quality_info
    
    def process_single_video(self, video_path, video_name=None):
        """Process a single video with enhanced analysis"""
        if video_name is None:
            video_name = Path(video_path).stem
        
        print(f"\n{'='*60}")
        print(f"Enhanced Processing: {video_name}")
        print(f"{'='*60}")
        
        # Create temporary frame directory
        temp_frames_dir = self.output_dir / 'temp_frames'
        temp_frames_dir.mkdir(exist_ok=True)
        
        # Extract frames
        extracted_frames = self.extract_frames_from_video(video_path, temp_frames_dir)
        
        if not extracted_frames:
            print("No frames extracted!")
            return []
        
        # Analyze each frame
        background_candidates = []
        all_quality_scores = []
        
        print(f"Analyzing {len(extracted_frames)} frames...")
        
        for frame_path in tqdm(extracted_frames, desc="Background + quality analysis"):
            is_bg, yolo_det, detr_det, quality_info = self.is_background_frame(frame_path)
            
            frame_analysis = {
                'frame_path': frame_path,
                'is_background': is_bg,
                'yolo_detections': len(yolo_det),
                'detr_detections': len(detr_det),
                'video_name': video_name
            }
            
            if is_bg and quality_info:
                frame_analysis.update(quality_info)
                background_candidates.append(frame_analysis)
                all_quality_scores.append(quality_info['information_score'])
        
        # Statistics
        self.stats['total_frames_extracted'] += len(extracted_frames)
        self.stats['total_background_frames'] += len(background_candidates)
        
        if all_quality_scores:
            avg_quality = np.mean(all_quality_scores)
            self.stats['avg_information_score'] = avg_quality
            
            # Filter high-quality backgrounds
            high_quality_frames = [
                f for f in background_candidates 
                if f['information_score'] >= self.config['info_quality_threshold']
            ]
            self.stats['high_quality_backgrounds'] += len(high_quality_frames)
            
            print(f"Background frames found: {len(background_candidates)}")
            print(f"High-quality backgrounds: {len(high_quality_frames)} (≥{self.config['info_quality_threshold']:.2f})")
            print(f"Average information score: {avg_quality:.3f}")
            
            # Select optimal frames
            selected_frames = self.select_optimal_frames(
                high_quality_frames if high_quality_frames else background_candidates
            )
            
        else:
            selected_frames = background_candidates
            print(f"Background frames found: {len(background_candidates)} (no quality scoring)")
        
        # Clean up temp frames
        try:
            shutil.rmtree(temp_frames_dir)
        except:
            pass
        
        return selected_frames
    
    def select_optimal_frames(self, background_frames):
        """Select optimal frames using enhanced strategy"""
        if not background_frames:
            return []
        
        # Sort by information score
        sorted_frames = sorted(background_frames, 
                              key=lambda x: x.get('information_score', 0.0), 
                              reverse=True)
        
        # Apply selection limits
        max_frames = min(len(sorted_frames), self.config['max_background_frames'])
        min_frames = min(len(sorted_frames), self.config['min_background_frames'])
        
        if self.config['selection_strategy'] == 'info_weighted':
            # Select top frames by information score
            selected = sorted_frames[:max_frames]
            
        elif self.config['selection_strategy'] == 'clustering':
            # Use DINO clustering for diversity
            if self.use_dino_quality and len(sorted_frames) > min_frames:
                selected = self._cluster_based_selection(sorted_frames, max_frames)
            else:
                selected = sorted_frames[:max_frames]
                
        else:  # random
            np.random.shuffle(sorted_frames)
            selected = sorted_frames[:max_frames]
        
        # Ensure minimum frames
        if len(selected) < min_frames and len(sorted_frames) >= min_frames:
            selected = sorted_frames[:min_frames]
        
        self.stats['selected_frames'] += len(selected)
        
        return selected
    
    def _cluster_based_selection(self, frames, max_frames):
        """Use DINO clustering for diverse frame selection"""
        try:
            if len(frames) <= max_frames:
                return frames
            
            # Extract frame paths for clustering
            frame_paths = [f['frame_path'] for f in frames]
            
            # Create temp directory for clustering
            temp_cluster_dir = self.output_dir / 'temp_clustering'
            temp_cluster_dir.mkdir(exist_ok=True)
            
            # Copy frames to temp directory
            for i, frame_path in enumerate(frame_paths):
                temp_path = temp_cluster_dir / f"frame_{i:04d}.jpg"
                shutil.copy2(frame_path, temp_path)
            
            # Extract DINO features
            temp_features_file = temp_cluster_dir / 'features.pkl'
            features_data = self.dino_extractor.process_image_directory(
                str(temp_cluster_dir),
                str(temp_features_file)
            )
            
            if features_data:
                # Perform clustering
                clusterer = DINOClusterer()
                
                with open(temp_features_file, 'rb') as f:
                    dino_data = pickle.load(f)
                
                features_array = np.array([f['global_features'] for f in dino_data['features']])
                
                # Determine clusters
                n_clusters = min(max_frames, max(3, len(frames) // 5))
                frames_per_cluster = max(1, max_frames // n_clusters)
                
                cluster_labels = clusterer.perform_clustering(
                    features_array, 
                    n_clusters=n_clusters,
                    method='kmeans'
                )
                
                # Select representative frames with quality weighting
                selected_indices = []
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_frames = [frames[i] for i in range(len(frames)) if cluster_mask[i]]
                    
                    if cluster_frames:
                        # Sort cluster by quality and take best
                        cluster_frames.sort(key=lambda x: x.get('information_score', 0.0), reverse=True)
                        n_from_cluster = min(frames_per_cluster, len(cluster_frames))
                        selected_indices.extend(cluster_frames[:n_from_cluster])
                
                # Clean up
                shutil.rmtree(temp_cluster_dir)
                
                return selected_indices[:max_frames]
            
            else:
                # Fallback to quality-based selection
                return frames[:max_frames]
                
        except Exception as e:
            print(f"Clustering selection error: {e}")
            return frames[:max_frames]
    
    def save_background_dataset(self, selected_frames):
        """Save selected background frames as DETR dataset"""
        if not selected_frames:
            print("No frames to save!")
            return
        
        print(f"\nCreating background dataset with {len(selected_frames)} frames...")
        
        # Split into train/val
        np.random.seed(42)
        shuffled = selected_frames.copy()
        np.random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * 0.8)
        train_frames = shuffled[:split_idx]
        val_frames = shuffled[split_idx:]
        
        # Copy frames and create annotations
        self._save_split_data(train_frames, 'train')
        self._save_split_data(val_frames, 'val')
        
        # Save quality report
        self._save_quality_report(selected_frames)
        
        print(f"✓ Background dataset created:")
        print(f"  Train: {len(train_frames)} frames")
        print(f"  Val: {len(val_frames)} frames")
        print(f"  Location: {self.output_dir}")
    
    def _save_split_data(self, frames, split):
        """Save frames and create COCO annotations for split"""
        # Copy images
        for frame_info in frames:
            frame_path = frame_info['frame_path']
            frame_name = Path(frame_path).name
            dst_path = self.output_dir / split / frame_name
            shutil.copy2(frame_path, dst_path)
        
        # Create COCO annotations
        images = []
        for img_id, frame_info in enumerate(frames, 1):
            frame_path = frame_info['frame_path']
            frame_name = Path(frame_path).name
            
            img = cv2.imread(frame_path)
            if img is not None:
                height, width, _ = img.shape
                images.append({
                    "id": img_id,
                    "file_name": frame_name,
                    "width": width,
                    "height": height,
                    "information_score": frame_info.get('information_score', 0.0)
                })
        
        # COCO structure
        coco_data = {
            "info": {
                "description": f"High-quality background frames for DETR training ({split} set)",
                "version": "2.0",
                "year": 2025,
                "contributor": "Enhanced Background Selector with DINO",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": images,
            "annotations": [],  # No objects in background frames
            "categories": [{
                "id": 1,
                "name": "surgical_tool",
                "supercategory": "medical_instrument"
            }]
        }
        
        # Save annotations
        annotations_path = self.output_dir / 'annotations' / f'{split}_annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    def _save_quality_report(self, selected_frames):
        """Save detailed quality analysis report"""
        quality_scores = [f.get('information_score', 0.0) for f in selected_frames]
        
        report = {
            "analysis_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_selected_frames": len(selected_frames),
                "dino_analysis_enabled": self.use_dino_quality,
                "selection_strategy": self.config['selection_strategy']
            },
            "quality_statistics": {
                "mean_score": float(np.mean(quality_scores)) if quality_scores else 0.0,
                "std_score": float(np.std(quality_scores)) if quality_scores else 0.0,
                "min_score": float(np.min(quality_scores)) if quality_scores else 0.0,
                "max_score": float(np.max(quality_scores)) if quality_scores else 0.0,
                "percentiles": {
                    "90th": float(np.percentile(quality_scores, 90)) if quality_scores else 0.0,
                    "75th": float(np.percentile(quality_scores, 75)) if quality_scores else 0.0,
                    "50th": float(np.percentile(quality_scores, 50)) if quality_scores else 0.0,
                    "25th": float(np.percentile(quality_scores, 25)) if quality_scores else 0.0
                }
            },
            "configuration": self.config,
            "processing_stats": self.stats,
            "frame_details": selected_frames
        }
        
        report_path = self.output_dir / 'quality_reports' / 'background_selection_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Quality report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Background Frame Selector with DINO Information Analysis'
    )
    
    parser.add_argument('--videos_dir', type=str,
                        help='Directory containing videos')
    parser.add_argument('--single_video', type=str,
                        help='Process single video file')
    parser.add_argument('--output_dir', type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Datasets/Detr/Background',
                        help='Output directory')
    
    # Model paths
    parser.add_argument('--yolo_model', type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt')
    parser.add_argument('--detr_model', type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/detr_inference_model.pth')
    
    # Enhanced parameters
    parser.add_argument('--use_dino_quality', action='store_true', default=True,
                        help='Use DINO information richness analysis')
    parser.add_argument('--info_threshold', type=float, default=0.4,
                        help='Minimum information richness threshold')
    parser.add_argument('--yolo_threshold', type=float, default=0.3,
                        help='YOLO confidence threshold')
    parser.add_argument('--detr_threshold', type=float, default=0.8,
                        help='DETR confidence threshold')
    parser.add_argument('--max_background_frames', type=int, default=50,
                        help='Maximum background frames to select')
    
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.videos_dir and not args.single_video:
        print("Error: Must specify either --videos_dir or --single_video")
        return 1
    
    # Initialize enhanced selector
    selector = EnhancedBackgroundSelector(
        yolo_model_path=args.yolo_model,
        detr_model_path=args.detr_model,
        output_dir=args.output_dir,
        device=args.device,
        use_dino_quality=args.use_dino_quality
    )
    
    # Update configuration
    selector.config.update({
        'yolo_threshold': args.yolo_threshold,
        'detr_threshold': args.detr_threshold,
        'info_quality_threshold': args.info_threshold,
        'max_background_frames': args.max_background_frames
    })
    
    print("=== ENHANCED BACKGROUND FRAME SELECTOR ===")
    print(f"DINO Quality Analysis: {args.use_dino_quality}")
    print(f"Info Threshold: {args.info_threshold}")
    print(f"YOLO Threshold: {args.yolo_threshold}")
    print(f"DETR Threshold: {args.detr_threshold}")
    print("=" * 50)
    
    # Load models
    if not selector.load_models():
        print("Failed to load models!")
        return 1
    
    # Process videos
    try:
        all_selected_frames = []
        
        if args.single_video:
            # Process single video
            if os.path.exists(args.single_video):
                selected = selector.process_single_video(args.single_video)
                all_selected_frames.extend(selected)
            else:
                print(f"Video not found: {args.single_video}")
                return 1
                
        else:
            # Process video directory
            videos_path = Path(args.videos_dir)
            video_files = list(videos_path.glob('*.mp4'))
            
            if not video_files:
                print(f"No MP4 files found in {args.videos_dir}")
                return 1
            
            selector.stats['total_videos'] = len(video_files)
            
            for video_file in video_files:
                selected = selector.process_single_video(video_file)
                all_selected_frames.extend(selected)
        
        # Create final dataset
        if all_selected_frames:
            selector.save_background_dataset(all_selected_frames)
            
            print(f"\n=== FINAL RESULTS ===")
            print(f"Total background frames selected: {len(all_selected_frames)}")
            print(f"Average information score: {selector.stats['avg_information_score']:.3f}")
            print(f"High-quality ratio: {selector.stats['high_quality_backgrounds']}/{selector.stats['total_background_frames']}")
            print(f"Dataset location: {args.output_dir}")
            print("=" * 50)
            
            return 0
        else:
            print("No background frames selected!")
            return 1
            
    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())