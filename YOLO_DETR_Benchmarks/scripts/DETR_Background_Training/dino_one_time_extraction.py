#!/usr/bin/env python3
"""
ONE-TIME DINO EXTRACTION FROM E:\PicsOnly
=========================================

Jednorazowa ekstrakcja najciekawszych frames używając DINO + YOLO/DETR classification.
Tworzy reusable test dataset dla wielokrotnego testowania strategii.

Usage: python dino_one_time_extraction.py
"""

import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel
from sklearn.cluster import KMeans
from datetime import datetime
from ultralytics import YOLO
from transformers import DetrForObjectDetection, DetrImageProcessor
import warnings
warnings.filterwarnings('ignore')

class DINOTestDatasetExtractor:
    def __init__(self):
        self.config = {
            'source_dir': "E:/PicsOnly",  # Source images
            'output_dir': "test_dino_dataset",  # Reusable test dataset
            'frames_target': 200,  # Total frames to extract
            'clustering_threshold': 0.5,  # More diverse clusters
            'min_cluster_size': 3,
            'yolo_model_path': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolov8n_tooltip_best_from_vscode.pt",
            'detr_model_path': "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/detr_inference_model.pth",
            'yolo_threshold': 0.4,
            'detr_threshold': 0.7,
            'train_val_split': 0.8
        }
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup output directories
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        self.interesting_frames_dir = self.output_dir / "interesting_frames"
        self.train_dir = self.interesting_frames_dir / "train"
        self.val_dir = self.interesting_frames_dir / "val"
        self.annotations_dir = self.interesting_frames_dir / "annotations"
        
        for dir_path in [self.train_dir, self.val_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        # Tracking
        self.extraction_log = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'frames_extracted': 0,
            'tooltip_frames': [],
            'background_frames': [],
            'uncertain_frames': []
        }
    
    def _init_models(self):
        """Initialize DINO, YOLO and DETR models"""
        print("Loading models...")
        
        # DINO for feature extraction
        self.dino_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        self.dino_model = ViTModel.from_pretrained('facebook/dino-vitb16').to(self.device)
        self.dino_model.eval()
        
        # YOLO for tooltip detection
        try:
            self.yolo_model = YOLO(self.config['yolo_model_path'])
            print("YOLO model loaded")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            self.yolo_model = None
        
        # DETR for tooltip detection
        try:
            self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detr_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50", 
                num_labels=1, 
                ignore_mismatched_sizes=True
            )
            
            checkpoint = torch.load(self.config['detr_model_path'], map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.detr_model.load_state_dict(state_dict, strict=False)
            self.detr_model.to(self.device)
            self.detr_model.eval()
            print("DETR model loaded")
        except Exception as e:
            print(f"Warning: Could not load DETR model: {e}")
            self.detr_model = None
    
    def extract_dino_features(self, image_path):
        """Extract DINO features from image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return features.flatten()
    
    def classify_frame(self, image_path):
        """Classify frame as TOOLTIP, BACKGROUND or UNCERTAIN using YOLO+DETR"""
        
        # Default to BACKGROUND if models not available
        if self.yolo_model is None and self.detr_model is None:
            return "BACKGROUND"
        
        yolo_detected = False
        detr_detected = False
        
        # YOLO detection
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(str(image_path), conf=self.config['yolo_threshold'])
                if len(results[0].boxes) > 0:
                    yolo_detected = True
            except:
                pass
        
        # DETR detection  
        if self.detr_model is not None:
            try:
                image = Image.open(image_path).convert('RGB')
                inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.detr_model(**inputs)
                
                # Post-process
                target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
                results = self.detr_processor.post_process_object_detection(
                    outputs, 
                    target_sizes=target_sizes,
                    threshold=self.config['detr_threshold']
                )[0]
                
                if len(results["scores"]) > 0:
                    detr_detected = True
            except:
                pass
        
        # Classification logic (based on DINO paper strategy)
        if yolo_detected:
            return "TOOLTIP"  # YOLO detected -> TOOLTIP FRAME
        elif not yolo_detected and not detr_detected:
            return "BACKGROUND"  # Neither detected -> BACKGROUND FRAME
        else:
            return "UNCERTAIN"  # DETR only -> UNCERTAIN (skip)
    
    def extract_diverse_frames(self):
        """Extract diverse frames from E:\PicsOnly"""
        print(f"\nExtracting frames from {self.config['source_dir']}...")
        
        source_dir = Path(self.config['source_dir'])
        if not source_dir.exists():
            print(f"ERROR: Source directory not found: {source_dir}")
            return []
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(source_dir.glob(ext)))
        
        if not all_images:
            print(f"ERROR: No images found in {source_dir}")
            return []
        
        print(f"Found {len(all_images)} images in source directory")
        
        # Sample images if too many
        if len(all_images) > self.config['frames_target'] * 2:
            np.random.seed(42)
            all_images = np.random.choice(all_images, self.config['frames_target'] * 2, replace=False)
        
        # Extract DINO features for clustering
        print("Extracting DINO features...")
        features_list = []
        valid_images = []
        
        for i, img_path in enumerate(all_images):
            if i % 10 == 0:
                print(f"Processing {i}/{len(all_images)}...")
            
            try:
                features = self.extract_dino_features(img_path)
                features_list.append(features)
                valid_images.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if not features_list:
            print("ERROR: No features extracted")
            return []
        
        features_array = np.array(features_list)
        
        # Clustering for diversity
        print(f"\nClustering {len(features_array)} images...")
        n_clusters = min(20, len(features_array) // 5)  # More clusters for diversity
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_array)
        
        # Select diverse samples from each cluster
        selected_images = []
        samples_per_cluster = max(self.config['frames_target'] // n_clusters, 10)
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Select up to samples_per_cluster from each cluster
                n_samples = min(samples_per_cluster, len(cluster_indices))
                selected_indices = np.random.choice(cluster_indices, n_samples, replace=False)
                
                for idx in selected_indices:
                    selected_images.append(valid_images[idx])
        
        # Limit to target number
        if len(selected_images) > self.config['frames_target']:
            selected_images = selected_images[:self.config['frames_target']]
        
        print(f"Selected {len(selected_images)} diverse images")
        return selected_images
    
    def process_and_save_frames(self, selected_images):
        """Process selected frames and save with classification"""
        print("\nClassifying and saving frames...")
        
        all_frames_data = []
        
        for i, img_path in enumerate(selected_images):
            if i % 10 == 0:
                print(f"Processing {i}/{len(selected_images)}...")
            
            # Classify frame
            frame_type = self.classify_frame(img_path)
            
            # Generate new name
            frame_name = f"frame_{i:05d}_{frame_type.lower()}.jpg"
            
            # Initially save to train dir
            dest_path = self.train_dir / frame_name
            
            # Copy image
            shutil.copy2(img_path, dest_path)
            
            # Track frame data
            frame_data = {
                'path': dest_path,
                'name': frame_name,
                'type': frame_type,
                'original': str(img_path)
            }
            all_frames_data.append(frame_data)
            
            # Log frame type
            if frame_type == "TOOLTIP":
                self.extraction_log['tooltip_frames'].append(frame_name)
            elif frame_type == "BACKGROUND":
                self.extraction_log['background_frames'].append(frame_name)
            else:
                self.extraction_log['uncertain_frames'].append(frame_name)
        
        # Split train/val
        np.random.seed(42)
        np.random.shuffle(all_frames_data)
        
        split_idx = int(len(all_frames_data) * self.config['train_val_split'])
        train_frames = all_frames_data[:split_idx]
        val_frames = all_frames_data[split_idx:]
        
        # Move val frames
        for frame_data in val_frames:
            src = frame_data['path']
            dst = self.val_dir / frame_data['name']
            shutil.move(str(src), str(dst))
            frame_data['path'] = dst
        
        print(f"\nFrame distribution:")
        print(f"  Total: {len(all_frames_data)} frames")
        print(f"  Train: {len(train_frames)} frames")
        print(f"  Val: {len(val_frames)} frames")
        print(f"  Tooltip frames: {len(self.extraction_log['tooltip_frames'])}")
        print(f"  Background frames: {len(self.extraction_log['background_frames'])}")
        print(f"  Uncertain frames: {len(self.extraction_log['uncertain_frames'])}")
        
        return train_frames, val_frames
    
    def create_annotations(self, train_frames, val_frames):
        """Create COCO format annotations"""
        print("\nCreating COCO annotations...")
        
        def create_coco_dict(frames, split_name):
            images = []
            annotations = []
            ann_id = 1
            
            for i, frame_data in enumerate(frames, 1):
                # Load image to get dimensions
                img = cv2.imread(str(frame_data['path']))
                if img is None:
                    continue
                    
                height, width = img.shape[:2]
                
                # Add image entry
                images.append({
                    "id": i,
                    "file_name": frame_data['name'],
                    "width": width,
                    "height": height,
                    "frame_type": frame_data['type']
                })
                
                # Add dummy annotation for tooltip frames
                if frame_data['type'] == "TOOLTIP":
                    annotations.append({
                        "id": ann_id,
                        "image_id": i,
                        "category_id": 1,
                        "bbox": [width//4, height//4, width//2, height//2],  # Dummy bbox
                        "area": (width//2) * (height//2),
                        "iscrowd": 0
                    })
                    ann_id += 1
            
            return {
                "info": {
                    "description": f"DINO extracted test dataset ({split_name})",
                    "version": "1.0",
                    "year": 2024,
                    "date_created": datetime.now().isoformat()
                },
                "images": images,
                "annotations": annotations,
                "categories": [{
                    "id": 1,
                    "name": "surgical_tool",
                    "supercategory": "medical_instrument"
                }]
            }
        
        # Create and save annotations
        train_annotations = create_coco_dict(train_frames, "train")
        val_annotations = create_coco_dict(val_frames, "val")
        
        train_ann_path = self.annotations_dir / "train_annotations.json"
        val_ann_path = self.annotations_dir / "val_annotations.json"
        
        with open(train_ann_path, 'w') as f:
            json.dump(train_annotations, f, indent=2)
        
        with open(val_ann_path, 'w') as f:
            json.dump(val_annotations, f, indent=2)
        
        print(f"Annotations saved to {self.annotations_dir}")
    
    def save_metadata(self):
        """Save extraction metadata"""
        self.extraction_log['frames_extracted'] = (
            len(self.extraction_log['tooltip_frames']) + 
            len(self.extraction_log['background_frames'])
        )
        
        metadata_path = self.output_dir / "dino_extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.extraction_log, f, indent=2)
        
        print(f"\nMetadata saved to {metadata_path}")
    
    def run(self):
        """Run complete DINO extraction pipeline"""
        print("="*60)
        print("ONE-TIME DINO EXTRACTION FOR STRATEGY TESTING")
        print("="*60)
        
        # Check if dataset already exists
        if (self.train_dir / "frame_00000_background.jpg").exists():
            print("\nWARNING: Test dataset already exists!")
            response = input("Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Using existing dataset.")
                return
        
        # Extract diverse frames
        selected_images = self.extract_diverse_frames()
        
        if not selected_images:
            print("ERROR: No images extracted")
            return
        
        # Process and save frames
        train_frames, val_frames = self.process_and_save_frames(selected_images)
        
        # Create annotations
        self.create_annotations(train_frames, val_frames)
        
        # Save metadata
        self.save_metadata()
        
        print("\n" + "="*60)
        print("DINO EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"Dataset saved to: {self.output_dir}")
        print(f"Ready for strategy testing!")
        print("="*60)

def main():
    extractor = DINOTestDatasetExtractor()
    extractor.run()

if __name__ == "__main__":
    main()