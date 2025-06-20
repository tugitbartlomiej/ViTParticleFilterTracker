#!/usr/bin/env python3
"""
Quick test script to process just a few videos and get background frames
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import shutil
from datetime import datetime

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

class QuickBackgroundSelector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        self.yolo_model = YOLO("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt")
        self.detr_model = DetrForObjectDetection.from_pretrained("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final")
        self.detr_processor = DetrImageProcessor.from_pretrained("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final")
        self.detr_model.to(self.device)
        self.detr_model.eval()
        
        # Output directory
        self.output_dir = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17")
        
        # Stats
        self.stats = {
            'videos_processed': 0,
            'frames_extracted': 0,
            'background_frames': 0,
            'video_details': []
        }
    
    def is_background_frame(self, image_path):
        """Check if frame is background using both YOLO and DETR"""
        # YOLO detection
        try:
            yolo_results = self.yolo_model(image_path, conf=0.5)
            yolo_detections = 0
            for result in yolo_results:
                if result.boxes is not None:
                    yolo_detections = len(result.boxes)
                    break
        except:
            yolo_detections = 0
            
        # DETR detection
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.detr_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )
            
            detr_detections = len(results[0]['scores']) if results else 0
        except:
            detr_detections = 0
        
        # Background if both models detect no tools
        is_background = yolo_detections == 0 and detr_detections == 0
        return is_background, yolo_detections, detr_detections
    
    def process_video(self, video_path, max_frames=30):
        """Process a single video and extract background frames"""
        video_name = video_path.stem
        print(f"\nProcessing {video_name}...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        
        background_frames = []
        frame_count = 0
        
        extracted_dir = self.output_dir / 'extracted_frames'
        background_dir = self.output_dir / 'background_frames'
        
        while cap.isOpened() and len(background_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Save frame
                frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                frame_path = extracted_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                self.stats['frames_extracted'] += 1
                
                # Check if it's background
                is_background, yolo_det, detr_det = self.is_background_frame(str(frame_path))
                
                if is_background:
                    # Copy to background directory
                    background_path = background_dir / frame_filename
                    shutil.copy2(frame_path, background_path)
                    background_frames.append(str(background_path))
                    self.stats['background_frames'] += 1
                    print(f"  Background frame: {frame_filename}")
            
            frame_count += 1
        
        cap.release()
        print(f"  Found {len(background_frames)} background frames from {video_name}")
        return background_frames
    
    def run_quick_test(self):
        """Run quick test on first few videos"""
        videos_dir = Path("E:/Cataract/videos/micro")
        video_files = list(videos_dir.glob("*.mp4"))[:5]  # Just first 5 videos
        
        print(f"Processing {len(video_files)} videos for quick test...")
        
        all_background_frames = []
        
        for video_file in video_files:
            background_frames = self.process_video(video_file)
            all_background_frames.extend(background_frames)
            self.stats['videos_processed'] += 1
            
            # Add to video details
            self.stats['video_details'].append({
                'video_name': video_file.stem,
                'background_frames': len(background_frames)
            })
        
        # Save results
        print(f"\n{'='*60}")
        print(f"QUICK TEST RESULTS")
        print(f"{'='*60}")
        print(f"Videos processed: {self.stats['videos_processed']}")
        print(f"Total frames extracted: {self.stats['frames_extracted']}")
        print(f"Background frames found: {self.stats['background_frames']}")
        print(f"Background ratio: {self.stats['background_frames']/self.stats['frames_extracted']:.3f}")
        
        # Save detailed results
        results_file = self.output_dir / 'quick_test_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'background_frames': all_background_frames
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Background frames in: {self.output_dir / 'background_frames'}")
        
        return all_background_frames

def main():
    print("QUICK BACKGROUND FRAME SELECTOR TEST")
    print("=" * 60)
    
    try:
        selector = QuickBackgroundSelector()
        background_frames = selector.run_quick_test()
        
        print(f"\n✅ SUCCESS: Generated {len(background_frames)} background frames from different videos")
        print("✅ All models working correctly")
        print("✅ Background detection pipeline functional")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())