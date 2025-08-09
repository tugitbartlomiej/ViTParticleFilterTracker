#!/usr/bin/env python3
"""
Single Video Test for DETR Background Training Pipeline
======================================================

Quick test script to process just one video and verify the complete pipeline works.
"""

import os
import shutil
from pathlib import Path

# Create single video directory for test
test_video_dir = Path("E:/Cataract/videos/test_single")
test_video_dir.mkdir(exist_ok=True)

# Copy one video for testing
source_video = Path("E:/Cataract/videos/micro/test01.mp4")
target_video = test_video_dir / "test01.mp4"

if source_video.exists() and not target_video.exists():
    print(f"Copying {source_video} to {target_video}")
    shutil.copy2(source_video, target_video)
    print("Test video ready")
else:
    print(f"Test video already exists: {target_video}")

# Create test config
config_content = """# Single Video Test Configuration
video_directory: "E:/Cataract/videos/test_single"
tooltip_dataset_path: "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Yolo/yolo_dataset_20250218/images/train"
tooltip_annotations_path: "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Yolo/yolo_dataset_20250218/coco_annotations_from_yolo_dataset_20250218.json"
original_model_checkpoint: "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/models/DETR/checkpoint_epoch_100.pth"
output_directory: "pipeline_output_single_test"

# DINO extraction parameters
dino_model: "facebook/dino-vitb16"
frames_per_video: 30
clustering_threshold: 0.7
min_cluster_size: 3

# Dataset mixing parameters  
tooltip_ratio: 0.7
background_ratio: 0.3
validation_split: 0.2

# Training parameters
gentle_lr: 1e-6
max_epochs: 1
batch_size: 1
gradient_accumulation_steps: 4

# Execution settings
python_executable: "py"
max_retries: 1
timeout_hours: 2
"""

config_path = Path("pipeline_config_single_test.yaml")
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"Created test config: {config_path}")
print(f"\nRun test with:\npy main_pipeline.py --config {config_path}")