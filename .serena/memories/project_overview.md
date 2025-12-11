# ViTParticleFilterTracker - Project Overview

## Purpose
Advanced surgical tool detection and tracking system for PhD research.
Specializes in cataract surgery video analysis using multiple AI models (YOLO, DETR, DINO).

## Core Problem Solved
- Detecting surgical instruments in operation videos
- Reducing false positives on clean eye images
- Background frame extraction for improved training

## Tech Stack
- **Python**: 3.11 (required - use `py -3.11`)
- **Deep Learning**: PyTorch, torchvision
- **Object Detection**: YOLO (ultralytics), DETR (HuggingFace transformers)
- **Feature Extraction**: DINO (facebook/dino-vitb16)
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, pycocotools
- **Visualization**: matplotlib, seaborn
- **Config**: YAML (pyyaml)
- **Progress**: tqdm

## Key Dependencies
```
torch
torchvision
ultralytics
transformers
pycocotools
pyyaml
opencv-python
numpy
matplotlib
seaborn
tqdm
timm
```

## Main Components

### 1. Annotators/
Detection and annotation systems:
- **DetrAnnotator/** - DETR training and inference
- **Yolo/** - YOLO detection and prediction
- **DeepSortYolo/** - Object tracking with DeepSORT
- **TimesFormer/** - Temporal video analysis
- **OpencvTrackerAnnotator/** - OpenCV-based tracking
- **Datasets/** - Training datasets (COCO format for DETR, YOLO format)

### 2. YOLO_DETR_Benchmarks/
Main research and benchmarking module:
- **scripts/DETR_Background_Training/** - Background training pipeline
- **Intelligent_Background_Selector_2025-07-17/** - Frame selection
- **DINO_Frame_Selection/** - DINO-based frame clustering
- **Advanced_Analysis/** - Performance analysis
- **Benchmarks/** - Comparison tests

### 3. BackgroundFinetuned/
Fine-tuned DETR models with background awareness:
- main.py - Main entry point
- main_train.py - Training script
- main_generate.py - Generation/inference

### 4. BareDetr/
Minimalist DETR implementation:
- Clean transformer-based object detection
- Modular design

### 5. Eden/
Cluster training infrastructure:
- SLURM scripts for HPC clusters
- Dataset preparation utilities
- COCO ↔ YOLO format converters

## Data Locations
- **Videos**: `E:/Cataract/videos/` (50+ surgical videos)
- **YOLO Dataset**: `Annotators/Datasets/Yolo/`
- **DETR Dataset**: `Annotators/Datasets/Detr/` (COCO format)
- **Models**: `YOLO_DETR_Benchmarks/models/`

## Pipeline Architecture
The main training pipeline follows 5 stages:
1. **DINO Extraction** - Extract frames using DINO clustering
2. **Dataset Mixing** - Combine tooltip (70%) and background (30%) frames
3. **Model Preparation** - Load pretrained DETR
4. **Gentle Training** - Fine-tune with low learning rate (1e-6)
5. **Validation** - Evaluate on test set
