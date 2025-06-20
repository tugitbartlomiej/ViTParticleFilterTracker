# DINO-Based Intelligent Frame Selection for DETR Training

This system uses DINO (Self-Supervised Vision Transformer) to intelligently select the most informative frames from surgical videos for DETR training. DINO's semantic understanding helps identify diverse and representative training samples, addressing the original DETR training problems.

## Background

The original DETR training suffered from:
- **Fixed predictions**: DETR predicts exactly 100 objects per image regardless of actual content
- **No background class**: Only "tool" class, no proper negative examples
- **False positives**: Many predictions on background regions
- **Poor performance**: Due to lack of diverse, representative training data

## Solution: DINO + Background Integration

This system provides:
1. **DINO-based selection**: Uses semantic understanding to select informative frames
2. **Background integration**: Combines with background frames for negative examples
3. **Intelligent clustering**: Groups similar frames and selects representatives
4. **Balanced datasets**: Creates proper train/val splits with diverse examples

## Directory Structure

```
DINO_Frame_Selection/
├── dino_frame_selector.py              # Main orchestration script
├── integrated_detr_dataset_creator.py  # Combines DINO + background frames
├── scripts/
│   ├── dino_feature_extractor.py       # DINO feature extraction
│   └── dino_clustering.py              # Clustering and selection
├── features/                           # DINO features storage
├── selected_frames/                    # Selected informative frames
└── detr_dataset/                       # Final DETR-ready dataset
```

## Key Features

### 1. DINO Feature Extraction
- Uses pre-trained DINO models (ViT-S/16, ViT-B/16, etc.)
- Extracts rich semantic features from surgical frames
- Captures spatial relationships and tool contexts
- Provides 384-768 dimensional feature vectors

### 2. Intelligent Clustering
- Groups frames by semantic similarity
- Selects representative frames from each cluster
- Multiple selection strategies:
  - **Centroid**: Closest to cluster center
  - **Diverse**: Maximum diversity within cluster
  - **Quality**: Based on feature quality metrics

### 3. Integration with Background Frames
- Combines DINO-selected frames with background frames
- Creates balanced positive/negative examples
- Addresses original DETR training limitations
- Generates proper COCO-style annotations

## Usage

### Quick Start

```bash
# Select frames from videos using DINO
python dino_frame_selector.py --video_dir "E:/Cataract/videos/micro" --output_dir "dino_results" --n_clusters 25 --frames_per_cluster 4

# Create integrated dataset with background frames
python integrated_detr_dataset_creator.py --dino_frames "dino_results/clustering_results/selected_frames" --background_frames "../Background_Extraction/background_frames" --output_dir "complete_detr_dataset"
```

### Advanced Usage

```bash
# Step 1: Extract frames from videos
python dino_frame_selector.py \
    --video_dir "E:/Cataract/videos/micro" \
    --output_dir "dino_selection" \
    --frame_interval 30 \
    --dino_model "dino_vitb16" \
    --n_clusters 30 \
    --frames_per_cluster 3 \
    --selection_method "diverse" \
    --create_detr_dataset

# Step 2: Create integrated dataset
python integrated_detr_dataset_creator.py \
    --dino_frames "dino_selection/clustering_results/selected_frames" \
    --background_frames "../Background_Extraction/background_frames" \
    --output_dir "integrated_dataset" \
    --balance_ratio 0.7 \
    --train_ratio 0.8
```

### Process from Existing Frames

```bash
# If you already have extracted frames
python dino_frame_selector.py \
    --input_dir "path/to/extracted/frames" \
    --output_dir "dino_results" \
    --n_clusters 20 \
    --frames_per_cluster 5
```

## Parameters

### DINO Frame Selector
- `--dino_model`: Model variant (`dino_vits16`, `dino_vitb16`, etc.)
- `--n_clusters`: Number of clusters for grouping frames
- `--frames_per_cluster`: Frames to select per cluster
- `--selection_method`: Selection strategy (`centroid`, `diverse`, `quality`)
- `--frame_interval`: Extract every N frames from videos
- `--max_images`: Limit number of images to process

### Integrated Dataset Creator
- `--balance_ratio`: Ratio of informative to background frames
- `--train_ratio`: Train/validation split ratio
- `--output_dir`: Directory for final integrated dataset

## Output Structure

### DINO Selection Results
```
dino_results/
├── extracted_frames/           # Extracted video frames
├── features/
│   ├── dino_features.pkl      # DINO feature vectors
│   └── dino_features_metadata.json
├── clustering_results/
│   ├── selected_frames/       # Selected representative frames
│   ├── clustering_results.json
│   ├── cluster_visualization.png
│   └── cluster_sizes.png
└── dino_selection_report.json
```

### Integrated Dataset
```
integrated_dataset/
├── train/                     # Training images
├── val/                       # Validation images
├── annotations/
│   ├── train_annotations.json # COCO-style annotations
│   └── val_annotations.json
└── dataset_report.json
```

## Technical Details

### DINO Models
- **dino_vits16**: ViT-Small with 16×16 patches (384-dim features)
- **dino_vitb16**: ViT-Base with 16×16 patches (768-dim features)
- **dino_vits8**: ViT-Small with 8×8 patches (higher resolution)

### Clustering Algorithms
- **K-means**: Primary clustering method
- **DBSCAN**: Alternative density-based clustering
- **PCA**: For visualization and dimensionality reduction

### Selection Strategies
- **Centroid**: Selects frames closest to cluster center
- **Diverse**: Maximizes diversity within clusters
- **Quality**: Based on feature norm and quality metrics

## Integration with Existing System

This system integrates with the existing SignificantImageSelector:

1. **Builds upon**: Previous YOLO annotation + visual feature approach
2. **Enhances with**: DINO's semantic understanding
3. **Combines with**: Background frame extraction
4. **Addresses**: DETR training limitations

## Benefits for DETR Training

1. **Diverse Examples**: DINO clustering ensures diverse training samples
2. **Semantic Similarity**: Groups frames by semantic content, not just visual
3. **Reduced Dataset Size**: Selects most informative frames, reducing training time
4. **Better Generalization**: Diverse examples improve model robustness
5. **Background Integration**: Proper negative examples reduce false positives

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- sklearn
- matplotlib
- opencv-python
- numpy
- tqdm
- Pillow

## Installation

```bash
pip install torch torchvision sklearn matplotlib opencv-python numpy tqdm pillow
```

## Next Steps

1. **Run DINO selection** on your surgical video dataset
2. **Combine with background frames** for complete dataset
3. **Add actual tool annotations** to selected frames
4. **Train DETR model** with integrated dataset
5. **Compare performance** with original DETR training

This system provides a significant improvement over random frame selection by leveraging DINO's semantic understanding to create more informative and diverse training datasets for DETR.