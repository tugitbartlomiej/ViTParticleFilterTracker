# Background Frame Extraction for DETR Training

This tool extracts background frames (frames without surgical tools) from cataract surgery videos to improve DETR training by providing negative examples.

## Problem

DETR was trained without background/negative examples, causing it to predict tools everywhere. This tool extracts clean background frames to add to the training dataset.

## Directory Structure

```
Background_Extraction/
├── extract_background_frames.py    # Main script
├── scripts/
│   ├── extract_frames.py          # Extract frames from videos
│   └── identify_background_frames.py  # Identify background frames
├── extracted_frames/               # Temporary extracted frames
├── background_frames/              # Selected background frames
└── detr_background_dataset/        # DETR-ready dataset
    ├── train/                      # Training background frames
    ├── val/                        # Validation background frames
    └── annotations/                # COCO-style annotations
```

## Usage

### Quick Start
```bash
cd YOLO_DETR_Benchmarks/Background_Extraction
python extract_background_frames.py
```

### Advanced Usage
```bash
# Extract from custom video directory
python extract_background_frames.py --video_dir "E:/Cataract/videos/micro"

# Adjust frame extraction interval (every 60 frames instead of 30)
python extract_background_frames.py --frame_interval 60

# Limit background frames and adjust detection sensitivity
python extract_background_frames.py --max_frames 500 --confidence 0.5

# Clean temporary files after processing
python extract_background_frames.py --clean_temp
```

## Process

1. **Frame Extraction**: Extracts frames from videos at specified intervals
2. **Object Detection**: Uses YOLO to detect objects in frames
3. **Background Identification**: Identifies frames without surgical tools
4. **Quality Filtering**: Filters out blurry or poor quality frames
5. **Dataset Creation**: Creates DETR-compatible dataset structure

## Parameters

- `--video_dir`: Directory containing videos (default: E:/Cataract/videos/micro)
- `--frame_interval`: Extract every N frames (default: 30)
- `--confidence`: YOLO confidence threshold (default: 0.3)
- `--max_frames`: Maximum background frames to keep (default: 1000)
- `--clean_temp`: Remove temporary extracted frames after processing

## Output

- `background_frames/`: Selected high-quality background frames
- `detr_background_dataset/`: DETR-ready dataset with train/val splits
- `analysis_results.json`: Detailed analysis of all processed frames
- `background_frames.json`: Metadata about selected background frames

## Integration with DETR Training

To use these background frames in DETR training:

1. Combine with existing tool detection dataset
2. Add background/no_object class to COCO annotations
3. Retrain DETR with both positive and negative examples
4. Reduce `num_queries` from 100 to ~20 for better performance

## Requirements

- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO
- Python 3.7+

## Quality Metrics

Frames are filtered based on:
- Blur score (Laplacian variance > 100)
- Brightness (30-225 range)
- Contrast (standard deviation > 20)
- Object detection results