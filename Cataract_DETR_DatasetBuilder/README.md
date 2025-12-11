# Cataract DETR Dataset Builder

Purpose: create a high‑quality, DETR‑friendly detection dataset from cataract surgery videos (e.g., `E:\Cataract\videos\micro`).

Outcome:
- Diverse frames (reduced redundancy, filtered blur)
- Optional pseudo‑labels using YOLO weights you provide
- COCO annotations with consistent integer `category_id` (default: 1)
- Grouped train/val/test split by video to prevent leakage

## Requirements
- Python 3.9+
- pip packages: `opencv-python`, `numpy`, `Pillow`
- Optional (for pseudo‑labels): `ultralytics` (YOLOv8)

Install:
```
pip install opencv-python numpy Pillow ultralytics
# for COCO mAP evaluation
pip install pycocotools  # Linux/WSL
# or on Windows
pip install pycocotools-windows
```

Note:
- On Windows WSL: `E:\` is usually `/mnt/e` and `F:\` is `/mnt/f`. Adjust paths accordingly.
- Avoid pre-generating augmented images; prefer runtime augmentations during training.

## Quick Start
1) Configure `builder_config.yaml`.
2) Run end‑to‑end builder:
```
python Cataract_DETR_DatasetBuilder/build_dataset.py --config Cataract_DETR_DatasetBuilder/builder_config.yaml
```
3) Outputs (defaults):
- `output/images/*.jpg` – selected frames
- `output/annotations_train.json`, `annotations_val.json`, `annotations_test.json`
- `output/frames_manifest.json` – metadata of selected frames
- `output/pseudo_annotations.json` – if YOLO pseudo‑labels enabled

## Config Overview (builder_config.yaml)
- `source_videos_dir`: root with cataract videos
- `output_dir`: destination folder for images + annotations
- `sampling`: frame selection params
  - `target_fps`: sample rate in frames per second (e.g., 2)
  - `max_frames_per_video`: hard cap per video (optional)
  - `blur_threshold`: variance of Laplacian min value (e.g., 80)
  - `phash_threshold`: perceptual hash Hamming distance to drop near duplicates (e.g., 10)
- `category`: class settings
  - `id`: integer class id (recommended 1)
  - `name`: class name (e.g., "tooltip")
- `yolo_pseudo_labels`: optional inference to auto-label
  - `enabled`: true/false
  - `weights_path`: path to YOLOv8 weights (`.pt`)
  - `conf_threshold`: 0.25–0.5 recommended
- `negatives`: control negative frames in the final dataset
  - `include`: true/false
  - `max_ratio`: e.g., 0.3 means up to 30% negatives per split
- `splits`: grouped by video (no leakage)
  - `train`: 0.7, `val`: 0.15, `test`: 0.15
  
## Recommended Targets (single class)
- Start: 3k–6k diverse frames total (≥10 videos). If possible, aim for ~10k.
- Keep negatives: 10–30% of frames without tools to help DETR calibrate background.
- Prefer quality over quantity: reduce near-duplicates and blur.

## Commands
- Extract frames only:
```
python Cataract_DETR_DatasetBuilder/extract_and_select_frames.py \
  --videos /mnt/e/Cataract/videos/micro \
  --out /mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/output_cataract/images \
  --target-fps 2 --blur-th 80 --phash-th 10 --max-per-video 1500
```
- Pseudo‑labels with YOLO:
```
python Cataract_DETR_DatasetBuilder/run_yolo_pseudo_labels.py \
  --images /mnt/f/.../output_cataract/images \
  --weights /mnt/f/.../YOLO_DETR_Benchmarks/models/YOLO/epoch100.pt \
  --out /mnt/f/.../output_cataract/pseudo_annotations.json \
  --category-id 1 --category-name tooltip --conf 0.25
```
- Build grouped COCO splits:
```
python Cataract_DETR_DatasetBuilder/split_and_build_coco.py \
  --images /mnt/f/.../output_cataract/images \
  --preds /mnt/f/.../output_cataract/pseudo_annotations.json \
  --out /mnt/f/.../output_cataract --category-id 1 --category-name tooltip \
  --train 0.7 --val 0.15 --test 0.15 --include-negatives --neg-ratio 0.3
```

## Offline COCO mAP (test)
After training, evaluate on test split without retraining:
```
python Cataract_DETR_DatasetBuilder/eval_coco_offline.py \
  --images_dir /mnt/f/.../output_cataract/images \
  --test_json /mnt/f/.../output_cataract/annotations_test.json \
  --model_checkpoint facebook/detr-resnet-50 \
  --weights_path /mnt/f/.../output_cataract/detr_tooltip_train/best_model_state.pth \
  --batch_size 8 --conf_threshold 0.0
```

## Notes
- The builder uses filename convention `<video_stem>_frame_<idx>.jpg` to group by original video.
- COCO `category_id` is always integer and consistent across files.
- If `ultralytics` is unavailable, you can skip pseudo‑labels and later import your own annotations.
