# DETR Single-Class Training Playbook (tooltip)

This playbook documents an end-to-end, reproducible procedure to train a single-class DETR for surgical tool tip detection.

## 0) Environment Setup
- Python 3.9+
- Install core deps:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA/CPU as needed
pip install transformers pillow opencv-python numpy pyyaml
```
- Optional (for dataset builder pseudo-labels):
```
pip install ultralytics
```
- COCO evaluation:
```
# Linux/WSL
pip install pycocotools
# Windows
pip install pycocotools-windows
```

## 1) Build Dataset From Videos
- Configure `Cataract_DETR_DatasetBuilder/builder_config.yaml`:
  - `source_videos_dir`: `E:/Cataract/videos/micro` (WSL: `/mnt/e/Cataract/videos/micro`)
  - `output_dir`: final dataset root
  - `category`: `{ id: 1, name: tooltip }`
  - `sampling`: `target_fps=2`, `blur_threshold=80`, `phash_threshold=10`
  - `yolo_pseudo_labels`: `enabled: true`, `weights_path: <your yolo .pt>`, `conf_threshold: 0.25`
  - `negatives`: `include: true`, `max_ratio: 0.3`
  - `splits`: `train=0.7`, `val=0.15`, `test=0.15`
- Run:
```
python Cataract_DETR_DatasetBuilder/build_dataset.py --config Cataract_DETR_DatasetBuilder/builder_config.yaml
```
- Outputs:
  - `images/`, `annotations_train.json`, `annotations_val.json`, `annotations_test.json`

Validation checklist:
- Ensure `category_id` is integer and consistent (1) across all annotations.
- Ensure `images` count per split is reasonable (ideally 3k–6k total to start, aim 8k–12k if possible).
- Verify negative images exist and are included up to 30% per split.
- Confirm no leakage: images from the same video appear only in one split.

## 2) Train Single-Class DETR
- Command (example):
```
python Cataract_DETR_DatasetBuilder/detr_single_class_train.py \
  --images_dir F:/.../output_cataract/images \
  --train_json F:/.../output_cataract/annotations_train.json \
  --val_json F:/.../output_cataract/annotations_val.json \
  --output_dir F:/.../output_cataract/detr_tooltip_train \
  --model_checkpoint facebook/detr-resnet-50 \
  --epochs 50 --batch_size 8 --use_amp --coco_eval_every 1
```
- Hyperparameters:
  - `epochs`: 50 (increase to 100 if loss continues to decrease and val mAP improves)
  - `batch_size`: 8 (increase if VRAM allows; reduce if OOM)
  - `lr`: 1e-4; `lr_backbone`: 1e-5; `weight_decay`: 1e-4
  - `num_queries`: 100 (default for DETR)
- What to monitor:
  - `train_loss` vs `val_loss` (divergence → overfitting)
  - COCO `mAP@0.5`, `mAP@0.5:0.95`, `AR@100` each epoch
  - If mAP plateaus and val loss oscillates → consider early stop or LR decay

Optional scheduler suggestion:
- Cosine with warmup (not implemented by default). If needed, we can add it.

Resuming:
- Script saves `best_model_state.pth` (by val loss). To resume training with this state:
  - We can extend the trainer to accept `--resume_state <pth>` (not added yet). For now, load weights manually before continuing.

## 3) Final Test Evaluation
- Command:
```
python Cataract_DETR_DatasetBuilder/eval_coco_offline.py \
  --images_dir F:/.../output_cataract/images \
  --test_json F:/.../output_cataract/annotations_test.json \
  --model_checkpoint facebook/detr-resnet-50 \
  --weights_path F:/.../output_cataract/detr_tooltip_train/best_model_state.pth \
  --batch_size 8 --conf_threshold 0.0
```
- Adjust `--conf_threshold` (0.0–0.5) only for analysis; mAP integrates over thresholds anyway.

## 4) Quality Controls & Tips
- Dataset size:
  - If <3k images: prioritize diversity over count (more videos, varied scenes).
- Negatives:
  - Keep 10–30% negatives per split; improves no‑object calibration and false positive control.
- Resolution:
  - Keep input resolution default of the HF image processor (resizes internally). If objects are small, consider larger DETR input size later.
- Post‑processing:
  - For DETR, NMS is usually not needed, but you can try NMS in downstream inference for robustness.

## 5) Common Pitfalls
- pycocotools install issues on Windows → use `pycocotools-windows`.
- Mixed `category_id` types (float vs int) → ensure all are `int`.
- Leakage between splits → must group by video.
- Over-augmentation via on-disk copies → avoid; prefer training-time aug.

## 6) Extension Paths
- Swap backbone (e.g., ResNet‑101) by changing `--model_checkpoint`.
- Try Deformable DETR / DINO once baseline is stable.
- Add learning rate scheduler and early stopping hooks.

---

This guide accompanies the session configs in `configs/` and the builder/trainer scripts in `Cataract_DETR_DatasetBuilder/`.
