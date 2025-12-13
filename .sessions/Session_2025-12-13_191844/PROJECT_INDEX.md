# Project Index - ViTParticleFilterTracker

**Generated:** 2025-12-13 19:24:16
**Method:** Directory scan

---

## TestDatasetGenerator/ (NEW - this session)
Files: 9

| File | Description |
|------|-------------|
| `extract_random_frames.py` | Extract random frames from videos |
| `detect_duplicates.py` | Detect duplicates vs training set |
| `detr_auto_annotate.py` | DETR batch auto-annotation |
| `review_annotations.py` | OpenCV manual review UI |
| `coco_to_yolo.py` | COCO to YOLO converter |
| `annotate_test_frames.py` | Manual annotation (COCO) |
| `annotate_test_frames_yolo.py` | Manual annotation (YOLO) |
| `annotate_with_detr.py` | Combined DETR + manual |
| `run_pipeline.py` | Pipeline runner |

---

## AdvancedDatasetSelection/
Files: 28

Key files:
- `main_selection_pipeline.py` - Main pipeline
- `dino_extractor.py` - DINO feature extraction
- `cluster_selector.py` - Clustering-based selection
- `detr_el2n_scorer.py` - DETR EL2N scoring

---

## YOLO_DETR_Benchmarks/scripts/
Files: 48

Key files:
- `benchmark_yolo_vs_detr_q81_multi_epoch.py` - Main benchmark
- `detr_train_optimized.py` - DETR training
- `main_pipeline.py` - Background training pipeline

---

## Annotators/
Files: 55

Key files:
- `detr_annotator_inference_range_images.py` - DETR inference
- `opencv_annotation_tracker_glob_images.py` - OpenCV annotation

---

## BareDetr/
Files: 10

Core DETR implementation:
- `detr.py` - DETR model
- `backbone.py` - ResNet backbone
- `transformer.py` - Transformer architecture
- `inference.py` - Inference script

---

## Statistics

| Module | Files |
|--------|-------|
| TestDatasetGenerator | 9 |
| AdvancedDatasetSelection | 28 |
| YOLO_DETR_Benchmarks | 48 |
| Annotators | 55 |
| BareDetr | 10 |
| **TOTAL** | **150+** |

---

*Index generated: 2025-12-13 19:24*
