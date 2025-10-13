# Medical Object Detection Benchmarking Guidelines
## DETR vs YOLO for Surgical Tool Detection - Evidence-Based Best Practices

**Version:** 2.0
**Date:** 2025-10-13
**Context:** Clinical AI for Surgical Tool Detection
**Evidence Base:** Scientific literature + Project experience

---

## Executive Summary

This document provides **evidence-based guidelines** for benchmarking DETR and YOLO models in medical imaging, specifically for surgical tool detection. Guidelines are derived from:

1. **Academic Research**: RT-DETR (CVPR 2024), Medical DETR studies (NYU 2025)
2. **Clinical Applications**: Diabetic retinopathy, surgical instrument tracking
3. **Project Experience**: Query 81 innovation, background fine-tuning success

**Key Finding**: DETR models require **different optimization strategies** for medical imaging compared to natural images (MS COCO). Simpler architectures often outperform complex ones.

---

## Table of Contents

1. [Medical Imaging Context](#1-medical-imaging-context)
2. [DETR Training for Medical Applications](#2-detr-training-for-medical-applications)
3. [Critical Benchmarking Metrics](#3-critical-benchmarking-metrics)
4. [Evaluation Methodology](#4-evaluation-methodology)
5. [DETR vs YOLO Performance Expectations](#5-detr-vs-yolo-performance-expectations)
6. [Implementation Checklist](#6-implementation-checklist)
7. [Scientific References](#7-scientific-references)

---

## 1. Medical Imaging Context

### 1.1 Unique Characteristics of Medical Images vs Natural Images

Based on **"Understanding differences in applying DETR to natural and medical images"** (NYU 2025):

| Characteristic | Natural Images (MS COCO) | Medical Images (Surgical) |
|----------------|--------------------------|---------------------------|
| **Objects per image** | 7.33 average | 1-2 average |
| **Object classes** | 80 classes | 1-2 classes (tool, background) |
| **Object size variance** | σ = 0.16 | σ = 0.025 (uniform) |
| **Resolution** | 224×224 - 640×640 | 1920×1080 - 4K |
| **Background** | Complex, varied | Homogeneous (surgical field) |
| **Temporal continuity** | Single frames | Video sequences (30-60 FPS) |

**Clinical Implication**: Medical images require **simpler DETR architectures** with fewer object queries and potentially single-scale encoders.

### 1.2 Critical Requirements for Surgical Applications

**Safety Requirements** (from clinical literature):
- **Sensitivity (Recall)**: ≥95% - Cannot miss surgical instruments
- **NPV (Negative Predictive Value)**: ≥99% - "No tool" must be reliable
- **Temporal Stability**: High - Flickering detections disrupt workflow
- **Latency**: <100ms - Real-time feedback essential

**Trust Requirements**:
- **Precision**: ≥85% - Minimize false alarms
- **Confidence Calibration**: ECE <0.10 - Confidence scores must be trustworthy
- **Consistency**: Low variance across cases

---

## 2. DETR Training for Medical Applications

### 2.1 Architecture Simplification (Evidence-Based)

**Finding** (NYU Medical DETR Study): *"Simpler and shallower architectures often achieve equal or superior results with less computational cost"*

#### Recommended Modifications for Surgical Tool Detection:

**1. Reduce Object Queries**
```python
# Natural images (MS COCO standard):
num_queries = 300  # ❌ Too many for medical

# Medical images (recommended):
num_queries = 50-100  # ✅ Optimal for 1-2 objects per frame
```

**Evidence**: Medical images contain <10 objects (typically 1-2), so 300 queries lead to increased false positives and slower learning.

**2. Simplify Encoder Complexity**
```python
# Complex (natural images):
encoder_layers = 6
multi_scale_features = [C3, C4, C5]  # 3 scales

# Simplified (medical images):
encoder_layers = 3-4  # ✅ Reduce by 33-50%
multi_scale_features = [C5]  # ✅ Single scale sufficient
```

**Evidence**: Medical images have uniform object sizes (σ=0.025), so multi-scale fusion provides minimal benefit and increases overfitting risk.

**3. Decoder Optimization**
```python
# Keep decoder relatively standard:
decoder_layers = 6  # Standard
# But use simpler query initialization:
query_init = "static"  # ✅ Random learnable embeddings work best
```

### 2.2 Loss Function Configuration

**Critical for Surgical Tool Detection**:

```python
# DETR Config for surgical tools:
config = DetrConfig(
    num_labels=2,  # background + surgical_tool

    # Hungarian matching costs:
    class_cost=1.5,      # ⬆ Increased (was 1.0) - better classification
    bbox_cost=5.0,       # Standard
    giou_cost=2.0,       # Standard

    # EOS (no-object) coefficient:
    eos_coefficient=0.05  # ⬇ Decreased (was 0.1) - reduce background bias
)
```

**Rationale**:
- **Higher class_cost**: Medical images require precise class discrimination (tool vs background)
- **Lower eos_coefficient**: Default 0.1 creates bias toward "no object" - problematic when most frames contain tools

**Evidence**: From project experience (ANALIZA_TRENINGU_DETR.md) and RT-DETR medical studies.

### 2.3 Training Strategies for Background Class Integration

**Problem**: Original DETR trained only on tool detection → false positives on background frames

**Solution**: Multi-AI consensus background extraction + gentle fine-tuning

#### Stage 1: Background Frame Extraction
```python
# Multi-model consensus system (proven in project):
background_candidates = []

for frame in video_frames:
    yolo_detects_tool = YOLO.detect(frame)
    detr_detects_tool = DETR.detect(frame)

    # Background = BOTH models say "no tool"
    if not yolo_detects_tool and not detr_detects_tool:
        background_candidates.append(frame)

# DINO clustering for diversity:
diverse_backgrounds = DINO.cluster_and_select(
    background_candidates,
    n_clusters=5,
    frames_per_cluster=20
)
```

**Result**: 96% frame reduction (2,500 → 100 high-quality backgrounds)

#### Stage 2: Fine-tuning Strategy

**Successful Strategy** (from project validation):

```python
# Ultra-gentle fine-tuning (proven effective):
config = {
    "learning_rate": 1e-8,           # ⬇ Ultra-low
    "backbone_lr": 1e-9,             # ⬇ Freeze-like
    "classifier_lr": 1e-7,           # ⬆ Allow adaptation
    "epochs": 3,                     # Short
    "tooltip_ratio": 0.90,           # 90% original class
    "background_ratio": 0.10,        # 10% new class
    "freeze_backbone": False,        # Light adaptation
}

# Result: 70% detection, 90.6% confidence, NO catastrophic forgetting
```

**Alternative Strategies** (from literature):
- **EMA Teacher Model** (OD-DETR 2024): Exponential moving average stabilization
- **Curriculum Learning**: Start with easy examples, gradually add difficult
- **Progressive Fine-tuning**: Multi-stage with increasing background ratio

### 2.4 Dataset Requirements

**Minimum Standards**:
- **Training set**: ≥1,000 annotated frames
- **Validation set**: ≥200 frames
- **Background frames**: 10-20% of training set
- **Format**: COCO JSON (standard for DETR)
- **Annotation quality**: Expert-verified bounding boxes

**Data Augmentation** (medical-specific):
```python
transforms = A.Compose([
    A.RandomRotation(degrees=10),           # ✅ Surgical rotation
    A.ColorJitter(brightness=0.1, contrast=0.1),  # ✅ Light variation
    A.RandomAffine(translate=(0.1, 0.1)),   # ✅ Camera movement
    A.GaussianBlur(kernel_size=3, p=0.3),   # ✅ Surgical smoke
    # ❌ Avoid aggressive augmentations (flips, crops) - medical context matters
])
```

---

## 3. Critical Benchmarking Metrics

### 3.1 Standard COCO Metrics (Mandatory)

**Primary Metrics**:

1. **mAP@[0.5:0.95]** - Primary benchmark
   - Target for surgical tools: **>50%**
   - Strict metric, penalizes imprecise localization

2. **mAP@0.5** - Clinical acceptability threshold
   - Target: **>70%**
   - More lenient, acceptable for tool presence detection

3. **Precision** (Positive Predictive Value)
   - Target: **>85%**
   - Clinical meaning: "When we detect a tool, how often are we correct?"
   - High precision = surgeon trust

4. **Recall** (Sensitivity)
   - Target: **>95%**
   - Clinical meaning: "Are we detecting all instruments?"
   - **Most critical for safety** - cannot miss tools

**Scale-Specific AP** (important for surgical tools):
- **AP_small**: Small instruments (forceps tips)
- **AP_medium**: Standard tools
- **AP_large**: Tools close to camera

### 3.2 Medical-Specific Metrics (Critical for Clinical Deployment)

#### A. False Negative Analysis (Safety Critical)

**Classification by Clinical Risk**:

| Type | Description | Risk Level | Acceptability |
|------|-------------|------------|---------------|
| Partial Occlusion Miss | Tool under tissue/blood | Low | Acceptable |
| Full Visibility Miss | Clear tool missed | High | **Unacceptable** |
| Critical Instrument Miss | Needle, blade missed | Critical | **Zero tolerance** |

#### B. False Positive Analysis

**Classification by Clinical Impact**:

| Type | Description | Impact | Example |
|------|-------------|--------|---------|
| Background Confusion | Surgical drape as tool | Low | Easily dismissed |
| Near-Miss Detection | Wrong tool class | Medium | Forceps → scissors |
| Phantom Detection | Empty field detection | High | Erodes trust |

**Implementation**:
```python
def classify_false_positives(fp_detections, gt_annotations):
    """Categorize FPs by clinical risk"""
    for fp in fp_detections:
        nearest_gt = find_nearest_gt(fp, gt_annotations)
        distance = calculate_distance(fp, nearest_gt)

        if distance > 200px:
            category = "phantom"  # High risk
        elif distance < 50px:
            category = "near_miss"  # Medium risk
        else:
            category = "background_confusion"  # Low risk
```

#### C. Diagnostic Odds Ratio (DOR)

```python
DOR = (TP * TN) / (FP * FN)
```

**Clinical Thresholds**:
- **DOR > 100**: Excellent diagnostic accuracy
- **DOR 50-100**: Good (acceptable for surgical deployment)
- **DOR < 50**: Poor

### 3.3 Temporal Stability Metrics (Video-Specific)

**Critical for surgical video applications** (DETR advantage expected here):

#### A. Temporal IoU
```python
def calculate_temporal_iou(predictions_by_frame):
    """
    Measure bounding box stability across consecutive frames.

    Returns: avg_temporal_iou (higher = more stable tracking)
    """
    temporal_ious = []

    for track in associate_detections_across_frames(predictions_by_frame):
        for i in range(len(track) - 1):
            iou = calculate_iou(track[i]['bbox'], track[i+1]['bbox'])
            temporal_ious.append(iou)

    return np.mean(temporal_ious)
```

**Clinical Thresholds**:
- **>0.85**: Excellent (smooth tracking)
- **0.70-0.85**: Acceptable (minor jitter)
- **<0.70**: Poor (disruptive)

**Research Hypothesis**: DETR's global attention → higher temporal IoU than YOLO's frame-independent CNN

#### B. Detection Flicker Rate
```python
def calculate_flicker_rate(predictions, ground_truth):
    """
    Count frames where object: detected → NOT detected → detected
    (Flicker = model loses track then recovers)
    """
    flicker_events = 0

    for obj_track in ground_truth:
        detected = [is_detected(pred, obj_track[i])
                   for i, pred in enumerate(predictions)]

        # Count True → False → True patterns
        for i in range(1, len(detected) - 1):
            if detected[i-1] and not detected[i] and detected[i+1]:
                flicker_events += 1

    return flicker_events / len(ground_truth)  # Rate per object
```

**Clinical Thresholds**:
- **<1%**: Excellent (imperceptible)
- **1-5%**: Acceptable
- **>5%**: Unacceptable (disruptive)

#### C. Track Fragmentation
```python
def calculate_fragmentation(predictions, ground_truth):
    """
    For continuously visible tool, count separate track segments.
    Ideal: 1 segment. Fragmented: multiple segments.
    """
    fragmentation_scores = []

    for gt_track in ground_truth.continuous_tracks:
        pred_segments = find_matching_segments(predictions, gt_track)
        fragmentation_scores.append(len(pred_segments))

    fragmentation_rate = sum(f > 1 for f in fragmentation_scores) / len(fragmentation_scores)
    return fragmentation_rate
```

**Clinical Threshold**: <5% fragmentation rate

### 3.4 Performance Metrics (Real-Time Requirements)

#### A. Latency Breakdown
```python
def detailed_latency_breakdown(model, test_images):
    """Measure complete inference pipeline"""
    results = {
        'preprocessing': [],
        'forward_pass': [],
        'postprocessing': [],  # Includes NMS for YOLO
        'total': []
    }

    for image in test_images:
        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        preprocessed = preprocess(image)
        results['preprocessing'].append((time.perf_counter() - t0) * 1000)

        # Stage 2: Forward pass
        t0 = time.perf_counter()
        outputs = model(preprocessed)
        results['forward_pass'].append((time.perf_counter() - t0) * 1000)

        # Stage 3: Postprocessing
        t0 = time.perf_counter()
        if model_type == 'yolo':
            final = apply_nms(outputs)  # NMS adds 2-10ms
        else:  # DETR
            final = threshold_filter(outputs)  # Just thresholding
        results['postprocessing'].append((time.perf_counter() - t0) * 1000)

    return {
        stage: {
            'mean_ms': np.mean(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99)
        }
        for stage, times in results.items()
    }
```

**Expected Results** (RTX 3070):
```
YOLO:
  Preprocessing: 2-3ms
  Forward pass: 5-8ms
  Postprocessing (NMS): 2-5ms
  Total: ~10-15ms → 60-100 FPS ✅ Real-time

DETR:
  Preprocessing: 3-5ms
  Forward pass: 30-50ms
  Postprocessing: 1-2ms
  Total: ~35-55ms → 18-28 FPS ⚠ Acceptable for some applications

RT-DETR:
  Total: ~10-20ms → 50-100 FPS ✅ Real-time competitive with YOLO
```

#### B. NMS Impact Analysis (YOLO-Specific)

**Research Question**: How does NMS affect YOLO accuracy and speed?

```python
def analyze_nms_impact(yolo_model, test_images):
    """Measure NMS overhead and accuracy impact"""
    results = []

    for iou_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for conf_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            # Pre-NMS
            t0 = time.time()
            raw_outputs = yolo_model.predict(test_images, conf=conf_thresh, nms=False)
            pre_nms_time = time.time() - t0

            # Post-NMS
            t0 = time.time()
            final_outputs = apply_nms(raw_outputs, iou_threshold=iou_thresh)
            nms_time = time.time() - t0

            results.append({
                'iou_threshold': iou_thresh,
                'conf_threshold': conf_thresh,
                'nms_latency_ms': nms_time * 1000,
                'nms_overhead_%': (nms_time / pre_nms_time) * 100,
                'reduction_rate': 1 - (len(final_outputs) / len(raw_outputs))
            })

    return pd.DataFrame(results)
```

**Clinical Implication**: DETR's NMS-free architecture = more predictable latency for real-time surgical systems

#### C. GPU Memory Profiling
```python
def measure_vram_usage(model, test_images, batch_sizes=[1, 2, 4]):
    """Peak VRAM across batch sizes"""
    results = []

    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        batch = create_batch(test_images[:bs])

        with torch.no_grad():
            _ = model(batch)

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        results.append({
            'batch_size': bs,
            'peak_vram_mb': peak_mb,
            'vram_per_image_mb': peak_mb / bs
        })

    return pd.DataFrame(results)
```

**Expected**:
- **YOLO**: 2-4 GB
- **DETR**: 4-8 GB
- **Clinical significance**: Lower VRAM = deployable on edge devices in OR

---

## 4. Evaluation Methodology

### 4.1 Dataset Preparation

**Mandatory Requirements**:
```yaml
dataset:
  # MUST be identical for both models
  test_images: "Datasets/TooltipMining/val"
  annotations: "annotations/val_annotations.json"  # COCO format

  # Minimum size
  min_images: 1000

  # Video sequences for temporal metrics
  video_sequences: true
  fps: 30
```

### 4.2 Hardware Standardization

**Fair Comparison Protocol**:
```python
# Identical hardware for both models
GPU = "NVIDIA RTX 3070"  # or Tesla T4/V100
CUDA_VERSION = "12.8"
PYTORCH_VERSION = "2.8.0"
PRECISION = "FP32"  # Fair comparison (use FP16 separately)
BATCH_SIZE = 1  # Real-time simulation
```

### 4.3 Reproducibility Protocol

```python
# Set all random seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# Deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Reproducibility > speed
```

### 4.4 Benchmark Execution Steps

**Step 1: Model Verification**
```bash
# Verify models exist and are correct format
ls -lh models/YOLO/epoch100.pt
ls -lh DETR/detr_inference_model_final/
```

**Step 2: Run Inference**
```bash
cd Advanced_Analysis

# YOLO inference
python run_inference.py --model_type yolo --config config.yaml

# DETR inference
python run_inference.py --model_type detr --config config.yaml
```

**Step 3: Calculate Metrics**
```bash
python evaluate_metrics.py --config config.yaml
```

**Step 4: Generate Report**
```bash
python visualize_results.py --config config.yaml
```

### 4.5 Output Structure

```
benchmark_results/
├── yolo_predictions.json          # COCO format detections
├── detr_predictions.json          # COCO format detections
├── yolo_performance.json          # FPS, VRAM, timing
├── detr_performance.json          # FPS, VRAM, timing
├── benchmark_report.json          # Consolidated metrics
├── comparison_video.mp4           # Side-by-side visualization
├── map_comparison.png             # mAP bar chart
├── temporal_iou_comparison.png    # Stability analysis
├── fps_comparison.png             # Performance chart
└── pr_curves_comparison.png       # Precision-Recall curves
```

---

## 5. DETR vs YOLO Performance Expectations

### 5.1 Literature Benchmarks

#### RT-DETR vs YOLOv8 (CVPR 2024 - MS COCO)

| Model | AP@50:95 | FPS (T4 GPU) | VRAM | NMS Required |
|-------|----------|--------------|------|--------------|
| YOLOv8-L | 52.9% | 71 | 2.5GB | Yes |
| RT-DETR-R50 | **53.1%** | **108** | 3.8GB | **No** |
| YOLOv8-X | 53.9% | 50 | 3.2GB | Yes |
| RT-DETR-R101 | **54.3%** | **74** | 5.1GB | **No** |

**Key Finding**: RT-DETR achieves **better accuracy** with **higher speed** at cost of increased VRAM

#### Medical Imaging - Diabetic Retinopathy (2025)

| Model | Precision | Recall | mAP@50 | mAP@50:95 |
|-------|-----------|--------|--------|-----------|
| YOLOv5 | 0.82 | 0.78 | 0.81 | 0.63 |
| YOLOv8 | 0.86 | 0.83 | 0.85 | 0.68 |
| DETR | 0.85 | 0.79 | 0.83 | 0.68 |
| **RT-DETR** | **0.90** | **0.85** | **0.88** | **0.76** |

**Key Finding**: RT-DETR shows **consistent advantages** in medical imaging precision

### 5.2 Expected Results for Surgical Tool Detection

**Based on project experience + literature**:

#### YOLO (YOLOv8, 100 epochs)
```
Expected Performance:
  mAP@50: 60-75%
  mAP@50:95: 45-60%
  Precision: 85-90%
  Recall: 90-95%

Temporal Metrics:
  Temporal IoU: 0.70-0.80 (frame-independent)
  Flicker rate: 2-5%
  Fragmentation: 10-15%

Performance:
  FPS: 60-100 (RTX 3070)
  Latency: 10-15ms
  VRAM: 2-4 GB

Strengths:
  ✅ Fast inference
  ✅ Low memory
  ✅ Mature architecture
  ✅ Small object detection

Limitations:
  ⚠ NMS dependency
  ⚠ Lower temporal stability
  ⚠ Frame-independent processing
```

#### DETR (Base, 100 epochs)
```
Expected Performance:
  mAP@50: 40-55%
  mAP@50:95: 30-45%
  Precision: 50-70%
  Recall: 90-100%

Temporal Metrics:
  Temporal IoU: 0.80-0.90 (attention benefit)
  Flicker rate: 0.5-2%
  Fragmentation: 5-10%

Performance:
  FPS: 18-28 (RTX 3070)
  Latency: 35-55ms
  VRAM: 4-8 GB

Strengths:
  ✅ No NMS
  ✅ High temporal stability
  ✅ Global context understanding
  ✅ End-to-end differentiable

Limitations:
  ⚠ Slower inference
  ⚠ Higher memory
  ⚠ Requires more training epochs
```

#### RT-DETR (If available)
```
Expected Performance:
  mAP@50: 65-80%
  mAP@50:95: 50-65%
  Precision: 85-90%
  Recall: 90-95%

Temporal Metrics:
  Temporal IoU: 0.85-0.92
  Flicker rate: 0.5-1.5%
  Fragmentation: 3-7%

Performance:
  FPS: 50-100 (RTX 3070)
  Latency: 10-20ms
  VRAM: 3-6 GB

Strengths:
  ✅ Best of both worlds
  ✅ Real-time speed
  ✅ High accuracy
  ✅ No NMS

Ideal for:
  Clinical deployment
```

### 5.3 Decision Matrix for Model Selection

```
Choose YOLO if:
├─ Real-time performance critical (>60 FPS required)
├─ Hardware limited (edge devices, mobile OR systems)
├─ Tools well-separated (minimal overlap)
├─ Latency consistency important
└─ Training time limited

Choose DETR if:
├─ Accuracy paramount over speed
├─ Complex spatial relationships between tools
├─ Temporal stability critical (tracking applications)
├─ End-to-end differentiability needed (future fine-tuning)
└─ Can accept 20-30 FPS

Choose RT-DETR if:
├─ Need BOTH accuracy AND speed
├─ Can afford slightly higher VRAM
├─ Want NMS-free architecture benefits
├─ Clinical deployment target
└─ Best overall balance
```

---

## 6. Implementation Checklist

### 6.1 Pre-Benchmark Setup

- [ ] **Environment**
  - [ ] Python 3.11 installed
  - [ ] PyTorch 2.8.0+ with CUDA support
  - [ ] All dependencies: `pip install -r Advanced_Analysis/requirements.txt`

- [ ] **Models**
  - [ ] YOLO model trained (100+ epochs)
  - [ ] DETR model trained (100+ epochs)
  - [ ] Models in correct format (YOLO: .pt, DETR: HuggingFace)

- [ ] **Dataset**
  - [ ] Test images prepared (≥1,000 images)
  - [ ] COCO annotations validated
  - [ ] Video sequences available (for temporal metrics)
  - [ ] Label mapping configured correctly

### 6.2 Configuration Verification

```yaml
# Advanced_Analysis/config.yaml

models:
  yolo:
    path: "models/YOLO/epoch100.pt"  # ✅ Check exists
  detr:
    path: "DETR/detr_inference_model_final"  # ✅ HuggingFace format

dataset:
  images_dir: "../BackgroundFinetuned/Datasets/TooltipMining/val"
  annotations_path: "../BackgroundFinetuned/Datasets/TooltipMining/annotations/val_annotations.json"

inference_params:
  confidence_threshold: 0.5  # Standard
  iou_threshold: 0.5         # mAP@.50 threshold

label_map:
  yolo: { 0: 1 }  # Map to COCO category
  detr: { 0: 1 }
```

### 6.3 Execution Checklist

- [ ] **Step 1: Run Benchmark**
  ```bash
  cd Advanced_Analysis
  python run_benchmark.py  # Complete pipeline
  ```

- [ ] **Step 2: Verify Outputs**
  - [ ] `yolo_predictions.json` created
  - [ ] `detr_predictions.json` created
  - [ ] `benchmark_report.json` generated

- [ ] **Step 3: Generate Visualizations**
  ```bash
  python visualize_results.py
  ```

- [ ] **Step 4: Validate Results**
  - [ ] mAP values reasonable
  - [ ] FPS measurements consistent
  - [ ] Temporal metrics calculated

### 6.4 Results Validation

**Sanity Checks**:
```python
# Check prediction counts
yolo_pred_count = len(yolo_predictions)
detr_pred_count = len(detr_predictions)
assert yolo_pred_count > 0, "YOLO produced no predictions"
assert detr_pred_count > 0, "DETR produced no predictions"

# Check mAP range
assert 0 < map_yolo < 1, "YOLO mAP out of range"
assert 0 < map_detr < 1, "DETR mAP out of range"

# Check FPS reasonable
assert 10 < fps_yolo < 200, "YOLO FPS unreasonable"
assert 5 < fps_detr < 100, "DETR FPS unreasonable"
```

---

## 7. Scientific References

### Primary Research Papers

1. **Zhao, Y., et al. (2024)**. "DETRs Beat YOLOs on Real-time Object Detection"
   *CVPR 2024*
   - RT-DETR architecture and benchmarking methodology
   - NMS impact analysis
   - Speed vs accuracy trade-offs

2. **Xu, Y., et al. (2025)**. "Understanding differences in applying DETR to natural and medical images"
   *NYU Medical School*
   - Medical imaging requires simpler architectures
   - Object query optimization for medical datasets
   - Multi-scale feature fusion analysis

3. **He, W., et al. (2025)**. "Object Detection for Medical Image Analysis: Insights from the RT-DETR Model"
   - RT-DETR for diabetic retinopathy detection
   - Medical imaging benchmarks
   - Precision: 0.90, Recall: 0.85, mAP50: 0.88

4. **Loza, et al. (2023)**. "Real-time surgical tool detection with multi-scale positional encoding"
   *Healthcare Technology Letters*
   - Dense transformer architectures
   - Surgical tool detection challenges
   - Real-time processing requirements

### Project-Specific Innovations

5. **Query 81 Innovation** (ViTParticleFilterTracker Project)
   - Single query specialization for tool detection
   - 97% confidence, 84.8% IoU, zero false positives
   - `QUERY_81_SOLUTION_DOCUMENTATION.md`

6. **Background Fine-tuning Success** (DETR Project)
   - Ultra-gentle strategy: lr=1e-8, 90% tooltip / 10% background
   - 70% detection, 90.6% confidence, no catastrophic forgetting
   - `STRATEGY_TESTING_RESULTS.md`

### Additional Reading

7. **COCO Evaluation Toolkit**: Official pycocotools documentation
8. **Ultralytics YOLOv8**: Official benchmarks and documentation
9. **HuggingFace DETR**: Transformers library implementation

---

## Appendix: Common Pitfalls & Solutions

### Data Pitfalls

❌ **Pitfall**: Using different test sets for YOLO and DETR
✅ **Solution**: Ensure identical `images_dir` and `annotations_path` in config

❌ **Pitfall**: COCO format errors (mismatched image IDs)
✅ **Solution**: Validate annotations with `pycocotools.coco.COCO()`

❌ **Pitfall**: Class ID mismatch between model and annotations
✅ **Solution**: Configure `label_map` correctly in config.yaml

### Metric Pitfalls

❌ **Pitfall**: Comparing mAP@50 to mAP@50:95
✅ **Solution**: Report both, use mAP@50:95 as primary

❌ **Pitfall**: Ignoring scale-specific AP
✅ **Solution**: Analyze AP_small, AP_medium, AP_large separately

❌ **Pitfall**: Trusting single metric
✅ **Solution**: Use comprehensive evaluation (accuracy + temporal + performance)

### Implementation Pitfalls

❌ **Pitfall**: No GPU warm-up before timing
✅ **Solution**: Run 5-10 warm-up iterations before measuring FPS

❌ **Pitfall**: Inconsistent batch size
✅ **Solution**: Use batch_size=1 for fair real-time comparison

❌ **Pitfall**: Mixing CPU and GPU inference
✅ **Solution**: Both models on same GPU with same precision

### Clinical Pitfalls

❌ **Pitfall**: Optimizing for mAP only
✅ **Solution**: **Recall is more critical** for safety - cannot miss tools

❌ **Pitfall**: Ignoring false negatives
✅ **Solution**: Analyze FN by type (partial occlusion, full visibility, critical instrument)

❌ **Pitfall**: Dismissing temporal metrics
✅ **Solution**: Stability matters in surgery - flickering detections disrupt workflow

---

## Conclusion

This document provides **evidence-based, clinically-grounded guidelines** for benchmarking DETR and YOLO in surgical tool detection. Key takeaways:

1. **Medical imaging requires different optimization** than natural images
2. **Simpler DETR architectures often outperform** complex ones in medical contexts
3. **Safety metrics (Recall, NPV) are more critical** than raw accuracy
4. **Temporal stability is essential** for video-based surgical applications
5. **RT-DETR represents best-of-both-worlds** for clinical deployment

**Recommended Approach**:
1. Start with baseline YOLO vs DETR comparison using this framework
2. Analyze results across all three metric categories (accuracy, temporal, performance)
3. Consider RT-DETR if deployment requires both speed and accuracy
4. Validate with clinical domain experts before deployment

---

**Document Version:** 2.0
**Last Updated:** 2025-10-13
**Maintained By:** ViTParticleFilterTracker Project Team
**License:** CC-BY 4.0
