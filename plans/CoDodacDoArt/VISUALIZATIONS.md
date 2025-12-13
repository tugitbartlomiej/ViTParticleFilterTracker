# Visualization Proposals for IEEE ACCESS Article

**Date:** 2025-12-13
**Purpose:** Catalog existing visualizations and propose new ones for article

---

## Existing Visualizations in Project

### 1. Fourier Spectral Analysis
**Location:** `AdvancedDatasetSelection/paper_visualizations/`

#### 1.1 Discriminative Features Visualization
**Script:** `visualize_discriminative_features.py`
**Status:** ✅ READY TO USE

**Outputs:**
- `discriminative_batch_n50.png` - Statistical comparison (tooltip vs background)
  - Box plots: Top 6 features
  - Violin plots: Feature distributions
  - Ring energy distribution (10 rings)
  - Effect size summary (Cohen's d)

**Recommended for Article:**
- **Figure X:** "Discriminative Fourier Features for Tooltip Detection"
- **Location:** Section 3.2.5 or Results
- **Priority:** 🔴 CRITICAL
- **Modifications needed:**
  - Add subfigure labels (a), (b), (c), (d)
  - Increase font sizes for publication
  - Add p-value annotations (*** for p<0.001)

**Commands to regenerate:**
```powershell
cd F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations
py -3.11 visualize_discriminative_features.py --batch --n_samples 50 --show
```

---

#### 1.2 Fourier Spectrum Gallery
**Script:** `visualize_fourier_spectrum.py`
**Status:** ✅ READY TO USE

**Outputs:**
- `fourier_gallery_n5.png` - Side-by-side comparison of 5 images
  - Original RGB image
  - Grayscale
  - FFT magnitude spectrum
  - Frequency bands overlay
  - Feature statistics table

**Recommended for Article:**
- **Figure Y:** "Fourier Transform Analysis of Surgical Frames"
- **Location:** Methodology Section 3.2.1
- **Priority:** 🟡 HIGH
- **Modifications:**
  - Select 3 representative images (tooltip, background, occluded)
  - Highlight ring bands in spectrum
  - Add interpretive annotations

---

#### 1.3 Discriminative Feature Summary
**Script:** `visualize_discriminative_features.py --summary`
**Data:** `output/discriminative_features.json`
**Status:** ✅ READY TO USE

**Outputs:**
- `discriminative_summary.png`:
  - Top 15 features by AUC (horizontal bar chart)
  - Top 15 by Cohen's d
  - Ring energy comparison (tooltip vs background)
  - Summary statistics table

**Recommended for Article:**
- **Figure Z:** "Feature Ranking and Discriminative Power"
- **Location:** Results Section 5.X
- **Priority:** 🔴 CRITICAL

---

### 2. DINO Feature Visualization
**Location:** `AdvancedDatasetSelection/paper_visualizations/`

#### 2.1 DINO Clustering (t-SNE)
**Script:** `visualize_clustering.py`
**Status:** ✅ READY TO USE

**Outputs:**
- PCA projection (2D) with cluster colors
- Selected representatives marked with stars
- Cluster size distribution histogram
- EL2N distribution per cluster (box plot)

**Recommended for Article:**
- **Figure W:** "DINO Feature Space and Cluster-Based Selection"
- **Location:** Methodology Section 3.2.2
- **Priority:** 🟡 HIGH
- **Modifications:**
  - Add convex hull boundaries around clusters
  - Annotate example images at cluster centers
  - Show connection lines to selected representatives

---

### 3. Checkpoint Analysis Visualizations
**Location:** `Eden/Scripts/CheckpointAnalizis/output/detr/`

#### 3.1 Training Loss Curve
**File:** `detr_training_loss.png` / `detr_training_loss.pdf`
**Status:** ✅ READY TO USE

**Contents:**
- Training loss vs epoch (40-160)
- Min loss annotation
- Trend line

**Recommended for Article:**
- **Figure A:** "DETR Training Loss Progression"
- **Location:** Results Section 5.1
- **Priority:** 🔴 CRITICAL
- **Modifications:**
  - Combine with validation loss (if available)
  - Add epoch milestones (60, 100, 140)
  - Annotate convergence plateau

---

#### 3.2 Weight Norm Evolution
**File:** `detr_weight_norm.png`
**Status:** ✅ READY TO USE

**Recommended for Article:**
- **Supplementary Material** or **Appendix**
- **Priority:** 🟢 MEDIUM

---

#### 3.3 Key Layer Evolution
**File:** `detr_layer_evolution.png`
**Status:** ✅ READY TO USE

**Contents:**
- Classification head weight norm
- BBox predictor norm
- Encoder self-attention norm
- Decoder self-attention norm

**Recommended for Article:**
- **Figure B:** "DETR Layer Weight Evolution During Training"
- **Location:** Results or Discussion
- **Priority:** 🟡 HIGH
- **Shows:** Model stability, no catastrophic changes

---

#### 3.4 Adam Moments
**File:** `detr_adam_moments.png`
**Status:** ✅ READY TO USE

**Recommended for Article:**
- **Supplementary Material**
- **Priority:** 🟢 MEDIUM
- **Audience:** Researchers interested in optimizer dynamics

---

#### 3.5 Training Summary Dashboard
**File:** `detr_training_summary.png`
**Status:** ✅ READY TO USE

**Contents:**
- 4-panel summary: Loss, Weight Norm, LR Schedule, Text Summary

**Recommended for Article:**
- **Figure C:** "DETR Training Dynamics Overview"
- **Location:** Results Section 5.1
- **Priority:** 🔴 CRITICAL
- **Most comprehensive single figure**

---

### 4. YOLO vs DETR Benchmark Visualizations
**Location:** `YOLO_DETR_Benchmarks/benchmark_results/`

#### 4.1 Existing Benchmark Plots
**Status:** ❓ CHECK IF EXISTS

**Needed Visualizations:**
- mAP comparison bar chart (YOLO vs DETR)
- Precision-Recall curves
- Confidence threshold sweep
- Size-based performance (medium vs large objects)

**If missing:** Need to create from JSON data in `benchmark_results/`

---

## NEW Visualizations to Create

### 5. Multi-Epoch Query Specialization
**Priority:** 🔴 CRITICAL
**Location:** NEW (needs to be created)

**Data Sources:**
- Checkpoint analysis results (epochs 40-160)
- Query hit rate distributions per epoch

**Proposed Figure:**
```
Figure: "Query Specialization Evolution Across Training"

Layout: 2 subplots
(a) Line plot: Query 81 hit rate vs epoch
    - X-axis: Epoch (40, 60, 80, 100, 120, 140, 160)
    - Y-axis: % detections from Query 81
    - Annotate emergence (@60), stabilization (@100)

(b) Stacked bar chart: Top 5 queries per epoch
    - Show how query distribution changes
    - Query 81 grows, others shrink
    - Color-coded per query
```

**Creation steps:**
1. Extract query stats from each checkpoint
2. Aggregate hit rates
3. Create matplotlib figure
4. Save as PDF for article

---

### 6. Per-Patient Generalization Heatmap
**Priority:** 🔴 CRITICAL
**Location:** NEW (needs to be created)

**Data Sources:**
- Cross-patient test results (Tables 8-9 in article)
- Per-patient mAP breakdown

**Proposed Figure:**
```
Figure: "Cross-Patient Generalization Matrix"

Layout: Heatmap + bar chart
(a) Heatmap: Training patient (rows) × Test patient (columns)
    - Color intensity = mAP@0.5
    - Diagonal = same-patient (should be high)
    - Off-diagonal = cross-patient (DETR advantage here)

(b) Bar chart: DETR vs YOLO generalization gap per patient
    - X-axis: Test patients
    - Y-axis: mAP difference (DETR - YOLO)
    - Show DETR's consistent advantage
```

---

### 7. Attention Map Visualization (Query 81)
**Priority:** 🟡 HIGH
**Location:** NEW (needs DETR inference code)

**Proposed Figure:**
```
Figure: "Query 81 Attention Patterns"

Layout: 3 rows × 3 columns (9 examples)
Each row:
- Original image
- Query 81 attention heatmap
- Detection result (bbox + confidence)

Cases to show:
Row 1: Clear tooltip → Focused attention on tip
Row 2: Background (clean eye) → Diffuse attention, low conf
Row 3: Partial occlusion → Attention on visible part
```

**Creation steps:**
1. Load DETR checkpoint
2. Hook into decoder attention weights
3. Extract Query 81 attention maps
4. Overlay on images
5. Create grid visualization

---

### 8. Background Training Impact
**Priority:** 🟡 HIGH
**Location:** NEW (needs to be created)

**Data Sources:**
- Model before background training
- Model after background training
- Test on clean eye images

**Proposed Figure:**
```
Figure: "Background-Aware Training Effectiveness"

Layout: 2 subplots
(a) Confusion matrix style:
    - Before: Tooltip TP=80%, Clean FP=45%
    - After:  Tooltip TP=79.5%, Clean FP=8%
    - Show 82% FP reduction

(b) Detection confidence distributions:
    - Tooltip frames: High confidence maintained
    - Background frames: Confidence drops significantly
    - Overlapping histograms
```

---

### 9. Dataset Selection Pipeline Flowchart
**Priority:** 🟡 HIGH
**Location:** NEW (diagram to create)

**Proposed Figure:**
```
Figure: "4-Stage Intelligent Dataset Selection Pipeline"

Layout: Horizontal flowchart with 4 stages
Stage 1: Fourier Pre-filtering
  - Input: 90,000 frames
  - Filter: High/Low ratio, entropy
  - Output: 50,000 frames

Stage 2: DINO Feature Extraction
  - 768-dim embeddings
  - Cosine similarity matrix
  - Output: Feature space

Stage 3: K-Center Greedy / Cluster Selection
  - Diversity guarantee
  - 2x oversampling
  - Output: 40,000 diverse frames

Stage 4: EL2N Ranking
  - Difficulty scoring
  - Select hardest 20,000
  - Output: Final training set

Show reduction numbers and example images at each stage
```

**Tool:** Draw.io, tikz, or matplotlib flowchart

---

### 10. Fourier Frequency Rings Explanation
**Priority:** 🟡 HIGH
**Location:** NEW (pedagogical figure)

**Proposed Figure:**
```
Figure: "Fourier Frequency Band Decomposition"

Layout: 1 large + 10 small subplots
(a) Full FFT spectrum with ring overlays (0-9)
    - Concentric circles showing ring boundaries
    - Color-coded energy levels

(b-k) 10 mini subplots: Individual ring reconstructions
    - Ring 0: Low freq (smooth structure)
    - Ring 1: Low-mid (large textures)
    - ...
    - Ring 9: Highest freq (noise/fine edges)
```

**Purpose:** Help readers understand ring-based feature extraction

---

### 11. LR Scheduler Comparison
**Priority:** 🟢 MEDIUM
**Location:** NEW (ablation experiment)

**Proposed Figure:**
```
Figure: "Learning Rate Schedule Impact on DETR Training"

Layout: 3 subplots
(a) LR vs Epoch for 3 schedules:
    - Constant (flat line)
    - Step decay (staircase)
    - Cosine (smooth curve)

(b) Training loss for each:
    - Constant: Oscillations
    - Step: Sudden drops
    - Cosine: Smooth convergence

(c) Final mAP comparison (bar chart):
    - Constant: 76.2%
    - Step: 78.1%
    - Cosine: 80.8%
```

---

### 12. Cluster-Based vs K-Center Comparison
**Priority:** 🟡 HIGH
**Location:** NEW (ablation experiment)

**Proposed Figure:**
```
Figure: "Dataset Selection Method Comparison"

Layout: 2 subplots
(a) Feature space coverage (PCA):
    - K-Center selected points (blue triangles)
    - Cluster selected points (red circles)
    - Show better coverage with cluster method

(b) mAP vs Selection method (bar chart):
    - Random: 72.3%
    - Fourier only: 74.1%
    - K-Center: 78.5%
    - Cluster (centroid): 79.8%
    - Cluster (max-EL2N): 80.8%
```

---

## Visualization Creation Priority List

### Phase 1: Use Existing (High Priority)
1. ✅ Fourier discriminative features (batch n50)
2. ✅ DETR training summary dashboard
3. ✅ Training loss curve
4. ✅ Layer weight evolution
5. ✅ DINO clustering (t-SNE)

**Action:** Export at 300 DPI, add subfigure labels

---

### Phase 2: Create New (Critical)
6. ❌ Multi-epoch query specialization timeline
7. ❌ Per-patient generalization heatmap
8. ❌ Dataset selection pipeline flowchart

**Action:** Write visualization scripts, generate figures

---

### Phase 3: Create New (High)
9. ❌ Attention map visualization (Query 81)
10. ❌ Background training impact
11. ❌ Fourier rings explanation
12. ❌ Cluster vs K-Center comparison

**Action:** Run new experiments if needed, create visualizations

---

### Phase 4: Optional Enhancements
13. ❌ LR scheduler comparison
14. ❌ Inference time breakdown pie chart
15. ❌ Edge deployment capability matrix

---

## Figure Quality Guidelines for IEEE ACCESS

### Resolution
- **Vector formats preferred:** PDF, EPS
- **Raster if needed:** PNG at 300 DPI minimum
- **Avoid:** JPEG (lossy compression)

### Size
- **Single column:** 3.5 inches (8.9 cm) wide
- **Double column:** 7.16 inches (18.2 cm) wide
- **Height:** No strict limit, but keep < 9 inches

### Fonts
- **Minimum readable size:** 8pt after scaling
- **Recommended:** 10-12pt for labels, 14pt for titles
- **Font family:** Sans-serif (Arial, Helvetica) or serif (Times)

### Colors
- **Use colorblind-friendly palettes**
- **Ensure grayscale readability** (for print)
- **Recommended tools:** ColorBrewer, Seaborn palettes

### Captions
- **Detailed descriptions** (readers should understand without reading text)
- **Subfigure labels:** (a), (b), (c) explained in caption
- **Reference in text:** "As shown in Fig. X(a), ..."

---

## Scripts to Assist Figure Creation

### Matplotlib Template (Publication Quality)
```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# IEEE ACCESS figure settings
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(7.16, 6))  # Double column

# ... plotting code ...

# Save in multiple formats
plt.savefig('figure_name.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure_name.png', bbox_inches='tight', dpi=300)
```

---

## Recommended Color Schemes

### Tooltip vs Background (Colorblind Safe)
- **Tooltip:** `#e74c3c` (red)
- **Background:** `#3498db` (blue)
- **Neutral:** `#95a5a6` (gray)

### Multi-class (for query distribution, etc.)
```python
import seaborn as sns
colors = sns.color_palette("colorblind", 10)
```

### Heatmaps
- **Diverging:** "RdYlBu" (red-yellow-blue)
- **Sequential:** "viridis", "plasma"
- **Avoid:** "jet" (not colorblind friendly)

---

## Figure Checklist Before Submission

- [ ] Resolution ≥ 300 DPI
- [ ] Fonts readable at print size (≥8pt)
- [ ] Subfigure labels (a), (b), (c) present
- [ ] Axes labeled with units
- [ ] Legend included (if multiple series)
- [ ] Colorblind-friendly palette
- [ ] Grayscale readable (test print)
- [ ] Caption written (detailed, standalone)
- [ ] Referenced in article text
- [ ] Source code/data archived for reproducibility

---

**Report Generated:** 2025-12-13
**Visualization Count:** 21 proposed (7 existing, 14 new)
**Critical Figures:** 5-7 for main article
**Supplementary Figures:** 3-5 for appendix
