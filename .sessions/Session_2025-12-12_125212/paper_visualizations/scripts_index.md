# Paper Visualization Scripts Index

**Created:** 2025-12-12
**Location:** `AdvancedDatasetSelection/paper_visualizations/`

## Scripts Overview

### 1. DINO Features Visualization
**File:** `visualize_dino_features.py`
**Path:** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\visualize_dino_features.py`

**Features:**
- Multi-head attention maps visualization
- CLS token feature extraction
- Attention distribution histograms
- All heads comparison grid
- Overlay heatmaps on original images

**Usage:**
```powershell
py -3.11 paper_visualizations/visualize_dino_features.py --num_images 5 --model dino_vits16
```

**Arguments:**
- `--images_dir` - Input images directory
- `--output_dir` - Output directory (default: `output/dino/`)
- `--num_images` - Number of images to process (default: 5)
- `--model` - DINO model variant: `dino_vits16`, `dino_vits8`, `dino_vitb16`, `dino_vitb8`
- `--show` - Display plots interactively

---

### 2. Fourier Spectrum Analysis
**File:** `visualize_fourier_spectrum.py`
**Path:** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\visualize_fourier_spectrum.py`

**Features:**
- 2D FFT magnitude and phase spectrum
- Frequency band decomposition (low/mid/high)
- Radial frequency profile
- Directional energy analysis (horizontal/vertical/diagonal)
- Band reconstruction visualization
- Spectral entropy calculation

**Usage:**
```powershell
py -3.11 paper_visualizations/visualize_fourier_spectrum.py --num_images 10
```

**Arguments:**
- `--images_dir` - Input images directory
- `--output_dir` - Output directory (default: `output/fourier/`)
- `--num_images` - Number of images to process (default: 10)
- `--show` - Display plots interactively

---

### 3. FastSAM Segmentation
**File:** `visualize_fastsam_segmentation.py`
**Path:** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\visualize_fastsam_segmentation.py`

**Features:**
- Automatic mask generation with FastSAM
- Segment overlay visualization (random colors)
- Complexity metrics (coverage, segment count, edge density)
- Segment statistics bar charts
- Fallback to edge-based segmentation if FastSAM unavailable

**Usage:**
```powershell
py -3.11 paper_visualizations/visualize_fastsam_segmentation.py --num_images 10
```

**Arguments:**
- `--images_dir` - Input images directory
- `--output_dir` - Output directory (default: `output/fastsam/`)
- `--num_images` - Number of images to process (default: 10)
- `--model` - FastSAM model: `FastSAM-s.pt` or `FastSAM-x.pt`
- `--show` - Display plots interactively

---

### 4. Feature Clustering with t-SNE
**File:** `visualize_clustering.py`
**Path:** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\paper_visualizations\visualize_clustering.py`

**Features:**
- K-Means clustering on combined features (DINO + color/texture)
- t-SNE dimensionality reduction
- Elbow method plot for optimal k selection
- Cluster size distribution
- Cluster compactness analysis
- Sample images from each cluster

**Usage:**
```powershell
py -3.11 paper_visualizations/visualize_clustering.py --num_images 500 --n_clusters 10
```

**Arguments:**
- `--images_dir` - Input images directory
- `--output_dir` - Output directory (default: `output/clustering/`)
- `--num_images` - Number of images to process (default: 500)
- `--n_clusters` - Number of clusters (default: 10)
- `--show` - Display plots interactively

---

## Output Directory Structure
```
paper_visualizations/output/
├── dino/
│   ├── {image_name}_dino_analysis.png
│   ├── {image_name}_all_heads.png
│   └── dino_comparison_grid.png
├── fourier/
│   ├── {image_name}_fourier_analysis.png
│   └── fourier_comparison_grid.png
├── fastsam/
│   ├── {image_name}_fastsam_analysis.png
│   └── fastsam_comparison_grid.png
└── clustering/
    ├── clustering_analysis.png
    └── cluster_samples.png
```

## Dependencies
- Python 3.11
- PyTorch 2.7.1+cu118
- torchvision
- opencv-python 4.11.0
- matplotlib
- seaborn
- scikit-learn 1.7.1
- tqdm
- Pillow
- ultralytics (for FastSAM)

## Quick Test Commands
```powershell
cd F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection

# Test single script (Fourier - fastest)
py -3.11 paper_visualizations/visualize_fourier_spectrum.py --num_images 3

# Test all scripts
py -3.11 paper_visualizations/visualize_dino_features.py --num_images 2
py -3.11 paper_visualizations/visualize_fourier_spectrum.py --num_images 3
py -3.11 paper_visualizations/visualize_fastsam_segmentation.py --num_images 3
py -3.11 paper_visualizations/visualize_clustering.py --num_images 100 --n_clusters 5
```
