# Advanced Dataset Selection Pipeline

Pipeline do inteligentnej selekcji najlepszych zdjec treningowych dla DETR.

## Uruchomienie

```bash
# Z glownego folderu projektu:
cd F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker

# Podstawowe uruchomienie (50 zdjec):
py -3.11 AdvancedDatasetSelection/main_selection_pipeline.py --config AdvancedDatasetSelection/config.yaml --target-size 50

# Pelny dataset (5000 zdjec):
py -3.11 AdvancedDatasetSelection/main_selection_pipeline.py --config AdvancedDatasetSelection/config.yaml --target-size 5000

# Testy:
py -3.11 AdvancedDatasetSelection/test_pipeline.py
```

## Output

Wybrane zdjecia trafiaja do:
```
AdvancedDatasetSelection/output/selected_dataset/
├── images/           # Wybrane zdjecia (img_00000.jpg, img_00001.jpg, ...)
├── selection_report.json
└── visualizations/
    ├── pca_coverage.png
    ├── difficulty_distribution.png
    ├── sam_complexity.png
    ├── fourier_diversity.png
    └── selection_summary.png
```

## Jak dziala pipeline

### 4-etapowa selekcja:

1. **Fourier Pre-filtering** - usuwa redundantne obrazy (obecnie wylaczony dla danych medycznych)
2. **Feature Extraction** - DINO (768-dim) + SAM complexity
3. **k-Center Greedy** - wybiera roznorodne obrazy (2x target_size)
4. **EL2N Ranking** - wybiera najtrudniejsze obrazy (final target_size)

### Przeplyw:
```
Input (2083 imgs) -> Fourier -> k-Center (100) -> EL2N -> Output (50 imgs)
```

## Konfiguracja (config.yaml)

Najwazniejsze parametry:

```yaml
datasets:
  new_source: "E:/cataract_surgery_Instruments_detection.v1i.coco/train"
  output: "./output/selected_dataset"

fourier:
  similarity_threshold: 1.1  # >1 = wylaczony (dla danych medycznych)

selection:
  k_center:
    oversampling_factor: 2.0  # 2x wiecej niz target przed EL2N

weights:
  dino_diversity: 0.35
  el2n_difficulty: 0.30
  sam_complexity: 0.20
  fourier_uniqueness: 0.15
```

## Struktura modulow

```
AdvancedDatasetSelection/
├── main_selection_pipeline.py   # Glowny orchestrator
├── config.yaml                  # Konfiguracja
├── test_pipeline.py             # Testy (6/6 passed)
│
├── feature_extractors/
│   ├── dino_extractor.py        # DINO ViT-B/16 (768-dim features)
│   ├── sam_extractor.py         # SAM complexity (bez modelu)
│   └── fourier_analyzer.py      # Analiza czestotliwosci
│
├── selection_methods/
│   ├── el2n_scorer.py           # EL2N difficulty scoring
│   ├── k_center_greedy.py       # Diversity selection
│   └── combined_selector.py     # 4-step hybrid selector
│
└── utils/
    ├── coco_handler.py          # COCO format I/O
    └── visualization.py         # Wykresy PCA, histogramy
```

## Wymagania

```
torch>=2.0
torchvision
numpy
scipy
scikit-learn
opencv-python
matplotlib
tqdm
pyyaml
```

## Przyklad wyniku

```
Selection complete!
  Input: 2083 images
  Output: 50 images
  Reduction: 2.4%

Metryki:
  EL2N difficulty mean: 0.68
  SAM complexity mean: 0.38
```
