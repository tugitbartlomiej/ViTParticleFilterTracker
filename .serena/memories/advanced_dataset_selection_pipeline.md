# Advanced Dataset Selection Pipeline

## Lokalizacja
`AdvancedDatasetSelection/` - pipeline do inteligentnej selekcji zdjęć treningowych dla DETR

## Główne komponenty

### 1. Main Pipeline (`main_selection_pipeline.py`)
- **Klasa:** `AdvancedDatasetSelectionPipeline`
- **Metody:**
  - `run()` - główny orchestrator 4-etapowej selekcji
  - `_load_images()` - ładowanie ścieżek obrazów z folderu
  - `_copy_selected_images()` - kopiowanie wybranych do output
  - `_generate_report()` - tworzenie JSON raportu

### 2. Feature Extractors (`feature_extractors/`)

#### FourierAnalyzer (`fourier_analyzer.py`)
- **Klasa:** `FourierAnalyzer`
- **Funkcja:** Analiza widma częstotliwości obrazów (5-dim features)
- **Metody:**
  - `compute_features()` - FFT + band energy extraction
  - `compute_features_batch()` - batch processing
  - `filter_redundant()` - usuwanie podobnych obrazów (cosine similarity)
  - `analyze_diversity()` - statystyki podobieństwa
- **Uwaga:** Dla danych medycznych threshold musi być >0.999 (wszystkie obrazy mają similarity ~0.9999)

#### DINOExtractor (`dino_extractor.py`)
- **Klasa:** `DINOExtractor`
- **Funkcja:** Ekstrakcja 768-dim CLS token z DINO ViT-B/16
- **Metody:**
  - `extract_features()` - single image
  - `extract_batch()` - batch processing
  - `compute_similarity_matrix()` - pairwise cosine similarity

#### SAMExtractor (`sam_extractor.py`)
- **Klasa:** `SAMExtractor`
- **Funkcja:** Complexity scoring (proxy bez modelu SAM)
- **Metody:**
  - `compute_complexity_without_sam()` - edge + blob detection
  - `process_batch()` - batch processing
- **Metryki:** edge_density, blob_count, texture_score, complexity_score

### 3. Selection Methods (`selection_methods/`)

#### EL2NScorer (`el2n_scorer.py`)
- **Klasa:** `EL2NScorer`
- **Funkcja:** Scoring trudności próbek: ||softmax(pred) - one_hot(label)||₂
- **Metody:**
  - `train_proxy_model()` - trenuje prosty CNN na danych
  - `compute_el2n_scores()` - oblicza EL2N dla wszystkich samples
  - `rank_by_difficulty()` - sortuje od najtrudniejszych
- **Funkcja standalone:** `compute_proxy_el2n_from_features()` - EL2N z gotowych features

#### KCenterGreedy (`k_center_greedy.py`)
- **Klasa:** `KCenterGreedy`
- **Funkcja:** Selekcja różnorodnych próbek (maximizes min distance)
- **Metody:**
  - `select()` - główna selekcja k samples
  - `analyze_diversity()` - statystyki wybranych
- **Metryki:** euclidean distance

#### CombinedSelector (`combined_selector.py`)
- **Klasa:** `CombinedSelector`
- **Funkcja:** 4-etapowa hybrydowa selekcja
- **Pipeline:**
  1. Fourier pre-filtering (redundancy removal)
  2. Feature extraction (DINO 768-dim + SAM complexity)
  3. k-Center Greedy (diversity: 2x target_size)
  4. EL2N ranking (difficulty: final target_size)
- **Metody:**
  - `select_optimal_subset()` - główna metoda selekcji
  - `_extract_all_features()` - ekstrakcja wszystkich features
  - `_combine_features()` - łączenie DINO + SAM

### 4. Utils (`utils/`)

#### COCOHandler (`coco_handler.py`)
- **Funkcja:** I/O dla formatu COCO annotations

#### Visualizer (`visualization.py`)
- **Klasa:** `Visualizer`
- **Metody:**
  - `plot_pca_coverage()` - PCA projekcja wybranych/niewybranych
  - `plot_difficulty_distribution()` - histogram EL2N scores
  - `plot_sam_complexity()` - histogram SAM complexity
  - `plot_selection_summary()` - podsumowanie wielopanelowe

## Konfiguracja (`config.yaml`)
```yaml
datasets:
  new_source: "E:/cataract_surgery_Instruments_detection.v1i.coco/train"
  output: "./output/selected_dataset"

fourier:
  similarity_threshold: 1.1  # >1 = wyłączony

selection:
  k_center:
    oversampling_factor: 2.0

weights:
  dino_diversity: 0.35
  el2n_difficulty: 0.30
  sam_complexity: 0.20
  fourier_uniqueness: 0.15
```

## Uruchomienie
```bash
cd F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker
py -3.11 AdvancedDatasetSelection/main_selection_pipeline.py --config AdvancedDatasetSelection/config.yaml --target-size 50
```

## Output
```
AdvancedDatasetSelection/output/selected_dataset/
├── images/                    # Wybrane obrazy
├── selection_report.json      # Raport z metrykami
└── visualizations/            # Wykresy PCA, histogramy
```

## Typowy wynik
```
Input: 2083 images → k-Center: 100 → EL2N: 50 images
EL2N difficulty mean: 0.68
SAM complexity mean: 0.38
```
