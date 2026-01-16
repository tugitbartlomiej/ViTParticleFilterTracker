# Indeks Klas Projektu (Serena MCP)

**Ostatnia aktualizacja:** 2026-01-16 22:05:00
**Metoda skanowania:** Serena MCP `get_symbols_overview`
**Liczba klas:** 12
**Liczba plików:** ~80

---

## Indeks według modułów

### AdvancedDatasetSelection/

#### main_selection_pipeline.py
- **`AdvancedDatasetSelectionPipeline`** - Główny pipeline do selekcji danych
  - `__init__()` - inicjalizacja
  - `load_datasets()` - ładowanie datasetów
  - `run()` - uruchomienie pipeline
  - `_generate_output()` - generowanie wyniku
  - `_generate_visualizations()` - wizualizacje
  - `_generate_report()` - raport

#### feature_extractors/
- **dino_extractor.py**
  - `DINOExtractor` - Ekstraktor cech DINOv2 ViT-L/16
    - `extract_cls_features()` - ekstrakcja CLS token
    - `compute_features_batch()` - batch processing
    - `compute_similarity_matrix()` - macierz podobieństwa

- **fastsam_extractor.py**
  - `FastSAMExtractor` - Ekstraktor cech FastSAM (kompleksowość sceny)
    - `extract_masks()` - ekstrakcja masek
    - `compute_scene_complexity()` - obliczanie złożoności
    - `compute_complexity_batch()` - batch processing

- **fourier_analyzer.py**
  - `FourierAnalyzer` - Analiza częstotliwościowa obrazów
    - `compute_frequency_features()` - cechy FFT
    - `compute_similarity_matrix()` - macierz podobieństwa
    - `filter_redundant()` - filtrowanie redundantnych
    - `analyze_diversity()` - analiza różnorodności

#### selection_methods/
- **cluster_selector.py**
  - `ClusterBasedSelector` - Selekcja oparta na klastrowaniu K-means
    - `extract_all_features()` - ekstrakcja wszystkich cech
    - `combine_features()` - łączenie cech z wagami PCA
    - `cluster_features()` - klasteryzacja FAISS GPU
    - `select_representatives()` - wybór reprezentantów
    - `select_optimal_subset()` - optymalny podzbiór
    - `generate_selection_report()` - raport selekcji

- **detr_el2n_scorer.py**
  - `DETR_EL2N_Scorer` - Scorer EL2N używający DETR Query 81
    - `compute_single_image_difficulty()` - trudność obrazu
    - `compute_el2n_scores()` - obliczanie EL2N
    - `save_selected_visualizations()` - wizualizacje

---

### YOLO_DETR_Benchmarks/scripts/

#### benchmark_yolo_vs_detr_q81_all_epoch.py
- **Główne funkcje benchmarkowe:**
  - `load_yolo_model()` - ładowanie YOLO
  - `load_detr_model()` - ładowanie DETR
  - `prepare_coco_annotations()` - przygotowanie annotacji (z exclude_images dla data leakage)
  - `run_inference_on_split()` - inferencja na splicie
  - `evaluate_predictions()` - ewaluacja predykcji
  - `generate_report()` - generowanie raportu
- **Klasy:**
  - `DFLoss` - niestandardowa funkcja straty

#### DETR_Background_Training/
- `BackgroundTaskExecutor` - wykonawca zadań w tle
- `StatusManager` - zarządzanie statusem pipeline
- `PipelineMonitor` - monitorowanie postępu

---

### BareDetr/

#### data/dataset.py
- **`CocoDetectionDataset`** - Dataset COCO dla DETR
  - `__init__()` - inicjalizacja z transformacjami
  - `__getitem__()` - pobieranie próbki

#### engine.py
- **`SmoothedValue`** - wygładzanie wartości metrycznych
- **`MetricLogger`** - logowanie metryk podczas treningu
- **Funkcje:**
  - `train_one_epoch()` - trening jednej epoki
  - `evaluate()` - ewaluacja modelu

---

## Statystyki

| Moduł | Pliki | Klasy |
|-------|-------|-------|
| AdvancedDatasetSelection | 30 | 6 |
| YOLO_DETR_Benchmarks | 49 | 3+ |
| BareDetr | 6 | 3 |
| **TOTAL** | **85** | **12+** |

---

## Kluczowe klasy według funkcji

### Feature Extractors
- `DINOExtractor` - cechy semantyczne DINOv2 (AdvancedDatasetSelection)
- `FastSAMExtractor` - kompleksowość wizualna (AdvancedDatasetSelection)
- `FourierAnalyzer` - analiza częstotliwościowa (AdvancedDatasetSelection)

### Selektory/Pipeline
- `AdvancedDatasetSelectionPipeline` - główny pipeline selekcji
- `ClusterBasedSelector` - selekcja K-means z FAISS GPU
- `DETR_EL2N_Scorer` - scoring trudności EL2N

### Datasety
- `CocoDetectionDataset` - dataset COCO dla DETR (BareDetr)

### Benchmarki
- `DFLoss` - loss dla YOLO
- Funkcje benchmarkowe w `benchmark_yolo_vs_detr_q81_all_epoch.py`

### Training Utils
- `SmoothedValue` - wygładzanie metryk
- `MetricLogger` - logowanie postępu

---

## Ostatnie zmiany (2026-01-16)

### Data Leakage Prevention
Dodano mechanizm wykluczania leaked images w benchmarkach:
- `LEAKED_IMAGES_FILE` - plik JSON z listą 48 leaked images
- `prepare_coco_annotations(exclude_images=)` - parametr wykluczania
- `datasets_full` vs `datasets_clean` - rozdzielenie datasetów

---

*Indeks wygenerowany przez Serena MCP: 2026-01-16 22:05:00*
*Projekt: ViTParticleFilterTracker*
