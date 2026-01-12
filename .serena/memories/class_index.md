# Indeks Klas Projektu (Serena MCP)

**Ostatnia aktualizacja:** 2026-01-12 06:00:00
**Metoda skanowania:** Serena MCP `get_symbols_overview`
**Liczba klas:** 12
**Liczba plików:** 80+

---

## Indeks według modułów

### AdvancedDatasetSelection/

#### main_selection_pipeline.py
- **`AdvancedDatasetSelectionPipeline`** - główny pipeline selekcji datasetu
  - `__init__()` - inicjalizacja pipeline
  - `load_datasets()` - ładowanie datasetów
  - `run()` - uruchomienie pipeline
  - `_generate_output()` - generowanie wyników
  - `_generate_visualizations()` - wizualizacje
  - `_generate_report()` - raport końcowy

#### feature_extractors/dino_extractor.py
- **`DINOExtractor`** - ekstrakcja cech DINO
  - `__init__()` - inicjalizacja modelu DINO
  - `_initialize_model()` - ładowanie modelu
  - `extract_cls_features()` - ekstrakcja CLS token
  - `compute_features_batch()` - batch processing
  - `compute_similarity_matrix()` - macierz podobieństwa

#### feature_extractors/fourier_analyzer.py
- **`FourierAnalyzer`** - analiza Fouriera obrazów
  - `compute_frequency_features()` - cechy częstotliwościowe
  - `compute_features_batch()` - batch processing
  - `filter_redundant()` - filtrowanie redundantnych
  - `analyze_diversity()` - analiza różnorodności

#### selection_methods/cluster_selector.py
- **`ClusterBasedSelector`** - selekcja klastrowa z FAISS
  - `extract_all_features()` - ekstrakcja cech
  - `combine_features()` - łączenie cech
  - `cluster_features()` - klastrowanie FAISS GPU
  - `select_representatives()` - wybór reprezentantów
  - `select_optimal_subset()` - optymalny podzbiór

#### selection_methods/detr_el2n_scorer.py
- **`DETR_EL2N_Scorer`** - scorer EL2N dla DETR
  - `compute_single_image_difficulty()` - trudność obrazu
  - `compute_el2n_scores()` - obliczanie EL2N
  - `get_detection_report()` - raport detekcji
  - `save_selected_visualizations()` - wizualizacje

#### utils/coco_handler.py
- **`COCOHandler`** - obsługa formatu COCO
  - `load_annotations()` - ładowanie adnotacji
  - `get_image_paths()` - ścieżki obrazów
  - `filter_by_image_ids()` - filtrowanie
  - `merge_datasets()` - łączenie datasetów
  - `save_annotations()` - zapis adnotacji

---

### Eden/Scripts/Train20kDataset_YOLO_31.12.25/

#### yolo_train_20k_finetune.py
- **`YOLOFineTuner`** - trening YOLO (v1 - z bugiem resume)
  - `train()` - trening z resume=True (BŁĘDNE)
  - `_save_best_model()` - zapis najlepszego modelu

#### yolo_train_20k_finetune_v2.py (NOWY)
- **`YOLOFineTunerV2`** - trening YOLO (v2 - poprawiony)
  - `__init__()` - ładuje pretrained weights (NIE resume)
  - `train()` - nowy trening bez resume=True
  - `_save_best_model()` - zapis najlepszego modelu
  - `_save_emergency_checkpoint()` - checkpoint awaryjny

---

### YOLO_DETR_Benchmarks/scripts/

#### benchmark_cataract_complete.py
- **`DFLoss`** - funkcja straty
- `benchmark_split()` - benchmark na podziale danych
- `evaluate()` - ewaluacja modelu

---

## Statystyki

| Moduł | Pliki | Klasy |
|-------|-------|-------|
| AdvancedDatasetSelection | 29 | 6 |
| YOLO_DETR_Benchmarks | 48 | 2 |
| Eden/Scripts | 2 | 2 |
| **TOTAL** | **79** | **10** |

---

## Kluczowe klasy według funkcji

### Feature Extractors
- `DINOExtractor` - cechy semantyczne DINO (AdvancedDatasetSelection)
- `FourierAnalyzer` - cechy częstotliwościowe FFT (AdvancedDatasetSelection)

### Selektory/Pipeline
- `AdvancedDatasetSelectionPipeline` - główny pipeline (AdvancedDatasetSelection)
- `ClusterBasedSelector` - selekcja klastrowa FAISS (AdvancedDatasetSelection)
- `DETR_EL2N_Scorer` - scorer trudności EL2N (AdvancedDatasetSelection)

### Training
- `YOLOFineTuner` - YOLO training v1 z bugiem (Eden)
- `YOLOFineTunerV2` - YOLO training v2 poprawiony (Eden) **NOWY**

### Utils
- `COCOHandler` - obsługa formatu COCO (AdvancedDatasetSelection)

---

## Ostatnie zmiany (2026-01-12)

### Dodano:
- `YOLOFineTunerV2` - poprawiony skrypt treningu YOLO
  - Fix: używa `--pretrained_path` zamiast `resume=True`
  - Pozwala trenować zakończone checkpointy dalej

---

*Indeks wygenerowany przez Serena MCP: 2026-01-12 06:00:00*
*Projekt: ViTParticleFilterTracker*
