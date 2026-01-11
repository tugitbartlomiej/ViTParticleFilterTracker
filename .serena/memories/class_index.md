# Indeks Klas Projektu (Serena MCP)

**Ostatnia aktualizacja:** 2026-01-11 (automatic Serena scan)
**Metoda skanowania:** Serena MCP `get_symbols_overview` z depth=1
**Liczba klas:** 115+
**Liczba plików przeskanowanych:** 60+

---

## Indeks według modułów

### AdvancedDatasetSelection/

#### feature_extractors/
- **dino_extractor.py**
  - `DINOExtractor` - ekstrakcja cech DINO ViT
    - `extract_cls_features()` - ekstrakcja CLS token
    - `compute_features_batch()` - batch processing
    - `compute_semantic_similarity()` - podobieństwo semantyczne
    - `compute_similarity_matrix()` - macierz podobieństwa
    - `_initialize_from_huggingface()` - ładowanie z HuggingFace
    - `_initialize_from_torch_hub()` - ładowanie z Torch Hub

- **fourier_analyzer.py**
  - `FourierAnalyzer` - analiza spektrum Fouriera
    - `compute_frequency_features()` - cechy częstotliwościowe
    - `compute_features_batch()` - przetwarzanie wsadowe
    - `filter_redundant()` - filtrowanie redundancji
    - `filter_redundant_streaming()` - streaming filtering
    - `analyze_diversity()` - analiza różnorodności

- **fastsam_extractor.py**
  - `FastSAMExtractor` - segmentacja FastSAM

- **sam_extractor.py**
  - `SAMExtractor` - segmentacja SAM

#### selection_methods/
- **detr_el2n_scorer.py**
  - `DETR_EL2N_Scorer` - scoring EL2N dla DETR Query 81
    - `compute_single_image_difficulty()` - trudność obrazu
    - `compute_el2n_scores()` - obliczanie EL2N
    - `get_detection_report()` - raport detekcji
    - `save_selected_visualizations()` - wizualizacje
    - `_save_detection_visualization()` - zapis wizualizacji detekcji

- **el2n_scorer.py**
  - `SimpleProxyModel(nn.Module)` - prosty model proxy
    - `forward()` - forward pass
  - `ImageDataset(Dataset)` - dataset obrazów
    - `__len__()`, `__getitem__()`
  - `EL2NScorer` - scoring EL2N
    - `train_proxy_model()` - trening proxy
    - `compute_el2n_scores()` - obliczanie EL2N
    - `compute_el2n_for_detection()` - EL2N dla detekcji
    - `rank_by_difficulty()` - ranking trudności
    - `save_scores()`, `load_scores()` - persistence

- **cluster_selector.py**
  - `ClusterBasedSelector` - selekcja oparta na klastrach
    - `extract_all_features()` - ekstrakcja cech
    - `cluster_features()` - klasteryzacja
    - `select_representatives()` - wybór reprezentantów
    - `select_optimal_subset()` - optymalny podzbiór
    - `_cluster_with_faiss_gpu()` - GPU klasteryzacja
    - `generate_selection_report()` - raport selekcji

- **combined_selector.py**
  - `CombinedSelector` - kombinowana selekcja wieloetapowa
    - `step1_fourier_prefilter()` - prefiltrowanie Fouriera
    - `step2_combine_features()` - kombinowanie cech
    - `step3_k_center_select()` - selekcja K-center
    - `step4_el2n_rank()` - ranking EL2N
    - `compute_combined_score()` - łączny score

- **k_center_greedy.py**
  - `KCenterGreedy` - zachłanna selekcja K-center
    - `compute_distances()` - obliczanie odległości
    - `select()` - selekcja punktów
    - `select_with_diversity_guarantee()` - selekcja z gwarancją różnorodności
    - `analyze_diversity()` - analiza różnorodności
  - `select_diverse_subset()` - funkcja pomocnicza

#### paper_visualizations/
- **visualize_clustering.py**
  - `ClusteringVisualizer` - wizualizacja klastrów z t-SNE
    - `compute_tsne()` / `compute_embedding()` - embedding
    - `visualize_clustering()` - główna wizualizacja
    - `visualize_cluster_samples()` - próbki z klastrów

- **visualize_fourier_spectrum.py**
  - `FourierVisualizer` - wizualizacja spektrum Fouriera
    - `compute_fft()` - obliczanie FFT
    - `visualize_single_image()` - pojedynczy obraz
    - `visualize_comparison()` - porównanie obrazów
    - `visualize_tooltip_vs_background()` - tooltip vs tło
    - `visualize_gallery()` - galeria
    - `visualize_extremes()` - ekstrema

- **visualize_discriminative_per_image.py**
  - `PerImageDiscriminativeVisualizer` - analiza cech dyskryminacyjnych per image
    - `visualize_single()` - pojedynczy obraz
    - `visualize_compare()` - porównanie dwóch obrazów
    - `visualize_compare_multi()` - tryb --multi (5 osobnych plików)
    - `_plot_ring_energy()` - wykres energii pierścieni
    - `_plot_top_features_compare()` - porównanie top cech

#### utils/
- **coco_handler.py**
  - `COCOHandler` - obsługa formatu COCO
    - `load_annotations()` - ładowanie adnotacji
    - `merge_datasets()` - łączenie datasetów
    - `filter_by_image_ids()` - filtrowanie po ID
    - `copy_selected_images()` - kopiowanie wybranych

#### main
- **main_selection_pipeline.py**
  - `AdvancedDatasetSelectionPipeline` - główny pipeline selekcji
    - `load_datasets()` - ładowanie danych
    - `run()` - uruchomienie pipeline'u
    - `_generate_visualizations()` - wizualizacje
    - `_generate_report()` - generowanie raportu

---

### YOLO_DETR_Benchmarks/

#### scripts/
- **cadtd_benchmark.py**
  - `CaDTDBenchmark` - benchmark CaDTD dataset
    - `step1_convert_voc_to_coco()` - konwersja VOC
    - `step2_run_yolo_inference()` - YOLO inference
    - `step3_run_detr_inference()` - DETR inference
    - `evaluate_predictions()` - ewaluacja
    - `run_full_benchmark()` - pełny benchmark

- **benchmark_visualizer.py**
  - `BenchmarkVisualizer` - wizualizacja wyników benchmarków
    - `sample_images()` - próbkowanie obrazów
    - `add_model()` - dodanie modelu
    - `visualize_sample()` - wizualizacja próbki
  - `YOLOWrapper` - wrapper dla modelu YOLO
  - `DETRWrapper` - wrapper dla modelu DETR

- **detr_train_optimized.py**
  - `SurgicalToolDataset(Dataset)` - dataset narzędzi chirurgicznych

- **multi_epoch_query_benchmark.py**
  - `MultiEpochQueryBenchmark` - benchmark query po wielu epokach
    - `analyze_query_distribution()` - analiza rozkładu
    - `run_inference_single_epoch()` - inference dla epoki
    - `plot_epoch_progression()` - wykres progresji

- **generate_benchmark_report.py**
  - `BenchmarkReportGenerator` - generator raportów
    - `generate_comparison_images()` - porównanie obrazów
    - `generate_metrics_infographic()` - infografika metryk
    - `generate_markdown_report()` - raport MD

- **voc_to_coco_converter.py**
  - `VOCToCOCOConverter` - konwerter VOC do COCO

#### scripts/DETR_Background_Training/
- **main_pipeline.py**
  - `PipelineConfig` - konfiguracja pipeline'u
    - `from_yaml()`, `to_yaml()` - serializacja YAML
  - `DETRPipelineOrchestrator` - orkiestrator pipeline'u
    - `stage_1_dino_extraction()` - DINO ekstrakcja
    - `stage_2_dataset_mixing()` - miksowanie datasetów
    - `stage_3_model_preparation()` - przygotowanie modelu
    - `stage_4_gentle_training()` - delikatny trening
    - `stage_5_validation()` - walidacja
    - `generate_final_report()` - raport końcowy

- **status_manager.py**
  - `TaskStatus(Enum)` - enum statusów
  - `ProgressType(Enum)` - typy postępu
  - `TaskProgress` - dataclass postępu
  - `TaskError` - dataclass błędów
  - `TaskInfo` - informacje o tasku
  - `StatusManager` - manager statusów
    - `create_task()`, `start_task()`, `complete_task()`
    - `update_progress()` - aktualizacja postępu
    - `get_summary_report()` - raport podsumowujący

- **pipeline_monitor.py**
  - `LogProgressPattern` - wzorce postępu w logach
  - `LogFileMonitor(FileSystemEventHandler)` - monitor plików logów
  - `ProcessMonitor` - monitor procesów
  - `TaskMonitor` - monitor tasków
  - `PipelineMonitor` - główny monitor pipeline'u
    - `start_monitoring()`, `stop_monitoring()`
    - `get_pipeline_health()` - zdrowie pipeline'u

- **background_task_executor.py**
  - `TaskExecutionConfig` - konfiguracja wykonania
  - `BackgroundProcess` - proces w tle
    - `start()`, `terminate()`, `wait_for_completion()`
  - `BackgroundTaskExecutor` - executor tasków
    - `execute_task()`, `execute_task_with_retry()`

- **progress_reporter.py**
  - `ProgressReporter` - raportowanie postępu
    - `start_stage()`, `complete_stage()`
    - `with_progress()`, `timed_stage()`
  - `TimedStageContext` - context manager dla etapów

- **strategy_trainer.py**
  - `COCODataset(Dataset)` - dataset COCO
  - `MixedDataset(Dataset)` - mieszany dataset
  - `StrategyTrainer` - trener strategii
    - `train_epoch()`, `validate_epoch()`, `train()`

#### DINO_Frame_Selection/
- **dino_frame_selector.py**
  - `DINOFrameSelector` - selektor ramek DINO
    - `extract_frames_from_videos()` - ekstrakcja z wideo
    - `extract_dino_features()` - cechy DINO
    - `perform_clustering_and_selection()` - klasteryzacja i selekcja
    - `create_detr_dataset()` - tworzenie datasetu DETR

- **enhanced_background_selector.py**
  - `EnhancedBackgroundSelector` - zaawansowany selektor tła
    - `load_models()` - ładowanie modeli
    - `detect_tools_yolo()`, `detect_tools_detr()` - detekcja
    - `is_background_frame()` - sprawdzenie tła
    - `select_optimal_frames()` - optymalny wybór

- **validation_framework.py**
  - `ValidationFramework` - framework walidacji
    - `validate_training_efficiency()` - efektywność treningu
    - `validate_model_performance()` - wydajność modelu
    - `validate_attention_alignment()` - wyrównanie attention

- **progressive_fine_tuning_strategy.py**
  - `ProgressiveFinetuningStrategy` - strategia progresywnego fine-tuningu
    - `analyze_frame_quality_distribution()` - rozkład jakości
    - `create_stage_datasets()` - datasety etapów

- **scripts/dino_feature_extractor.py**
  - `DINOFeatureExtractor` - ekstraktor cech DINO
    - `extract_features()`, `extract_patch_features()`
    - `process_image_directory()` - przetwarzanie katalogu

- **scripts/dino_clustering.py**
  - `DINOClustering` - klasteryzacja DINO
    - `find_optimal_clusters()` - optymalna liczba klastrów
    - `perform_clustering()` - klasteryzacja
    - `select_representative_frames()` - reprezentatywne ramki
    - `visualize_clusters()` - wizualizacja

---

### BareDetr/

- **data/dataset.py**
  - `CocoDetectionDataset(torch.utils.data.Dataset)` - bazowy dataset COCO

- **data/transforms.py**
  - `Compose` - kompozycja transformacji
  - `Normalize` - normalizacja
  - `ToTensor` - konwersja do tensora
  - `RandomHorizontalFlip` - losowe odbicie poziome
  - `RandomResize` - losowa zmiana rozmiaru
  - `FixedResize` - stała zmiana rozmiaru
  - `get_transforms()` - funkcja fabrykująca

- **engine.py**
  - `SmoothedValue` - wygładzona wartość metryki
  - `MetricLogger` - logger metryk treningowych
    - `log_every()` - logowanie co N kroków
  - `train_one_epoch()`, `evaluate()` - główne funkcje treningowe

---

### Annotators/

#### DetrAnnotator/
- **detr_annotator_train.py**
  - `SurgicalToolDataset(Dataset)` - dataset DETR
  - `train_epoch()`, `evaluate_model()`, `save_checkpoint()`

- **coco-augmentation.py**
  - `COCOAugmenter` - augmentacja COCO
    - `_create_augmentation_pipeline()` - pipeline augmentacji
    - `augment_dataset()` - augmentacja datasetu

#### Hybrid/
- **yolo_opencv_hybrid.py**
  - `ParticleFilter` - filtr cząsteczkowy dla trackingu
    - `init_around()` - inicjalizacja wokół punktu
    - `predict()`, `update()`, `resample()`
    - `estimate()` - estymacja pozycji

#### DeepSortYolo/
- **deep_sort_yolo_predict.py**
  - `ObjectTracker` - tracker obiektów Deep SORT + YOLO
    - `convert_yolo_to_sort_format()` - konwersja formatu
    - `process_video_file()` - przetwarzanie wideo

#### TimesFormer/
- **timesformer-surgical-train.py**
  - `SurgicalVideoDataset(Dataset)` - dataset wideo

#### Utils/
- **yolo_to_coco_converter.py**
  - `YOLOtoCOCOConverter` - konwerter YOLO do COCO
    - `convert_bbox_yolo_to_coco()` - konwersja bbox
    - `convert()` - główna konwersja

---

## Statystyki

| Moduł | Pliki | Klasy |
|-------|-------|-------|
| AdvancedDatasetSelection | 29 | 24 |
| YOLO_DETR_Benchmarks/scripts | 20 | 18 |
| YOLO_DETR_Benchmarks/DETR_Background_Training | 15 | 22 |
| YOLO_DETR_Benchmarks/DINO_Frame_Selection | 12 | 12 |
| BareDetr | 6 | 8 |
| Annotators | 47 | 15 |
| **TOTAL** | **~130** | **~100** |

---

## Kluczowe klasy według funkcji

### Datasety
- `CocoDetectionDataset` - bazowy dataset COCO (BareDetr)
- `SurgicalToolDataset` - dataset narzędzi chirurgicznych (wielokrotnie)
- `MixedDataset` - mieszany dataset tooltip/background
- `ImageDataset` - prosty dataset obrazów
- `SurgicalVideoDataset` - dataset wideo chirurgicznego

### Feature Extractors
- `DINOExtractor` - cechy DINO ViT (AdvancedDatasetSelection)
- `DINOFeatureExtractor` - ekstrakcja cech DINO (YOLO_DETR)
- `FourierAnalyzer` - analiza Fouriera
- `FastSAMExtractor` / `SAMExtractor` - segmentacja

### Selektory
- `CombinedSelector` - główny selektor wieloetapowy
- `ClusterBasedSelector` - selekcja klastrowa
- `KCenterGreedy` - zachłanna selekcja
- `DETR_EL2N_Scorer` - scoring trudności
- `EL2NScorer` - ogólny scorer EL2N
- `DINOFrameSelector` - selekcja ramek DINO
- `EnhancedBackgroundSelector` - selektor tła

### Pipeline & Orchestration
- `AdvancedDatasetSelectionPipeline` - główny pipeline selekcji
- `DETRPipelineOrchestrator` - orkiestrator treningu DETR
- `StatusManager` - zarządzanie statusem
- `PipelineMonitor` - monitoring pipeline
- `BackgroundTaskExecutor` - wykonywanie tasków w tle
- `ProgressReporter` - raportowanie postępu

### Wizualizacje
- `FourierVisualizer` - spektrum Fouriera
- `ClusteringVisualizer` - klastry t-SNE
- `PerImageDiscriminativeVisualizer` - cechy dyskryminacyjne per image
- `BenchmarkVisualizer` - wyniki benchmarków
- `BenchmarkReportGenerator` - generator raportów

### Training
- `StrategyTrainer` - trener strategii
- `ProgressiveFinetuningStrategy` - strategia progresywnego fine-tuningu
- `ValidationFramework` - framework walidacji

### Benchmarks
- `CaDTDBenchmark` - benchmark CaDTD
- `MultiEpochQueryBenchmark` - benchmark query po wielu epokach

### Converters
- `VOCToCOCOConverter` - VOC do COCO
- `YOLOtoCOCOConverter` - YOLO do COCO
- `COCOAugmenter` - augmentacja COCO

### Transforms (BareDetr)
- `Compose`, `Normalize`, `ToTensor`
- `RandomHorizontalFlip`, `RandomResize`, `FixedResize`

### Tracking
- `ParticleFilter` - filtr cząsteczkowy
- `ObjectTracker` - Deep SORT + YOLO tracker

### Metrics & Logging
- `SmoothedValue` - wygładzona wartość
- `MetricLogger` - logger metryk

---

*Indeks wygenerowany przez Serena MCP: 2026-01-11*
*Projekt: ViTParticleFilterTracker*
*Metoda: `get_symbols_overview` z depth=1 dla pełnej widoczności metod*
