# Indeks Klas Projektu (Auto-generated)

**Ostatnia aktualizacja:** 2025-12-13 01:30:00
**Metoda skanowania:** Grep `^class \w+` + ręczna kategoryzacja
**Liczba klas:** 107
**Liczba plików przeskanowanych:** 50+

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

- **fourier_analyzer.py**
  - `FourierAnalyzer` - analiza spektrum Fouriera
    - `compute_frequency_features()` - cechy częstotliwościowe
    - `compute_features_batch()` - przetwarzanie wsadowe
    - `filter_redundant()` - filtrowanie redundancji
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

- **el2n_scorer.py**
  - `SimpleProxyModel(nn.Module)` - prosty model proxy
  - `ImageDataset(Dataset)` - dataset obrazów
  - `EL2NScorer` - scoring EL2N
    - `train_proxy_model()` - trening proxy
    - `compute_el2n_scores()` - obliczanie EL2N
    - `rank_by_difficulty()` - ranking trudności

- **cluster_selector.py**
  - `ClusterBasedSelector` - selekcja oparta na klastrach
    - `extract_all_features()` - ekstrakcja cech
    - `cluster_features()` - klasteryzacja
    - `select_representatives()` - wybór reprezentantów
    - `select_optimal_subset()` - optymalny podzbiór

- **combined_selector.py**
  - `CombinedSelector` - kombinowana selekcja wieloetapowa
    - `step1_fourier_prefilter()` - prefiltrowanie Fouriera
    - `step2_combine_features()` - kombinowanie cech
    - `step3_k_center_select()` - selekcja K-center
    - `step4_el2n_rank()` - ranking EL2N

- **k_center_greedy.py**
  - `KCenterGreedy` - zachłanna selekcja K-center
    - `compute_distances()` - obliczanie odległości
    - `select()` - selekcja punktów

#### paper_visualizations/
- **visualize_clustering.py**
  - `ClusteringVisualizer` - wizualizacja klastrów z t-SNE
    - `compute_tsne()` - embedding t-SNE
    - `visualize_clustering()` - główna wizualizacja

- **visualize_fourier_spectrum.py**
  - `FourierVisualizer` - wizualizacja spektrum Fouriera
    - `compute_fft()` - obliczanie FFT
    - `visualize_single_image()` - pojedynczy obraz
    - `visualize_comparison()` - porównanie obrazów

- **visualize_dino_features.py**
  - `DINOVisualizer` - wizualizacja cech DINO
    - `get_attention_maps()` - mapy uwagi

- **visualize_discriminative_per_image.py**
  - `PerImageDiscriminativeVisualizer` - analiza cech dyskryminacyjnych per image
    - `visualize_single()` - pojedynczy obraz
    - `visualize_compare()` - porównanie dwóch obrazów
    - `visualize_compare_multi()` - tryb --multi (5 osobnych plików)

- **visualize_discriminative_features.py**
  - `DiscriminativeVisualizer` - batch analiza cech dyskryminacyjnych
    - `compute_batch_features()` - cechy dla batcha
    - `generate_report()` - raport JSON

- **visualize_fastsam_segmentation.py**
  - `FastSAMVisualizer` - wizualizacja segmentacji FastSAM

#### utils/
- **coco_handler.py**
  - `COCOHandler` - obsługa formatu COCO
    - `load_annotations()` - ładowanie adnotacji
    - `merge_datasets()` - łączenie datasetów

- **visualization.py**
  - `Visualizer` - narzędzia wizualizacji
    - `plot_pca_coverage()` - pokrycie PCA
    - `plot_difficulty_distribution()` - rozkład trudności

#### main
- **main_selection_pipeline.py**
  - `AdvancedDatasetSelectionPipeline` - główny pipeline selekcji
    - `load_datasets()` - ładowanie danych
    - `run()` - uruchomienie pipeline'u
    - `_generate_visualizations()` - wizualizacje

---

### YOLO_DETR_Benchmarks/

#### scripts/
- **cadtd_benchmark.py**
  - `CaDTDBenchmark` - benchmark CaDTD dataset

- **benchmark_visualizer.py**
  - `BenchmarkVisualizer` - wizualizacja wyników benchmarków
  - `YOLOWrapper` - wrapper dla modelu YOLO
  - `DETRWrapper` - wrapper dla modelu DETR

- **detr_train_optimized.py**
  - `SurgicalToolDataset(Dataset)` - dataset narzędzi chirurgicznych

- **yolo-train.py**
  - `YOLOTrainer` - trener YOLO

- **detr_training_degradation_analysis.py**
  - `QueryAnalyzer` - analiza query DETR
  - `DETRBenchmark` - benchmark DETR

- **voc_to_coco_converter.py**
  - `VOCToCOCOConverter` - konwerter VOC do COCO

- **multi_epoch_query_benchmark.py**
  - `MultiEpochQueryBenchmark` - benchmark query po wielu epokach

- **generate_benchmark_report.py**
  - `BenchmarkReportGenerator` - generator raportów

#### scripts/DETR_Background_Training/
- **main_pipeline.py**
  - `PipelineConfig` - konfiguracja pipeline'u
  - `DETRPipelineOrchestrator` - orkiestrator pipeline'u
    - `stage_1_dino_extraction()` - DINO ekstrakcja
    - `stage_2_dataset_mixing()` - miksowanie datasetów
    - `stage_3_model_preparation()` - przygotowanie modelu
    - `stage_4_gentle_training()` - delikatny trening
    - `stage_5_validation()` - walidacja

- **status_manager.py**
  - `TaskStatus(Enum)` - enum statusów
  - `ProgressType(Enum)` - typy postępu
  - `TaskProgress` - dataclass postępu
  - `TaskError` - dataclass błędów
  - `TaskInfo` - informacje o tasku
  - `StatusManager` - manager statusów

- **pipeline_monitor.py**
  - `LogProgressPattern` - wzorce postępu w logach
  - `LogFileMonitor(FileSystemEventHandler)` - monitor plików logów
  - `ProcessMonitor` - monitor procesów
  - `TaskMonitor` - monitor tasków
  - `PipelineMonitor` - główny monitor pipeline'u

- **background_task_executor.py**
  - `TaskExecutionConfig` - konfiguracja wykonania
  - `BackgroundProcess` - proces w tle
  - `BackgroundTaskExecutor` - executor tasków

- **progress_reporter.py**
  - `ProgressReporter` - raportowanie postępu
  - `TimedStageContext` - context manager dla etapów

- **strategy_trainer.py**
  - `COCODataset(Dataset)` - dataset COCO
  - `MixedDataset(Dataset)` - mieszany dataset
  - `StrategyTrainer` - trener strategii

- **strategy_testing_framework.py**
  - `StrategyTestingFramework` - framework testowania strategii

- **model_comparison_validator.py**
  - `DETRModelComparator` - porównywarka modeli DETR

- **dino_one_time_extraction.py**
  - `DINOTestDatasetExtractor` - jednorazowa ekstrakcja DINO

- **simple_pipeline.py**
  - `SimpleDETRPipeline` - uproszczony pipeline

- **run_strategy_tests.py**
  - `StrategyTestRunner` - runner testów strategii

- **pipeline_watcher.py**
  - `PipelineWatcher` - obserwator pipeline'u

- **local/mixed_gentle_training.py**
  - `COCODataset(Dataset)` - lokalny dataset COCO
  - `MixedDataset(Dataset)` - lokalny mieszany dataset

- **local/detr_local_train.py**
  - `SurgicalToolDataset(Dataset)` - lokalny dataset

#### DINO_Frame_Selection/
- **visualize_dino_attention.py**
  - `DINOAttentionVisualizer` - wizualizacja uwagi DINO

- **validation_framework.py**
  - `ValidationFramework` - framework walidacji

- **scripts/dino_feature_extractor.py**
  - `DINOFeatureExtractor` - ekstraktor cech DINO

- **scripts/dino_clustering.py**
  - `DINOClustering` - klasteryzacja DINO

- **realtime_pipeline_monitor.py**
  - `RealTimePipelineMonitor` - monitor w czasie rzeczywistym

- **enhanced_background_selector.py**
  - `EnhancedBackgroundSelector` - zaawansowany selektor tła

- **dino_information_analyzer.py**
  - `DINOInformationAnalyzer` - analizator informacji DINO

- **dino_frame_selector.py**
  - `DINOFrameSelector` - selektor ramek DINO

- **integrated_detr_dataset_creator.py**
  - `IntegratedDETRDatasetCreator` - twórca datasetu DETR

- **progressive_fine_tuning_strategy.py**
  - `ProgressiveFinetuningStrategy` - strategia progresywnego fine-tuningu
  - `ProgressiveTrainer` - trener progresywny

---

### Eden/Scripts/

- **YOLOYamlCreator.py**
  - `YOLOYamlCreator` - twórca plików YAML dla YOLO

- **COCOToYOLOLabelsConverter.py**
  - `COCOToYOLOLabelsConverter` - konwerter etykiet COCO do YOLO

- **yolo-train.py / yolo-train_the_best_11062025.py**
  - `YOLOTrainer` - trener YOLO (wersje lokalne)

- **detr_train_optimized.py**
  - `SurgicalToolDataset(Dataset)` - dataset dla treningu DETR

- **20kDataset_11.12.25/detr_train_optimized.py**
  - `SurgicalToolDataset(Dataset)` - dataset 20k fine-tuning
    - **LR Reset Fix** (linie 592-614) - naprawiony bug resetowania LR po resume

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

- **engine.py**
  - `SmoothedValue` - wygładzona wartość metryki
  - `MetricLogger` - logger metryk treningowych

---

### Annotators/

#### DetrAnnotator/
- **detr_annotator_train.py / detr_annotator_train_corrected.py**
  - `SurgicalToolDataset(Dataset)` - dataset DETR

- **detr_annotator_train_range_eden.py**
  - `Config` - konfiguracja
  - `SurgicalToolDataset` - dataset z range

- **coco-augmentation.py**
  - `COCOAugmenter` - augmentacja COCO

- **coco-augmentation_range.py**
  - `COCORangeAugmenter` - augmentacja z range

#### Hybrid/
- **yolo_opencv_hybrid.py**
  - `ParticleFilter` - filtr cząsteczkowy dla trackingu

#### DeepSortYolo/
- **deep_sort_yolo_predict.py**
  - `ObjectTracker` - tracker obiektów Deep SORT + YOLO

#### TimesFormer/
- **timesformer-visualize.py**
  - `SurgicalVideoPredictor` - predyktor wideo chirurgicznego

- **timesformer-surgical-train.py**
  - `SurgicalVideoDataset(Dataset)` - dataset wideo

- **timesformer-auto-annotator.py**
  - `TimesformerAutoAnnotator` - auto-adnotator

#### Yolo/
- **yolo-train.py**
  - `YOLOTrainer` - trener YOLO

#### Utils/
- **yolo_to_coco_converter.py**
  - `YOLOtoCOCOConverter` - konwerter YOLO do COCO

#### OpencvTrackerAnnotator/
- **coco-yolo-converter.py**
  - `COCOtoYOLOConverter` - konwerter COCO do YOLO

---

### Cataract_DETR_DatasetBuilder/

- **eval_coco_offline.py**
  - `InferenceCocoImages(Dataset)` - dataset do inferencji

- **detr_single_class_train.py**
  - `CocoSingleClassDataset(Dataset)` - dataset single class

---

### Utilities/

- **copy_code_only.py**
  - `CodeOnlyCopier` - kopiowanie tylko kodu (bez danych)

- **.sessions/tools/save_session.py**
  - `Colors` - kolory do terminala

---

## Statystyki

| Moduł | Klasy |
|-------|-------|
| AdvancedDatasetSelection | 22 |
| YOLO_DETR_Benchmarks/scripts | 15 |
| YOLO_DETR_Benchmarks/DETR_Background_Training | 25 |
| YOLO_DETR_Benchmarks/DINO_Frame_Selection | 12 |
| Eden/Scripts | 6 |
| BareDetr | 8 |
| Annotators | 14 |
| Cataract_DETR_DatasetBuilder | 2 |
| Other | 3 |
| **TOTAL** | **107** |

---

## Kluczowe klasy według funkcji

### Datasety
- `CocoDetectionDataset` - bazowy dataset COCO (BareDetr)
- `SurgicalToolDataset` - dataset narzędzi chirurgicznych (wielokrotnie)
- `MixedDataset` - mieszany dataset tooltip/background
- `ImageDataset` - prosty dataset obrazów

### Feature Extractors
- `DINOExtractor` - cechy DINO ViT
- `DINOFeatureExtractor` - ekstrakcja cech DINO (YOLO_DETR)
- `FourierAnalyzer` - analiza Fouriera
- `FastSAMExtractor` / `SAMExtractor` - segmentacja

### Selektory
- `CombinedSelector` - główny selektor wieloetapowy
- `ClusterBasedSelector` - selekcja klastrowa
- `KCenterGreedy` - zachłanna selekcja
- `DETR_EL2N_Scorer` - scoring trudności
- `DINOFrameSelector` - selekcja ramek DINO
- `EnhancedBackgroundSelector` - selektor tła

### Pipeline & Orchestration
- `AdvancedDatasetSelectionPipeline` - główny pipeline selekcji
- `DETRPipelineOrchestrator` - orkiestrator treningu DETR
- `StatusManager` - zarządzanie statusem
- `PipelineMonitor` - monitoring
- `BackgroundTaskExecutor` - wykonywanie tasków w tle
- `ProgressReporter` - raportowanie postępu

### Wizualizacje
- `FourierVisualizer` - spektrum Fouriera
- `ClusteringVisualizer` - klastry t-SNE
- `DINOVisualizer` / `DINOAttentionVisualizer` - cechy DINO
- `PerImageDiscriminativeVisualizer` - cechy dyskryminacyjne per image
- `DiscriminativeVisualizer` - batch cechy dyskryminacyjne
- `BenchmarkVisualizer` - wyniki benchmarków

### Training
- `YOLOTrainer` - trener YOLO
- `StrategyTrainer` - trener strategii
- `ProgressiveTrainer` - trener progresywny

### Transforms
- `Compose`, `Normalize`, `ToTensor`
- `RandomHorizontalFlip`, `RandomResize`, `FixedResize`

### Benchmarks
- `CaDTDBenchmark` - benchmark CaDTD
- `DETRBenchmark` - benchmark DETR
- `MultiEpochQueryBenchmark` - benchmark query

### Converters
- `VOCToCOCOConverter` - VOC do COCO
- `COCOToYOLOLabelsConverter` - COCO do YOLO labels
- `YOLOtoCOCOConverter` - YOLO do COCO
- `COCOtoYOLOConverter` - COCO do YOLO

---

*Indeks wygenerowany automatycznie: 2025-12-13 01:30:00*
*Projekt: ViTParticleFilterTracker*
