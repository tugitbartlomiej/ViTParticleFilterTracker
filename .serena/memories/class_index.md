# Indeks Klas Projektu

**Ostatnia aktualizacja:** 2025-12-12 20:25:00
**Liczba klas:** 98
**Liczba plików:** 78

---

## Indeks według modułów

### AdvancedDatasetSelection/

#### feature_extractors/
- **dino_extractor.py:30**
  - `DINOExtractor` - ekstrakcja cech DINO ViT

- **fastsam_extractor.py:30**
  - `FastSAMExtractor` - segmentacja FastSAM

- **fourier_analyzer.py:22**
  - `FourierAnalyzer` - analiza spektrum Fouriera

- **sam_extractor.py:24**
  - `SAMExtractor` - segmentacja SAM

#### selection_methods/
- **cluster_selector.py:37**
  - `ClusterBasedSelector` - selekcja oparta na klastrach

- **combined_selector.py:29**
  - `CombinedSelector` - kombinowana selekcja

- **detr_el2n_scorer.py:32**
  - `DETR_EL2N_Scorer` - scoring EL2N dla DETR z Q81

- **el2n_scorer.py**
  - `SimpleProxyModel(nn.Module):25` - prosty model proxy
  - `ImageDataset(Dataset):56` - dataset obrazów
  - `EL2NScorer:81` - scoring EL2N

- **k_center_greedy.py:19**
  - `KCenterGreedy` - zachłanna selekcja K-center

#### paper_visualizations/
- **visualize_clustering.py:54**
  - `ClusteringVisualizer` - wizualizacja klastrów z t-SNE

- **visualize_dino_features.py:49**
  - `DINOVisualizer` - wizualizacja cech DINO

- **visualize_fastsam_segmentation.py:55**
  - `FastSAMVisualizer` - wizualizacja segmentacji FastSAM

- **visualize_fourier_spectrum.py:162**
  - `FourierVisualizer` - wizualizacja spektrum Fouriera

#### utils/
- **coco_handler.py:17**
  - `COCOHandler` - obsługa formatu COCO

- **visualization.py:15**
  - `Visualizer` - narzędzia wizualizacji

#### main
- **main_selection_pipeline.py:65**
  - `AdvancedDatasetSelectionPipeline` - główny pipeline selekcji

---

### YOLO_DETR_Benchmarks/scripts/

#### Main Scripts
- **benchmark_visualizer.py**
  - `BenchmarkVisualizer:54` - wizualizacja benchmarków
  - `YOLOWrapper:361` - wrapper YOLO
  - `DETRWrapper:395` - wrapper DETR

- **cadtd_benchmark.py:53**
  - `CaDTDBenchmark` - benchmark CaDTD

- **detr_annotator_train_corrected.py:47**
  - `SurgicalToolDataset(Dataset)` - dataset narzędzi chirurgicznych

- **detr_train_optimized.py:51**
  - `SurgicalToolDataset(Dataset)` - zoptymalizowany dataset

- **detr_training_degradation_analysis.py**
  - `QueryAnalyzer:74` - analiza query DETR
  - `DETRBenchmark:162` - benchmark DETR

- **generate_benchmark_report.py:22**
  - `BenchmarkReportGenerator` - generator raportów

- **multi_epoch_query_benchmark.py:24**
  - `MultiEpochQueryBenchmark` - benchmark wielu epok

- **voc_to_coco_converter.py:12**
  - `VOCToCOCOConverter` - konwerter VOC→COCO

- **yolo-train.py:36**
  - `YOLOTrainer` - trener YOLO

#### DETR_Background_Training/
- **background_task_executor.py**
  - `TaskExecutionConfig:38` - konfiguracja tasków
  - `BackgroundProcess:64` - proces w tle
  - `BackgroundTaskExecutor:360` - executor tasków

- **detr_background_finetune_fixed.py:52**
  - `SurgicalToolDataset(Dataset)` - dataset z fixami

- **detr_background_finetune_single.py:25**
  - `SurgicalToolDataset(Dataset)` - single dataset

- **detr_background_only_finetune.py:46**
  - `BackgroundDataset(Dataset)` - dataset tła

- **detr_finetune_background.py:46**
  - `SurgicalToolDataset(Dataset)` - finetune tła

- **dino_one_time_extraction.py:28**
  - `DINOTestDatasetExtractor` - jednorazowa ekstrakcja DINO

- **main_pipeline.py**
  - `PipelineConfig:54` - konfiguracja pipeline'u
  - `DETRPipelineOrchestrator:139` - orkiestrator pipeline'u

- **model_comparison_validator.py:24**
  - `DETRModelComparator` - porównanie modeli DETR

- **pipeline_monitor.py**
  - `LogProgressPattern:35` - wzorce postępu
  - `LogFileMonitor(FileSystemEventHandler):82` - monitor logów
  - `ProcessMonitor:185` - monitor procesów
  - `TaskMonitor:256` - monitor tasków
  - `PipelineMonitor:379` - główny monitor

- **pipeline_watcher.py:25**
  - `PipelineWatcher` - obserwator pipeline'u

- **progress_reporter.py**
  - `ProgressReporter:34` - reporter postępu
  - `TimedStageContext:257` - kontekst etapów

- **run_strategy_tests.py:21**
  - `StrategyTestRunner` - runner testów strategii

- **simple_pipeline.py:28**
  - `SimpleDETRPipeline` - prosty pipeline DETR

- **status_manager.py**
  - `TaskStatus(Enum):28` - enum statusów
  - `ProgressType(Enum):37` - typy postępu
  - `TaskProgress:45` - postęp zadań
  - `TaskError:58` - błędy zadań
  - `TaskInfo:72` - info o zadaniach
  - `StatusManager:89` - manager statusów

- **strategy_testing_framework.py:25**
  - `StrategyTestingFramework` - framework testów

- **strategy_trainer.py**
  - `COCODataset(Dataset):26` - dataset COCO
  - `MixedDataset(Dataset):90` - mieszany dataset
  - `StrategyTrainer:122` - trener strategii

#### local/
- **detr_local_train.py:29**
  - `SurgicalToolDataset(Dataset)` - lokalny dataset

- **mixed_gentle_training.py**
  - `COCODataset(Dataset):24` - COCO dataset
  - `MixedDataset(Dataset):91` - mieszany dataset

---

### BareDetr/

- **data/dataset.py:10**
  - `CocoDetectionDataset(torch.utils.data.Dataset)` - bazowy dataset COCO

- **data/transforms.py**
  - `Compose:11` - kompozycja transformacji
  - `Normalize:27` - normalizacja
  - `ToTensor:44` - konwersja do tensora
  - `RandomHorizontalFlip:53` - losowe odbicie
  - `RandomResize:78` - losowa zmiana rozmiaru
  - `FixedResize:106` - stała zmiana rozmiaru

- **engine.py**
  - `SmoothedValue:251` - wygładzona wartość
  - `MetricLogger:292` - logger metryk

---

### Annotators/

#### DetrAnnotator/
- **coco-augmentation.py:55**
  - `COCOAugmenter` - augmentacja COCO

- **coco-augmentation_range.py:53**
  - `COCORangeAugmenter` - augmentacja zakresowa

- **coco_augmentation_archivier.py:14**
  - `COCOAugmenter` - archiwalna augmentacja

- **detr_annotator_train.py:16**
  - `SurgicalToolDataset(torch.utils.data.Dataset)` - bazowy dataset

- **detr_annotator_train_corrected.py:22**
  - `SurgicalToolDataset(Dataset)` - poprawiony dataset

- **detr_annotator_train_range.py:14**
  - `SurgicalToolDataset(torch.utils.data.Dataset)` - zakresowy dataset

- **detr_annotator_train_range_eden.py**
  - `Config:20` - konfiguracja Eden
  - `SurgicalToolDataset(torch.utils.data.Dataset):67` - dataset Eden

- **detr_annotator_train_range_optimized.py:27**
  - `SurgicalToolDataset(torch.utils.data.Dataset)` - zoptymalizowany

- **detr_annotator_train_range_optimized_augmented.py:29**
  - `SurgicalToolDataset(torch.utils.data.Dataset)` - z augmentacją

- **detr_annotator_inference_range_images.py:13**
  - `SurgicalToolDataset(Dataset)` - inference dataset

#### DeepSortYolo/
- **deep_sort_yolo_predict.py:15**
  - `ObjectTracker` - tracker obiektów

- **sort/sort.py**
  - `KalmanBoxTracker(object):94` - tracker Kalman
  - `Sort(object):199` - algorytm SORT

#### Hybrid/
- **yolo_opencv_hybrid.py:50**
  - `ParticleFilter` - filtr cząsteczkowy

#### TimesFormer/
- **timesformer-auto-annotator.py:11**
  - `TimesformerAutoAnnotator` - auto-annotator

- **timesformer-surgical-train.py:13**
  - `SurgicalVideoDataset(Dataset)` - dataset video

- **timesformer-surgical-test.py:11**
  - `SurgicalVideoPredictor` - predyktor

- **timesformer-visualize*.py**
  - `SurgicalVideoPredictor` - wizualizacja (3 warianty)

#### Utils/
- **yolo_to_coco_converter.py:10**
  - `YOLOtoCOCOConverter` - konwerter YOLO→COCO

#### Utils/SignificantImageSelector/
- **image_feature_extractor.py:21**
  - `ImageFeatureExtractor` - ekstraktor cech obrazów

- **annotation_feature_extractor.py:18**
  - `YOLOAnnotationFeatureExtractor` - ekstraktor cech YOLO

- **image_clustering.py:20**
  - `ImageClusterer` - klasteryzacja obrazów

- **results_visualizer.py:23**
  - `ResultsVisualizer` - wizualizacja wyników

#### OpencvTrackerAnnotator/
- **coco-yolo-converter.py:13**
  - `COCOtoYOLOConverter` - konwerter COCO→YOLO

#### Yolo/
- **yolo-train.py:27**
  - `YOLOTrainer` - trener YOLO

---

### BackgroundFinetuned/

#### Main Scripts
- **evaluate_mixed_detr.py:50**
  - `MixedDetrDataset(torch.utils.data.Dataset)` - ewaluacja

- **main_train.py:19**
  - `ConfigurableTrainer` - konfigurowalny trener

- **main_generate.py:79**
  - `DatasetGenerator` - generator datasetów

- **production_finetune_mixed.py:122**
  - `MixedDetrDataset(Dataset)` - produkcyjny mixed dataset

- **production_finetune_mixed_v_2.py:103**
  - `MixedDetrDataset(Dataset)` - v2

- **production_q81_trainer.py**
  - `BalancedSurgicalDataset(Dataset):38` - zbalansowany dataset
  - `DETRLoss(nn.Module):177` - loss DETR

- **proper_detr_trainer.py**
  - `ProperDETRDataset(Dataset):34` - proper dataset
  - `HungarianMatcher(nn.Module):164` - Hungarian matcher
  - `DETRLoss(nn.Module):233` - loss funkcja

- **query81_trainer.py**
  - `MixedDataset(Dataset):30` - mixed dataset
  - `Query81SpecializedLoss(nn.Module):97` - specjalizowany loss Q81
  - `Query81Trainer:151` - trener Q81

- **simple_production_trainer.py**
  - `SimpleDETRDataset(Dataset):24` - prosty dataset
  - `SimpleQ81Loss(nn.Module):93` - prosty loss Q81

- **simple_q81_finetuner.py**
  - `SimpleDataset(Dataset):25` - prosty dataset
  - `Query81Loss(nn.Module):88` - Q81 loss

- **test_model.py**
  - `ValidationDataset(Dataset):36` - walidacyjny dataset
  - `ModelTester:71` - tester modeli

- **test_query81_quality.py:28**
  - `Query81Tester` - tester jakości Q81

- **visualize_q81_results.py:22**
  - `QueryVisualizationTester` - wizualizacja Q81

- **wsl6_production_detector.py:37**
  - `ProductionQuery81Detector` - produkcyjny detektor

#### Src/trainer/
- **mixed_detr_finetune.py**
  - `MixedDatasetItem:138` - item datasetu
  - `MixedCOCODataset(Dataset):144` - mixed COCO

- **mixed_detr_trainer_v2.py**
  - `BackgroundAndToolMixDataset(Dataset):54` - mix tła i narzędzi
  - `SimplifiedDETRLoss(nn.Module):185` - uproszczony loss
  - `MixedDETRTrainerV2:198` - trener v2

#### Src/dataset_creator/
- **dino_information_analyzer.py:41**
  - `DINOInformationAnalyzer` - analizator DINO

- **intelligent_background_frame_selectorDYD.py**
  - `TorchLoadPatcher:67` - patcher torch.load
  - `IntelligentBackgroundFrameSelector:108` - inteligentny selektor

- **mix_background_and_tooltip_coco.py:50**
  - `Coco` - helper COCO

---

## Statystyki

| Moduł | Pliki | Klasy |
|-------|-------|-------|
| AdvancedDatasetSelection | 17 | 18 |
| YOLO_DETR_Benchmarks | 26 | 48 |
| BareDetr | 4 | 9 |
| Annotators | 21 | 28 |
| BackgroundFinetuned | 22 | 31 |
| **TOTAL** | **90** | **134** |

---

## Kluczowe klasy według funkcji

### Datasety (Dataset/torch.utils.data.Dataset)
- `SurgicalToolDataset` - główny dataset narzędzi (wiele wariantów)
- `MixedDataset` / `MixedCOCODataset` - mieszane datasety
- `BackgroundDataset` - dataset tła
- `CocoDetectionDataset` - bazowy COCO

### Trenery
- `ConfigurableTrainer` - konfigurowalny trener
- `Query81Trainer` - specjalizowany trener Q81
- `StrategyTrainer` - trener strategii
- `YOLOTrainer` - trener YOLO

### Feature Extractors
- `DINOExtractor` - cechy DINO ViT
- `FourierAnalyzer` - analiza Fouriera
- `FastSAMExtractor` / `SAMExtractor` - segmentacja

### Selektory/Pipeline
- `AdvancedDatasetSelectionPipeline` - główny pipeline
- `CombinedSelector` - kombinowana selekcja
- `ClusterBasedSelector` - selekcja klastrowa
- `DETRPipelineOrchestrator` - orkiestrator DETR

### Wizualizacje
- `FourierVisualizer` - spektrum Fouriera
- `ClusteringVisualizer` - klastry t-SNE
- `DINOVisualizer` - cechy DINO
- `BenchmarkVisualizer` - benchmarki

### Loss Functions (nn.Module)
- `DETRLoss` - standardowy loss DETR
- `Query81Loss` / `Query81SpecializedLoss` - loss Q81
- `HungarianMatcher` - matching Węgierski

---

*Indeks wygenerowany: 2025-12-12 20:25:00*
*Projekt: ViTParticleFilterTracker*
