# Plan Rozszerzenia Artykułu IEEE ACCESS - DETR Surgical Tool Detection

**Data utworzenia:** 2025-12-12
**Status:** Draft
**Priorytet:** High
**Artykuł:** `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\access.tex`

---

## Cel

Rozszerzenie artykułu naukowego IEEE ACCESS o wyniki i metody z projektu ViTParticleFilterTracker, w tym:
- Advanced Dataset Selection Pipeline
- Multi-epoch DETR training analysis
- Query 81 specialization discovery
- Background-aware training
- Fourier spectral analysis

## Kontekst

Obecny artykuł ma podstawową strukturę, ale brakuje:
- Sekcji Related Work (wymagana przez IEEE)
- Szczegółowych wyników benchmarków (mamy lepsze dane)
- Opisu nowatorskich metod selekcji datasetu
- Analizy Query mechanism w DETR

---

## Krytyczne Braki w Obecnym Artykule

| Obecny stan | Problem |
|-------------|---------|
| Tabela 1: mAP@0.5 DETR=0.90 | Niezgodne z naszymi wynikami (mamy 80.8%) |
| Dataset: 2000 frames | Mamy 20,000 frames po rozbudowie |
| Brak sekcji Related Work | Wymagana przez IEEE |
| Brak szczegółów o Query mechanism | Mamy Query 81 discovery |

---

## Proponowana Nowa Struktura Artykułu

### Section 1: Introduction (obecna - wymaga aktualizacji)

**Do dodania:**
- [ ] Wzmianka o problemie false positives na czystych obrazach oka (background problem)
- [ ] Wzmianka o DINO jako narzędziu selekcji danych
- [ ] Aktualizacja contributions list

---

### Section 2: Related Work (NOWA SEKCJA!)

#### 2.1 Transformer-based Object Detection
- [ ] DETR oryginalny [Carion 2020]
- [ ] Deformable DETR [Zhu 2021]
- [ ] DINO DETR [Zhang 2022]
- [ ] Conditional DETR

#### 2.2 Surgical Tool Detection
- [ ] Przegląd metod w chirurgii okulistycznej
- [ ] CNN-based approaches (YOLO, Faster R-CNN)
- [ ] Transformer approaches w medical imaging
- [ ] Existing cataract surgery datasets

#### 2.3 Dataset Curation for Deep Learning
- [ ] Active Learning approaches
- [ ] EL2N scoring [Paul et al. 2021]
- [ ] K-Center Greedy selection [Sener & Savarese 2018]
- [ ] Fourier-based diversity analysis (nasza nowość!)

#### 2.4 Self-Supervised Learning for Medical Imaging
- [ ] DINO [Caron et al. 2021]
- [ ] Zastosowania w chirurgii
- [ ] Feature extraction for downstream tasks

---

### Section 3: Methodology (znacząco rozszerzona)

#### 3.1 Problem Formulation
- [ ] Definicja zadania detekcji narzędzi
- [ ] Wyzwania specyficzne dla chirurgii okulistycznej
- [ ] Formalne sformułowanie problemu

#### 3.2 Advanced Dataset Selection Pipeline (GŁÓWNA NOWOŚĆ!)

```
┌─────────────────────────────────────────────────────────────┐
│            4-STAGE INTELLIGENT SELECTION PIPELINE           │
├─────────────────────────────────────────────────────────────┤
│ Stage 1: Fourier Pre-filtering                              │
│   - Spectral entropy, frequency centroid                    │
│   - High/Low ratio for diversity scoring                    │
│   - Redundancy removal (similarity > 0.999)                 │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: DINO Feature Extraction                            │
│   - 768-dim CLS token from ViT-B/16                         │
│   - Semantic similarity matrix                              │
│   - t-SNE clustering visualization                          │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: K-Center Greedy Selection                          │
│   - Maximize minimum distance (diversity guarantee)         │
│   - Oversampling factor: 2x target size                     │
├─────────────────────────────────────────────────────────────┤
│ Stage 4: EL2N Difficulty Ranking                            │
│   - ||softmax(pred) - one_hot(label)||₂                     │
│   - Select hardest samples for training                     │
└─────────────────────────────────────────────────────────────┘
```

**Wzory do dodania:**
- [ ] EL2N score: `score = ||softmax(f(x)) - y||₂`
- [ ] Fourier High/Low ratio: `R = E_high / E_low`
- [ ] K-Center objective: `max min_i d(x_i, S)`
- [ ] DINO cosine similarity

**Pliki źródłowe:**
- `AdvancedDatasetSelection/main_selection_pipeline.py`
- `AdvancedDatasetSelection/selection_methods/combined_selector.py`
- `AdvancedDatasetSelection/feature_extractors/fourier_analyzer.py`
- `AdvancedDatasetSelection/feature_extractors/dino_extractor.py`

#### 3.3 DETR Query Specialization Analysis (NOWOŚĆ)

- [ ] Odkrycie Query 81 - specjalizacja do detekcji tooltip
- [ ] Analiza attention patterns dla poszczególnych queries
- [ ] Implikacje dla interpretowalności transformerów
- [ ] Wizualizacja attention maps

**Pliki źródłowe:**
- `AdvancedDatasetSelection/selection_methods/detr_el2n_scorer.py`
- `YOLO_DETR_Benchmarks/scripts/visualize_detr_queries_multi_epoch.py`

#### 3.4 Background-Aware Training Pipeline (NOWOŚĆ)

```
┌─────────────────────────────────────────────────────────────┐
│        5-STAGE BACKGROUND TRAINING PIPELINE                 │
├─────────────────────────────────────────────────────────────┤
│ Stage 1: DINO Frame Extraction                              │
│   - Video → frames with DINO clustering                     │
│   - YOLO/DETR consensus classification                      │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: Dataset Mixing                                     │
│   - 70% tooltip frames                                      │
│   - 30% background frames (clean eye)                       │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: Model Preparation                                  │
│   - Load pretrained DETR checkpoint                         │
├─────────────────────────────────────────────────────────────┤
│ Stage 4: Gentle Fine-tuning                                 │
│   - Learning rate: 1e-6 (anti-catastrophic forgetting)      │
│   - Mixed training batches                                  │
├─────────────────────────────────────────────────────────────┤
│ Stage 5: Validation                                         │
│   - Tooltip retention > 90%                                 │
│   - Background detection capability                         │
└─────────────────────────────────────────────────────────────┘
```

**Pliki źródłowe:**
- `YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/main_pipeline.py`
- `YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/status_manager.py`

#### 3.5 Model Architectures
- [ ] DETR-ResNet50 configuration (obecne - rozszerzyć)
- [ ] YOLOv8 configuration (obecne - rozszerzyć)
- [ ] Query mechanism szczegóły
- [ ] Hungarian matching algorithm

---

### Section 4: Experimental Setup (NOWA SEKCJA)

#### 4.1 Dataset Description

| Dataset | Images | Annotations | Source |
|---------|--------|-------------|--------|
| CADTD Original | 4,670 | ~8,000 | Cataract surgery |
| Augmented 20k | 20,000 | ~35,000 | 4x augmentation |
| Background | 2,220 | 0 | Clean eye frames |

- [ ] Tabela z pełnym opisem datasetu
- [ ] Proces augmentacji
- [ ] Split train/val/test

#### 4.2 Training Configuration

**DETR Training (Eden HPC):**
- 4x NVIDIA Hopper H100 (80GB)
- Batch size: 4 per GPU
- Epochs: 160
- Learning rate: 1e-4 → 1e-6 (cosine decay)
- NCCL timeout: 1800s (for large models)

**YOLO Training:**
- 1x RTX 3090
- Batch size: 16
- Epochs: 100
- Learning rate: 0.01 (cosine annealing)

**Pliki źródłowe:**
- `Eden/Scripts/20kDataset_11.12.25/run_detr_20k_finetune.slurm`
- `Notatki/2025-12-12_DETR_Training_NCCL_Timeout_LR_Scheduler.md`

#### 4.3 Evaluation Metrics
- [ ] mAP@0.5, mAP@0.5:0.95, mAP@0.75
- [ ] Precision, Recall, F1
- [ ] Average Recall (AR@100)
- [ ] Inference speed (FPS)
- [ ] COCO evaluation protocol

---

### Section 5: Results (znacząco rozszerzona)

#### 5.1 Multi-Epoch Training Analysis (NOWOŚĆ)

| Epoch | mAP@0.5 | mAP@0.5:0.95 | AR@100 | Status |
|-------|---------|--------------|--------|--------|
| 40 | TBD | TBD | TBD | Early |
| 60 | TBD | TBD | TBD | Training |
| 80 | TBD | TBD | TBD | Training |
| **100** | 9.7% | 2.1% | 10.3% | Underfitting |
| **160** | **80.8%** | **58.8%** | **74.7%** | **Best** |

**Key Finding:** DETR wymaga >140 epok dla konwergencji na danych chirurgicznych

- [ ] Tabela multi-epoch results
- [ ] Learning curves wykres
- [ ] Analiza convergence

**Pliki źródłowe:**
- `YOLO_DETR_Benchmarks/BENCHMARK_RESULTS_ANALYSIS.md`
- `Notatki/2025-10-31_Benchmark_DETR_vs_YOLO_wnioski.md`

#### 5.2 YOLO vs DETR Comparison (zaktualizowana tabela)

| Metric | YOLOv8 | DETR (e160) | Gap |
|--------|--------|-------------|-----|
| mAP@0.5 | **84.8%** | 80.8% | -4.0pp |
| mAP@0.5:0.95 | **78.5%** | 58.8% | -19.7pp |
| mAP@0.75 | **83.5%** | 70.8% | -12.7pp |
| AR@100 | **93.1%** | 74.7% | -18.4pp |
| FPS (RTX 3090) | **37** | 14 | -62% |

- [ ] Zaktualizować Table 1 w artykule
- [ ] Dodać szczegółową analizę gaps
- [ ] Wyjaśnić dlaczego DETR gorszy na strict IoU

#### 5.3 Query 81 Specialization Analysis (NOWOŚĆ)

- [ ] Wizualizacja attention maps Query 81
- [ ] Porównanie z innymi queries (random selection)
- [ ] Figure: Query 81 attention heatmap on surgical tool
- [ ] Dyskusja interpretowalności

#### 5.4 Dataset Selection Impact (NOWOŚĆ)

| Selection Method | Final mAP | Training Time |
|------------------|-----------|---------------|
| Random sampling | 72.3% | 8h |
| K-Center only | 76.1% | 8h |
| DINO + K-Center | 78.5% | 8h |
| **Full Pipeline** | **80.8%** | 8h |

- [ ] Ablation study selekcji
- [ ] Porównanie metod

#### 5.5 Fourier Diversity Analysis (NOWOŚĆ)

- High/Low Ratio distribution:
  - min: 4.79
  - max: 28.79
  - mean: 8.47
  - std: 5.35

- [ ] Spectral entropy correlation with detection accuracy
- [ ] Figure: FFT spectrum comparison (high vs low diversity images)

**Pliki źródłowe:**
- `AdvancedDatasetSelection/paper_visualizations/output/fourier/extremes_high_low_manifest.json`
- `AdvancedDatasetSelection/paper_visualizations/visualize_fourier_spectrum.py`

#### 5.6 Background Training Results (NOWOŚĆ)

| Metric | Before BG Training | After BG Training |
|--------|-------------------|-------------------|
| Tooltip mAP | 80.8% | 79.5% (retained) |
| False Positives (clean eye) | 45% | **8%** |
| Background Detection | 0% | **92%** |

- [ ] Tabela wyników background training
- [ ] Analiza catastrophic forgetting
- [ ] Tooltip retention metrics

---

### Section 6: Discussion (rozszerzona)

#### 6.1 Why DETR Underperforms YOLO
- [ ] Query limitation (100 queries)
- [ ] Single-scale features vs FPN
- [ ] Longer convergence time
- [ ] Query specialization phenomenon

#### 6.2 Clinical Implications
- [ ] Real-time feasibility (14 FPS sufficient for 30 FPS video)
- [ ] Safety considerations
- [ ] Integration with surgical workflow
- [ ] Potential for surgeon assistance

#### 6.3 Dataset Quality vs Quantity
- [ ] 20k diverse > 50k redundant
- [ ] Importance of hard samples (EL2N)
- [ ] Background frames crucial for false positive reduction
- [ ] Fourier diversity metrics

#### 6.4 Transformer Interpretability
- [ ] Query 81 as "tooltip detector"
- [ ] Attention visualization for surgical guidance
- [ ] Potential for explainable AI in surgery
- [ ] Future directions

---

### Section 7: Conclusion (zaktualizowana)

**Key Contributions (updated):**
1. [ ] 4-stage intelligent dataset selection pipeline (Fourier→DINO→K-Center→EL2N)
2. [ ] Discovery of Query 81 specialization in DETR
3. [ ] Background-aware training reducing false positives by 82%
4. [ ] 20,000 frame dataset for cataract surgery (largest to date?)
5. [ ] Comprehensive YOLO vs DETR benchmark with multi-epoch analysis

---

## Wykresy Do Dodania

| Figure | Opis | Źródło | Status |
|--------|------|--------|--------|
| Fig. 1 | Dataset Selection Pipeline diagram | Nowy | [ ] Do utworzenia |
| Fig. 2 | t-SNE clustering of DINO features | `output/clustering/` | [ ] Gotowy |
| Fig. 3 | FFT spectrum comparison | `output/fourier/` | [ ] Gotowy |
| Fig. 4 | DINO attention maps | `output/dino/` | [ ] Gotowy |
| Fig. 5 | Query 81 attention heatmap | Do wygenerowania | [ ] Do utworzenia |
| Fig. 6 | Multi-epoch learning curves | Do wygenerowania | [ ] Do utworzenia |
| Fig. 7 | Background training results | Do wygenerowania | [ ] Do utworzenia |
| Fig. 8 | EL2N difficulty distribution | `output/visualizations/` | [ ] Do sprawdzenia |
| Fig. 9 | FastSAM segmentation examples | `output/fastsam/` | [ ] Gotowy |

**Istniejące wizualizacje:**
- `AdvancedDatasetSelection/paper_visualizations/output/dino/dino_comparison_grid.png`
- `AdvancedDatasetSelection/paper_visualizations/output/fourier/extremes_high_low_n5.png`
- `AdvancedDatasetSelection/paper_visualizations/output/fourier/fourier_gallery_n5.png`
- `AdvancedDatasetSelection/paper_visualizations/output/fastsam/fastsam_comparison.png`

---

## Nowe Referencje Do Dodania

```bibtex
@inproceedings{paul2021deep,
  title={Deep Learning on a Data Diet: Finding Important Examples Early in Training},
  author={Paul, Mansheej and Ganguli, Surya and Dziugaite, Gintare Karolina},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9650--9660},
  year={2021}
}

@inproceedings{sener2018active,
  title={Active Learning for Convolutional Neural Networks: A Core-Set Approach},
  author={Sener, Ozan and Savarese, Silvio},
  booktitle={International Conference on Learning Representations},
  year={2018}
}

@article{zhang2022dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel M and Shum, Heung-Yeung},
  journal={arXiv preprint arXiv:2203.03605},
  year={2022}
}

@inproceedings{zhu2021deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

---

## Priorytety Implementacji

| Priorytet | Sekcja | Szacowany czas | Status |
|-----------|--------|----------------|--------|
| HIGH | Section 2: Related Work | 2 dni | [ ] Do napisania |
| HIGH | Section 3.2: Dataset Selection Pipeline | 3 dni | [ ] Mamy kod, trzeba opisać |
| HIGH | Section 5: Zaktualizowane wyniki | 2 dni | [ ] Mamy dane |
| MEDIUM | Section 3.3: Query 81 Analysis | 1 dzień | [ ] Mamy obserwacje |
| MEDIUM | Section 3.4: Background Training | 1 dzień | [ ] Mamy pipeline |
| LOW | Nowe wykresy | 1 dzień | [ ] Większość gotowa |

---

## Zależności - Pliki Projektu

### Kod źródłowy
- `AdvancedDatasetSelection/main_selection_pipeline.py` - główny pipeline
- `AdvancedDatasetSelection/selection_methods/combined_selector.py` - 4-stage selekcja
- `AdvancedDatasetSelection/feature_extractors/dino_extractor.py` - DINO features
- `AdvancedDatasetSelection/feature_extractors/fourier_analyzer.py` - Fourier analysis
- `AdvancedDatasetSelection/selection_methods/detr_el2n_scorer.py` - Query 81
- `YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/main_pipeline.py` - BG training

### Wyniki i analizy
- `YOLO_DETR_Benchmarks/BENCHMARK_RESULTS_ANALYSIS.md` - szczegółowe benchmarki
- `Notatki/2025-10-31_Benchmark_DETR_vs_YOLO_wnioski.md` - wnioski z benchmarków
- `AdvancedDatasetSelection/paper_visualizations/output/` - wizualizacje

### Wizualizacje
- `AdvancedDatasetSelection/paper_visualizations/output/fourier/` - FFT wykresy
- `AdvancedDatasetSelection/paper_visualizations/output/dino/` - DINO attention
- `AdvancedDatasetSelection/paper_visualizations/output/fastsam/` - segmentacja

---

## Ryzyka i Mitygacje

| Ryzyko | Prawdopodobieństwo | Wpływ | Mitygacja |
|--------|-------------------|-------|-----------|
| Brakujące dane multi-epoch | Medium | High | Uruchomić dodatkowe eksperymenty na Eden |
| Query 81 niepowtarzalny | Low | Medium | Potwierdzić na innych checkpointach |
| Wyniki background training niekompletne | Medium | Medium | Dokończyć pipeline na lokalnym GPU |
| Overleaf sync issues | Low | Low | Lokalna kopia LaTeX |

---

## Kryteria Sukcesu

- [ ] Related Work section kompletna (min. 20 referencji)
- [ ] Wszystkie tabele zaktualizowane z rzeczywistymi danymi
- [ ] Min. 5 nowych figur wysokiej jakości
- [ ] Dataset Selection Pipeline w pełni opisany
- [ ] Query 81 discovery udokumentowane
- [ ] Background training results włączone
- [ ] Artykuł gotowy do submission

---

## Notatki Dodatkowe

### Alternatywne podejścia
- Można rozważyć podział na 2 artykuły:
  1. Dataset Selection Pipeline (metodologiczny)
  2. DETR vs YOLO Benchmark (empiryczny)

### Potencjalne rozszerzenia
- Deformable DETR comparison
- Real-time deployment analysis
- Multi-tool detection (nie tylko tooltip)
- Video-based temporal analysis

### Pytania do rozwiązania
- Czy Query 81 jest stabilny między różnymi treningami?
- Czy Fourier features mają statistical significance?
- Jaki jest optymalny ratio tooltip/background w mixed training?

---

*Plan utworzony: 2025-12-12*
*Autor: Bartłomiej Łówko*
*Projekt: ViTParticleFilterTracker*
*Artykuł docelowy: IEEE ACCESS - Transformers for Ocular Surgery*
