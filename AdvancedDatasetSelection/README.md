# Advanced Dataset Selection Pipeline

Pipeline do inteligentnej selekcji najlepszych zdjec treningowych dla DETR fine-tuning.

**Nowa wersja (v2.0)** wykorzystuje cluster-based selection oparty na literaturze:
- **ELFS (ICLR 2025)**: Label-free coreset selection z DINO + clustering
- **CCS (ICLR 2023)**: Coverage-centric selection z EL2N jako cecha

## Szybki start

```powershell
cd F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection

# Test na 5 obrazach (cluster-based, domyslna metoda):
py -3.11 main_selection_pipeline.py --config config.yaml --target 5

# Cluster-based z wyborem najtrudniejszych probek per klaster:
py -3.11 main_selection_pipeline.py --config config.yaml --target 50 --method cluster --strategy max_el2n

# Legacy k-Center (dla porownania):
py -3.11 main_selection_pipeline.py --config config.yaml --target 50 --method kcenter

# Pelna selekcja (500 obrazow):
py -3.11 main_selection_pipeline.py --config config.yaml --target 500
```

### Parametry CLI

| Argument | Opis |
|----------|------|
| `--target N` | Liczba obrazow do wybrania |
| `--method {cluster,kcenter}` | Metoda selekcji (cluster = nowa, kcenter = legacy) |
| `--strategy {centroid,max_el2n,medoid}` | Strategia wyboru reprezentanta klastra |
| `--output DIR` | Katalog wyjsciowy |

---

## Architektura Pipeline (Cluster-Based - Nowa)

```
+---------------------------------------------------------------------------+
|                    INPUT: N obrazow (np. 248)                             |
+---------------------------------------------------------------------------+
                                 |
                                 v
+---------------------------------------------------------------------------+
|  STAGE 1: EKSTRAKCJA CECH (wszystkie rownolegle)                          |
|---------------------------------------------------------------------------+
|                                                                           |
|  +------------+  +------------+  +-----------+  +------------+            |
|  |  FOURIER   |  |   DINOv3   |  |    SAM    |  |   EL2N     |            |
|  |  (9 cech)  |  | (1024 cech)|  | (1 cecha) |  | (1 cecha)  |            |
|  +------------+  +------------+  +-----------+  +------------+            |
|        |               |              |               |                   |
|        v               v              v               v                   |
|  Low/Mid/High    CLS token      Complexity      Difficulty                |
|  H/V/D1/D2       embeddings     score           score                     |
|                                                                           |
|  KLUCZOWA ROZNICA: EL2N jest teraz CECHA, nie tylko ranking!              |
|                                                                           |
+---------------------------------------------------------------------------+
                                 |
                                 v
+---------------------------------------------------------------------------+
|  STAGE 2: KOMBINACJA CECH                                                 |
|---------------------------------------------------------------------------+
|                                                                           |
|  Laczone w jeden wektor cech per obraz:                                   |
|                                                                           |
|  [DINO (1024) | Fourier (9) | SAM (1) | EL2N (1)] = 1035 wymiarow        |
|                                                                           |
|  + Z-score normalization (StandardScaler)                                 |
|                                                                           |
+---------------------------------------------------------------------------+
                                 |
                                 v
+---------------------------------------------------------------------------+
|  STAGE 3: K-MEANS CLUSTERING                                              |
|---------------------------------------------------------------------------+
|                                                                           |
|  Grupowanie obrazow w K klastrow (K = target_size)                        |
|                                                                           |
|  Kazdy klaster = grupa podobnych obrazow w przestrzeni cech               |
|  (semantycznie, teksturowo, i pod wzgledem trudnosci!)                    |
|                                                                           |
|  N obrazow ------------> K klastrow                                       |
|                                                                           |
+---------------------------------------------------------------------------+
                                 |
                                 v
+---------------------------------------------------------------------------+
|  STAGE 4: WYBOR REPREZENTANTOW                                            |
|---------------------------------------------------------------------------+
|                                                                           |
|  Z kazdego klastra wybieramy 1 reprezentanta:                             |
|                                                                           |
|  Strategie:                                                               |
|  - 'centroid': najblizszy do srodka klastra (najbardziej typowy)         |
|  - 'max_el2n': najtrudniejszy w klastrze (active learning)               |
|  - 'medoid':   minimalizujacy dystans do pozostalych                     |
|                                                                           |
|  K klastrow ------------> K reprezentantow                                |
|                                                                           |
+---------------------------------------------------------------------------+
                                 |
                                 v
+---------------------------------------------------------------------------+
|                    OUTPUT: K wybranych obrazow                            |
|---------------------------------------------------------------------------+
|  + Gwarantowane pokrycie wszystkich klastrow (coverage)                   |
|  + Reprezentatywnosc dla calego datasetu                                  |
|  + EL2N uzyty do grupowania, nie tylko rankingu                           |
+---------------------------------------------------------------------------+
```

---

## Porownanie Metod Selekcji

| Aspekt | **Cluster-Based (Nowa)** | k-Center Greedy (Legacy) |
|--------|--------------------------|--------------------------|
| EL2N | Cecha w przestrzeni klasteringu | Tylko ranking koncowy |
| Pokrycie | Gwarantowane (1 z klastra) | Greedy, moze pominac regiony |
| Trudnosc | Moze preferowac trudne (`max_el2n`) | Trudne dopiero w rankingu |
| Zlozonosc | O(NK) | O(N^2) dla k-Center |
| Literatura | ELFS, CCS (ICLR 2023/2025) | k-Center (1985) |

### Kiedy uzyc ktorej metody?

- **Cluster-Based (`--method cluster`)**:
  - Domyslna, zalecana dla wiekszosci przypadkow
  - Lepsza gdy zalezy na coverage calego datasetu
  - Strategia `max_el2n` dla active learning

- **k-Center (`--method kcenter`)**:
  - Dla porownania z baseline
  - Gdy chcesz silniejszy nacisk na roznorodnosc Fouriera

---

## Strategie Wyboru Reprezentantow

| Strategia | Opis | Kiedy uzywac |
|-----------|------|--------------|
| `centroid` | Wybiera probke najblizej srodka klastra | Najbardziej typowe probki |
| `max_el2n` | Wybiera najtrudniejsza probke z klastra | Active learning, trenowanie na slabosciach modelu |
| `medoid` | Minimalizuje dystans do wszystkich w klastrze | Gdy klastry sa bardzo zroznicowane |

---

## Komponenty i ich cechy

| Komponent | Wymiar | Cel | Uzycie w pipeline |
|-----------|--------|-----|-------------------|
| **DINO** | 1024 | Semantyka (co jest na obrazie) | Cecha w klasteringu |
| **Fourier** | 9 | Tekstury, czestotliwosci | Cecha w klasteringu |
| **SAM** | 1 | Zlozonosc sceny | Cecha w klasteringu |
| **EL2N** | 1 | Trudnosc dla modelu | **CECHA** w klasteringu! |

**Razem:** 1035 wymiarow per obraz

---

## Cechy Fouriera (9 wymiarow)

### Pasma czestotliwosci (3 cechy):
- **Low Band Energy** (0-10% promienia): ogolne ksztalty, tlo
- **Mid Band Energy** (10-50% promienia): tekstury, krawedzie
- **High Band Energy** (50-100% promienia): drobne detale, szum

### Metryki globalne (2 cechy):
- **Spectral Entropy**: rownomiernosc rozkladu energii
- **Frequency Centroid**: srodek ciezkosci czestotliwosci

### Cechy kierunkowe (4 cechy):
- **Horizontal Energy**: energia krawedzi poziomych (0 deg)
- **Vertical Energy**: energia krawedzi pionowych (90 deg)
- **Diagonal1 Energy**: energia krawedzi ukosnych (45 deg)
- **Diagonal2 Energy**: energia krawedzi ukosnych (-45 deg)

---

## Output

```
AdvancedDatasetSelection/output/selected_dataset/
+-- images/                          # Wybrane zdjecia
|   +-- img_00000.jpg
|   +-- img_00001.jpg
|   +-- ...
+-- selection_report.json            # Statystyki selekcji
+-- visualizations/
    +-- pca_coverage.png             # Pokrycie przestrzeni cech DINO
    +-- difficulty_distribution.png  # Rozklad EL2N scores
    +-- sam_complexity.png           # Zlozonosc scen
    +-- fourier_diversity.png        # Analiza Fouriera + similarity matrix
    +-- cluster_visualization.png    # NOWE: wizualizacja klastrow
    +-- selection_summary.png        # Podsumowanie etapow
```

---

## Konfiguracja (config.yaml)

```yaml
models:
  dino:
    model_name: dinov3_vitl16
    model_path: ../External/Models/dinov3-vitl16-pretrain-lvd1689m
    device: cuda
  detr:
    checkpoint: ../Eden/Checkpoints/DETR/checkpoint_epoch_170.pth
    device: cuda
    num_labels: 1

datasets:
  new_source: E:/cataract_surgery_Instruments_detection.v1i.coco/valid
  output: ./output/selected_dataset

dino:
  feature_dim: 1024  # DINOv3 ViT-L/16
  batch_size: 8

selection:
  # Metoda selekcji: 'cluster' (nowa, zalecana) lub 'kcenter' (legacy)
  method: cluster
  # Strategia wyboru reprezentanta klastra:
  # - 'centroid': najblizszy do srodka klastra
  # - 'max_el2n': najtrudniejszy w klastrze
  # - 'medoid': minimalizujacy dystans do wszystkich
  strategy: centroid

output:
  target_size: 50
  generate_visualizations: true

processing:
  use_cache: false
  cache_dir: ./output/feature_cache
```

---

## Struktura modulow

```
AdvancedDatasetSelection/
+-- main_selection_pipeline.py   # Glowny orchestrator (obsluguje obie metody)
+-- config.yaml                  # Konfiguracja
+-- README.md                    # Dokumentacja
|
+-- feature_extractors/
|   +-- dino_extractor.py        # DINOv3 ViT-L/16 (1024-dim, HuggingFace)
|   +-- sam_extractor.py         # SAM complexity proxy
|   +-- fourier_analyzer.py      # FFT analysis (9 cech)
|
+-- selection_methods/
|   +-- cluster_selector.py      # NOWY: Cluster-based selection (ELFS/CCS)
|   +-- detr_el2n_scorer.py      # DETR-based EL2N scoring
|   +-- combined_selector.py     # Legacy: k-Center + EL2N ranking
|   +-- k_center_greedy.py       # Legacy: k-Center algorithm
|   +-- el2n_scorer.py           # Proxy EL2N (prosty CNN)
|
+-- utils/
    +-- coco_handler.py          # COCO format I/O
    +-- visualization.py         # Wykresy + cluster visualization
```

---

## Przyklad wyniku (Cluster-Based)

```
============================================================
Starting Advanced Dataset Selection Pipeline
============================================================
Loading new dataset from E:/cataract_surgery_Instruments_detection.v1i.coco/valid
Found 248 images in new dataset
Total images loaded: 248
Input: 248 images
Target: 50 images

============================================================
Starting Cluster-Based Selection Pipeline
Input: 248 images -> Target: 50 images
============================================================

[Stage 1] Extracting all features...
Extracting Fourier features: 100%|####################| 248/248
Computing SAM proxy scores: 100%|#####################| 248/248
Loading model from local path: .../dinov3-vitl16-pretrain-lvd1689m
Model loaded successfully (feature_dim=1024)
Extracting dinov3_vitl16 features: 100%|##############| 248/248
Computing EL2N scores using DETR model...
DETR model loaded (epoch 170, loss 0.1628)
Computing DETR EL2N: 100%|############################| 248/248
EL2N stats: mean=0.456, std=0.234

[Stage 2] Combining features into unified space...
Combined features shape: (248, 1035)
Applied z-score normalization

[Stage 3] Clustering into 50 clusters...
Cluster sizes: min=2, max=12, mean=4.9

[Stage 4] Selecting representatives (centroid strategy)...
Selected 50 samples from 50 clusters

============================================================
Selection Complete!
  Input: 248 images
  Output: 50 images
  Clusters: 50
  Coverage: 100.0%
  EL2N (selected): 0.489 +/- 0.198
  EL2N (all): 0.456
============================================================

Selection complete!
  Input: 248 images
  Output: 50 images
  Reduction: 20.2%
```

---

## Wizualizacje

### Cluster Visualization (NOWE)
- PCA projekcja z kolorami klastrow
- Czerwone gwiazdki = wybrani reprezentanci
- Rozklad wielkosci klastrow
- Box plot EL2N per klaster

### PCA Coverage
Pokazuje jak wybrane probki pokrywaja przestrzen cech DINO.

### Difficulty Distribution
Porownuje rozklad EL2N scores miedzy wszystkimi obrazami a wybranymi.

### Fourier Diversity
Macierz podobienstwa Fouriera (Euclidean-based).

---

## Literatura

1. **ELFS (ICLR 2025)**: "Effective Label-Free Subset Selection for Fine-tuning"
   - DINO features + clustering + difficulty scores

2. **CCS (ICLR 2023)**: "Coverage-centric Coreset Selection"
   - EL2N jako cecha do stratified sampling

3. **EL2N (NeurIPS 2021)**: "Deep Learning on a Data Diet"
   - Error L2 Norm jako miara trudnosci probki

---

## Autor

Pipeline stworzony dla projektu PhD: ViT Particle Filter Tracker
Data: 2025-12-11 (v2.0 - Cluster-Based)
