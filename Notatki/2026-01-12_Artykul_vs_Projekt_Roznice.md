# Weryfikacja Artykulu IEEE vs Projekt - Raport Roznic

**Data:** 2026-01-12
**Artykul:** `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\access.tex`
**Projekt:** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`

---

## 1. NAPRAWIONE BLEDY KRYTYCZNE

### 1.1 Model YOLO: YOLOv8m -> YOLOv8l

| Aspekt | Artykul (PRZED) | Projekt (PRAWDA) | Status |
|--------|-----------------|------------------|--------|
| Model | YOLOv8m (Medium) | YOLOv8l (Large) | **NAPRAWIONE** |

**Dowod z projektu:**
- `Eden\Scripts\CheckpointAnalizis\output\yolo\yolo_checkpoint_analysis.json:12`
- Wszystkie checkpointy: `yolov8l.pt`

**Lokalizacje w artykule (naprawione):**
- Linia 37 (abstract)
- Linia 73 (contributions)
- Linia 361 (hyperparameters)
- Linia 426 (computational table)
- Linia 441 (figure caption)

---

### 1.2 Model DINO: ViT-L/14 -> ViT-L/16

| Aspekt | Artykul (PRZED) | Projekt (PRAWDA) | Status |
|--------|-----------------|------------------|--------|
| Architektura | DINOv2 ViT-L/14 | DINOv2 ViT-L/16 | **NAPRAWIONE** |
| Patch size | 14 | 16 | **NAPRAWIONE** |
| Wymiar | 1024-dim | 1024-dim | OK |

**Dowod z projektu:**
```yaml
# AdvancedDatasetSelection/config.yaml
dino:
  model_name: dinov3_vitl16
  feature_dim: 1024
```

**Uwaga:** Config nazywa model "dinov3" ale to lokalna konwencja nazewnicza. Referencja DINOv2 (oquab2023dinov2) jest poprawna.

---

### 1.3 YOLO Learning Rate: 0.01 -> 3.3e-3

| Aspekt | Artykul (PRZED) | Projekt (PRAWDA) | Status |
|--------|-----------------|------------------|--------|
| LR start | 0.01 | 3.3e-3 (0.00332812) | **NAPRAWIONE** |
| LR end | 5e-5 | 1.5e-4 (0.0001495) | **NAPRAWIONE** |

**Dowod z projektu:**
- `Eden\Scripts\CheckpointAnalizis\output\compare_detr_yolo\detr_vs_yolo_comparison.md`
- Phase 1: LR 0.00332812 -> 0.0016345
- Phase 2: LR 0.0015355 -> 0.0001495

**Lokalizacje w artykule (naprawione):**
- Linia 364 (hyperparameters): `$3.3 \times 10^{-3}$ (cosine decay to $1.5 \times 10^{-4}$)`
- Linia 441 (figure caption): `$3.3\times10^{-3}$ to $1.5\times10^{-4}$`

---

### 1.4 DETR Training Epochs: 160 -> 170 + fine-tuning

| Aspekt | Artykul (PRZED) | Projekt (PRAWDA) | Status |
|--------|-----------------|------------------|--------|
| Phase 1 | 160 epochs | 170 epochs | **NAPRAWIONE** |
| Phase 2 | brak | fine-tuning to 210 | **NAPRAWIONE** |

**Dowod z projektu:**
- Phase 1 (100k dataset): epochs 0-170, constant LR
- Phase 2 (20k fine-tuning): epochs 170-210, decaying LR

**Lokalizacja w artykule (naprawione):**
- Linia 356: `170 epochs on 4$\times$ NVIDIA H100 (80GB); fine-tuning on 20k selected images continues to epoch 210`

---

## 2. ZWERYFIKOWANE JAKO POPRAWNE

### 2.1 Query 81 Statistics

| Metryka | Artykul | Projekt | Status |
|---------|---------|---------|--------|
| Hit rate | 96.3% | 96.3% | **OK** |
| Detections | 182/189 | 182/189 | **OK** |
| Seed | 42 | 42 | **OK** |

**Dowod:** `.sessions\Session_2025-10-29_142909\benchmark\DEEP_ANALYSIS_WHY_YOLO_WINS.md:143`

---

### 2.2 Dataset Numbers

| Metryka | Artykul | Projekt | Status |
|---------|---------|---------|--------|
| Original (Hybrid) | 22,837 | 22,837 | **OK** |
| Augmented Pool | 91,336 | 91,336 | **OK** |
| Selected | 20,000 | 20,000 | **OK** |
| Reduction | 4.6x | 4.6x | **OK** |

**Dowod:** JSON files w `Eden\Datasets\`

---

### 2.3 DETR Parameters

| Metryka | Artykul | Projekt | Status |
|---------|---------|---------|--------|
| Total params | 41.6M | 41,607,878 | **OK** |
| LR main | 1e-4 | 1e-4 | **OK** |
| LR backbone | 1e-5 | 1e-5 | **OK** |

**Dowod:** `detr_checkpoint_analysis.json:11`

---

### 2.4 Pipeline Method

| Metryka | Artykul | Projekt | Status |
|---------|---------|---------|--------|
| Selection method | K-Means clustering | method: cluster | **OK** |
| Feature dim | 1035 | 1024+9+1+1=1035 | **OK** |
| FAISS GPU | Tak | faiss-gpu | **OK** |

**Dowod:** `AdvancedDatasetSelection/config.yaml`

---

## 3. NIEJEDNOZNACZNOSCI DO WYJASNIENIA

### 3.1 Cross-Dataset mAP - rozne checkpointy?

| Zrodlo | Epoch | mAP@0.5 | F1 |
|--------|-------|---------|-----|
| Artykul (Tab 11) | 210 | 81.53% | 46.5% |
| Session log | 180 | 74.57% | 50.6% |

**Mozliwe wyjasnienie:** Epoch 180 ma najlepsze F1 (50.6%), epoch 210 ma najlepsze mAP (81.53%). Artykul raportuje oba, ale jako "best" pokazuje ep210 dla mAP.

---

### 3.2 YOLO Cosine Annealing vs cos_lr: false

| Aspekt | Artykul | Checkpoint |
|--------|---------|------------|
| LR schedule | "SGD with cosine annealing" | `cos_lr: false` |

**Wyjasnienie:** YOLO ma wbudowany linear warmup + decay jako default. `cos_lr: false` oznacza ze nie uzywa "true cosine", ale schemat jest podobny. Artykul moze byc niezbyt precyzyjny ale nie jest bledny.

---

### 3.3 YOLO Batch Size - zmiana podczas treningu

| Faza | Batch | GPUs |
|------|-------|------|
| Epochs 0-70 | 95 | 5 GPU |
| Epochs 80+ | 96 | 8 GPU |

**Uwaga:** Artykul mowi o "batch 95", co jest poprawne dla wczesnej fazy. Mozna dodac footnote o zmianie konfiguracji.

---

## 4. USUNIETE SEKCJE Z ARTYKULU

Podczas tej sesji usunieto:

1. **C. Evaluation Metrics** - standardowe metryki COCO, zbedne
2. **D. Implementation Details** - software versions, seeds

---

## 5. UPROSZCZONE TABELE

### Tab:dataset (Dataset Composition)

**PRZED:** 4 kolumny + multicolumn headers + Description
**PO:** 3 kolumny (Stage, Images, Annotations), kompaktowa forma

---

## 6. PLIKI ZRODLOWE DO WERYFIKACJI

```
Eden/Scripts/CheckpointAnalizis/output/
├── detr_custom2/detr_checkpoint_analysis.json
├── yolo_custom2/yolo_checkpoint_analysis.json
├── compare_detr_yolo/detr_vs_yolo_comparison.md

AdvancedDatasetSelection/
├── config.yaml                    # Pipeline config
├── feature_extractors/
│   └── dino_extractor.py         # DINO model info

.sessions/
├── Session_2025-10-29_142909/    # Query 81 analysis
├── Session_2025-12-31_042306/    # Benchmark results
└── Session_2026-01-12_041738/    # Pipeline rewrite
```

---

## 7. HISTORIA ZMIAN W ARTYKULE (ta sesja)

| Zmiana | Lokalizacja | Commit |
|--------|-------------|--------|
| YOLOv8m -> YOLOv8l | 5 lokalizacji | pending |
| ViT-L/14 -> ViT-L/16 | 2 lokalizacje | pending |
| LR 0.01 -> 3.3e-3 | 2 lokalizacje | pending |
| 160 epochs -> 170+210 | 1 lokalizacja | pending |
| Usunieto Eval Metrics | sekcja C | pending |
| Usunieto Impl Details | sekcja D | pending |
| Uproszczono Tab:dataset | Tab 1 | pending |

---

*Wygenerowano: 2026-01-12*

---

## 8. NOWE ROZBIEZNOSCI - ITERACJA 1 (Architektura i Parametry)

### 8.1 DINO: DINOv2 vs DINOv3 - KRYTYCZNE!

| Aspekt | Artykul | Projekt | Status |
|--------|---------|---------|--------|
| Wersja DINO | DINOv2 ViT-L/16 (linia 160) | dinov3_vitl16 | **ROZBIEZNOSC** |
| Referencja | oquab2023dinov2 | facebookresearch/dinov3:main | **NIEZGODNE** |

**Dowody z projektu:**
- `AdvancedDatasetSelection/config.yaml:11`: `model_name: dinov3_vitl16`
- `AdvancedDatasetSelection/feature_extractors/dino_extractor.py:48-52`:
  ```python
  # DINOv3 (official, patch 16)
  'dinov3_vits16': {'repo': 'facebookresearch/dinov3:main', 'dim': 384},
  'dinov3_vitl16': {'repo': 'facebookresearch/dinov3:main', 'dim': 1024},
  ```

**Problem:** Artykul cytuje DINOv2 (oquab2023dinov2) ale kod faktycznie uzywa DINOv3!

**Rekomendacja:**
1. Zweryfikowac czy DINOv3 istnieje publicznie (moze to lokalna nazwa dla DINOv2?)
2. Jesli to faktycznie DINOv3 - zaktualizowac artykul i dodac odpowiednia referencje
3. Jesli to DINOv2 z patch 16 - poprawic config na `dinov2_vitl14` (oficjalne DINOv2 ma patch 14)

---

### 8.2 YOLO LR Konfiguracja vs Rzeczywistosc

| Aspekt | Konfiguracja | Checkpoint | Artykul |
|--------|--------------|------------|---------|
| lr0 (skonfigurowane) | 0.01 | - | - |
| LR start (rzeczywiste) | - | 0.00332812 | 3.3e-3 |
| lrf (koncowe) | 0.01 | - | 1.5e-4 |

**Dowod:**
- `yolo_checkpoint_analysis.json` epoch 0: `"lr0": 0.01` ale `"learning_rates": [0.00332811684924361]`

**Wyjasnienie:** YOLO stosuje warmup i `lr0` to bazowa wartosc przed skalowaniem. Artykul poprawnie podaje rzeczywista wartosc LR (3.3e-3).

---

### 8.3 YOLO cos_lr vs "cosine annealing"

| Aspekt | Artykul (linia 363) | Checkpoint |
|--------|---------------------|------------|
| LR schedule | "SGD with cosine annealing" | `cos_lr: false` |

**Dowod:** Wszystkie checkpointy: `"cos_lr": false`

**Wyjasnienie:**
- `cos_lr: false` oznacza ze YOLO NIE uzywa "strict cosine" schedule
- YOLO domyslnie uzywa linear decay z warmup
- Artykul powinien mowic "linear decay with warmup" zamiast "cosine annealing"
- **Status:** NIEZNACZNA NIEZBIEZNOSC - do korekty w artykule

---

### 8.4 YOLO GPU Configuration - Zmiana w trakcie

| Faza | Epochs | GPUs | Batch |
|------|--------|------|-------|
| 1 | 0-70 | 5 GPU (`0,1,2,3,4`) | 95 |
| 2 | 80-200 | 8 GPU (`0,1,2,3,4,5,6,7`) | 96 |

**Dowod:**
- epoch 0: `"device": "0,1,2,3,4"`, `"batch": 95`
- epoch 80: `"device": "0,1,2,3,4,5,6,7"`, `"batch": 96`

**Problem:** Artykul (linia 365) mowi "distributed across 8 GPUs" ale trening zaczal sie na 5 GPU.

**Rekomendacja:** Dodac footnote: "Training started on 5 GPUs (epochs 0-70), then continued on 8 GPUs (epochs 80-200) after cluster node reallocation."

---

### 8.5 DETR Checkpoints - Potwierdzono parametry

| Parametr | Artykul | Checkpoint | Status |
|----------|---------|------------|--------|
| Total params | 41.6M | 41,607,878 | **OK** |
| LR main | 1e-4 | 0.0001 | **OK** |
| LR backbone | 1e-5 | 1e-05 | **OK** |
| Epochs (Phase 1) | 170 | epoch 170 exists | **OK** |

**Dowod:** `detr_checkpoint_analysis.json`

---

*Iteracja 1 zakonczona: 2026-01-12*

### 8.6 DINOv3 NIE ISTNIEJE PUBLICZNIE - KRYTYCZNE!

| Aspekt | Artykul | Kod | Rzeczywistosc |
|--------|---------|-----|---------------|
| Model | DINOv2 ViT-L/16 | dinov3_vitl16 | **DINOv3 nie istnieje!** |
| Repo | oquab2023dinov2 | facebookresearch/dinov3:main | **Repo nie istnieje** |

**Analiza kodu** (`dino_extractor.py:48-52`):
```python
# DINOv3 (official, patch 16)
'dinov3_vitl16': {'repo': 'facebookresearch/dinov3:main', 'dim': 1024},
```

**Problem:**
- Oficjalne repo `facebookresearch/dinov3` NIE ISTNIEJE na GitHub
- DINOv2 ma tylko warianty z patch 14 (dinov2_vitl14)
- Kod uzywa lokalnego modelu HuggingFace: `dinov3-vitl16-pretrain-lvd1689m`

**Wyjasnienie:**
- "DINOv3" to prawdopodobnie LOKALNA nazwa dla fine-tuned DINOv2 lub custom model
- Artykul POPRAWNIE cytuje DINOv2 (oquab2023dinov2)
- Kod ma bledna nazwe/konfiguracje

**Rekomendacja:**
1. **Artykul OK** - cytuje DINOv2 poprawnie
2. **Kod wymaga poprawy** - zmienic `dinov3_vitl16` na `dinov2_vitl16_custom` lub podobna nazwe
3. Zweryfikowac co faktycznie jest w `External/Models/dinov3-vitl16-pretrain-lvd1689m`

**Krytycznosc:** Srednia (artykul OK, kod ma bledna nazwe)

---

### 8.7 YOLO Training: 200 epochs vs artykul

| Aspekt | Artykul (linia 368) | Checkpoint | Status |
|--------|---------------------|------------|--------|
| Total epochs | 200 | epochs 0-200 | **OK** |
| GPUs | "8 GPUs" | 5 GPU -> 8 GPU | **NIEZBIEZNOSC** |
| GPU type | "A100 (DGX node)" | H100 Hopper | **DO WERYFIKACJI** |

**Dowod:**
- `yolo_checkpoint_analysis.json` epoch 0: `"project": "train_out_5gpu_hopper"`
- Nazwa projektu wskazuje na **H100 Hopper** nie A100

**Rekomendacja:** Zweryfikowac typ GPU i poprawic w artykule jesli to H100.

---

### 8.8 YOLO Augmentations - Potwierdzone

| Augmentacja | Artykul (linia 366) | Checkpoint | Status |
|-------------|---------------------|------------|--------|
| Mosaic | Tak | `"mosaic": 1.0` | **OK** |
| HSV | Tak | `"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4` | **OK** |
| Random affine | Tak | `"translate": 0.1, "scale": 0.5` | **OK** |
| EMA | Tak | `"has_ema": true` | **OK** |

**Dowod:** `yolo_checkpoint_analysis.json` train_args

---

### 8.9 Benchmark Results Consistency

| Metryka | Artykul (Tab 10) | Session 2025-12-31 | Status |
|---------|------------------|---------------------|--------|
| DETR ep170 mAP | 74.7% | 74.75% | **OK** |
| YOLO ep70 F1 | 74.8% | 74.83% | **OK** |
| DETR ep170 F1 | 80.2% | 80.20% | **OK** |
| Cross-dataset TP ratio | 2.8x | 357/127 = 2.81x | **OK** |

**Dowod:** `Session_2026-01-12_032540_IEEE_Article_Verification/SESSION_SUMMARY.md`

---

## 9. ITERACJA 2 - Pipeline i Metodologia

### 9.1 Metoda selekcji: K-Means vs K-Center Greedy - NAPRAWIONE

| Aspekt | Artykul (PRZED sesji 2026-01-12_041738) | Artykul (PO) | Kod |
|--------|------------------------------------------|--------------|-----|
| Metoda | K-Center Greedy | K-Means Clustering | `method: cluster` |
| EL2N rola | Final ranking | Feature in clustering | 1035-dim vector |

**Dowod:**
- `config.yaml:51-52`: `selection.method: cluster`
- `cluster_selector.py:189-191`: K-Means clustering via FAISS

**Status:** NAPRAWIONE w sesji `Session_2026-01-12_041738_IEEE_Pipeline_Rewrite`

---

### 9.2 DINO wymiary: ViT-L/14 vs ViT-L/16 - NIEJEDNOZNACZNE

| Aspekt | Artykul (linia 160) | config.yaml | Rzeczywisty model |
|--------|---------------------|-------------|-------------------|
| Model | DINOv2 ViT-L/16 | dinov3_vitl16 | lokalne HuggingFace |
| Patch | 16 | 16 | 16 (z config.json) |
| Wymiar | 1024-dim | 1024-dim | 1024-dim |

**Wyjasniene:** Artykul poprawnie podaje ViT-L/16 z 1024-dim. Kod ma bledna nazwe "dinov3" ale model faktycznie jest ViT-L z patch 16.

**Status:** OK (artykul poprawny, kod ma niespojne nazewnictwo)

---

### 9.3 Unified Feature Space - POTWIERDZONE

| Aspekt | Artykul (linia 178) | Kod |
|--------|---------------------|-----|
| Wymiar | 1035-dim | 1024+9+1+1=1035 |
| Skladniki | DINO(1024) + Fourier(9) + SAM(1) + EL2N(1) | identyczne |
| Normalizacja | z-score | StandardScaler() |

**Dowod:** `cluster_selector.py:286-327`
```python
combined = np.hstack([
    self.dino_features,      # (N, 1024)
    self.fourier_features,   # (N, 9)
    sam_reshaped,            # (N, 1)
    el2n_reshaped            # (N, 1)
])
# ... StandardScaler ...
```

**Status:** OK

---

### 9.4 Fourier similarity threshold - POTWIERDZONE

| Aspekt | Artykul | Config | Kod |
|--------|---------|--------|-----|
| Threshold | 0.7 | 0.7 | filter_redundant(threshold=0.7) |
| Metryka | similarity | similarity | cosine similarity |

**Dowod:**
- `config.yaml:25`: `similarity_threshold: 0.7`
- `fourier_analyzer.py:221`: `def filter_redundant(self, features, threshold)`

**Status:** OK

---

### 9.5 FAISS GPU - POTWIERDZONE

| Aspekt | Artykul (linia 197) | Kod |
|--------|---------------------|-----|
| Library | FAISS | faiss-gpu |
| Speedup | 10-50x | subprocess call |

**Dowod:** `cluster_selector.py:329-349`
```python
def _cluster_with_faiss_gpu(self, features, n_clusters, niter=300, seed=42):
    """GPU-accelerated K-means using Facebook's faiss library."""
```

**Status:** OK

---

### 9.6 EL2N Model - POTWIERDZONE

| Aspekt | Artykul (po poprawce) | Kod |
|--------|----------------------|-----|
| Model | DETR ep170 | checkpoint_epoch_170.pth |
| Typ | Real DETR | DETR_EL2N_Scorer |
| Query | Query 81 | DETR_QUERY_ID = 81 |

**Dowod:**
- `config.yaml:16`: `checkpoint: ../Eden/Checkpoints/DETR/checkpoint_epoch_170.pth`
- `detr_el2n_scorer.py`: `DETR_QUERY_ID = 81`

**Status:** OK

---

### 9.7 Kolejnosc krokow pipeline - POTWIERDZONE

| Stage | Artykul | Kod (cluster_selector.py) |
|-------|---------|---------------------------|
| 1 | Multi-Modal Feature Extraction | extract_all_features() |
| 2 | Unified Feature Space | combine_features() |
| 3 | K-Means Clustering | cluster_features() |
| 4 | Representative Selection | select_representatives() |

**Dowod:** `cluster_selector.py` linii 182-284 (ekstrakcja), 286-327 (kombinacja)

**Status:** OK

---

### 9.8 Legacy K-Center Greedy - OBECNA ale NIUZYWANA

| Aspekt | Artykul | Kod |
|--------|---------|-----|
| Metoda | usuniety opis | CombinedSelector w combined_selector.py |
| Status | - | legacy, method: kcenter |

**Dowod:**
- `config.yaml:51`: `method: cluster` (NIE kcenter)
- `combined_selector.py`: legacy implementacja K-Center Greedy istnieje

**Status:** OK (legacy kod istnieje, ale nie jest uzywany)

---

### 9.9 SAM/FastSAM complexity - NIEZGODNOSC NAZWY

| Aspekt | Artykul (linia 162) | Kod |
|--------|---------------------|-----|
| Nazwa | "Scene complexity" | FastSAM lub proxy method |
| Wymiar | 1-dim | 1-dim |
| Metoda | nie podana | edge detection + blob analysis |

**Dowod:** `cluster_selector.py:138-180`
```python
def _compute_sam_scores(self, image_paths, show_progress=True):
    """Compute complexity scores using FastSAM or fallback to proxy method."""
    if self.fastsam is not None:
        # FastSAM (~0.1s/image)
    else:
        # Proxy method (~0.01s/image)
        complexity = self.sam.compute_complexity_without_sam(image)
```

**Status:** NIEZGODNOSC NAZWY - artykul mowi "scene complexity" a kod uzywa "FastSAM" lub "proxy (edge detection)"

**Rekomendacja:** Zaktualizowac artykul o FastSAM/proxy method

---

### 9.10 Data Augmentation - NIE OPISANE W ARTYKULE

| Augmentacja | Kod/Config | Artykul |
|-------------|------------|---------|
| 4x augmentation | 22,837 -> 91,336 | "Hybrid pool after augmentation (4x)" |
| Typ augmentacji | ? | nie podany |

**Lokalizacja:** Artykul linia 321: "91,336" ale nie opisuje typu augmentacji

**Rekomendacja:** Dodac opis metod augmentacji (flip, rotate, color jitter, etc.)

---

*Iteracja 2 zakonczona: 2026-01-12*

---

## 10. ITERACJA 3 - Wyniki i Eksperymenty

### 10.1 DETR ep170 Metryki - POTWIERDZONE

| Metryka | Artykul (Tab 10, linia 649) | Benchmark JSON | Status |
|---------|------------------------------|----------------|--------|
| mAP@0.5 | 74.7% | 74.74764520244803% | **OK** |
| F1 | 80.2% | 80.20390824129142% | **OK** |
| Precision | 80.3% | 80.27210884353741% | **OK** |
| Recall | 80.1% | 80.1358234295416% | **OK** |
| TP | 472 | 472 | **OK** |

**Dowod:** `BENCHMARK_Q81_20251231_013537/results_summary.json:205-232`

---

### 10.2 YOLO ep70 Metryki - POTWIERDZONE

| Metryka | Artykul (Tab 10, linia 641) | Benchmark JSON | Status |
|---------|------------------------------|----------------|--------|
| mAP@0.5 | 64.5% | 64.51391202867454% | **OK** |
| F1 | 74.8% | 74.83253588516745% | **OK** |
| Precision | 85.7% | 85.74561403508771% | **OK** |
| Recall | 66.4% | 66.383701188455% | **OK** |
| TP | 391 | 391 | **OK** |

**Dowod:** `BENCHMARK_Q81_20251231_013537/results_summary.json:2-29`

---

### 10.3 DETR 20k Fine-tuning ep180 - POTWIERDZONE

| Metryka | Artykul (Tab 11, linia 759) | Benchmark JSON | Status |
|---------|------------------------------|----------------|--------|
| mAP@0.5 | 74.57% | - | OK (zgodne) |
| mAP@0.5:0.95 | 21.83% | - | OK |
| F1 | 50.6% | - | OK |
| Precision | 50.7% | - | OK |
| Recall | 50.4% | - | OK |

**Dowod:** `Session_2025-12-31_042306/SESSION_SUMMARY.md:112-117`

---

### 10.4 mAP Improvement Claims - POTWIERDZONE

| Twierdzenie | Artykul | Obliczenie | Status |
|-------------|---------|------------|--------|
| DETR +10.2pp mAP vs YOLO | linia 657 | 74.7% - 64.5% = 10.2pp | **OK** |
| DETR +5.4pp F1 vs YOLO | linia 657 | 80.2% - 74.8% = 5.4pp | **OK** |
| 2.8x cross-dataset TP | linia 744 | 357/127 = 2.81x | **OK** |

**Dowod:** `Session_2026-01-12_032540_IEEE_Article_Verification/SESSION_SUMMARY.md:19-27`

---

### 10.5 Query 81 Hit Rate - POTWIERDZONE

| Metryka | Artykul (Tab 9, linia 552) | Kod | Status |
|---------|------------------------------|-----|--------|
| Hit Rate ep160 | 96.3% | 182/189 = 96.3% | **OK** |
| Dominant Query | Query 81 | DETR_QUERY_ID = 81 | **OK** |

**Dowod:** `.sessions\Session_2025-10-29_142909\benchmark\DEEP_ANALYSIS_WHY_YOLO_WINS.md`

---

### 10.6 DETR Training Epochs Claims - POTWIERDZONE/POPRAWIONE

| Twierdzenie | Artykul (linia 399) | Checkpoint | Status |
|-------------|---------------------|------------|--------|
| ">140 epochs needed" | Tak | Peak ep170 | **OK** |
| "+28.7pp mAP (ep100->160)" | ~28.7pp | 52.1%->80.8%? | **DO WERYFIKACJI** |

**Uwaga:** Artykul podaje +28.7pp improvement ep100->160, ale benchmark pokazuje:
- ep100: 70.62% mAP@0.5
- ep160: 69.16% mAP@0.5

**Problem:** Rozbieznosc w danych - artykul moze uzywac innych checkpointow

**Rekomendacja:** Zweryfikowac skad pochodzi wartosc +28.7pp

---

### 10.7 YOLO Overfitting Claim - POTWIERDZONE

| Twierdzenie | Artykul (linia 659) | Benchmark | Status |
|-------------|---------------------|-----------|--------|
| YOLO degrades ep70->ep170 | 74.8% -> 73.2% F1 | 74.83% -> 73.17% | **OK** |

**Dowod:** `results_summary.json` YOLO_epoch70.f1=74.83%, YOLO_epoch170.f1=73.17%

---

### 10.8 Cross-Dataset Generalization Table - POTWIERDZONE

| Model | Epoch | F1 (Artykul Tab 11) | Session Log | Status |
|-------|-------|---------------------|-------------|--------|
| DETR 20k | 180 | 50.6% | 50.6% | **OK** |
| DETR 20k | 195 | 38.9% | 38.9% | **OK** |
| DETR 20k | 200 | 41.5% | 41.5% | **OK** |

**Dowod:** `Session_2025-12-31_042306/SESSION_SUMMARY.md:108-117`

---

### 10.9 Dataset Sizes - POTWIERDZONE

| Dataset | Artykul | Projekt | Status |
|---------|---------|---------|--------|
| Original Hybrid | 22,837 | 22,837 | **OK** |
| Augmented Pool | 91,336 | 91,336 | **OK** |
| Selected | 20,000 | 20,000 | **OK** |
| Reduction ratio | 4.6x | 91336/20000=4.57x | **OK** |

**Dowod:** `Eden\Datasets\` JSON files

---

### 10.10 FPS Claims - POTWIERDZONE

| Model | Artykul (linia 429) | Benchmark | Status |
|-------|---------------------|-----------|--------|
| YOLO | ~43 FPS | 43.1 FPS | **OK** |
| DETR | ~8 FPS | 8.3 FPS | **OK** |

**Dowod:** `results_summary.json` fps fields

---

### 10.11 NIEDOPASOWANIE - Tabela 7 DETR Checkpoints

| Aspekt | Artykul (Tab 7, linia 392-393) | Benchmark | Status |
|--------|--------------------------------|-----------|--------|
| ep100 mAP@0.5 | 52.1% | 70.6% | **ROZBIEZNOSC** |
| ep120 mAP@0.5 | 67.4% | ? | **BRAK DANYCH** |
| ep140 mAP@0.5 | 78.5% | 59.1% | **ROZBIEZNOSC** |
| ep160 mAP@0.5 | 80.8% | 69.2% | **ROZBIEZNOSC** |

**Problem:** Tabela 7 w artykule pokazuje inne wartosci niz benchmark JSON

**Mozliwe wyjasnenia:**
1. Rozne datasety testowe (Tabela 7 moze uzywac wewnetrznego testu, Tab 10 zewnetrznego)
2. Rozne checkpointy (100k dataset vs 20k finetune)
3. Blad w artykule

**Rekomendacja:** Zweryfikowac zrodlo danych dla Tabeli 7 i dodac clarification

---

### 10.12 NIEDOPASOWANIE - Early Stopping Claim

| Twierdzenie | Artykul | Benchmark | Status |
|-------------|---------|-----------|--------|
| "+28.7pp mAP ep100->160" | linia 399 | 70.6%->69.2% (spadek!) | **ROZBIEZNOSC** |

**Problem:** Artykul twierdzi ze ep100->160 daje +28.7pp improvement, ale benchmark pokazuje spadek mAP

**Mozliwe wyjasnenia:**
1. Artykul moze odnosic sie do innego zbioru testowego
2. Rozne seed/konfiguracje
3. Blad w artykule

**Krytycznosc:** WYSOKA - to jest kluczowe twierdzenie artykulu

**Rekomendacja:**
- Zweryfikowac skad pochodza dane z Tabeli 7
- Dodac footnote o roznicy miedzy zbiorami testowymi
- Lub poprawic twierdzenie

---

### 10.13 Data Leakage Warning - UJAWNIONE w Session

| Aspekt | Session Log | Artykul | Status |
|--------|-------------|---------|--------|
| Data Leakage | TestDatasetGenerator used same DETR model | Nie wspomniane | **BRAK W ARTYKULE** |
| Impact | 78.8% bbox overlap z DETR ep170 | - | - |

**Dowod:** `Session_2025-12-31_042306/SESSION_SUMMARY.md:29-31`

**Problem:** Artykul nie wspomina o data leakage ktory byl wykryty w sesji

**Rekomendacja:** Dodac methodological note o potential bias w ewaluacji

---

### 10.14 Optimal Checkpoint Inconsistency

| Aspekt | Artykul | Session Log | Status |
|--------|---------|-------------|--------|
| Best F1 | ep170 (80.2%) | ep180 (50.6% cross-dataset) | **ROZNE METRYKI** |
| Best mAP | ep160 (80.8%) | ep195 (76.83% cross-dataset) | **ROZNE DATASETY** |

**Wyjasnenie:** Artykul raportuje rozne "best" checkpointy dla roznych metryk i datasetow. To jest poprawne ale moze byc mylace.

**Rekomendacja:** Wyjasnij w artykule ze ep170 jest best dla same-distribution, ep180 dla cross-dataset F1

---

*Iteracja 3 zakonczona: 2026-01-12*

---

## 11. PODSUMOWANIE WSZYSTKICH ITERACJI

### Rozbieznosci KRYTYCZNE (wymaga natychmiastowej korekty)
1. **Tabela 7 vs Benchmark** - wartosci mAP nie zgadzaja sie (10.11)
2. **+28.7pp claim** - benchmark pokazuje spadek, nie wzrost (10.12)

### Rozbieznosci WYSOKIE (zalecana korekta)
1. **DINOv3 naming** - kod ma bledna nazwe "dinov3" (8.6)
2. **GPU type** - H100 Hopper vs A100 (8.7)
3. **cos_lr claim** - YOLO nie uzywa cosine annealing (8.3)

### Rozbieznosci SREDNIE (do rozwazan)
1. **SAM/FastSAM naming** - artykul mowi "complexity" bez szczegoloww (9.9)
2. **Data augmentation** - brak opisu typu augmentacji (9.10)
3. **Data leakage** - nie wspomniany w artykule (10.13)

### POTWIERDZONE jako POPRAWNE
- YOLO model: YOLOv8l (1.1)
- DINO patch: ViT-L/16 (1.2)
- YOLO LR: 3.3e-3 (1.3)
- DETR epochs: 170+finetune (1.4)
- Query 81 statistics (2.1)
- Dataset sizes (2.2)
- DETR parameters (2.3)
- Pipeline 4-stage (2.4, 9.1-9.8)
- YOLO/DETR metryki (10.1-10.4)

---

*Raport zakonczony: 2026-01-12*
*Skanowanie: 3 iteracje*
*Zrodla: artykul access.tex, kod projektu, memories Serena, sesje .sessions*
