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
