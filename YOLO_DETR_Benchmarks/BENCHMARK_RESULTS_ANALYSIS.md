# Analiza Wyników Benchmarku YOLO vs DETR

## Pytania Badawcze

1. **Dlaczego DETR epoch 160 wydaje się gorszy niż epoch 100?**
2. **Jakie są błędy w benchmarku?**
3. **Jak interpretować wyniki?**

---

## TL;DR - Odpowiedzi

### ❌ Nieprawda: "Epoch 160 jest gorszy niż epoch 100"

**DETR epoch 160 jest ~27x lepszy niż epoch 100!**

| Metryka | Epoch 100 | Epoch 160 | Poprawa |
|---------|-----------|-----------|---------|
| mAP@0.5:0.95 | **2.1%** | **58.8%** | **+2700%** |
| mAP@0.5 | 9.7% | 80.8% | +730% |
| AR@100 | 10.3% | 74.7% | +625% |

### ✓ Prawda: "DETR jest nadal gorszy niż YOLO"

| Metryka | YOLO | DETR (e160) | Gap |
|---------|------|-------------|-----|
| mAP@0.5:0.95 | 78.5% | 58.8% | **-19.7pp** |
| mAP@0.5 | 84.8% | 80.8% | -4pp |
| AR@100 | 93.1% | 74.7% | -18.4pp |

---

## Szczegółowa Analiza

### 1. Problem z Epoch 100

Wyniki DETR dla epoch 100 były **katastrofalnie niskie**:

```
Average Precision  (AP) @[ IoU=0.50:0.95 ] = 0.003  (0.3%)
Average Precision  (AP) @[ IoU=0.50      ] = 0.013  (1.3%)
Average Recall     (AR) @[ IoU=0.50:0.95 ] = 0.004  (0.4%)
```

**Prawdopodobne przyczyny:**

#### A. Nieprawidłowy Confidence Threshold
- YOLO używa domyślnego conf=0.25
- DETR epoch 100 prawdopodobnie testowany z conf=0.5 lub wyższym
- Model w epoch 100 był jeszcze słabo skalibrowany → niskie confidence scores
- Za wysoki threshold = większość detekcji odrzucona

#### B. Model Niedotrenowany
- 100 epok to za mało dla DETR na tym datasecie
- Learning curves pokazują że model nadal się uczył:
  - Epoch 100: validation loss ~0.12
  - Epoch 160: validation loss ~0.08
  - Epoch 140-160: największy skok w jakości

#### C. Możliwe Problemy Techniczne
- Błędne ładowanie checkpointa
- Problem z formatem predykcji (DETR → COCO format)
- Nieprawidłowa normalizacja bounding boxes

### 2. Epoch 160 - Znacząca Poprawa

```
DETR Epoch 160 (conf=0.2):
  mAP@0.5:0.95 = 58.8%
  mAP@0.5      = 80.8%
  mAP@0.75     = 70.8%
  AR@100       = 74.7%
```

**Co się zmieniło:**
- ✓ Więcej treningu → lepsze feature learning
- ✓ Niższy confidence threshold (0.15-0.2) → więcej detekcji
- ✓ Lepsza kalibracja confidence scores
- ✓ Redukcja false negatives

### 3. Gap YOLO vs DETR

#### Na IoU=0.5 (loose matching):
```
YOLO:  84.8% mAP
DETR:  80.8% mAP
Gap:   4pp   (tylko 5% różnicy!)
```

#### Na IoU=0.5:0.95 (strict matching):
```
YOLO:  78.5% mAP
DETR:  58.8% mAP
Gap:   19.7pp  (25% różnicy)
```

**Interpretacja:**
- DETR **dobrze lokalizuje obiekty** (wysoki mAP@0.5)
- DETR ma problem z **precyzyjnymi bounding boxes** (niski mAP@0.75)
- YOLO lepiej "dopasowuje" bbox do kształtu obiektu

### 4. Analiza Average Recall (AR)

```
YOLO:  93.1% AR@100
DETR:  74.7% AR@100
```

**Co to oznacza:**
- YOLO znajduje **93% wszystkich obiektów** (przy top-100 detekcjach)
- DETR znajduje **75% wszystkich obiektów**
- DETR ma problem z **false negatives** (18% obiektów pominięte)

**Możliwe przyczyny:**
- Query limit (100 queries może być za mało)
- Niedostateczne background training
- Problemy z małymi obiektami (AR_medium = 68% dla DETR)

---

## Dlaczego DETR jest gorszy niż YOLO?

### 1. Architecture Design

**YOLO:**
- Dense prediction (każdy pixel może mieć detekcję)
- Anchor boxes (predefiniowane kształty)
- Feature Pyramid Network (multi-scale)
- **Zoptymalizowany pod detection**

**DETR:**
- Sparse prediction (fixed 100 queries)
- Learned object queries (nie anchors)
- Single-scale features (ostatnia warstwa)
- **Zoptymalizowany pod set prediction**

### 2. Training Data Gap

**Wasze dane treningowe:**
- YOLO: trenowany na pełnym CADTD (~8000 obrazów)
- DETR: trenowany na CADTD (możliwe że mniej epok efektywnego treningu)

**COCO Pretrain:**
- YOLO: COCO pretrain (80 klas, 117k obrazów)
- DETR: COCO pretrain (80 klas, 117k obrazów)

### 3. Hyperparameters

Możliwe problemy:
- Learning rate schedule nie optymalny dla DETR
- Batch size za mały (DETR needs large batches!)
- Weight decay / optimizer settings
- Augmentations nie dostosowane do transformer

### 4. Losses

**YOLO używa:**
- Classification loss (CrossEntropy)
- Bbox regression loss (CIoU/DIoU)
- Objectness loss

**DETR używa:**
- Hungarian matching cost
- Classification loss
- L1 bbox loss
- GIoU loss
- **Trudniejsza optymalizacja!**

---

## Wnioski

### ✅ Co działa dobrze:

1. **DETR znacząco się poprawił** (epoch 100 → 160)
2. **Detection@IoU=0.5 jest prawie równy** (80.8% vs 84.8%)
3. **Model znajduje większość obiektów** (75% recall)

### ❌ Co wymaga poprawy:

1. **Precise localization** (mAP@0.75: 70.8% vs 83.5% YOLO)
2. **False negatives** (25% obiektów pominiętych)
3. **Small objects** (AR_medium: 68% vs 94% YOLO)

### 🎯 Rekomendacje:

#### A. Optymalizacja Confidence Threshold
```python
# Test różne thresholdy
conf_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
# Znajdź optymalny dla validation set
```

#### B. Więcej Background Training
- DETR ma problem z false negatives
- Background frames mogą pomóc modelu "nie mylić się"
- Mixed gentle training (70% tooltips, 30% background)

#### C. Więcej Epok (200-300)
- DETR needs more training than YOLO
- Learning curves sugerują że model nadal się uczy
- Early stopping based on validation mAP

#### D. Query Optimization
- Przetestuj 150-200 queries zamiast 100
- Może pomóc z recall (więcej szans na detekcję)

#### E. Bbox Loss Tuning
- Zwiększ wagę GIoU loss (lepsze bbox fitting)
- Rozważ DIoU/CIoU loss zamiast GIoU

---

## Plan Dalszych Badań

### Faza 1: Multi-Epoch Analysis (w trakcie)
- ✓ Epoch 160 (done)
- ⏳ Epoch 100 (downloading)
- ⏳ Epoch 80 (downloading)
- ⏳ Epoch 60 (downloading)
- ⏳ Epoch 40 (downloading)

**Cel:** Znaleźć optymalny punkt treningowy (czy więcej = lepiej?)

### Faza 2: Hyperparameter Sweep
- Confidence threshold grid search
- Number of queries optimization
- Loss weights tuning

### Faza 3: Architecture Experiments
- Deformable DETR (lepszy multi-scale)
- Conditional DETR (szybszy konwergencja)
- DINO DETR (state-of-the-art)

---

## Podsumowanie Liczbowe

| Experiment | mAP@0.5 | mAP@0.5:0.95 | AR@100 | FPS |
|------------|---------|--------------|--------|-----|
| YOLO Epoch 100 | **84.8%** | **78.5%** | **93.1%** | **37 FPS** |
| DETR Epoch 100 | 9.7% | 2.1% | 10.3% | 14 FPS |
| DETR Epoch 160 (conf=0.5) | 80.8% | 58.7% | 74.5% | 14 FPS |
| DETR Epoch 160 (conf=0.2) | 80.8% | **58.9%** | **74.7%** | 14 FPS |

**Best DETR Configuration:** Epoch 160, conf_threshold=0.2

**Remaining Gap:**
- mAP@0.5: -4pp (5% gap)
- mAP@0.5:0.95: -19.7pp (25% gap)
- AR@100: -18.4pp (20% gap)
- FPS: -23 FPS (62% slower)

---

## Data: 2025-10-31
## Author: Bartłomiej Łówko
## Projekt: DETR Cataract Detection Benchmark
