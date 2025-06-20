# Jak naprawić problem DETR false positives

## Problem
DETR wykrywa obiekty tam gdzie ich nie ma, ponieważ:
1. **num_queries=100** - model MUSI przewidzieć 100 obiektów
2. **Brak klasy "background"** - model nie ma jak oznaczyć "brak obiektu"
3. **Threshold za niski** - przyjmuje słabe predykcje

## Rozwiązania

### 1. PILNE - Popraw dataset (wymagane ponowne trenowanie)

#### A. Dodaj klasę background do COCO:
```json
{
  "categories": [
    {"id": 0, "name": "background", "supercategory": "none"},
    {"id": 1, "name": "tool", "supercategory": "none"}
  ]
}
```

#### B. Dodaj "puste" obrazy do datasetu:
- Obrazy bez żadnych narzędzi
- Oznaczone jako klasa 0 (background) lub bez annotacji

### 2. Popraw konfigurację treningu

#### A. W `detr_train_optimized.py`:
```python
# Zmniejsz liczbę queries
parser.add_argument("--num_queries", type=int, default=20)  # było 100

# Dodaj weighted loss dla background
class_weights = torch.tensor([0.1, 1.0])  # [background, tool]
```

#### B. Dodaj proper loss function:
```python
def compute_loss_with_background_penalty(outputs, targets, num_classes):
    # Penalizuj false positives na background
    pass
```

### 3. Popraw post-processing

#### A. Zwiększ threshold:
```python
CONFIDENCE_THRESHOLD = 0.7  # było 0.25
```

#### B. Dodaj NMS (Non-Maximum Suppression):
```python
def apply_nms(boxes, scores, threshold=0.5):
    # Usuń overlapping boxes
    pass
```

#### C. Filtruj według klasy:
```python
# Usuń wszystkie predykcje klasy 0 (background)
valid_predictions = predictions[predictions.labels != 0]
```

## Kroki do wdrożenia

### Szybkie rozwiązanie (bez re-treningu):
1. ✅ Użyj `test_detr_fixed.py` z wysokim threshold
2. ✅ Dodaj NMS filtering
3. ✅ Filtruj według klasy

### Właściwe rozwiązanie (z re-treningiem):
1. 🔄 Popraw dataset - dodaj background class
2. 🔄 Dodaj "negative samples" (obrazy bez narzędzi)
3. 🔄 Zmniejsz num_queries do 20
4. 🔄 Dodaj class weighting
5. 🔄 Re-trenuj model

## Dlaczego YOLO nie ma tego problemu?

**YOLO**:
- Grid-based detection - przewiduje tylko gdzie są obiekty
- Ma implicit background class
- Może zwrócić 0 detections

**DETR**:
- Set prediction - MUSI zwrócić fixed number (100) obiektów
- Bez proper background class się "myled"
- Hungarian matching wymusza assignment nawet dla false positives

## Wnioski
Problem jest w **architekturze treningu**, nie w inference. DETR potrzebuje:
1. Proper background handling
2. Negative samples w datasecie
3. Balanced loss function
4. Aggressive post-processing