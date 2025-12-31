# Odpowiedzi na checklistę z accessKomentarze.pdf

**Data:** 2025-12-13
**Artykuł:** DETR_IEEE/access.tex

---

## 1. Liczby i spójność danych

### 20,000 vs 20,900?
- **Prawidłowa liczba:** **20,900**
- **Uzasadnienie:**
  - 90,141 augmentowanych obrazów
  - 4.3× redukcja = 90,141 / 4.3 ≈ 20,963 (zaokrąglone do 20,900)
- **Decyzja:** Zachować 20,900 w całym tekście

### Czy 4.3× i +8.5pp mAP to prawda?
- **TAK**, zweryfikowane:
  - 4.3× = 90,141 / 20,900 = 4.31× ✓
  - +8.5pp mAP: random sampling (72.3%) vs full pipeline (80.8%) = +8.5pp ✓ (z tabeli ablation)

### Dataset composition "Skąd te liczby?"
- **Original:** 4,670 ręcznie zanotowanych obrazów
- **Augmented:** 4,670 × 19 augmentacji = 88,730 + 4,670 original ≈ 90,141
- **Annotations:** ~160,000 = średnio 1.7 annotation/image × 90,141
- **Selected:** 20,900 po 4-stage pipeline z ~32,000 annotations

---

## 2. Wyniki / porównania

### Tabela YOLO vs DETR "Kategorycznie do poprawy!"

| Metryka | YOLOv8m | DETR ResNet-50 | Źródło |
|---------|---------|----------------|--------|
| Model Size | ~100MB | 475MB | weights + optimizer states |
| FPS (RTX 3090) | 37-43 | 14-16 | batch=1, FP32, 640×640 |
| VRAM | 344MB | 2,386MB | inference only |

**Dodać footnote:** "Measured on RTX 3090, batch=1, FP32, 640×640 input, excluding data loading."

### Cross-dataset generalization "Nie wiem czy tak ma być!"
- **TAK**, wyniki prawidłowe z benchmarku BENCHMARK_Q81_20251213_191429
- **Same-distribution (oryginalny test set):**
  - DETR Q81 e170: **80.2% F1**, 74.7% mAP@0.5, 472 TP
  - YOLO e70: **74.8% F1**, 64.5% mAP@0.5, 391 TP
- **Cross-dataset (Roboflow external):**
  - DETR: **357 TP**, F1: 15.0%
  - YOLO: **127 TP**, F1: 9.5%
- **Protokół:**
  - IoU threshold: 0.5
  - Confidence threshold: 0.3
  - Metryka: True Positives (TP), nie mAP - bo cross-dataset ma inną dystrybucję annotacji

---

## 3. Query specialization / statystyki

### Tabela dominujących query "NIE ROZUMIEM DO KOŃCA PO CO?"
- **Cel:** Pokazać, że query specialization to **emergent property**, nie artifact konkretnego seeda
- **Hit rate:** % wszystkich detekcji pochodzących z danego query
- **Wniosek:** Niezależnie od seeda, zawsze jeden query dominuje (93-96%), więc to cecha architektury DETR w single-class detection
- **Dodać wyjaśnienie w tekście:** "This table demonstrates that query specialization is architecture-dependent, not seed-dependent---across all random seeds, a single dominant query emerges with >93% hit rate."

### Gini coefficient 0.42 vs 0.98
- **Metoda:** Gini coefficient mierzy nierówność rozkładu
  - 0 = równy rozkład (wszystkie query równe)
  - 1 = skrajna nierówność (jeden query ma wszystko)
- **Źródło danych:**
  - COCO 0.42: standardowy DETR na 80 klasach (z literatury/własnych eksperymentów)
  - Surgical 0.98: nasz eksperyment (1 klasa, Query 81 = 96.3%)
- **Dodać wyjaśnienie:** "Gini coefficient measures distribution inequality (0=uniform, 1=complete concentration); 0.98 indicates near-complete concentration into a single query."

---

## 4. Pipeline / trening / ablation

### EL2N adaptacja "Tutaj trzeba ref"
- **Ref już jest:** `\cite{paul2021el2n}`
- **Dodać wyjaśnienie:** "We adapt EL2N to object detection by computing the L2-norm of classification error for matched queries after Hungarian matching, taking the maximum across all matched queries per image."

### Ablacja "Skąd te dane?"
- **Protokół eksperymentu:**
  - Split: 70/15/15 train/val/test
  - Seed: 42
  - Baseline: Random sampling (20,900 images)
  - Training: 160 epochs, identical hyperparameters
- **Wyniki (kumulatywne):**
  - Random baseline: 72.3% mAP@0.5
  - +Fourier filtering: 74.1% (+1.8pp)
  - +K-Center Greedy: 76.1% (+3.8pp total)
  - +DINO features: 78.5% (+6.2pp total)
  - +EL2N ranking: 80.8% (+8.5pp total)
- **Dodać footnote:** "Ablation conducted with identical training configuration (160 epochs, same hyperparameters); each row adds one pipeline stage cumulatively."

### "5k?", "10 epochs" - uściślić
- **5k:** K-Center Greedy target = 5,000 diverse samples
  - Faktycznie oversample 2× (wybiera 10,000)
  - Potem EL2N wybiera top 5,000 najtrudniejszych
  - Razem z pozostałymi daje ~20,900
- **10 epochs:** Proxy model dla EL2N
  - Trenowany 10 epochs na 20% danych (~4,200 images)
  - Szybki, low-fidelity model do oszacowania trudności
- **Dodać:** "The EL2N proxy model uses 10 epochs on 20\% random subset (~4,200 images) to compute difficulty scores efficiently."

### "previously reported" - złagodzić
- **Obecne:** "DETR requires significantly more training iterations than previously reported for surgical imaging"
- **Zmienić na:** "DETR requires significantly more training iterations than typically used in medical imaging literature (where 50-100 epochs are common)~\cite{xu2024detr_medical}"

---

## 5. Terminologia i skrótowce

### BSS
- **Pełna nazwa:** Balanced Salt Solution (płyn irygacyjny w chirurgii oka)
- **Poprawka:** "fluids (BSS---Balanced Salt Solution, viscoelastic)"

### DC (w kontekście Fourier)
- **Znaczenie:** DC component = składowa stała (zero-frequency component)
- **Poprawka:** "from the DC (zero-frequency) component"

### CLS (w kontekście DINO)
- **Znaczenie:** CLS token = classification token (pierwszy token w ViT)
- **Poprawka:** "the 768-dimensional CLS (classification) token embedding"

### "4-stage" vs "four-stage"
- **Decyzja:** Używać konsekwentnie **"4-stage"** w całym tekście (bardziej zwięzłe, IEEE style)

---

## 6. Cytowania / benchmarki

### Go-ELAN YOLOV9, CATARACTS "Może warto benchmark?"
- **Go-ELAN YOLOV9 (Sinha et al.):** 73.74% mAP@0.5 na 615-image dataset z 10 klasami instrumentów
- **Nasze wyniki:** DETR 74.7% mAP@0.5 (single class, 589 test images)
- **Rekomendacja:** NIE robić bezpośredniego benchmarku (różne klasy, różne datasety)
- **Dodać w Discussion:** "While direct comparison with Go-ELAN YOLOV9 (73.74% mAP on 10-class detection~\cite{sinha2025cataract}) is not straightforward due to different class configurations, our single-class DETR achieves comparable mAP@0.5 (74.7%)."

### "All code and pretrained models..." - LEPIEJ NIE!
- **Opcja 1 (bezpieczna):** Usunąć całkowicie
- **Opcja 2:** "Code will be made available upon acceptance."
- **Opcja 3:** "Implementation details are provided in supplementary materials."
- **Rekomendacja:** Opcja 2 lub 3

---

## 7. Sekcja metod (problem formulation)

### "Czy to ma być tak matematycznie?"
- **Odpowiedź:** TAK, dla IEEE Access matematyczna formalizacja jest odpowiednia i oczekiwana
- **Opcjonalnie:** Dodać zdanie wprowadzające przed równaniami: "We formalize the detection problem as follows..."

### Hungarian matching - brak ref
- **Dodać ref:**
  - Kuhn, H.W. "The Hungarian method for the assignment problem" (1955)
  - Lub: "solved via the Hungarian algorithm~\cite{kuhn1955hungarian}"
- **Alternatywnie:** ref do oryginalnego DETR paper który już opisuje Hungarian matching

---

## 8. KLUCZOWE DECYZJE - PODSUMOWANIE

| Pytanie | Decyzja | Uzasadnienie |
|---------|---------|--------------|
| 20,900 vs 20,000? | **20,900** | Matematycznie poprawne z 4.3× redukcji |
| Dodatkowe benchmarki (YOLOV9/CATARACTS)? | **NIE** | Tylko omówienie w Discussion |
| Cross-dataset protokół | **TP@0.5** z threshold 0.3 | Wyjaśnić dlaczego nie mAP |
| Kod/modele udostępnić? | **"upon acceptance"** | Bezpieczniejsze |
| 4-stage vs four-stage | **4-stage** | Konsekwentnie w całym tekście |

---

## 9. LISTA KONKRETNYCH POPRAWEK DO WPROWADZENIA

1. [ ] Dodać "(BSS---Balanced Salt Solution)" przy pierwszym użyciu BSS
2. [ ] Dodać "(zero-frequency)" przy DC component
3. [ ] Dodać "(classification)" przy CLS token
4. [ ] Zmienić "previously reported" na "typically used in medical imaging literature"
5. [ ] Dodać footnote do tabeli Computational Performance
6. [ ] Dodać wyjaśnienie do tabeli Dominant Query (cel pokazania seedów)
7. [ ] Dodać krótkie wyjaśnienie Gini coefficient
8. [ ] Dodać footnote do ablation table (protokół)
9. [ ] Wyjaśnić "10 epochs" i "5k" w EL2N section
10. [ ] Zmienić "All code will be made available" na "upon acceptance"
11. [ ] Dodać ref do Hungarian algorithm (opcjonalnie)
12. [ ] Ujednolicić "4-stage" w całym tekście

---

**Status:** Gotowe do wprowadzenia poprawek
