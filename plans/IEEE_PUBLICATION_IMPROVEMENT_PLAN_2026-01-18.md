# Plan Poprawek Publikacji IEEE ACCESS

**Data:** 2026-01-18
**Artykuł:** `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\access.tex`
**Status:** W trakcie rewizji

---

## PODSUMOWANIE STANU

### ✅ Już ukończone (z poprzednich sesji):
- Ujednolicenie konfiguracji YOLO (200 epok, batch 95)
- Poprawka rozmiaru datasetu (20,900)
- Skrócenie abstraktu do ~195 słów
- Poprawka keywords (8 haseł, alfabetycznie)
- Parametry pipeline selekcji (k=5000, EL2N proxy-model)
- Metadane IEEE Access (autorzy, afiliacja, funding)
- Checklist grafik (DPI OK)
- Bibliografia sprawdzona

### 🔴 Do zrobienia - KRYTYCZNE (P0)
### 🟡 Do zrobienia - WAŻNE (P1)
### 🟢 Do zrobienia - OPCJONALNE (P2)

---

## 🔴 P0 - KRYTYCZNE (przed submission)

### 1. Aktualizacja wyników YOLO 500 epok
**Status:** ⏳ Trening w toku (Job 1523642)
**Szacowany czas:** 2-3h po zakończeniu treningu

- [ ] Poczekać na zakończenie treningu YOLO (ep200→500)
- [ ] Uruchomić benchmark YOLO ep500 vs DETR
- [ ] Zaktualizować Table 5 (Training Configuration): epochs 200→500
- [ ] Zaktualizować Table 7 (Main Results): nowe mAP/F1 dla YOLO
- [ ] Zaktualizować Figure 5 (Training curves): dodać krzywe 200-500
- [ ] Sprawdzić czy YOLO ep500 poprawia cross-patient generalization

**Lokalizacje w artykule:**
- Linia ~360 (hyperparameters)
- Linia ~420 (results tables)
- Linia ~450 (figures)

---

### 2. Spójność liczb w pipeline selekcji
**Problem:** Artykuł mówi 5000 obrazów + oversampling 2000 = 20000. To się nie zgadza.
**Szacowany czas:** 3-4h

- [ ] Narysować schemat przepływu danych:
  ```
  90,389 (augmented) 
    → Fourier filtering (~80k)
    → DINO+K-Center (25k)
    → EL2N scoring (20,900)
  ```
- [ ] Zaktualizować Section III-B (Data Selection Pipeline)
- [ ] Dodać nową figurę: "Data Flow Diagram"
- [ ] Upewnić się że wszystkie liczby są spójne

**Lokalizacje w artykule:**
- Linia ~180-220 (Methodology - Data Selection)
- Table 1 (Dataset Composition)

---

### 3. Multi-seed experiments (minimum 3 seedy)
**Problem:** Wszystko z seed=42, recenzenci medyczni wymagają stabilności
**Szacowany czas:** 8-12h (compute) + 2h (analiza)

- [ ] Uruchomić DETR evaluation z seedami: 42, 123, 456
- [ ] Uruchomić YOLO evaluation z seedami: 42, 123, 456
- [ ] Obliczyć średnią ± std dla:
  - mAP@0.5
  - F1 score
  - Query concentration metrics
- [ ] Zaktualizować Table 7 z format: `XX.X ± Y.Y`
- [ ] Dodać zdanie w Implementation Details o multiple seeds

**Lokalizacje w artykule:**
- Linia ~350 (Implementation Details)
- Tables 7, 8, 9 (Results)

---

### 4. Fair Comparison DETR vs YOLO
**Problem:** Różne GPU, batch size, augmentacje - to nie jest fair comparison
**Szacowany czas:** 2-3h (tekst)

- [ ] Dodać nową subsekcję "Fair Comparison Protocol" w Methodology
- [ ] Wyjaśnić:
  - Ten sam data split (train/val/test)
  - Te same definicje (IoU=0.5, confidence threshold)
  - Porównanie po compute budget, nie epokach
- [ ] Alternatywnie: przyznać wprost że porównujemy "best-practice"
- [ ] Złagodzić wnioski: "w naszej konfiguracji" zamiast "YOLO jest gorsze"

**Lokalizacje w artykule:**
- Nowa sekcja ~III-D
- Discussion section

---

### 5. Naukowe uzasadnienie selekcji danych
**Status:** ✅ Przygotowane (Serena memory)
**Szacowany czas:** 1-2h (integracja)

- [ ] Dodać cytaty do Related Work:
  - Paul et al. "Deep Learning on a Data Diet" (NeurIPS 2021) - EL2N
  - "Blind Coreset Selection" (ICLR 2025)
  - Tancik et al. "Fourier Features" (NeurIPS 2020)
- [ ] Dodać uzasadnienie feature importance w Methodology
- [ ] Zaktualizować Discussion o naukowe podstawy wyboru cech

**Źródło:** `dataset_selection_scientific_justification.md` (Serena memory)

---

## 🟡 P1 - WAŻNE (wzmacnia wiarygodność)

### 6. Cross-patient generalization - protokół i metryki
**Problem:** YOLO 0% mAP wygląda podejrzanie
**Szacowany czas:** 4-5h

- [ ] Sprawdzić jeszcze raz konfigurację (czy wszystko OK)
- [ ] Dodać przykładowe obrazki: co YOLO widzi vs nie wykrywa
- [ ] Wyjaśnić DLACZEGO tak się dzieje (overfitting do features szpitala?)
- [ ] Zmienić język: "brak transferu w naszym protokole" zamiast "totalna porażka"
- [ ] Rozważyć przeniesienie do Appendix jeśli wyniki są niestabilne

**Lokalizacje w artykule:**
- Section IV-C (Cross-Patient Generalization)
- Table 11

---

### 7. Query 81 analysis - spójność liczb
**Problem:** 96.3% vs 42-48% w różnych miejscach
**Szacowany czas:** 2-3h

- [ ] Jasno zdefiniować "detekcja":
  - Po confidence threshold?
  - Po dopasowaniu do GT?
- [ ] Zrobić tabelkę hit-rate dla różnych thresholdów (0.1, 0.3, 0.5, 0.7, 0.9)
- [ ] Upewnić się że Figure i Table mają te same liczby

**Lokalizacje w artykule:**
- Section IV-B (Query Specialization)
- Table 9, Figure 6

---

### 8. Czas wykonania pipeline selekcji
**Problem:** Nie wiadomo czy zysk rekompensuje koszt selekcji
**Szacowany czas:** 1-2h

- [ ] Zmierzyć czas każdego etapu:
  - Fourier extraction: ~X min
  - DINO extraction: ~X min
  - K-Center clustering: ~X min
  - EL2N scoring: ~X min
- [ ] Pokazać bilans: "selekcja zajęła X godzin, zaoszczędziliśmy Y godzin treningu"
- [ ] Dodać do Table 6 (Computational Resources)

---

### 9. Definicja "tooltip" - obrazek
**Problem:** Nie wiadomo co dokładnie oznacza bbox
**Szacowany czas:** 1h

- [ ] Dodać Figure 1b: przykład bbox na narzędziu
- [ ] Wyjaśnić: czy to czubek czy całe narzędzie?
- [ ] Dodać do Section II (Problem Definition)

---

## 🟢 P2 - OPCJONALNE (jeśli czas pozwoli)

### 10. RT-DETR baseline
**Problem:** Wspominamy ale nie testujemy
**Szacowany czas:** 8-12h (trening) + 2h (analiza)

- [ ] Opcjonalnie: wytrenować RT-DETR na tym samym datasecie
- [ ] Alternatywnie: napisać wprost że RT-DETR to future work
- [ ] Zaktualizować Related Work + Discussion

---

### 11. Etyka i zgody
**Problem:** Brak informacji o danych medycznych
**Szacowany czas:** 30min

- [ ] Dodać 3-5 zdań:
  - Dane są anonimowe
  - IRB approval (lub wyjaśnienie że dataset publiczny)
  - Data availability statement

**Lokalizacja:** Section I lub nowa sekcja "Ethics Statement"

---

### 12. Ablacja MAX vs MEAN w EL2N
**Problem:** Używamy MAX po query bez uzasadnienia
**Szacowany czas:** 2-3h

- [ ] Uruchomić ablację: max vs mean vs weighted
- [ ] Pokazać że max działa najlepiej
- [ ] Dodać wyjaśnienie intuicji

---

### 13. Drobiazgi formatowania
**Szacowany czas:** 1h

- [ ] Usunąć placeholdery ("xxxx 00, 0000")
- [ ] Poprawić literówki w nazwiskach
- [ ] Sprawdzić czy Table 16 nie jest ucięta
- [ ] Wybrać jedno: "Q81" albo "Query 81"
- [ ] Podkreślić główny wynik (40× poprawa) na początku abstraktu

---

## HARMONOGRAM SUGEROWANY

### Tydzień 1: Krytyczne (P0)
| Dzień | Zadanie | Czas |
|-------|---------|------|
| 1 | Aktualizacja wyników YOLO 500 (po treningu) | 3h |
| 2 | Spójność liczb pipeline | 4h |
| 3-4 | Multi-seed experiments | 10h |
| 5 | Fair comparison + naukowe uzasadnienie | 4h |

### Tydzień 2: Ważne (P1)
| Dzień | Zadanie | Czas |
|-------|---------|------|
| 1 | Cross-patient analysis | 5h |
| 2 | Query 81 spójność | 3h |
| 3 | Czas pipeline + definicja tooltip | 3h |
| 4 | Review i proofreading | 4h |

### Tydzień 3: Opcjonalne (P2) + Submission
| Dzień | Zadanie | Czas |
|-------|---------|------|
| 1-2 | RT-DETR (opcjonalnie) | 12h |
| 3 | Drobiazgi + etyka | 2h |
| 4 | Final review | 4h |
| 5 | **SUBMISSION** | - |

---

## SZACOWANY CZAS CAŁKOWITY

| Priorytet | Zadania | Czas |
|-----------|---------|------|
| P0 (krytyczne) | 5 zadań | ~20-25h |
| P1 (ważne) | 4 zadania | ~10-15h |
| P2 (opcjonalne) | 4 zadania | ~15-20h |
| **TOTAL** | | **~45-60h** |

---

## ZALEŻNOŚCI

```
YOLO Training (Job 1523642)
    ↓
Aktualizacja wyników (P0.1)
    ↓
Multi-seed experiments (P0.3)
    ↓
Cross-patient re-analysis (P1.6)
```

---

## PLIKI DO MODYFIKACJI

1. **`access.tex`** - główny artykuł
2. **`figures/`** - nowe figury (data flow, tooltip example)
3. **`tables/`** - zaktualizowane tabele z ±std

---

*Plan utworzony: 2026-01-18*
*Projekt: ViTParticleFilterTracker*
*YOLO Training Job: 1523642 (PENDING)*
