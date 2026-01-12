# Plan Rewizji Artykułu IEEE - DETR Surgical Tool Detection

**Data utworzenia:** 2026-01-11
**Status:** Do realizacji
**Priorytet:** Wysoki (przed submisją)

---

## 🔴 KRYTYCZNE - Major Revisions (Muszą być zrobione)

### 1. Naprawić spójność liczb w pipeline selekcji danych
- [ ] **Problem:** Stage 3 mówi o k=5000, oversampling 2k, ale finalny zbiór = 20,000
- [ ] Utworzyć **flowchart/tabelę** pokazującą:
  ```
  raw (4,670) → augment (90,141) → Fourier filter → DINO embed →
  K-Center (k=?) → oversample (?) → EL2N → final (20,000)
  ```
- [ ] Podać konkretne N na **każdym etapie**
- [ ] Wyjaśnić wprost: czy selekcja była na augmented pool czy przed augmentacją
- [ ] Ujednolicić symbole w całym artykule

### 2. Dodać sekcję "Fair Comparison Protocol" (DETR vs YOLO)
- [ ] Opisać:
  - Identyczny split train/val/test
  - Identyczny preprocessing
  - Identyczne definicje klasy i IoU threshold
- [ ] Porównać po **liczbie kroków/iteracji** (nie tylko epokach)
- [ ] Dodać info o ujednoliconej augmentacji (lub wyjaśnić różnice)
- [ ] Złagodzić wnioski: "YOLO overfits" → "w tej konfiguracji YOLO wykazuje degradację"
- [ ] Ewentualnie: dodać 2-3 seedy dla potwierdzenia

### 3. Dodać istotność statystyczną (Multi-seed)
- [ ] Uruchomić **3-5 seedów** dla kluczowych settingów:
  - DETR_20k
  - YOLO_best
  - Cross-dataset evaluation
- [ ] Raportować **mean ± std**
- [ ] Opcjonalnie: bootstrap CI po wideo

### 4. Ujednolicić metryki Cross-dataset
- [ ] **Problem:** F1 ~15% vs mAP 74-81% wygląda jak sprzeczność
- [ ] Wprowadzić spójne nazewnictwo i trzymać wszędzie:
  - `DETR_full` (train na pełnym / baseline)
  - `DETR_20k` (po selekcji)
  - `DETR_20k_ft` (fine-tune)
  - Analogicznie YOLO
- [ ] Dla external dataset raportować **te same metryki** (mAP + F1 w tym samym progu)
- [ ] Lub jasno rozdzielić: "mAP protocol" vs "operating point @conf=0.3"

### 5. Wyjaśnić ekstremalny wynik YOLO 0% na cross-dataset
- [ ] Zweryfikować konfigurację (czy nie ma błędu)
- [ ] Sprawdzić czy threshold 0.3 jest odpowiedni dla obu modeli
- [ ] Dodać analizę **dlaczego** YOLO całkowicie zawodzi
- [ ] Złagodzić język: "complete failure" → "near-zero transfer under our protocol"
- [ ] Dodać 2-3 przykłady jakościowe (visualizations)

### 6. Naprawić niespójność Query Specialization
- [ ] **Problem:** 96.3% vs 42-48% w różnych miejscach
- [ ] Zdefiniować formalnie: co liczysz jako "detection"
  - Po NMS? (tu go nie ma)
  - Po threshold?
  - Po matching z GT?
- [ ] Zrobić tabelę/wykres: **hit-rate(q) vs threshold (0.1-0.9)**
- [ ] Wtedy wszystko się spina

---

## 🟡 ŚREDNIO WAŻNE - Minor Revisions

### 7. Dodać koszt obliczeniowy pipeline selekcji
- [ ] Zmierzyć czas przetwarzania 90k obrazów przez pipeline
- [ ] Dodać tabelę:
  | Etap | Czas |
  |------|------|
  | Fourier filtering | X min |
  | DINO embedding | X min |
  | K-Center Greedy | X min |
  | EL2N scoring | X min |
  | **Total curation** | X min |
  | **Training time saved** | X hours |
- [ ] Pokazać że zysk > koszt

### 8. Uzasadnić brak RT-DETR w benchmarku
- [ ] Dodać zdanie w metodologii: celem było zbadanie fundamentów mechanizmu atencji na standardowym DETR
- [ ] Optymalizacja prędkości (RT-DETR) to krok wtórny
- [ ] Lub: dodać RT-DETR do benchmarku (jeśli czas pozwala)

### 9. Przeanalizować proporcje augmentacji w wybranych 20k
- [ ] Sprawdzić: jaka proporcja wybranych 20,000 pochodzi z:
  - Oryginalnych obrazów
  - Augmentowanych wersji
- [ ] Czy pipeline nie wybiera głównie augmentowanych wersji tych samych obrazów?
- [ ] Dodać analizę/tabelę

### 10. Dodać sekcję Ethics/Data Governance
- [ ] 3-5 zdań w sekcji Data Availability lub osobnej Ethics:
  - Zgoda na użycie danych
  - Anonimizacja
  - IRB/komisja bioetyczna (lub wyjaśnienie czemu nie dotyczy)
  - Czy dataset będzie publiczny

### 11. Uzasadnić adaptację EL2N dla detekcji
- [ ] Wyjaśnić: dlaczego **max** po matched queries, a nie suma/średnia?
- [ ] Dodać mini-ablation (nawet na małej próbce):
  | Wariant | mAP |
  |---------|-----|
  | max | X |
  | mean | X |
  | cls+bbox | X |

---

## 🟢 DROBNE - Editorial

### 12. Naprawić formatowanie
- [ ] Usunąć/poprawić placeholdery:
  - "xxxx 00, 0000"
  - "VOLUME 4, 2016"
  - "10.1109ACCESS.2024.DOI"
- [ ] Poprawić literówki w nazwiskach
- [ ] Naprawić Table 16 (ucięta kolumna R - Recall)
- [ ] Sprawdzić wszystkie tabele/figury

### 13. Poprawić język i styl
- [ ] Skrócić abstrakt do ~200 słów
- [ ] Rozbić długie zdania (>100 słów)
- [ ] Ujednolicić nazewnictwo: Q81 vs Query 81 (wybrać jedno)
- [ ] Złagodzić "marketingowy" język:
  - "winner-take-all" → "dominant query phenomenon"
  - "catastrophic" → "severe"
- [ ] Native speaker proofreading

### 14. Doprecyzować definicję zadania
- [ ] Dodać rysunek/mini-schemat pokazujący:
  - Co to "tooltip" - tip czy całe narzędzie?
  - Jak wygląda bbox w adnotacjach
- [ ] 2 zdania definicji

### 15. Wzmocnić kluczowe wyniki w Abstract/Intro
- [ ] Przenieść "40× improvement on cross-dataset" na początek abstraktu
- [ ] To jest ważniejsze niż 8.5pp na zbiorze wewnętrznym
- [ ] Dodać quantifiable impact

---

## 📋 Kolejność realizacji (sugerowana)

1. **Najpierw:** Krytyczne #1-6 (wymagają zmian w tekście i potencjalnie nowych eksperymentów)
2. **Potem:** Średnie #7-11 (wymagają dodatkowych analiz)
3. **Na końcu:** Drobne #12-15 (editorial, można zrobić równolegle)

---

## 🔬 Dodatkowe eksperymenty do rozważenia

- [ ] Multi-seed (3-5) dla głównych wyników
- [ ] Ablation: query count (10-20 vs 100)
- [ ] RT-DETR benchmark (opcjonalnie)
- [ ] Analiza proporcji augmented/original w wybranych 20k

---

## 📊 Szacowany nakład pracy

| Kategoria | Szacowany czas |
|-----------|----------------|
| Krytyczne (tekst) | 8-12h |
| Krytyczne (eksperymenty) | 4-8h GPU |
| Średnie | 4-6h |
| Drobne | 2-3h |
| **Razem** | ~20-30h |

---

*Plan utworzony na podstawie 4 niezależnych recenzji/analiz artykułu*
*Ostatnia aktualizacja: 2026-01-11*
