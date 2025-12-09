# 📋 BENCHMARK RULES - Zasady Benchmarkowania YOLO vs DETR

**Utworzono:** 2025-10-31
**Autor:** Claude Code
**Status:** OBOWIĄZUJĄCE

---

## 🎯 CELE BENCHMARKU

Benchmarki YOLO vs DETR mają na celu:
1. Obiektywne porównanie wydajności obu modeli
2. Identyfikację mocnych i słabych stron każdego modelu
3. Diagnozowanie problemów (overfit, query specialization, confidence calibration)
4. Dostarczenie **wizualnych dowodów** sukcesu/porażki modeli
5. Generowanie **szczegółowych raportów MD** z analizą

---

## 📂 STRUKTURA FOLDERÓW

```
YOLO_DETR_Benchmarks/
├── Benchmarks/                          ← GŁÓWNY FOLDER Z SESJAMI
│   └── {DATA}_{GODZINA}_{NAZWA_BENCHMARKU}/
│       ├── BENCHMARK_REPORT.md          ← KOMPLETNY RAPORT (WYMAGANY!)
│       ├── summary.json                 ← Surowe metryki
│       ├── yolo_baseline/
│       │   ├── yolo_predictions.json
│       │   └── yolo_performance.json
│       ├── detr_*/
│       │   ├── detr_predictions.json
│       │   └── detr_performance.json
│       └── visualizations/              ← WIZUALIZACJE (WYMAGANE!)
│           ├── yolo/                    ← Obrazy z detekcjami YOLO
│           ├── detr/                    ← Obrazy z detekcjami DETR
│           ├── comparison/              ← Side-by-side porównania
│           ├── infographics/            ← Wykresy i infografiki
│           │   ├── metrics_comparison.png
│           │   ├── detection_counts.png
│           │   └── epoch_progression.png (dla multi-epoch)
│           └── success_failure_analysis.json
│
├── Scripts/
│   ├── generate_benchmark_report.py    ← Generator raportów
│   ├── multi_epoch_query_benchmark.py  ← Multi-epoch benchmark
│   └── [inne utility scripts]
│
└── Advanced_Analysis/
    └── [istniejące skrypty]
```

---

## ✅ OBOWIĄZKOWE ELEMENTY BENCHMARKU

### 1. **BENCHMARK_REPORT.md** (WYMAGANY!)

Każdy benchmark **MUSI** zawierać raport MD z następującymi sekcjami:

#### a) **Executive Summary**
- Data i godzina benchmarku
- Werdykt (kto wygrał)
- Tabela z głównymi metrykami
- Osadzone infografiki

#### b) **Detailed Analysis**
- **"Why YOLO is Better"** - konkretne powody z liczbami
  - Przykłady success cases (3-5 obrazów)
  - Metryki pokazujące przewagę

- **"Why DETR Has Advantages"** - obszary gdzie DETR jest lepszy
  - Przykłady success cases
  - Metryki pokazujące przewagę

#### c) **Configuration Analysis**
- Dla DETR: porównanie różnych threshold/konfiguracji
- Dla multi-epoch: porównanie wszystkich epok
- Tabele z wynikami

#### d) **Visual Comparisons**
- **MINIMUM 10 obrazów** side-by-side comparison
- Osadzone w raporcie MD
- Pokazujące gdzie każdy model lepszy/gorszy

#### e) **Conclusions**
- Jasne wnioski: kto wygrał i dlaczego
- Rekomendacje dla produkcji
- Rekomendacje dla dalszego research

### 2. **INFOGRAFIKI** (WYMAGANE!)

Każdy benchmark musi zawierać:

#### a) **metrics_comparison.png**
- 6+ wykresów słupkowych
- Porównanie mAP@0.5, mAP@0.5:0.95, AR@100, AR medium, etc.
- Annotations: "YOLO wins by X%" / "DETR wins by X%"
- Kolorowe (zielony=YOLO, czerwony=DETR)

#### b) **detection_counts.png**
- Histogram liczby detekcji per obraz
- Boxplot porównawczy
- Mean/median annotations

#### c) **epoch_progression.png** (dla multi-epoch)
- 4 subplots:
  - mAP progression
  - AR progression
  - Query 81 dominance
  - Query diversity (active queries)

### 3. **OBRAZY Z DETEKCJAMI** (WYMAGANE!)

#### a) **comparison/** - Side-by-side porównania
- MINIMUM 20 obrazów
- Lewy panel: YOLO (zielone bbox)
- Prawy panel: DETR (czerwone bbox)
- Labels z confidence scores
- Nagłówki "YOLO" i "DETR"

#### b) **yolo/** i **detr/** - Pojedyncze obrazy
- Minimum 20 obrazów każdy
- Bounding boxy z confidence scores
- Labels z nazwami klas

### 4. **success_failure_analysis.json**

```json
{
  "yolo_success_cases": [
    {
      "image": "frame_001.jpg",
      "yolo_detections": 5,
      "detr_detections": 2,
      "difference": 3
    }
  ],
  "detr_success_cases": [...]
}
```

---

## 🔍 TYPY BENCHMARKÓW

### 1. **Single Epoch Benchmark**
- Porównanie jednej epoki DETR vs YOLO
- Testowanie różnych confidence thresholds
- Focus: optymalizacja hyperparametrów

### 2. **Multi-Epoch Benchmark**
- Porównanie epoch 100, 120, 140, 160
- Analiza Query 81 dominacji per epoka
- Diagnoza overfittingu
- Focus: wybór najlepszej epoki

### 3. **Query Analysis Benchmark**
- Porównanie Query 81 vs All Queries
- Analiza query diversity
- Testowanie ensemble strategii
- Focus: zrozumienie jak DETR używa queries

### 4. **Configuration Sweep**
- Test wielu konfiguracji (TTA, box refinement, etc.)
- Grid search po hyperparametrach
- Focus: znalezienie optimal config

---

## 📊 METRYKI DO RAPORTOWANIA

### Obowiązkowe metryki:
1. **mAP@0.5** - główna metryka sukcesu
2. **mAP@0.5:0.95** - strict localization
3. **mAP@0.75** - tight boxes
4. **AR@100** - recall (ile obiektów znajdzie)
5. **AR medium** - KRYTYCZNE dla surgical tools!
6. **AR large** - dla dużych narzędzi
7. **FPS** - szybkość
8. **VRAM** - użycie pamięci

### Dodatkowe dla multi-epoch:
9. **Query 81 usage %** - czy model używa tylko jednego query
10. **Active queries** - ile z 100 queries jest używanych
11. **Query diversity** - czy model ma ensemble benefit

---

## 🎨 STANDARDY WIZUALIZACJI

### Kolory:
- **YOLO:** Zielony (#00ff00 / lime)
- **DETR:** Czerwony (#ff0000 / red)
- **GT (Ground Truth):** Niebieski (#0000ff / blue)

### Fonty:
- Tytuły: 16-20pt, bold
- Labels: 12-14pt
- Annotations: 10-12pt

### Jakość:
- PNG, DPI=300 minimum
- Rozmiar obrazów: 16:9 aspect ratio dla infografik
- Side-by-side: każdy panel min 800px wide

---

## 🚫 CZEGO UNIKAĆ

### ❌ NIE WOLNO:
1. **Brakujących wizualizacji** - każdy benchmark MUSI mieć obrazy
2. **Raportów bez konkretnych liczb** - zawsze podawaj gaps i %
3. **Benchmarków bez YOLO baseline** - zawsze porównuj z YOLO
4. **Ignorowania Query 81** - zawsze sprawdzaj czy DETR używa tylko jednego query
5. **Testowania tylko jednej epoki** - sprawdź kilka, może wcześniejsza lepsza
6. **Brakujących success cases** - zawsze pokaż konkretne przykłady
7. **Raportów bez konkluzji** - zawsze napisz jasny werdykt

---

## 🔄 WORKFLOW BENCHMARKINGU

### Krok 1: Przygotowanie
```bash
# Pobierz checkpointy
scp eden-cluster:~/DETR/Checkpoints/checkpoint_epoch_*.pth Eden/Checkpoints/DETR/

# Sprawdź czy YOLO checkpoint jest dostępny
ls YOLO_DETR_Benchmarks/models/YOLO/epoch100.pt
```

### Krok 2: Uruchomienie benchmarku
```bash
# Single epoch
cd YOLO_DETR_Benchmarks/Advanced_Analysis
py -3.11 simple_epoch160_test.py

# Multi-epoch
py -3.11 ../Scripts/multi_epoch_query_benchmark.py --epochs 100 120 140 160
```

### Krok 3: Generowanie raportu
```bash
cd YOLO_DETR_Benchmarks
py -3.11 Scripts/generate_benchmark_report.py \
    --benchmark-dir "Benchmarks/[folder]" \
    --images-dir "../BackgroundFinetuned/Datasets/TooltipMining/train" \
    --annotations "../BackgroundFinetuned/Datasets/TooltipMining/annotations/tool_train_annotations.json"
```

### Krok 4: Weryfikacja
Sprawdź czy istnieją:
- [ ] BENCHMARK_REPORT.md
- [ ] visualizations/infographics/metrics_comparison.png
- [ ] visualizations/infographics/detection_counts.png
- [ ] visualizations/comparison/ (min 20 obrazów)
- [ ] success_failure_analysis.json

### Krok 5: Archiwizacja
Folder benchmarku jest KOMPLETNY i gotowy do archiwizacji.

---

## 🎯 CHECKLIST PRZED ZAKOŃCZENIEM

Przed zakończeniem benchmarku SPRAWDŹ:

- [ ] BENCHMARK_REPORT.md istnieje i ma wszystkie sekcje
- [ ] Infografiki PNG są wygenerowane (metrics, detection_counts)
- [ ] Są obrazy side-by-side comparison (min 20)
- [ ] Jest analiza success/failure cases
- [ ] Wszystkie metryki są policzone (mAP, AR, FPS, VRAM)
- [ ] Jest jasny werdykt: kto wygrał i dlaczego
- [ ] Są rekomendacje dla produkcji i research
- [ ] Folder jest nazwany zgodnie z konwencją: {DATA}_{NAZWA}
- [ ] Wszystkie pliki są w odpowiednich podfolderach

---

## 📝 NAMING CONVENTIONS

### Folders:
```
YYYY-MM-DD_HH-MM-SS_{BENCHMARK_NAME}/
```

Przykłady:
- `2025-10-31_17-15-00_DETR_Epoch160_vs_YOLO_Epoch100_Confidence_Optimization`
- `2025-11-01_10-30-00_Multi_Epoch_Query_Analysis`
- `2025-11-02_14-00-00_Deformable_DETR_vs_YOLO`

### Files:
- `BENCHMARK_REPORT.md` - główny raport
- `summary.json` - surowe metryki
- `success_failure_analysis.json` - analiza przypadków
- `metrics_comparison.png` - infografika metryk
- `detection_counts.png` - infografika detekcji
- `epoch_progression.png` - (dla multi-epoch)

---

## 🔬 SPECJALNE PRZYPADKI

### Multi-Epoch Benchmark:
- Zawsze testuj minimum 4 epoki
- Sprawdź Query 81 dominację dla każdej epoki
- Zdiagnozuj overfit (czy późniejsze epoki są gorsze)
- Wygeneruj epoch_progression.png

### Query Analysis:
- Sprawdź Query 81 usage %
- Policz active queries (ile z 100 jest używanych)
- Przetestuj inference only z Query 81 vs all queries
- Porównaj ensemble benefit

### Configuration Sweep:
- Test minimum 5 konfiguracji
- Zapisz każdą jako osobny subfolder
- Wygeneruj tabelę porównawczą wszystkich config
- Pokaż optimal configuration

---

## 📚 PRZYKŁADOWE RAPORTY

Dobre przykłady benchmarków:
1. `Benchmarks/2025-10-31_17-15-00_DETR_Epoch160_vs_YOLO_Epoch100_Confidence_Optimization/`
   - Kompletny raport
   - Infografiki
   - Success/failure analysis

---

## 🚀 CIĄGŁE UDOSKONALANIE

Te zasady będą ewoluować. Po każdym benchmarku:
1. Sprawdź co można poprawić
2. Zaktualizuj BENCHMARK_RULES.md
3. Dodaj nowe best practices
4. Dokumentuj lessons learned

---

**PAMIĘTAJ:** Benchmark bez wizualizacji i raportu to **NIEPEŁNY BENCHMARK**!

Każdy benchmark musi być:
- ✅ Reproducible (dokładna dokumentacja jak uruchomić)
- ✅ Visual (obrazy pokazujące gdzie modele lepsze/gorsze)
- ✅ Analytical (głęboka analiza dlaczego model wygrał/przegrał)
- ✅ Actionable (konkretne rekomendacje co dalej robić)

---

**Koniec dokumentu**
