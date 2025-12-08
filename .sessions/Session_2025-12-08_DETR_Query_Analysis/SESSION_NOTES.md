# Sesja 2025-12-08: Analiza DETR Query i Generalizacji YOLO vs DETR

## Cel Sesji
Analiza która query w DETR odpowiada za detekcję tooltips oraz porównanie generalizacji YOLO vs DETR.

---

## Wykonane Zadania

### 1. Utworzono skrypt wizualizacji query DETR
**Plik:** `scripts/visualize_detr_queries_multi_epoch.py`

Funkcjonalność:
- Unikalne kolory dla każdego z 100 queries DETR
- Śledzenie które query wykrywa które obiekty
- Porównania między epokami (100, 120, 140, 160, 170)
- Statystyki użycia queries
- Heatmapy aktywności

**Wynik kluczowy:**
- Query 81 odpowiada za ~50% wszystkich detekcji
- Tylko 3-5 queries jest aktywnych z 100 dostępnych
- Q81, Q94, Q61 to dominujące queries

### 2. Utworzono benchmark YOLO vs DETR Q81
**Plik:** `scripts/benchmark_yolo_vs_detr_q81_multi_epoch.py`

Funkcjonalność:
- DETR używa TYLKO Query 81
- Konfigurowalne epoki dla YOLO i DETR
- Konfigurowalne thresholdy (standardowy 30%, wysoki 85%)
- Wizualizacje dla obu thresholdów
- Obliczanie TP/FP/FN/Precision/Recall

### 3. Przeprowadzono benchmark generalizacji
**Folder wyników:** `Benchmarks/BENCHMARK_Q81_20251208_044457/`

Testowane modele:
- YOLO: epoch 100, 120, 170
- DETR Q81: epoch 170

---

## Kluczowe Odkrycia

### YOLO - Silny Overfitting
| Epoch | TP (train) | TP (valid) | Wniosek |
|-------|------------|------------|---------|
| 100 | 101 | 1 | Słaba generalizacja |
| 120 | 65 | 1 | Degradacja |
| 170 | 29 | 0 | Całkowity overfitting |

### DETR Q81 - Lepsza Generalizacja
| Epoch | TP (train) | TP (valid) | Wniosek |
|-------|------------|------------|---------|
| 170 | 316 | 20 | Zachowuje zdolność detekcji |

### Przyczyna
- YOLO trenowany na Dataset A (jedno oko)
- Testowany na Dataset B (inne oko, ta sama operacja)
- YOLO nauczył się specyficznych cech jednego oka
- DETR (Transformer) lepiej generalizuje dzięki global attention

---

## Utworzone Pliki

```
scripts/
├── visualize_detr_queries_multi_epoch.py   # NOWY - wizualizacja queries
└── benchmark_yolo_vs_detr_q81_multi_epoch.py  # NOWY - benchmark Q81

Benchmarks/
├── DETR_QUERY_VISUALIZATION_20251208_*/    # Wyniki wizualizacji query
└── BENCHMARK_Q81_20251208_044457/          # Główny benchmark
    ├── results_summary.json
    ├── BENCHMARK_REPORT.md
    ├── WNIOSKI_ANALIZA_GENERALIZACJI.md    # Dokument z wnioskami
    └── visualizations/
```

---

## Rekomendacje na Przyszłość

1. **Używać DETR Q81** jako bazowego detektora
2. **Mixed dataset training** - trenować na wielu pacjentach
3. **Query optimization** - badać jak poprawić Q81
4. **High threshold (85%)** dla precyzyjnych detekcji

---

## Komendy do Uruchomienia

### Wizualizacja Query DETR:
```powershell
py -3.11 "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\scripts\visualize_detr_queries_multi_epoch.py"
```

### Benchmark YOLO vs DETR Q81:
```powershell
py -3.11 "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\scripts\benchmark_yolo_vs_detr_q81_multi_epoch.py"
```

---

## Status
**ZAKOŃCZONA** - Wnioski zapisane, gotowe do commit.
