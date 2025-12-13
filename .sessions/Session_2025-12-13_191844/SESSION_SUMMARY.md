# Session Summary: TestDatasetGenerator Pipeline

## Metadata
- **Date:** 2025-12-13
- **Time:** 17:40 - 19:18
- **Status:** Completed
- **Type:** Tools

## Objective
Stworzenie kompletnego pipeline'u do generowania test datasetu z filmów wideo operacji zaćmy, który:
1. Ekstrahuje losowe klatki z wielu filmów
2. Wykrywa duplikaty względem 90k treningowego datasetu
3. Automatycznie adnotuje narzędzia chirurgiczne (DETR)
4. Pozwala na manualny review i korektę (OpenCV)
5. Eksportuje do formatu YOLO

## Context
Użytkownik potrzebował test datasetu niezależnego od treningowego (90k obrazów). Filmy źródłowe: train03-train25, test01-test04 (27 filmów po ~20k klatek). Wymaganie: brak duplikatów z treningowym datasetem.

## Actions Taken

### 1. Utworzenie struktury projektu
- Folder: `TestDatasetGenerator/`
- Podfoldery: `output/test_frames/`, `output/reports/`, `output/yolo_annotations/`

### 2. Skrypt ekstrakcji klatek (`extract_random_frames.py`)
- Losowe próbkowanie z 27 filmów wideo
- Równomierne rozłożenie klatek między filmami
- Format nazwy: `{video}_frame_{XXXXXXX}.jpg`

### 3. Skrypt wykrywania duplikatów (`detect_duplicates.py`)
- Porównanie nazw plików (video + frame index)
- Perceptual hashing (pHash) dla wizualnego porównania
- Wykrywanie klatek czasowo bliskich treningowym

### 4. Pipeline adnotacji (3 skrypty):
- **`detr_auto_annotate.py`** - batch DETR inference (threshold 80%)
- **`review_annotations.py`** - OpenCV UI do przeglądania/korekty
- **`coco_to_yolo.py`** - konwersja COCO JSON → YOLO .txt

### 5. Dodatkowe skrypty:
- `annotate_test_frames.py` - adnotacja manualna (COCO output)
- `annotate_test_frames_yolo.py` - adnotacja manualna (YOLO output)
- `annotate_with_detr.py` - kombinacja DETR + manual
- `run_pipeline.py` - uruchamia extract + detect w sekwencji
- `config.yaml` - konfiguracja ścieżek i parametrów

### 6. Modyfikacje na życzenie użytkownika:
- Zmiana threshold DETR: 30% → 80%
- Dodanie opcji `--start` do kontynuowania review od określonego obrazu
- Analiza i wyodrębnienie przejrzanych adnotacji (589 z 1040)

## Results

### Key Findings
- **2448 klatek** wyekstrahowanych z 27 filmów
- **1772 detekcji DETR** przy threshold 80% (72.4% detection rate)
- **589 obrazów** zaadnotowanych po przeglądzie (1-1040)
- **125 obrazów** ręcznie poprawionych
- **196 obrazów** usuniętych/pominiętych

### Statistics
| Metryka | Wartość |
|---------|---------|
| Wszystkie klatki | 2448 |
| Przejrzane (1-1040) | 1040 |
| Zaakceptowane | 589 |
| Pominięte | 451 |
| Do przejrzenia (1041+) | 1010 |

### Issues Encountered
1. **Sortowanie plików** - nazwy typu `train05_frame_0001234.jpg` wymagały custom sort key
2. **DETR checkpoint loading** - użycie `strict=False` i `ignore_mismatched_sizes=True`
3. **Plik COCO zawierał wszystkie obrazy** - stworzono oddzielny plik tylko z zaadnotowanymi

## Files Generated/Modified

### Nowe pliki w `TestDatasetGenerator/`:
```
TestDatasetGenerator/
├── config.yaml
├── requirements.txt
├── run_pipeline.py
├── extract_random_frames.py
├── detect_duplicates.py
├── detr_auto_annotate.py          # Pipeline krok 1
├── review_annotations.py          # Pipeline krok 2 (+ --start option)
├── coco_to_yolo.py                # Pipeline krok 3
├── annotate_test_frames.py
├── annotate_test_frames_yolo.py
├── annotate_with_detr.py
└── output/
    ├── test_frames/               # 2448 obrazów
    ├── detr_detections_coco.json  # Auto-detekcje DETR
    ├── annotations_reviewed_coco.json     # Po review (wszystkie)
    ├── annotations_reviewed_1040_coco.json # Tylko przejrzane (589)
    └── annotations_clean_coco.json        # Wszystkie z adnotacjami (1599)
```

## Commands Used

```powershell
# Ekstrakcja klatek
py -3.11 extract_random_frames.py

# Auto-adnotacja DETR (80% threshold)
py -3.11 detr_auto_annotate.py

# Review w OpenCV
py -3.11 review_annotations.py
py -3.11 review_annotations.py --start 1041  # kontynuacja

# Eksport do YOLO
py -3.11 coco_to_yolo.py --input output/annotations_reviewed_1040_coco.json
```

## Next Steps
- [ ] Kontynuować review od obrazu 1041 (`--start 1041`)
- [ ] Eksportować 589 zaadnotowanych obrazów do YOLO
- [ ] Uruchomić wykrywanie duplikatów względem treningowego datasetu
- [ ] Opcjonalnie: obniżyć threshold DETR dla większej liczby auto-detekcji

## Pipeline Summary

```
┌─────────────────────────────┐
│  Videos (27 files)          │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  extract_random_frames.py   │ → 2448 frames
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  detect_duplicates.py       │ → Check vs 90k training
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  detr_auto_annotate.py      │ → COCO JSON (auto)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  review_annotations.py      │ → COCO JSON (verified)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  coco_to_yolo.py            │ → YOLO dataset
└─────────────────────────────┘
```
