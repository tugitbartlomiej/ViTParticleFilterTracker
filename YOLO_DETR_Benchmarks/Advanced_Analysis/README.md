# Środowisko do Zaawansowanej Analizy Modeli Detekcji

Ten folder zawiera kompletne środowisko do porównywania wydajności modeli YOLO i DETR. Zostało ono zintegrowane z istniejącą strukturą folderu `YOLO_DETR_Benchmarks`.

## Plan Działania

### Krok 1: Konfiguracja
1.  **Zainstaluj zależności**: Uruchom `pip install -r requirements.txt` w swoim wirtualnym środowisku.
2.  **Edytuj `config.yaml`**: Otwórz plik `config.yaml` i uzupełnij DOKŁADNE ścieżki do:
    *   Twojego modelu YOLOv8 (wytrenowanego przez 100 epok).
    *   Twojego modelu DETR (wytrenowanego przez 100 epok).
    *   Katalogu z obrazami testowymi.
    *   Pliku z adnotacjami w formacie COCO (`ground_truth.json`).

### Krok 2: Uruchomienie Inferencji
Uruchom skrypt `run_inference.py`, aby wygenerować pliki z predykcjami dla obu modeli.

```bash
# Generowanie predykcji dla YOLO
python run_inference.py --model_type yolo

# Generowanie predykcji dla DETR
python run_inference.py --model_type detr
```

Albo uruchom cały benchmark jednym poleceniem:

```bash
python run_benchmark.py                 # YOLO + DETR + metryki
python run_benchmark.py --only yolo     # tylko YOLO end-to-end
python run_benchmark.py --only detr     # tylko DETR end-to-end
```

### Krok 3: Obliczenie Metryk
Po wygenerowaniu plików z predykcjami, uruchom skrypt `evaluate_metrics.py`.

```bash
python evaluate_metrics.py
```

### Krok 4: Wygenerowanie Wizualizacji
Aby stworzyć wizualne porównania, uruchom `visualize_results.py`.

```bash
python visualize_results.py
```

## Co Gdzie Znaleźć?
-   **Konfiguracja**: `config.yaml`
-   **Skrypty**: `run_inference.py`, `evaluate_metrics.py`, `visualize_results.py`
-   **Wyniki**: Wszystkie wygenerowane pliki (predykcje, raporty, filmy, wykresy) znajdą się w folderze `benchmark_results`.

## Dodatkowe uwagi (ważne dla poprawności i wydajności)
-   `run_inference.py` dopasowuje `image_id` na podstawie COCO `ground_truth.json` (jeśli ścieżka jest ustawiona w `config.yaml`). Dzięki temu ewaluacja COCO używa poprawnych ID obrazów.
-   Logi wydajności (FPS, VRAM) są zapisywane automatycznie do `yolo_performance.json` i `detr_performance.json` w katalogu wyjściowym. `evaluate_metrics.py` wczyta je, jeśli są dostępne.
-   Jeśli mapowanie klas modelu na ID kategorii COCO nie jest 1:1, możesz dodać w `config.yaml` sekcję `label_map`:

```yaml
label_map:
  yolo: { 0: 1, 1: 2 }   # YOLO klasa 0 -> COCO id 1, YOLO klasa 1 -> COCO id 2, itd.
  detr: { 0: 1, 1: 2 }   # analogicznie dla DETR, jeśli wymagane
```

-   Jeśli nie ustawisz `label_map`:
    - YOLO użyje domyślnie `category_id = cls_idx + 1`.
    - DETR użyje `category_id = cls_idx` (zakłada zgodność z COCO).

### Konwersja DETR `.pth` → folder HuggingFace
Jeśli masz checkpoint `checkpoint_epoch_100.pth`, możesz go przekonwertować do formatu HuggingFace za pomocą skryptu:

```bash
python ../../YOLO_DETR_Benchmarks/models/convert_detr_checkpoint.py
```

Po konwersji wskaż w `config.yaml` ścieżkę do folderu z `config.json` i `model.safetensors` (np. `../../YOLO_DETR_Benchmarks/DETR/detr_inference_model_final`).
