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
