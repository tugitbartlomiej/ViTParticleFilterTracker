# Sesja: YOLO vs DETR — zapis i instrukcja uruchomienia

Uwaga: Ten dokument to Session Guide dla Codex (Codex CLI).

Poniżej zapis schematu sesji (reproducible runbook) oraz krótka instrukcja jak uruchomić benchmark z tego folderu.

## Konfiguracja środowiska
- Wymagania: Python 3.11, GPU opcjonalnie (zalecane), pip.
- Wejście do katalogu:
  ```bash
  cd YOLO_DETR_Benchmarks/Advanced_Analysis
  ```
- Wirtualne środowisko i zależności:
  - Linux/WSL:
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
  - Windows (PowerShell):
    ```powershell
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

## Edycja config.yaml (obowiązkowe)
Ustaw ścieżki zgodnie z Twoim środowiskiem:
- `models.yolo.path`: np. `../models/YOLO/yolo_inference_model_final/yolo_inference_model.pt`
- `models.detr.path`: folder HuggingFace z `config.json` i `model.safetensors` (np. `../DETR/detr_inference_model_final`)
- `dataset.images_dir`: katalog obrazów testowych (np. `../Datasets/Yolo/images/val`)
- `dataset.annotations_path`: COCO GT JSON zgodny z powyższymi obrazami
- `label_map`: zwykle `{ yolo: {0:0}, detr: {0:0} }`

Jeśli masz tylko checkpoint DETR `.pth`, skonwertuj:
```bash
python ../../YOLO_DETR_Benchmarks/models/convert_detr_checkpoint.py
```

## Szybki start (end-to-end)
```bash
python run_benchmark.py                 # YOLO + DETR + metryki
# tryby selektywne
python run_benchmark.py --only yolo
python run_benchmark.py --only detr
```

## Krok po kroku
```bash
python run_inference.py --model_type yolo   # zapisze yolo_predictions.json + yolo_performance.json
python run_inference.py --model_type detr   # zapisze detr_predictions.json + detr_performance.json
python evaluate_metrics.py                  # stworzy benchmark_report.json
python visualize_results.py                 # wideo + wykresy porównawcze
```

## Artefakty wyjściowe
W `benchmark_results/` znajdziesz: `yolo_predictions.json`, `detr_predictions.json`, `*_performance.json`, `benchmark_report.json`, `comparison_video.mp4`, `map_comparison.png`, `temporal_iou_comparison.png`, `fps_comparison.png`.

## Wskazówki i debug
- Spójność `image_id`: nazwy plików w COCO `images.file_name` muszą odpowiadać plikom w `images_dir`.
- Szybki smoke test: wskaż w `config.yaml` mały katalog z ~10 obrazami.
- Jeśli mAP dziwnie niski: sprawdź `label_map` oraz progi `confidence_threshold`/`iou_threshold` w `inference_params`.

## Przykładowy log sesji (do powtórzenia)
```bash
# (1) aktywacja venv i instalacja zależności
source .venv/bin/activate && pip install -r requirements.txt
# (2) edycja Advanced_Analysis/config.yaml (ścieżki modeli i danych)
# (3) pełny benchmark
python run_benchmark.py
# (4) artefakty
ls benchmark_results
```
