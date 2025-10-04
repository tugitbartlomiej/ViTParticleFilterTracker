Instrukcja uruchomienia benchmarku YOLO vs DETR (Python 3.11)

Wymagania:
- Python 3.11 (GPU opcjonalnie, zalecane dla wydajności)
- System Windows lub WSL/Linux

1) Wejdź do folderu z narzędziami analitycznymi
```
cd YOLO_DETR_Benchmarks/Advanced_Analysis
```

2) (Opcjonalnie) stwórz wirtualne środowisko i zainstaluj zależności
- Windows (PowerShell):
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- Linux/WSL:
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Skonfiguruj ścieżki w `config.yaml` (domyślna konfiguracja wskazuje na repozytoryjne ścieżki względne):
- YOLO: `../models/YOLO/yolo_inference_model_final/yolo_inference_model.pt` (możesz zmienić na `../models/YOLO/epoch100.pt`)
- DETR: `../DETR/detr_inference_model_final` (folder Hugging Face z `config.json` i `model.safetensors`)
- Zbiór testowy: `../Datasets/Yolo/images/val`
- Adnotacje COCO: `../Datasets/Detr/coco_annotations_from_yolo_dataset_20250218.json`

Uwaga: Jeśli masz tylko checkpoint DETR `.pth` (np. `models/DETR/checkpoint_epoch_100.pth`), przekonwertuj go:
```
python ../../YOLO_DETR_Benchmarks/models/convert_detr_checkpoint.py
```
Po konwersji użyj folderu wyjściowego (`../DETR/detr_inference_model_final`) w `config.yaml`.

4) Uruchom pełny benchmark (inferencja + metryki)
```
python run_benchmark.py
```
Tryby selektywne:
```
python run_benchmark.py --only yolo   # tylko YOLO
python run_benchmark.py --only detr   # tylko DETR
```
Lub ręcznie (krok po kroku):
```
python run_inference.py --model_type yolo
python run_inference.py --model_type detr
python evaluate_metrics.py
```

5) Wizualizacje (wideo porównawcze + wykresy)
```
python visualize_results.py
```

6) Wyniki
Wszystko trafia do `YOLO_DETR_Benchmarks/Advanced_Analysis/benchmark_results/`:
- `yolo_predictions.json`, `detr_predictions.json`
- `yolo_performance.json`, `detr_performance.json` (FPS, VRAM, liczba klatek)
- `benchmark_report.json` (metryki COCO + stabilność czasowa + wydajność)
- `comparison_video.mp4`, `map_comparison.png`, `temporal_iou_comparison.png`, `fps_comparison.png`

Wskazówki:
- Jeśli zmienisz `dataset.images_dir`, ewaluacja COCO zostanie ograniczona do obrazów, dla których wygenerowano predykcje.
- `label_map` w `config.yaml` wymusza zgodność klas (dla Twojego zestawu: YOLO 0 → COCO 0, DETR 0 → COCO 0).

