# CataractAI Workbench

Aplikacja desktopowa PyQt6 konsolidujaca caly pipeline ML do detekcji narzedzi chirurgicznych w operacjach zaćmy. Opakowuje istniejace skrypty CLI, pipeline'y treningowe, benchmarki i zarzadzanie klastrem HPC w jeden interfejs z 6 zakladkami.

**98 plikow Python | ~15 000 linii kodu produkcyjnego | 338 testow | Python 3.11**

---

## Spis tresci

- [Architektura](#architektura)
- [Instalacja](#instalacja)
- [Uruchamianie](#uruchamianie)
- [Zakladki](#zakladki)
  - [1. Dataset Selection](#1-dataset-selection)
  - [2. Training](#2-training)
  - [3. Benchmarks](#3-benchmarks)
  - [4. Eden HPC](#4-eden-hpc)
  - [5. Visualization](#5-visualization)
  - [6. Experiments](#6-experiments)
- [Architektura kodu](#architektura-kodu)
- [Backend Adaptery](#backend-adaptery)
- [Testy](#testy)
- [Struktura katalogow](#struktura-katalogow)

---

## Architektura

```
┌─────────────────────────────────────────────────────────────┐
│                      MainWindow (PyQt6)                     │
│  ┌──────┬──────────┬───────────┬──────┬────────┬──────────┐ │
│  │ Data │ Training │ Benchmark │ Eden │  Viz   │ Experim. │ │
│  │ Sel. │          │           │ HPC  │        │          │ │
│  └──┬───┴────┬─────┴─────┬────┴──┬───┴───┬────┴────┬─────┘ │
│     │        │           │       │       │         │        │
│  ┌──┴────────┴───────────┴───────┴───────┴─────────┴──┐     │
│  │              Signal Bus (event system)              │     │
│  └──┬────────────────────────────────────────────┬────┘     │
│     │                                            │          │
│  ┌──┴──────────────┐          ┌──────────────────┴──┐       │
│  │  Core Services  │          │  Backend Adaptery   │       │
│  │  - Checkpoint   │          │  - Training         │       │
│  │  - Parameter    │          │  - Inference (ABC)  │       │
│  │  - Dataset      │          │  - Dataset Sel.     │       │
│  │  - SSH Manager  │          │  - Benchmark        │       │
│  │  - Log Parser   │          │  - Eden             │       │
│  └─────────────────┘          │  - Metrics          │       │
│                               └─────────────────────┘       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Status Bar: GPU VRAM | Eden status | Active jobs    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │  Istniejacy kod    │
                    │  (3 pliki zmienione│
                    │   ~15 linii)       │
                    └────────────────────┘
```

---

## Instalacja

```bash
# Wymagany Python 3.11
py -3.11 -m pip install -r CataractAI_Workbench/requirements.txt
```

### Zaleznosci

| Pakiet | Wersja | Rola |
|--------|--------|------|
| PyQt6 | >= 6.5.0 | Framework GUI |
| pyqtgraph | >= 0.13.0 | Wykresy real-time (loss, LR) |
| matplotlib | >= 3.7.0 | Wykresy statyczne (publikacyjne) |
| paramiko | >= 3.3.0 | SSH do klastra Eden |
| scp | >= 0.14.0 | Transfer plikow SCP |
| psutil | >= 5.9.0 | Monitoring GPU/CPU |
| numpy | >= 1.24.0 | Obliczenia numeryczne |
| Pillow | >= 10.0.0 | Przetwarzanie obrazow |
| PyYAML | >= 6.0 | Konfiguracja YAML |

Opcjonalnie (do pelnej funkcjonalnosci):
- `torch` + `torchvision` -- trening i inferencja DETR
- `ultralytics` -- inferencja YOLO
- `pynvml` -- monitoring VRAM GPU w status barze

---

## Uruchamianie

```bash
# Jako modul
py -3.11 -m CataractAI_Workbench.app.main

# Lub po instalacji (pip install -e CataractAI_Workbench/)
cataract-workbench
```

Aplikacja uruchamia sie z dark theme (Fusion) i splash screenem.

---

## Zakladki

### 1. Dataset Selection

Interfejs do pipeline'u `AdvancedDatasetSelection` -- selekcja optymalnego podzbioru obrazow z duzego datasetu.

```
┌──[Config Panel]──┬──[Results Panel]──────────────────────────┐
│ Method: cluster   │ [Run Pipeline]  [Stop]  Progress: [====] │
│ Strategy: centroid│                                           │
│ Target: 20000     │ ┌──[Feature Viz]──────┬──[Gallery]──────┐│
│ Weights:          │ │ PCA | Fourier | SAM │ Thumbnails      ││
│  dino: 0.35       │ │ EL2N | Cluster      │ Selected: 20000 ││
│  fourier: 0.15    │ └────────────────────┴─────────────────┘│
│ [Load] [Save]     │ ┌──[Log Console]──────────────────────── ││
│                   │ │ 14:32:01 Extracting DINO features...  ││
└───────────────────┴──────────────────────────────────────────┘
```

**Funkcje:**
- Edytor konfiguracji mapujacy `config.yaml`
- 5 typow wizualizacji: PCA coverage, Fourier diversity, SAM complexity, EL2N distribution, cluster map
- Galeria wybranych obrazow z lazy loading
- Pipeline uruchamiany w QThread z progress barem

### 2. Training

Trening lokalny DETR/YOLO z wykresami na zywo i narzedziem do analizy parametrow.

```
┌──[Config]────────┬──[Training Monitor]───────────────────────┐
│ Model: DETR       │ [Start]  [Pause]  [Stop]                  │
│ Epochs: 5         │ ┌──[Live Charts (pyqtgraph)]─────────────┐│
│ Batch: 2          │ │ Loss ────v           LR ──v             ││
│ LR: 2e-6          │ │     \                    \              ││
│ AMP: [x]          │ └────────────────────────────────────────┘│
│                   │ ┌──[Checkpoints]──┬──[Log Console]───────┐│
│ [Load] [Save]     │ │ ep5_best.pth    │ Epoch 3 loss=0.108   ││
└───────────────────┴──────────────────────────────────────────┘
```

**Dwie pod-zakladki:**
- **Training** -- konfiguracja, uruchomienie, wykresy live (pyqtgraph), menedzer checkpointow
- **Training Analyzer** -- 3 narzedzia:
  - **Checkpoint Inspector** -- statystyki warstw, histogramy wag, porownanie dwoch checkpointow
  - **Parameter Advisor** -- rekomendacje LR, batch size, VRAM, early stopping na podstawie reguł
  - **Dataset Analyzer** -- analiza kompozycji COCO (rozmiary bbox, aspect ratio, heatmapa, balans kategorii)

### 3. Benchmarks

Uruchamianie skryptow benchmarkowych i przegladanie wynikow.

**Funkcje:**
- Automatyczne odkrywanie skryptow `benchmark_*.py` w `YOLO_DETR_Benchmarks/scripts/`
- Dynamiczne parsowanie `--help` na formularz parametrow
- Izolacja przez QProcess (nie blokuje GUI)
- Tabele wynikow (mAP, Precision, Recall, F1) z wykresami
- Eksport do JSON, HTML, CSV

### 4. Eden HPC

Polaczenie SSH z klastrem Eden (Paramiko, 2-hop przez jump host) i zarzadzanie SLURM.

```
┌──[Cluster Status]───┬──[Jobs]───────────────────────────────┐
│ Nodes:               │ #1524831 yolo_lr001  hopper RUNNING   │
│  dgx-1: 8xA100 [4]  │ #1523692 yolo_500ep  hopper COMPLETED │
│  hopper: 8xH100 [7] │ [Submit] [Cancel] [Refresh]           │
│                      ├──[Remote Log Viewer]──────────────────┤
│ ┌──[File Transfer]── │ Epoch 45/330  loss: 0.45  mAP: 0.965 │
│ │ Upload / Download  │ [Live] [Pause] [Clear]                │
└──────────────────────┴───────────────────────────────────────┘
```

**Funkcje:**
- Dwuetapowe SSH: `ssh.mini.pw.edu.pl` (jump) -> `eden` (klaster)
- Dashboard klastra: status wezlow, GPU, partycje
- Zarzadzanie SLURM: `squeue`, `sbatch`, `scancel`
- Streaming logow w czasie rzeczywistym (`tail -f` przez SSH)
- Transfer plikow (SFTP) z paskiem postepu
- Przegladanie zdalnych sesji Eden
- Mini terminal SSH

### 5. Visualization

Dashboard wizualizacji laczacy metryki z treningu lokalnego i Eden.

**Cztery pod-zakladki:**
- **Training Dashboard** -- ujednolicone wykresy loss/LR z wielu runow, aktualizowane na zywo przez signal bus
- **Dataset Explorer** -- przegladarka obrazow z overlayem bounding boxow COCO, filtrowanie po kategorii
- **Model Comparison** -- porownanie side-by-side predykcji dwoch modeli na tych samych obrazach
- **Inference Tester** -- wizualne testowanie modeli:
  - Ladowanie DETR lub YOLO checkpoint
  - Pojedyncza inferencja z overlayem bbox + ground truth
  - Batch test z metrykami (precision, recall, F1)
  - Tablica detekcji, macierz pomylek
  - Eksport wynikow do CSV/JSON

### 6. Experiments

Dziennik eksperymentow zintegrowany z istniejacym systemem sesji (`.sessions/`).

**Funkcje:**
- Przegladanie i filtrowanie sesji (po typie, tekscie)
- Tworzenie nowych sesji z formularzem (typ, tematy, cel)
- Edytor Markdown z renderowaniem (headings, listy, checkboxy, kod, linki)
- Kolorowe tagi tematow
- Statusy: In Progress / Completed / Failed
- Otwieranie Timeline View i Graph View (jesli istnieja pliki HTML w `.sessions/analysis/`)
- Statystyki: liczba sesji, podzial wg typow i tematow

---

## Architektura kodu

### Wzorce projektowe (SOLID)

| Wzorzec | Zastosowanie |
|---------|-------------|
| **ABC + Polimorfizm** | `ModelInference` -> `DETRInference`, `YOLOInference` |
| **Factory** | `InferenceFactory.create("detr")` |
| **Singleton** | `get_signal_bus()` -- globalny event bus |
| **Adapter** | Backend adaptery opakowujace istniejacy kod |
| **Observer** | PyQt signals przez SignalBus |
| **Composition** | Widgety skladane z mniejszych komponentow |
| **Service Layer** | `AnalysisService(ABC)` -> Checkpoint, Parameter, Dataset |
| **Template Method** | `PipelineWorker.run_task()` nadpisywany w podklasach |

### Threading

| Operacja | Model | Powod |
|----------|-------|-------|
| Feature extraction | QThread | GPU-bound, progress |
| Trening lokalny | QThread | GPU-bound, epoch callbacks |
| Benchmarki | QProcess | Izolacja, skrypty CLI |
| Komendy SSH | QThread | I/O-bound |
| Streaming logow | QThread | Ciagle I/O |
| Transfer plikow | QThread | I/O-bound, progress |
| Renderowanie GUI | Main thread | Wymaganie Qt |

### Signal Bus

Globalny event bus (`app/core/signal_bus.py`) do komunikacji miedzy komponentami:

```python
from CataractAI_Workbench.app.core.signal_bus import get_signal_bus

bus = get_signal_bus()
bus.epoch_completed.connect(my_handler)       # Training -> Viz dashboard
bus.ssh_connected.connect(update_status)      # Eden -> Status bar
bus.pipeline_finished.connect(show_results)   # Pipeline -> Results viewer
```

---

## Backend Adaptery

Adaptery opakowuja istniejacy kod projektu. Minimalne zmiany w oryginalnych plikach (3 pliki, ~15 linii):

| Adapter | Opakowuje | Zmiana w oryginale |
|---------|-----------|-------------------|
| `training_adapter.py` | `BackgroundFinetuned/main_train.py` | +`epoch_callback` param |
| `dataset_selection_adapter.py` | `AdvancedDatasetSelection/main_selection_pipeline.py` | -- |
| `inference_adapter.py` | torch/transformers + ultralytics | -- |
| `benchmark_adapter.py` | `YOLO_DETR_Benchmarks/scripts/benchmark_*.py` | -- |
| `eden_adapter.py` | `SSHManager` + `EdenClient` | -- |
| `metrics.py` | Czyste funkcje: IoU, precision, recall, F1 | -- |

Zmiany w istniejacym kodzie (backward-compatible, opcjonalne parametry):
- `AdvancedDatasetSelection/selection_methods/cluster_selector.py` -- `progress_callback=None`
- `AdvancedDatasetSelection/selection_methods/combined_selector.py` -- `progress_callback=None`
- `BackgroundFinetuned/main_train.py` -- `epoch_callback=None`

---

## Testy

```bash
# Uruchom wszystkie testy
py -3.11 -m pytest CataractAI_Workbench/tests/ -v

# Uruchom bez testow Qt (szybciej, bez GUI)
py -3.11 -m pytest CataractAI_Workbench/tests/ -v --ignore=CataractAI_Workbench/tests/test_widgets.py

# Pojedynczy modul
py -3.11 -m pytest CataractAI_Workbench/tests/test_metrics.py -v
```

### Pokrycie testami

| Modul testowy | Testow | Co testuje |
|---------------|--------|-----------|
| `test_metrics.py` | 29 | IoU, precision/recall/F1, COCO loading, batch eval |
| `test_config_manager.py` | 23 | YAML/JSON load/save, nested access, round-trip |
| `test_session_manager.py` | 61 | Session CRUD, parsowanie, filtry, statystyki |
| `test_markdown_renderer.py` | 49 | Headings, listy, checkboxy, code blocks, edge cases |
| `test_log_parser.py` | 47 | Parsowanie logow YOLO/DETR, detekcja formatu |
| `test_model_registry.py` | 30 | Rejestracja, skanowanie, persistencja, usuwanie |
| `test_services.py` | 46 | Checkpoint/Parameter/Dataset analysis services |
| `test_inference_adapter.py` | 30 | ABC, factory, DETR/YOLO, `_require` helper |
| `test_widgets.py` | 23 | LogConsole, ProgressPanel, ConfigForm, FilePicker, BBoxPainter |
| **Suma** | **338** | |

Testy nie wymagaja GPU, torch ani polaczenia SSH -- uzywaja mockow i plikow tymczasowych.

---

## Struktura katalogow

```
CataractAI_Workbench/
├── app/
│   ├── main.py                              # Entry point, QApplication, splash
│   ├── main_window.py                       # QMainWindow + 6 tabow + status bar
│   ├── theme.py                             # Dark Fusion theme + QSS
│   │
│   ├── core/                                # Rdzen (non-GUI)
│   │   ├── config_manager.py                # YAML + JSON config
│   │   ├── signal_bus.py                    # Globalny event bus
│   │   ├── worker_base.py                   # QThread base z sygnalami
│   │   ├── ssh_manager.py                   # Paramiko SSH + jump host
│   │   ├── eden_client.py                   # SLURM API
│   │   ├── log_parser.py                    # Parser logow YOLO/DETR
│   │   ├── model_registry.py                # Rejestr modeli
│   │   ├── project_paths.py                 # Sciezki projektu
│   │   ├── analysis_services.py             # Re-export serwisow
│   │   └── services/
│   │       ├── base.py                      # AnalysisService(ABC)
│   │       ├── checkpoint_service.py        # Analiza checkpointow
│   │       ├── parameter_service.py         # Rekomendacje parametrow
│   │       └── dataset_service.py           # Analiza datasetu COCO
│   │
│   ├── tabs/
│   │   ├── dataset_selection/               # Tab 1
│   │   │   ├── tab_widget.py
│   │   │   ├── config_editor.py
│   │   │   ├── pipeline_runner.py
│   │   │   ├── feature_viz.py
│   │   │   └── results_viewer.py
│   │   │
│   │   ├── training/                        # Tab 2
│   │   │   ├── tab_widget.py
│   │   │   ├── config_editor.py
│   │   │   ├── training_runner.py
│   │   │   ├── live_chart.py
│   │   │   ├── checkpoint_manager.py
│   │   │   ├── training_analyzer.py
│   │   │   ├── checkpoint_inspector.py
│   │   │   ├── parameter_advisor.py
│   │   │   ├── dataset_analyzer_widget.py
│   │   │   └── theme_constants.py
│   │   │
│   │   ├── benchmarks/                      # Tab 3
│   │   │   ├── tab_widget.py
│   │   │   ├── benchmark_runner.py
│   │   │   ├── results_viewer.py
│   │   │   ├── results_charts.py
│   │   │   └── report_generator.py
│   │   │
│   │   ├── eden/                            # Tab 4
│   │   │   ├── tab_widget.py
│   │   │   ├── cluster_dashboard.py
│   │   │   ├── job_manager.py
│   │   │   ├── log_viewer.py
│   │   │   ├── file_transfer.py
│   │   │   └── session_browser.py
│   │   │
│   │   ├── visualization/                   # Tab 5
│   │   │   ├── tab_widget.py
│   │   │   ├── training_dashboard.py
│   │   │   ├── dataset_explorer.py
│   │   │   ├── model_comparison.py
│   │   │   ├── inference_tester.py
│   │   │   ├── inference_controls.py
│   │   │   ├── inference_image_viewer.py
│   │   │   ├── inference_stats.py
│   │   │   ├── inference_worker.py
│   │   │   └── prediction_renderer.py
│   │   │
│   │   └── experiments/                     # Tab 6
│   │       ├── tab_widget.py
│   │       ├── session_manager.py
│   │       ├── experiment_editor.py
│   │       ├── markdown_renderer.py
│   │       ├── session_list_widget.py
│   │       ├── new_session_dialog.py
│   │       └── topic_tags_widget.py
│   │
│   └── widgets/                             # Reuzywalne widgety
│       ├── log_console.py
│       ├── progress_panel.py
│       ├── config_form.py
│       ├── image_gallery.py
│       ├── file_picker.py
│       ├── chart_widget.py
│       ├── bbox_painter.py
│       └── ssh_terminal.py
│
├── backend/                                 # Adaptery
│   ├── training_adapter.py
│   ├── inference_adapter.py
│   ├── dataset_selection_adapter.py
│   ├── benchmark_adapter.py
│   ├── eden_adapter.py
│   └── metrics.py
│
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_config_manager.py
│   ├── test_session_manager.py
│   ├── test_markdown_renderer.py
│   ├── test_log_parser.py
│   ├── test_model_registry.py
│   ├── test_services.py
│   ├── test_inference_adapter.py
│   └── test_widgets.py
│
├── requirements.txt
└── setup.py
```

---

## Kontekst projektu

CataractAI Workbench jest czescia projektu PhD **ViTParticleFilterTracker** -- systemu detekcji narzedzi chirurgicznych w operacjach zacmy. Projekt obejmuje:

- **AdvancedDatasetSelection/** -- pipeline selekcji optymalnego podzbioru z 91k obrazow (DINO, Fourier, SAM, EL2N, k-center)
- **BackgroundFinetuned/** -- trening DETR z mixed gentle fine-tuning
- **YOLO_DETR_Benchmarks/** -- 26+ skryptow benchmarkowych porownujacych YOLO vs DETR
- **Eden/** -- skrypty do klastra HPC (SLURM, GPU A100/H100)
- **.sessions/** -- system dokumentacji eksperymentow (40+ sesji)

Workbench konsoliduje te komponenty w jeden interfejs, zachowujac istniejacy kod nienaruszony (adapter-first, 3 pliki zmienione, ~15 linii).
