# BackgroundFinetuned Architecture Refactoring Plan

**Data utworzenia:** 2026-01-18
**Status:** Draft
**Priorytet:** High

---

## Cel

Transformacja chaotycznej struktury folderu BackgroundFinetuned w dobrze zorganizowany, łatwy w utrzymaniu projekt z czystą architekturą, eliminacją duplikacji kodu i efektywnym zarządzaniem zasobami.

## Kontekst

### Zidentyfikowane problemy

| Problem | Skala | Wpływ |
|---------|-------|-------|
| Duplikacja trenerów | 10 plików z ~70% wspólnego kodu | Trudność utrzymania |
| Chaos organizacyjny | 48 plików Python w root | Nawigacja niemożliwa |
| Nadmiar checkpointów | 109 plików .pth = 34GB | Zajętość dysku |
| Rozproszone temp folders | 8 folderów w różnych lokalizacjach | Bałagan |
| Cache/śmieci | `__pycache__`, logi, `.zip` | Niepotrzebne pliki |

### Statystyki przed refaktoryzacją
- **Pliki Python w root:** 48
- **Pliki trenerów:** 10
- **Duplikacja kodu:** ~70%
- **Checkpointy w output:** 109 (34GB)
- **Temp folders:** 8
- **Czas znalezienia pliku:** ~5-10 minut

## Wymagania

### Funkcjonalne
- [ ] Jeden zunifikowany system treningowy ze strategiami
- [ ] Scentralizowane zarządzanie konfiguracją (YAML)
- [ ] Automatyczne zarządzanie checkpointami (retention policy)
- [ ] Czytelna struktura katalogów

### Techniczne
- [ ] Zachowanie historii git przy przenoszeniu plików
- [ ] Backward compatibility dla istniejących skryptów
- [ ] Testy jednostkowe dla kluczowych komponentów
- [ ] Dokumentacja API

## Plan Implementacji

### Faza 1: Krytyczne Czyszczenie (Priorytet: CRITICAL)

#### 1.1 Usunięcie śmieci
- [ ] Usunąć `__pycache__/` (wszystkie lokalizacje)
- [ ] Usunąć `*.log` (test_*.log w root)
- [ ] Usunąć `triple_comparison.zip`
- [ ] Usunąć `test_dir/` (jeśli puste)

#### 1.2 Archiwizacja starych checkpointów
- [ ] Zachować tylko 5 najnowszych checkpointów w `models/trained/`
- [ ] Przenieść pozostałe 104 checkpointy → `archive/old_models/`
- [ ] Opcjonalnie: skompresować archiwum (34GB → ~15GB)

#### 1.3 Konsolidacja temp folders
- [ ] Zweryfikować zawartość wszystkich `temp_*` folderów
- [ ] Usunąć puste/niepotrzebne temp foldery
- [ ] Utworzyć scentralizowany `tmp/` w root z `.gitignore`

### Faza 2: Konsolidacja Kodu (Priorytet: HIGH)

#### 2.1 Nowa struktura katalogów
```
src/
├── core/           # config, constants, exceptions, paths
├── models/         # detr/, yolo/, factory
├── training/       # base_trainer, strategies/, callbacks, metrics
├── data/           # loaders/, processors/, combiners/, validators
├── analysis/       # checkpoint_analyzer, query_analyzer, model_comparator
├── visualization/  # training_plots, detection_visualizer, query_visualizer
├── validation/     # dataset_validator, model_validator
└── utils/          # logging, file_utils, gpu_utils, coco_utils
```

#### 2.2 Migracja trenerów (10 → 4 strategie)
Pliki do połączenia:
- [ ] `production_finetune_mixed.py` → `src/training/strategies/mixed_trainer.py`
- [ ] `production_finetune_mixed_v_2.py` → DELETE (duplikat)
- [ ] `production_q81_trainer.py` → `src/training/strategies/query81_trainer.py`
- [ ] `proper_detr_trainer.py` → `src/training/strategies/detr_trainer.py`
- [ ] `query81_trainer.py` → MERGE → `query81_trainer.py`
- [ ] `simple_production_trainer.py` → `src/training/strategies/production_trainer.py`
- [ ] `simple_q81_finetuner.py` → MERGE → `query81_trainer.py`
- [ ] `wsl6_production_detector.py` → MERGE → `production_trainer.py`
- [ ] `Src/trainer/mixed_detr_finetune.py` → MERGE → `mixed_trainer.py`
- [ ] `Src/trainer/mixed_detr_trainer_v2.py` → MERGE → `mixed_trainer.py`

Nowy system:
- [ ] Utworzyć `src/training/base_trainer.py` (abstrakcyjna klasa)
- [ ] Utworzyć `scripts/train.py` - unified CLI
- [ ] Testować każdą strategię osobno

#### 2.3 Migracja data processing
- [ ] `Src/dataset_creator/` → `src/data/processors/`
- [ ] `intelligent_background_frame_selectorDYD.py` → `background_selector.py`
- [ ] `dino_information_analyzer.py` → `dino_analyzer.py`
- [ ] `dataset_combiner_v2.py` → `src/data/combiners/dataset_combiner.py`
- [ ] Usunąć `dataset_combiner.py` (stara wersja)

#### 2.4 Migracja analizy i wizualizacji
- [ ] `analyze_detr_checkpoint*.py` (3 pliki) → `src/analysis/checkpoint_analyzer.py`
- [ ] `visualize_*.py` (5 plików) → `src/visualization/` (3 moduły)
- [ ] `compare_models_on_yolo.py` → `src/analysis/model_comparator.py`
- [ ] `find_tooltip_queries.py` → `src/analysis/query_analyzer.py`

#### 2.5 Migracja validation
- [ ] `Src/Validation/` → `src/validation/`
- [ ] `dataset_quality_validation.py` → `dataset_validator.py`
- [ ] `tool_only_validation.py` → `model_validator.py`
- [ ] `dual_detection_visualizer.py` → `../visualization/`

### Faza 3: Modernizacja Konfiguracji (Priorytet: MEDIUM)

#### 3.1 YAML configuration system
- [ ] Utworzyć `configs/base_config.yaml`
- [ ] Utworzyć `configs/environments/development.yaml`
- [ ] Utworzyć `configs/environments/production.yaml`
- [ ] Utworzyć `configs/environments/testing.yaml`
- [ ] Utworzyć `configs/experiments/query81_experiment.yaml`
- [ ] Utworzyć `configs/experiments/mixed_training_experiment.yaml`
- [ ] Migrować `Settings/*.json` → `configs/*.yaml`

#### 3.2 Centralna konfiguracja ścieżek
- [ ] Utworzyć `src/core/paths.py` z klasą `ProjectPaths`
- [ ] Refaktoryzować wszystkie hardkodowane ścieżki

### Faza 4: Testy i Dokumentacja (Priorytet: MEDIUM)

#### 4.1 Test infrastructure
- [ ] Utworzyć `tests/unit/test_models/`
- [ ] Utworzyć `tests/unit/test_training/`
- [ ] Utworzyć `tests/unit/test_data/`
- [ ] Utworzyć `tests/integration/test_pipelines/`
- [ ] Utworzyć `tests/fixtures/`
- [ ] Dodać `pytest.ini`, `conftest.py`

#### 4.2 Dokumentacja
- [ ] Przenieść `README.md` → `docs/README.md` (rozszerzyć)
- [ ] Przenieść `CLAUDE.md` → `docs/sessions/`
- [ ] Przenieść `Commands.md` → `docs/sessions/`
- [ ] Utworzyć `docs/api/` (auto-generated)
- [ ] Utworzyć `docs/tutorials/quick_start.md`
- [ ] Utworzyć `docs/tutorials/training_guide.md`

## Architektura / Design

### Docelowa struktura katalogów

```
BackgroundFinetuned/
│
├── src/                              # GŁÓWNY KOD ŹRÓDŁOWY
│   ├── core/
│   │   ├── config.py                 # Centralna konfiguracja
│   │   ├── constants.py              # Stałe projektu
│   │   ├── exceptions.py             # Custom exceptions
│   │   └── paths.py                  # Zarządzanie ścieżkami
│   │
│   ├── models/
│   │   ├── detr/
│   │   │   ├── model.py
│   │   │   ├── processor.py
│   │   │   └── converter.py
│   │   ├── yolo/
│   │   │   ├── model.py
│   │   │   └── converter.py
│   │   └── factory.py
│   │
│   ├── training/
│   │   ├── base_trainer.py           # Abstract BaseTrainer
│   │   ├── strategies/
│   │   │   ├── detr_trainer.py       # ZASTĘPUJE 10 TRAINERÓW
│   │   │   ├── query81_trainer.py
│   │   │   ├── mixed_trainer.py
│   │   │   └── production_trainer.py
│   │   ├── callbacks.py
│   │   ├── metrics.py
│   │   └── utils.py
│   │
│   ├── data/
│   │   ├── loaders/
│   │   ├── processors/
│   │   ├── combiners/
│   │   └── validators.py
│   │
│   ├── analysis/
│   ├── visualization/
│   ├── validation/
│   └── utils/
│
├── scripts/                          # ENTRY POINTS
│   ├── train.py                      # Unified CLI
│   ├── evaluate.py
│   ├── generate_dataset.py
│   ├── analyze_model.py
│   └── visualize_results.py
│
├── configs/                          # KONFIGURACJE
│   ├── base_config.yaml
│   ├── environments/
│   └── experiments/
│
├── tests/                            # TESTY
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── data/                             # DANE (z Datasets/)
│   ├── background/
│   ├── tooltip_mining/
│   └── mixed/
│
├── models/                           # MODELE
│   ├── checkpoints/
│   └── trained/
│
├── outputs/                          # WYJŚCIA (z output/)
│   ├── training_runs/
│   └── analysis_reports/
│
├── logs/
├── tmp/                              # SCENTRALIZOWANE TEMP
├── archive/                          # ARCHIWUM
│   ├── old_models/
│   └── old_results/
│
├── docs/
│   ├── api/
│   ├── architecture/
│   ├── tutorials/
│   └── sessions/
│
├── README.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

### Mapowanie zmian

| Kategoria | Ilość | Opis |
|-----------|-------|------|
| DELETE | 17 | Śmieci, duplikaty, temp folders |
| RENAME | 10 | Lepsze nazwy plików/folderów |
| MOVE | 35+ | Przeniesienie do właściwych lokalizacji |
| CREATE | 25 | Nowe foldery i pliki strukturalne |
| MERGE | 8 grup | Konsolidacja duplikatów |

## Zależności

- Python 3.11+
- PyTorch & Transformers (DETR)
- Ultralytics (YOLO)
- COCO API
- Pillow, OpenCV
- pytest (testy)
- pydantic (walidacja config)

## Ryzyka i Mitygacje

| Ryzyko | Prawdop. | Wpływ | Mitygacja |
|--------|----------|-------|-----------|
| Zepsucie importów | High | High | Testować każdy krok, backup branch |
| Utrata historii git | Medium | Medium | Używać `git mv` zamiast copy+delete |
| Broken scripts | High | Medium | Legacy folder jako backup przez miesiąc |
| Brakujące zależności | Low | Low | requirements.txt z pinami wersji |
| Regression bugs | Medium | High | Testy integracyjne przed/po |

## Kryteria Sukcesu

### Metryki docelowe
- [ ] Pliki Python w root: 48 → 5-8
- [ ] Pliki trenerów: 10 → 1 system (4 strategie)
- [ ] Duplikacja kodu: ~70% → <10%
- [ ] Aktywne checkpointy: 109 → 5-10 (reszta w archive)
- [ ] Temp folders: 8 → 1 (scentralizowany)
- [ ] Czas znalezienia pliku: ~5-10 min → <1 min
- [ ] Testy jednostkowe: 0 → 20+
- [ ] Dokumentacja API: 0% → 80%

## Powiązane Pliki

- `BackgroundFinetuned/REFACTORING_PLAN.md` (istniejący plan)
- `BackgroundFinetuned/ARCHITECTURE_DIAGRAMS.md` (istniejące diagramy)
- `BackgroundFinetuned/CLAUDE.md` (historia sesji)
- `BackgroundFinetuned/Commands.md` (komendy)

## Opcje Wykonania

| Opcja | Zakres | Szacowany czas |
|-------|--------|----------------|
| **A) Quick wins** | Cleanup śmieci, archive checkpoints | 1 dzień |
| **B) Pełna refaktoryzacja** | Wszystkie 4 fazy | 7-10 dni |
| **C) Tylko trenerzy** | Konsolidacja 10→4 strategie | 3 dni |

## Notatki

### Przed rozpoczęciem
1. Utworzyć backup branch: `git checkout -b backup/pre-refactoring`
2. Uruchomić istniejące testy (jeśli są)
3. Zweryfikować działanie głównych skryptów

### Alternatywne podejścia
- Można rozważyć użycie `cookiecutter` template dla Python ML projects
- Dla konfiguracji: Hydra zamiast czystego YAML
- Dla trenerów: PyTorch Lightning jako base framework

### Uwagi bezpieczeństwa
- NIGDY nie usuwać bez jawnego zatwierdzenia użytkownika
- Każda większa zmiana = osobny commit
- Zachować stare pliki w `legacy/` przez minimum 30 dni

---

*Plan utworzony: 2026-01-18*
*Projekt: ViTParticleFilterTracker/BackgroundFinetuned*
*Agent: project-architect (ID: a5bffa3)*
