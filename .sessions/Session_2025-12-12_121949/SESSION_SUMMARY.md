# Session Summary: Session_2025-12-12_121949

## Metadata
- **Date:** 2025-12-12
- **Time:** 12:19:49
- **Duration:** ~1.5 godziny
- **Status:** Active (job w kolejce)
- **Type:** Training | SSH

## Objective
Przygotować dataset 20k wybranych obrazów do fine-tuningu modelu DETR na klastrze obliczeniowym EDEN. Połączyć adnotacje z dwóch źródeł i uruchomić trening.

## Context
- Poprzednio wybrano 20,000 najlepszych obrazów z datasetu używając AdvancedDatasetSelection
- Obrazy pochodzą z dwóch źródeł:
  - 48 obrazów z Roboflow (`_annotations.coco.json`) - 640x640
  - 19,952 obrazów z augmented dataset (`augmented_coco_20250417_030014.json`) - 1920x1080
- Model DETR był wcześniej trenowany do epoch 170 na dużym datasecie 90k

## Actions Taken

### 1. Analiza plików adnotacji
- Przeanalizowano strukturę obu plików JSON (format COCO)
- Zidentyfikowano różnice:
  - Różne `category_id` (1 vs 0)
  - Różne rozdzielczości obrazów
  - Kolizje ID

### 2. Stworzenie skryptu mergującego
- `merge_annotations.py` - łączy adnotacje z dwóch źródeł
- Filtruje tylko obrazy istniejące w katalogu `images/`
- Unifikuje category_id do 1 (tooltip)
- Przypisuje nowe sekwencyjne ID

### 3. Naprawa duplikatów bounding boxów
- Wykryto 215 obrazów z >1 bbox (216 duplikatów)
- Przyczyna: augmentacja geometryczna dzieliła bbox przy krawędzi
- `fix_duplicate_bboxes.py` - zachowuje tylko bbox z największą area
- Wynik: 20,000 obrazów = 20,000 adnotacji (1:1)

### 4. Wizualizacja
- `visualize_annotations.py` - rysuje bbox na obrazach
- Przetestowano na 100 losowych obrazach
- Zweryfikowano poprawność po naprawie duplikatów

### 5. Aktualizacja archiwum tar
- Dodano `merged_20k_annotations.json` do `20kSelectedImages.tar`
- Archiwum gotowe do przesłania na EDEN

### 6. Przygotowanie skryptu SLURM
- `run_detr_20k_finetune.slurm` - konfiguracja treningu
- Resume z `checkpoint_epoch_170.pth`
- Parametry fine-tuningu: LR=5e-5, backbone_LR=5e-6

### 7. Uruchomienie na EDEN
- Przesłano pliki na klaster
- Rozwiązano problemy: DOS line endings, partycja, time limit
- Job wysłany: `1453424`

## Results

### Key Findings
- Dataset 20k ma spójne adnotacje w formacie COCO
- Każdy obraz ma dokładnie 1 bounding box
- Job w kolejce z priorytetem (wszystkie GPU zajęte)

### Metrics/Data
| Metryka | Wartość |
|---------|---------|
| Obrazy | 20,000 |
| Adnotacje | 20,000 |
| Duplikaty usunięte | 216 |
| Kategorie | 1 (tooltip) |
| Rozdzielczości | 640x640, 1920x1080 |

### SLURM Configuration
| Parametr | Wartość |
|----------|---------|
| Partycja | long |
| GPU | 4 (dowolne) |
| Czas | 5 dni |
| RAM | 250GB |
| Job ID | 1453424 |

### Issues Encountered
- **DOS line breaks** → Konwersja CRLF→LF
- **Invalid partition dgx** → Usunięto, tylko `long`
- **PartitionTimeLimit** → Zmniejszono z 7 na 5 dni
- **Duplikaty bbox** → Skrypt fix_duplicate_bboxes.py

## Conclusions
- Dataset 20k jest gotowy do treningu
- Adnotacje zostały pomyślnie połączone i naprawione
- Job czeka w kolejce na wolne GPU
- Checkpointy będą zapisywane co 5 epok dla bezpieczeństwa

## Next Steps
- [ ] Monitorować status joba: `squeue -u bpiotrowski`
- [ ] Po starcie: sprawdzić logi `logs/detr_20k_finetune_1453424.log`
- [ ] Po zakończeniu: ewaluacja modelu na zbiorze walidacyjnym
- [ ] Porównanie z modelem trenowanym na 90k

## Files Generated

### Skrypty mergowania i naprawy
- `AdvancedDatasetSelection/output/selected_dataset/merge_annotations.py`
- `AdvancedDatasetSelection/output/selected_dataset/fix_duplicate_bboxes.py`
- `AdvancedDatasetSelection/output/selected_dataset/visualize_annotations.py`

### Adnotacje
- `AdvancedDatasetSelection/output/selected_dataset/merged_20k_annotations.json` - GŁÓWNY PLIK
- `AdvancedDatasetSelection/output/selected_dataset/merged_20k_annotations_fixed.json` - kopia
- `AdvancedDatasetSelection/output/selected_dataset/merged_20k_annotations_with_duplicates.json.bak` - backup

### Archiwum
- `AdvancedDatasetSelection/output/selected_dataset/20kSelectedImages.tar` - zaktualizowane

### SLURM
- `Eden/Scripts/20kDataset_11.12.25/run_detr_20k_finetune.slurm`
- `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py` (bez zmian)

### Wizualizacje
- `AdvancedDatasetSelection/output/selected_dataset/visualizations_bbox/` - 100 testowych
- `AdvancedDatasetSelection/output/selected_dataset/visualizations_bbox_fixed/` - 50 po naprawie

## Commands Used

### Lokalne (Windows)
```powershell
# Merge adnotacji
py -3.11 "F:\...\merge_annotations.py"

# Naprawa duplikatów
py -3.11 "F:\...\fix_duplicate_bboxes.py"

# Wizualizacja
py -3.11 "F:\...\visualize_annotations.py" --max 100 --random

# Wizualizacja wszystkich 20k
py -3.11 "F:\...\visualize_annotations.py" --max 20000
```

### EDEN (Linux)
```bash
# Transfer plików
scp 20kSelectedImages.tar bpiotrowski@eden:/mnt/evafs/faculty/home/bpiotrowski/
scp run_detr_20k_finetune.slurm bpiotrowski@eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/

# Uruchomienie
sbatch run_detr_20k_finetune.slurm

# Monitoring
squeue -u bpiotrowski
squeue -u bpiotrowski --start
sfree
sinfo -p long -o "%P %l"

# Naprawa skryptu
sed -i 's/7-00:00:00/5-00:00:00/' run_detr_20k_finetune.slurm
```

## Related Work
- **Previous session:** `Session_2025-12-11_205605` - Advanced Dataset Selection z DETR Q81
- **Checkpoint source:** `/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/checkpoint_epoch_170.pth`
- **Dataset archive:** `/mnt/evafs/faculty/home/bpiotrowski/20kSelectedImages.tar`
