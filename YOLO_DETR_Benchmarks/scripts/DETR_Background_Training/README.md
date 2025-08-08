# DETR Background Training

Ten folder zawiera wszystkie skrypty i pliki związane z fine-tuningiem modelu DETR z dodatkową klasą "background".

## Struktura folderu

```
DETR_Background_Training/
├── README.md                                    # Ten plik - instrukcje
├── README_finetune.md                          # Szczegółowe instrukcje fine-tuningu
├── detr_finetune_background.py                 # Główny skrypt Python fine-tuningu
├── run_detr_finetune_background.slurm          # Skrypt SLURM dla hopper (8 GPU)
└── run_detr_finetune_background_pascal.slurm   # Skrypt SLURM dla pascal (4 GPU)
```

## Wymagania przed uruchomieniem

1. **Checkpoint oryginalnego modelu**: `./Checkpoints/final_checkpoint.pth`
2. **Dataset background**: `/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_Background_dataset.zip`
3. **Oryginalny dataset**: dostępny w archiwum `datasets_20250606.tar.gz`

## Szybkie uruchomienie

### Opcja 1: hopper (8 GPU) - zalecana
```bash
cd /mnt/evafs/faculty/home/bpiotrowski/DETR
sbatch DETR_Background_Training/run_detr_finetune_background.slurm
```

### Opcja 2: pascal (4 GPU) - zapasowa
```bash
cd /mnt/evafs/faculty/home/bpiotrowski/DETR
sbatch DETR_Background_Training/run_detr_finetune_background_pascal.slurm
```

## Monitorowanie

```bash
# Status zadania
squeue -u $USER

# Podgląd logów
tail -f logs/detr_finetune_bg_hopper_*.log
# lub
tail -f logs/detr_finetune_bg_pascal_*.log

# Sprawdzenie dostępności węzłów
sfree
```

## Wyniki

Po zakończeniu treningu model zostanie zapisany w:
- **Najlepszy model**: `./best_models/DETR_fine_tuned_background_model/`
- **Checkpointy**: `./Checkpoints/DETR_fine_tuned_background_model/`
- **Logi**: `./output_DETR_fine_tuned_background_model/logs/`

## Szczegółowe instrukcje

Zobacz `README_finetune.md` dla pełnych instrukcji, parametrów i troubleshootingu.

## Klasy w modelu po fine-tuningu

- **Klasa 1**: `surgical_tool` (zachowane wagi z oryginalnego modelu)
- **Klasa 2**: `background` (nowo wytrenowana klasa)

## Kontakt

W przypadku problemów sprawdź logi lub skontaktuj się z administratorem klastra.