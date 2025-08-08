# DETR Fine-tuning dla klasy "background"

## Przegląd

Ten zestaw skryptów umożliwia fine-tuning wytrenowanego modelu DETR z dodatkową klasą "background". Skrypty są zoptymalizowane dla treningu rozproszonego (DDP) na 8 GPU w klastrze Eden.

## Pliki

- `scripts/detr_finetune_background.py` - główny skrypt fine-tuningu
- `scripts/run_detr_finetune_background.slurm` - plik wsadowy SLURM
- `README_finetune.md` - ten plik z instrukcjami

## Wymagania

1. Wytrenowany checkpoint DETR (100+ epok) w `./Checkpoints/final_checkpoint.pth`
2. Dataset z klasą background w `/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_Background_dataset.zip`
3. Oryginalny dataset z narzędziami chirurgicznymi w `datasets_20250606.tar.gz`
4. Środowisko conda `yolo_py310` z zainstalowanymi zależnościami

## Strategia fine-tuningu

### Architektura modelu
- Dodanie nowej klasy "background" (ID: 2) do istniejących klas
- Modyfikacja warstwy klasyfikacyjnej (`class_embed`) dla nowej liczby klas
- Kopiowanie wag dla istniejących klas, inicjalizacja losowa dla nowej klasy

### Strategia uczenia
- **Backbone (ResNet)**: zamrożony lub bardzo niski LR (1e-6)
- **Encoder**: niski LR (5e-6) 
- **Decoder**: średni LR (1e-4)
- **Classification head**: wysoki LR (5e-4)

### Balansowanie danych
- Kombinacja oryginalnego datasetu + dataset background
- Parametr `background_weight` kontroluje proporcję próbek background
- Domyślnie: background stanowi 30% datasetu treningowego

## Użycie

### 1. Przygotowanie

Sprawdź ścieżki w pliku SLURM:
```bash
# W run_detr_finetune_background.slurm
CHECKPOINT_PATH="/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/final_checkpoint.pth"
```

### 2. Uruchomienie na klastrze

```bash
cd /mnt/evafs/faculty/home/bpiotrowski/DETR
sbatch scripts/run_detr_finetune_background.slurm
```

### 3. Monitorowanie

```bash
# Sprawdzenie statusu zadania
squeue -u $USER

# Podgląd logów
tail -f logs/detr_finetune_bg_8gpu_*.log

# TensorBoard (jeśli dostępny)
tensorboard --logdir=train_out_finetune_bg/logs
```

## Parametry fine-tuningu

### Kluczowe parametry w SLURM:

```bash
--epochs                    15          # Krótszy trening niż od zera
--batch_size                8           # Mniejszy batch dla fine-tuningu
--background_weight         0.3         # 30% próbek background
--freeze_backbone                       # Zamróż backbone
--lr_head                   5e-4        # Wysoki LR dla nowej warstwy
--lr_decoder                1e-4        # Średni LR dla dekodera
--lr_encoder                5e-6        # Niski LR dla enkodera
--patience                  8           # Early stopping
```

### Modyfikowalne parametry:

- `--background_weight`: waga próbek background (0.1-1.0)
- `--freeze_backbone`: czy zamrozić backbone
- `--freeze_encoder`: czy zamrozić encoder
- Learning rates dla poszczególnych komponentów
- `--epochs`: liczba epok (zalecane: 10-20)

## Struktura wyjściowa

```
├── output_DETR_fine_tuned_background_model/     # Główny folder wyjściowy
│   ├── logs/                                    # Logi TensorBoard
│   └── final_model/                            # Model po zakończeniu
├── Checkpoints/DETR_fine_tuned_background_model/ # Checkpointy co N epok
└── best_models/DETR_fine_tuned_background_model/  # Najlepszy model (lowest validation loss)
```

## Walidacja modelu

Po fine-tuningu model będzie miał:
- **Klasa 1**: `surgical_tool` (zachowane wagi)
- **Klasa 2**: `background` (nowe wagi)

Sprawdź konfigurację modelu:
```python
from transformers import DetrConfig, DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained("./best_models/DETR_fine_tuned_background_model/")
print("Liczba klas:", model.config.num_labels)
print("Mapowanie:", model.config.id2label)
```

## Troubleshooting

### Problem: Brak checkpointa
```
❌ Brak checkpointa w: /mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/final_checkpoint.pth
```
**Rozwiązanie**: Sprawdź ścieżkę `CHECKPOINT_PATH` w pliku SLURM i upewnij się że oryginalny trening został zakończony

### Problem: CUDA out of memory
**Rozwiązanie**: Zmniejsz `batch_size` z 8 na 4 lub 2

### Problem: Model nie uczy się klasy background
**Rozwiązanie**: 
- Zwiększ `background_weight` (np. do 0.5-1.0)
- Zwiększ `lr_head` (np. do 1e-3)
- Sprawdź czy dataset background ma poprawne anotacje (puste)

### Problem: Overfiiting na klasie background
**Rozwiązanie**:
- Zmniejsz `background_weight` (np. do 0.1-0.2)
- Zwiększ regularyzację (`weight_decay`)
- Zmniejsz `lr_head`

## Parametry domyślne vs zalecane

| Parametr | Domyślne | Zalecane | Uwagi |
|----------|----------|----------|-------|
| epochs | 15 | 10-20 | W zależności od konwergencji |
| batch_size | 8 | 4-8 | Dostosuj do pamięci GPU |
| background_weight | 0.3 | 0.2-0.5 | Zależy od wielkości datasetu |
| lr_head | 5e-4 | 1e-4 - 1e-3 | Najważniejszy parametr |
| patience | 8 | 5-10 | Wcześniejsze zatrzymanie |

## Dalsze kroki

Po zakończeniu fine-tuningu:

1. **Ewaluacja**: Przetestuj model na zestawie testowym
2. **Analiza**: Sprawdź metryki dla każdej klasy osobno
3. **Optymalizacja**: Dostosuj parametry jeśli potrzebne
4. **Deploy**: Użyj najlepszego modelu z `best_models/DETR_fine_tuned_background_model/`

## Kontakt

W przypadku problemów sprawdź logi lub skontaktuj się z administratorem klastra.