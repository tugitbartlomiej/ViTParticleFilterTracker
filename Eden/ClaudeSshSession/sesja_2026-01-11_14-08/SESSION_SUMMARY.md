# Sesja SSH Eden - 2026-01-11 14:08 UTC

**Data:** 2026-01-11
**Godzina:** 14:08 UTC (15:08 CET)
**Job IDs:** 1509211 (DETR), 1509212 (YOLO)
**Node:** dgx-4 (DETR planowany), dgx-2 (YOLO planowany)
**Status:** QUEUED (oba joby w kolejce)

---

## Podsumowanie Sesji

Naprawa błędu składni bash w skryptach SLURM i ponowne uruchomienie treningów DETR i YOLO na 20k dataset do epoch 500.

### Kluczowe Osiągnięcia:

1. **Zidentyfikowano przyczynę błędu joba DETR 1509138**
   - Błąd: `line 49: conditional binary o`
   - Przyczyna: `\!` zamiast `!` w warunkach bash `[[ ]]`
   - Dotyczyło linii 49, 76 w DETR i linii 173 w YOLO

2. **Naprawiono skrypty SLURM**
   - `run_detr_20k_resume_4gpu.slurm` - usunięto `\!` → `!`
   - `run_yolo_20k_resume_500.slurm` - usunięto `\!` → `!`

3. **Uruchomiono nowe joby**
   - DETR: Job 1509211 (ep295→500)
   - YOLO: Job 1509212 (ep200→500)

4. **Wyjaśniono lokalizacje datasetów**
   - DETR używa formatu COCO
   - YOLO konwertuje COCO→YOLO on-the-fly

---

## Struktura Katalogów na Eden

### DETR
```
/mnt/evafs/faculty/home/bpiotrowski/DETR/
├── ckpt_20k_finetune_v2_fixed/          # Working checkpoints (13GB)
│   ├── checkpoint_epoch_170.pth         # Start point
│   ├── checkpoint_epoch_175.pth
│   ├── ...
│   └── checkpoint_epoch_295.pth         # Resume point
├── Checkpoints/20k_finetune_v2_fixed/   # Archiwum (12GB)
│   ├── checkpoint_epoch_170.pth
│   ├── ...
│   ├── checkpoint_epoch_290.pth
│   └── tensorboard_logs/
├── train_out_20k_finetune/
│   └── logs/                            # TensorBoard runtime
├── best_20k_finetune/
├── run_detr_20k_resume_4gpu.slurm       # NAPRAWIONY skrypt
└── detr_train_optimized.py              # Skrypt treningowy
```

### YOLO
```
/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/
├── train_out_5gpu_hopper/exp/weights/
│   ├── best.pt                          # 87 MB - najlepszy (mAP 99.5%)
│   ├── last.pt                          # 87 MB - epoch 200
│   ├── epoch180.pt
│   └── epoch190.pt
├── run_yolo_20k_resume_500.slurm        # NAPRAWIONY skrypt
├── yolo_train_20k_finetune.py
├── train_out_500/                       # Output dla nowego treningu
├── ckpt_500/
└── best_500/
```

---

## Lokalizacje Datasetów

### DETR (format COCO)
| Element | Ścieżka |
|---------|---------|
| Obrazy (archiwum) | `~/20kSelectedImages.tar` |
| Adnotacje COCO | `~/merged_20k_annotations_fixed.json` |

### YOLO (format YOLO)
| Element | Opis |
|---------|------|
| Źródło | Te same obrazy co DETR |
| Konwersja | COCO→YOLO generowane on-the-fly w skrypcie |
| Labels | `$TMPDIR/yolo_dataset/labels/{train,val}/*.txt` |
| Config | `$TMPDIR/yolo_dataset/dataset.yaml` |

**UWAGA:** YOLO nie ma gotowych pre-generated labels na Eden - są tworzone przy każdym uruchomieniu z COCO JSON.

---

## Błąd Składni Bash - Szczegóły

### Problem
```bash
# ŹLE (powoduje błąd "conditional binary o"):
if [[ \! -d "$IMAGES_DIR" ]]; then

# POPRAWNIE:
if [[ ! -d "$IMAGES_DIR" ]]; then
```

### Dotknięte linie
| Plik | Linia | Błędne | Poprawne |
|------|-------|--------|----------|
| `run_detr_20k_resume_4gpu.slurm` | 49 | `\! -d` | `! -d` |
| `run_detr_20k_resume_4gpu.slurm` | 76 | `\! -f` | `! -f` |
| `run_yolo_20k_resume_500.slurm` | 173 | `\! -f` | `! -f` |

### Przyczyna
Prawdopodobnie escape'owanie podczas kopiowania/tworzenia plików na klastrze.

---

## Konfiguracja Treningu

### DETR (Job 1509211)
```bash
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --time=5-00:00:00

torchrun --nproc_per_node=3 detr_train_optimized.py \
    --images_dir "$IMAGES_DIR" \
    --annotations_path "$ANNOT_JSON" \
    --epochs 500 \
    --batch_size 4 \
    --lr 5e-5 \
    --lr_backbone 5e-6 \
    --lr_scheduler cosine \
    --lr_min 1e-7 \
    --save_interval 5 \
    --patience 50 \
    --resume_training
```

### YOLO (Job 1509212)
```bash
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --time=5-00:00:00

python3 yolo_train_20k_finetune.py \
    --dataset_yaml_path $DATASET_YAML \
    --epochs 500 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --save_period 10 \
    --patience 50 \
    --resume --resume_path last.pt
```

---

## Ważne Ścieżki

| Ścieżka | Opis |
|---------|------|
| `~/DETR/ckpt_20k_finetune_v2_fixed/checkpoint_epoch_295.pth` | DETR resume point |
| `~/DETR/Checkpoints/20k_finetune_v2_fixed/tensorboard_logs/` | DETR TensorBoard archiwum |
| `~/Yolo/20k_finetune/train_out_5gpu_hopper/exp/weights/last.pt` | YOLO resume (ep200) |
| `~/Yolo/20k_finetune/train_out_5gpu_hopper/exp/weights/best.pt` | YOLO best (mAP 99.5%) |
| `~/20kSelectedImages.tar` | Dataset archiwum (~9GB) |
| `~/merged_20k_annotations_fixed.json` | Adnotacje COCO |

---

## Komendy SSH

### Status kolejki
```bash
ssh eden-cluster "squeue -u bpiotrowski"
ssh eden-cluster "squeue -u bpiotrowski --start"  # Szacowany start
```

### Logi (gdy joby ruszą)
```bash
# DETR
ssh eden-cluster "tail -f ~/DETR/logs/detr_20k_4gpu_1509211.log"

# YOLO
ssh eden-cluster "tail -f ~/Yolo/20k_finetune/logs/yolo_20k_resume500_1509212.log"
```

### Dostępność GPU
```bash
ssh eden-cluster "sfree"
```

### TensorBoard
```bash
ssh eden-cluster "ls -la ~/DETR/train_out_20k_finetune/logs/"
ssh eden-cluster "ls -la ~/DETR/Checkpoints/20k_finetune_v2_fixed/tensorboard_logs/"
```

---

## Problemy i Rozwiązania

### Problem 1: DETR job 1509138 FAILED z błędem składni bash
**Błąd:** `line 49: conditional binary o`
**Przyczyna:** `\!` zamiast `!` w warunkach `[[ ]]`
**Rozwiązanie:** Edycja lokalna skryptów, usunięcie backslash przed `!`, scp na klaster

### Problem 2: Stare joby zakończyły się błędem
**Status:** Kolejka była pusta - joby już failowały
**Rozwiązanie:** Nie trzeba było anulować, uruchomiono nowe joby

---

## Status na Koniec Sesji

### Joby w kolejce
| Job ID | Model | Epochs | GPUs | Est. Start | Node |
|--------|-------|--------|------|------------|------|
| 1509211 | DETR | 295→500 | 3 | 2026-01-11 17:40 | dgx-4 |
| 1509212 | YOLO | 200→500 | 3 | 2026-01-12 05:16 | dgx-2 |

### Stan klastra (sfree)
```
Node      Free GPUs
dgx-1         8 / 8    (pełna dostępność)
dgx-2         0 / 8    (zajęty)
dgx-3         4 / 8    (częściowo wolny)
dgx-4         2 / 8    (częściowo wolny)
hopper        0 / 8    (zajęty)
hopper-2      8 / 8    (pełna dostępność)
```

### Następne kroki
- [ ] Monitorować start jobów DETR i YOLO
- [ ] Sprawdzić czy skrypty działają poprawnie po starcie
- [ ] Sprawdzić TensorBoard logi po starcie treningu
- [ ] Po zakończeniu: benchmark DETR ep500 vs YOLO ep500

---

## Pliki Skryptów (naprawione)

### run_detr_20k_resume_4gpu.slurm
Lokalizacja Eden: `/mnt/evafs/faculty/home/bpiotrowski/DETR/run_detr_20k_resume_4gpu.slurm`
- 3 GPU, partycja long
- Resume z ep295, target 500
- TensorBoard sync co 10 min
- **NAPRAWIONO:** linia 49, 76 - `\!` → `!`

### run_yolo_20k_resume_500.slurm
Lokalizacja Eden: `/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/run_yolo_20k_resume_500.slurm`
- 3 GPU, partycja long
- Resume z last.pt (ep200), target 500
- Output: train_out_500/
- **NAPRAWIONO:** linia 173 - `\!` → `!`

---

*Sesja zapisana: 2026-01-11 14:08 UTC*
*Projekt: ViTParticleFilterTracker*
*Treningi: DETR ep295→500, YOLO ep200→500*
*Status: Joby w kolejce po naprawie błędu składni bash*
