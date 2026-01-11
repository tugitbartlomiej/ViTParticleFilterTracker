# Sesja SSH Eden - 2026-01-11 01:30 UTC

**Data:** 2026-01-11
**Godzina:** 01:30 UTC (02:30 CET)
**Job IDs:** 1509138 (DETR), 1509140 (YOLO)
**Node:** dgx-2 (DETR), dgx-4 (YOLO) - zaplanowane
**Status:** QUEUED (oba joby w kolejce)

---

## Podsumowanie Sesji

Naprawa i ponowne uruchomienie treningu DETR oraz YOLO na 20k dataset do epoch 500.

### Kluczowe Osiagniecia:

1. **Zidentyfikowano problem z poprzednim jobem DETR (1505719)**
   - Skrypt `run_detr_20k_resume_295.slurm` uzywal ZLYCH argumentow:
     - `--coco_path` zamiast `--images_dir`
     - brak `--annotations_path` (WYMAGANY!)
     - `--resume <path>` zamiast `--resume_training`
     - `--save_every` zamiast `--save_interval`

2. **Naprawiono skrypt DETR** - `run_detr_20k_resume_4gpu.slurm`
   - Poprawne argumenty zgodne z `detr_train_optimized.py`
   - TensorBoard logi synchronizowane co 10 min do archiwum

3. **Przeanalizowano trening YOLO**
   - Odkryto ze YOLO jest na epoch 200 (nie 300!)
   - Utworzono `run_yolo_20k_resume_500.slurm` do kontynuacji

4. **Zidentyfikowano duplikaty checkpointow DETR**
   - `ckpt_20k_finetune_v2_fixed/` - working dir (27 ckpt, do ep295)
   - `Checkpoints/20k_finetune_v2_fixed/` - archiwum (25 ckpt, do ep290)
   - ~25GB duplikatow

---

## Struktura Katalogow na Eden

### DETR
```
/mnt/evafs/faculty/home/bpiotrowski/DETR/
├── ckpt_20k_finetune_v2_fixed/          # Working checkpoints (13GB)
│   ├── checkpoint_epoch_170.pth         # Start point
│   ├── checkpoint_epoch_175.pth
│   ├── ...
│   └── checkpoint_epoch_295.pth         # Ostatni (Jan 3)
├── Checkpoints/20k_finetune_v2_fixed/   # Archiwum (12GB)
│   ├── checkpoint_epoch_170.pth
│   ├── ...
│   ├── checkpoint_epoch_290.pth         # Brak ep295!
│   └── tensorboard_logs/
├── train_out_20k_finetune/
│   └── logs/                            # TensorBoard runtime
├── best_20k_finetune/
├── run_detr_20k_resume_4gpu.slurm       # NOWY - naprawiony skrypt
├── run_detr_20k_resume_500.slurm        # Wersja dla hopper
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
├── run_yolo_20k_resume_500.slurm        # NOWY - kontynuacja do 500
├── yolo_train_20k_finetune.py
├── train_out_500/                       # Output dla nowego treningu
├── ckpt_500/
└── best_500/
```

---

## Konfiguracja Treningu

### DETR (Job 1509138)
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

### YOLO (Job 1509140)
```bash
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --time=5-00:00:00

python3 yolo_train_20k_finetune.py \
    --epochs 500 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --save_period 10 \
    --patience 50 \
    --resume --resume_path last.pt
```

---

## Wazne Sciezki

| Sciezka | Opis |
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

### Logi (gdy joby rusza)
```bash
# DETR
ssh eden-cluster "tail -f ~/DETR/logs/detr_20k_4gpu_1509138.log"

# YOLO
ssh eden-cluster "tail -f ~/Yolo/20k_finetune/logs/yolo_20k_resume500_1509140.log"
```

### TensorBoard
```bash
# DETR logs
ssh eden-cluster "ls -la ~/DETR/train_out_20k_finetune/logs/"
ssh eden-cluster "ls -la ~/DETR/Checkpoints/20k_finetune_v2_fixed/tensorboard_logs/"
```

### Checkpointy
```bash
ssh eden-cluster "ls -la ~/DETR/ckpt_20k_finetune_v2_fixed/ | tail -5"
ssh eden-cluster "ls -la ~/Yolo/20k_finetune/train_out_5gpu_hopper/exp/weights/"
```

---

## Problemy i Rozwiazania

### Problem 1: DETR job 1505719 FAILED
**Przyczyna:** Skrypt `run_detr_20k_resume_295.slurm` uzywal zlych argumentow:
```bash
# ZLE:
--coco_path "$TMPDIR/20k_dataset"    # Nie istnieje!
--resume "$RESUME_CHECKPOINT"         # Nie istnieje!
--save_every 5                        # Nie istnieje!

# POPRAWNIE:
--images_dir "$IMAGES_DIR"
--annotations_path "$ANNOT_JSON"
--resume_training
--save_interval 5
```
**Rozwiazanie:** Utworzono nowy skrypt `run_detr_20k_resume_4gpu.slurm` z poprawnymi argumentami

### Problem 2: YOLO epoch confusion
**Przyczyna:** SESSION_SUMMARY mowil o epoch 300, ale checkpoint `last.pt` ma `epoch: 200`
**Rozwiazanie:** Sprawdzono metadane checkpointu - faktyczny stan to epoch 200

### Problem 3: Brak dostepnych GPU
**Przyczyna:** dgx-1, dgx-3 w stanie `drng` (draining), hopper-2 niedostepny
**Rozwiazanie:** Uzycie partycji `long` bez nodelist - SLURM sam wybierze dostepny node

---

## Status na Koniec Sesji

### Joby w kolejce
| Job ID | Model | Epochs | GPUs | Est. Start | Node |
|--------|-------|--------|------|------------|------|
| 1509138 | DETR | 295->500 | 3 | 05:36 | dgx-2 |
| 1509140 | YOLO | 200->500 | 3 | 13:05 | dgx-4 |

### Stan klastra (sfree)
```
Node      Free GPUs
dgx-1         3 / 8    (draining)
dgx-2         0 / 8
dgx-3         3 / 8    (draining)
dgx-4         0 / 8
hopper        2 / 8
hopper-2      8 / 8    (unavailable)
```

### Nastepne kroki
- [ ] Monitorowac start jobow
- [ ] Sprawdzic TensorBoard logi po starcie treningu
- [ ] Po zakonczeniu: benchmark DETR ep500 vs YOLO ep500

---

## Pliki Skryptow

### run_detr_20k_resume_4gpu.slurm (NOWY)
Lokalizacja: `/mnt/evafs/faculty/home/bpiotrowski/DETR/run_detr_20k_resume_4gpu.slurm`
- 3 GPU, partycja long
- Resume z ep295, target 500
- TensorBoard sync co 10 min

### run_yolo_20k_resume_500.slurm (NOWY)
Lokalizacja: `/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/run_yolo_20k_resume_500.slurm`
- 3 GPU, partycja long
- Resume z last.pt (ep200), target 500
- Output: train_out_500/

---

*Sesja zapisana: 2026-01-11 01:30 UTC*
*Projekt: ViTParticleFilterTracker*
*Treningi: DETR 20k ep295->500, YOLO 20k ep200->500*
