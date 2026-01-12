# Sesja SSH Eden - 2026-01-11 16:15 UTC

**Data:** 2026-01-11
**Godzina:** 16:15 UTC (17:15 CET)
**Job IDs:** 1509211 (DETR - running), 1509486 (YOLO - pending)
**Status:** DETR RUNNING, YOLO PENDING (poprawiony)

---

## Podsumowanie Sesji

Naprawa błędu YOLO fine-tuning - poprawna strategia kontynuacji treningu analogiczna do DETR.

### Problem Pierwotny (z poprzedniej sesji)

Job YOLO 1509301 failed z błędem:
```
AssertionError: last.pt training to 200 epochs is finished, nothing to resume.
```

Ultralytics YOLO nie pozwala na `resume=True` gdy trening zakończył się normalnie (200/200 epok).

### Rozwiązanie

**Strategia fine-tuning (analogiczna do DETR):**

| Model | Oryginalny trening | Fine-tune na 20k | Razem |
|-------|-------------------|------------------|-------|
| DETR  | epoch 170         | +330 epok        | 500   |
| YOLO  | epoch 200         | +300 epok        | 500   |

**Kluczowa zmiana:**
- NIE używać `--resume` (Ultralytics resume)
- Użyć `--pretrained_path` do załadowania weights
- Ustawić `--epochs 300` (nie 500!)
- Efektywnie: 200 + 300 = 500 epok

---

## Nowe Pliki na Eden

### yolo_train_20k_finetune_v2.py
```
~/Yolo/20k_finetune/yolo_train_20k_finetune_v2.py
```
Poprawiony skrypt z rozdzieleniem:
- `--pretrained_path`: ładuje weights (bez resume)
- `--resume`: tylko dla przerwanego treningu

### run_yolo_20k_finetune.slurm
```
~/Yolo/20k_finetune/run_yolo_20k_finetune.slurm
```
Poprawiony SLURM:
- `--epochs 300` (nie 500)
- `--pretrained_path $PRETRAINED_WEIGHTS`
- Bez `--resume`

---

## Konfiguracja Treningu

### YOLO Fine-tune (Job 1509486)
```bash
python3 yolo_train_20k_finetune_v2.py \
    --dataset_yaml_path $DATASET_YAML \
    --project_dir $PROJECT_ROOT/train_out_20k_finetune \
    --checkpoint_dir $PROJECT_ROOT/ckpt_20k_finetune \
    --best_model_dir $PROJECT_ROOT/best_20k_finetune \
    --model_size m \
    --epochs 300 \                    # 300 nowych epok
    --batch_size 30 \                 # 30 * 3 GPU = 90 effective
    --learning_rate 0.001 \           # Reduced for fine-tuning
    --pretrained_path $PRETRAINED_WEIGHTS  # epoch 200 weights
```

### DETR (Job 1509211) - bez zmian
```bash
torchrun --nproc_per_node=3 detr_train_optimized.py \
    --epochs 500 \
    --resume_training
```

---

## Struktura Katalogów

### YOLO
```
~/Yolo/20k_finetune/
├── train_out_5gpu_hopper/exp/weights/
│   ├── best.pt                     # Najlepszy z orig. treningu
│   └── last.pt                     # Epoch 200 (pretrained)
├── train_out_20k_finetune/         # NOWY output fine-tune
├── ckpt_20k_finetune/              # Checkpointy fine-tune
├── best_20k_finetune/              # Best model fine-tune
├── yolo_train_20k_finetune_v2.py   # NOWY poprawiony skrypt
├── run_yolo_20k_finetune.slurm     # NOWY poprawiony SLURM
└── logs/
    └── yolo_20k_finetune_1509486.log
```

### DETR
```
~/DETR/
├── ckpt_20k_finetune_v2_fixed/
│   └── checkpoint_epoch_*.pth
└── logs/
    └── detr_20k_4gpu_1509211.log
```

---

## Status Jobów

| Job ID | Model | Status | Strategia | Node |
|--------|-------|--------|-----------|------|
| 1509211 | DETR | ✅ RUNNING (5h) | ep170→500 | dgx-4 |
| 1509486 | YOLO | ⏳ PENDING | ep200→500 (300 new) | - |

---

## Komendy SSH

### Status kolejki
```bash
ssh eden-cluster "squeue -u bpiotrowski"
```

### Logi
```bash
# DETR
ssh eden-cluster "tail -f ~/DETR/logs/detr_20k_4gpu_1509211.log"

# YOLO (gdy ruszy)
ssh eden-cluster "tail -f ~/Yolo/20k_finetune/logs/yolo_20k_finetune_1509486.log"
```

### GPU availability
```bash
ssh eden-cluster "sfree"
```

---

## Historia Błędów i Napraw

### Błąd 1: batch_size (Job 1509212)
```
ValueError: 'batch=32' must be a multiple of GPU count 3
```
**Fix:** batch_size 32 → 30

### Błąd 2: resume logic (Job 1509301)
```
AssertionError: training to 200 epochs is finished, nothing to resume
```
**Fix:**
- Usunąć `--resume`
- Dodać `--pretrained_path`
- Zmienić epochs 500 → 300

### Błąd 3: wrong epochs (Job 1509485) - anulowany
**Fix:** epochs 500 → 300 (bo 200 już zrobione)

---

## Następne Kroki

- [ ] Monitorować start YOLO job 1509486
- [ ] Sprawdzić czy fine-tuning działa poprawnie
- [ ] Po zakończeniu DETR i YOLO: benchmark porównawczy
- [ ] Sprawdzić metryki: mAP, loss na 20k dataset

---

## Ważne Informacje

### Różnica między resume a pretrained
- `--resume`: Kontynuuje przerwany trening (wymaga niezakończonego checkpointu)
- `--pretrained_path`: Ładuje weights jako punkt startowy dla NOWEGO treningu

### Efektywna liczba epok
- YOLO: 200 (original) + 300 (fine-tune) = **500 total**
- DETR: 170 (original) + 330 (fine-tune) = **500 total**

---

*Sesja zapisana: 2026-01-11 16:15 UTC*
*Projekt: ViTParticleFilterTracker*
*Status: YOLO fine-tuning poprawiony i uruchomiony*
