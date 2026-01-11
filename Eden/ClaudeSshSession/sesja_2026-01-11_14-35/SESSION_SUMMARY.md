# Sesja SSH Eden - 2026-01-11 14:35 UTC

**Data:** 2026-01-11
**Godzina:** 14:35 UTC (15:35 CET)
**Job IDs:** 1509211 (DETR - kontynuacja), 1509301 (YOLO - nowy po naprawie)
**Node:** dgx-4 (DETR running)
**Status:** DETR RUNNING, YOLO PENDING

---

## Podsumowanie Sesji

Naprawa błędu YOLO job 1509212 - batch_size=32 nie dzieliło się przez 3 GPU. Zmieniono na batch_size=30 i uruchomiono ponownie jako job 1509301.

### Kluczowe Osiągnięcia:

1. **Sprawdzono status jobów z poprzedniej sesji**
   - DETR 1509211 - RUNNING, epoch 296/500
   - YOLO 1509212 - FAILED po 39 sekundach

2. **Zidentyfikowano przyczynę błędu YOLO**
   - Błąd: `ValueError: 'batch=32' must be a multiple of GPU count 3`
   - Ultralytics wymaga batch % gpu_count == 0
   - Sugestia: użyć batch_size 30 lub 33

3. **Naprawiono skrypt YOLO**
   - `batch_size`: 32 → 30
   - Efektywny batch: 96 → 90
   - Naprawiono również line endings (CRLF → LF)

4. **Uruchomiono nowy job YOLO**
   - Nowy Job ID: 1509301
   - Status: PENDING (Resources)

---

## Struktura Katalogów na Eden

### DETR
```
/mnt/evafs/faculty/home/bpiotrowski/DETR/
├── ckpt_20k_finetune_v2_fixed/          # Working checkpoints
│   └── checkpoint_epoch_295.pth         # Resume point (teraz ep296+)
├── Checkpoints/20k_finetune_v2_fixed/   # Archiwum
│   └── tensorboard_logs/
├── train_out_20k_finetune/
│   └── logs/                            # TensorBoard runtime
├── logs/
│   └── detr_20k_4gpu_1509211.log        # Aktualny log
├── run_detr_20k_resume_4gpu.slurm
└── detr_train_optimized.py
```

### YOLO
```
/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/
├── train_out_5gpu_hopper/exp/weights/
│   ├── best.pt                          # 87 MB - najlepszy (mAP 99.5%)
│   └── last.pt                          # 87 MB - epoch 200 (resume point)
├── run_yolo_20k_resume_500.slurm        # NAPRAWIONY (batch 30)
├── yolo_train_20k_finetune.py
├── logs/
│   ├── yolo_20k_resume500_1509212.log   # Stary (FAILED)
│   └── yolo_20k_resume500_1509301.log   # Nowy (aktualny)
├── train_out_500/                       # Output dla nowego treningu
├── ckpt_500/
└── best_500/
```

---

## Błąd YOLO i Naprawa

### Problem
```
ValueError: 'batch=32' must be a multiple of GPU count 3. 
Try 'batch=30' or 'batch=33', the nearest batch sizes evenly divisible by 3.
```

### Przyczyna
Ultralytics YOLO wymaga żeby batch_size było podzielne przez liczbę GPU przy treningu multi-GPU.

### Rozwiązanie
```bash
# PRZED (błędne):
--batch_size 32   # 32 % 3 = 2 ≠ 0

# PO (poprawne):
--batch_size 30   # 30 % 3 = 0 ✓
```

### Dodatkowy problem: Line Endings
```bash
sbatch: error: Batch script contains DOS line breaks (\r\n)
# Rozwiązanie:
sed -i 's/\r$//' run_yolo_20k_resume_500.slurm
```

---

## Konfiguracja Treningu

### DETR (Job 1509211) - BEZ ZMIAN
```bash
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --time=5-00:00:00

torchrun --nproc_per_node=3 detr_train_optimized.py \
    --epochs 500 \
    --batch_size 4 \
    --lr 5e-5 \
    --lr_backbone 5e-6 \
    --lr_scheduler cosine \
    --resume_training
```

### YOLO (Job 1509301) - NAPRAWIONY
```bash
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --time=5-00:00:00

python3 yolo_train_20k_finetune.py \
    --epochs 500 \
    --batch_size 30        # ZMIENIONO z 32
    --learning_rate 0.001 \
    --save_period 10 \
    --patience 50 \
    --resume --resume_path last.pt
```

---

## Ważne Ścieżki

| Ścieżka | Opis |
|---------|------|
| `~/DETR/logs/detr_20k_4gpu_1509211.log` | DETR aktualny log |
| `~/Yolo/20k_finetune/logs/yolo_20k_resume500_1509301.log` | YOLO nowy log |
| `~/Yolo/20k_finetune/run_yolo_20k_resume_500.slurm` | Naprawiony skrypt |
| `~/20kSelectedImages.tar` | Dataset archiwum |
| `~/merged_20k_annotations_fixed.json` | Adnotacje COCO |

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
ssh eden-cluster "tail -f ~/Yolo/20k_finetune/logs/yolo_20k_resume500_1509301.log"
```

### Dostępność GPU
```bash
ssh eden-cluster "sfree"
```

---

## Status na Koniec Sesji

### Joby
| Job ID | Model | Status | Epochs | Node |
|--------|-------|--------|--------|------|
| 1509211 | DETR | ✅ RUNNING | 296→500 | dgx-4 |
| 1509301 | YOLO | ⏳ PENDING | 200→500 | (Resources) |

### DETR Progress
- Aktualnie: Epoch 296/500
- Runtime: ~16 minut
- Loss: ~0.07-0.18

### Następne kroki
- [ ] Monitorować start YOLO job 1509301
- [ ] Sprawdzić czy batch_size=30 działa poprawnie
- [ ] Kontynuować monitoring DETR
- [ ] Po zakończeniu: benchmark DETR ep500 vs YOLO ep500

---

## Pliki Zmodyfikowane

### run_yolo_20k_resume_500.slurm
- Lokalizacja: `~/Yolo/20k_finetune/run_yolo_20k_resume_500.slurm`
- Zmiany:
  - `--batch_size 32` → `--batch_size 30`
  - `echo "Batch size: 32/GPU"` → `echo "Batch size: 30/GPU"`
  - Line endings: CRLF → LF (via sed)

---

*Sesja zapisana: 2026-01-11 14:35 UTC*
*Projekt: ViTParticleFilterTracker*
*Treningi: DETR ep296→500 (running), YOLO ep200→500 (pending po naprawie)*
*Status: YOLO naprawiony - batch_size 32→30*
