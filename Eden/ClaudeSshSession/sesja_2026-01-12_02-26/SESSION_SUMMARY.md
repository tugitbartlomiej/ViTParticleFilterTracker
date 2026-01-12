# Sesja SSH Eden - 2026-01-12 02:26 UTC

**Data:** 2026-01-12
**Godzina:** 02:26 UTC (03:26 CET)
**Job IDs:** 1509211 (DETR), 1509486 (YOLO)
**Status:** Monitorowanie (brak bezpośredniego SSH w tej sesji)

---

## Podsumowanie Sesji

Sesja dokumentacyjna - kontynuacja monitorowania jobów z poprzednich sesji. Brak bezpośredniego połączenia SSH. Główna praca w tej sesji dotyczyła weryfikacji artykułu IEEE ACCESS względem danych z projektu.

---

## Status Jobów (z poprzednich sesji)

| Job ID | Model | Status | Strategia | Node |
|--------|-------|--------|-----------|------|
| 1509211 | DETR | RUNNING | ep170→500 | dgx-4 |
| 1509486 | YOLO | PENDING/RUNNING | ep200→500 (300 new) | - |

---

## Struktura Katalogów na Eden

### DETR
```
~/DETR/
├── ckpt_20k_finetune_v2_fixed/
│   └── checkpoint_epoch_*.pth
├── train_out_20k_finetune/
│   └── logs/
└── logs/
    └── detr_20k_4gpu_1509211.log
```

### YOLO
```
~/Yolo/20k_finetune/
├── train_out_5gpu_hopper/exp/weights/
│   ├── best.pt
│   └── last.pt  (epoch 200 - pretrained)
├── train_out_20k_finetune/
├── ckpt_20k_finetune/
├── best_20k_finetune/
├── yolo_train_20k_finetune_v2.py
├── run_yolo_20k_finetune.slurm
└── logs/
    └── yolo_20k_finetune_1509486.log
```

---

## Konfiguracja Treningu

### DETR Job 1509211
```bash
torchrun --nproc_per_node=3 detr_train_optimized.py \
    --epochs 500 \
    --resume_training
```

### YOLO Job 1509486
```bash
python3 yolo_train_20k_finetune_v2.py \
    --epochs 300 \
    --batch_size 30 \
    --learning_rate 0.001 \
    --pretrained_path $PRETRAINED_WEIGHTS
```

---

## Komendy SSH do Monitorowania

```bash
# Status kolejki
ssh eden-cluster "squeue -u bpiotrowski"

# Logi DETR
ssh eden-cluster "tail -f ~/DETR/logs/detr_20k_4gpu_1509211.log"

# Logi YOLO
ssh eden-cluster "tail -f ~/Yolo/20k_finetune/logs/yolo_20k_finetune_1509486.log"

# GPU availability
ssh eden-cluster "sfree"
```

---

## Weryfikacja Danych z Projektu

W tej sesji zweryfikowano dane z benchmarków lokalnych:

### Fourier Features (`fourier_features.pkl`)
- 90,389 obrazów
- High/Low Ratio: min=4.75, max=28.83, mean=8.39
- **Brak obrazów poniżej R < 4.0** (0%)

### Benchmark DETR vs YOLO (`BENCHMARK_Q81_20251231_013537`)
- DETR ep170: mAP=74.75%, F1=80.2%
- YOLO ep70: mAP=64.51%, F1=74.8%
- Różnica mAP: +10.24pp
- Cross-dataset: 357 vs 127 TP (2.81×)

---

## Powiązane Sesje

- `sesja_2026-01-11_16-15` - Uruchomienie jobów, fix YOLO resume
- `sesja_2026-01-12_01-13` - Poprzednia dokumentacja

---

## Następne Kroki

- [ ] Połączyć się przez SSH i sprawdzić aktualny status jobów
- [ ] Sprawdzić postęp DETR (current epoch, loss)
- [ ] Sprawdzić czy YOLO już wystartował
- [ ] Po zakończeniu: benchmark porównawczy na 500 epokach

---

*Sesja zapisana: 2026-01-12 02:26 UTC*
*Projekt: ViTParticleFilterTracker*
*Status: Dokumentacja monitorowania*
