# Sesja SSH Eden - 2026-01-12 01:13 UTC

**Data:** 2026-01-12
**Godzina:** 01:13 UTC (02:13 CET)
**Job IDs:** 1509211 (DETR), 1509486 (YOLO)
**Status:** Kontynuacja monitorowania z poprzedniej sesji

---

## Podsumowanie Sesji

Sesja dokumentacyjna - brak bezpośredniego połączenia SSH. Kontynuacja monitorowania jobów uruchomionych w sesji `sesja_2026-01-11_16-15`.

### Status Jobów (z poprzedniej sesji)

| Job ID | Model | Status | Strategia | Oczekiwany czas |
|--------|-------|--------|-----------|-----------------|
| 1509211 | DETR | RUNNING | ep170→500 | ~24-36h |
| 1509486 | YOLO | PENDING→RUNNING | ep200→500 (300 new) | ~12-18h |

---

## Konfiguracja Treningu (przypomnienie)

### DETR Job 1509211
```bash
torchrun --nproc_per_node=3 detr_train_optimized.py \
    --epochs 500 \
    --resume_training
```
- **Punkt startowy:** epoch 170
- **Cel:** epoch 500
- **Node:** dgx-4

### YOLO Job 1509486
```bash
python3 yolo_train_20k_finetune_v2.py \
    --epochs 300 \
    --batch_size 30 \
    --learning_rate 0.001 \
    --pretrained_path $PRETRAINED_WEIGHTS
```
- **Punkt startowy:** epoch 200 weights (bez resume)
- **Cel:** 300 nowych epok (efektywnie 500 total)
- **Fix z poprzedniej sesji:** Użycie `--pretrained_path` zamiast `--resume`

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
├── train_out_20k_finetune/  (output fine-tune)
├── ckpt_20k_finetune/
├── best_20k_finetune/
├── yolo_train_20k_finetune_v2.py
├── run_yolo_20k_finetune.slurm
└── logs/
    └── yolo_20k_finetune_1509486.log
```

---

## Komendy SSH do Monitorowania

### Status kolejki
```bash
ssh eden-cluster "squeue -u bpiotrowski"
```

### Logi DETR
```bash
ssh eden-cluster "tail -f ~/DETR/logs/detr_20k_4gpu_1509211.log"
```

### Logi YOLO
```bash
ssh eden-cluster "tail -f ~/Yolo/20k_finetune/logs/yolo_20k_finetune_1509486.log"
```

### GPU availability
```bash
ssh eden-cluster "sfree"
```

---

## Powiązane Sesje

- `sesja_2026-01-11_16-15` - Uruchomienie jobów, fix YOLO resume
- `sesja_2026-01-11_14-35` - Poprzednie próby uruchomienia
- `sesja_2026-01-11_01-30` - Konfiguracja DETR/YOLO resume

---

## Następne Kroki

- [ ] Sprawdzić status jobów przez SSH
- [ ] Monitorować postęp treningu (epoch, loss)
- [ ] Po zakończeniu: uruchomić benchmark porównawczy
- [ ] Aktualizować artykuł IEEE ACCESS o wyniki 500 epoch

---

## Notatki

Ta sesja jest głównie dokumentacyjna. Poprzednia sesja `sesja_2026-01-11_16-15` zawiera pełne szczegóły konfiguracji i naprawy błędu YOLO resume.

Kluczowa różnica między DETR a YOLO w kontekście resume:
- **DETR:** Używa `--resume_training` z własnym checkpoint systemem
- **YOLO (Ultralytics):** Nie pozwala na resume zakończonego treningu - wymaga `--pretrained_path`

---

*Sesja zapisana: 2026-01-12 01:13 UTC*
*Projekt: ViTParticleFilterTracker*
*Status: Dokumentacja kontynuacji monitorowania*
