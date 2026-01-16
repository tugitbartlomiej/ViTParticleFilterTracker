# Sesja SSH Eden - 2026-01-12 05:10 UTC

**Data:** 2026-01-12
**Godzina:** 05:10 UTC (06:10 CET)
**Job IDs:** 1509211 (DETR), 1509798 (YOLO)
**Status:** DETR RUNNING, YOLO PENDING

---

## Podsumowanie Sesji

Naprawa błędu YOLO training (`resume=True` nie działa dla zakończonego checkpointu). Stworzono `yolo_train_20k_finetune_v2.py` z poprawioną logiką pretrained weights. Diagnoza problemów z klastrem: dgx-2/dgx-3 niedostępne (awaria klimatyzacji), niski priorytet fair-share.

---

## Status Jobów

| Job ID | Model | Status | Epoch | Node | GPU |
|--------|-------|--------|-------|------|-----|
| 1509211 | DETR | RUNNING (27h+) | 333-334/500 | dgx-4 | 3×A100 |
| 1509798 | YOLO v2 | PENDING (Priority 950) | - | - | 3 requested |

---

## Status Klastra

| Node | Free GPUs | Free RAM | Status |
|------|-----------|----------|--------|
| dgx-1 | 0/8 | 1146 GiB | brak GPU |
| dgx-2 | 8/8 | 1008 GiB | **DRAINED** (klimatyzacja) |
| dgx-3 | 8/8 | 1008 GiB | **DRAINED** (klimatyzacja) |
| dgx-4 | 1/8 | 5 GiB | DETR + 7 innych jobów |
| hopper | 7/8 | 4 GiB | ~55 CPU jobs (hbaniecki) |
| hopper-2 | 8/8 | 2015 GiB | **DOWN** |

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
├── yolo_train_20k_finetune.py     # v1 - z bugiem resume
├── yolo_train_20k_finetune_v2.py  # v2 - POPRAWIONY
├── run_yolo_20k_finetune.slurm
└── logs/
```

---

## Problem i Rozwiązanie

### Problem
```
AssertionError: last.pt training to 200 epochs is finished, nothing to resume.
```

### Przyczyna
Skrypt v1 używał `resume=True` dla zakończonego checkpointu (epoch 200).
Ultralytics odmawia "wznowienia" ukończonego treningu.

### Rozwiązanie
Stworzono `yolo_train_20k_finetune_v2.py`:
```python
# v1 (BŁĘDNE)
if self.resume and self.resume_path:
    train_args['resume'] = True  # NIE DZIAŁA!

# v2 (POPRAWNE)
self.model = YOLO(pretrained_path)  # Załaduj wagi
# NIE ustawiaj resume=True
results = self.model.train(epochs=300, ...)  # Nowe 300 epok
```

---

## Komendy SSH

```bash
# Status kolejki
ssh eden-cluster "squeue -u bpiotrowski"

# Logi DETR
ssh eden-cluster "tail -f ~/DETR/logs/detr_20k_4gpu_1509211.log"

# Logi YOLO (gdy wystartuje)
ssh eden-cluster "tail -f ~/Yolo/20k_finetune/logs/yolo_20k_finetune_1509798.log"

# Status klastra
ssh eden-cluster "sfree"
ssh eden-cluster "sinfo -N -l | grep -E 'dgx|hopper'"

# Priorytety
ssh eden-cluster "sprio -p hopper"
```

---

## Problem z Priorytetem

| User | Priorytet |
|------|-----------|
| hbaniecki | 147,702 |
| bsobieski | 147,068 |
| **bpiotrowski** | **~950** |

**Przyczyna:** DETR działa 27+ godzin → fair-share karze za długotrwałe użycie zasobów.

**Rozwiązanie:** Poczekać aż DETR skończy (~166 epok, ~12-14h), wtedy priorytet się odnowi.

---

## Pliki Utworzone/Zmodyfikowane

### Lokalnie
- `Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune_v2.py`

### Na Eden
- `~/Yolo/20k_finetune/yolo_train_20k_finetune_v2.py`

---

## Następne Kroki

- [ ] Monitorować DETR (epoch 333→500, ~12-14h)
- [ ] Po zakończeniu DETR: priorytet wzrośnie
- [ ] YOLO powinien wystartować automatycznie
- [ ] Sprawdzić: `ssh eden-cluster "squeue -u bpiotrowski"`

---

*Sesja zapisana: 2026-01-12 05:10 UTC*
*Projekt: ViTParticleFilterTracker*
