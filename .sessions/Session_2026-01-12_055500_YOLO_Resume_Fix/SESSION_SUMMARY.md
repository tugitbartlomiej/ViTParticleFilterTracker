# Session Summary: YOLO_Resume_Fix

## Metadata
- **Date:** 2026-01-12
- **Time:** 05:55 CET
- **Status:** In Progress
- **Type:** SSH / Training

## Objective
Naprawić błąd YOLO training na Eden gdzie job 1509301 failed z błędem:
```
AssertionError: last.pt training to 200 epochs is finished, nothing to resume.
```

## Context
- YOLO model wytrenowany do epoch 200 (checkpoint `last.pt`)
- Próba wznowienia treningu do 500 epok (300 nowych)
- Ultralytics `resume=True` nie pozwala "wznowić" zakończonego treningu
- DETR job 1509211 działa równolegle na dgx-4

## Problem Analysis

### Root Cause
Skrypt `yolo_train_20k_finetune.py` używał:
```python
if self.resume and self.resume_path:
    train_args['resume'] = True  # BŁĄD!
```

Ultralytics `resume=True` działa TYLKO dla przerwanego treningu, nie dla zakończonego.

### Solution
Stworzono `yolo_train_20k_finetune_v2.py` który:
1. Używa `--pretrained_path` zamiast `--resume_path`
2. Ładuje wagi jako pretrained model
3. Uruchamia NOWY trening na 300 epok (bez `resume=True`)

## Actions Taken

1. **Analiza błędu z logu** - zidentyfikowano AssertionError w Ultralytics
2. **Przeszukano istniejące skrypty** - znaleziono brakujący `yolo_train_20k_finetune_v2.py`
3. **Stworzono poprawiony skrypt v2** - nowa logika pretrained weights
4. **Upload na Eden** - `scp` do `~/Yolo/20k_finetune/`
5. **Próby uruchomienia joba**:
   - dgx-3: DRAINED (klimatyzacja)
   - dgx-2: DRAINING (klimatyzacja)
   - hopper: za mało RAM (15 GiB / 200 GiB wymagane)
   - hopper z 64GB RAM: kolejka (Priority)
   - dgx-4 z 2 GPU: kolejka (Priority 950 vs ~100,000)

## Results

### Key Findings
- dgx-2, dgx-3 niedostępne z powodu awarii klimatyzacji
- hopper obciążony przez ~40 CPU jobów (hbaniecki)
- Fair-share priorytet bardzo niski (950) bo DETR działa 15h
- Job 1509626 w kolejce - czeka na zasoby

### Cluster Status
| Node | GPUs Free | RAM Free | Status |
|------|-----------|----------|--------|
| dgx-1 | 0/8 | 1366 GiB | brak GPU |
| dgx-2 | 8/8 | 608 GiB | DRAINING (AC) |
| dgx-3 | 8/8 | 1008 GiB | DRAINED (AC) |
| dgx-4 | 2/8 | 27 GiB | DETR running |
| hopper | 3/8 | 15 GiB | RAM occupied |
| hopper-2 | 8/8 | 2015 GiB | DOWN |

### Current Jobs
| Job ID | Model | Status | Node |
|--------|-------|--------|------|
| 1509626 | YOLO v2 | PENDING (Priority) | - |
| 1509211 | DETR | RUNNING 15h | dgx-4 |

## Files Generated/Modified

### Created
- `Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune_v2.py` - poprawiony skrypt

### On Eden
- `~/Yolo/20k_finetune/yolo_train_20k_finetune_v2.py` - uploaded

## Commands Used

```bash
# Upload skryptu
scp yolo_train_20k_finetune_v2.py eden-cluster:~/Yolo/20k_finetune/

# Próby uruchomienia
sbatch --nodelist=dgx-3 run_yolo_20k_finetune.slurm  # FAILED - drained
sbatch --nodelist=dgx-2 run_yolo_20k_finetune.slurm  # FAILED - draining
sbatch -p hopper run_yolo_20k_finetune.slurm         # PENDING - no RAM
sbatch -p hopper --mem=64G run_yolo_20k_finetune.slurm  # PENDING - priority
sbatch -p long --nodelist=dgx-4 --gres=gpu:2 --mem=64G run_yolo_20k_finetune.slurm  # PENDING

# Diagnostyka
sfree
sinfo -N -l | grep dgx
squeue -u bpiotrowski
sprio -p long
```

## Next Steps
- [ ] Czekać na zwolnienie zasobów (dgx-2/dgx-3 po naprawie AC)
- [ ] Lub anulować DETR aby podnieść priorytet YOLO
- [ ] Monitorować status: `squeue -u bpiotrowski`
- [ ] Po starcie: `tail -f ~/Yolo/20k_finetune/logs/yolo_20k_finetune_1509626.log`

## Key Fix (v1 vs v2)

```python
# v1 (BŁĘDNE)
if self.resume and self.resume_path:
    train_args['resume'] = True  # NIE DZIAŁA dla zakończonego ckpt

# v2 (POPRAWNE)
self.model = YOLO(pretrained_path)  # Załaduj wagi
# NIE ustawiaj resume=True
results = self.model.train(epochs=300, ...)  # Nowe 300 epok
```

---
*Session saved: 2026-01-12 05:55 CET*
