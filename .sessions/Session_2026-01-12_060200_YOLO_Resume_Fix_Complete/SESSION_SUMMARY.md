# Session Summary: YOLO_Resume_Fix_Complete

## Metadata
- **Date:** 2026-01-12
- **Time:** 06:02 CET
- **Status:** Completed
- **Type:** SSH / Training / Tools

## Objective
1. Naprawić błąd YOLO training na Eden (AssertionError: nothing to resume)
2. Uruchomić trening YOLO 300 nowych epok (200→500)
3. Zaktualizować indeks klas projektu

## Context
- YOLO checkpoint `last.pt` wytrenowany do epoch 200
- Próba wznowienia treningu do 500 epok zakończyła się błędem
- Ultralytics `resume=True` nie działa dla zakończonego treningu
- DETR job 1509211 działa równolegle na dgx-4 (15h+)

## Actions Taken

### 1. Analiza błędu (z logu użytkownika)
```
AssertionError: last.pt training to 200 epochs is finished, nothing to resume.
```

### 2. Identyfikacja przyczyny
- Skrypt v1 używał `resume=True` dla zakończonego checkpointu
- Ultralytics odmawia "wznowienia" ukończonego treningu

### 3. Stworzenie poprawionego skryptu v2
- `yolo_train_20k_finetune_v2.py`
- Używa `--pretrained_path` zamiast `--resume_path`
- NIE ustawia `resume=True` - ładuje wagi i trenuje od nowa

### 4. Upload na Eden
```bash
scp yolo_train_20k_finetune_v2.py eden-cluster:~/Yolo/20k_finetune/
```

### 5. Próby uruchomienia joba
| Node | Wynik | Przyczyna |
|------|-------|-----------|
| dgx-3 | FAILED | DRAINED (klimatyzacja) |
| dgx-2 | FAILED | DRAINING (klimatyzacja) |
| hopper 200GB | PENDING | Za mało RAM (15 GiB wolne) |
| hopper 64GB | PENDING | Priority (950 vs 100,000+) |
| dgx-4 2GPU | PENDING | Priority |

### 6. Commit
```
d053b348 fix(training): add YOLO finetune v2 script with pretrained weights logic
```

### 7. Aktualizacja indeksu Serena MCP
- Dodano `YOLOFineTunerV2` do class_index

## Results

### Key Findings

#### Problem z klasorem Eden
| Node | Status | Uwagi |
|------|--------|-------|
| dgx-2 | DRAINING | awaria klimatyzacji |
| dgx-3 | DRAINED | awaria klimatyzacji |
| dgx-4 | RUNNING | DETR job, 2 GPU wolne |
| hopper | MIXED | 3 GPU, ale tylko 15 GiB RAM |
| hopper-2 | DOWN | problem z GPU |

#### Fair-share Priority
| Job | Priorytet |
|-----|-----------|
| pgarbacz | 99,207 |
| hbaniecki | 147,702 |
| **bpiotrowski (YOLO)** | **950** |

DETR działa 15h+ co obniża priorytet nowych jobów.

### Issues Encountered
1. `resume=True` nie działa dla zakończonych checkpointów - ROZWIĄZANE (v2)
2. dgx-2/dgx-3 niedostępne - NIEROZWIĄZANE (awaria AC)
3. Niski priorytet - NIEROZWIĄZANE (czekać lub anulować DETR)

## Files Generated/Modified

### Created
- `Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune_v2.py`
- `.sessions/Session_2026-01-12_055500_YOLO_Resume_Fix/`
- `.sessions/Session_2026-01-12_060200_YOLO_Resume_Fix_Complete/`

### On Eden
- `~/Yolo/20k_finetune/yolo_train_20k_finetune_v2.py`

### Updated
- Serena MCP memory: `class_index`

## Commands Used

### SSH/SCP
```bash
scp yolo_train_20k_finetune_v2.py eden-cluster:~/Yolo/20k_finetune/
ssh eden-cluster "sfree"
ssh eden-cluster "squeue -u bpiotrowski"
ssh eden-cluster "sinfo -N -l | grep dgx"
ssh eden-cluster "sprio -p long"
```

### SLURM
```bash
sbatch -p hopper --mem=64G run_yolo_20k_finetune.slurm
sbatch -p long --nodelist=dgx-4 --gres=gpu:2 --mem=64G run_yolo_20k_finetune.slurm
scancel 1509624 1509625
```

### Git
```bash
git add Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune_v2.py
git commit -m "fix(training): add YOLO finetune v2 script..."
```

### Serena MCP
```
mcp__plugin_serena_serena__activate_project
mcp__plugin_serena_serena__get_symbols_overview
mcp__plugin_serena_serena__write_memory (class_index)
```

## Current Jobs on Eden

| Job ID | Model | Status | Node |
|--------|-------|--------|------|
| 1509626 | YOLO v2 | PENDING (Priority 950) | - |
| 1509211 | DETR | RUNNING 15h | dgx-4 |

## Next Steps
- [ ] Czekać na zwolnienie zasobów (dgx-2/dgx-3 po naprawie AC)
- [ ] LUB anulować DETR aby podnieść priorytet YOLO
- [ ] Monitorować: `ssh eden-cluster "squeue -u bpiotrowski"`
- [ ] Po starcie YOLO: `tail -f ~/Yolo/20k_finetune/logs/yolo_20k_finetune_1509626.log`

## Key Fix (v1 vs v2)

```python
# v1 (BŁĘDNE) - resume zakończonego treningu
if self.resume and self.resume_path:
    train_args['resume'] = True  # BŁĄD! Checkpoint ukończony

# v2 (POPRAWNE) - pretrained weights, nowy trening
self.model = YOLO(pretrained_path)  # Załaduj wagi
# NIE ustawiaj resume=True
results = self.model.train(epochs=300, ...)  # 300 NOWYCH epok
```

---

*Session completed: 2026-01-12 06:02 CET*
*Commit: d053b348*
*Project: ViTParticleFilterTracker*
