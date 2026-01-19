# Session Summary: YOLO LR Fix & Fine-tune ep170-500

## Metadata
- **Date:** 2026-01-19
- **Time:** ~03:00 - 04:20 UTC
- **Status:** Active (Job 1523692 Running)
- **Type:** Training / SSH / Bug Fix
- **Job ID:** 1523692
- **Partition:** hopper (5 day limit)
- **GPU:** 4x NVIDIA H100 PCIe (81GB each)

## Objective
Naprawić problem z Learning Rate przy fine-tuningu YOLO i uruchomić trening od epoch 170 do 500.

## Context
Użytkownik zauważył że przy resume treningu YOLO Learning Rate może być źle ustawiony. Poprzedni trening (ep0-200) używał lr0=0.01 z decay do ~0.0001. Nowy trening powinien kontynuować z odpowiednim LR.

## Actions Taken

### 1. Analiza problemu LR
- Sprawdzono logi poprzednich sesji i checkpointów
- Odkryto że epoch 70 miał LR=0.006535, epoch 100 miał LR=0.00505
- Końcowy LR na epoch 200 ≈ 0.0001

### 2. Wykrycie krytycznego bugu
- **Problem:** `optimizer=auto` w Ultralytics IGNORUJE parametry `lr0` i `momentum`
- W logach widoczne: `optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0001'`
- Ultralytics auto wybierał lr=0.01 zamiast ustawionego 0.0001!

### 3. Naprawa skryptu Python
- Dodano `'optimizer': 'SGD'` do train_args (linia 189)
- Dodano parametr `--lrf` (default 0.1)
- Dodano `'momentum': 0.937` explicit

### 4. Konfiguracja treningu
- Start z epoch170.pt (nie epoch200)
- LR = 0.001 (fine-tune rate, 10x mniejszy niż oryginalny)
- lrf = 0.1 (final LR = 0.0001)
- 330 nowych epok (170 + 330 = 500 total)

### 5. Uruchomienie na Eden
- Kilka prób z różnymi partycjami (short, long, hopper)
- Finalnie: partycja hopper, 4x H100 GPU
- Job 1523692 uruchomiony i działa

## Results

### Key Findings
1. **Ultralytics bug:** `optimizer=auto` ignoruje `lr0` i `momentum` - MUSI być explicit `optimizer='SGD'`
2. **LR Schedule:** Oryginalny trening: 0.01→0.0001, Fine-tune: 0.001→0.0001
3. **Checkpoint:** epoch170.pt znajduje się w `~/Yolo/20k_finetune/` (root, nie w weights/)
4. **sfree vs sinfo:** `sfree` pokazuje lepsze info o dostępnych zasobach niż `sinfo`

### Issues Encountered

| Problem | Rozwiązanie |
|---------|-------------|
| `optimizer=auto` ignoruje lr0 | Użyj explicit `optimizer='SGD'` |
| sinfo pokazuje nodes jako "down" | Użyj `sfree` do sprawdzenia zasobów |
| Kolejka na short partition | Zmień na hopper partition |
| Brak epoch170 w expected path | Checkpoint był w root folder |

## Files Generated/Modified

### Modified
1. `Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune_v2.py`
   - Dodano `optimizer='SGD'` (linia 189)
   - Dodano parametr `--lrf`
   - Zmieniono default LR na 0.0001

2. `Eden/ClaudeSshSession/sesja_2026-01-18_yolo_resume/run_yolo_20k_resume_ep200_to_500.slurm`
   - Zmieniono na epoch170
   - Zmieniono partycję na hopper
   - Zmieniono na 4 GPU

### Created
1. `Eden/ClaudeSshSession/sesja_2026-01-19_yolo_finetune_ep170_500/`
   - `SESSION_SUMMARY.md`
   - `run_yolo_20k_ep170_to_500.slurm`

2. `.sessions/Session_2026-01-19_042000_YOLO_LR_Fix_Finetune/`
   - `README.md`
   - `SESSION_SUMMARY.md`

## Commands Used

```bash
# Sprawdzenie statusu
ssh eden-cluster "squeue -u bpiotrowski"
ssh eden-cluster "sfree"

# Anulowanie jobów
ssh eden-cluster "scancel <job_id>"

# Kopiowanie skryptów
scp script.py eden-cluster:~/Yolo/20k_finetune/
scp script.slurm eden-cluster:~/Yolo/20k_finetune/

# Uruchomienie joba
ssh eden-cluster "cd ~/Yolo/20k_finetune && sbatch run_yolo_20k_ep170_to_500.slurm"

# Sprawdzenie logów
ssh eden-cluster "tail -100 ~/Yolo/20k_finetune/logs/yolo_20k_ep170_to_500_1523692.log"

# Weryfikacja LR
ssh eden-cluster "grep -E '(optimizer|lr0|SGD)' ~/Yolo/20k_finetune/logs/*.log"
```

## Training Configuration

```python
optimizer = 'SGD'      # WAŻNE: explicit, nie auto!
lr0 = 0.001            # Fine-tune LR
lrf = 0.1              # Final LR ratio
momentum = 0.937
epochs = 330           # Nowe epoki
batch = 32
imgsz = 640
device = '0,1,2,3'     # 4x H100
model = 'epoch170.pt'  # Pretrained weights
```

## Performance Estimate

| Metryka | Wartość |
|---------|---------|
| 1 iteracja | ~1.30s |
| 1 epoka | ~12 min |
| 330 epok | ~66 godzin |
| **Szacowany czas** | **~2.75 dni** |
| Limit partycji | 5 dni |

## Next Steps
- [ ] Monitorować trening (loss, mAP)
- [ ] Sprawdzić checkpoint po epoch 10
- [ ] Po zakończeniu: walidacja na test set
- [ ] Porównanie z modelem z epoch 200

## Verification

Potwierdzone w logach:
```
optimizer: SGD(lr=0.001, momentum=0.937)
Starting training for 330 epochs...
```

---

*Session saved: 2026-01-19 04:20 UTC*
*Job 1523692 running on hopper (4x H100)*
