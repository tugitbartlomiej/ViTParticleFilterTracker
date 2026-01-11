# Session Summary: Session_2026-01-11_022500

## Metadata
- **Date:** 2026-01-11
- **Time:** 02:25:00 CET
- **Status:** Completed
- **Type:** SSH | Training

## Objective
Kontynuacja treningow DETR i YOLO na 20k dataset do epoch 500. Naprawa nieudanego joba DETR i uruchomienie obu treningow na klastrze Eden.

## Context
- Poprzednia sesja (2025-12-31) wykonala benchmark DETR 20k finetune
- DETR zatrzymal sie na epoch 295, YOLO na epoch 200
- Job DETR 1505719 FAILED z powodu zlych argumentow CLI

## Actions Taken

### 1. Wczytanie poprzedniej sesji
- Wczytano Session_2025-12-31_042306 z wynikami benchmarku
- Kluczowe: DETR ep180 osiagnal 77% mAP na zewnetrznym datasecie

### 2. Analiza nieudanego joba DETR
- Job 1505719 failed z bledem: `--images_dir, --annotations_path required`
- Przyczyna: skrypt `run_detr_20k_resume_295.slurm` uzywal ZLYCH argumentow:
  - `--coco_path` zamiast `--images_dir`
  - brak `--annotations_path`
  - `--resume <path>` zamiast `--resume_training`
  - `--save_every` zamiast `--save_interval`

### 3. Analiza duplikatow checkpointow DETR
- `ckpt_20k_finetune_v2_fixed/` - 27 ckpt (do ep295), 13GB
- `Checkpoints/20k_finetune_v2_fixed/` - 25 ckpt (do ep290), 12GB
- Archiwum brakuje ep295 (sync nie zdazyl)

### 4. Utworzenie naprawionego skryptu DETR
- Plik: `run_detr_20k_resume_4gpu.slurm`
- Poprawne argumenty zgodne z `detr_train_optimized.py`
- TensorBoard logi synchronizowane co 10 min

### 5. Analiza treningu YOLO
- Odkryto ze YOLO jest na epoch 200 (nie 300!)
- Sprawdzono metadane `last.pt`: `epoch: -1` (zakonczony), `epochs: 200`

### 6. Utworzenie skryptu YOLO resume
- Plik: `run_yolo_20k_resume_500.slurm`
- Resume z `last.pt` (ep200), target 500

### 7. Uruchomienie jobow na Eden
- Problemy z dostepnoscia GPU (dgx-1, dgx-3 w draining, hopper-2 unavailable)
- Finalne joby: 3 GPU, partycja long

### 8. Proba uruchomienia Serena MCP
- Serena dziala poprawnie (test reczny)
- Problem z polaczeniem Claude Code

## Results

### Key Findings

1. **Przyczyna bledu DETR job 1505719**
   - Skrypt uzywal kompletnie innych argumentow niz `detr_train_optimized.py` akceptuje
   - Ktos napisal skrypt bez sprawdzenia `--help`

2. **Stan checkpointow**
   - DETR: epoch 295 (working dir), epoch 290 (archiwum)
   - YOLO: epoch 200 (nie 300 jak myslano)

3. **Dostepnosc klastra**
   - dgx-1, dgx-3: draining (maintenance)
   - dgx-2, dgx-4: zajete
   - hopper: 2 wolne GPU
   - hopper-2: unavailable

### Issues Encountered

1. **Zle argumenty w skrypcie DETR**
   - Rozwiazanie: nowy skrypt z poprawnymi argumentami

2. **Confusion o epoch YOLO**
   - Rozwiazanie: sprawdzenie metadanych checkpointu

3. **Brak dostepnych GPU**
   - Rozwiazanie: joby w kolejce, rusza gdy sie zwolnia

4. **Serena MCP nie laczy sie**
   - Status: Serena dziala poprawnie, problem po stronie Claude Code

## Files Generated/Modified

### Eden (utworzone na klastrze)
- `/mnt/evafs/.../DETR/run_detr_20k_resume_4gpu.slurm`
- `/mnt/evafs/.../DETR/run_detr_20k_resume_500.slurm`
- `/mnt/evafs/.../DETR/run_detr_20k_resume_hopper2.slurm`
- `/mnt/evafs/.../Yolo/20k_finetune/run_yolo_20k_resume_500.slurm`

### Lokalne
- `Eden/ClaudeSshSession/sesja_2026-01-11_01-30/SESSION_SUMMARY.md`
- `Eden/ClaudeSshSession/sesja_2026-01-11_01-30/run_detr_20k_resume_4gpu.slurm`
- `Eden/ClaudeSshSession/sesja_2026-01-11_01-30/run_yolo_20k_resume_500.slurm`

## Commands Used

### SSH
```bash
ssh eden-cluster "squeue -u bpiotrowski"
ssh eden-cluster "squeue -u bpiotrowski --start"
ssh eden-cluster "sbatch <script.slurm>"
ssh eden-cluster "scancel <job_id>"
ssh eden-cluster "sinfo -p long -o '%N %G %a %t'"
```

### Analiza checkpointow
```bash
ssh eden-cluster "ls -la ~/DETR/ckpt_20k_finetune_v2_fixed/"
ssh eden-cluster "python3 -c 'import torch; ckpt = torch.load(...); print(ckpt.keys())'"
```

### Serena
```bash
cd serena && uv run serena start-mcp-server --context claude-code --project ...
```

## Job Queue Status (na koniec sesji)

| Job ID | Model | Epochs | GPUs | Est. Start | Node |
|--------|-------|--------|------|------------|------|
| 1509138 | DETR | 295->500 | 3 | 05:36 | dgx-2 |
| 1509140 | YOLO | 200->500 | 3 | 13:05 | dgx-4 |

## Next Steps
- [ ] Monitorowac start jobow DETR i YOLO
- [ ] Sprawdzic TensorBoard logi po starcie treningu
- [ ] Po zakonczeniu: benchmark DETR ep500 vs YOLO ep500
- [ ] Rozwiazac problem z Serena MCP (restart Claude Code)
- [ ] Zsynchronizowac brakujacy ep295 do archiwum DETR

## Configuration Summary

### DETR Training
- Resume: checkpoint_epoch_295.pth
- Target: 500 epochs
- GPUs: 3
- Batch: 4/GPU (effective 12)
- LR: 5e-5, backbone: 5e-6
- Scheduler: cosine, min_lr: 1e-7
- TensorBoard: sync co 10 min

### YOLO Training
- Resume: last.pt (epoch 200)
- Target: 500 epochs
- GPUs: 3
- Batch: 32/GPU (effective 96)
- LR: 0.001
- Patience: 50

---

*Sesja zapisana: 2026-01-11 02:25 CET*
*Projekt: ViTParticleFilterTracker*
*Treningi: DETR ep295->500, YOLO ep200->500*
