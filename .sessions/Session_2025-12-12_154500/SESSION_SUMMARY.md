# Session Summary: Session_2025-12-12_154500

## Metadata
- **Date:** 2025-12-12
- **Time:** 14:45 - 15:45 CET
- **Status:** Active (training running)
- **Type:** SSH | Training

## Objective
Uruchomić fine-tuning DETR na 20k wybranych obrazach na klastrze Eden, kontynuując od checkpoint epoch 170.

## Context
- Dataset 20k obrazów wybranych przez Advanced Dataset Selection pipeline
- Checkpoint z poprzedniego treningu: `checkpoint_epoch_170.pth`
- Partycja `long` była zajęta, użyto `experimental` (Pascal)

## Actions Taken

### 1. Analiza dostępności GPU
- `sfree` - sprawdzenie wolnych zasobów
- Pascal (experimental) wolny z 4x P100 (16GB)
- DGX zajęte (0 GPU wolnych)

### 2. Zmiana partycji SLURM
- Zmiana z `long` na `experimental`
- Problem: skrypt używał `-p long` nie `--partition=long`
- Rozwiązanie: `sed -i 's/-p long/-p experimental/'`

### 3. Naprawy ekstrakcji datasetu
- JSON annotations nie był w archiwum tar
- Przesłano `merged_20k_annotations.json` przez SCP
- Naprawiono logikę `mv images_tmp images` -> `mv images_tmp/* images/`

### 4. Naprawy treningu
| Problem | Rozwiązanie |
|---------|-------------|
| CUDA OOM (batch 16) | Zmniejszono batch_size do 4 |
| epochs=100 < checkpoint 170 | Zwiększono epochs do 300 |
| AMP AssertionError | Wyłączono AMP (`default=False`) |
| start_epoch +1 | Usunięto `+1` z `load_checkpoint()` |

### 5. Analiza 20k datasetu
- Przeanalizowano `selection_reasons.json` i `.txt`
- Zidentyfikowano kategorie: trudne (57), unikalne (2662), reprezentatywne (47)
- Top źródła: test01 (8.6%), train01 (7.4%)

### 6. Dokumentacja
- Utworzono `Eden/TrainingRules/EDEN_SLURM_GUIDE.md`
- Zapisano działający skrypt SLURM jako template

## Results

### Key Findings
- Pascal często wolny gdy DGX zajęte - warto sprawdzać `experimental`
- P100 (16GB) wymaga batch_size 4 (nie 16 jak A100/H100)
- AMP może powodować konflikt z optimizer state z checkpointu
- Checkpoint epoch N oznacza ukończoną epokę N

### Issues Encountered
1. **PartitionTimeLimit** - job w `long` gdy wymagał Pascala
2. **0 images** - JSON nie w archiwum, błędna logika mv
3. **OOM** - batch_size 16 za duży dla P100
4. **Instant completion** - epochs < checkpoint_epoch
5. **AMP AssertionError** - niekompatybilny optimizer state
6. **Epoch +1** - trening zaczynał od 171 zamiast 170

### Training Status
- **Job ID:** 1453787
- **Node:** pascal
- **Epoch:** 170/300 (running)
- **Speed:** ~3 sek/batch
- **Loss:** ~0.4-0.6

## Files Generated/Modified

### Utworzone
- `Eden/TrainingRules/EDEN_SLURM_GUIDE.md` - kompletny przewodnik
- `Eden/TrainingRules/run_detr_20k_finetune.slurm` - template skryptu

### Przesłane na Eden
- `merged_20k_annotations.json` -> `~/merged_20k_annotations.json`

### Zmodyfikowane na Eden
- `~/DETR/run_detr_20k_finetune.slurm` - partycja, batch_size, epochs
- `~/DETR/detr_train_optimized.py` - AMP default, start_epoch

## Commands Used

### SLURM
```bash
sbatch run_detr_20k_finetune.slurm
squeue -u bpiotrowski
scancel JOB_ID
sfree
sinfo -p experimental -o "%P %l"
```

### SSH
```bash
ssh eden-cluster "command"
scp file.json eden-cluster:~/
```

### Naprawy
```bash
sed -i 's/-p long/-p experimental/' run_detr_20k_finetune.slurm
sed -i 's/--batch_size 16/--batch_size 4/' run_detr_20k_finetune.slurm
sed -i 's/--epochs 100/--epochs 300/' run_detr_20k_finetune.slurm
sed -i 's/default=True,/default=False,/' detr_train_optimized.py
sed -i 's/start_epoch + 1/start_epoch/' detr_train_optimized.py
```

## Next Steps
- [ ] Monitorować trening (epoch 170-300)
- [ ] Sprawdzić validation loss po kilku epokach
- [ ] Zapisać best model do Checkpoints/best_20k_finetune/
- [ ] Porównać z poprzednim treningiem na pełnym datasecie
