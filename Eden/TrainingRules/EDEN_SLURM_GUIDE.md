# Eden SLURM Training Guide

## Ścieżki na Eden

```bash
# Home directory
/mnt/evafs/faculty/home/bpiotrowski/

# Alias
~/  # = /home2/faculty/bpiotrowski/

# Projekty
~/DETR/                          # DETR training scripts
~/DETR/Checkpoints/              # Archived checkpoints
~/DETR/logs/                     # SLURM job logs

# Conda
/mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
# Środowisko: yolo_py310

# TMPDIR (szybki dysk lokalny na nodzie)
/tmp/${USER}_${SLURM_JOB_ID}/
```

## Partycje i GPU

| Partycja | Timelimit | GPU | Uwagi |
|----------|-----------|-----|-------|
| `long` | 5 dni | DGX (8x A100/H100) | Główna partycja |
| `experimental` | 5 dni | Pascal (4x P100 16GB) | Często wolna! |
| `short` | ? | DGX | Krótkie joby |
| `debug` | ? | Wszystkie | Testowanie |
| `hopper` | ? | 8x H100 | Najnowsze GPU |

### Sprawdzanie dostępności
```bash
sfree                    # Wolne zasoby
sinfo -N -l             # Stan nodów
squeue -u bpiotrowski   # Moje joby
squeue -u bpiotrowski --start  # Przewidywany start
```

### GPU Memory
- **DGX (A100/H100)**: 40-80GB per GPU
- **Pascal (P100)**: 16GB per GPU - wymaga mniejszego batch_size!

## Szablon SLURM

```bash
#!/bin/bash
#SBATCH -A transformers_vsc          # Account
#SBATCH -p experimental              # Partycja (long/experimental/short)
#SBATCH --gres=gpu:4                 # Liczba GPU
#SBATCH --mem=250G                   # RAM
#SBATCH --time=5-00:00:00            # Max czas (D-HH:MM:SS)
#SBATCH --job-name=my_job            # Nazwa joba
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/my_job_%j.log  # %j = job ID
#SBATCH --error=logs/my_job_%j.log
#SBATCH --open-mode=append

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

# === TMPDIR ===
TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR"

# === CONDA ===
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

# === TRAINING ===
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29501 \
    my_script.py \
    --batch_size 4 \
    --epochs 300 \
    --lr 5e-5 \
    --lr_backbone 5e-6 \
    --lr_scheduler cosine \
    --lr_min 1e-7 \
    --resume_training  # jeśli wznawiasz z checkpointu
```

## Typowe Problemy i Rozwiązania

### 1. CUDA Out of Memory na Pascal
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Rozwiązanie:** Zmniejsz batch_size
- DGX: batch_size 16-32
- Pascal: batch_size 2-4

### 2. Checkpoint nie ładuje się
```
[Checkpoint] No checkpoints found
```
**Rozwiązanie:**
- Sprawdź nazwę pliku: `checkpoint_epoch_XXX.pth`
- Skopiuj ręcznie: `cp source/checkpoint_epoch_170.pth ./ckpt_dir/`

### 3. epochs < checkpoint_epoch = natychmiastowe zakończenie
```
Starting training loop from epoch 171...
Saving final model...
Training completed!
```
**Rozwiązanie:** Ustaw epochs > checkpoint_epoch
```bash
--epochs 300  # jeśli checkpoint jest z epoch 170
```

### 4. AMP/Mixed Precision conflict z checkpoint optimizer state
```
AssertionError: assert grad_scale is None and found_inf is None
```
**Przyczyna:** Checkpoint był trenowany z innymi ustawieniami AMP/GradScaler. Optimizer state z checkpointu jest niekompatybilny z nowym GradScaler.

**Rozwiązanie:** Wyłącz AMP w skrypcie Python:
```python
# Zmień default=True na default=False:
parser.add_argument("--use_amp", action='store_true', default=False)
```
**Lokalizacja:** `~/DETR/detr_train_optimized.py`

**Komenda naprawy:**
```bash
sed -i 's/default=True,/default=False,/' ~/DETR/detr_train_optimized.py
```

### 5. Resume training zaczyna od epoch+1 zamiast epoch
```
[Checkpoint] Successfully loaded state from epoch 170
Starting training loop from epoch 171...  # Powinno być 170!
```
**Przyczyna:** Funkcja `load_checkpoint` zwraca `start_epoch + 1`

**Rozwiązanie:** W `detr_train_optimized.py` linia ~271:
```python
# BYŁO:
return model, optimizer, scaler, start_epoch + 1
# POWINNO BYĆ:
return model, optimizer, scaler, start_epoch
```
**Komenda naprawy:**
```bash
sed -i 's/return model, optimizer, scaler, start_epoch + 1/return model, optimizer, scaler, start_epoch/' ~/DETR/detr_train_optimized.py
```

### 6. Collate error - różne rozmiary obrazów
```
Error during collate_fn: The expanded size of the tensor (1333) must match
```
**Rozwiązanie:** Napraw collate_fn lub użyj stałego rozmiaru obrazów

**Uwaga (DDP):** jeżeli `collate_fn`/dataset zwraca `None` tylko na jednym ranku, nie można robić lokalnego `continue`, bo ranki wykonują różne collectives (BROADCAST/REDUCE/BARRIER) i pojawia się błąd `Detected mismatch between collectives on ranks`.

**Praktyczny fix:** padding w collate_fn (żeby nie było wyjątków od różnych rozmiarów) + wspólny skip batcha na wszystkich rankach (np. `dist.all_reduce(MAX)` na fladze `batch is None`).

### 7. Job w kolejce (Priority)
```
(Priority) lub (Resources)
```
**Rozwiązanie:**
- Sprawdź `sfree` - może inna partycja jest wolna
- `experimental` często ma wolne GPU (Pascal)

### 8. Ekstrakcja tar - pliki w złym miejscu
```bash
# Problem: mv images_tmp images - gdy images/ już istnieje
# Rozwiązanie:
mv images_tmp/* images/ 2>/dev/null; rmdir images_tmp
```

### 9. Resume training używa STAREGO LR z checkpointu
```
[Checkpoint] Successfully loaded state from epoch 172
Starting training loop from epoch 172...
Current LR: 1.00e-04   # ← ZŁE! Powinno być 5e-5
```
**Przyczyna:** `optimizer.load_state_dict()` nadpisuje nowe wartości LR starymi z checkpointu. Scheduler też jest źle skonfigurowany (myśli że trening trwa 0→epochs zamiast start_epoch→epochs).

**Rozwiązanie:** Dodaj LR Reset po wczytaniu checkpointu w `detr_train_optimized.py`:
```python
# Po load_checkpoint():
if args.resume_training and latest_checkpoint_path:
    # Reset LR to new values
    optimizer.param_groups[0]['lr'] = args.lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = args.lr_backbone

    # Re-initialize scheduler for remaining epochs
    remaining_epochs = args.epochs - start_epoch
    if scheduler is not None and args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=args.lr_min
        )
```

**Lokalizacja w kodzie:** `~/DETR/detr_train_optimized.py` linie ~592-614

**Weryfikacja że fix działa:**
```
[LR Reset] Resetting learning rates to new fine-tuning values...
  Old LR (from checkpoint): 1.00e-04
  New LR: 5.00e-05 (backbone: 5.00e-06)
[Scheduler] Re-initialized cosine scheduler for 128 remaining epochs
Current LR: 5.00e-05   # ← POPRAWNE!
```

### 10. Brak LR Scheduler = oscylacje loss
```
Epoch 100: loss=0.19
Epoch 120: loss=0.23  # ← wzrost!
Epoch 140: loss=0.24  # ← dalszy wzrost!
Epoch 160: loss=0.16  # ← nagle spadek
```
**Przyczyna:** Stały LR bez decay powoduje niestabilność w późnych epokach treningu DETR (Hungarian matching).

**Rozwiązanie:** Zawsze używaj cosine scheduler z małym lr_min:
```bash
--lr_scheduler cosine \
--lr_min 1e-7
```

**Zalecane parametry fine-tuning:**
- LR: `5e-5` (2x mniej niż początkowe 1e-4)
- Backbone LR: `5e-6` (10x mniej niż main)
- Scheduler: `cosine`
- lr_min: `1e-7`

## Komendy SLURM

```bash
# Submit
sbatch script.slurm

# Status
squeue -u bpiotrowski
squeue -u bpiotrowski --start

# Anuluj
scancel JOB_ID

# Logi
tail -f ~/DETR/logs/job_NAME_JOBID.log

# Zasoby
sfree
sinfo -p experimental -o "%P %l"  # Limit partycji
```

## Skrypty na Eden

```
~/DETR/
├── detr_train_optimized.py      # Główny skrypt treningowy
├── run_detr_20k_finetune.slurm  # SLURM dla 20k dataset
├── Checkpoints/
│   ├── checkpoint_epoch_170.pth
│   └── 20k_finetune/            # Checkpointy z fine-tuningu
├── ckpt_20k_finetune/           # Working checkpoints (lokalnie)
├── best_20k_finetune/           # Best model
├── train_out_20k_finetune/      # Output
└── logs/                        # SLURM logs
```

## SSH z Windows

```bash
# ~/.ssh/config
Host eden-jump
    HostName ssh.mini.pw.edu.pl
    User piotrowskib2
    ServerAliveInterval 60

Host eden-cluster
    HostName eden
    User bpiotrowski
    ProxyJump eden-jump
    ServerAliveInterval 60
```

```bash
# Połączenie
ssh eden-cluster

# Kopiowanie plików
scp local_file.json eden-cluster:~/destination/
scp eden-cluster:~/remote_file.pth ./local/
```

## Checklist przed submitowaniem

- [ ] Partycja odpowiednia do dostępnych GPU (`sfree`)
- [ ] batch_size dopasowany do GPU (Pascal=4, DGX=16+)
- [ ] epochs > checkpoint_epoch (jeśli resume)
- [ ] Checkpoint skopiowany do working dir
- [ ] AMP wyłączone jeśli problemy z optimizer state
- [ ] JSON annotations przesłany na Eden
- [ ] Katalog logs/ istnieje
- [ ] **LR scheduler ustawiony** (`--lr_scheduler cosine --lr_min 1e-7`)
- [ ] **LR Reset fix** w `detr_train_optimized.py` (jeśli resume z nowym LR)
