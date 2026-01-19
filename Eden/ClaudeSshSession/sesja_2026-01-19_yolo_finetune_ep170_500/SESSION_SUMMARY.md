# Sesja SSH Eden - 2026-01-19 03:07 UTC

**Data:** 2026-01-19
**Godzina:** ~03:07 UTC
**Job ID:** 1523692
**Status:** RUNNING
**Partycja:** hopper (5 dni limit)
**GPU:** 4x NVIDIA H100 PCIe (81GB each)

---

## Podsumowanie Sesji

Uruchomienie fine-tuningu YOLO na 20k selected dataset od epoch 170 do 500.
Kluczowe poprawki:
1. **Learning Rate Fix** - zmiana z `optimizer=auto` (ignorował lr0) na `optimizer=SGD`
2. **LR = 0.001** (fine-tune rate, 10x mniejszy niż oryginalny 0.01)
3. **Start z epoch 170** checkpoint zamiast 200
4. **4x H100 GPU** na partycji hopper

## Struktura Katalogów na Eden

```
/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/
├── epoch170.pt                      # Checkpoint startowy
├── run_yolo_20k_ep170_to_500.slurm  # Skrypt SLURM
├── yolo_train_20k_finetune_v2.py    # Skrypt treningowy
├── logs/                            # Logi SLURM
│   └── yolo_20k_ep170_to_500_1523692.log
├── train_out_ep170_to_500/          # Output treningu
│   └── exp/
│       └── weights/
├── ckpt_ep170_to_500/               # Checkpointy
├── best_ep170_to_500/               # Najlepszy model
└── train_out_5gpu_hopper/           # Poprzedni trening (ep0-200)
```

## Konfiguracja Treningu

### SLURM
```bash
#SBATCH -p hopper
#SBATCH --gres=gpu:4
#SBATCH --mem=150G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=yolo_20k_ep170_500
```

### Training Args
```
optimizer=SGD          # WAŻNE: explicit, nie auto!
lr0=0.001              # Fine-tune LR (10x mniejszy niż oryginalny)
lrf=0.1                # Final LR = 0.001 * 0.1 = 0.0001
momentum=0.937
epochs=330             # Nowe epoki (170 + 330 = 500 total)
batch=32
imgsz=640
device=0,1,2,3
model=epoch170.pt      # Pretrained weights
```

### Learning Rate Schedule
| Etap | LR Start | LR End |
|------|----------|--------|
| Oryginalny (ep0-170) | 0.01 | ~0.0016 |
| Fine-tune (ep170-500) | 0.001 | 0.0001 |

## Ważne Ścieżki

| Ścieżka | Opis |
|---------|------|
| `~/Yolo/20k_finetune/epoch170.pt` | Checkpoint startowy |
| `~/Yolo/20k_finetune/train_out_ep170_to_500/` | Output treningu |
| `~/Yolo/20k_finetune/logs/yolo_20k_ep170_to_500_*.log` | Logi |
| `/tmp/bpiotrowski_1523692/yolo_dataset/` | Dataset (TMPDIR) |

## Komendy SSH

```bash
# Status joba
squeue -u bpiotrowski

# Wolne zasoby
sfree

# Logi na żywo
tail -f ~/Yolo/20k_finetune/logs/yolo_20k_ep170_to_500_1523692.log

# Sprawdź optimizer i LR
grep -E "(optimizer|lr0|SGD)" ~/Yolo/20k_finetune/logs/yolo_20k_ep170_to_500_1523692.log

# Anuluj job
scancel 1523692
```

## Problemy i Rozwiązania

### Problem 1: optimizer=auto ignoruje lr0
**Symptom:** W logach widać:
```
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.0001' and 'momentum=0.937'
optimizer: SGD(lr=0.01, momentum=0.9)
```

**Rozwiązanie:** Dodano explicit `optimizer='SGD'` do skryptu Python:
```python
train_args = {
    'optimizer': 'SGD',  # MUST specify, otherwise 'auto' ignores lr0!
    'lr0': self.learning_rate,
    'momentum': 0.937,
    ...
}
```

### Problem 2: Brak epoch170 checkpoint
**Symptom:** Skrypt szukał w złym miejscu

**Rozwiązanie:** Checkpoint był w `~/Yolo/20k_finetune/epoch170.pt` (root folder)

### Problem 3: Kolejka na short partition
**Symptom:** Job pending z powodu braku zasobów

**Rozwiązanie:** Zmiana na partycję `hopper` (wolne 4 GPU)

## Status na Koniec Sesji

```
JOBID     PARTITION  NAME         USER      ST  TIME   NODES  NODELIST
1523692   hopper     yolo_20k     bpiotrow  R   0:XX   1      hopper
```

**Potwierdzone w logach:**
```
optimizer: SGD(lr=0.001, momentum=0.937)
Starting training for 330 epochs...
```

## Zmodyfikowane Pliki

1. `Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune_v2.py`
   - Dodano `optimizer='SGD'` (linia 189)
   - Dodano parametr `--lrf` (default 0.1)
   - Zmieniono default `--learning_rate` na 0.0001

2. `Eden/ClaudeSshSession/sesja_2026-01-18_yolo_resume/run_yolo_20k_resume_ep200_to_500.slurm`
   - Zmieniono na partycję hopper
   - Zmieniono na 4 GPU
   - Zaktualizowano ścieżki dla epoch170

---

*Sesja zapisana: 2026-01-19 03:30 UTC*
*Projekt: ViTParticleFilterTracker*
*Job: 1523692 na hopper (4x H100)*
