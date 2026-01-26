# Sesja SSH Eden - 2026-01-20 13:59 UTC

**Data:** 2026-01-20
**Godzina:** ~13:59 UTC (14:59 CET)
**Job ID:** 1524831
**Status:** RUNNING
**Partycja:** hopper (7x H100 GPU)
**Batch:** 35 (5 per GPU)

---

## Podsumowanie Sesji

Ponowny fine-tuning YOLO od epoch 170 z **POPRAWIONYM Learning Rate** po nieudanym poprzednim treningu.

### Lekcja z poprzedniego treningu (Job 1523692):
- `lr0=0.001` byl ZA WYSOKI
- Model osiagnal najlepsze wyniki w epoch 1, potem sie pogorszal
- EarlyStopping zatrzymal trening po 51 epokach
- Najlepszy model (epoch 1): mAP50-95=0.961

### Nowa strategia:
- `lr0=0.0001` (10x mniejszy niz poprzednio)
- `lrf=0.1` -> final LR = 0.00001
- `patience=100` (wiecej cierpliwosci)
- 7x H100 GPU (zamiast 4)

## Struktura Katalogow na Eden

```
/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/
├── epoch170.pt                       # Checkpoint startowy
├── run_yolo_20k_ep170_lr0001.slurm   # Skrypt SLURM
├── yolo_train_20k_finetune_v2.py     # Skrypt treningowy
├── logs/                             # Logi SLURM
│   └── yolo_20k_ep170_lr0001_1524831.log
├── train_out_ep170_lr0001/           # Output treningu (NOWY)
│   └── exp/
│       └── weights/
├── ckpt_ep170_lr0001/                # Checkpointy (NOWY)
├── best_ep170_lr0001/                # Najlepszy model (NOWY)
├── train_out_ep170_to_500/           # Poprzedni trening (lr0=0.001)
│   └── exp/weights/best.pt           # mAP50-95=0.961
└── best_ep170_to_500/                # Poprzedni best model
```

## Konfiguracja Treningu

### SLURM
```bash
#SBATCH -p hopper
#SBATCH --gres=gpu:7
#SBATCH --mem=150G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=yolo_20k_lr0001
```

### Training Args (POPRAWIONE)
```
optimizer=SGD          # Explicit (nie auto!)
lr0=0.0001            # POPRAWIONE! (bylo 0.001)
lrf=0.1               # Final LR = 0.00001
momentum=0.937
epochs=330            # (170 + 330 = 500 total)
batch=35              # 5 per GPU x 7
imgsz=640
device=0,1,2,3,4,5,6  # 7 GPU
patience=100          # Zwiekszone (bylo 50)
model=epoch170.pt
```

### Porownanie LR
| Trening | lr0 | Final LR | GPU | Wynik |
|---------|-----|----------|-----|-------|
| Poprzedni (nieudany) | 0.001 | 0.0001 | 4x H100 | Best @ epoch 1, degradacja |
| **Nowy (poprawiony)** | **0.0001** | **0.00001** | **7x H100** | W trakcie |

### Porownanie z oryginalnym treningiem
| Etap | LR |
|------|-----|
| Oryginalny trening epoch 0 | 0.01 |
| Oryginalny trening epoch 170 | ~0.0016 (cosine decay) |
| Poprzedni fine-tune (nieudany) | 0.001 |
| **Obecny fine-tune** | **0.0001** |

## Komendy SSH

### Monitorowanie
```bash
# Status joba
squeue -u bpiotrowski

# Logi na zywo
tail -f ~/Yolo/20k_finetune/logs/yolo_20k_ep170_lr0001_1524831.log

# Sprawdz LR w logach
grep -E "(optimizer|lr0|SGD)" ~/Yolo/20k_finetune/logs/yolo_20k_ep170_lr0001_1524831.log

# Sprawdz metryki
grep -E "(mAP|Epoch)" ~/Yolo/20k_finetune/logs/yolo_20k_ep170_lr0001_1524831.log | tail -20
```

### Zarzadzanie jobem
```bash
# Anuluj job
scancel 1524831

# Sprawdz zasoby
sfree | grep hopper
```

## Wazne Sciezki

| Sciezka | Opis |
|---------|------|
| `~/Yolo/20k_finetune/epoch170.pt` | Checkpoint startowy |
| `~/Yolo/20k_finetune/train_out_ep170_lr0001/` | Output treningu |
| `~/Yolo/20k_finetune/logs/yolo_20k_ep170_lr0001_1524831.log` | Logi |
| `/tmp/bpiotrowski_1524831/yolo_dataset/` | Dataset (TMPDIR) |

## Problemy i Rozwiazania

### Problem 1: SSH timeout
**Symptom:** `Connection timed out` przy polaczeniu z eden-cluster
**Rozwiazanie:** Ponowne proby polaczenia

### Problem 2: DOS line endings
**Symptom:** `sbatch: error: Batch script contains DOS line breaks`
**Rozwiazanie:** `sed -i 's/\r$//' run_yolo_20k_ep170_lr0001.slurm`

### Problem 3: Batch size nie podzielny przez GPU
**Symptom:** `ValueError: 'batch=32' must be a multiple of GPU count 7`
**Rozwiazanie:** Zmiana batch z 32 na 35 (5 per GPU x 7)

### Problem 4: Poprzedni lr0=0.001 za wysoki
**Symptom:** Model osiagnal best w epoch 1, potem degradacja
**Rozwiazanie:** Zmniejszenie lr0 do 0.0001

## Status Treningu (poczatek)

### Potwierdzone w logach:
```
optimizer: SGD(lr=0.0001, momentum=0.937)
Starting training for 330 epochs...
Using 7x H100 PCIe GPUs (DDP mode)
```

### Loss po ~11 iteracjach (epoch 1):
| Metryka | Start | Aktualnie |
|---------|-------|-----------|
| box_loss | 0.86 | 0.45 |
| cls_loss | 1.31 | 0.28 |
| dfl_loss | 1.07 | 0.85 |

**Wniosek:** Model sie uczy - loss spada prawidlowo z lr0=0.0001

## Oczekiwane Wyniki

- Stopniowa poprawa metryk przez wiele epok (nie tylko epoch 1)
- mAP50-95 > 0.961 (poprzedni best)
- Brak wczesnego zatrzymania przez EarlyStopping

## Historia Jobow

| Job ID | Data | lr0 | GPU | Status |
|--------|------|-----|-----|--------|
| 1523692 | 2026-01-19 | 0.001 | 4x H100 | COMPLETED (early stop @ ep 51) |
| 1524827 | 2026-01-20 | 0.0001 | 4x H100 | CANCELLED (zamieniony na 7 GPU) |
| 1524828 | 2026-01-20 | 0.0001 | 7x H100 | FAILED (batch=32 nie podzielne) |
| **1524831** | 2026-01-20 | **0.0001** | **7x H100** | **RUNNING** |

---

*Sesja zapisana: 2026-01-20 14:05 UTC*
*Projekt: ViTParticleFilterTracker*
*Job: 1524831 na hopper (7x H100)*
*Korekta: LR 0.001 -> 0.0001, batch 32 -> 35*
