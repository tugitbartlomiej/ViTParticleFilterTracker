# Sesja SSH Eden - 2026-01-16 23:05 UTC

**Data:** 2026-01-16
**Godzina:** 23:05 UTC
**Job ID:** 1519552 (nowy), 1509798 (anulowany), 1509211 (anulowany)
**Status:** PENDING (nowy job submitted)

---

## Podsumowanie Sesji

Sesja poswiecona analizie stanu treningow YOLO i DETR na klastrze Eden oraz optymalizacji kolejki SLURM.

### Glowne dzialania:
1. Sprawdzenie statusu klastra - awaria klimatyzacji (dgx-2, dgx-3, hopper-2 DOWN)
2. Analiza kolejki SLURM - duze zatlocenie, array job owernera (74 zadania)
3. Znalezienie checkpointow YOLO 20k (ep200 COMPLETED) i DETR 20k (ep435)
4. Anulowanie DETR job (1509211) na zadanie uzytkownika
5. Anulowanie starego YOLO job (1509798) ktory czekal 4 dni na hopper
6. Stworzenie nowego skryptu YOLO dla partycji `short` z 2 GPU
7. Submisja nowego joba (1519552) - pozycja 7 w kolejce short

---

## Status Klastra

### Nody DOWN (awaria klimatyzacji):
| Node | Status | Powod |
|------|--------|-------|
| dgx-2 | down* | awaria klimatyzacji |
| dgx-3 | down* | awaria klimatyzacji |
| hopper-2 | down | gres/gpu count reporting |

### Nody dzialajace:
- hopper (8 GPU) - zajety przez vzaigraj
- dgx-1 (8 GPU) - running jobs
- dgx-4 (8 GPU) - running jobs

---

## Struktura Katalogow na Eden

```
/mnt/evafs/faculty/home/bpiotrowski/
├── Yolo/
│   └── 20k_finetune/
│       ├── train_out_5gpu_hopper/exp/weights/  # YOLO ep200 checkpoints
│       │   ├── best.pt (84MB)
│       │   ├── last.pt (84MB)
│       │   ├── epoch180.pt (167MB)
│       │   └── epoch190.pt (167MB)
│       ├── run_yolo_20k_finetune.slurm         # stary skrypt (long, 3GPU)
│       ├── run_yolo_20k_finetune_short.slurm   # NOWY skrypt (short, 2GPU)
│       ├── yolo_train_20k_finetune_v2.py
│       └── logs/
│
├── DETR/
│   ├── Checkpoints/                    # DETR 100k original (ep20-170)
│   ├── ckpt_20k_finetune_v2_fixed/    # DETR 20k finetune (ep170-435, 54 pliki)
│   └── train_out_20k_finetune/
│
└── ViT/                               # Stary projekt (nieaktualny)
```

---

## Konfiguracja Nowego Treningu YOLO

### Stara konfiguracja (anulowana):
```bash
#SBATCH -p long
#SBATCH --gres=gpu:3
#SBATCH --time=5-00:00:00
```

### Nowa konfiguracja (submitted):
```bash
#SBATCH -p short              # Inna kolejka - mniej zatloczona
#SBATCH --gres=gpu:2          # 2 GPU zamiast 3 - latwiej dostac slot
#SBATCH --time=20:00:00       # 20h wystarczy na 300 epok
#SBATCH --mem=150G
#SBATCH --job-name=yolo_20k_short
```

### Parametry treningu:
- **Start:** epoch 200 (last.pt)
- **Cel:** epoch 500 (300 nowych epok)
- **Batch size:** 32/GPU (effective 64)
- **Learning rate:** 0.001
- **Device:** 0,1 (2 GPU)
- **Output:** train_out_20k_finetune_short/

---

## Wazne Sciezki

| Sciezka | Opis |
|---------|------|
| `/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/` | YOLO 20k projekt |
| `.../train_out_5gpu_hopper/exp/weights/last.pt` | Pretrained weights (ep200) |
| `.../run_yolo_20k_finetune_short.slurm` | Nowy skrypt SLURM |
| `/mnt/evafs/faculty/home/bpiotrowski/DETR/ckpt_20k_finetune_v2_fixed/` | DETR 20k checkpointy |

---

## Komendy SSH

### Sprawdzenie statusu:
```bash
# Status jobów
ssh eden-cluster "squeue -u bpiotrowski"

# Status kolejki short
ssh eden-cluster "squeue -p short"

# Szczegoly joba
ssh eden-cluster "scontrol show job 1519552"

# Status nodów
ssh eden-cluster "sinfo -N -l"

# Wolne zasoby
ssh eden-cluster "sfree"
```

### Monitorowanie treningu:
```bash
# Logi
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/logs/yolo_20k_short_1519552.log"

# Checkpointy
ssh eden-cluster "ls -la /mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/train_out_20k_finetune_short/"
```

### Zarzadzanie jobami:
```bash
# Anuluj job
ssh eden-cluster "scancel JOB_ID"

# Submit nowy
ssh eden-cluster "cd /mnt/evafs/.../Yolo/20k_finetune && sbatch run_yolo_20k_finetune_short.slurm"
```

---

## Historia Jobow (2026-01-16)

| Job ID | Nazwa | Status | Uwagi |
|--------|-------|--------|-------|
| 1509211 | detr_20k_4gpu | CANCELLED | Anulowany na zadanie usera |
| 1509798 | yolo_20k_ft | CANCELLED | Czekal 4 dni na hopper |
| **1519552** | **yolo_20k_short** | **PENDING** | Nowy job, pozycja 7 w short |

---

## Problemy i Rozwiazania

### Problem 1: Dlugi czas oczekiwania (4 dni)
- **Przyczyna:** Partycja `hopper` zajeta przez array job (74 zadania)
- **Rozwiazanie:** Przesubmitowanie na partycje `short`

### Problem 2: Awaria klimatyzacji
- **Wplyw:** dgx-2, dgx-3, hopper-2 niedostepne
- **Rozwiazanie:** Uzywanie dgx-1, dgx-4 (dzialaja)

### Problem 3: Array job blokuje kolejke
- **Opis:** owerner ma 74 pending tasks przed userem
- **Rozwiazanie:** Zmiana partycji + mniejsza liczba GPU

---

## Status na Koniec Sesji

### YOLO 20k Finetune:
- **Completed:** ep200 (checkpointy dostepne)
- **Pending:** ep200 -> ep500 (job 1519552 w kolejce)
- **Pozycja:** 7 w kolejce `short`
- **Szacowany start:** kilka godzin (running jobs sa krotkie)

### DETR 20k Finetune:
- **Completed:** ep435 z 500 (87%)
- **Status:** Trening anulowany (na zadanie usera)
- **Checkpointy:** 54 pliki zachowane

### Klaster:
- 3/6 nodow GPU DOWN
- Duze zatlocenie
- Partycja `short` relatywnie wolna

---

## Nastepne Kroki

1. [ ] Monitorowac job 1519552 - czekac na start
2. [ ] Po zakonczeniu YOLO - uruchomic benchmark na nowych checkpointach
3. [ ] Rozwazyc wznowienie DETR do ep500 (opcjonalnie)

---

*Sesja zapisana: 2026-01-16 23:05*
*Projekt: ViTParticleFilterTracker*
