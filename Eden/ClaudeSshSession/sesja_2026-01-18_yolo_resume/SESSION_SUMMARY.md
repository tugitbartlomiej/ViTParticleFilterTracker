# Sesja SSH Eden - 2026-01-18 YOLO Resume

**Data:** 2026-01-18
**Cel:** Wznowienie treningu YOLO od epoch 200 do 500
**Status:** PENDING (skrypt gotowy do submisji)

---

## Podsumowanie

Poprzedni trening YOLO zatrzymał się na epoch 200. Trzeba dotrenować do epoch 500 (300 nowych epok).

## Skrypt SLURM

**Plik:** `run_yolo_20k_resume_ep200_to_500.slurm`

### Konfiguracja:
| Parametr | Wartość |
|----------|---------|
| Partycja | short |
| GPU | 2 |
| Pamięć | 150G |
| Czas | 20h |
| Start | epoch 200 |
| Cel | +300 epok (total 500) |
| Batch size | 32/GPU (64 effective) |
| LR | 0.001 |

---

## Komendy do uruchomienia

### 1. Skopiuj skrypt na Eden:
```bash
scp "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\ClaudeSshSession\sesja_2026-01-18_yolo_resume\run_yolo_20k_resume_ep200_to_500.slurm" eden-cluster:/mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/
```

### 2. Submit job:
```bash
ssh eden-cluster "cd /mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune && sbatch run_yolo_20k_resume_ep200_to_500.slurm"
```

### 3. Sprawdź status:
```bash
ssh eden-cluster "squeue -u bpiotrowski"
```

### 4. Monitoruj logi:
```bash
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/logs/yolo_20k_ep200_to_500_*.log"
```

---

## Struktura katalogów na Eden

```
~/Yolo/20k_finetune/
├── train_out_5gpu_hopper/exp/weights/   # Checkpoint ep200 (oryginalny)
│   ├── best.pt
│   └── last.pt
├── train_out_resume_ep170/exp/weights/  # Checkpoint z poprzedniego resume
├── run_yolo_20k_resume_ep200_to_500.slurm  # NOWY SKRYPT
├── yolo_train_20k_finetune_v2.py        # Skrypt treningowy
└── logs/
```

## Output (po zakończeniu)

```
~/Yolo/20k_finetune/
├── train_out_ep200_to_500/exp/weights/  # Nowe checkpointy
├── ckpt_ep200_to_500/                   # Backup checkpointy
├── best_ep200_to_500/                   # Best model
└── final_ep500/                         # Finalne checkpointy
```

---

*Sesja utworzona: 2026-01-18*
*Projekt: ViTParticleFilterTracker*
