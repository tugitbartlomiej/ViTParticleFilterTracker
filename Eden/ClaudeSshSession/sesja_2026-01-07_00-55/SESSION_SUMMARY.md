# Sesja SSH Eden - 2026-01-07 00:55 UTC

**Data:** 2026-01-07
**Godzina:** 00:55 UTC (01:55 CET)
**Job ID:** 1500982
**Node:** dgx-4 (4x A100 40GB)
**Status:** COMPLETED (SUCCESS)

---

## Podsumowanie Sesji

Uruchomienie treningu **YOLO 20k Fine-tune** na klastrze Eden. Kontynuacja z checkpoint epoch170.pt na 20,000 inteligentnie wybranych obrazach z Advanced Dataset Selection pipeline.

### Kluczowe Osiagnięcia:
1. Naprawiono błąd ekstrakcji archiwum (mv *.jpg - argument list too long)
2. Naprawiono błąd przekazywania zmiennych do heredoc Python (export przed heredoc)
3. Usunięto `set -o pipefail` które powodowało SIGPIPE przy `head -10`
4. Trening zakończony sukcesem z wybitnymi wynikami

### Wyniki Końcowe:
| Metryka | Wartość |
|---------|---------|
| mAP@0.5 | **99.5%** |
| mAP@0.5-0.95 | **96.3%** |
| Precision | 99.7% |
| Recall | 99.8% |

---

## Struktura Katalogów na Eden

```
/mnt/evafs/faculty/home/bpiotrowski/Yolo/
├── 20k_finetune/                              # NOWY - projekt 20k finetune
│   ├── logs/                                  # Logi SLURM
│   │   └── yolo_20k_finetune_1500982.log     # Log zakończonego treningu
│   ├── ckpt/                                  # Checkpointy (puste - ultralytics używa własnej ścieżki)
│   ├── best/                                  # Best model (puste)
│   ├── train_out/                             # Output (puste)
│   ├── train_out_5gpu_hopper/exp/weights/     # FAKTYCZNY OUTPUT (z config checkpoint)
│   │   ├── best.pt                            # 87 MB - najlepszy model (Jan 7 05:49)
│   │   ├── last.pt                            # 87 MB - ostatni checkpoint (Jan 7 05:49)
│   │   ├── epoch180.pt                        # 175 MB (Jan 7 02:39)
│   │   └── epoch190.pt                        # 175 MB (Jan 7 04:21)
│   ├── yolo_train_20k_finetune.py            # Skrypt treningowy Python
│   ├── run_yolo_20k_dgx3.slurm               # Skrypt SLURM (naprawiony)
│   └── yolo11n.pt                             # Model AMP test
├── ckpt_5gpu_hopper/                          # Stare checkpointy (do epoch170)
├── best_5gpu_hopper/                          # Stary best model
└── train_out_5gpu_hopper/                     # Stare outputy

/mnt/evafs/faculty/home/bpiotrowski/
├── 20kSelectedImages.tar                      # Archiwum 20k obrazów (~9GB)
├── merged_20k_annotations_fixed.json          # Adnotacje COCO (5.3 MB)
└── YOLO_Checkpoints/
    └── epoch170.pt                            # Checkpoint startowy
```

---

## Konfiguracja Treningu

### SLURM Configuration:
```bash
#SBATCH -A transformers_vsc
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=yolo_20k_dgx1
```

### Training Arguments:
```bash
python3 yolo_train_20k_finetune.py \
    --dataset_yaml_path $DATASET_YAML \
    --project_dir /mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/train_out \
    --checkpoint_dir /mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/ckpt \
    --best_model_dir /mnt/evafs/faculty/home/bpiotrowski/Yolo/20k_finetune/best \
    --model_size m \
    --epochs 300 \
    --batch_size 32 \
    --imgsz 640 \
    --learning_rate 0.001 \
    --workers 16 \
    --device 0,1,2,3 \
    --save_period 5 \
    --patience 30 \
    --freeze_backbone 0 \
    --resume --resume_path /mnt/evafs/faculty/home/bpiotrowski/YOLO_Checkpoints/epoch170.pt
```

### Ultralytics DDP Config:
- 4x NVIDIA A100-SXM4-40GB
- DDP: torch.distributed.run --nproc_per_node 4
- PyTorch 2.2.2+cu121
- Ultralytics 8.3.152

---

## Ważne Ścieżki

| Ścieżka | Opis |
|---------|------|
| `~/Yolo/20k_finetune/` | Główny folder projektu |
| `~/Yolo/20k_finetune/train_out_5gpu_hopper/exp/weights/best.pt` | **Najlepszy model (99.5% mAP)** |
| `~/Yolo/20k_finetune/logs/yolo_20k_finetune_1500982.log` | Log treningu |
| `~/YOLO_Checkpoints/epoch170.pt` | Checkpoint startowy |
| `~/20kSelectedImages.tar` | Archiwum obrazów |
| `~/merged_20k_annotations_fixed.json` | Adnotacje COCO |

---

## Komendy SSH

### Sprawdzanie statusu:
```bash
ssh eden-cluster "squeue -u bpiotrowski"
ssh eden-cluster "tail -50 ~/Yolo/20k_finetune/logs/yolo_20k_finetune_1500982.log"
```

### Sprawdzanie modeli:
```bash
ssh eden-cluster "ls -la ~/Yolo/20k_finetune/train_out_5gpu_hopper/exp/weights/"
ssh eden-cluster "find ~/Yolo/ -name '*.pt' -mtime -1 -ls"
```

### Kopiowanie modelu lokalnie:
```bash
scp eden-cluster:~/Yolo/20k_finetune/train_out_5gpu_hopper/exp/weights/best.pt ./yolo_20k_best.pt
```

---

## Problemy i Rozwiązania

### Problem 1: Exit code 13 (SIGPIPE)
**Przyczyna:** `set -o pipefail` + `ls -la | head -10` powoduje SIGPIPE gdy head zamyka potok
**Rozwiązanie:** Zmiana `set -euo pipefail` na `set -eu`

### Problem 2: Argument list too long
**Przyczyna:** `mv *.jpg images/` z 20k plików przekracza limit argumentów bash
**Rozwiązanie:** Użycie obrazów bezpośrednio z EXTRACT_DIR bez przenoszenia

### Problem 3: Zmienne nie przekazane do heredoc Python
**Przyczyna:** `export COCO_JSON IMAGES_DIR YOLO_DATASET_DIR` było PO heredoc
**Rozwiązanie:** Przeniesienie `export` PRZED `python3 << 'PYTHON_SCRIPT'`

### Problem 4: Requested node configuration not available
**Przyczyna:** Partycja `experimental` zamiast `long` dla dgx
**Rozwiązanie:** Zmiana na `-p long` i usunięcie `--nodelist`

---

## Status na Koniec Sesji

### Trening YOLO 20k Fine-tune:
- **Status:** COMPLETED (SUCCESS)
- **Job ID:** 1500982
- **Node:** dgx-4
- **Runtime:** ~5 godzin
- **Exit code:** 0

### Wyniki walidacji:
```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
  all       2000       2000      0.997      0.998      0.995      0.963
```

### Pliki wyjściowe:
- `best.pt` - 87 MB (Jan 7 05:49) - **GOTOWY DO UŻYCIA**
- `last.pt` - 87 MB (Jan 7 05:49)
- `epoch180.pt`, `epoch190.pt` - checkpointy pośrednie

---

## Następne Kroki

1. [ ] Skopiować `best.pt` lokalnie do benchmarków
2. [ ] Uruchomić benchmark YOLO vs DETR na zewnętrznym datasecie
3. [ ] Porównać wyniki z DETR 20k finetune (ep180: 77% mAP)
4. [ ] Zaktualizować artykuł IEEE z wynikami YOLO

---

*Sesja zapisana: 2026-01-07 00:55 UTC*
*Projekt: ViTParticleFilterTracker*
*Trening: YOLO 20k Fine-tune na 4x A100 GPU*
