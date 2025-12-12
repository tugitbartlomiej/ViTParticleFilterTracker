# Sesja SSH Eden - 2025-12-12 18:18 UTC

**Data:** 2025-12-12
**Godzina:** 18:18 UTC (19:18 CET)
**Job ID:** 1454252
**Node:** pascal
**Status:** RUNNING

---

## Podsumowanie Sesji

Sesja poświęcona naprawie NCCL timeout i DDP desynchronization w treningu DETR na 20k dataset. Dodano LR Scheduler (cosine annealing) i TensorBoard logging.

### Główne osiągnięcia:
1. Naprawiono NCCL timeout (zwiększono timeout, dodano barrier przed all_reduce)
2. Naprawiono DDP desync (usunięto barrier na None batch)
3. Dodano Cosine LR Scheduler (lr_min=1e-7)
4. Dodano TensorBoard logging (loss, LR, gradient norm, GPU memory)
5. Wznowiono trening z checkpoint_epoch_170.pth

---

## Struktura Katalogów na Eden

```
/mnt/evafs/faculty/home/bpiotrowski/DETR/
├── detr_train_optimized.py          # Główny skrypt treningu (ZMODYFIKOWANY)
├── run_detr_20k_finetune.slurm      # SLURM job script
├── launch_tensorboard.sh            # Skrypt uruchamiania TensorBoard
│
├── train_out_20k_finetune/          # OUTPUT_DIR
│   ├── logs/                        # TensorBoard events
│   │   └── events.out.tfevents.*
│   └── final_model/                 # Model HuggingFace po zakończeniu
│       ├── config.json
│       ├── model.safetensors
│       └── preprocessor_config.json
│
├── ckpt_20k_finetune/               # CHECKPOINT_DIR
│   └── checkpoint_epoch_*.pth       # Checkpointy PyTorch do resume
│
├── best_20k_finetune/               # BEST_MODEL_DIR
│   └── (najlepszy model wg val_loss)
│
├── Checkpoints/                     # ARCHIVE_DIR
│   └── 20k_finetune/                # Kopia bezpieczeństwa checkpointów
│
└── logs/                            # Logi SLURM
    └── detr_20k_finetune_*.log
```

---

## Konfiguracja Treningu

### SLURM
```bash
#SBATCH -A transformers_vsc
#SBATCH -p experimental
#SBATCH --gres=gpu:4
#SBATCH --mem=250G
#SBATCH --time=5-00:00:00
```

### Torchrun
```bash
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29501 \
    detr_train_optimized.py \
    --images_dir "$IMAGES_DIR" \
    --annotations_path "$ANNOT_JSON" \
    --output_dir ./train_out_20k_finetune \
    --checkpoint_dir ./ckpt_20k_finetune \
    --best_model_dir ./best_20k_finetune \
    --epochs 300 \
    --batch_size 4 \
    --num_workers 4 \
    --num_queries 100 \
    --save_interval 5 \
    --patience 30 \
    --augment \
    --model_checkpoint facebook/detr-resnet-50 \
    --lr 5e-5 \
    --lr_backbone 5e-6 \
    --weight_decay 1e-4 \
    --lr_scheduler cosine \
    --lr_min 1e-7 \
    --resume_training
```

### Zmienne NCCL
```bash
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800              # 30 min
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## Ważne Ścieżki

| Ścieżka | Opis |
|---------|------|
| `/mnt/evafs/faculty/home/bpiotrowski/DETR/` | Główny katalog projektu |
| `./train_out_20k_finetune/logs/` | TensorBoard logi |
| `./train_out_20k_finetune/final_model/` | Model HuggingFace |
| `./ckpt_20k_finetune/` | Checkpointy PyTorch |
| `./best_20k_finetune/` | Najlepszy model |
| `./Checkpoints/20k_finetune/` | Archiwum checkpointów |
| `./logs/detr_20k_finetune_1454252.log` | Log aktualnego joba |

---

## Komendy SSH

### Połączenie
```bash
# SSH config (~/.ssh/config)
Host eden-cluster
    HostName eden
    User bpiotrowski
    ProxyJump eden-jump

# Połączenie
ssh eden-cluster
```

### Monitoring joba
```bash
# Status kolejki
ssh eden-cluster "squeue -u bpiotrowski"

# Logi na żywo
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_20k_finetune_1454252.log"

# Tylko postęp
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_20k_finetune_1454252.log | grep -E 'Training|Validation|Epoch|Loss'"
```

### TensorBoard
```bash
# Z przekierowaniem portu
ssh -L 6006:localhost:6006 eden-cluster
tensorboard --logdir=/mnt/evafs/faculty/home/bpiotrowski/DETR/train_out_20k_finetune/logs --port=6006

# Lub skrypt
ssh eden-cluster "bash /mnt/evafs/faculty/home/bpiotrowski/DETR/launch_tensorboard.sh"
```

### Upload plików
```bash
scp -o ProxyJump=eden-jump "local_file.py" bpiotrowski@eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/
```

---

## Problemy i Rozwiązania

### Problem 1: NCCL Timeout (oryginalny)
- **Objaw:** `WorkNCCL ran for 600236 ms before timing out` na epoch 171
- **Przyczyna:** Brak synchronizacji DDP przed all_reduce
- **Rozwiązanie:**
  - `NCCL_TIMEOUT=1800`
  - `dist.barrier()` przed all_reduce

### Problem 2: DDP Desynchronization (po pierwszym fix)
- **Objaw:** `Rank 3 running BROADCAST, Rank 1 running REDUCE`
- **Przyczyna:** `dist.barrier()` na None batch gdy tylko niektóre ranki mają None
- **Rozwiązanie:** Usunięcie barrier na None batch

### Problem 3: Oscylujący Loss
- **Objaw:** Loss: 0.192 → 0.232 → 0.244 → 0.159
- **Przyczyna:** Stały LR=1e-4 przez cały trening
- **Rozwiązanie:** Cosine LR Scheduler (5e-5 → 1e-7)

---

## Status na Koniec Sesji

```
Job ID:     1454252
Status:     RUNNING
Node:       pascal
GPUs:       4x P100
Runtime:    ~25 min
Epoch:      171/300
Step:       ~350/1125 (31%)
Loss:       ~0.28
LR:         Cosine (starting 1e-4)
```

Trening przebiega stabilnie po naprawach DDP.

---

## Pliki Zmodyfikowane w Sesji

### Na Eden:
- `detr_train_optimized.py` - poprawki DDP + LR scheduler + TensorBoard
- `run_detr_20k_finetune.slurm` - zmienne NCCL + argumenty LR

### Lokalnie:
- `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py`
- `Eden/Scripts/20kDataset_11.12.25/run_detr_20k_finetune.slurm`
- `Eden/Scripts/20kDataset_11.12.25/launch_tensorboard.sh`
- `Notatki/2025-12-12_DETR_Training_NCCL_Timeout_LR_Scheduler.md`
- `Notatki/TreningUlepszanieZbioru/2025-12-12_DETR_Training_Checklist_Poprawnego_Uruchomienia.md`

---

*Sesja zapisana: 2025-12-12 18:18 UTC*
*Projekt: ViTParticleFilterTracker*
