# DETR Training na Eden - Kompletna Checklist Poprawnego Uruchomienia

**Data:** 2025-12-12
**Kontekst:** Synteza doświadczeń z wielu sesji treningowych DETR na klastrze Eden. Dokument opisuje wszystkie niezbędne kroki, poprawki i konfiguracje wymagane do stabilnego treningu.

---

## Podsumowanie wykonawcze

Aby poprawnie uruchomić trening DETR na klastrze Eden z wieloma GPU (DDP), należy:
1. Poprawić synchronizację DDP w kodzie Python
2. Skonfigurować zmienne środowiskowe NCCL
3. Dodać LR Scheduler (cosine annealing)
4. Przygotować prawidłowy dataset (format COCO)
5. Użyć odpowiednich ustawień SLURM

---

## 1. Poprawki kodu Python (`detr_train_optimized.py`)

### 1.1 Usunięcie `@lru_cache`
```python
# ZŁE - powoduje desync między procesami DDP
@lru_cache(maxsize=1000)
def load_image(path):
    ...

# DOBRE - bez cache
def load_image(path):
    ...
```
**Powód:** `@lru_cache` nie jest DDP-safe, różne procesy mają różne cache → różny timing.

### 1.2 Obsługa None batch (BEZ barrier!)
```python
# ZŁE - barrier przy None batch powoduje desync gdy tylko niektóre ranki mają None
if batch is None:
    dist.barrier()  # USUNĄĆ!
    continue

# DOBRE - po prostu skip, drop_last=True zapewnia równe batche
if batch is None:
    continue
```
**Powód:** Jeśli tylko niektóre ranki dostaną None batch, barrier spowoduje deadlock.

### 1.3 Barrier przed all_reduce
```python
# Na końcu epoki treningowej, przed agregacją loss:
dist.barrier()  # Poczekaj na wszystkie GPU
train_loss_tensor = torch.tensor(train_loss).to(device)
dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
train_loss = train_loss_tensor.item() / world_size
```

### 1.4 Broadcast decyzji early stopping
```python
# Rank 0 podejmuje decyzję, broadcast do wszystkich
dist.barrier()
if rank == 0:
    should_stop = torch.tensor([1 if patience_counter >= args.patience else 0], device=device)
else:
    should_stop = torch.tensor([0], device=device)  # Init z 0, nie z decyzji!
dist.broadcast(should_stop, src=0)

if should_stop.item() == 1:
    break
```

### 1.5 Barrier na końcu epoki
```python
# Po zakończeniu walidacji i zapisie checkpointu:
dist.barrier()  # Synchronizacja przed następną epoką
```

### 1.6 LR Scheduler
```python
# Tworzenie schedulera
if args.lr_scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min  # np. 1e-7
    )

# Po każdej epoce (po walidacji):
scheduler.step()

# Zapis/odczyt w checkpoint:
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # WAŻNE!
    'epoch': epoch,
    ...
}
```

---

## 2. Konfiguracja SLURM (`run_detr_*.slurm`)

### 2.1 Nagłówek SLURM
```bash
#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p experimental           # lub hopper dla H100
#SBATCH --gres=gpu:4              # liczba GPU
#SBATCH --mem=250G
#SBATCH --time=5-00:00:00         # 5 dni
#SBATCH --job-name=detr_20k_finetune
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR
#SBATCH --output=logs/detr_%j.log
#SBATCH --error=logs/detr_%j.log
#SBATCH --open-mode=append
```

### 2.2 Zmienne NCCL (KRYTYCZNE!)
```bash
# DDP
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export OMP_NUM_THREADS=1

# NCCL - zapobieganie timeout
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,NET
export NCCL_TIMEOUT=1800              # 30 min (default ~10 min)
export NCCL_ASYNC_ERROR_HANDLING=1    # deprecated, ale zostawić
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
```

### 2.3 Komenda torchrun
```bash
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29501 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29501 \
    detr_train_optimized.py \
    --images_dir "$IMAGES_DIR" \
    --annotations_path "$ANNOT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
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

---

## 3. Przygotowanie danych

### 3.1 Format COCO
```json
{
  "images": [
    {"id": 1, "file_name": "frame_00001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, width, height], "area": 1000, "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "tooltip"}
  ]
}
```

### 3.2 Krytyczne wymagania
| Aspekt | Wymaganie |
|--------|-----------|
| `category_id` | Musi być **1** (nie 0!) i zgodne między train/eval |
| `bbox` format | `[x, y, width, height]` w pikselach (NIE `[x1,y1,x2,y2]`!) |
| `file_name` | Musi dokładnie odpowiadać nazwom plików obrazów |
| Współrzędne | W pikselach, NIE znormalizowane [0,1] |
| Zakresy | `x >= 0`, `y >= 0`, `x+w <= width`, `y+h <= height` |

---

## 4. Typowe błędy i rozwiązania

### 4.1 NCCL Timeout
```
WorkNCCL(SeqNum=8898, OpType=ALLREDUCE) ran for 600236 milliseconds before timing out
```
**Rozwiązanie:**
- Zwiększ `NCCL_TIMEOUT=1800`
- Dodaj barrier przed all_reduce
- Usuń barrier przy None batch

### 4.2 DDP Desynchronization
```
RuntimeError: Detected mismatch between collectives on ranks.
Rank 3 is running BROADCAST, but Rank 1 is running REDUCE
```
**Rozwiązanie:**
- Usuń `dist.barrier()` przy None batch
- Popraw broadcast early stopping (rank 0 tworzy tensor, inne init z 0)

### 4.3 NaN w wagach
**Rozwiązanie:**
- Wyłącz Mixed Precision: `--use_amp False` (lub usuń flagę)
- Zmniejsz learning rate
- Sprawdź gradient clipping: `--max_grad_norm 0.1`

### 4.4 mAP ~1% przy ewaluacji
**Rozwiązanie:**
- Sprawdź `category_id` zgodność (GT vs predykcje)
- Sprawdź format bbox (`[x,y,w,h]` vs `[x1,y1,x2,y2]`)
- Sprawdź `file_name` zgodność

### 4.5 Oscylujący loss
```
Epoch 100: 0.192 → Epoch 120: 0.232 → Epoch 140: 0.244
```
**Rozwiązanie:**
- Dodaj LR Scheduler (cosine annealing)
- Zmniejsz bazowy LR dla fine-tuningu (np. 5e-5 zamiast 1e-4)

---

## 5. Checklist przed uruchomieniem

- [ ] `@lru_cache` usunięty z kodu
- [ ] Brak `dist.barrier()` przy None batch
- [ ] `dist.barrier()` przed all_reduce na końcu epoki
- [ ] Early stopping z broadcast (rank 0 decyduje)
- [ ] LR Scheduler skonfigurowany (cosine)
- [ ] Scheduler state zapisywany w checkpoint
- [ ] `NCCL_TIMEOUT=1800` w SLURM
- [ ] `drop_last=True` w DataLoader
- [ ] `category_id=1` w annotations JSON
- [ ] Bbox w formacie `[x,y,w,h]` w pikselach
- [ ] Mixed Precision wyłączony (dla stabilności)

---

## 6. Komendy SSH do Eden

```bash
# Konfiguracja SSH (~/.ssh/config)
Host eden-jump
    HostName ssh.mini.pw.edu.pl
    User piotrowskib2
    ServerAliveInterval 60

Host eden-cluster
    HostName eden
    User bpiotrowski
    ProxyJump eden-jump
    ServerAliveInterval 60

# Upload plików
scp -o ProxyJump=eden-jump "local_file.py" bpiotrowski@eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/

# Uruchomienie treningu
ssh eden-cluster "cd /mnt/evafs/faculty/home/bpiotrowski/DETR && sbatch run_detr_20k_finetune.slurm"

# Monitoring
ssh eden-cluster "squeue -u bpiotrowski"
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_*.log"
```

---

## 7. Hiperparametry rekomendowane

| Parametr | Wartość | Uwagi |
|----------|---------|-------|
| LR (główny) | 5e-5 | Dla fine-tuningu; 1e-4 dla treningu od zera |
| LR (backbone) | 5e-6 | 10x mniejszy niż główny |
| LR Scheduler | cosine | Z `lr_min=1e-7` |
| Batch size/GPU | 4-8 | Zależnie od pamięci GPU |
| Gradient clipping | 0.1 | `max_grad_norm` |
| Weight decay | 1e-4 | |
| Patience | 30 | Early stopping |
| Save interval | 5-10 | Epoki między checkpointami |

---

## Powiązane pliki

### Skrypty treningowe:
- `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py`
- `Eden/Scripts/20kDataset_11.12.25/run_detr_20k_finetune.slurm`
- `Eden/Scripts/20kDataset_11.12.25/launch_tensorboard.sh`

### Narzędzia diagnostyczne:
- `Eden/Checkpoints/DETR/analyze_checkpoints.py`

### Poprzednie notatki:
- `Notatki/2025-10-31_DETR_EDEN_training_podsumowanie.md`
- `Notatki/2025-10-31_Benchmark_DETR_vs_YOLO_wnioski.md`
- `Notatki/2025-12-12_DETR_Training_NCCL_Timeout_LR_Scheduler.md`

---

*Notatka wygenerowana: 2025-12-12*
*Projekt: ViTParticleFilterTracker*
