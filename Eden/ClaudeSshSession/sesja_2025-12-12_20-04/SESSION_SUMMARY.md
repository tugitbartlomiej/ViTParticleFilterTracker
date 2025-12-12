# Sesja SSH Eden - 2025-12-12 20:04 CET

**Data:** 2025-12-12
**Godzina:** 20:04 CET (19:04 UTC)
**Job ID:** 1454279
**Node:** pascal
**Status:** RUNNING

---

## Podsumowanie Sesji

Naprawa krytycznego błędu DDP desync przy przejściu epoch 171→172. Problem: walidacyjny DataLoader i Sampler nie miały `drop_last=True`, co powodowało różną liczbę batchy na różnych rankach → deadlock.

### Główne osiągnięcia:
1. Zidentyfikowano root cause: brak `drop_last=True` w val_sampler i val_dataloader
2. Naprawiono `val_sampler` - dodano `drop_last=True` do DistributedSampler
3. Naprawiono `val_dataloader` - dodano `drop_last=True` do DataLoader
4. Naprawiono `find_latest_checkpoint()` - teraz sprawdza `final_checkpoint.pth` jako pierwszą opcję
5. Wznowiono trening z `final_checkpoint.pth` (epoch 171)
6. Job 1454279 działa stabilnie - epoch 172 w trakcie (21% @ 20:04)

---

## Struktura Katalogów na Eden

```
/mnt/evafs/faculty/home/bpiotrowski/DETR/
├── detr_train_optimized.py          # Główny skrypt treningu (ZMODYFIKOWANY - drop_last fix)
├── run_detr_20k_finetune.slurm      # SLURM job script
│
├── train_out_20k_finetune/          # OUTPUT_DIR
│   ├── logs/                        # TensorBoard events
│   └── final_model/                 # Model HuggingFace po zakończeniu
│
├── ckpt_20k_finetune/               # CHECKPOINT_DIR
│   ├── checkpoint_epoch_170.pth     # Checkpoint przed crashem
│   └── final_checkpoint.pth         # Checkpoint z epoch 171 (używany do resume)
│
├── best_20k_finetune/               # BEST_MODEL_DIR
│
├── Checkpoints/                     # ARCHIVE_DIR
│   └── 20k_finetune/
│       ├── checkpoint_epoch_170.pth
│       └── final_checkpoint.pth
│
└── logs/                            # Logi SLURM
    └── detr_20k_finetune_1454279.log
```

---

## Kluczowe Poprawki Kodu

### 1. val_sampler z drop_last (linie 388-396)
```python
val_sampler = DistributedSampler(
    val_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
    drop_last=True  # CRITICAL: ensures same batch count across all ranks
)
```

### 2. val_dataloader z drop_last (linie 411-423)
```python
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=val_sampler,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=True,
    persistent_workers=True if args.num_workers > 0 else False,
    prefetch_factor=2 if args.num_workers > 0 else None,
    drop_last=True,  # CRITICAL: ensures same batch count across all ranks
)
```

### 3. find_latest_checkpoint() - priorytet final_checkpoint.pth
```python
def find_latest_checkpoint(checkpoint_dir):
    # First check for final_checkpoint.pth (saved on crash/completion)
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if final_checkpoint.exists():
        print(f"[Checkpoint] Found final_checkpoint.pth - using it for resume")
        return final_checkpoint
    # Then look for epoch checkpoints...
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

### Argumenty treningu
```bash
--lr 5e-5 --lr_backbone 5e-6 --lr_scheduler cosine --lr_min 1e-7
--batch_size 4 --epochs 300 --patience 30 --resume_training
```

---

## Ważne Ścieżki

| Ścieżka | Opis |
|---------|------|
| `/mnt/evafs/faculty/home/bpiotrowski/DETR/` | Główny katalog projektu |
| `./ckpt_20k_finetune/final_checkpoint.pth` | Checkpoint użyty do resume (epoch 171) |
| `./logs/detr_20k_finetune_1454279.log` | Log aktualnego joba |

---

## Komendy SSH

### Monitoring
```bash
# Status joba
ssh eden-cluster "squeue -u bpiotrowski"

# Logi na żywo
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_20k_finetune_1454279.log"

# Tylko postęp
ssh eden-cluster "tail -30 /path/to/log | grep -E 'Training|Validation|Epoch'"
```

### Upload
```bash
scp -o ProxyJump=eden-jump "detr_train_optimized.py" bpiotrowski@eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/
```

---

## Problemy i Rozwiązania

### Problem: DDP Desync przy Validation
- **Objaw:** `RuntimeError: Detected mismatch between collectives on ranks. Rank 1 running BROADCAST, Rank 0 running REDUCE`
- **Przyczyna:** `val_sampler` i `val_dataloader` bez `drop_last=True` → różne ranki mają różną liczbę batchy
- **Rozwiązanie:** Dodanie `drop_last=True` do obu

### Problem: Resume nie znajduje final_checkpoint.pth
- **Przyczyna:** `find_latest_checkpoint()` szukało tylko `checkpoint_epoch_*.pth`
- **Rozwiązanie:** Dodanie sprawdzenia `final_checkpoint.pth` jako pierwszej opcji

---

## Status na Koniec Sesji

```
Job ID:     1454279
Status:     RUNNING
Node:       pascal
GPUs:       4x P100
Runtime:    ~19 min
Epoch:      172/300
Progress:   21% (234/1125 batches)
Loss:       0.4300
Speed:      2.91s/it
LR:         Cosine (starting 5e-5)
```

Trening przebiega stabilnie. Kluczowy test: czy walidacja przejdzie bez desync.

---

## Pliki Zmodyfikowane w Sesji

### Na Eden:
- `detr_train_optimized.py` - poprawki drop_last + find_latest_checkpoint

### Lokalnie:
- `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py`
- `.sessions/Session_2025-12-12_200437/`
- `Eden/ClaudeSshSession/sesja_2025-12-12_20-04/`

---

*Sesja zapisana: 2025-12-12 20:04 CET*
*Projekt: ViTParticleFilterTracker*
