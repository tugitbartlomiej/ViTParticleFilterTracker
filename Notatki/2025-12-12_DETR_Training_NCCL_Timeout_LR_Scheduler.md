# DETR Training: Rozwiązanie NCCL Timeout + LR Scheduler

**Data:** 2025-12-12
**Kontekst:** Trening DETR na Eden (cluster SLURM) zakończył się błędem NCCL timeout na epoce 171. Analiza checkpointów wykazała problemy ze stałym learning rate.

---

## Główne Wnioski

### Problem 1: NCCL Timeout (crash treningu)
- **Objaw:** `WorkNCCL(SeqNum=8898, OpType=ALLREDUCE) ran for 600236 milliseconds before timing out`
- **Przyczyna:** Desynchronizacja procesów DDP - jeden GPU kończył szybciej niż inne
- **Lokalizacja błędu:** Epoch 171, krok 1121/1125 (99.6% ukończenia epoki)

### Problem 2: Oscylujący Loss (analiza checkpointów)
- Loss nie spadał monotonicznie: `0.192 → 0.232 → 0.244 → 0.159 → 0.163`
- Stały LR=1e-4 przez cały trening powodował "skakanie" w późnych epokach
- Brak LR scheduler = model nie konwerguje stabilnie

---

## Szczegóły

### Rozwiązanie NCCL Timeout

**Zmiany w `detr_train_optimized.py`:**

1. **Usunięcie `@lru_cache`** - nie jest DDP-safe, powoduje różnice w timing między procesami

2. **Dodanie `dist.barrier()` przy None batch:**
```python
if batch is None:
    dist.barrier()  # Synchronizacja wszystkich procesów
    continue
```

3. **Barrier przed aggregacją loss:**
```python
dist.barrier()  # Poczekaj na wszystkie GPU
train_loss_tensor = torch.tensor(train_loss).to(device)
dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
```

4. **Broadcast decyzji early stopping:**
```python
should_stop = torch.tensor([1 if patience_counter >= args.patience else 0], device=device)
dist.broadcast(should_stop, src=0)  # Wszystkie procesy kończą razem
```

**Zmiany w SLURM script:**
```bash
export NCCL_TIMEOUT=1800  # 30 min zamiast 10
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Rozwiązanie problemu LR

**Implementacja LR Scheduler:**

| Scheduler | Opis | Kiedy używać |
|-----------|------|--------------|
| `cosine` | Cosine annealing | Domyślny, płynne zmniejszanie |
| `cosine_warmup` | Warmup + cosine | Gdy model niestabilny na początku |
| `step` | StepLR | Spadek co N epok |
| `plateau` | ReduceOnPlateau | Automatyczna redukcja |

**Nowe argumenty:**
```bash
--lr_scheduler cosine
--lr_min 1e-7
--warmup_epochs 5  # dla cosine_warmup
```

### Analiza Checkpointów (wyniki)

```
Epoch    Loss       LR (main)    Weight Norm   Status
60       0.2161     1.00e-04     8.4487        OK
100      0.1921     1.00e-04     8.5007        OK (najlepszy do tej pory)
120      0.2324     1.00e-04     8.5754        Wzrost!
140      0.2443     1.00e-04     8.6531        Wzrost!
160      0.1587     1.00e-04     8.6885        NAJLEPSZY
170      0.1628     1.00e-04     8.7076        Lekki wzrost
```

**Wnioski z analizy:**
- Brak NaN/Inf - wagi zdrowe
- Weight norm stabilny (+3% przez 110 epok) - normalne
- Adam moments stabilne (v ≈ 0.000005)
- Problem: stały LR powoduje oscylacje

---

## Powiązane Pliki

### Zmodyfikowane:
- `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py` - główny skrypt treningu
- `Eden/Scripts/20kDataset_11.12.25/run_detr_20k_finetune.slurm` - SLURM job

### Utworzone:
- `Eden/Checkpoints/DETR/analyze_checkpoints.py` - skrypt analizy checkpointów
- `Eden/Checkpoints/DETR/checkpoint_analysis.json` - wyniki analizy

### Checkpointy:
- `Eden/Checkpoints/DETR/checkpoint_epoch_60.pth`
- `Eden/Checkpoints/DETR/checkpoint_epoch_100.pth`
- `Eden/Checkpoints/DETR/checkpoint_epoch_120.pth`
- `Eden/Checkpoints/DETR/checkpoint_epoch_140.pth`
- `Eden/Checkpoints/DETR/checkpoint_epoch_160.pth`
- `Eden/Checkpoints/DETR/checkpoint_epoch_170.pth`

---

## Komendy do wznowienia treningu

```bash
# Wgraj poprawione pliki na Eden
scp "Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py" eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/
scp "Eden/Scripts/20kDataset_11.12.25/run_detr_20k_finetune.slurm" eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/

# Uruchom trening
ssh eden
cd /mnt/evafs/faculty/home/bpiotrowski/DETR
sbatch run_detr_20k_finetune.slurm
```

---

## Zalecenia na przyszłość

1. **Zawsze używaj LR scheduler** - cosine annealing jako domyślny
2. **Włącz TensorBoard logging** - dla pełnej analizy krzywych uczenia
3. **Monitoruj NCCL_DEBUG=INFO** - dla wczesnego wykrywania problemów synchronizacji
4. **Zapisuj checkpoint_analysis.json** - dla retrospektywnej analizy treningu

---

*Notatka wygenerowana: 2025-12-12*
*Projekt: ViTParticleFilterTracker*
