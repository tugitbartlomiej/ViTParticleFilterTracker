# Session Summary: DETR_Training_DDP_Validation_Fix

## Metadata
- **Date:** 2025-12-12
- **Time:** 19:45-20:04 CET
- **Status:** In Progress (monitoring)
- **Type:** SSH/Training
- **Job ID:** 1454279
- **Node:** pascal (4x P100 GPUs)

## Objective
Fix DDP desynchronization crash that occurred at epoch 171→172 transition in DETR training on Eden cluster.

## Context
This session continues from session `sesja_2025-12-12_18-18` where:
- Job 1454252 crashed after completing epoch 171
- Error: `RuntimeError: Detected mismatch between collectives on ranks. Rank 1 is running BROADCAST, Rank 0 is running REDUCE`
- `final_checkpoint.pth` was saved before crash (epoch 171 state)

## Root Cause Analysis

The crash occurred because **validation DataLoader** was missing `drop_last=True`:
1. With 20k images, 90% train / 10% val split → ~2000 validation images
2. 4 GPUs × batch_size 4 = 16 images per step
3. 2000 / 16 = 125 batches... but with remainder!
4. Different ranks received different batch counts → DDP desync during validation

## Actions Taken

### 1. Identified Missing `drop_last` in Validation
```python
# Problem: val_sampler missing drop_last
val_sampler = DistributedSampler(val_dataset, shuffle=False)  # NO drop_last!

# Problem: val_dataloader missing drop_last
val_dataloader = DataLoader(val_dataset, sampler=val_sampler)  # NO drop_last!
```

### 2. Applied Fix to `detr_train_optimized.py`

**Fix 1: val_sampler (lines 388-396)**
```python
val_sampler = DistributedSampler(
    val_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
    drop_last=True  # CRITICAL: ensures same batch count across all ranks
)
```

**Fix 2: val_dataloader (lines 411-423)**
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

**Fix 3: find_latest_checkpoint() (lines 208-240)**
```python
def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file based on epoch number or final_checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # First check for final_checkpoint.pth (saved on crash/completion)
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if final_checkpoint.exists():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Found final_checkpoint.pth - using it for resume")
        return final_checkpoint

    # Then look for epoch checkpoints
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    # ... rest of function
```

### 3. Uploaded and Started New Job
```bash
scp -o ProxyJump=eden-jump "detr_train_optimized.py" bpiotrowski@eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/
ssh eden-cluster "cd /mnt/evafs/faculty/home/bpiotrowski/DETR && sbatch run_detr_20k_finetune.slurm"
# Job ID: 1454279
```

### 4. Verified Resume from final_checkpoint.pth
```
[Checkpoint] Found final_checkpoint.pth - using it for resume
[Checkpoint] Loading checkpoint from: ckpt_20k_finetune/final_checkpoint.pth
[Checkpoint] LR scheduler state loaded
[Checkpoint] Successfully loaded state from epoch 171
Starting training loop from epoch 171...
--- Epoch 172/300 ---
Training E172:   0%|          | 0/1125 [00:00<?, ?it/s]
```

## Results

### Key Findings
1. **Root cause:** Validation DataLoader without `drop_last=True` causes DDP desync when val_dataset size not perfectly divisible by (world_size × batch_size)
2. **Solution:** Add `drop_last=True` to both `DistributedSampler` and `DataLoader` for validation
3. **Checkpoint resume:** Modified `find_latest_checkpoint()` to prioritize `final_checkpoint.pth`

### Current Status (as of 20:04 CET)
```
Job ID:     1454279
Status:     RUNNING
Node:       pascal (4x P100)
Runtime:    19:06
Epoch:      172/300
Progress:   21% (234/1125 batches)
Loss:       0.4300
Speed:      2.91s/it
ETA:        ~43 min remaining for epoch 172
```

### Issues Encountered
1. **Initial confusion:** SLURM script echoes "Resume from checkpoint_epoch_170.pth" but Python actually loads `final_checkpoint.pth`
2. **Slow first batch:** First batch took 72.56s/it (normal - JIT compilation), then stabilized at 2.9s/it

## Files Generated/Modified

### On Eden:
- `/mnt/evafs/faculty/home/bpiotrowski/DETR/detr_train_optimized.py` - DDP fixes

### Locally:
- `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py` - Updated local copy
- `.sessions/Session_2025-12-12_200437/` - This session

## Commands Used

### SSH Monitoring
```bash
# Check job status
ssh eden-cluster "squeue -u bpiotrowski"

# View logs
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_20k_finetune_1454279.log"

# Check training progress
ssh eden-cluster "tail -30 /path/to/log | grep -E 'Training|Validation|Epoch'"

# Upload fixed script
scp -o ProxyJump=eden-jump "detr_train_optimized.py" bpiotrowski@eden:/mnt/evafs/faculty/home/bpiotrowski/DETR/
```

## Next Steps
- [ ] Monitor epoch 172 completion and validation phase
- [ ] Verify validation completes without DDP desync (CRITICAL TEST)
- [ ] Confirm epoch 173 starts successfully
- [ ] If successful, let training continue to epoch 300

## Related Sessions
- `Eden/ClaudeSshSession/sesja_2025-12-12_18-18/` - Previous session with NCCL timeout fix
- `Notatki/TreningUlepszanieZbioru/2025-12-12_DETR_Training_Checklist_Poprawnego_Uruchomienia.md` - Training checklist

---

*Session saved: 2025-12-12 20:04 CET*
*Project: ViTParticleFilterTracker*
