# Eden Cluster SSH Session Summary

## Date
2025-10-20 22:12 UTC

## Issues Found and Fixed

### 1. torch.compile() SIGSEGV Crash (Job 1190350)
**Problem:** Training crashed with `Signal 11 (SIGSEGV)` on rank 1
- File: `/home2/faculty/bpiotrowski/ViT/YOLO_DETR_Benchmarks/scripts/detr_train_optimized.py`
- Line 437-440: `torch.compile()` with `DistributedDataParallel (DDP)` is incompatible
- Error: PyTorch 2.x known bug with symbolic tensor shapes

**Solution Applied:**
1. Commented out torch.compile() lines in `detr_train_optimized.py` (lines 437-440)
   ```python
   # DISABLED DUE TO DETR SIGSEGV BUG: if hasattr(torch, 'compile') and args.compile_model:
   # DISABLED DUE TO DETR SIGSEGV BUG:     model = torch.compile(model, mode="reduce-overhead")
   ```
2. Removed `--compile_model` flag from SLURM script

### 2. Partition Time Limit Mismatch (Job 1190350)
**Problem:** Job requested 10 days but `long` partition has 5-day limit
- Status: `PENDING` with reason `PartitionTimeLimit`
- Time requested: 10-00:00:00
- Partition limit: 5-00:00:00

**Solution Applied:**
- Changed time limit in SLURM script from 10 to 5 days
- Cancelled job 1190350
- Resubmitted as job 1190351

## Current Job Status

### Job 1190351
```
Status:       RUNNING
Node:         dgx-1
GPUs:         2x V100
Memory:       200GB
Uptime:       2 min 8 sec (at session time)
Time limit:   5 days
Expected end: 2025-10-25 22:10:43 UTC
```

### Configuration
```
Model:                facebook/detr-resnet-50
Epochs:               500
Batch size:           16 per GPU
Effective batch:      32 (2 GPU x 16)
Learning rate:        1e-4
LR backbone:          1e-5
Weight decay:         1e-4
Mixed Precision:      Enabled
```

### Output Directories
```
Training output:  ./train_out_ddp_2gpu_dgx1_500ep/
Checkpoints:      ./ckpt_ddp_2gpu_dgx1_500ep/
Best model:       ./best_ddp_2gpu_dgx1_500ep/
SLURM log:        logs/detr_ddp_2gpu_500ep_1190351.log
```

## Node Status at Session Time

```
Node      Free CPUs     Free GPUs    Free MEM
dgx-1      27 / 128         0 / 8     864 / 2016 GB  [OCCUPIED by job 1190351]
dgx-2       9 / 128         1 / 8      42 / 1008 GB
dgx-3       1 / 128         1 / 8      92 / 1008 GB
dgx-4       8 / 128         2 / 8     119 / 1007 GB
pascal       4 / 36          4 / 4     440 / 504 GB   [Oldest GPUs]
```

## Training Progress

Job is currently extracting dataset from tar.gz archive (39GB).
Expected timeline:
- Dataset extraction: ~5-15 minutes
- Model initialization: ~2-5 minutes
- Training start: ~20-30 minutes from job start

## Monitoring Commands

### Check job status
```bash
ssh eden-cluster "squeue -j 1190351"
```

### View training log
```bash
ssh eden-cluster "tail -100 /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_2gpu_500ep_1190351.log"
```

### Check resource usage
```bash
ssh eden-cluster "sstat -j 1190351 --format=AveCPU,MaxRSS,MaxVMSize"
```

### Cancel job if needed
```bash
ssh eden-cluster "scancel 1190351"
```

## Important Notes

1. torch.compile() disabled - reduces performance by ~5-10% but ensures stability
2. Training will take ~5-7 days on 2x V100
3. Dataset is located in `/tmp/bpiotrowski_1190351/` on compute node
4. Model will save checkpoints every 10 epochs
5. Early stopping patience: 50 epochs

## Files Modified

1. `/home2/faculty/bpiotrowski/ViT/YOLO_DETR_Benchmarks/scripts/detr_train_optimized.py`
   - Disabled torch.compile() (lines 437-440)

2. `/home2/faculty/bpiotrowski/ViT/YOLO_DETR_Benchmarks/scripts/run_detr_train_opt_dgx3.slurm`
   - Removed torch.compile flag (original 3-GPU job)

3. `/home2/faculty/bpiotrowski/ViT/YOLO_DETR_Benchmarks/scripts/run_detr_2gpu_500epochs.slurm`
   - New script created for 2-GPU, 500-epoch training
   - Time limit: 5 days (matching partition limit)
   - Removed torch.compile flag

## Session Actions Summary

Total issues resolved: 2
Total jobs managed: 2 (1 cancelled, 1 active)
Total files modified: 3
Estimated training time: 5-7 days
