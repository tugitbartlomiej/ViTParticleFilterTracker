# Eden Cluster SSH Session Archive
**Session Date:** 2025-10-20 22:12 UTC

## Directory Contents

### Documentation
- **SESSION_SUMMARY.md** - Complete session overview with all actions taken
- **README.md** - This file

### Configuration & Scripts
- **run_detr_2gpu_500epochs.slurm** - SLURM job script for 2-GPU training on DGX-1
- **detr_train_optimized_excerpt.py** - Training script excerpt showing key sections

### Logs & Status
- **training_log.txt** - Current training output (updated at session end)
- **cluster_status.txt** - Cluster node availability snapshot
- **torch_compile_fix.txt** - Code section showing the torch.compile() bug fix

## Quick Summary

### Issues Resolved
1. **SIGSEGV Crash** - torch.compile() incompatible with DDP
   - Solution: Disabled torch.compile() in training script
   - Status: FIXED

2. **Job Timeout** - Time limit exceeded partition maximum
   - Requested: 10 days
   - Partition max: 5 days
   - Solution: Changed to 5 days
   - Status: FIXED

### Current Job
- **Job ID:** 1190351
- **Status:** RUNNING on DGX-1
- **Configuration:** 2x V100 GPUs, 500 epochs, batch size 16
- **Expected Duration:** 5-7 days
- **Time Limit:** 2025-10-25 22:10 UTC

### Key Metrics
```
Effective batch size:  32 (2 GPU x 16)
Learning rate:         1e-4
Mixed Precision:       Enabled
Early stopping patience: 50 epochs
```

## Monitoring Job

Check job status:
```bash
ssh eden-cluster "squeue -j 1190351"
```

View latest logs:
```bash
ssh eden-cluster "tail -100 /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_2gpu_500ep_1190351.log"
```

## Important Files on Eden

- **SLURM script:** `/home2/faculty/bpiotrowski/ViT/YOLO_DETR_Benchmarks/scripts/run_detr_2gpu_500epochs.slurm`
- **Training script:** `/home2/faculty/bpiotrowski/ViT/YOLO_DETR_Benchmarks/scripts/detr_train_optimized.py`
- **Training log:** `/mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_2gpu_500ep_1190351.log`
- **Output directory:** `/mnt/evafs/faculty/home/bpiotrowski/DETR/train_out_ddp_2gpu_dgx1_500ep/`

## Training Output Locations

- Checkpoints: `./ckpt_ddp_2gpu_dgx1_500ep/`
- Best model: `./best_ddp_2gpu_dgx1_500ep/`
- Training logs: `./train_out_ddp_2gpu_dgx1_500ep/`

## Next Steps

1. Monitor job completion (check back after 5-7 days)
2. Verify model quality metrics
3. Transfer results back to local machine
4. Update project documentation with results

## Session Archive Purpose

This directory contains a snapshot of the entire SSH session:
- All configuration files used
- All scripts executed
- Current logs and status
- Bug fixes applied
- Cluster status at session time

Use this for reference, debugging, or documentation purposes.
