# DETR Checkpoint Archival System Implementation

**Implementation Date:** 2025-10-20
**Job ID:** 1190356
**Script:** run_detr_2gpu_500epochs_with_archival.slurm

## Problem Solved

Training was starting from epoch 0 instead of resuming from checkpoint_epoch_125.pth. Additionally, checkpoints need to be archived to permanent storage for backup and recovery.

## Solution Overview

Enhanced SLURM script with automatic checkpoint archival system that:
1. Resumes training from latest checkpoint (epoch 125)
2. Periodically syncs checkpoints to archive directory every 10 minutes
3. Ensures final sync on job completion (success or failure)

## Implementation Details

### Checkpoint Directories

- **Working Directory:** `./ckpt_ddp_2gpu_dgx1_500ep/`
  - Used during training
  - Contains checkpoint_epoch_125.pth (starting point)

- **Archive Directory:** `/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/`
  - Permanent storage location
  - Synchronized every 10 minutes during training
  - Final sync on job completion

### Key Features

1. **Periodic Sync (Background Process)**
   - Runs every 10 minutes (600 seconds)
   - Uses rsync with --update flag (only newer files)
   - Non-blocking - runs in parallel with training
   - PID tracked for cleanup

2. **Exit Trap Handler**
   - Triggered on EXIT, INT, TERM signals
   - Ensures checkpoints are saved even if job crashes
   - Performs final sync before job termination

3. **Resume Training**
   - `--resume_training` flag enabled
   - Automatically loads latest checkpoint from working directory
   - Should resume from epoch 126 (after checkpoint_epoch_125.pth)

### Code Structure

```bash
# Checkpoint archival setup
ARCHIVE_DIR=/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints
CHECKPOINT_DIR=./ckpt_ddp_2gpu_dgx1_500ep

# Sync function
sync_checkpoints() {
    echo "[$(date)] Syncing checkpoints to archive..."
    rsync -av --update "$CHECKPOINT_DIR/" "$ARCHIVE_DIR/"
    echo "[$(date)] Sync completed"
}

# Exit trap
trap 'echo "EXIT trap triggered"; sync_checkpoints' EXIT INT TERM

# Background sync loop (every 10 minutes)
(
    while true; do
        sleep 600
        sync_checkpoints
    done
) &
SYNC_PID=$!

# Training with --resume_training flag
torchrun ... detr_train_optimized.py ... --resume_training

# Cleanup
kill $SYNC_PID 2>/dev/null || true
sync_checkpoints  # Final sync
```

## Files Modified

1. **run_detr_2gpu_500epochs_with_archival.slurm** (NEW)
   - Enhanced version with checkpoint archival
   - Location: `/home2/faculty/bpiotrowski/ViT/YOLO_DETR_Benchmarks/scripts/`

2. **checkpoint_epoch_125.pth** (COPIED)
   - Copied from archive to working directory
   - Source: `/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/`
   - Destination: `./ckpt_ddp_2gpu_dgx1_500ep/`
   - Size: 475MB

## Deployment Steps

1. Cancelled previous job 1190351 (training from epoch 0)
2. Copied checkpoint_epoch_125.pth to working directory
3. Created enhanced SLURM script with archival
4. Fixed DOS line endings (CRLF → LF)
5. Submitted job 1190356

## Verification Checklist

- [x] Job submitted successfully (ID: 1190356)
- [x] Running on dgx-1 with 2 GPUs
- [ ] Training resumed from epoch 126 (to be verified)
- [ ] Background sync process started (to be verified)
- [ ] Checkpoints appearing in archive directory (to be verified)

## Monitoring

Check training progress:
```bash
ssh eden-cluster "tail -f /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_2gpu_500ep_1190356.log"
```

Verify checkpoint sync:
```bash
ssh eden-cluster "ls -lh /mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/"
```

Check job status:
```bash
ssh eden-cluster "squeue -u bpiotrowski"
```

## Expected Behavior

1. Job starts and extracts dataset to /tmp
2. Training script loads checkpoint_epoch_125.pth
3. Prints "Resuming from epoch 126" or similar
4. Background sync starts and logs every 10 minutes
5. Checkpoints saved to working dir every 10 epochs
6. Each checkpoint synced to archive within 10 minutes
7. On completion, final sync ensures all checkpoints archived

## Troubleshooting

If training starts from epoch 0 again:
- Check checkpoint file exists in working directory
- Verify --resume_training flag is set
- Check detr_train_optimized.py checkpoint loading logic

If sync fails:
- Verify archive directory permissions
- Check disk space in archive location
- Review rsync output in logs

## Session Archive

Session summary available at:
`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\ClaudeSshSession\sesja_2025-10-20_23-XX\`

---
**Status:** Implementation complete, training in progress
**Next Steps:** Monitor training logs to verify checkpoint resume and archival functioning
