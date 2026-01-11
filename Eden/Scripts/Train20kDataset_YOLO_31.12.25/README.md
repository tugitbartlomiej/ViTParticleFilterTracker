# YOLO 20k Fine-Tuning Scripts

Fine-tuning YOLOv8 on 20,000 intelligently selected images from the Advanced Dataset Selection pipeline.

## Overview

This training continues from **epoch 170** checkpoint, using the same dataset that achieved **37x improvement** in cross-dataset generalization for DETR (2% → 74.57% mAP).

## Dataset

- **Source**: Advanced Dataset Selection pipeline output
- **Size**: 20,000 carefully selected images from 90,141 augmented pool
- **Selection criteria**: DETR EL2N scoring + diversity sampling
- **Format**: COCO annotations (converted to YOLO format during training)
- **Archive**: `/mnt/evafs/faculty/home/bpiotrowski/20kSelectedImages.tar`
- **Annotations**: `/mnt/evafs/faculty/home/bpiotrowski/merged_20k_annotations.json`

## Files

| File | Description |
|------|-------------|
| `yolo_train_20k_finetune.py` | Main training script with YOLOv8 Ultralytics API |
| `run_yolo_20k_finetune.slurm` | SLURM script for DGX-2 (4x V100 32GB) |
| `run_yolo_20k_finetune_dgx1.slurm` | SLURM script for DGX-1 (8x V100 32GB) |
| `run_yolo_20k_finetune_pascal.slurm` | SLURM script for Pascal GPUs (2x 16GB) |
| `launch_tensorboard.sh` | TensorBoard monitoring script |

## Training Configuration

| Parameter | DGX-2 | DGX-1 | Pascal |
|-----------|-------|-------|--------|
| GPUs | 4x V100 | 8x V100 | 2x Pascal |
| Batch/GPU | 32 | 32 | 16 |
| Effective batch | 128 | 256 | 32 |
| Learning rate | 0.001 | 0.001 | 0.0005 |
| Epochs | 300 | 300 | 300 |
| Image size | 640 | 640 | 640 |
| Time limit | 5 days | 5 days | 7 days |

## Key Differences from Initial Training

1. **Reduced Learning Rate**: 0.001 vs 0.01 (10x lower for fine-tuning)
2. **Resume from checkpoint**: epoch170.pt instead of pretrained weights
3. **Curated dataset**: 20k selected vs 90k augmented images
4. **Extended patience**: 30 epochs early stopping

## Quick Start

### 1. Upload to Eden

```bash
scp -r Eden/Scripts/Train20kDataset_YOLO_31.12.25/* \
    bpiotrowski@eden.ue.katowice.pl:/mnt/evafs/faculty/home/bpiotrowski/YOLO_20k/
```

### 2. Prepare Checkpoint

Ensure `epoch170.pt` is in the correct location:
```bash
# On Eden
ls -la /mnt/evafs/faculty/home/bpiotrowski/YOLO_Checkpoints/epoch170.pt
```

If not present, copy from training output:
```bash
cp /path/to/YOLO_EDEN_TRAIN/exp/weights/epoch170.pt \
   /mnt/evafs/faculty/home/bpiotrowski/YOLO_Checkpoints/
```

### 3. Submit Job

```bash
# DGX-2 (recommended - 4 GPUs)
sbatch run_yolo_20k_finetune.slurm

# DGX-1 (8 GPUs, fastest)
sbatch run_yolo_20k_finetune_dgx1.slurm

# Pascal (fallback option)
sbatch run_yolo_20k_finetune_pascal.slurm
```

### 4. Monitor Training

```bash
# Check job status
squeue -u $USER

# View logs
tail -f /mnt/evafs/faculty/home/bpiotrowski/YOLO_20k/logs/yolo_20k_finetune_*.log

# Start TensorBoard (in separate terminal)
./launch_tensorboard.sh 6007

# SSH tunnel for TensorBoard (from local machine)
ssh -L 6007:localhost:6007 bpiotrowski@eden.ue.katowice.pl
```

## Output Structure

```
/mnt/evafs/faculty/home/bpiotrowski/YOLO_20k/
├── logs/                    # SLURM job logs
├── train_out_20k/           # Training output (Ultralytics format)
│   └── exp/
│       ├── weights/
│       │   ├── best.pt
│       │   ├── last.pt
│       │   └── epoch*.pt
│       └── results.csv
├── ckpt_yolo_20k/           # Periodic checkpoint backups
├── best_yolo_20k/           # Best model copy
└── yolo_train_20k_finetune.py
```

## Checkpoints

- **Periodic saves**: Every 5 epochs
- **Auto-sync**: Every 10 minutes to `/mnt/evafs/faculty/home/bpiotrowski/YOLO_Checkpoints/20k_finetune/`
- **Best model**: Saved to `best_yolo_20k/` at end of training

## Expected Outcomes

Based on DETR 20k finetune results, we expect:
- Improved cross-dataset generalization
- Better tooltip detection on external datasets
- Reduced overfitting compared to 90k training

## Benchmarking

After training, run the benchmark script to compare with DETR:
```bash
py YOLO_DETR_Benchmarks/scripts/benchmark_yolo_vs_detr_q81_multi_epoch.py
```

## Related Scripts

- DETR 20k finetune: `Eden/Scripts/Train20kDataset_11.12.25/`
- Original YOLO training: `Eden/Scripts/run_yolo-train_the_best_11062025.slurm`
- Benchmark: `YOLO_DETR_Benchmarks/scripts/benchmark_yolo_vs_detr_q81_multi_epoch.py`

## Troubleshooting

### Out of Memory
Reduce batch size in SLURM script:
```bash
--batch_size 16  # or 8 for Pascal
```

### Checkpoint Not Found
Script will fall back to pretrained YOLOv8m if epoch170.pt is missing.

### Permission Denied
Ensure conda environment is activated:
```bash
source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310
```

---

**Created**: 2025-12-31
**Author**: Claude Code Assistant
**Related Session**: Session_2025-12-31_042306
