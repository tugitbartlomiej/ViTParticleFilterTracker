# EDEN Commands - Session 2025-12-12

## Job Info
- **Job ID:** 1453424
- **Partition:** long
- **Time Limit:** 5 days
- **GPUs:** 4
- **Status:** PD (Priority) - waiting for resources

## Monitoring Commands

```bash
# Check job status
squeue -u bpiotrowski

# Check estimated start time
squeue -u bpiotrowski --start

# Check free resources
sfree

# Check partition limits
sinfo -p long -o "%P %l"

# Watch job (updates every 60s)
watch -n 60 'squeue -u bpiotrowski; sfree'
```

## After Job Starts

```bash
# Check logs
tail -f logs/detr_20k_finetune_1453424.log

# Check GPU usage
nvidia-smi

# Check training progress
grep -i "epoch\|loss" logs/detr_20k_finetune_1453424.log
```

## Checkpoints Location
```
/mnt/evafs/faculty/home/bpiotrowski/DETR/Checkpoints/20k_finetune/
/mnt/evafs/faculty/home/bpiotrowski/DETR/best_20k_finetune/
```

## Cancel Job (if needed)
```bash
scancel 1453424
```

## Resubmit
```bash
sbatch run_detr_20k_finetune.slurm
```
