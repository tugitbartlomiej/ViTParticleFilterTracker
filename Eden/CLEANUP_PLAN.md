# Plan czyszczenia Eden — 2026-03-20

## Do usunięcia (~70 GB)

### 1. DUPLIKAT ckpt_20k_finetune_v2_fixed/ (26 GB)
Kopia tego samego co Checkpoints/20k_finetune_v2_fixed/
```
rm -rf ~/DETR/ckpt_20k_finetune_v2_fixed/
```

### 2. Stare checkpointy 20k_ftv2 — zachować 5, usunąć 49 (~23 GB)
Zachować: ep170, ep210 (best mAP), ep265, ep435 (latest), final_checkpoint.pth
```
# ZACHOWAĆ:
# ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_170.pth
# ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth
# ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_265.pth
# ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_435.pth
# ~/DETR/Checkpoints/20k_finetune_v2_fixed/final_checkpoint.pth

# USUNĄĆ: wszystkie inne checkpoint_epoch_*.pth z tego katalogu
```

### 3. Stare early-training checkpointy ep20-130 (~10 GB)
```
rm ~/DETR/Checkpoints/checkpoint_epoch_20.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_30.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_40.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_50.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_60.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_70.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_80.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_90.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_100.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_110.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_120.pth
rm ~/DETR/Checkpoints/checkpoint_epoch_130.pth
```
ZACHOWAĆ: checkpoint_epoch_140.pth, checkpoint_epoch_150.pth, checkpoint_epoch_160.pth, checkpoint_epoch_170.pth, final_checkpoint.pth

### 4. Stary v1 fine-tune (7.5 GB)
```
rm -rf ~/DETR/ckpt_20k_finetune/
```

### 5. Stary 2-GPU eksperyment (2.8 GB)
```
rm -rf ~/DETR/ckpt_ddp_2gpu_dgx1_500ep/
```

### 6. Stare logi (~1 GB)
```
rm ~/DETR/logs/*.log
```

## ZACHOWAĆ (krytyczne)
- ~/DETR/Checkpoints/checkpoint_epoch_170.pth — START ablacji
- ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth — best mAP artykuł
- ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_265.pth
- ~/DETR/Checkpoints/20k_finetune_v2_fixed/final_checkpoint.pth
- ~/datasets_20250606.tar.gz — pula 91k
- ~/20kSelectedImages.tar — 20k wybranych
- ~/DETR/best_*/ — best models

## Szacowane oszczędności: ~70 GB
## Po czyszczeniu: ~88 GB wolnego (wystarczy na ablację)
