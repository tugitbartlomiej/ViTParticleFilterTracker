# External benchmark v1 - cross-dataset

- Wariant: **v1** (FULL pipeline (baseline, ep270))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\ckpt_20k_finetune\checkpoint_epoch_270.pth` (epoka 270)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 21.40% | 5.54% | 18.9% | 12.3% | 40.3% | 946 | 6721 | 1400 |
| CLEAN | 2333 | 21.40% | 5.54% | 18.9% | 12.3% | 40.3% | 946 | 6721 | 1400 |
| split_train | 2083 | 15.35% | 3.78% | 15.4% | 9.9% | 34.6% | 726 | 6622 | 1370 |
| split_valid | 200 | 86.62% | 24.83% | 86.0% | 81.9% | 90.5% | 181 | 40 | 19 |
| split_test | 50 | 68.24% | 21.50% | 52.7% | 39.8% | 78.0% | 39 | 59 | 11 |
