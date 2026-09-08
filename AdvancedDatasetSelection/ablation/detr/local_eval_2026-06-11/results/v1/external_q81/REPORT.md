# External benchmark v1 - cross-dataset (protokol Q81)

- Wariant: **v1** (FULL pipeline (baseline, ep270))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\ckpt_20k_finetune\checkpoint_epoch_270.pth` (epoka 270)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 26.04% | 6.68% | 44.8% | 60.3% | 35.6% | 835 | 549 | 1511 |
| CLEAN | 2333 | 26.04% | 6.68% | 44.8% | 60.3% | 35.6% | 835 | 549 | 1511 |
| split_train | 2083 | 19.64% | 4.87% | 38.0% | 53.8% | 29.4% | 616 | 528 | 1480 |
| split_valid | 200 | 85.74% | 24.23% | 91.4% | 92.3% | 90.5% | 181 | 15 | 19 |
| split_test | 50 | 69.75% | 22.04% | 80.9% | 86.4% | 76.0% | 38 | 6 | 12 |
