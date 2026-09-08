# External benchmark v0 - cross-dataset (protokol Q81)

- Wariant: **v0** (UNCURATED baseline (ep170, przed fine-tune))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\ckpt_20k_finetune\checkpoint_epoch_170.pth` (epoka 170)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 3.56% | 0.72% | 14.5% | 14.6% | 14.5% | 339 | 1975 | 2007 |
| CLEAN | 2333 | 3.56% | 0.72% | 14.5% | 14.6% | 14.5% | 339 | 1975 | 2007 |
| split_train | 2083 | 4.09% | 0.86% | 15.2% | 15.3% | 15.1% | 317 | 1749 | 1779 |
| split_valid | 200 | 2.03% | 0.37% | 8.5% | 8.5% | 8.5% | 17 | 182 | 183 |
| split_test | 50 | 2.70% | 0.39% | 10.1% | 10.2% | 10.0% | 5 | 44 | 45 |
