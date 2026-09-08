# External benchmark v5 - cross-dataset

- Wariant: **v5** (RANDOM)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v5\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 1.37% | 0.29% | 8.9% | 6.9% | 12.3% | 289 | 3895 | 2057 |
| CLEAN | 2333 | 1.37% | 0.29% | 8.9% | 6.9% | 12.3% | 289 | 3895 | 2057 |
| split_train | 2083 | 1.52% | 0.33% | 9.5% | 7.4% | 13.1% | 274 | 3426 | 1822 |
| split_valid | 200 | 0.56% | 0.09% | 4.1% | 3.1% | 6.0% | 12 | 371 | 188 |
| split_test | 50 | 0.46% | 0.09% | 4.0% | 3.0% | 6.0% | 3 | 98 | 47 |
