# External benchmark v4 - cross-dataset

- Wariant: **v4** (QUALITY_ONLY)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v4\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 1.54% | 0.34% | 8.1% | 6.4% | 11.1% | 260 | 3821 | 2086 |
| CLEAN | 2333 | 1.54% | 0.34% | 8.1% | 6.4% | 11.1% | 260 | 3821 | 2086 |
| split_train | 2083 | 1.65% | 0.38% | 8.7% | 6.8% | 11.8% | 248 | 3385 | 1848 |
| split_valid | 200 | 1.10% | 0.14% | 3.2% | 2.5% | 4.5% | 9 | 352 | 191 |
| split_test | 50 | 0.97% | 0.21% | 4.4% | 3.4% | 6.0% | 3 | 84 | 47 |
