# External benchmark v5 - cross-dataset (protokol Q81)

- Wariant: **v5** (RANDOM)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v5\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 0.23% | 0.04% | 2.5% | 2.9% | 2.2% | 52 | 1737 | 2294 |
| CLEAN | 2333 | 0.23% | 0.04% | 2.5% | 2.9% | 2.2% | 52 | 1737 | 2294 |
| split_train | 2083 | 0.21% | 0.04% | 2.6% | 3.0% | 2.2% | 47 | 1527 | 2049 |
| split_valid | 200 | 1.27% | 0.24% | 2.7% | 3.0% | 2.5% | 5 | 164 | 195 |
| split_test | 50 | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% | 0 | 46 | 50 |
