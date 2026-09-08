# External benchmark v6 - cross-dataset (protokol Q81)

- Wariant: **v6** (DIVERSITY_ONLY (DINO+K-Center))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v6\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 3.67% | 0.86% | 16.1% | 17.7% | 14.7% | 345 | 1607 | 2001 |
| CLEAN | 2333 | 3.67% | 0.86% | 16.1% | 17.7% | 14.7% | 345 | 1607 | 2001 |
| split_train | 2083 | 3.98% | 0.96% | 16.6% | 18.4% | 15.2% | 318 | 1412 | 1778 |
| split_valid | 200 | 5.30% | 0.91% | 12.0% | 12.5% | 11.5% | 23 | 161 | 177 |
| split_test | 50 | 1.79% | 0.45% | 9.1% | 10.5% | 8.0% | 4 | 34 | 46 |
