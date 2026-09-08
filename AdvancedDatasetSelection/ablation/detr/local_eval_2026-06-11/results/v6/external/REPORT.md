# External benchmark v6 - cross-dataset

- Wariant: **v6** (DIVERSITY_ONLY (DINO+K-Center))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v6\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 1.80% | 0.47% | 10.3% | 8.0% | 14.7% | 346 | 4001 | 2000 |
| CLEAN | 2333 | 1.80% | 0.47% | 10.3% | 8.0% | 14.7% | 346 | 4001 | 2000 |
| split_train | 2083 | 1.89% | 0.50% | 10.7% | 8.3% | 15.2% | 319 | 3542 | 1777 |
| split_valid | 200 | 3.14% | 0.54% | 7.8% | 5.9% | 11.5% | 23 | 368 | 177 |
| split_test | 50 | 0.60% | 0.11% | 5.5% | 4.2% | 8.0% | 4 | 91 | 46 |
