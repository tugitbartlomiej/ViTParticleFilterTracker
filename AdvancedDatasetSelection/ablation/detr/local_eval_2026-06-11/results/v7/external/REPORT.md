# External benchmark v7 - cross-dataset

- Wariant: **v7** (EL2N_ONLY (random+EL2N rerank))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v7\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 3.63% | 0.67% | 9.6% | 7.1% | 14.6% | 343 | 4470 | 2003 |
| CLEAN | 2333 | 3.63% | 0.67% | 9.6% | 7.1% | 14.6% | 343 | 4470 | 2003 |
| split_train | 2083 | 3.81% | 0.72% | 9.9% | 7.4% | 15.1% | 316 | 3975 | 1780 |
| split_valid | 200 | 5.78% | 0.94% | 7.8% | 5.8% | 12.0% | 24 | 393 | 176 |
| split_test | 50 | 1.56% | 0.29% | 3.9% | 2.9% | 6.0% | 3 | 102 | 47 |
