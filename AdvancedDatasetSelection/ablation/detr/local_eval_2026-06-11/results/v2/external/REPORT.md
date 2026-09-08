# External benchmark v2 - cross-dataset

- Wariant: **v2** (NO_FOURIER)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v2\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 2.19% | 0.41% | 10.4% | 7.9% | 15.3% | 360 | 4210 | 1986 |
| CLEAN | 2333 | 2.19% | 0.41% | 10.4% | 7.9% | 15.3% | 360 | 4210 | 1986 |
| split_train | 2083 | 2.31% | 0.45% | 10.8% | 8.1% | 15.9% | 333 | 3755 | 1763 |
| split_valid | 200 | 2.11% | 0.30% | 8.2% | 6.2% | 12.0% | 24 | 360 | 176 |
| split_test | 50 | 0.61% | 0.10% | 4.1% | 3.1% | 6.0% | 3 | 95 | 47 |
