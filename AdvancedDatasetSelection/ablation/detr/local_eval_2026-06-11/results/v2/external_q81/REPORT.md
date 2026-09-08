# External benchmark v2 - cross-dataset (protokol Q81)

- Wariant: **v2** (NO_FOURIER)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v2\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 0.37% | 0.06% | 4.8% | 5.0% | 4.6% | 107 | 2012 | 2239 |
| CLEAN | 2333 | 0.37% | 0.06% | 4.8% | 5.0% | 4.6% | 107 | 2012 | 2239 |
| split_train | 2083 | 0.33% | 0.05% | 4.4% | 4.6% | 4.2% | 87 | 1811 | 2009 |
| split_valid | 200 | 3.04% | 0.44% | 10.6% | 11.2% | 10.0% | 20 | 158 | 180 |
| split_test | 50 | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% | 0 | 43 | 50 |
