# External benchmark v4 - cross-dataset (protokol Q81)

- Wariant: **v4** (QUALITY_ONLY)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v4\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 2.94% | 0.62% | 12.0% | 13.2% | 10.9% | 256 | 1682 | 2090 |
| CLEAN | 2333 | 2.94% | 0.62% | 12.0% | 13.2% | 10.9% | 256 | 1682 | 2090 |
| split_train | 2083 | 3.26% | 0.70% | 12.8% | 14.1% | 11.7% | 245 | 1492 | 1851 |
| split_valid | 200 | 1.66% | 0.19% | 4.4% | 4.9% | 4.0% | 8 | 155 | 192 |
| split_test | 50 | 2.07% | 0.44% | 6.8% | 7.9% | 6.0% | 3 | 35 | 47 |
