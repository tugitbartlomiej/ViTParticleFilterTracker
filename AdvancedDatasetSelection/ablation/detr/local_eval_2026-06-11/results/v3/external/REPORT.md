# External benchmark v3 - cross-dataset

- Wariant: **v3** (NO_EL2N)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v3\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 1.51% | 0.37% | 7.9% | 5.9% | 12.1% | 283 | 4550 | 2063 |
| CLEAN | 2333 | 1.51% | 0.37% | 7.9% | 5.9% | 12.1% | 283 | 4550 | 2063 |
| split_train | 2083 | 1.69% | 0.42% | 8.3% | 6.1% | 12.6% | 265 | 4050 | 1831 |
| split_valid | 200 | 0.56% | 0.07% | 5.2% | 3.9% | 8.0% | 16 | 399 | 184 |
| split_test | 50 | 0.48% | 0.11% | 2.6% | 1.9% | 4.0% | 2 | 101 | 48 |
