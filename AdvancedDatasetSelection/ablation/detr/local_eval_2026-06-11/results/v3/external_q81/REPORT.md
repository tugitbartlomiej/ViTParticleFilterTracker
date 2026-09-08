# External benchmark v3 - cross-dataset (protokol Q81)

- Wariant: **v3** (NO_EL2N)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v3\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 0.29% | 0.04% | 2.9% | 3.0% | 2.9% | 69 | 2263 | 2277 |
| CLEAN | 2333 | 0.29% | 0.04% | 2.9% | 3.0% | 2.9% | 69 | 2263 | 2277 |
| split_train | 2083 | 0.27% | 0.03% | 2.7% | 2.7% | 2.7% | 56 | 2026 | 2040 |
| split_valid | 200 | 0.81% | 0.11% | 6.5% | 6.5% | 6.5% | 13 | 187 | 187 |
| split_test | 50 | 0.00% | 0.00% | 0.0% | 0.0% | 0.0% | 0 | 50 | 50 |
