# External benchmark v7 - cross-dataset (protokol Q81)

- Wariant: **v7** (EL2N_ONLY (random+EL2N rerank))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v7\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| FULL | 2333 | 4.32% | 0.80% | 15.1% | 15.6% | 14.6% | 343 | 1861 | 2003 |
| CLEAN | 2333 | 4.32% | 0.80% | 15.1% | 15.6% | 14.6% | 343 | 1861 | 2003 |
| split_train | 2083 | 4.69% | 0.87% | 15.6% | 16.1% | 15.1% | 316 | 1652 | 1780 |
| split_valid | 200 | 5.77% | 0.93% | 12.2% | 12.4% | 12.0% | 24 | 169 | 176 |
| split_test | 50 | 1.46% | 0.25% | 6.5% | 7.0% | 6.0% | 3 | 40 | 47 |
