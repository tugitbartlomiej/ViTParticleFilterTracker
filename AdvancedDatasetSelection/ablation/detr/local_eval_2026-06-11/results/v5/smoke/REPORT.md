# Smoke test v5 (po 50 obrazow z obu datasetow)

- Wariant: **v5** (RANDOM)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v5\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| external_test_smoke | 50 | 0.46% | 0.09% | 4.0% | 3.0% | 6.0% | 3 | 98 | 47 |
| samedist_test_smoke | 50 | 100.00% | 91.67% | 81.3% | 68.5% | 100.0% | 50 | 23 | 0 |
