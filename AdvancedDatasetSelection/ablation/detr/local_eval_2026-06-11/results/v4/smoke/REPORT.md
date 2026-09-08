# Smoke test v4 (po 50 obrazow z obu datasetow)

- Wariant: **v4** (QUALITY_ONLY)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v4\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| external_test_smoke | 50 | 0.97% | 0.21% | 4.4% | 3.4% | 6.0% | 3 | 84 | 47 |
| samedist_test_smoke | 50 | 100.00% | 94.12% | 66.7% | 50.0% | 100.0% | 50 | 50 | 0 |
