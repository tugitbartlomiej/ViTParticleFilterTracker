# Smoke test v6 (po 50 obrazow z obu datasetow)

- Wariant: **v6** (DIVERSITY_ONLY (DINO+K-Center))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v6\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| external_test_smoke | 50 | 0.60% | 0.11% | 5.5% | 4.2% | 8.0% | 4 | 91 | 46 |
| samedist_test_smoke | 50 | 100.00% | 94.95% | 66.7% | 50.0% | 100.0% | 50 | 50 | 0 |
