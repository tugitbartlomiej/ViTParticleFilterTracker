# Smoke test v3 (po 50 obrazow z obu datasetow)

- Wariant: **v3** (NO_EL2N)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v3\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| external_test_smoke | 50 | 0.48% | 0.11% | 2.6% | 1.9% | 4.0% | 2 | 101 | 48 |
| samedist_test_smoke | 50 | 100.00% | 95.51% | 66.7% | 50.0% | 100.0% | 50 | 50 | 0 |
