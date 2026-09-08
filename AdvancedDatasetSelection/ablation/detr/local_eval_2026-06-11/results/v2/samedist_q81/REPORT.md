# Same-distribution v2 (video-level split)

- Wariant: **v2** (NO_FOURIER)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v2\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | 10652 | 45.99% | 40.48% | 55.8% | 56.2% | 55.5% | 5985 | 4667 | 4801 |
| test | 9140 | 47.51% | 42.02% | 57.5% | 57.8% | 57.2% | 5280 | 3860 | 3951 |
