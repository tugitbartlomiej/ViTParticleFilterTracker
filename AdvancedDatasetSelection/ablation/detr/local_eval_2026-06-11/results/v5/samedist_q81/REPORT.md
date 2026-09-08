# Same-distribution v5 (video-level split)

- Wariant: **v5** (RANDOM)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v5\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | 10652 | 42.68% | 36.81% | 57.2% | 68.0% | 49.4% | 5328 | 2503 | 5458 |
| test | 9140 | 42.99% | 37.74% | 57.2% | 70.1% | 48.3% | 4460 | 1906 | 4771 |
