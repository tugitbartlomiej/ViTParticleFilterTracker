# Same-distribution v4 (video-level split)

- Wariant: **v4** (QUALITY_ONLY)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v4\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | 10652 | 83.84% | 74.75% | 87.0% | 87.9% | 86.1% | 9283 | 1274 | 1503 |
| test | 9140 | 85.10% | 76.21% | 88.0% | 88.8% | 87.3% | 8058 | 1017 | 1173 |
