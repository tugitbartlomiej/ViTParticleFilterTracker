# Same-distribution v3 (video-level split)

- Wariant: **v3** (NO_EL2N)
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\ablation_best\v3\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | 10652 | 58.12% | 53.85% | 64.2% | 64.6% | 63.8% | 6883 | 3769 | 3903 |
| test | 9140 | 59.24% | 54.62% | 65.3% | 65.7% | 65.0% | 6001 | 3139 | 3230 |
