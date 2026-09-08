# Same-distribution v6 (video-level split)

- Wariant: **v6** (DIVERSITY_ONLY (DINO+K-Center))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v6\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | 10652 | 98.01% | 87.99% | 99.2% | 100.0% | 98.5% | 10625 | 5 | 161 |
| test | 9140 | 97.94% | 88.75% | 99.4% | 99.9% | 98.8% | 9120 | 8 | 111 |
