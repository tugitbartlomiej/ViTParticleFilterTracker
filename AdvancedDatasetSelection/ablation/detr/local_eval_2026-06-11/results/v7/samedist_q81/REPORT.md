# Same-distribution v7 (video-level split)

- Wariant: **v7** (EL2N_ONLY (random+EL2N rerank))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v7\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | 10652 | 98.01% | 91.98% | 99.3% | 100.0% | 98.7% | 10649 | 3 | 137 |
| test | 9140 | 97.96% | 92.26% | 99.4% | 99.8% | 98.9% | 9126 | 14 | 105 |
