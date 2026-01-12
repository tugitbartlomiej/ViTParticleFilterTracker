# DETR vs YOLO training comparison (two phases)

## Inputs
- DETR phase1: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr_custom2\detr_checkpoint_analysis.json`
- DETR phase2: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr_custom_finetune\detr_checkpoint_analysis.json`
- YOLO phase1: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\yolo_custom2\yolo_checkpoint_analysis.json`
- YOLO phase2: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\yolo_custom_finetune\yolo_checkpoint_analysis.json`
- YOLO phase1 CSV: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\exp\results.csv`
- YOLO phase2 CSV: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart\exp\results.csv`

## Phase 1 (100k, epoch <= 170)
| Metric | DETR | YOLO |
|---|---|---|
| LR start -> end | 0.0001 -> 0.0001 | 0.00332812 -> 0.0016345 |
| LR backbone start -> end | 1e-05 -> 1e-05 | N/A |
| Best metric | train loss 0.1586986621776661 @ 160 | mAP50-95 0.94078 @ 170 |
| Weight norm range | 8.422172231248767 -> 8.707576377428913 | 6.375064229682447 -> 12.396746071250025 |

## Phase 2 (fine-tuning, 170-200)
| Metric | DETR | YOLO |
|---|---|---|
| LR start -> end | 0.0001 -> 4.372534316686896e-05 | 0.0015355 -> 0.0001495 |
| LR backbone start -> end | 1e-05 -> 4.383851333019196e-06 | N/A |
| Best metric | train loss 0.12050509931571991 @ 195 | mAP50-95 0.96332 @ 189 |
| Weight norm range | 8.707576377428913 -> 8.711658748331375 | 8.481072474031732 -> 8.531500911454893 |
