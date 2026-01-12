# DETR vs YOLO training comparison (two phases)

## Inputs
- DETR phase1: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr_custom2\detr_checkpoint_analysis.json`
- DETR phase2: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\detr_custom_finetune\detr_checkpoint_analysis.json`
- YOLO phase1: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\yolo_custom2\yolo_checkpoint_analysis.json`
- YOLO phase2: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\yolo_custom_finetune\yolo_checkpoint_analysis.json`
- YOLO phase1 CSV: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\exp\results.csv`
- YOLO phase2 CSV: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\YOLO_EDEN_TRAIN\YoloTreningSesnions\20KTreningFrom170epochStart\exp\results.csv`

## Dataset phases
- Phase 1 (epoch <= 170): ~100k images
- Phase 2 (epoch 170-200): 20k images (fine-tune)

## Plots
- Training overview: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\compare_detr_yolo\detr_vs_yolo_comparison.png`
- DETR layer evolution: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\compare_detr_yolo\detr_layer_evolution_compare.png`

## Phase 1 (epoch <= 170)
| Metric | DETR | YOLO |
|---|---|---|
| LR start -> end | 0.0001 -> 0.0001 | 0.00332812 -> 0.0016345 |
| LR backbone start -> end | 1e-05 -> 1e-05 | N/A |
| Best metric | train loss 0.1586986621776661 @ 160 | mAP50-95 0.94078 @ 170 |
| Weight norm range | 8.422172231248767 -> 8.707576377428913 | 6.375064229682447 -> 12.396746071250025 |
| Weight norm max range | 158.31597900390625 -> 158.71160888671875 | 35.2136344909668 -> 463.07391357421875 |
| YOLO weight std range | N/A | 7.211425235025577 -> 24.634247916577326 |

## Phase 2 (epoch 170-200)
| Metric | DETR | YOLO |
|---|---|---|
| LR start -> end | 0.0001 -> 4.372534316686896e-05 | 0.0015355 -> 0.0001495 |
| LR backbone start -> end | 1e-05 -> 4.383851333019196e-06 | N/A |
| Best metric | train loss 0.12050509931571991 @ 195 | mAP50-95 0.96332 @ 189 |
| Weight norm range | 8.707576377428913 -> 8.711658748331375 | 8.481072474031732 -> 8.531500911454893 |
| Weight norm max range | 158.70492553710938 -> 158.71160888671875 | 106.78174591064453 -> 109.51588439941406 |
| YOLO weight std range | N/A | 8.73956352162836 -> 8.827853977316888 |

## DETR layer norm drift (earliest -> latest)
| Layer | Norm change (%) |
|---|---|
| class_labels_classifier.weight | +43.90% |
| model.encoder.layers.0.self_attn.out_proj.weight | +4.67% |
| model.backbone.conv_encoder.model.layer4.2.conv3.weight | +3.31% |
| model.decoder.layers.0.self_attn.out_proj.weight | +2.85% |
| bbox_predictor.layers.2.weight | +0.30% |

## Notes
- DETR metrics are losses (lower is better). YOLO metrics are mAP (higher is better).
- Weight statistics come from checkpoint analysis; YOLO std is only available for YOLO checkpoints.
