# Integration Plan: TimeSformer + DETR + Temporal DINO

## Cel
Stworzenie systemu przewidywania przebicia błony oka łączącego:
- **DETR/YOLO**: spatial object detection (tooltip position)
- **TimeSformer**: temporal action recognition (current action)
- **Temporal DINO**: future action anticipation (prediction)

## Architektura

```
┌─────────────────────────────────────────────────────────┐
│           Video Input (8 frames @ 30 FPS)               │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
        ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ DETR/YOLO    │          │ TimeSformer  │
│ (Spatial)    │          │ (Temporal)   │
│              │          │              │
│ Output:      │          │ Output:      │
│ - Tooltip    │          │ - Current    │
│   boxes      │          │   action     │
│ - Confidence │          │ - Confidence │
└──────┬───────┘          └──────┬───────┘
       │                         │
       └────────┬────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Temporal DINO    │
      │ (Future Predict) │
      │                  │
      │ Input:           │
      │ - Past frames    │
      │ - Tooltip track  │
      │ - Current action │
      │                  │
      │ Output:          │
      │ - Future action  │
      │ - Time to event  │
      │ - Confidence     │
      └─────────┬────────┘
                │
                ▼
        ┌──────────────┐
        │ Final Output │
        │              │
        │ "PRZEBICIE"  │
        │ "za 0.5s"    │
        │ conf: 0.92   │
        └──────────────┘
```

## Etapy Implementacji

### Faza 1: Przygotowanie środowiska
- [ ] Clone Temporal DINO repo: `git clone https://github.com/IzzeddinTeeti/ssl_pred`
- [ ] Zainstaluj dependencies
- [ ] Przetestuj Temporal DINO na sample data

### Faza 2: Feature Extraction Pipeline
- [ ] Zmodyfikuj TimeSformer do ekstrahowania features (nie tylko logits)
- [ ] Dodaj trajectory tracker dla tooltip boxes (DETR output)
- [ ] Stwórz unified feature vector: [spatial_features, temporal_features, trajectory]

### Faza 3: Temporal DINO Adaptation
- [ ] Pretrain Temporal DINO na surgical videos (self-supervised)
- [ ] Fine-tune na dataset z labeled "przebicie" events
- [ ] Dodaj temporal offset prediction (ile sekund do przebicia)

### Faza 4: Integration
- [ ] Zintegruj wszystkie modele w jeden pipeline
- [ ] Dodaj postprocessing (smoothing, confidence thresholding)
- [ ] Real-time inference optimization

### Faza 5: Evaluation
- [ ] Test na validation set
- [ ] Measure: Precision, Recall, F1 dla prediction
- [ ] Measure: Mean Time Error (jak dokładnie przewiduje moment)

## Struktura kodu

```
ViTParticleFilterTracker/
├── Annotators/
│   ├── DetrAnnotator/           # Już masz
│   ├── TimesFormer/             # Już masz
│   └── TemporalDINO/            # NOWE
│       ├── ssl_pred/            # Clone z GitHub
│       ├── adapter.py           # Adapter dla surgical videos
│       └── trainer.py
├── ActionAnticipation/           # NOWE
│   ├── pipeline.py              # Main integration
│   ├── trajectory_tracker.py   # Tooltip trajectory analysis
│   ├── feature_fusion.py       # Combine DETR + TimeSformer
│   └── inference.py            # Real-time prediction
└── configs/
    └── anticipation_config.yaml
```

## Konfiguracja

```yaml
# anticipation_config.yaml

models:
  detr:
    checkpoint: "DETR_Checkpoints/checkpoint_epoch_140.pth"
    confidence_threshold: 0.7

  timesformer:
    checkpoint: "Annotators/TimesFormer/Models/best_surgical_timesformer.pth"
    num_frames: 8

  temporal_dino:
    backbone: "transformer"
    pretrained: true
    num_future_frames: 4  # Przewiduj 4 klatki w przód (~0.13s @ 30fps)

pipeline:
  input_fps: 30
  prediction_horizon: 0.5  # Przewiduj 0.5s w przód
  confidence_threshold: 0.8

actions:
  - name: "przebicie"
    alert_time: 0.5  # Alert 0.5s przed wydarzeniem
  - name: "bezposrednie_zagrozenie"
    alert_time: 1.0
```

## Expected Performance

Based on papers:
- **Temporal DINO**: +9.9 PP improvement in action prediction
- **TimeSformer**: ~85-90% accuracy on surgical actions (Twoje wyniki)
- **DETR**: ~90% tooltip detection (Twoje wyniki)

**Combined expected performance:**
- Action anticipation accuracy: 75-85%
- Time-to-event prediction: ±0.2s mean error
- False positive rate: <10%

## Timeline

- Week 1-2: Setup Temporal DINO + initial tests
- Week 3-4: Feature extraction pipeline
- Week 5-6: Model integration
- Week 7-8: Testing + optimization

## Resources

### Papers
1. Temporal DINO: https://arxiv.org/abs/2308.04589
2. SurgFUTR (future reference): https://arxiv.org/abs/2510.12904
3. TimeSformer: https://arxiv.org/abs/2102.05095

### Code
1. Temporal DINO: https://github.com/IzzeddinTeeti/ssl_pred
2. CAMMA surgical tools: https://github.com/CAMMA-public
3. TimeSformer: https://github.com/facebookresearch/TimeSformer

### Datasets (do pretraining)
- CholecT50: https://github.com/CAMMA-public/cholect50
- Surgical Workflows: https://github.com/CAMMA-public/SurgLatentGraph
- Twoje własne cataract surgery videos

## Notes

- **Self-supervised pretraining**: Temporal DINO nie wymaga labeled data do pretrainingu
- **Fine-tuning**: Dopiero fine-tuning wymaga labeled "przebicie" events
- **Real-time**: Może wymagać model optimization (pruning, quantization) dla real-time
- **GPU**: Cały pipeline może wymagać minimum 8GB VRAM

## Contact dla pomocy

- Temporal DINO: Izzeddin Teeti (ICCV 2023)
- SurgFUTR: Saurav Sharma (ssharma@unistra.fr)
- CAMMA group: https://camma.unistra.fr/
