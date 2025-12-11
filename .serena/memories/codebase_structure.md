# Codebase Structure

## Root Directory
```
ViTParticleFilterTracker/
├── Annotators/                 # Detection and annotation systems
├── BackgroundFinetuned/        # Fine-tuned DETR models
├── BackgroundFinetuned_wt_dev_refactor/  # Refactored version
├── BareDetr/                   # Minimalist DETR implementation
├── Cataract_DETR_DatasetBuilder/  # Dataset building tools
├── Claude/                     # Claude AI integration
├── Eden/                       # HPC cluster scripts
├── Notatki/                    # Notes (Polish)
├── serena/                     # Serena MCP server (venv)
├── YOLO_DETR_Benchmarks/       # Main benchmarking module
├── YOLO_DETR_BENCHAMRKS_VER2_CODEX/  # Alternative version
├── CLAUDE.md                   # Project instructions
├── AGENTS.md                   # Task Master guide
├── README.md                   # Project documentation
└── *.py                        # Root utility scripts
```

## Key Directories Detail

### Annotators/
```
Annotators/
├── DetrAnnotator/              # DETR training/inference
│   ├── detr_annotator_inference_range_images.py
│   └── ...
├── Yolo/                       # YOLO detection
│   ├── yolo_predict.py
│   └── ...
├── DeepSortYolo/               # Tracking
├── TimesFormer/                # Video temporal analysis
├── OpencvTrackerAnnotator/     # OpenCV tracking
├── Datasets/                   # Training data
│   ├── Yolo/
│   └── Detr/
└── Utils/                      # Utility functions
```

### YOLO_DETR_Benchmarks/
```
YOLO_DETR_Benchmarks/
├── scripts/
│   └── DETR_Background_Training/  # Main pipeline
│       ├── main_pipeline.py       # Orchestrator
│       ├── simple_pipeline.py     # Simplified version
│       ├── background_task_executor.py
│       ├── status_manager.py
│       ├── pipeline_monitor.py
│       ├── progress_reporter.py
│       ├── dataset_mixer.py
│       └── *.yaml                 # Configs
├── Intelligent_Background_Selector_2025-07-17/
├── DINO_Frame_Selection/
├── Advanced_Analysis/
├── Benchmarks/
├── models/                     # Trained models
│   ├── YOLO/
│   └── DETR/
└── Datasets/
```

### BackgroundFinetuned/
```
BackgroundFinetuned/
├── main.py                     # Entry point
├── main_train.py               # Training
├── main_generate.py            # Generation/inference
└── ...
```

## Important Files

### Entry Points
- `YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/main_pipeline.py`
- `Annotators/DetrAnnotator/detr_annotator_inference_range_images.py`
- `Annotators/Yolo/yolo_predict.py`
- `BackgroundFinetuned/main.py`

### Configuration
- `YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/pipeline_config.yaml`
- `YOLO_DETR_Benchmarks/benchmark_config.yaml`

### Models (trained)
- `YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt`
- `YOLO_DETR_Benchmarks/DETR/detr_inference_model_final/`
- `Eden/DETR_Checkpoints/` (checkpoints from cluster training)

## External Data Paths
- Videos: `E:/Cataract/videos/`
- Micro videos: `E:/Cataract/videos/micro/`
- Test videos: `E:/Cataract/videos/test_single/`
