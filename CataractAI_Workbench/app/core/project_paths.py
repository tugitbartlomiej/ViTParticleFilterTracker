"""Centralized project paths configuration."""

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Find the ViTParticleFilterTracker root directory."""
    # Walk up from this file to find the project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "CLAUDE.md").exists() or (parent / "CataractAI_Workbench").exists():
            return parent
    return current.parents[3]  # fallback: 4 levels up from core/


PROJECT_ROOT = _find_project_root()

# Existing project directories
ADVANCED_DATASET_SELECTION = PROJECT_ROOT / "AdvancedDatasetSelection"
BACKGROUND_FINETUNED = PROJECT_ROOT / "BackgroundFinetuned"
YOLO_DETR_BENCHMARKS = PROJECT_ROOT / "YOLO_DETR_Benchmarks"
EDEN_SCRIPTS = PROJECT_ROOT / "Eden" / "Scripts"
EXTERNAL = PROJECT_ROOT / "External"

# Config files
DATASET_SELECTION_CONFIG = ADVANCED_DATASET_SELECTION / "config.yaml"
TRAINING_SETTINGS = BACKGROUND_FINETUNED / "Settings" / "Windows" / "TrainingSettings.json"

# Model cache (from External/model_cache_config.py)
MODEL_CACHE_DIR = EXTERNAL / "Models"

# Workbench directories
WORKBENCH_ROOT = PROJECT_ROOT / "CataractAI_Workbench"
WORKBENCH_CONFIGS = WORKBENCH_ROOT / "app" / "resources"

# Default data paths (can be overridden by config)
DEFAULT_DATASET_ROOT = Path("E:/cataract_surgery_Instruments_detection.v1i.coco")
DEFAULT_VIDEO_DIR = Path("E:/Cataract/videos")


def setup_model_cache():
    """Configure model cache directories (mirrors External/model_cache_config.py)."""
    cache_dir = str(MODEL_CACHE_DIR)
    os.environ.setdefault("TORCH_HOME", cache_dir)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(MODEL_CACHE_DIR / "datasets"))

    try:
        import torch
        torch.hub.set_dir(cache_dir)
    except ImportError:
        pass
