"""Shared fixtures for CataractAI Workbench tests."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_coco_annotations(tmp_path):
    """Create a minimal COCO-format annotation file."""
    data = {
        "images": [
            {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img_002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 50, 100, 80], "area": 8000, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "tool"},
            {"id": 2, "name": "background"},
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def sample_session_dir(tmp_path):
    """Create a mock session directory with SESSION_SUMMARY.md."""
    session_name = "Session_2026-01-15_143022_Test_Session"
    session_dir = tmp_path / session_name
    session_dir.mkdir()
    summary = '''# Session Summary: Test Session

## Metadata
- **Date:** 2026-01-15
- **Time:** 14:30:22
- **Type:** Training
- **Status:** Completed
- **Topics:** DETR, YOLO

## Objective
Test the training pipeline with new parameters.

## Actions Taken
1. Configured training parameters
2. Ran training for 5 epochs

## Key Findings
- Loss decreased from 0.5 to 0.1
- mAP improved to 0.85

## Files Modified
- main_train.py
- config.yaml

## Next Steps
- [ ] Run full training
- [x] Verify results
'''
    (session_dir / "SESSION_SUMMARY.md").write_text(summary, encoding="utf-8")
    return tmp_path, session_name


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample YAML config file."""
    config = """
selection_method: cluster
strategy: centroid
target_size: 20000
features:
  dino_pca_dim: 32
  use_fourier: true
weights:
  dino: 0.35
  fourier: 0.15
  sam: 0.20
  el2n: 0.30
"""
    path = tmp_path / "config.yaml"
    path.write_text(config)
    return str(path)


@pytest.fixture
def sample_config_json(tmp_path):
    """Create a sample JSON config file."""
    config = {
        "model": "detr",
        "epochs": 5,
        "batch_size": 2,
        "learning_rate": 2e-6,
        "amp": True,
        "early_stopping": {"enabled": True, "patience": 6},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(path)
