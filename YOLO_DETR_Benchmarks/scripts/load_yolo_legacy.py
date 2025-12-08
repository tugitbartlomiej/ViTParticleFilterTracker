#!/usr/bin/env python3
"""
Helper to load legacy YOLO checkpoints with PyTorch 2.6+
Uses weights_only=False for trusted checkpoints
"""

import torch
import sys

# Create compatibility stubs for removed ultralytics classes
import ultralytics.utils.loss as loss_module

# DFLoss was removed in newer ultralytics - create stub for pickle compatibility
if not hasattr(loss_module, 'DFLoss'):
    class DFLoss:
        """Compatibility stub for legacy DFLoss class"""
        pass
    loss_module.DFLoss = DFLoss
    sys.modules['ultralytics.utils.loss'].DFLoss = DFLoss

from ultralytics import YOLO

# Monkey-patch torch.load to use weights_only=False
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """Override torch.load to always use weights_only=False"""
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

def load_yolo_legacy(checkpoint_path, device='cuda'):
    """
    Load legacy YOLO checkpoint with PyTorch 2.6+ compatibility

    Args:
        checkpoint_path: Path to YOLO .pt file
        device: 'cuda' or 'cpu'

    Returns:
        YOLO model
    """
    try:
        model = YOLO(checkpoint_path)
        model.to(device)
        return model
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        raise

if __name__ == "__main__":
    # Test loading
    checkpoint = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/YOLO_EDEN_TRAIN/epoch100.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading YOLO from: {checkpoint}")
    model = load_yolo_legacy(checkpoint, device)
    print(f"✅ Successfully loaded YOLO model")
    print(f"Model type: {type(model)}")
