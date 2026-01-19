#!/usr/bin/env python3
"""
YOLO Fine-tuning on 20k Selected Dataset - V2 (FIXED)
======================================================

Fine-tune YOLOv8 model using pretrained weights from epoch 200,
training 300 NEW epochs (for effective total of 500).

KEY FIX (v2):
- Uses --pretrained_path to LOAD weights (not resume)
- Trains 300 NEW epochs (NOT resume old training)
- Ultralytics resume=True ONLY works for interrupted training
- For completed training, we start a NEW training with pretrained weights

Usage:
    python yolo_train_20k_finetune_v2.py \
        --dataset_yaml_path /path/to/20k_dataset.yaml \
        --pretrained_path /path/to/last.pt \
        --epochs 300 \
        --learning_rate 0.001
"""

import argparse
import os
import shutil
import signal
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*NVML.*")

import torch
from ultralytics import YOLO


class YOLOFineTunerV2:
    """
    YOLO Fine-tuning trainer for 20k selected dataset - V2 (FIXED).

    KEY DIFFERENCE from v1:
    - Does NOT use resume=True (that only works for interrupted training)
    - Loads pretrained weights and starts NEW training for specified epochs
    - This allows training completed checkpoints for additional epochs
    """

    def __init__(
        self,
        dataset_yaml_path: str,
        model_size: str = 'm',
        epochs: int = 300,
        batch_size: int = 16,
        imgsz: int = 640,
        project_dir: str = './yolo_20k_finetune',
        checkpoint_dir: str = './ckpt_yolo_20k',
        best_model_dir: str = './best_yolo_20k',
        learning_rate: float = 0.0001,
        lrf: float = 0.1,  # Final LR = lr0 * lrf
        pretrained_path: str = None,  # CHANGED: pretrained, not resume
        workers: int = 4,
        device: str = None,
        save_period: int = 5,
        patience: int = 30,
        verbose: bool = True,
        freeze_backbone: int = 0,
    ):
        """Initialize YOLO fine-tuner v2."""
        # Disable wandb
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"

        self.dataset_yaml_path = Path(dataset_yaml_path)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.project_dir = Path(project_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_model_dir = Path(best_model_dir)
        self.learning_rate = learning_rate
        self.lrf = lrf
        self.pretrained_path = pretrained_path
        self.workers = workers
        self.save_period = save_period
        self.patience = patience
        self.verbose = verbose
        self.freeze_backbone = freeze_backbone

        # Device setup
        if device is None:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    self.device = ','.join([str(i) for i in range(gpu_count)])
                else:
                    self.device = '0'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Verify dataset exists
        if not self.dataset_yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml_path}")

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        # KEY FIX: Load pretrained weights as starting point for NEW training
        if self.pretrained_path and Path(self.pretrained_path).exists():
            print(f"\n{'='*60}")
            print("YOLO 20K FINE-TUNING V2 - PRETRAINED WEIGHTS")
            print(f"{'='*60}")
            print(f"  Pretrained: {self.pretrained_path}")
            print(f"  Strategy: Load weights, train {self.epochs} NEW epochs")
            print(f"  NOTE: This is NOT resume - it's a new training session!")
            # Load model with pretrained weights
            self.model = YOLO(self.pretrained_path)
        else:
            print(f"\n{'='*60}")
            print("YOLO 20K FINE-TUNING V2 - FRESH START")
            print(f"{'='*60}")
            print(f"  Model: YOLOv8{self.model_size}")
            self.model = YOLO(f'yolov8{self.model_size}.pt')

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        self._print_config()

    def _print_config(self):
        """Print training configuration."""
        print(f"\nConfiguration:")
        print(f"  Dataset: {self.dataset_yaml_path}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs} (NEW epochs, not continuation)")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.imgsz}")
        print(f"  Learning rate: {self.learning_rate} -> {self.learning_rate * self.lrf} (lrf={self.lrf})")
        print(f"  Save period: {self.save_period}")
        print(f"  Patience: {self.patience}")
        print(f"  Freeze backbone: {self.freeze_backbone} layers")
        print(f"  Project dir: {self.project_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")

        if torch.cuda.is_available():
            print(f"\nGPU Info:")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")

        print(f"{'='*60}\n")

    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signal."""
        print("\n[INTERRUPT] Saving emergency checkpoint...")
        emergency_path = self.checkpoint_dir / "emergency_checkpoint.pt"
        self.model.save(str(emergency_path))
        print(f"  Saved to: {emergency_path}")
        sys.exit(0)

    def train(self):
        """Run fine-tuning."""
        print("Starting YOLO fine-tuning on 20k dataset...")
        print(f"Training for {self.epochs} NEW epochs...")

        try:
            # Training arguments
            # KEY FIX: NO resume=True! This is a NEW training session
            train_args = {
                'data': str(self.dataset_yaml_path),
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.imgsz,
                'device': self.device,
                'project': str(self.project_dir),
                'name': 'exp',
                'pretrained': True,  # Use pretrained weights
                'verbose': self.verbose,
                'workers': self.workers,
                'optimizer': 'SGD',  # MUST specify optimizer, otherwise 'auto' ignores lr0!
                'lr0': self.learning_rate,
                'lrf': self.lrf,  # Final LR = lr0 * lrf
                'momentum': 0.937,
                'save_period': self.save_period,
                'patience': self.patience,
                'exist_ok': True,
                'save': True,
                'save_json': True,
                'plots': True,
                'val': True,
                # Fine-tuning specific
                'freeze': self.freeze_backbone,
                'cos_lr': False,  # Linear LR for fine-tuning stability
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.0,  # No warmup for bias (already trained)
                # Regularization
                'weight_decay': 0.0005,
                'dropout': 0.0,
                # DO NOT set resume=True - we're starting a NEW training!
            }

            # Run training
            results = self.model.train(**train_args)

            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)

            # Copy best model
            self._save_best_model()

            return results

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Saving checkpoint...")
            self._save_emergency_checkpoint()
            sys.exit(0)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            self._save_emergency_checkpoint()
            sys.exit(1)

    def _save_best_model(self):
        """Copy best model to dedicated directory."""
        possible_paths = [
            self.project_dir / "exp" / "weights" / "best.pt",
            self.project_dir / "exp" / "best.pt",
            self.project_dir / "weights" / "best.pt",
        ]

        for path in possible_paths:
            if path.exists():
                dst = self.best_model_dir / "best.pt"
                shutil.copy(path, dst)
                print(f"Best model saved to: {dst}")
                return

        print("Warning: Could not find best.pt")

    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint."""
        try:
            path = self.checkpoint_dir / "emergency_checkpoint.pt"
            self.model.save(str(path))
            print(f"Emergency checkpoint: {path}")
        except Exception as e:
            print(f"Could not save emergency checkpoint: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLO Fine-tuning on 20k Selected Dataset (V2 - FIXED)'
    )

    # Paths
    parser.add_argument('--dataset_yaml_path', type=str, required=True,
                        help='Path to dataset YAML')
    parser.add_argument('--project_dir', type=str, default='./yolo_20k_finetune',
                        help='Project output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt_yolo_20k',
                        help='Checkpoint directory')
    parser.add_argument('--best_model_dir', type=str, default='./best_yolo_20k',
                        help='Best model directory')

    # Model
    parser.add_argument('--model_size', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')

    # Pretrained weights (NOT resume!)
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained weights (e.g., last.pt from epoch 200)')

    # Training
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of NEW epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate (should match final LR from previous training)')
    parser.add_argument('--lrf', type=float, default=0.1,
                        help='Final LR ratio (final_lr = lr0 * lrf)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (e.g., 0 or 0,1,2,3)')

    # Checkpointing
    parser.add_argument('--save_period', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')

    # Fine-tuning specific
    parser.add_argument('--freeze_backbone', type=int, default=0,
                        help='Number of backbone layers to freeze')

    parser.add_argument('--verbose', action='store_true', default=True)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("="*60)
    print("YOLO 20K FINE-TUNING SCRIPT V2 (FIXED)")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("="*60)

    # Create trainer and run
    trainer = YOLOFineTunerV2(
        dataset_yaml_path=args.dataset_yaml_path,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        project_dir=args.project_dir,
        checkpoint_dir=args.checkpoint_dir,
        best_model_dir=args.best_model_dir,
        learning_rate=args.learning_rate,
        lrf=args.lrf,
        pretrained_path=args.pretrained_path,
        workers=args.workers,
        device=args.device,
        save_period=args.save_period,
        patience=args.patience,
        verbose=args.verbose,
        freeze_backbone=args.freeze_backbone,
    )

    trainer.train()

    print("\nFine-tuning completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
