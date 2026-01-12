#!/usr/bin/env python3
"""
YOLO Fine-tuning on 20k Selected Dataset - V2
==============================================

Fine-tune YOLOv8 model from pretrained weights.
Properly handles continuing training beyond original epochs.

Key fix: Separates loading weights from resume logic.
- --pretrained_path: Load weights from checkpoint (starts fresh training)
- --resume: Actually resume interrupted training (only for mid-training)
"""

import argparse
import os
import shutil
import signal
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*NVML.*")

import torch
from ultralytics import YOLO


class YOLOFineTuner:
    def __init__(
        self,
        dataset_yaml_path: str,
        model_size: str = 'm',
        epochs: int = 500,
        batch_size: int = 16,
        imgsz: int = 640,
        project_dir: str = './yolo_20k_finetune',
        checkpoint_dir: str = './ckpt_yolo_20k',
        best_model_dir: str = './best_yolo_20k',
        learning_rate: float = 0.001,
        pretrained_path: str = None,
        resume: bool = False,
        workers: int = 4,
        device: str = None,
        save_period: int = 5,
        patience: int = 30,
        verbose: bool = True,
        freeze_backbone: int = 0,
    ):
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
        self.pretrained_path = pretrained_path
        self.resume = resume
        self.workers = workers
        self.save_period = save_period
        self.patience = patience
        self.verbose = verbose
        self.freeze_backbone = freeze_backbone

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

        if not self.dataset_yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml_path}")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # FIXED: Load model from pretrained_path if provided (as pretrained, NOT resume)
        if self.pretrained_path:
            print(f"\n{'='*60}")
            print("YOLO 20K FINE-TUNING - FROM PRETRAINED WEIGHTS")
            print(f"{'='*60}")
            print(f"  Pretrained weights: {self.pretrained_path}")
            print(f"  Training {self.epochs} NEW epochs (not resuming)")
            self.model = YOLO(self.pretrained_path)
        else:
            print(f"\n{'='*60}")
            print("YOLO 20K FINE-TUNING - FRESH START")
            print(f"{'='*60}")
            print(f"  Model: YOLOv8{self.model_size}")
            self.model = YOLO(f'yolov8{self.model_size}.pt')

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        self._print_config()

    def _print_config(self):
        print(f"\nConfiguration:")
        print(f"  Dataset: {self.dataset_yaml_path}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.imgsz}")
        print(f"  Learning rate: {self.learning_rate}")
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
        print("\n[INTERRUPT] Saving emergency checkpoint...")
        emergency_path = self.checkpoint_dir / "emergency_checkpoint.pt"
        self.model.save(str(emergency_path))
        print(f"  Saved to: {emergency_path}")
        sys.exit(0)

    def train(self):
        print("Starting YOLO fine-tuning on 20k dataset...")

        try:
            train_args = {
                'data': str(self.dataset_yaml_path),
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.imgsz,
                'device': self.device,
                'project': str(self.project_dir),
                'name': 'exp',
                'pretrained': True,
                'verbose': self.verbose,
                'workers': self.workers,
                'lr0': self.learning_rate,
                'lrf': 0.01,
                'save_period': self.save_period,
                'patience': self.patience,
                'exist_ok': True,
                'save': True,
                'save_json': True,
                'plots': True,
                'val': True,
                'freeze': self.freeze_backbone,
                'cos_lr': True,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'weight_decay': 0.0005,
                'dropout': 0.0,
            }

            # FIXED: Only add resume=True for actual interrupted training resume
            # NOT for continuing training with more epochs from a completed checkpoint
            if self.resume:
                print("[WARNING] Using resume=True - only use for interrupted training!")
                train_args['resume'] = True

            results = self.model.train(**train_args)

            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)

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
        try:
            path = self.checkpoint_dir / "emergency_checkpoint.pt"
            self.model.save(str(path))
            print(f"Emergency checkpoint: {path}")
        except Exception as e:
            print(f"Could not save emergency checkpoint: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO Fine-tuning on 20k Selected Dataset (V2)'
    )

    parser.add_argument('--dataset_yaml_path', type=str, required=True)
    parser.add_argument('--project_dir', type=str, default='./yolo_20k_finetune')
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt_yolo_20k')
    parser.add_argument('--best_model_dir', type=str, default='./best_yolo_20k')
    parser.add_argument('--model_size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'])

    # FIXED: New argument for pretrained weights (not resume)
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained weights (e.g., last.pt from epoch 200)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted training (NOT for completed training!)')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_period', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--freeze_backbone', type=int, default=0)
    parser.add_argument('--verbose', action='store_true', default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("YOLO 20K FINE-TUNING SCRIPT (V2 - FIXED RESUME LOGIC)")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("="*60)

    trainer = YOLOFineTuner(
        dataset_yaml_path=args.dataset_yaml_path,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        project_dir=args.project_dir,
        checkpoint_dir=args.checkpoint_dir,
        best_model_dir=args.best_model_dir,
        learning_rate=args.learning_rate,
        pretrained_path=args.pretrained_path,
        resume=args.resume,
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
