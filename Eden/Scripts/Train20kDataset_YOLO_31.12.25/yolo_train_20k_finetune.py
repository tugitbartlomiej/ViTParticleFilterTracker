#!/usr/bin/env python3
"""
YOLO Fine-tuning on 20k Selected Dataset
=========================================

Fine-tune YOLOv8 model starting from epoch 170 checkpoint,
similar to DETR 20k finetune approach.

Key differences from original training:
- Resume from epoch170.pt checkpoint
- Reduced learning rate (0.001 instead of 0.01) for fine-tuning
- Training on 20,000 intelligently selected images
- Extended patience for convergence

Usage:
    python yolo_train_20k_finetune.py \
        --dataset_yaml_path /path/to/20k_dataset.yaml \
        --resume_path /path/to/epoch170.pt \
        --epochs 300 \
        --learning_rate 0.001
"""

import argparse
import json
import os
import shutil
import signal
import sys
import warnings
from pathlib import Path

import yaml

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*NumPy.*")
warnings.filterwarnings("ignore", message=".*NVML.*")

import torch
from ultralytics import YOLO


def coco_to_yolo_format(coco_json_path: Path, output_dir: Path, images_dir: Path) -> Path:
    """
    Convert COCO format annotations to YOLO format.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]

    Args:
        coco_json_path: Path to COCO JSON annotations
        output_dir: Directory for YOLO labels
        images_dir: Directory containing images

    Returns:
        Path to generated dataset.yaml
    """
    print(f"\n{'='*60}")
    print("Converting COCO to YOLO format")
    print(f"{'='*60}")

    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    labels_dir = output_dir / "labels" / "train"
    images_out_dir = output_dir / "images" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # Build image id to info mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Build image id to annotations mapping
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)

    # Category mapping (YOLO uses 0-indexed classes)
    categories = coco_data.get('categories', [])
    cat_id_to_yolo_id = {}
    class_names = []
    for idx, cat in enumerate(sorted(categories, key=lambda x: x['id'])):
        cat_id_to_yolo_id[cat['id']] = idx
        class_names.append(cat['name'])

    print(f"  Categories: {class_names}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")

    # Convert each image
    converted_count = 0
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        file_name = img_info['file_name']

        # Get annotations for this image
        annotations = image_id_to_anns.get(img_id, [])

        # Create YOLO label file
        label_file = labels_dir / (Path(file_name).stem + ".txt")

        with open(label_file, 'w') as f:
            for ann in annotations:
                # COCO bbox: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']

                # Convert to YOLO: center_x, center_y, width, height (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height

                # Clip to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                # Get YOLO class id
                yolo_class = cat_id_to_yolo_id.get(ann['category_id'], 0)

                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # Copy/symlink image if not already in output
        src_image = images_dir / file_name
        dst_image = images_out_dir / file_name

        if src_image.exists() and not dst_image.exists():
            # Use symlink on Linux, copy on Windows
            try:
                dst_image.symlink_to(src_image)
            except OSError:
                shutil.copy2(src_image, dst_image)

        converted_count += 1

    print(f"  Converted: {converted_count} images")

    # Create dataset.yaml
    yaml_content = {
        'path': str(output_dir),
        'train': 'images/train',
        'val': 'images/train',  # Use same for val (or split if needed)
        'names': {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"  Dataset YAML: {yaml_path}")
    print(f"{'='*60}\n")

    return yaml_path


class YOLOFineTuner:
    """
    YOLO Fine-tuning trainer for 20k selected dataset.

    Key features:
    - Resume from existing checkpoint (epoch 170)
    - Reduced learning rate for fine-tuning
    - Extended training with patience
    - Checkpoint saving every N epochs
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
        learning_rate: float = 0.001,  # Reduced LR for fine-tuning
        resume: bool = True,
        resume_path: str = None,
        workers: int = 4,
        device: str = None,
        save_period: int = 5,
        patience: int = 30,
        verbose: bool = True,
        freeze_backbone: int = 0,  # Number of backbone layers to freeze
    ):
        """Initialize YOLO fine-tuner."""
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
        self.resume = resume
        self.resume_path = resume_path
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
        if self.resume and self.resume_path:
            print(f"\n{'='*60}")
            print("YOLO 20K FINE-TUNING - RESUME FROM CHECKPOINT")
            print(f"{'='*60}")
            print(f"  Checkpoint: {self.resume_path}")
            self.model = YOLO(self.resume_path)
        else:
            print(f"\n{'='*60}")
            print("YOLO 20K FINE-TUNING - FRESH START")
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
        """Handle interrupt signal."""
        print("\n[INTERRUPT] Saving emergency checkpoint...")
        emergency_path = self.checkpoint_dir / "emergency_checkpoint.pt"
        self.model.save(str(emergency_path))
        print(f"  Saved to: {emergency_path}")
        sys.exit(0)

    def train(self):
        """Run fine-tuning."""
        print("Starting YOLO fine-tuning on 20k dataset...")

        try:
            # Training arguments
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
                'lrf': 0.01,  # Final LR = lr0 * lrf
                'save_period': self.save_period,
                'patience': self.patience,
                'exist_ok': True,
                'save': True,
                'save_json': True,
                'plots': True,
                'val': True,
                # Fine-tuning specific
                'freeze': self.freeze_backbone,  # Freeze first N layers
                'cos_lr': True,  # Cosine LR scheduler
                'warmup_epochs': 3,  # Warmup epochs
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                # Regularization (reduced for fine-tuning)
                'weight_decay': 0.0005,
                'dropout': 0.0,
            }

            # Add resume if checkpoint exists
            if self.resume and self.resume_path:
                train_args['resume'] = True

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
        description='YOLO Fine-tuning on 20k Selected Dataset'
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

    # Resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to checkpoint (e.g., epoch170.pt)')

    # Training
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs (total, not additional)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate (reduced for fine-tuning)')
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

    # COCO conversion
    parser.add_argument('--coco_json', type=str, default=None,
                        help='Path to COCO JSON (for conversion)')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Path to images directory (for COCO conversion)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for YOLO format (for conversion)')

    parser.add_argument('--verbose', action='store_true', default=True)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("="*60)
    print("YOLO 20K FINE-TUNING SCRIPT")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("="*60)

    # Handle COCO to YOLO conversion if requested
    if args.coco_json and args.images_dir and args.output_dir:
        yaml_path = coco_to_yolo_format(
            coco_json_path=Path(args.coco_json),
            output_dir=Path(args.output_dir),
            images_dir=Path(args.images_dir)
        )
        args.dataset_yaml_path = str(yaml_path)

    # Create trainer and run
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
        resume=args.resume,
        resume_path=args.resume_path,
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
