"""
DETR Local Training Script - Windows Compatible
Fine-tune DETR with background class on local machine
"""

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch._dynamo
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, Dataset
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
)

# Torch compile config (suppress errors for stability)
torch._dynamo.config.suppress_errors = True

class SurgicalToolDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, augment=False):
        print("Initializing background dataset...")
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augment = augment

        # Simple augmentations for local training
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        else:
            self.transform = None

        # Load annotations
        print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        self.categories = coco_data['categories']
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}

        self.image_id_to_image = {img['id']: img for img in coco_data['images']}
        self.image_id_to_anns = {}
        
        print("Mapping annotations to images...")
        for ann in tqdm(coco_data['annotations']):
            img_id = ann['image_id']
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            ann['bbox'] = [float(x) for x in ann['bbox']]
            self.image_id_to_anns[img_id].append(ann)

        self.image_ids = []
        all_image_ids_in_json = set(self.image_id_to_image.keys())

        for img_id in self.image_id_to_anns.keys():
             if img_id in all_image_ids_in_json:
                  self.image_ids.append(img_id)

        images_with_anns_set = set(self.image_ids)
        for img_id in all_image_ids_in_json:
             if img_id not in images_with_anns_set:
                 self.image_ids.append(img_id)
                 if img_id not in self.image_id_to_anns:
                      self.image_id_to_anns[img_id] = []

        self.image_ids = sorted(list(set(self.image_ids)))

        print(f"Dataset initialized. Using {len(self.image_ids)} images.")

    def __len__(self):
        return len(self.image_ids)

    @lru_cache(maxsize=64)  # Smaller cache for local machine
    def _load_image(self, image_path):
        """Cache image loading"""
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_image[image_id]
        annotations = self.image_id_to_anns.get(image_id, [])
        
        formatted_annotations = []
        for ann in annotations:
             formatted_annotations.append({
                 'bbox': ann['bbox'],
                 'category_id': ann['category_id'],
                 'area': ann.get('area', float(ann['bbox'][2] * ann['bbox'][3])),
                 'iscrowd': ann.get('iscrowd', 0)
             })

        image_path = str(self.images_dir / image_info['file_name'])
        try:
            image = self._load_image(image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        if self.augment and self.transform:
            image = self.transform(image)

        target = {'image_id': image_id, 'annotations': formatted_annotations}

        try:
             encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        except Exception as e:
             print(f"Error processing image {image_id} ({image_path}) with processor: {e}")
             return None

        pixel_values = encoding["pixel_values"].squeeze(0)
        pixel_mask = encoding["pixel_mask"].squeeze(0)

        if not encoding["labels"]:
             labels = {'class_labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.tensor([], dtype=torch.float32)}
        else:
             labels = encoding["labels"][0]

        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    try:
        batch_size = len(batch)
        first_item = batch[0]
        
        pixel_values = torch.empty(
            batch_size, 
            *first_item["pixel_values"].shape, 
            dtype=first_item["pixel_values"].dtype
        )
        pixel_mask = torch.empty(
            batch_size, 
            *first_item["pixel_mask"].shape, 
            dtype=first_item["pixel_mask"].dtype
        )
        
        for i, item in enumerate(batch):
            pixel_values[i] = item["pixel_values"]
            pixel_mask[i] = item["pixel_mask"]
            
        labels = [item["labels"] for item in batch]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}
    except Exception as e:
        print(f"Error during collate_fn: {e}")
        return None

def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file based on epoch number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"[Checkpoint] Directory not found: {checkpoint_dir}")
        return None

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        print(f"[Checkpoint] No checkpoints found in {checkpoint_dir}")
        return None

    try:
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        print(f"[Checkpoint] Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        print(f"[Checkpoint] Error parsing checkpoint filenames: {e}")
        return None

def load_checkpoint_for_negative_training(checkpoint_path, device, id2label, label2id, model_checkpoint_name="facebook/detr-resnet-50"):
    """Loads model from checkpoint for negative sample training (background images as hard negatives)"""
    print(f"[Fine-tuning] Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"[Fine-tuning] Checkpoint loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # Create new config with updated classes
        config = DetrConfig.from_pretrained(model_checkpoint_name)
        config.num_labels = len(id2label)
        config.id2label = id2label
        config.label2id = label2id
        
        print(f"[Fine-tuning] Creating model with {len(id2label)} classes")
        
        # Create model with new config
        model = DetrForObjectDetection(config).to(device)
        
        # Load state dict from checkpoint (handle nested structure)
        if 'model_state_dict' in checkpoint:
            if 'model_state_dict' in checkpoint['model_state_dict']:
                # Double nested structure
                state_dict = checkpoint['model_state_dict']['model_state_dict']
            else:
                # Single nested structure
                state_dict = checkpoint['model_state_dict']
        else:
            # Direct state dict
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            new_state_dict[key] = v
        
        # No modification of classification head needed - background training uses same classes
        # Background images serve as negative examples (hard negatives) with empty annotations
        print(f"[Fine-tuning] Using original classification head with {len(id2label)} classes")
        print(f"[Fine-tuning] Background images will be negative examples to reduce false positives")
        
        # Ensure all tensors in state_dict are on the correct device
        for key in new_state_dict:
            if torch.is_tensor(new_state_dict[key]):
                new_state_dict[key] = new_state_dict[key].to(device)
        
        # Load modified state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"[Fine-tuning] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[Fine-tuning] Unexpected keys: {unexpected_keys}")
        print(f"[Fine-tuning] Model successfully loaded from checkpoint for negative training")
            
        return model
        
    except Exception as e:
        print(f"[Fine-tuning] Error loading model from checkpoint: {e}")
        raise e

def train(args):
    print("[INFO] Starting DETR background fine-tuning (Local)...")
    print(f"Arguments: {args}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Setup directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = output_dir / "logs"
    logging_dir.mkdir(exist_ok=True)
    # writer = SummaryWriter(log_dir=str(logging_dir))
    writer = None  # Temporarily disabled TensorBoard logging

    # Load original categories from tooltip model checkpoint
    print("Loading categories from tooltip model...")
    
    try:
        tooltip_checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'model_config' in tooltip_checkpoint and 'id2label' in tooltip_checkpoint['model_config']:
            # Load from model config if available
            id2label = tooltip_checkpoint['model_config']['id2label']
            # Convert string keys to integers
            id2label = {int(k): v for k, v in id2label.items()}
        else:
            # Fallback: load from original tooltip dataset
            tooltip_dataset_path = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\coco_annotations_from_yolo_dataset_20250218.json"
            with open(tooltip_dataset_path, 'r') as f:
                tooltip_data = json.load(f)
            tooltip_categories = tooltip_data['categories']
            id2label = {cat['id']: cat['name'] for cat in tooltip_categories}
        
        label2id = {v: k for k, v in id2label.items()}
        print(f"Original tooltip categories loaded: {id2label}")
        print("Background images will be used as negative samples (hard negatives) to reduce false positives")
        
    except Exception as e:
        print(f"Error loading tooltip categories: {e}")
        print("Using fallback: single tool class")
        id2label = {0: 'tool'}
        label2id = {'tool': 0}

    # Load processor
    print("Loading image processor...")
    processor = DetrImageProcessor.from_pretrained(
        args.model_checkpoint,
        size={"longest_edge": 1333, "shortest_edge": 800},
        do_resize=True,
        do_normalize=True
    )

    # Create dataset with background images (negative samples/hard negatives)
    print("Creating dataset with background images as negative samples...")
    print("These images have empty annotations and will help reduce false positives")
    full_dataset = SurgicalToolDataset(
        images_dir=args.background_images_dir,
        annotations_file=args.background_annotations_path,
        processor=processor,
        augment=args.augment
    )

    # Split dataset
    if args.train_val_split < 1.0:
        train_size = int(args.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        val_dataset = None

    print(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
    num_workers = min(args.num_workers, 2)  # Limit for local machine
    print(f"Using {num_workers} workers")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == 'cuda',
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == 'cuda',
        )

    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        model = load_checkpoint_for_negative_training(
            args.checkpoint_path,
            device,
            id2label,
            label2id,
            args.model_checkpoint
        )
    else:
        print("No checkpoint provided, starting from pretrained model")
        config = DetrConfig.from_pretrained(args.model_checkpoint)
        config.num_labels = len(id2label)
        config.id2label = id2label
        config.label2id = label2id
        model = DetrForObjectDetection(config).to(device)
    
    # Compile model if available and requested
    if hasattr(torch, 'compile') and args.compile_model:
        print("Compiling model...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"Compilation failed: {e}, continuing without compilation")
    
    print(f"Model loaded successfully")

    # Setup optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        fused=False,  # Compatibility fix
        foreach=False
    )

    # Initialize GradScaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp and device.type == 'cuda')
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume_training:
        latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint_path:
            print(f"Resuming from {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)

    # Training loop
    print(f"Starting training from epoch {start_epoch}...")
    print(f"Mixed Precision: {'Enabled' if args.use_amp and device.type == 'cuda' else 'Disabled'}")
    
    best_val_loss = float('inf')
    patience_counter = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
                
            model.train()
            train_loss = 0.0
            processed_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Training E{epoch+1}")

            optimizer.zero_grad()
            
            for i, batch in enumerate(progress_bar):
                if batch is None:
                    continue

                # Move batch to device
                batch_on_device = {
                    'pixel_values': batch['pixel_values'].to(device, non_blocking=True),
                    'pixel_mask': batch['pixel_mask'].to(device, non_blocking=True),
                    'labels': []
                }
                
                for label_dict in batch['labels']:
                    processed_dict = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in label_dict.items()
                    }
                    batch_on_device['labels'].append(processed_dict)

                # Forward pass with mixed precision
                with autocast(enabled=args.use_amp and device.type == 'cuda'):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss / args.gradient_accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Update stats
                batch_loss = loss.item() * args.gradient_accumulation_steps
                train_loss += batch_loss
                processed_batches += 1
                
                progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

            avg_train_loss = train_loss / processed_batches
            
            print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
            if writer:
                writer.add_scalar("Loss/train", avg_train_loss, epoch)

            # Validation
            if val_dataloader:
                model.eval()
                val_loss = 0.0
                processed_val_batches = 0
                
                print("Running validation...")
                progress_bar_val = tqdm(val_dataloader, desc=f"Validation E{epoch+1}")

                with torch.no_grad():
                    for batch_val in progress_bar_val:
                        if batch_val is None:
                            continue

                        batch_on_device_val = {
                            'pixel_values': batch_val['pixel_values'].to(device, non_blocking=True),
                            'pixel_mask': batch_val['pixel_mask'].to(device, non_blocking=True),
                            'labels': []
                        }
                        
                        for label_dict in batch_val['labels']:
                            processed_dict_val = {
                                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                for k, v in label_dict.items()
                            }
                            batch_on_device_val['labels'].append(processed_dict_val)

                        with autocast(enabled=args.use_amp and device.type == 'cuda'):
                            outputs_val = model(**batch_on_device_val)
                            loss_val = outputs_val.loss

                        batch_loss_val = loss_val.item()
                        val_loss += batch_loss_val
                        processed_val_batches += 1
                        
                        progress_bar_val.set_postfix({"loss": f"{batch_loss_val:.4f}"})

                avg_val_loss = val_loss / processed_val_batches
                
                print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
                if writer:
                    writer.add_scalar("Loss/validation", avg_val_loss, epoch)

                # Save checkpoint
                if (epoch + 1) % args.save_interval == 0:
                    chkpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': avg_val_loss,
                    }, chkpt_path)
                    print(f"Checkpoint saved: {chkpt_path}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    print(f"Validation improved: {best_val_loss:.4f} → {avg_val_loss:.4f}")
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = checkpoint_dir / "best_model.pth"
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved: {best_model_path}")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epoch(s)")
                    if patience_counter >= args.patience:
                        print("Early stopping triggered")
                        break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model
        print("Saving final model...")
        final_path = checkpoint_dir / "final_model.pth"
        torch.save(model.state_dict(), final_path)
        print(f"Final model saved: {final_path}")
        
        if writer:
            writer.close()
        print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description="DETR Local Training - Background Fine-tuning")
    
    # Paths - with Windows defaults
    parser.add_argument("--checkpoint_path", type=str, 
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\DETR\detr_inference_model.pth",
                       help="Path to pretrained DETR checkpoint (.pth file)")
    parser.add_argument("--background_images_dir", type=str, 
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\Background\train",
                       help="Path to background images directory")
    parser.add_argument("--background_annotations_path", type=str,
                       default=r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\Datasets\Detr\Background\annotations\train_annotations.json",
                       help="Path to background annotations JSON file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_local")
    parser.add_argument("--output_dir", type=str, default="./output_local")
    
    # Model
    parser.add_argument("--model_checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--num_queries", type=int, default=100)
    
    # Dataset
    parser.add_argument("--train_val_split", type=float, default=0.85)
    parser.add_argument("--augment", action='store_true', help="Enable data augmentation")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (start small for local)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Windows)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_backbone", type=float, default=5e-6, help="Backbone learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    
    # Optimization
    parser.add_argument("--use_amp", action='store_true', default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (increase if small batch)")
    parser.add_argument("--compile_model", action='store_true',
                        help="Use torch.compile (experimental)")
    
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--resume_training", action='store_true')
    
    args = parser.parse_args()
    
    print("="*60)
    print("[DETR] Local Training")
    print("="*60)
    
    train(args)

if __name__ == "__main__":
    main()