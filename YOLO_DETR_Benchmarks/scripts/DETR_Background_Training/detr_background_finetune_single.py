import argparse
import json
import os
from functools import lru_cache  # Dla cache'owania
from pathlib import Path

import torch
import torch._dynamo
from PIL import Image
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
)

torch._dynamo.config.suppress_errors = True  # Dla torch.compile
# torch._dynamo.config.guard_fail_fn = lambda guard, *_: None  # Not available in this PyTorch version

# --- Optimized Dataset Class ---
class SurgicalToolDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, augment=False):
        print("Initializing background dataset...")
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augment = augment

        # Bardziej wydajne augmentacje
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = None

        # Load annotations
        print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
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

        print(f"Background dataset initialized. Using {len(self.image_ids)} images found in annotations and image list.")

    def __len__(self):
        return len(self.image_ids)

    @lru_cache(maxsize=128)  # Cache dla często używanych obrazów
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
             print(f"Target annotations passed: {target['annotations']}")
             return None

        pixel_values = encoding["pixel_values"].squeeze(0)
        pixel_mask = encoding["pixel_mask"].squeeze(0)

        if not encoding["labels"]:
             labels = {'class_labels': torch.tensor([], dtype=torch.int64), 'boxes': torch.tensor([], dtype=torch.float32)}
        else:
             labels = encoding["labels"][0]

        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}

# --- Optimized Collate Function ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    try:
        # Pre-allocate tensory dla wydajności
        batch_size = len(batch)
        first_item = batch[0]
        
        # Alokuj tensory z góry
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
        
        # Wypełnij tensory
        for i, item in enumerate(batch):
            pixel_values[i] = item["pixel_values"]
            pixel_mask[i] = item["pixel_mask"]
            
        labels = [item["labels"] for item in batch]
        return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}
    except Exception as e:
        print(f"Error during collate_fn: {e}")
        return None

# --- Checkpoint Helper Functions ---
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

def load_checkpoint_and_modify_for_background(checkpoint_path, device, id2label, label2id, model_checkpoint_name="facebook/detr-resnet-50"):
    """Loads model from checkpoint and modifies classification head for background class"""
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
        model = DetrForObjectDetection(config).to(device, memory_format=torch.contiguous_format)
        
        # Load state dict from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Debug: Print some key information
        print(f"[Fine-tuning] Total keys in checkpoint: {len(new_state_dict)}")
        class_keys = [k for k in new_state_dict.keys() if 'class' in k.lower()]
        print(f"[Fine-tuning] Keys containing 'class': {class_keys}")
        embed_keys = [k for k in new_state_dict.keys() if 'embed' in k.lower()]
        print(f"[Fine-tuning] Keys containing 'embed': {embed_keys}")
        
        # Handle classification head modification
        original_num_classes = 1  # Original model had surgical_tool only
        new_num_classes = len(id2label)
        
        if new_num_classes != original_num_classes:
            print(f"[Fine-tuning] Modifying classification head from {original_num_classes} to {new_num_classes} classes")
            
            # Get original classification weights (use correct key names from checkpoint)
            class_weight_key = 'class_labels_classifier.weight'
            class_bias_key = 'class_labels_classifier.bias'
            
            if class_weight_key not in new_state_dict:
                print(f"[Fine-tuning] Available keys: {list(new_state_dict.keys())[:10]}...")
                raise KeyError(f"Expected key '{class_weight_key}' not found in checkpoint")
                
            old_class_weight = new_state_dict[class_weight_key].to(device)  # Shape: [2, hidden_dim] (1 class + no-object)
            old_class_bias = new_state_dict[class_bias_key].to(device)      # Shape: [2]
            
            # Create new classification head weights - ensure on correct device
            hidden_dim = old_class_weight.size(1)
            new_class_weight = torch.zeros(new_num_classes + 1, hidden_dim, dtype=old_class_weight.dtype, device=device)  # +1 for no-object
            new_class_bias = torch.zeros(new_num_classes + 1, dtype=old_class_bias.dtype, device=device)
            
            # Initialize new weights
            torch.nn.init.xavier_uniform_(new_class_weight)
            torch.nn.init.constant_(new_class_bias, 0.)
            
            # Copy weights for existing classes
            num_classes_to_copy = min(original_num_classes + 1, new_num_classes + 1)
            new_class_weight[:num_classes_to_copy] = old_class_weight[:num_classes_to_copy]
            new_class_bias[:num_classes_to_copy] = old_class_bias[:num_classes_to_copy]
            
            # Update state dict with new classification head
            new_state_dict[class_weight_key] = new_class_weight
            new_state_dict[class_bias_key] = new_class_bias
            
            print(f"[Fine-tuning] Copied weights for {original_num_classes} existing classes")
            print(f"[Fine-tuning] Added background class")
        
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
        print(f"[Fine-tuning] Model successfully loaded and modified from checkpoint")
            
        return model
        
    except Exception as e:
        print(f"[Fine-tuning] Error loading model from checkpoint: {e}")
        raise e

def load_checkpoint(checkpoint_path, model, optimizer, scaler, device):
    """Loads model and optimizer state from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        print(f"[Checkpoint] Checkpoint file not found: {checkpoint_path}")
        return model, optimizer, scaler, 0

    print(f"[Checkpoint] Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle both DDP and non-DDP checkpoints
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        print(f"[Checkpoint] Successfully loaded state from epoch {start_epoch}")
        return model, optimizer, scaler, start_epoch + 1
    except Exception as e:
        print(f"[Checkpoint] Error loading checkpoint: {e}")
        return model, optimizer, scaler, 0

# --- Main Training Function ---
def train(args):
    print("Starting DETR background fine-tuning (Single GPU)...")
    print(f"Arguments passed or defaults used: {args}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable TF32 for A100/H100 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    best_model_dir = Path(args.best_model_dir)
    final_model_dir = output_dir / "final_model"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = output_dir / "logs"
    logging_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(logging_dir))

    # Define categories for fine-tuning (original + background)
    print("Setting up categories for background fine-tuning...")
    
    try:
        # We know the original model had 1 class (surgical_tool)
        # Now we add background class
        combined_categories = [
            {'id': 1, 'name': 'surgical_tool', 'supercategory': 'medical_instrument'},
            {'id': 2, 'name': 'background', 'supercategory': 'scene'}
        ]
        
        id2label = {cat['id']: cat['name'] for cat in combined_categories}
        label2id = {v: k for k, v in id2label.items()}
        
        print(f"Categories for fine-tuning: {id2label}")
            
    except Exception as e:
        print(f"Error setting up categories: {e}")
        return

    # Load processor
    print(f"Loading image processor from checkpoint: {args.model_checkpoint}")
    try:
        processor = DetrImageProcessor.from_pretrained(
            args.model_checkpoint,
            size={"longest_edge": 1333, "shortest_edge": 800},  # Replace deprecated max_size
            do_resize=True,
            do_normalize=True
        )
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    # Create dataset
    print("Creating background dataset...")
    try:
        full_dataset = SurgicalToolDataset(
            images_dir=args.background_images_dir,
            annotations_file=args.background_annotations_path,
            processor=processor,
            augment=args.augment
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

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

    # Adjust num_workers for single GPU
    effective_num_workers = min(args.num_workers, 4)  # Limit for single GPU
    print(f"Using {effective_num_workers} workers (requested: {args.num_workers})")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if effective_num_workers > 0 else False,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=effective_num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if effective_num_workers > 0 else False,
            prefetch_factor=2 if effective_num_workers > 0 else None,
        )

    # Load model from checkpoint and modify for background class
    print(f"Loading and modifying model from checkpoint: {args.checkpoint_path}")
    
    try:
        model = load_checkpoint_and_modify_for_background(
            args.checkpoint_path,
            device,
            id2label,
            label2id,
            args.model_checkpoint
        )
        
        # Compile model dla dodatkowej wydajności (PyTorch 2.0+)
        if hasattr(torch, 'compile') and args.compile_model:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        
        print(f"Model loaded successfully on {device}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Setup optimizer
    try:
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
            fused=False,  # Disable fused AdamW to fix dtype mismatch with AMP
            foreach=False  # For full dtype compatibility
        )
    except Exception as e:
        print(f"Error setting up optimizer: {e}")
        return

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=args.use_amp)
    
    # Load checkpoint if resuming (this would be for resuming fine-tuning)
    start_epoch = 0
    if args.resume_training:
        latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint_path:
            model, optimizer, scaler, start_epoch = load_checkpoint(
                latest_checkpoint_path, model, optimizer, scaler, device
            )

    # Training loop
    print(f"Starting background fine-tuning loop from epoch {start_epoch}...")
    print(f"Mixed Precision Training: {'Enabled' if args.use_amp else 'Disabled'}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Gradient accumulation
    accumulation_steps = args.gradient_accumulation_steps

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n--- Background Fine-tuning Epoch {epoch+1}/{args.epochs} ---")
                
            model.train()
            train_loss = 0.0
            processed_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Background Fine-tuning E{epoch+1}")

            optimizer.zero_grad()
            
            for i, batch in enumerate(progress_bar):
                if batch is None:
                    continue

                # Move batch to device (optimized)
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
                with autocast(enabled=args.use_amp):
                    outputs = model(**batch_on_device)
                    loss = outputs.loss / accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Update stats
                batch_loss = loss.item() * accumulation_steps
                train_loss += batch_loss
                processed_batches += 1
                
                progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
                # Log less frequently for performance
                if i % 10 == 0:
                    writer.add_scalar("Loss/train_batch", batch_loss, epoch * len(train_dataloader) + i)

            avg_train_loss = train_loss / processed_batches
            
            print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

            # Validation
            if val_dataloader:
                model.eval()
                val_loss = 0.0
                processed_val_batches = 0
                
                print("Running validation...")
                progress_bar_val = tqdm(val_dataloader, desc=f"Validation E{epoch+1}")

                with torch.no_grad():
                    for i_val, batch_val in enumerate(progress_bar_val):
                        if batch_val is None:
                            continue

                        # Move batch to device (optimized)
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

                        with autocast(enabled=args.use_amp):
                            outputs_val = model(**batch_on_device_val)
                            loss_val = outputs_val.loss

                        batch_loss_val = loss_val.item()
                        val_loss += batch_loss_val
                        processed_val_batches += 1
                        
                        progress_bar_val.set_postfix({"loss": f"{batch_loss_val:.4f}"})

                avg_val_loss = val_loss / processed_val_batches
                
                print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
                writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch)

                # Save checkpoint and best model
                if (epoch + 1) % args.save_interval == 0:
                    chkpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': avg_val_loss,
                    }, chkpt_path)
                    print(f"Checkpoint saved to {chkpt_path}")

                # Save best model
                if avg_val_loss < best_val_loss:
                    print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving best model...")
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    model.save_pretrained(str(best_model_dir))
                    processor.save_pretrained(str(best_model_dir))
                    print(f"Best model saved to: {best_model_dir}")
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve for {patience_counter} epoch(s).")
                    if patience_counter >= args.patience:
                        print(f"Early stopping triggered.")
                        break

    except KeyboardInterrupt:
        print("\nBackground fine-tuning interrupted by user.")
    finally:
        # Save final model
        print("\nSaving final background fine-tuned model...")
        try:
            model.save_pretrained(str(final_model_dir))
            processor.save_pretrained(str(final_model_dir))
            print(f"Final model saved to: {final_model_dir}")
            
            # Save final checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, checkpoint_dir / "final_checkpoint.pth")
            
        except Exception as e:
            print(f"Error saving final model: {e}")
            
        writer.close()
        
        print("Background fine-tuning completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DETR with background class - Single GPU Version")
    
    # Paths
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to pretrained DETR checkpoint (.pth file)")
    parser.add_argument("--background_images_dir", type=str, required=True)
    parser.add_argument("--background_annotations_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--best_model_dir", type=str, default="./best_model")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # Model
    parser.add_argument("--model_checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--num_queries", type=int, default=100)
    
    # Dataset
    parser.add_argument("--train_val_split", type=float, default=0.9)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--background_weight", type=float, default=1.0,
                       help="Weight for background samples (kept for compatibility)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    
    # Optimization
    parser.add_argument("--use_amp", action='store_true', default=True, 
                        help="Use automatic mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--compile_model", action='store_true', default=False,
                        help="Use torch.compile for model optimization (requires PyTorch 2.0+)")
    
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume_training", action='store_true')
    
    args = parser.parse_args()
    
    train(args)