import argparse
import json
import os
from functools import lru_cache  # Dla cache'owania
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
)

torch._dynamo.config.suppress_errors = True  # Dla torch.compile

# --- Setup DDP ---
def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not running in distributed mode")
        return None, None, None
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Optimized Dataset Class ---
class SurgicalToolDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, augment=False):
        if dist.get_rank() == 0:
            print("Initializing dataset...")
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augment = augment

        # Bardziej wydajne augmentacje
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # Usuń ColorJitter jeśli nie jest krytyczny - jest wolny
                # transforms.RandomApply([
                #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
                # ], p=0.8),
            ])
        else:
            self.transform = None

        # Load annotations
        if dist.get_rank() == 0:
            print(f"Loading annotations from: {annotations_file}")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        self.categories = coco_data['categories']
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}

        self.image_id_to_image = {img['id']: img for img in coco_data['images']}
        self.image_id_to_anns = {}
        
        if dist.get_rank() == 0:
            print("Mapping annotations to images...")
            ann_iterator = tqdm(coco_data['annotations'])
        else:
            ann_iterator = coco_data['annotations']
            
        for ann in ann_iterator:
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

        if dist.get_rank() == 0:
            print(f"Dataset initialized. Using {len(self.image_ids)} images found in annotations and image list.")

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
            if dist.get_rank() == 0:
                print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Error loading image {image_path}: {e}")
            return None

        if self.augment and self.transform:
            image = self.transform(image)

        target = {'image_id': image_id, 'annotations': formatted_annotations}

        try:
             encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        except Exception as e:
             if dist.get_rank() == 0:
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
        if dist.get_rank() == 0:
            print(f"Error during collate_fn: {e}")
        return None

# --- Checkpoint Helper Functions (modified for DDP) ---
def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file based on epoch number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Directory not found: {checkpoint_dir}")
        return None

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        if dist.get_rank() == 0:
            print(f"[Checkpoint] No checkpoints found in {checkpoint_dir}")
        return None

    try:
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Error parsing checkpoint filenames: {e}")
        return None

def load_checkpoint(checkpoint_path, model, optimizer, scaler, device):
    """Loads model and optimizer state from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Checkpoint file not found: {checkpoint_path}")
        return model, optimizer, scaler, 0

    if dist.get_rank() == 0:
        print(f"[Checkpoint] Loading checkpoint from: {checkpoint_path}")
    
    # Use map_location to handle DDP saved models
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

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
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Successfully loaded state from epoch {start_epoch}")
        return model, optimizer, scaler, start_epoch + 1
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Error loading checkpoint: {e}")
        return model, optimizer, scaler, 0

# --- Main Training Function ---
def train(args):
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    if rank == 0:
        print("Starting training script...")
        print(f"Arguments passed or defaults used: {args}")
        print(f"Running on {world_size} GPUs")

    # Setup device
    device = torch.device(f'cuda:{local_rank}')
    
    # Enable TF32 for A100/H100 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup directories (only on main process)
    if rank == 0:
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
    
    # Ensure all processes wait for directories to be created
    dist.barrier()

    # Load categories
    if rank == 0:
        print("Loading category information from annotations file...")
    
    try:
        with open(args.annotations_path, 'r') as f:
            coco_data = json.load(f)
        categories = coco_data.get('categories', [])
        if not categories:
             raise ValueError("No 'categories' section found or it's empty in annotations file.")
        id2label = {cat['id']: cat['name'] for cat in categories}
        label2id = {v: k for k, v in id2label.items()}
        if rank == 0:
            print(f"Categories loaded: {id2label}")
    except Exception as e:
        if rank == 0:
            print(f"Error loading categories: {e}")
        cleanup_ddp()
        return

    # Load processor
    if rank == 0:
        print(f"Loading image processor from checkpoint: {args.model_checkpoint}")
    try:
        processor = DetrImageProcessor.from_pretrained(args.model_checkpoint)
    except Exception as e:
        if rank == 0:
            print(f"Error loading processor: {e}")
        cleanup_ddp()
        return

    # Create dataset
    if rank == 0:
        print("Creating datasets...")
    try:
        full_dataset = SurgicalToolDataset(
            images_dir=args.images_dir,
            annotations_file=args.annotations_path,
            processor=processor,
            augment=args.augment
        )
    except Exception as e:
        if rank == 0:
            print(f"Error creating dataset: {e}")
        cleanup_ddp()
        return

    # Split dataset
    if args.train_val_split < 1.0:
        train_size = int(args.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        val_dataset = None

    if rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation dataset size: {len(val_dataset)}")

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    val_sampler = None
    if val_dataset:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

    # Create optimized data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Important for DDP
        persistent_workers=True if args.num_workers > 0 else False,  # Optymalizacja
        prefetch_factor=2 if args.num_workers > 0 else None,  # Optymalizacja
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    # Load model
    if rank == 0:
        print(f"Loading model configuration from checkpoint: {args.model_checkpoint}")
    
    try:
        config = DetrConfig.from_pretrained(
            args.model_checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        config.num_queries = args.num_queries
        
        model = DetrForObjectDetection.from_pretrained(
            args.model_checkpoint,
            config=config,
            ignore_mismatched_sizes=True
        ).to(device)
        
        # Compile model dla dodatkowej wydajności (PyTorch 2.0+)
        if hasattr(torch, 'compile') and args.compile_model:
            if rank == 0:
                print("Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
        
        # Wrap model in DDP
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False  # Optymalizacja - ustaw True tylko jeśli potrzebne
        )
        
        if rank == 0:
            print(f"Model loaded successfully with DDP on GPU {local_rank}")
            
    except Exception as e:
        if rank == 0:
            print(f"Error loading model: {e}")
        cleanup_ddp()
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
            fused=True if torch.cuda.is_available() else False  # Fused optimizer dla lepszej wydajności
        )
    except Exception as e:
        if rank == 0:
            print(f"Error setting up optimizer: {e}")
        cleanup_ddp()
        return

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=args.use_amp)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_training:
        latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint_path:
            # Note: pass the base model (not DDP wrapped) for loading
            model.module, optimizer, scaler, start_epoch = load_checkpoint(
                latest_checkpoint_path, model.module, optimizer, scaler, device
            )

    # Training loop
    if rank == 0:
        print(f"Starting training loop from epoch {start_epoch}...")
        print(f"Mixed Precision Training: {'Enabled' if args.use_amp else 'Disabled'}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Gradient accumulation
    accumulation_steps = args.gradient_accumulation_steps

    try:
        for epoch in range(start_epoch, args.epochs):
            # Set epoch for distributed sampler (important for shuffling)
            train_sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
                
            model.train()
            train_loss = 0.0
            processed_batches = 0
            
            # Create progress bar only on main process
            if rank == 0:
                progress_bar = tqdm(train_dataloader, desc=f"Training E{epoch+1}")
            else:
                progress_bar = train_dataloader

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
                
                if rank == 0:
                    progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
                    # Log less frequently for performance
                    if i % 10 == 0:
                        writer.add_scalar("Loss/train_batch", batch_loss, epoch * len(train_dataloader) + i)

            # Aggregate loss across all processes
            train_loss_tensor = torch.tensor(train_loss).to(device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / (processed_batches * world_size)
            
            if rank == 0:
                print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
                writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

            # Validation
            if val_dataloader:
                if val_sampler:
                    val_sampler.set_epoch(epoch)
                    
                model.eval()
                val_loss = 0.0
                processed_val_batches = 0
                
                if rank == 0:
                    print("Running validation...")
                    progress_bar_val = tqdm(val_dataloader, desc=f"Validation E{epoch+1}")
                else:
                    progress_bar_val = val_dataloader

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
                        
                        if rank == 0:
                            progress_bar_val.set_postfix({"loss": f"{batch_loss_val:.4f}"})

                # Aggregate validation loss
                val_loss_tensor = torch.tensor(val_loss).to(device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor.item() / (processed_val_batches * world_size)
                
                if rank == 0:
                    print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
                    writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch)

                    # Save checkpoint and best model (only on main process)
                    if (epoch + 1) % args.save_interval == 0:
                        chkpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict(),  # Save unwrapped model
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
                        model.module.save_pretrained(str(best_model_dir))
                        processor.save_pretrained(str(best_model_dir))
                        print(f"Best model saved to: {best_model_dir}")
                    else:
                        patience_counter += 1
                        if rank == 0:
                            print(f"Validation loss did not improve for {patience_counter} epoch(s).")
                        if patience_counter >= args.patience:
                            if rank == 0:
                                print(f"Early stopping triggered.")
                            break

    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user.")
    finally:
        # Save final model (only on main process)
        if rank == 0:
            print("\nSaving final model...")
            try:
                model.module.save_pretrained(str(final_model_dir))
                processor.save_pretrained(str(final_model_dir))
                print(f"Final model saved to: {final_model_dir}")
                
                # Save final checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, checkpoint_dir / "final_checkpoint.pth")
                
            except Exception as e:
                print(f"Error saving final model: {e}")
                
            writer.close()
            
        # Clean up DDP
        cleanup_ddp()
        
        if rank == 0:
            print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR with DDP - Optimized Version")
    
    # Paths
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--annotations_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_ddp")
    parser.add_argument("--best_model_dir", type=str, default="./best_model_ddp")
    parser.add_argument("--output_dir", type=str, default="./output_ddp")
    
    # Model
    parser.add_argument("--model_checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--num_queries", type=int, default=100)
    
    # Dataset
    parser.add_argument("--train_val_split", type=float, default=0.9)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)  # Zwiększone z 2 na 4
    
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