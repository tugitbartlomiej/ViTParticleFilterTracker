import argparse
import json
import math
import os
# lru_cache removed - not DDP-safe
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

    # NOTE: lru_cache removed - not DDP-safe, can cause process desync
    def _load_image(self, image_path):
        """Load image from path"""
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
    """Finds the latest checkpoint file based on epoch number or final_checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Directory not found: {checkpoint_dir}")
        return None

    # First check for final_checkpoint.pth (saved on crash/completion)
    final_checkpoint = checkpoint_dir / "final_checkpoint.pth"
    if final_checkpoint.exists():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Found final_checkpoint.pth - using it for resume")
        return final_checkpoint

    # Then look for epoch checkpoints
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

def load_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler, device):
    """Loads model, optimizer, scaler, and scheduler state from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Checkpoint file not found: {checkpoint_path}")
        return model, optimizer, scaler, scheduler, 0

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

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if dist.get_rank() == 0:
                print(f"[Checkpoint] LR scheduler state loaded")

        start_epoch = checkpoint.get('epoch', 0)
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Successfully loaded state from epoch {start_epoch}")
        return model, optimizer, scaler, scheduler, start_epoch
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Error loading checkpoint: {e}")
        return model, optimizer, scaler, scheduler, 0

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
            shuffle=False,
            drop_last=True  # CRITICAL: ensures same batch count across all ranks
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
            drop_last=True,  # CRITICAL: ensures same batch count across all ranks
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
        # DISABLED DUE TO DETR SIGSEGV BUG: if hasattr(torch, 'compile') and args.compile_model:
        # DISABLED DUE TO DETR SIGSEGV BUG:     if rank == 0:
        # DISABLED DUE TO DETR SIGSEGV BUG:         print("Compiling model with torch.compile...")
        # DISABLED DUE TO DETR SIGSEGV BUG:     model = torch.compile(model, mode="reduce-overhead")
        
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

    # Setup Learning Rate Scheduler
    scheduler = None
    if args.lr_scheduler != "none":
        if rank == 0:
            print(f"Setting up LR scheduler: {args.lr_scheduler}")

        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr_min
            )
        elif args.lr_scheduler == "cosine_warmup":
            # Cosine with linear warmup
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    # Linear warmup
                    return (epoch + 1) / args.warmup_epochs
                else:
                    # Cosine annealing after warmup
                    progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                    return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * (1 + math.cos(math.pi * progress)) / 2

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif args.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_step_size,
                gamma=args.lr_gamma
            )
        elif args.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.lr_gamma,
                patience=5,
                min_lr=args.lr_min,
                verbose=(rank == 0)
            )

        if rank == 0 and scheduler:
            print(f"  LR scheduler initialized: {type(scheduler).__name__}")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_training:
        latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint_path:
            # Note: pass the base model (not DDP wrapped) for loading
            model.module, optimizer, scaler, scheduler, start_epoch = load_checkpoint(
                latest_checkpoint_path, model.module, optimizer, scaler, scheduler, device
            )

    # Training loop
    if rank == 0:
        print(f"Starting training loop from epoch {start_epoch}...")
        print(f"Mixed Precision Training: {'Enabled' if args.use_amp else 'Disabled'}")
        print(f"LR Scheduler: {args.lr_scheduler}")
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.2e}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Gradient accumulation
    accumulation_steps = args.gradient_accumulation_steps

    # For tracking DETR loss components
    import time

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()

            # Set epoch for distributed sampler (important for shuffling)
            train_sampler.set_epoch(epoch)

            if rank == 0:
                print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
                # Log GPU memory at epoch start
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved")
                    writer.add_scalar("GPU/memory_allocated_GB", gpu_mem_allocated, epoch)
                    writer.add_scalar("GPU/memory_reserved_GB", gpu_mem_reserved, epoch)

            model.train()
            train_loss = 0.0
            processed_batches = 0
            # Track DETR loss components
            epoch_loss_ce = 0.0
            epoch_loss_bbox = 0.0
            epoch_loss_giou = 0.0
            epoch_grad_norm = 0.0
            
            # Create progress bar only on main process
            if rank == 0:
                progress_bar = tqdm(train_dataloader, desc=f"Training E{epoch+1}")
            else:
                progress_bar = train_dataloader

            optimizer.zero_grad()
            
            for i, batch in enumerate(progress_bar):
                if batch is None:
                    # Skip None batches - drop_last=True ensures same batch count across ranks
                    # DO NOT use barrier here - it causes desync if only some ranks get None
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        epoch_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Update stats
                batch_loss = loss.item() * accumulation_steps
                train_loss += batch_loss
                processed_batches += 1

                # Track DETR loss components (if available)
                if hasattr(outputs, 'loss_dict'):
                    loss_dict = outputs.loss_dict
                    if 'loss_ce' in loss_dict:
                        epoch_loss_ce += loss_dict['loss_ce'].item()
                    if 'loss_bbox' in loss_dict:
                        epoch_loss_bbox += loss_dict['loss_bbox'].item()
                    if 'loss_giou' in loss_dict:
                        epoch_loss_giou += loss_dict['loss_giou'].item()

                if rank == 0:
                    progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
                    # Log every 50 batches for TensorBoard (balance detail vs performance)
                    if i % 50 == 0:
                        global_step = epoch * len(train_dataloader) + i
                        writer.add_scalar("Loss/train_batch", batch_loss, global_step)

                        # Log loss components if available
                        if hasattr(outputs, 'loss_dict'):
                            loss_dict = outputs.loss_dict
                            if 'loss_ce' in loss_dict:
                                writer.add_scalar("Loss_Components/ce_batch", loss_dict['loss_ce'].item(), global_step)
                            if 'loss_bbox' in loss_dict:
                                writer.add_scalar("Loss_Components/bbox_batch", loss_dict['loss_bbox'].item(), global_step)
                            if 'loss_giou' in loss_dict:
                                writer.add_scalar("Loss_Components/giou_batch", loss_dict['loss_giou'].item(), global_step)

            # Synchronize all processes before aggregation to prevent NCCL timeout
            dist.barrier()

            # Aggregate loss across all processes
            train_loss_tensor = torch.tensor(train_loss).to(device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)

            # Aggregate batch count as well (processes might have different counts due to None batches)
            batch_count_tensor = torch.tensor(processed_batches).to(device)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / max(batch_count_tensor.item(), 1)
            
            if rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f} (time: {epoch_time:.1f}s)")

                # Log epoch metrics to TensorBoard
                writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
                writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

                # Log average gradient norm
                if processed_batches > 0 and epoch_grad_norm > 0:
                    avg_grad_norm = epoch_grad_norm / processed_batches
                    writer.add_scalar("Gradients/norm_avg", avg_grad_norm, epoch)
                    print(f"  Avg Gradient Norm: {avg_grad_norm:.4f}")

                # Log DETR loss components (epoch averages)
                if processed_batches > 0:
                    if epoch_loss_ce > 0:
                        writer.add_scalar("Loss_Components/ce_epoch", epoch_loss_ce / processed_batches, epoch)
                    if epoch_loss_bbox > 0:
                        writer.add_scalar("Loss_Components/bbox_epoch", epoch_loss_bbox / processed_batches, epoch)
                    if epoch_loss_giou > 0:
                        writer.add_scalar("Loss_Components/giou_epoch", epoch_loss_giou / processed_batches, epoch)

                # Flush TensorBoard writer periodically
                writer.flush()

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
                            # Skip None batches - DO NOT use barrier here
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

                # Synchronize all processes before validation loss aggregation
                dist.barrier()

                # Aggregate validation loss
                val_loss_tensor = torch.tensor(val_loss).to(device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

                # Aggregate batch count
                val_batch_count_tensor = torch.tensor(processed_val_batches).to(device)
                dist.all_reduce(val_batch_count_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_tensor.item() / max(val_batch_count_tensor.item(), 1)
                
                if rank == 0:
                    print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
                    writer.add_scalar("Loss/validation_epoch", avg_val_loss, epoch)

                # Synchronize before checkpoint saving
                dist.barrier()

                # Step LR scheduler (after validation)
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    # Log current LR
                    if rank == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        current_lr_backbone = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else current_lr
                        print(f"  LR: {current_lr:.2e} (backbone: {current_lr_backbone:.2e})")
                        writer.add_scalar("LR/main", current_lr, epoch)
                        writer.add_scalar("LR/backbone", current_lr_backbone, epoch)

                if rank == 0:
                    # Save checkpoint and best model (only on main process)
                    if (epoch + 1) % args.save_interval == 0:
                        chkpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                        checkpoint_data = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict(),  # Save unwrapped model
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'loss': avg_val_loss,
                        }
                        if scheduler is not None:
                            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(checkpoint_data, chkpt_path)
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
                        print(f"Validation loss did not improve for {patience_counter} epoch(s).")

                # Synchronize all ranks before checking early stopping
                dist.barrier()

                # Broadcast early stopping decision from rank 0 to all processes
                if rank == 0:
                    should_stop = torch.tensor([1 if patience_counter >= args.patience else 0], device=device)
                else:
                    should_stop = torch.tensor([0], device=device)
                dist.broadcast(should_stop, src=0)

                if should_stop.item() == 1:
                    if rank == 0:
                        print(f"Early stopping triggered.")
                    break

            # End of epoch - synchronize all ranks
            dist.barrier()

    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user.")
    finally:
        # Synchronize all ranks before final cleanup (with timeout to avoid hang)
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Final barrier failed: {e}")

        # Save final model (only on main process)
        if rank == 0:
            print("\nSaving final model...")
            try:
                model.module.save_pretrained(str(final_model_dir))
                processor.save_pretrained(str(final_model_dir))
                print(f"Final model saved to: {final_model_dir}")

                # Save final checkpoint
                final_checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }
                if scheduler is not None:
                    final_checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(final_checkpoint_data, checkpoint_dir / "final_checkpoint.pth")

            except Exception as e:
                print(f"Error saving final model: {e}")

            writer.close()

        # Wait for rank 0 to finish saving before cleanup
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

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
    parser.add_argument("--use_amp", action='store_true', default=False,
                        help="Use automatic mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--compile_model", action='store_true', default=False,
                        help="Use torch.compile for model optimization (requires PyTorch 2.0+)")

    # Learning Rate Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "cosine_warmup", "step", "plateau"],
                        help="LR scheduler type: none, cosine, cosine_warmup, step, plateau")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs (for cosine_warmup)")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--lr_step_size", type=int, default=30,
                        help="Step size for StepLR scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.1,
                        help="Gamma for StepLR/ReduceLROnPlateau scheduler")
    
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume_training", action='store_true')
    
    args = parser.parse_args()
    
    train(args)