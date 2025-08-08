import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
)

torch._dynamo.config.suppress_errors = True

def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not running in distributed mode")
        return None, None, None
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

class SurgicalToolDataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, augment=False, is_background=False):
        if dist.get_rank() == 0:
            print(f"Initializing {'background' if is_background else 'original'} dataset...")
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.augment = augment
        self.is_background = is_background

        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = None

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
            print(f"{'Background' if is_background else 'Original'} dataset initialized. Using {len(self.image_ids)} images.")

    def __len__(self):
        return len(self.image_ids)

    @lru_cache(maxsize=128)
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
        if dist.get_rank() == 0:
            print(f"Error during collate_fn: {e}")
        return None

def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest fine-tuning checkpoint file based on epoch number."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Directory not found: {checkpoint_dir}")
        return None

    # Look for fine-tuning checkpoints first
    checkpoint_files = list(checkpoint_dir.glob("DETR_fine_tuned_background_model_checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        # Fallback to generic pattern
        checkpoint_files = list(checkpoint_dir.glob("*checkpoint_epoch_*.pth"))
    
    if not checkpoint_files:
        if dist.get_rank() == 0:
            print(f"[Checkpoint] No fine-tuning checkpoints found in {checkpoint_dir}")
        return None

    try:
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Found latest fine-tuning checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Error parsing checkpoint filenames: {e}")
        return None

def load_model_from_checkpoint(checkpoint_path, device, new_num_classes, id2label, label2id):
    """Load model from PyTorch checkpoint and modify for fine-tuning with new class"""
    if dist.get_rank() == 0:
        print(f"[Fine-tuning] Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        if dist.get_rank() == 0:
            print(f"[Fine-tuning] Checkpoint loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # Create base model with default config (1 class - surgical_tool)
        config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
        original_num_classes = 1  # We know original model had 1 class
        
        # Update config for new number of classes
        config.num_labels = new_num_classes
        config.id2label = id2label
        config.label2id = label2id
        
        if dist.get_rank() == 0:
            print(f"[Fine-tuning] Original model has {original_num_classes} classes")
            print(f"[Fine-tuning] New model will have {new_num_classes} classes")
        
        # Create model with original structure first
        model = DetrForObjectDetection(config).to(device)
        
        # Load state dict from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Handle classification head modification
        if new_num_classes != original_num_classes:
            if dist.get_rank() == 0:
                print(f"[Fine-tuning] Modifying classification head from {original_num_classes} to {new_num_classes} classes")
            
            # Get original classification weights
            old_class_weight = new_state_dict['class_embed.weight']  # Shape: [2, hidden_dim] (1 class + no-object)
            old_class_bias = new_state_dict['class_embed.bias']      # Shape: [2]
            
            # Create new classification head weights
            hidden_dim = old_class_weight.size(1)
            new_class_weight = torch.zeros(new_num_classes + 1, hidden_dim)  # +1 for no-object
            new_class_bias = torch.zeros(new_num_classes + 1)
            
            # Initialize new weights
            torch.nn.init.xavier_uniform_(new_class_weight)
            torch.nn.init.constant_(new_class_bias, 0.)
            
            # Copy weights for existing classes
            num_classes_to_copy = min(original_num_classes + 1, new_num_classes + 1)
            new_class_weight[:num_classes_to_copy] = old_class_weight[:num_classes_to_copy]
            new_class_bias[:num_classes_to_copy] = old_class_bias[:num_classes_to_copy]
            
            # Update state dict
            new_state_dict['class_embed.weight'] = new_class_weight
            new_state_dict['class_embed.bias'] = new_class_bias
            
            if dist.get_rank() == 0:
                print(f"[Fine-tuning] Copied weights for {original_num_classes} existing classes")
                print(f"[Fine-tuning] Initialized new classification head: {new_class_weight.shape}")
        
        # Load modified state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if dist.get_rank() == 0:
            if missing_keys:
                print(f"[Fine-tuning] Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[Fine-tuning] Unexpected keys: {unexpected_keys}")
            print(f"[Fine-tuning] Model successfully loaded and modified from checkpoint")
            
        return model
        
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[Fine-tuning] Error loading model from checkpoint: {e}")
        raise e

def load_checkpoint(checkpoint_path, model, optimizer, scaler, device):
    """Loads model and optimizer state from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        if dist.get_rank() == 0:
            print(f"[Checkpoint] Checkpoint file not found: {checkpoint_path}")
        return model, optimizer, scaler, 0

    if dist.get_rank() == 0:
        print(f"[Checkpoint] Loading checkpoint from: {checkpoint_path}")
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        state_dict = checkpoint['model_state_dict']
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

def train(args):
    rank, world_size, local_rank = setup_ddp()
    
    if rank == 0:
        print("Starting DETR fine-tuning for background class...")
        print(f"Arguments: {args}")
        print(f"Running on {world_size} GPUs")

    device = torch.device(f'cuda:{local_rank}')
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
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
    
    dist.barrier()

    # Load categories from both datasets
    if rank == 0:
        print("Loading categories from original and background datasets...")
    
    try:
        # Load original categories
        with open(args.original_annotations_path, 'r') as f:
            original_coco_data = json.load(f)
        original_categories = original_coco_data.get('categories', [])
        
        # Load background categories  
        with open(args.background_annotations_path, 'r') as f:
            background_coco_data = json.load(f)
        background_categories = background_coco_data.get('categories', [])
        
        # Create combined categories with background class
        # Assume background will be class ID 2 (after surgical_tool which is ID 1)
        combined_categories = original_categories.copy()
        background_class = {
            'id': len(original_categories) + 1,  # Next available ID
            'name': 'background',
            'supercategory': 'scene'
        }
        combined_categories.append(background_class)
        
        id2label = {cat['id']: cat['name'] for cat in combined_categories}
        label2id = {v: k for k, v in id2label.items()}
        
        if rank == 0:
            print(f"Combined categories: {id2label}")
            print(f"Background class ID: {background_class['id']}")
            
    except Exception as e:
        if rank == 0:
            print(f"Error loading categories: {e}")
        cleanup_ddp()
        return

    # Load processor (use default DETR processor)
    if rank == 0:
        print(f"Loading image processor from facebook/detr-resnet-50")
    try:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    except Exception as e:
        if rank == 0:
            print(f"Error loading processor: {e}")
        cleanup_ddp()
        return

    # Create datasets
    if rank == 0:
        print("Creating datasets...")
    try:
        # Original dataset (with surgical tools)
        original_dataset = SurgicalToolDataset(
            images_dir=args.original_images_dir,
            annotations_file=args.original_annotations_path,
            processor=processor,
            augment=args.augment,
            is_background=False
        )
        
        # Background dataset (empty annotations, background class)
        background_dataset = SurgicalToolDataset(
            images_dir=args.background_images_dir,
            annotations_file=args.background_annotations_path,
            processor=processor,
            augment=args.augment,
            is_background=True
        )
        
        # Combine datasets with proper weighting
        if args.background_weight > 0:
            # Create weighted combination
            background_copies = int(len(original_dataset) * args.background_weight / len(background_dataset))
            background_copies = max(1, background_copies)  # At least 1 copy
            
            background_datasets = [background_dataset] * background_copies
            full_dataset = ConcatDataset([original_dataset] + background_datasets)
            
            if rank == 0:
                print(f"Using {background_copies} copies of background dataset for weighting")
        else:
            full_dataset = ConcatDataset([original_dataset, background_dataset])
            
    except Exception as e:
        if rank == 0:
            print(f"Error creating datasets: {e}")
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
        print(f"Original dataset size: {len(original_dataset)}")
        print(f"Background dataset size: {len(background_dataset)}")
        print(f"Combined dataset size: {len(full_dataset)}")
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

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
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

    # Load model from checkpoint and modify for new classes
    try:
        model = load_model_from_checkpoint(
            args.checkpoint_path,
            device,
            len(id2label),
            id2label,
            label2id
        )
        
        # Apply fine-tuning strategy
        if args.freeze_backbone:
            if rank == 0:
                print("[Fine-tuning] Freezing backbone parameters")
            for name, param in model.named_parameters():
                if "backbone" in name:
                    param.requires_grad = False
        
        if args.freeze_encoder:
            if rank == 0:
                print("[Fine-tuning] Freezing encoder parameters")
            for name, param in model.named_parameters():
                if "transformer.encoder" in name:
                    param.requires_grad = False
        
        # Compile model if requested
        if hasattr(torch, 'compile') and args.compile_model:
            if rank == 0:
                print("Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
        
        # Wrap model in DDP
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False
        )
        
        if rank == 0:
            print(f"Model successfully wrapped in DDP on GPU {local_rank}")
            
    except Exception as e:
        if rank == 0:
            print(f"Error loading model: {e}")
        cleanup_ddp()
        return

    # Setup optimizer with different learning rates for fine-tuning
    try:
        param_groups = []
        
        # Classification head gets higher learning rate
        class_embed_params = [p for n, p in model.named_parameters() 
                             if "class_embed" in n and p.requires_grad]
        if class_embed_params:
            param_groups.append({
                "params": class_embed_params,
                "lr": args.lr_head,
                "name": "classification_head"
            })
        
        # Decoder gets medium learning rate
        decoder_params = [p for n, p in model.named_parameters() 
                         if "transformer.decoder" in n and p.requires_grad 
                         and "class_embed" not in n]
        if decoder_params:
            param_groups.append({
                "params": decoder_params,
                "lr": args.lr_decoder,
                "name": "decoder"
            })
        
        # Encoder gets lower learning rate (if not frozen)
        encoder_params = [p for n, p in model.named_parameters() 
                         if "transformer.encoder" in n and p.requires_grad]
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": args.lr_encoder,
                "name": "encoder"
            })
        
        # Backbone gets lowest learning rate (if not frozen)
        backbone_params = [p for n, p in model.named_parameters() 
                          if "backbone" in n and p.requires_grad]
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": args.lr_backbone,
                "name": "backbone"
            })
        
        # Other parameters
        other_params = [p for n, p in model.named_parameters() 
                       if not any(key in n for key in ["class_embed", "transformer.decoder", 
                                                      "transformer.encoder", "backbone"]) 
                       and p.requires_grad]
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": args.lr,
                "name": "other"
            })
        
        if rank == 0:
            print("[Fine-tuning] Parameter groups:")
            for group in param_groups:
                print(f"  {group['name']}: {len(group['params'])} params, LR: {group['lr']}")
        
        optimizer = torch.optim.AdamW(
            param_groups, 
            weight_decay=args.weight_decay,
            fused=True if torch.cuda.is_available() else False
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
            model.module, optimizer, scaler, start_epoch = load_checkpoint(
                latest_checkpoint_path, model.module, optimizer, scaler, device
            )

    # Training loop
    if rank == 0:
        print(f"Starting fine-tuning loop from epoch {start_epoch}...")
        print(f"Mixed Precision Training: {'Enabled' if args.use_amp else 'Disabled'}")
        print(f"Fine-tuning strategy: freeze_backbone={args.freeze_backbone}, freeze_encoder={args.freeze_encoder}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    accumulation_steps = args.gradient_accumulation_steps

    try:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\n--- Fine-tuning Epoch {epoch+1}/{args.epochs} ---")
                
            model.train()
            train_loss = 0.0
            processed_batches = 0
            
            # Create progress bar only on main process
            if rank == 0:
                progress_bar = tqdm(train_dataloader, desc=f"Fine-tuning E{epoch+1}")
            else:
                progress_bar = train_dataloader

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

                    # Save checkpoint and best model
                    if (epoch + 1) % args.save_interval == 0:
                        chkpt_path = checkpoint_dir / f"DETR_fine_tuned_background_model_checkpoint_epoch_{epoch+1}.pth"
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'loss': avg_val_loss,
                            'num_classes': len(id2label),
                            'id2label': id2label,
                            'label2id': label2id,
                        }, chkpt_path)
                        print(f"DETR_fine_tuned_background_model checkpoint saved to {chkpt_path}")

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving best model...")
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        model.module.save_pretrained(str(best_model_dir))
                        processor.save_pretrained(str(best_model_dir))
                        print(f"Best fine-tuned model saved to: {best_model_dir}")
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
            print("\nFine-tuning interrupted by user.")
    finally:
        # Save final model
        if rank == 0:
            print("\nSaving final fine-tuned model...")
            try:
                model.module.save_pretrained(str(final_model_dir))
                processor.save_pretrained(str(final_model_dir))
                print(f"Final fine-tuned model saved to: {final_model_dir}")
                
                # Save final checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'num_classes': len(id2label),
                    'id2label': id2label,
                    'label2id': label2id,
                }, checkpoint_dir / "DETR_fine_tuned_background_model_final_checkpoint.pth")
                
            except Exception as e:
                print(f"Error saving final model: {e}")
                
            writer.close()
            
        cleanup_ddp()
        
        if rank == 0:
            print("Fine-tuning completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DETR with background class - DDP Version")
    
    # Paths
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to pretrained DETR checkpoint (.pth file)")
    parser.add_argument("--original_images_dir", type=str, required=True,
                       help="Directory with original images")
    parser.add_argument("--original_annotations_path", type=str, required=True,
                       help="Path to original annotations JSON")
    parser.add_argument("--background_images_dir", type=str, required=True,
                       help="Directory with background images")
    parser.add_argument("--background_annotations_path", type=str, required=True,
                       help="Path to background annotations JSON")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_DETR_fine_tuned_background_model")
    parser.add_argument("--best_model_dir", type=str, default="./best_models/DETR_fine_tuned_background_model")
    parser.add_argument("--output_dir", type=str, default="./output_DETR_fine_tuned_background_model")
    
    # Fine-tuning strategy
    parser.add_argument("--freeze_backbone", action='store_true',
                       help="Freeze backbone weights during fine-tuning")
    parser.add_argument("--freeze_encoder", action='store_true',
                       help="Freeze encoder weights during fine-tuning")
    parser.add_argument("--background_weight", type=float, default=0.5,
                       help="Weight for background samples (0.5 means background samples are 50% of original)")
    
    # Learning rates for different components
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4,
                       help="Learning rate for classification head")
    parser.add_argument("--lr_decoder", type=float, default=5e-5,
                       help="Learning rate for decoder")
    parser.add_argument("--lr_encoder", type=float, default=1e-5,
                       help="Learning rate for encoder")
    parser.add_argument("--lr_backbone", type=float, default=1e-6,
                       help="Learning rate for backbone")
    
    # Dataset
    parser.add_argument("--train_val_split", type=float, default=0.9)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Training
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of fine-tuning epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    
    # Optimization
    parser.add_argument("--use_amp", action='store_true', default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--compile_model", action='store_true', default=False)
    
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume_training", action='store_true')
    
    args = parser.parse_args()
    
    train(args)