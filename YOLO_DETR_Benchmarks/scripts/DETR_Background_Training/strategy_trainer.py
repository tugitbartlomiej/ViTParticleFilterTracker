#!/usr/bin/env python3
"""
STRATEGY TRAINER - Parametryzowany script do testowania strategii
================================================================

Łatwo konfigurowalny trainer używany przez strategy testing framework.
Implementuje różne strategie uczenia z focus na preventing catastrophic forgetting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
import json
import os
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.optim import AdamW
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime

class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, processor, prefix=""):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.prefix = prefix
        
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        print(f"{prefix} Dataset: {len(self.image_ids)} images, {len(self.annotations)} annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = self.images_dir / image_info['file_name']
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Get annotations for this image
        image_annotations = self.annotations_by_image.get(image_id, [])
        
        # Prepare annotations for DETR processor
        annotations = []
        if image_annotations:
            for ann in image_annotations:
                annotations.append({
                    'image_id': image_id,
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    'iscrowd': ann.get('iscrowd', 0)
                })
        
        # Create proper COCO format target
        target = {
            'image_id': image_id,
            'annotations': annotations
        }
        
        # Process with DETR processor
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'pixel_mask': encoding['pixel_mask'].squeeze(),
            'labels': encoding['labels'][0] if 'labels' in encoding else None
        }

class MixedDataset(Dataset):
    """Dataset with configurable tooltip/background mixing"""
    def __init__(self, tooltip_dataset, background_dataset, tooltip_ratio=0.9):
        self.tooltip_dataset = tooltip_dataset
        self.background_dataset = background_dataset
        self.tooltip_ratio = tooltip_ratio
        
        tooltip_len = len(tooltip_dataset)
        background_len = len(background_dataset)
        
        # Calculate effective length
        self.length = max(tooltip_len, int(background_len / (1 - tooltip_ratio)))
        
        print(f"Mixed Dataset: {self.length} samples ({tooltip_ratio*100:.0f}% tooltip)")
        print(f"Tooltip pool: {tooltip_len}, Background pool: {background_len}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if random.random() < self.tooltip_ratio:
            tooltip_idx = idx % len(self.tooltip_dataset)
            return self.tooltip_dataset[tooltip_idx]
        else:
            bg_idx = idx % len(self.background_dataset)
            return self.background_dataset[bg_idx]

def collate_fn(batch):
    """Custom collate function for DETR training - handles variable sizes"""
    # DETR processor already handles batching - just return individual items
    return batch[0] if len(batch) == 1 else batch

class StrategyTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        print("STRATEGY TRAINER INITIALIZATION")
        print("="*50)
        print(f"Strategy: {args.strategy_name}")
        print(f"Device: {self.device}")
        print(f"Learning Rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print(f"Tooltip Ratio: {args.tooltip_ratio}")
        print(f"Freeze Backbone: {args.freeze_backbone}")
        print("="*50)
        
        # Initialize components
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = self._load_model()
        self.dataloaders = self._prepare_data()
        self.optimizer = self._setup_optimizer()
        
        # Training tracking
        self.training_log = {
            'strategy': args.strategy_name,
            'start_time': datetime.now().isoformat(),
            'config': vars(args),
            'epochs': [],
            'best_val_loss': float('inf')
        }
    
    def _load_model(self):
        """Load and configure model based on strategy"""
        print("\nLoading model...")
        
        # Load base model with correct class count
        # DETR adds 1 for "no object" class, so checkpoint with 2 classes needs num_labels=1
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,  # Will create 2 classes: tooltip + no_object
            ignore_mismatched_sizes=True
        )
        
        # Load pretrained tooltip weights
        print(f"Loading checkpoint: {self.args.checkpoint_path}")
        checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights - checkpoint should match model now (both 2 classes)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        
        print("Model loaded successfully with tooltip weights")
        
        model.to(self.device)
        
        # Apply freezing strategy
        if self.args.freeze_backbone:
            self._apply_freezing_strategy(model)
        
        return model
    
    def _apply_freezing_strategy(self, model):
        """Apply layer freezing based on strategy"""
        print("\nApplying freezing strategy...")
        
        frozen_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            
            # Freeze backbone and encoder (keep decoder and classifier trainable)
            if ('backbone' in name or 
                'encoder' in name or
                'input_projection' in name or
                'query_position_embeddings' in name):
                
                param.requires_grad = False
                frozen_params += 1
                print(f"Frozen: {name}")
        
        print(f"Frozen {frozen_params}/{total_params} parameter groups")
        
        # Verify trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def _prepare_data(self):
        """Prepare datasets and dataloaders"""
        print("\nPreparing datasets...")
        
        # Load tooltip dataset
        tooltip_dataset = COCODataset(
            self.args.tooltip_images_dir,
            self.args.tooltip_annotations_path,
            self.processor,
            "TOOLTIP"
        )
        
        # Load background dataset
        background_dataset = COCODataset(
            self.args.background_images_dir,
            self.args.background_annotations_path,
            self.processor,
            "BACKGROUND"
        )
        
        # Create mixed dataset
        mixed_dataset = MixedDataset(
            tooltip_dataset, 
            background_dataset, 
            self.args.tooltip_ratio
        )
        
        # Split train/val
        train_size = int(0.9 * len(mixed_dataset))
        val_size = len(mixed_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            mixed_dataset, [train_size, val_size]
        )
        
        # Create dataloaders - use batch_size=1 to avoid tensor size issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # Force batch_size=1 for variable image sizes
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Force batch_size=1 for variable image sizes
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        return {'train': train_loader, 'val': val_loader}
    
    def _setup_optimizer(self):
        """Setup optimizer with strategy-specific learning rates"""
        
        if hasattr(self.args, 'lr_backbone') and hasattr(self.args, 'lr_classifier'):
            # Differential learning rates
            print(f"\nUsing differential learning rates:")
            print(f"  Backbone: {self.args.lr_backbone}")
            print(f"  Classifier: {self.args.lr_classifier}")
            
            param_groups = [
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if 'backbone' in n and p.requires_grad],
                    'lr': self.args.lr_backbone
                },
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if 'backbone' not in n and p.requires_grad],
                    'lr': self.args.lr_classifier
                }
            ]
            
            optimizer = AdamW(param_groups, weight_decay=self.args.weight_decay)
        else:
            # Standard learning rate
            print(f"\nUsing standard learning rate: {self.args.lr}")
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        
        return optimizer
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\nTraining Epoch {epoch}...")
        
        for i, batch in enumerate(self.dataloaders['train']):
            # Handle single item batch
            pixel_values = batch['pixel_values'].unsqueeze(0).to(self.device)
            pixel_mask = batch['pixel_mask'].unsqueeze(0).to(self.device)
            labels = [batch['labels']] if batch['labels'] is not None else [{'class_labels': torch.tensor([], dtype=torch.long), 'boxes': torch.tensor([], dtype=torch.float32).reshape(0, 4)}]
            
            # Move labels to device
            if labels[0] is not None:
                labels[0]['class_labels'] = labels[0]['class_labels'].to(self.device)
                labels[0]['boxes'] = labels[0]['boxes'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(self.dataloaders['train'])}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\nValidating Epoch {epoch}...")
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloaders['val']):
                # Handle single item batch
                pixel_values = batch['pixel_values'].unsqueeze(0).to(self.device)
                pixel_mask = batch['pixel_mask'].unsqueeze(0).to(self.device)
                labels = [batch['labels']] if batch['labels'] is not None else [{'class_labels': torch.tensor([], dtype=torch.long), 'boxes': torch.tensor([], dtype=torch.float32).reshape(0, 4)}]
                
                # Move labels to device
                if labels[0] is not None:
                    labels[0]['class_labels'] = labels[0]['class_labels'].to(self.device)
                    labels[0]['boxes'] = labels[0]['boxes'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Run complete training"""
        print(f"\n{'='*60}")
        print(f"STRATEGY: {self.args.strategy_name.upper()}")
        print(f"Starting training for {self.args.epochs} epochs...")
        print(f"Learning Rate: {self.args.lr}")
        print(f"Tooltip Ratio: {self.args.tooltip_ratio}")
        print(f"Freeze Backbone: {getattr(self.args, 'freeze_backbone', False)}")
        print(f"{'='*60}")
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\n{'*'*40}")
            print(f"EPOCH {epoch}/{self.args.epochs}")
            print(f"{'*'*40}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate_epoch(epoch)
            
            # Log epoch results
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            }
            self.training_log['epochs'].append(epoch_data)
            
            print(f"\nEPOCH {epoch} SUMMARY:")
            print(f"  Training Loss:   {train_loss:.6f}")
            print(f"  Validation Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < self.training_log['best_val_loss']:
                self.training_log['best_val_loss'] = val_loss
                print(f"  *** NEW BEST VALIDATION LOSS: {val_loss:.6f} ***")
                
                # Save model
                os.makedirs(self.args.output_dir, exist_ok=True)
                model_path = os.path.join(self.args.output_dir, 'mixed_gentle_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"  Model saved to: {model_path}")
            else:
                print(f"  No improvement (best: {self.training_log['best_val_loss']:.6f})")
        
        # Save training log
        log_path = os.path.join(self.args.output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED!")
        print(f"Strategy: {self.args.strategy_name}")
        print(f"Best validation loss: {self.training_log['best_val_loss']:.6f}")
        print(f"Training log saved to: {log_path}")
        print(f"Model saved to: {os.path.join(self.args.output_dir, 'mixed_gentle_model.pth')}")
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Strategy-based DETR training")
    
    # Strategy info
    parser.add_argument("--strategy_name", type=str, default="test", help="Strategy name")
    
    # Paths
    parser.add_argument("--tooltip_images_dir", type=str, required=True)
    parser.add_argument("--tooltip_annotations_path", type=str, required=True)
    parser.add_argument("--background_images_dir", type=str, required=True)
    parser.add_argument("--background_annotations_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./strategy_output")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-7, help="Learning rate")
    parser.add_argument("--lr_backbone", type=float, help="Backbone learning rate (differential)")
    parser.add_argument("--lr_classifier", type=float, help="Classifier learning rate (differential)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    
    # Data mixing
    parser.add_argument("--tooltip_ratio", type=float, default=0.9)
    
    # Strategy options
    parser.add_argument("--freeze_backbone", action="store_true")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = StrategyTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()